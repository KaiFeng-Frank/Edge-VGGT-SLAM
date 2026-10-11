"""Research-only channel-first original-parameter DPVO source-frame Update.

Derived from Princeton-VL/DPVO commit 0ac95b656d1fda91c271d2a106460d19ad966fc7
net.py Update and blocks.py SoftAgg/GatedResidual. Keep the adjacent MIT LICENSE,
Copyright (c) 2022 Princeton University. No model weights or inputs are bundled.

The frozen v34 dependency graph is retained; learned Linear transforms are
expressed as 1x1 Conv and original channel-only LayerNorm as centered population
variance over C. Real arithmetic agrees; convolution/reduction order can change
FP32 rounding. This is not bitwise equivalence, accepted inference or speed proof.
There is no token/source removal, altered group degree, clipping or new parameter.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

SUBMODULE_NAMES = ("c1", "c2", "norm", "agg_kk", "agg_ij", "gru", "corr", "d", "w")
PARENT_SOURCE_SHA256 = "b89b9b84d3afb47fd926decddde0077222dc07930bf75ee08acb410dc4f6f3f4"
AGGREGATION_EXPRESSION = "torch.softmax(original_g_conv(values), dim=group_axis_BCDM_2_or_3)"
CHANNEL_NORM_FORMULA = "(x-mean_C(x))/sqrt(mean_C((x-mean_C(x))^2)+original_eps)*original_gamma+original_beta"
SIGMOID_OPTIONS = ("stable", "original")
STABLE_SIGMOID_FORMULA = "exp(min(x,0))/(1+exp(-abs(x)))"


def _is_residual(module):
    return tuple(module._modules) == ("gate", "res") and isinstance(module.gate, nn.Sequential) and isinstance(module.res, nn.Sequential)


def _validate_module(module, channels):
    if isinstance(module, nn.Linear):
        if module.in_features <= 0 or module.out_features <= 0:
            raise ValueError("Invalid original Linear dimensions")
    elif isinstance(module, nn.LayerNorm):
        if tuple(module.normalized_shape) != (channels,) or not module.elementwise_affine or module.bias is None:
            raise ValueError("Expected original affine channel-only LayerNorm")
        if not math.isfinite(module.eps) or module.eps <= 0:
            raise ValueError("Original LayerNorm epsilon must be finite and positive")
    elif isinstance(module, nn.Sequential):
        for child in module.children():
            _validate_module(child, channels)
    elif _is_residual(module):
        if tuple(type(x) for x in module.gate) != (nn.Linear, nn.Sigmoid):
            raise ValueError("Expected unchanged original GatedResidual gate")
        if tuple(type(x) for x in module.res) != (nn.Linear, nn.ReLU, nn.Linear):
            raise ValueError("Expected unchanged original GatedResidual residual")
        _validate_module(module.gate, channels)
        _validate_module(module.res, channels)
    elif tuple(module._modules) == ("f", "g", "h") and getattr(module, "expand", None) is True:
        for child in module.children():
            if not isinstance(child, nn.Linear):
                raise ValueError("Expected original SoftAgg Linear f/g/h")
            _validate_module(child, channels)
    elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Identity)):
        pass
    elif type(module).__name__ == "GradientClip" and not tuple(module.parameters()) and not tuple(module.buffers()) and not tuple(module.children()):
        # The pinned official inference forward is identity; backward clipping
        # is never used because this wrapper requires no_grad/eval.
        pass
    else:
        raise ValueError("Unsupported original arithmetic module " + type(module).__name__)


def _linear(module, value):
    return F.conv2d(value, module.weight.reshape(module.out_features, module.in_features, 1, 1), module.bias)


def _channel_norm(module, value):
    mean = torch.mean(value, dim=1, keepdim=True)
    centered = value - mean
    variance = torch.mean(centered * centered, dim=1, keepdim=True)
    normalized = centered / torch.sqrt(variance + module.eps)
    return normalized * module.weight.reshape(1, -1, 1, 1) + module.bias.reshape(1, -1, 1, 1)


def _apply(module, value):
    if isinstance(module, nn.Linear):
        return _linear(module, value)
    if isinstance(module, nn.LayerNorm):
        return _channel_norm(module, value)
    if isinstance(module, nn.Sequential):
        for child in module.children():
            value = _apply(child, value)
        return value
    if _is_residual(module):
        return value + _apply(module.gate, value) * _apply(module.res, value)
    if isinstance(module, nn.ReLU):
        # Out-of-place preserves all branches, including shared net inputs to
        # delta and weight. The official inplace ReLUs follow fresh Linear values.
        return torch.relu(value)
    if isinstance(module, nn.Sigmoid):
        return torch.sigmoid(value)
    if isinstance(module, nn.Identity) or type(module).__name__ == "GradientClip":
        return value
    raise RuntimeError("Unvalidated arithmetic module")


def _aggregate(module, value, axis):
    weights = torch.softmax(_linear(module.g, value), dim=axis)
    grouped = torch.sum(_linear(module.f, value) * weights, dim=axis, keepdim=True)
    return _linear(module.h, grouped)


def _tensor_identity(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            value.dtype, value.device, tuple(value.stride()))


class SourceFrameUpdateChannelFirst(nn.Module):
    """Original-parameter arithmetic over CPU FP32 [B,C,D,M].

    The owner packs/restores channel-first arrays outside the neural graph.
    Complete source topology and inverse edge permutation remain the v32 plan.
    validate_inputs and verify_original_binding run outside timed/traced forward.
    Positive B,D,M are supported; first fixed export uses B1,C384,D10,M96.
    """
    def __init__(self, original_update, *, terminal_sigmoid="stable"):
        super().__init__()
        if terminal_sigmoid not in SIGMOID_OPTIONS:
            raise ValueError("terminal_sigmoid must be stable or original")
        if not isinstance(original_update, nn.Module) or any(v.training for v in original_update.modules()):
            raise ValueError("Expected original Update in eval mode")
        if set(original_update._modules) != set(SUBMODULE_NAMES):
            raise ValueError("Expected exactly the original nine Update child modules")
        for name in SUBMODULE_NAMES:
            module = getattr(original_update, name, None)
            if not isinstance(module, nn.Module):
                raise ValueError("Original Update missing module " + name)
        for name, module in original_update._modules.items():
            # Exactly the original registration names and module objects; no
            # prefix, parameter clone, altered LayerNorm or replacement Linear.
            self.add_module(name, module)
        if not isinstance(self.norm, nn.LayerNorm) or len(self.norm.normalized_shape) != 1:
            raise ValueError("Expected original channel-only LayerNorm")
        self.channels = int(self.norm.normalized_shape[0])
        first_corr_linear = next((v for v in self.corr.modules() if isinstance(v, nn.Linear)), None)
        if first_corr_linear is None or first_corr_linear.out_features != self.channels:
            raise ValueError("Expected original correlation Linear encoder")
        self.correlation_channels = int(first_corr_linear.in_features)
        for name in ("agg_kk", "agg_ij"):
            aggregate = getattr(self, name)
            if getattr(aggregate, "expand", None) is not True:
                raise ValueError("Only original expanded SoftAgg is supported")
            for component in ("f", "g", "h"):
                linear = getattr(aggregate, component, None)
                if not isinstance(linear, nn.Linear) or (linear.in_features, linear.out_features) != (self.channels, self.channels):
                    raise ValueError("Expected original per-channel SoftAgg f/g/h")
        if not isinstance(self.w, nn.Sequential) or len(self.w) < 2 or not isinstance(self.w[-1], nn.Sigmoid):
            raise ValueError("Expected original weight prefix and terminal Sigmoid")
        if list(self.state_dict()) != list(original_update.state_dict()):
            raise ValueError("Wrapper does not preserve the complete original state_dict keys")
        for value in self.parameters():
            if value.dtype != torch.float32 or value.device.type != "cpu":
                raise ValueError("Original parameter bindings must be CPU FP32")
        for module in tuple(self.children()):
            _validate_module(module, self.channels)
        self.terminal_sigmoid = terminal_sigmoid
        self._parameter_bindings = {name: _tensor_identity(v) for name, v in self.named_parameters()}
        self._buffer_bindings = {name: _tensor_identity(v) for name, v in self.named_buffers()}
        self._module_bindings = {name: id(v) for name, v in self.named_modules() if name}
        self.eval()
        self.verify_original_binding(original_update)

    def verify_original_binding(self, original_update):
        """Check object/identity/version bindings; not a replacement raw hash."""
        if self.training or original_update.training:
            raise ValueError("Update must remain in eval mode")
        original_parameters = dict(original_update.named_parameters())
        current_parameters = dict(self.named_parameters())
        original_buffers = dict(original_update.named_buffers())
        current_buffers = dict(self.named_buffers())
        if original_parameters.keys() != current_parameters.keys() or original_buffers.keys() != current_buffers.keys():
            raise ValueError("Changed original parameter/buffer names")
        if list(self.state_dict()) != list(original_update.state_dict()):
            raise ValueError("Changed original state_dict names")
        for name, value in current_parameters.items():
            if value is not original_parameters[name] or _tensor_identity(value) != self._parameter_bindings[name]:
                raise ValueError("Changed original parameter binding " + name)
        for name, value in current_buffers.items():
            if value is not original_buffers[name] or _tensor_identity(value) != self._buffer_bindings[name]:
                raise ValueError("Changed original buffer binding " + name)
        if {name: id(v) for name, v in self.named_modules() if name} != self._module_bindings:
            raise ValueError("Changed wrapper submodule identity")
        if any(getattr(self, name) is not getattr(original_update, name) for name in SUBMODULE_NAMES):
            raise ValueError("Changed original submodule identity")
        if any(v.training for v in self.modules()) or any(v.training for v in original_update.modules()):
            raise ValueError("A bound original submodule is not in eval mode")
        return {
            "original_parameter_objects_shared": True,
            "original_module_objects_shared": True,
            "state_dict_names_preserved": True,
            "state_tensors": len(self.state_dict()),
            "terminal_sigmoid": self.terminal_sigmoid,
            "stable_formula": STABLE_SIGMOID_FORMULA if self.terminal_sigmoid == "stable" else None,
            "original_terminal_sigmoid_module_retained": True,
            "input_output_layout": "B,C,D,M",
            "channel_LayerNorm_axis": 1,
            "channel_LayerNorm_formula": CHANNEL_NORM_FORMULA,
            "LayerNorm_configs": {name: {"channels": int(v.normalized_shape[0]), "eps": float(v.eps),
                                          "elementwise_affine": bool(v.elementwise_affine)}
                                  for name, v in self.named_modules() if isinstance(v, nn.LayerNorm)},
            "Linear_expression": "conv2d(x, original_weight.reshape(out,in,1,1), original_bias)",
            "raw_parameter_SHA_required_before_and_after_by_owner": True,
            "no_complete_weight_mutation_detection_from_identity_only": True,
        }

    def validate_inputs(self, net, inp, corr):
        if self.training or any(v.training for v in self.modules()):
            raise ValueError("Update must remain in eval mode")
        for name, value in (("net", net), ("inp", inp), ("corr", corr)):
            if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.dtype != torch.float32 or value.device.type != "cpu":
                raise ValueError(name + " must be CPU FP32 [B,C,D,M]")
            if not value.is_contiguous() or value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError(name + " must be contiguous, finite and not require gradients")
        if any(n <= 0 for n in net.shape) or net.shape != inp.shape or net.shape[1] != self.channels:
            raise ValueError("Invalid channel-first net/inp shape")
        if corr.shape[0] != net.shape[0] or corr.shape[2:] != net.shape[2:] or corr.shape[1] != self.correlation_channels:
            raise ValueError("Invalid channel-first correlation shape")
        return dict(batch=net.shape[0], channels=self.channels, target_degree=net.shape[2],
                    patches=net.shape[3], correlation_channels=self.correlation_channels)

    def forward(self, net, inp, corr):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("SourceFrameUpdateChannelFirst requires eval and torch.no_grad")
        net = _channel_norm(self.norm, net + inp + _apply(self.corr, corr))
        previous = torch.cat((torch.zeros_like(net[:, :, :1]), net[:, :, :-1]), dim=2)
        net = net + _apply(self.c1, previous)
        following = torch.cat((net[:, :, 1:], torch.zeros_like(net[:, :, -1:])), dim=2)
        net = net + _apply(self.c2, following)
        net = net + _aggregate(self.agg_kk, net, axis=2)
        net = net + _aggregate(self.agg_ij, net, axis=3)
        net = _apply(self.gru, net)
        delta = _apply(self.d, net)
        if self.terminal_sigmoid == "original":
            weight = _apply(self.w, net)
        else:
            logits = net
            for module in tuple(self.w.children())[:-1]:
                logits = _apply(module, logits)
            numerator = torch.exp(torch.minimum(logits, torch.zeros_like(logits)))
            weight = numerator / (1 + torch.exp(-torch.abs(logits)))
        return net, delta, weight
