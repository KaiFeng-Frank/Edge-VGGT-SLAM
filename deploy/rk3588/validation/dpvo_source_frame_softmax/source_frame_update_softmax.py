"""Research-only direct-softmax complete-source-frame DPVO Update.

Primary semantics: princeton-vl/DPVO commit
0ac95b656d1fda91c271d2a106460d19ad966fc7, dpvo/net.py Update and
dpvo/blocks.py SoftAgg/GatedResidual. Keep the adjacent original MIT license.

This v34 variant changes only the expressed group normalization relative to
the frozen v33 source-frame wrapper. It retains the original parameter objects
and dependency graph; direct torch.softmax may change reduction rounding.
The separately considered channel-first Conv variant is NOT implemented here.
No original pretrained weights or inputs are bundled. Source topology must be
validated by the frozen v32 source_frame_plan before constructing these tensors.
Changing GEMM/reduction shapes can change rounding: this is not an equivalence,
accepted RKNN execution, trajectory, latency or 20 Hz claim.
"""
import torch
from torch import nn


SUBMODULE_NAMES = ("c1", "c2", "norm", "agg_kk", "agg_ij", "gru", "corr", "d", "w")
PARENT_SOURCE_SHA256 = "cf1adb2224750cd0659139689f4b8a5a8107f2e733e8d1c6f86ea009c86a1bb2"
AGGREGATION_EXPRESSION = "torch.softmax(original_g(values), dim=group_axis)"
SIGMOID_OPTIONS = ("stable", "original")
STABLE_SIGMOID_FORMULA = "exp(min(x,0))/(1+exp(-abs(x)))"


def _aggregate(module, values, axis):
    """All members are real: no Gather, padded groups, masks or Where."""
    weights = torch.softmax(module.g(values), dim=axis)
    grouped = torch.sum(module.f(values) * weights, dim=axis, keepdim=True)
    return module.h(grouped)


def _tensor_identity(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            value.dtype, value.device, tuple(value.stride()))


class SourceFrameUpdate(nn.Module):
    """Use [B,D,M,C] target-major values for one complete source frame.

    forward(net, inp, corr) returns net/delta/weight in the same target-major
    order. The owner restores original global edge order using the v32 plan and
    validates every block before committing net or invoking BA. No per-block
    success is an independent pose.

    Call validate_inputs and verify_original_binding outside timed/exported
    execution. forward is inference-only and contains the neural arithmetic,
    without data-dependent validation or source topology construction. Torch
    supports positive B,D,M; first planned static export is B1,D10,M96,C384.
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
            "raw_parameter_SHA_required_before_and_after_by_owner": True,
            "no_complete_weight_mutation_detection_from_identity_only": True,
        }

    def validate_inputs(self, net, inp, corr):
        """CPU FP32 finite shape checks, to call before any forward/export."""
        if self.training or any(v.training for v in self.modules()):
            raise ValueError("Update must remain in eval mode")
        for name, value in (("net", net), ("inp", inp), ("corr", corr)):
            if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.dtype != torch.float32 or value.device.type != "cpu":
                raise ValueError(name + " must be CPU FP32 [B,D,M,C]")
            if not value.is_contiguous() or value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError(name + " must be contiguous, finite and not require gradients")
        if any(n <= 0 for n in net.shape) or net.shape != inp.shape or net.shape[-1] != self.channels:
            raise ValueError("Invalid net/inp source-block shape")
        if corr.shape[:3] != net.shape[:3] or corr.shape[-1] != self.correlation_channels:
            raise ValueError("Invalid correlation source-block shape")
        return {"batch": net.shape[0], "target_degree": net.shape[1],
                "patches": net.shape[2], "channels": self.channels,
                "correlation_channels": self.correlation_channels}

    def forward(self, net, inp, corr):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("SourceFrameUpdate requires eval and torch.no_grad")
        net = self.norm(net + inp + self.corr(corr))
        previous = torch.cat((torch.zeros_like(net[:, :1]), net[:, :-1]), dim=1)
        net = net + self.c1(previous)
        # c2 must consume the c1-updated state, including its boundary biases.
        following = torch.cat((net[:, 1:], torch.zeros_like(net[:, -1:])), dim=1)
        net = net + self.c2(following)
        net = net + _aggregate(self.agg_kk, net, axis=1)
        net = net + _aggregate(self.agg_ij, net, axis=2)
        net = self.gru(net)
        delta = self.d(net)
        if self.terminal_sigmoid == "original":
            weight = self.w(net)
        else:
            logits = net
            for module in tuple(self.w.children())[:-1]:
                logits = module(logits)
            numerator = torch.exp(torch.minimum(logits, torch.zeros_like(logits)))
            weight = numerator / (1 + torch.exp(-torch.abs(logits)))
        return net, delta, weight
