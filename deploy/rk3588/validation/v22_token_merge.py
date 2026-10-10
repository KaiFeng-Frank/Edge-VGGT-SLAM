"""Research-only CPU port of Co-Me's 2D merge/split and confidence predictor.

Derived from co-me-tokens/CoMe, commit
5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833, by the Co-Me authors:
https://github.com/co-me-tokens/CoMe
This adaptation is under CC BY-NC 4.0, not the repository's BSD code license:
https://creativecommons.org/licenses/by-nc/4.0/

Only the original norm2 -> MLP branch is compressed by a caller. Attention,
LayerScale and the full per-token residual are outside this helper. Restoring
a mean token repeats its MLP output; it does not reconstruct lost features.
This is an approximation experiment, not unchanged-model numerical equality
or a complete Co-Me deployment. No production entry point imports this file.

Each scene plan treats views separately. Every view's five special tokens
remain individual rows. 3x3 patches are ordered by group row/column then
within-group row/column; restore returns original image row-major order.
The released BF16 confidence model consumes DINO block[14] patch output,
before the final DINO norm. Its mask must be rebuilt for each input scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_COMMIT = '5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833'
SOURCE_PINS = {
    'src/accelerate/token_merger/co_me_2d.py': '5f3ad3dc44fb1b9d6afb427b3005f0d98fdddd33ce601c4db13aa57f05665d9c',
    'src/accelerate/common/confidence_predictor.py': 'cadde9b423499aa9d49ca72ed185625c2346d60a875501481a1aa9d396d32e73',
    'src/accelerate/vggt/direct.py': '70d46598daa04e89ebe5d54ddd39614d57e1eda1938d1c03eaa1c62eb5956f05',
}
RELEASED_CHECKPOINT_SHA256 = '95b0a004544c96da5c4e55748e8cf4837608627a6414563af06cb3ee25acd29f'


def _readonly(value):
    value = np.array(value, copy=True)
    value.flags.writeable = False
    return value


def _pair(value, name):
    if len(value) != 2 or any(type(n) is not int or n < 1 for n in value):
        raise ValueError(f'{name} must contain two positive integers')
    return tuple(value)


@dataclass(frozen=True)
class MergePlan:
    grid_hw: tuple[int, int]
    group_hw: tuple[int, int]
    special_tokens: int
    views: int
    compact_rows: int
    merge_mask: np.ndarray
    view_offsets: np.ndarray
    reverse_indices: np.ndarray
    merged_patch_mask: np.ndarray
    token_weights: np.ndarray
    representative_indices: np.ndarray
    mean_input_indices: np.ndarray
    mean_output_indices: np.ndarray
    _torch_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def tokens_per_view(self):
        return self.special_tokens + self.grid_hw[0] * self.grid_hw[1]

    def summary(self):
        original = self.views * self.tokens_per_view
        return {'views': self.views, 'grid_hw': list(self.grid_hw),
                'group_hw': list(self.group_hw), 'special_tokens_per_view': self.special_tokens,
                'original_rows': original, 'compact_rows': self.compact_rows,
                'merged_groups': int(self.merge_mask.sum()), 'total_groups': int(self.merge_mask.size),
                'merged_patch_tokens': int(self.merged_patch_mask.sum()),
                'unmerged_patch_tokens': int(self.merged_patch_mask.size - self.merged_patch_mask.sum()),
                'row_reduction_fraction': 1 - self.compact_rows / original,
                'view_offsets': self.view_offsets.tolist(),
                'residual_merged': False, 'attention_merged': False,
                'token_weights_applied_to_MLP': False, 'research_only': True}


def build_plan(mask, *, grid_hw=(18, 24), group_hw=(3, 3), special_tokens=5):
    """Build one immutable scene plan from NP bool [views, group_h, group_w]."""
    height, width = _pair(grid_hw, 'grid_hw')
    gh, gw = _pair(group_hw, 'group_hw')
    if (height % gh or width % gw or type(special_tokens) is not int or special_tokens < 0
            or not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 3
            or mask.shape != (mask.shape[0], height // gh, width // gw) or mask.shape[0] < 1):
        raise ValueError('Bool per-view merge mask must match the divisible patch grid')
    views = mask.shape[0]
    count = special_tokens + height * width
    if views * count > 65536:
        raise ValueError('Research merge plan exceeds its bounded token budget')
    groups = np.arange(height * width).reshape(height // gh, gh, width // gw, gw)
    groups = groups.transpose(0, 2, 1, 3).reshape(-1, gh * gw)
    representatives, mean_inputs, mean_outputs, weights = [], [], [], []
    reverse = np.empty((views, count), np.int64)
    patch_mask = np.zeros((views, height * width), np.bool_)
    offsets = [0]
    cursor = 0
    for view in range(views):
        for special in range(special_tokens):
            representatives.append(view * count + special)
            weights.append(1.)
            reverse[view, special] = cursor
            cursor += 1
        for group, do_merge in zip(groups, mask[view].reshape(-1)):
            originals = group + special_tokens
            if do_merge:
                representatives.append(view * count + int(originals[0]))
                mean_inputs.append(view * count + originals)
                mean_outputs.append(cursor)
                weights.append(float(gh * gw))
                reverse[view, originals] = cursor
                patch_mask[view, group] = True
                cursor += 1
            else:
                representatives.extend((view * count + originals).tolist())
                weights.extend([1.] * len(group))
                reverse[view, originals] = np.arange(cursor, cursor + len(group))
                cursor += len(group)
        offsets.append(cursor)
    return MergePlan((height, width), (gh, gw), special_tokens, views, cursor,
        _readonly(mask), _readonly(np.asarray(offsets, np.int64)), _readonly(reverse),
        _readonly(patch_mask.reshape(views, height, width)), _readonly(np.asarray(weights, np.float32)),
        _readonly(np.asarray(representatives, np.int64)),
        _readonly(np.asarray(mean_inputs, np.int64).reshape(-1, gh * gw)),
        _readonly(np.asarray(mean_outputs, np.int64)))


def _is_numpy(value):
    if isinstance(value, np.ndarray):
        if value.dtype not in (np.float16, np.float32, np.float64):
            raise ValueError('NumPy tokens must have a floating dtype')
        return True
    import torch
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or not value.is_floating_point():
        raise ValueError('Only CPU NumPy/Torch floating tokens are supported')
    return False


def _torch_indices(plan):
    if not plan._torch_cache:
        import torch
        plan._torch_cache.update({
            'representatives': torch.tensor(plan.representative_indices, dtype=torch.long),
            'mean_inputs': torch.tensor(plan.mean_input_indices.reshape(-1), dtype=torch.long),
            'mean_outputs': torch.tensor(plan.mean_output_indices, dtype=torch.long),
            'reverse': torch.tensor(plan.reverse_indices.reshape(-1), dtype=torch.long),
        })
    return plan._torch_cache


def merge_tokens(tokens, plan):
    """Gather/mean original MLP inputs; return compact [1, M, C].

    Accept [views, N, C] (DINO/frame) or [1, views*N, C] (global). CPU Torch
    means retain input dtype, including BF16; no implicit FP32 promotion.
    The same fixed plan may be used for all MLPs of its one captured scene.
    """
    numpy = _is_numpy(tokens)
    if (tokens.ndim != 3 or tokens.shape[-1] < 1
            or tokens.shape[:2] not in ((plan.views, plan.tokens_per_view),
                                       (1, plan.views * plan.tokens_per_view))):
        raise ValueError('Tokens do not match the plan view/grid layout')
    flat = tokens.reshape(plan.views * plan.tokens_per_view, tokens.shape[-1])
    if numpy:
        compact = np.take(flat, plan.representative_indices, axis=0)
        if len(plan.mean_output_indices):
            compact[plan.mean_output_indices] = np.take(flat, plan.mean_input_indices, axis=0).mean(axis=1)
    else:
        ids = _torch_indices(plan)
        compact = flat.index_select(0, ids['representatives'])
        if len(plan.mean_output_indices):
            means = flat.index_select(0, ids['mean_inputs']).reshape(
                -1, plan.group_hw[0] * plan.group_hw[1], flat.shape[-1]).mean(dim=1)
            compact.index_copy_(0, ids['mean_outputs'], means)
    return compact.reshape(1, plan.compact_rows, tokens.shape[-1])


def restore_tokens(compact, plan, *, layout='per_view'):
    """Gather compact MLP results to original rows, without scaling by weight."""
    numpy = _is_numpy(compact)
    if (compact.ndim != 3 or compact.shape[:2] != (1, plan.compact_rows)
            or compact.shape[-1] < 1 or layout not in ('per_view', 'global')):
        raise ValueError('Compact MLP result or requested restore layout is invalid')
    if numpy:
        result = np.take(compact[0], plan.reverse_indices.reshape(-1), axis=0)
    else:
        result = compact[0].index_select(0, _torch_indices(plan)['reverse'])
    shape = ((plan.views, plan.tokens_per_view, compact.shape[-1]) if layout == 'per_view'
             else (1, plan.views * plan.tokens_per_view, compact.shape[-1]))
    return result.reshape(shape)


def merge_mask_from_confidence(confidence, *, grid_hw=(18, 24), group_hw=(3, 3), threshold=.5):
    """Released pairwise-rank rule for one scene, strict < cross-view quantile.

    Input CPU Torch [views,H,W] or [views,H*W,1] is the predictor's exp+1
    output. Original BF16 block means precede a FP32 quantile. Ties may merge
    fewer than half the groups. No forced quota or confidence clipping.
    """
    import torch
    height, width = _pair(grid_hw, 'grid_hw')
    gh, gw = _pair(group_hw, 'group_hw')
    if (not isinstance(confidence, torch.Tensor) or confidence.device.type != 'cpu'
            or not confidence.is_floating_point() or confidence.ndim != 3
            or height % gh or width % gw or not np.isfinite(threshold) or not 0 <= threshold <= 1
            or confidence.shape[0] < 1 or confidence.shape[1:] not in ((height, width), (height * width, 1))
            or not torch.isfinite(confidence).all()):
        raise ValueError('Finite CPU confidence matching the scene grid and a quantile in [0,1] are required')
    views = confidence.shape[0]
    groups = confidence.reshape(views, height // gh, gh, width // gw, gw)
    means = groups.permute(0, 1, 3, 2, 4).reshape(views, -1, gh * gw).mean(dim=-1)
    percentile = torch.quantile(means.float().reshape(1, -1), float(threshold), dim=-1, keepdim=True)
    return (means.reshape(1, -1) < percentile).reshape(views, height // gh, width // gw).numpy()


def build_confidence_predictor(*, hidden_dims=(1024, 128, 1), activation='expp1'):
    """Exact released no-register GlobalAttentionConfidence Torch equations.

    Creates CPU modules with the original 18 state keys. BF16 is explicitly
    chosen by load_released_predictor; unsupported CPU BF16 raises naturally.
    Small alternative dimensions are for synthetic equation checks only.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    if (len(hidden_dims) != 3 or any(type(n) is not int or n < 1 for n in hidden_dims)
            or hidden_dims[2] != 1 or activation not in ('expp1', 'none')):
        raise ValueError('Predictor requires three positive dimensions, scalar output and a known activation')
    input_dim, dim, output_dim = hidden_dims

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(dim, 3 * dim, bias=True)
            self.proj = nn.Linear(dim, dim, bias=True)
        def forward(self, x):
            batch, count, _ = x.shape
            q, k, v = self.qkv(x).reshape(batch, count, 3, 1, dim).permute(2, 0, 3, 1, 4).unbind(0)
            result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
            return self.proj(result.transpose(1, 2).reshape(batch, count, dim))

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(dim, 2 * dim, bias=True)
            self.act = nn.GELU(approximate='none')
            self.fc2 = nn.Linear(2 * dim, dim, bias=True)
        def forward(self, x): return self.fc2(self.act(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-5)
            self.attn = Attention()
            self.norm2 = nn.LayerNorm(dim, eps=1e-5)
            self.mlp = Mlp()
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class GlobalAttentionConfidence(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj1 = nn.Linear(input_dim, dim, bias=True)
            self.block = Block()
            self.proj2 = nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=1, padding=1, padding_mode='reflect', bias=True),
                nn.GELU(approximate='none'),
                nn.Conv2d(dim, output_dim, 3, stride=1, padding=1, padding_mode='reflect', bias=True))
        def forward(self, tokens, BPHWC):
            if (not isinstance(tokens, torch.Tensor) or tokens.device.type != 'cpu'
                    or len(BPHWC) != 5 or any(type(n) is not int or n < 1 for n in BPHWC)):
                raise ValueError('Predictor accepts only CPU Torch and an explicit BPHWC shape')
            batch, views, height, width, channels = BPHWC
            if (channels != input_dim or min(height, width) < 2
                    or tuple(tokens.shape) != (batch, views, height, width, channels)
                    or tokens.dtype != self.proj1.weight.dtype):
                raise ValueError('Predictor input shape/dtype must match the explicit scene; no dtype fallback')
            x = self.proj1(tokens.reshape(batch, views * height * width, channels))
            x = self.block(x)
            x = x.reshape(batch * views, height, width, dim).permute(0, 3, 1, 2)
            x = self.proj2(x).permute(0, 2, 3, 1).flatten(1, 2)
            return torch.exp(x) + 1. if activation == 'expp1' else x
    return GlobalAttentionConfidence().eval()


def load_released_predictor(checkpoint, config, *, expected_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
                            expected_config_sha256):
    """Load the separately released Co-Me predictor on CPU in original BF16.

    Neither this loader nor plan construction runs the original VGGT model.
    The source checkpoint is not included in this repository.
    """
    import torch
    for path, expected in ((checkpoint, expected_checkpoint_sha256), (config, expected_config_sha256)):
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected)
                or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected):
            raise ValueError('Released confidence file SHA256 mismatch')
    configuration = json.loads(Path(config).read_text())
    fields = {'predictor_dims': [1024, 128, 1], 'predictor_active': 'expp1',
              'predictor_num_register': 0, 'loss_function': 'pairwise-rank',
              'inject_block_idx': 14, 'train_dtype': 'torch.bfloat16'}
    if any(configuration.get(k) != v for k, v in fields.items()):
        raise ValueError('Released confidence architecture or training metadata changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or v.dtype != torch.bfloat16
                                          or v.device.type != 'cpu' or not torch.isfinite(v).all()
                                          for v in state.values()):
        raise ValueError('Exact released finite CPU BF16 state is required')
    model = build_confidence_predictor().to(dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
