"""CPU reference operations for the pinned DPVO neural-update research probe.

No CUDA extension, pretrained network or VO/BA implementation is imported here.
Indices and floating-point inputs must be CPU tensors. Floating reductions are
mathematically equivalent references, not a promise of CUDA bitwise equality.
DPVO commit: 0ac95b656d1fda91c271d2a106460d19ad966fc7.

Derived from DPVO (MIT), Copyright (c) 2022 Princeton Vision & Learning Lab.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn.functional as F

DPVO_COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
SOURCE_SHA256 = {
    "dpvo/fastba/ba.cpp": "cb0514d680cb1e44ba511cb82dde441fe35998f36706cc73322715f9706ed82f",
    "dpvo/altcorr/correlation_kernel.cu": "85667383d879c9093cee5b88ae231f562d98e751b4a549917ebda105a1a291e0",
    "dpvo/altcorr/correlation.py": "367fe95eff8c076b4ec3c4830ff9a9626a8bca4abadc9042a05bd3f58d577f4a",
    "dpvo/blocks.py": "9c9e2848da524829374e56bb1b14a891e9e2c19bc683af8e958e8982ba33d661",
    "dpvo/net.py": "1aee33d54f0b1cc1f3f842b60790ff1299ef9c1e20bca6467ce7114dca802acc",
    "dpvo/utils.py": "5bc5820014045b6979d9979ae80a2b519c64f9491ae83e909a77731152751a4a",
}
SCATTER_SEMANTICS_SOURCE = (
    "https://github.com/rusty1s/pytorch_scatter/tree/2.1.2/torch_scatter"
)


def _cpu(value, name):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU torch.Tensor")
    return value


def _indices(value, name):
    _cpu(value, name)
    if value.dtype != torch.int64:
        raise ValueError(f"{name} must have int64 dtype")
    return value


def neighbors(kk, jj):
    """Return int64 predecessor/successor edge indices, with -1 at boundaries.

    This is ba.cpp's stable jj sort inside each kk group. Tied destinations
    retain input-edge order. Negative group/destination IDs remain sortable.
    """
    _indices(kk, "kk")
    _indices(jj, "jj")
    if kk.ndim != 1 or jj.shape != kk.shape:
        raise ValueError("kk and jj must have the same one-dimensional shape")
    previous = torch.full_like(kk, -1)
    following = torch.full_like(kk, -1)
    if kk.numel() > 1:
        order = torch.argsort(jj, stable=True)
        order = order[torch.argsort(kk[order], stable=True)]
        same_group = kk[order[1:]] == kk[order[:-1]]
        left, right = order[:-1][same_group], order[1:][same_group]
        previous[right], following[left] = left, right
    return previous, following


def _scatter_inputs(src, index, dim):
    _cpu(src, "src")
    _indices(index, "index")
    if src.ndim < 1 or type(dim) is not int or not -src.ndim <= dim < src.ndim:
        raise ValueError("dim must identify a source dimension")
    dim %= src.ndim
    if index.ndim < 1 or index.ndim > src.ndim:
        raise ValueError("index must have one through src.ndim dimensions")
    if index.numel() and bool((index < 0).any()):
        raise ValueError("scatter group indices must be nonnegative")
    # Match torch_scatter.utils.broadcast, including per-batch group indices.
    if index.ndim == 1:
        index = index.reshape((1,) * dim + tuple(index.shape))
    index = index.reshape(tuple(index.shape) + (1,) * (src.ndim - index.ndim))
    try:
        index = index.expand(src.shape)
    except RuntimeError as exc:
        raise ValueError("index does not broadcast to src at dim") from exc
    size = list(src.shape)
    size[dim] = int(index.max()) + 1 if index.numel() else 0
    return index, dim, size


def scatter_sum(src, index, dim=1):
    """Dense group sum with missing groups zero and size max(index)+1.

    Only the arguments used by official DPVO are exposed; no out/dim_size API.
    Source order is retained in the input and expanded output indexing.
    """
    index, dim, size = _scatter_inputs(src, index, dim)
    result = src.new_zeros(size)
    return result.scatter_add_(dim, index, src)


def scatter_softmax(src, index, dim=1):
    """Stable independent softmax over each group, in original source order."""
    index, dim, size = _scatter_inputs(src, index, dim)
    if not src.is_floating_point():
        raise ValueError("scatter_softmax requires a floating-point source")
    maxima = src.new_full(size, -torch.inf)
    maxima.scatter_reduce_(dim, index, src, reduce="amax", include_self=True)
    exponential = (src - maxima.gather(dim, index)).exp()
    totals = src.new_zeros(size).scatter_add_(dim, index, exponential)
    return exponential / totals.gather(dim, index)


def _features(value, name, ndim):
    _cpu(value, name)
    if value.ndim != ndim or value.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(f"{name} must be a {ndim}D FP16/FP32/FP64 CPU tensor")
    return value


def _radius(radius):
    if type(radius) is not int or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    return radius


def _coordinates(coords, name):
    _cpu(coords, name)
    if coords.dtype != torch.float32 or not bool(torch.isfinite(coords).all()):
        raise ValueError(f"{name} must contain finite FP32 xy coordinates")
    # Native CUDA converts floor to signed int32. Refuse undefined conversions.
    if coords.numel() and bool((coords.abs() >= 2**30).any()):
        raise ValueError("coordinates exceed the defined native int32 sampling range")
    return coords


def patchify(net, coords, radius, mode="bilinear"):
    """DPVO altcorr forward: xy centers, floor grid, zero padding, then bilinear.

    net is [B,C,H,W], coords FP32 [B,M,2]. Bilinear output is
    [B,M,C,2R+1,2R+1]. As in the official wrapper, any other mode returns
    the raw zero-padded [B,M,C,2R+2,2R+2] integer samples. FP32 coordinate
    fractions are not silently cast to the feature dtype in this wrapper.
    """
    _features(net, "net", 4)
    _coordinates(coords, "coords")
    _radius(radius)
    if coords.ndim != 3 or coords.shape[0] != net.shape[0] or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [B,M,2] matching net batch")
    batch, channels, height, width = net.shape
    if min(batch, channels, height, width) < 1:
        raise ValueError("net batch/channel/spatial dimensions must be positive")
    count = coords.shape[1]
    diameter = 2 * radius + 2
    offset = torch.arange(-radius, radius + 2, dtype=torch.int64)
    base = coords.floor().to(torch.int64)
    xs = base[..., 0, None, None] + offset[None, :]
    ys = base[..., 1, None, None] + offset[:, None]
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    indices = (ys.clamp(0, height-1) * width + xs.clamp(0, width-1)).reshape(batch, 1, -1)
    samples = net.reshape(batch, channels, height*width).gather(2, indices.expand(-1, channels, -1))
    samples = samples.reshape(batch, channels, count, diameter, diameter).permute(0, 2, 1, 3, 4)
    samples = torch.where(valid[:, :, None], samples, samples.new_zeros(()))
    if mode != "bilinear":
        return samples
    fraction = (coords - coords.floor())[:, :, None, None, None]
    dx, dy = fraction.unbind(-1)
    d = diameter - 1
    x00 = (1-dy) * (1-dx) * samples[..., :d, :d]
    x01 = (1-dy) * dx * samples[..., :d, 1:]
    x10 = dy * (1-dx) * samples[..., 1:, :d]
    x11 = dy * dx * samples[..., 1:, 1:]
    return x00 + x01 + x10 + x11


def _corr_inputs(fmap1, fmap2, coords, ii, jj, radius):
    _features(fmap1, "fmap1", 5)
    _features(fmap2, "fmap2", 5)
    _coordinates(coords, "coords")
    _indices(ii, "ii")
    _indices(jj, "jj")
    _radius(radius)
    batch, patch_count, channels, ph, pw = fmap1.shape
    if min(batch, patch_count, channels, ph, pw, *fmap2.shape) < 1 or channels % 8:
        raise ValueError("feature dimensions must be positive and channels divisible by 8, as in CUDA")
    if fmap2.shape[0] != batch or fmap2.shape[2] != channels or fmap2.dtype != fmap1.dtype:
        raise ValueError("feature batch/channel/dtype must match")
    if ii.ndim != 1 or jj.shape != ii.shape or coords.shape != (batch, ii.numel(), 2, ph, pw):
        raise ValueError("ii/jj [E] and coords [B,E,2,patchH,patchW] must agree")
    if ii.numel() and (bool((ii < 0).any()) or bool((ii >= patch_count).any()) or bool((jj < 0).any()) or bool((jj >= fmap2.shape[1]).any())):
        raise ValueError("patch/view index outside its corresponding feature tensor")


def corr(fmap1, fmap2, coords, ii, jj, radius=1, dropout=1):
    """Forward CPU reference for the actual altcorr CUDA kernel.

    fmap1 [B,Npatch,C,patchH,patchW], fmap2 [B,Nview,C,H,W],
    coords FP32 [B,E,2,patchH,patchW] (xy). ii selects patches, jj views.
    Output [B,E,offsetX,offsetY,patchY,patchX], offset sizes 2R+1.
    Integer dot products are computed BEFORE the four bilinear terms; there
    is no channel normalization. CUDA's final x/y-offset permutation is kept.
    The dropout argument affects only upstream backward, not this forward.
    Training/dropout backward and CUDA floating-reduction bitwise identity
    are deliberately not asserted by this inference reference.
    """
    _corr_inputs(fmap1, fmap2, coords, ii, jj, radius)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
        raise ValueError("dropout must lie in [0,1]; it does not change forward")
    batch, _, _, ph, pw = fmap1.shape
    edge_count = ii.numel()
    diameter = 2 * radius + 2
    height, width = fmap2.shape[-2:]
    # Temporary gathered channel features are bounded to at most 64 edges.
    # No [B,E,C,H,W] selected-view copy or CUDA extension is constructed.
    raw = fmap1.new_empty(batch, edge_count, diameter, diameter, ph, pw)
    target_hwc = fmap2.permute(0, 1, 3, 4, 2)
    batch_index = torch.arange(batch)[:, None, None, None]
    for start in range(0, edge_count, 64):
        end = min(start + 64, edge_count)
        patch = fmap1[:, ii[start:end]]
        floor = coords[:, start:end].floor().to(torch.int64)
        view_index = jj[start:end][None, :, None, None]
        for oy in range(diameter):
            y = floor[:, :, 1] + (oy-radius)
            for ox in range(diameter):
                x = floor[:, :, 0] + (ox-radius)
                valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                sampled = target_hwc[batch_index, view_index, y.clamp(0,height-1), x.clamp(0,width-1)].permute(0,1,4,2,3)
                sampled = torch.where(valid[:, :, None], sampled, sampled.new_zeros(()))
                dot = (patch * sampled).sum(2)
                raw[:, start:end, oy, ox] = torch.where(valid, dot, dot.new_zeros(()))
    # coords derivatives are absent from CorrLayer.backward in the original.
    xy = coords.detach()
    dx = (xy[:, :, 0] - xy[:, :, 0].floor()).to(fmap1.dtype)[:, :, None, None]
    dy = (xy[:, :, 1] - xy[:, :, 1].floor()).to(fmap2.dtype)[:, :, None, None]
    d = diameter - 1
    out = (1-dx) * (1-dy) * raw[:, :, :d, :d]
    out = out + dx * (1-dy) * raw[:, :, :d, 1:]
    out = out + (1-dx) * dy * raw[:, :, 1:, :d]
    out = out + dx * dy * raw[:, :, 1:, 1:]
    return out.permute(0, 1, 3, 2, 4, 5)


def pyramidify(fmap, lvls=(1, 4)):
    """Official utils.py independent avg_pool2d kernel=stride=level pyramid."""
    _features(fmap, "fmap", 5)
    if min(fmap.shape) < 1 or not lvls:
        raise ValueError("fmap dimensions and the level list must be nonempty")
    batch, views, channels, height, width = fmap.shape
    result = []
    for level in lvls:
        if type(level) is not int or not 1 <= level <= min(height, width):
            raise ValueError("pooling level must be a positive integer within spatial dimensions")
        pooled = F.avg_pool2d(fmap.reshape(batch*views, channels, height, width), level, stride=level)
        result.append(pooled.reshape(batch, views, channels, height//level, width//level))
    return result


def corr_two_levels(gmap, fmap, coords, ii, jj, radius=3, levels=(1, 4), dropout=1):
    """Official CorrBlock forward with level interleaved as the innermost axis.

    The default two levels are (1,4); output is [B,E,2*(2R+1)^2*P_H*P_W].
    Unlike concatenating complete level vectors, stack(...,-1).reshape(...)
    retains the ordering consumed by the original Update.corr Linear weights.
    """
    if len(levels) != 2:
        raise ValueError("corr_two_levels requires exactly two ordered levels")
    pyramid = pyramidify(fmap, levels)
    outputs = [corr(gmap, target, coords/level, ii, jj, radius, dropout) for target, level in zip(pyramid, levels)]
    return torch.stack(outputs, -1).reshape(gmap.shape[0], ii.numel(), 2*(2*radius+1)**2*gmap.shape[-2]*gmap.shape[-1])
