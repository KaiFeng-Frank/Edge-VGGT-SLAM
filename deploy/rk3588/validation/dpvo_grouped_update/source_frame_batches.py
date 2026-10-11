"""Research-only full-graph degree-batched DPVO Update transport.

Princeton-VL/DPVO commit 0ac95b656d1fda91c271d2a106460d19ad966fc7:
neighbors(kk,jj), SoftAgg(kk), SoftAgg(ii*12345+jj). Adjacent MIT LICENSE.
No neural execution, graph mutation, padding, dropping edges or adoption here.
The callback owns the original-parameter model/SDK and its numerical gate.
Only complete equal-degree source blocks become separate neural batch items.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np


PLAN_SOURCE_SHA256 = "04e240c2c35bd1359dbe5649fb038490a0a7ff99654d8d112ad7c83acc87ac2e"
CF_SOURCE_SHA256 = "7d89d91c64158d34f72b17091ebeaf06369ef71cb330495f6c3ec39092035c57"
INPUT_NAMES = ("net", "inp", "corr")
OUTPUT_NAMES = ("net", "delta", "weight")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _owned_readonly(value):
    result = np.array(value, order="C", copy=True)
    result.flags.writeable = False
    return result


def load_pinned_plan(path=None):
    """Load the exact frozen v32 plan; no replacement grouping algorithm."""
    path = Path(path) if path is not None else Path(__file__).with_name("source_frame_plan.py")
    if _sha(path.read_bytes()) != PLAN_SOURCE_SHA256:
        raise ValueError("Frozen complete-source plan SHA differs")
    name = "_v36_source_plan_" + PLAN_SOURCE_SHA256
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


@dataclass(frozen=True)
class SnapshotIdentity:
    graph_revision: int
    geometry_revision: int
    feature_generation: int
    recurrent_revision: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (
                self.graph_revision, self.geometry_revision,
                self.feature_generation, self.recurrent_revision)):
            raise ValueError("Explicit nonnegative integer snapshot revisions required")


@dataclass(frozen=True)
class DegreeBatch:
    degree: int
    patches: int
    source_slots: tuple
    target_slots: tuple
    canonical_edge_indices: np.ndarray  # [sources,degree,patches]

    def shape(self, channels):
        return (len(self.source_slots), channels, self.degree, self.patches)

    def metadata(self):
        return dict(source_slots=list(self.source_slots),
                    target_slots=[list(v) for v in self.target_slots],
                    sources=len(self.source_slots), degree=self.degree,
                    patches=self.patches, edges=int(self.canonical_edge_indices.size),
                    padding=False, dependency_groups_split=False,
                    physical_layout="source_batch,channels,target,patch")


@dataclass(frozen=True)
class DegreePlan:
    edges: int
    patches: int
    batches: tuple
    topology_sha256: str
    frame_capacity: int

    def metadata(self):
        return dict(edges=self.edges, patches=self.patches,
                    full_source_blocks=sum(len(b.source_slots) for b in self.batches),
                    neural_calls=len(self.batches), topology_sha256=self.topology_sha256,
                    all_edges_exactly_once=True, original_edge_order_restored=True,
                    padding=False, batch_items_independent_sources=True,
                    batches=[b.metadata() for b in self.batches])


def build_degree_plan(ii, jj, kk, *, patches_per_frame=48, frame_capacity=64,
                      max_target_degree=25, max_sources_per_batch=None,
                      plan_source=None):
    """Deterministic equal-D batches, optionally split only along source axis.

    A short final batch is an actual different static N shape, never padded.
    The caller must compile/support every requested (N,D,M), or reject it.
    """
    if max_sources_per_batch is not None and (
            type(max_sources_per_batch) is not int or max_sources_per_batch <= 0):
        raise ValueError("max_sources_per_batch must be a positive integer or None")
    frozen = load_pinned_plan(plan_source)
    sources = frozen.build_source_frame_plan(ii, jj, kk,
        patches_per_frame=patches_per_frame, frame_capacity=frame_capacity,
        max_target_degree=max_target_degree)
    grouped = {}
    for source in sources:
        grouped.setdefault(int(source.sorted_target_slots.size), []).append(source)
    batches = []
    for degree, members in sorted(grouped.items()):
        step = len(members) if max_sources_per_batch is None else max_sources_per_batch
        for offset in range(0, len(members), step):
            group = members[offset:offset + step]
            indices = np.stack([v.canonical_edge_indices.reshape(degree, patches_per_frame)
                                for v in group])
            batches.append(DegreeBatch(degree, patches_per_frame,
                tuple(v.source_slot for v in group),
                tuple(tuple(int(t) for t in v.sorted_target_slots) for v in group),
                _owned_readonly(indices)))
    indices = np.concatenate([v.canonical_edge_indices.ravel() for v in batches])
    if not np.array_equal(np.sort(indices), np.arange(ii.size, dtype=np.int64)):
        raise AssertionError("Every original edge must appear exactly once")
    fingerprint = hashlib.sha256()
    for value in (ii, jj, kk):
        fingerprint.update(np.ascontiguousarray(value).tobytes())
    fingerprint.update(str((patches_per_frame, frame_capacity)).encode("ascii"))
    return DegreePlan(int(ii.size), patches_per_frame, tuple(batches),
                      fingerprint.hexdigest(), frame_capacity)


def _validate_inputs(plan, net, inp, corr):
    values = (net, inp, corr)
    for name, value in zip(INPUT_NAMES, values):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 3:
            raise ValueError(name + " must be FP32 NumPy [1,edges,channels]")
        if value.shape[0] != 1 or value.shape[1] != plan.edges or value.shape[2] <= 0:
            raise ValueError("Full ordered graph input dimensions differ")
        if not np.isfinite(value).all():
            raise ValueError("Finite original full graph input required")
    if net.shape != inp.shape:
        raise ValueError("Original net/context channel shape differs")
    return values


def pack_batch(values, batch):
    """Own an FP32 [sources,C,D,M] copy, sorted independently per source."""
    if not isinstance(values, np.ndarray) or values.ndim != 3 or values.shape[0] != 1:
        raise ValueError("Expected original [1,edges,C]")
    picked = values[0, batch.canonical_edge_indices, :]
    return np.array(picked.transpose(0, 3, 1, 2), order="C", copy=True)


def execute_grouped_update(plan, net, inp, corr, execute, *,
                           snapshot_identity, current_identity,
                           max_payload_bytes=512 * 1024 * 1024):
    """Stage every source result, then restore full [1,E,C] atomically.

    execute(batch, feeds) returns exactly {net,delta,weight} CF FP32 arrays.
    No caller input is modified, and any failure/revision change returns nothing.
    This performs finite/shape/weight-domain validation only, not numerical
    acceptance. A board owner must compare original full Update and all relevant
    outputs at the unchanged gate before any experimental recurrent submission.
    """
    if not isinstance(snapshot_identity, SnapshotIdentity) or not callable(current_identity):
        raise ValueError("Explicit immutable snapshot and owner revision reader required")
    if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
        raise ValueError("Positive explicit staging byte budget required")
    original = _validate_inputs(plan, net, inp, corr)
    channels = net.shape[2]
    output_sizes = (channels, 2, 2)
    input_bytes = sum(v.nbytes for v in original)
    output_bytes = 4 * plan.edges * sum(output_sizes)
    # Conservative double payload reserve for advanced-index packing and
    # inverse-layout copies. This excludes model/SDK allocations and is not
    # a whole-process memory guarantee.
    largest_batch_bytes = max(8 * b.canonical_edge_indices.size * (
        sum(v.shape[2] for v in original) + sum(output_sizes)) for b in plan.batches)
    if input_bytes + output_bytes + largest_batch_bytes > max_payload_bytes:
        raise ValueError("Owned staging plus largest batch exceeds explicit byte budget")
    snapshot = tuple(_owned_readonly(v) for v in original)
    staged = {name: np.full((1, plan.edges, c), np.nan, dtype=np.float32)
              for name, c in zip(OUTPUT_NAMES, output_sizes)}
    for batch in plan.batches:
        if current_identity() != snapshot_identity:
            raise RuntimeError("Stale graph/geometry/features/recurrent snapshot")
        feeds = {name: pack_batch(value, batch) for name, value in zip(INPUT_NAMES, snapshot)}
        result = execute(batch, feeds)
        if not isinstance(result, dict) or set(result) != set(OUTPUT_NAMES):
            raise ValueError("Exactly three canonical output names required")
        for name, c in zip(OUTPUT_NAMES, output_sizes):
            value = result[name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != batch.shape(c):
                raise ValueError("Malformed channel-first source-batch output " + name)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite source-batch output " + name)
            if name == "weight" and (np.any(value < 0) or np.any(value > 1)):
                raise ValueError("Original terminal weight domain violated")
            canonical = value.transpose(0, 2, 3, 1).reshape(-1, c)
            staged[name][0, batch.canonical_edge_indices.ravel(), :] = canonical
        del result, feeds, value, canonical
    if current_identity() != snapshot_identity:
        raise RuntimeError("Stale result before complete full-graph restore")
    if not all(np.isfinite(v).all() for v in staged.values()):
        raise AssertionError("Incomplete full-graph staging; no partial result returned")
    return staged


def static_budget(plan, channels=384, correlation_channels=882):
    """Exact original Linear MAC count; no assumed hardware timing/speedup."""
    if channels != 384 or correlation_channels != 882:
        raise ValueError("Budget coefficients are only pinned original DPVO P3/C384")
    source_count = sum(len(b.source_slots) for b in plan.batches)
    patch_groups = source_count * plan.patches
    pair_groups = sum(len(b.source_slots) * b.degree for b in plan.batches)
    mac = 2699520 * plan.edges + 147456 * (patch_groups + pair_groups)
    return dict(Linear_MAC=mac, Linear_GMAC=mac / 1e9,
        unique_patch_groups=patch_groups, unique_source_target_groups=pair_groups,
        FP32_full_inputs_bytes=4 * plan.edges * (channels * 2 + correlation_channels),
        FP32_full_outputs_bytes=4 * plan.edges * (channels + 4),
        calls=len(plan.batches), MAC_reduced=False,
        graph_shapes=[dict(net=list(b.shape(channels)), inp=list(b.shape(channels)),
                           corr=list(b.shape(correlation_channels)),
                           delta=list(b.shape(2)), weight=list(b.shape(2))) for b in plan.batches],
        compilation_for_N_greater_than_one_verified=False,
        timing_or_20Hz_claim=False)
