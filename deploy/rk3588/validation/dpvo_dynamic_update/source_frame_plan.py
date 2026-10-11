"""Pure NumPy layout plan for a complete source-frame DPVO Update block.

This is topology validation, not neural execution. The proposed tensor layout is
[batch, sorted_target, source_patch, channels]. No edge, recurrent state or group
is dropped. Real-valued dependency equivalence is not FP32 equivalence evidence.
"""
from dataclasses import dataclass

import numpy as np


def _owned_readonly(value):
    result = np.array(value, dtype=np.int64, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class SourceFramePlan:
    source_slot: int
    patches_per_frame: int
    sorted_target_slots: np.ndarray
    original_edge_indices: np.ndarray
    canonical_edge_indices: np.ndarray
    original_to_canonical: np.ndarray

    @property
    def tensor_shape_without_batch_channels(self):
        return (int(self.sorted_target_slots.size), self.patches_per_frame)

    def to_metadata(self):
        return {
            "source_slot": self.source_slot,
            "sorted_target_slots": self.sorted_target_slots.tolist(),
            "degree": int(self.sorted_target_slots.size),
            "edges": int(self.original_edge_indices.size),
            "canonical_shape": list(self.tensor_shape_without_batch_channels),
            "original_edge_order_preserved_on_restore": True,
            "all_patch_target_members_present": True,
            "duplicate_edges": False,
        }


def build_source_frame_plan(ii, jj, kk, *, patches_per_frame=96,
                            frame_capacity=64, max_target_degree=25):
    """Return one complete immutable plan per sorted source frame.

    Inputs are the exact ordered graph's C-order int64 vectors. Each distinct
    (source,target) pair must contain every source patch exactly once. Reject
    missing/duplicate members, mixed source patch ownership, collisions and
    overflow rather than padding, truncating or changing grouping semantics.
    """
    for name, value in (("ii", ii), ("jj", jj), ("kk", kk)):
        if not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.ndim != 1:
            raise ValueError(name + " must be a one-dimensional int64 NumPy array")
    if not ii.size or ii.shape != jj.shape or ii.shape != kk.shape:
        raise ValueError("Expected nonempty equally sized edge vectors")
    if any(type(v) is not int or v <= 0 for v in (patches_per_frame, frame_capacity, max_target_degree)):
        raise ValueError("Capacities must be positive integers")
    if frame_capacity > 12345:
        raise ValueError("Frame domain must exclude ii*12345+jj label collisions")
    if frame_capacity > np.iinfo(np.int64).max // patches_per_frame:
        raise ValueError("Patch ID domain overflows int64")
    if np.any(ii < 0) or np.any(ii >= frame_capacity) or np.any(jj < 0) or np.any(jj >= frame_capacity):
        raise ValueError("Source or target frame outside the declared domain")
    if np.any(kk < 0) or np.any(kk >= patches_per_frame * frame_capacity) or not np.array_equal(ii, kk // patches_per_frame):
        raise ValueError("Patch ownership must be ii == kk//patches_per_frame")
    plans = []
    for source in np.unique(ii):
        original = np.flatnonzero(ii == source)
        targets = np.unique(jj[original])
        degree = int(targets.size)
        if degree > max_target_degree:
            raise ValueError("Source target degree exceeds declared capacity; no edge removal")
        if original.size != degree * patches_per_frame:
            raise ValueError("Missing or duplicate edge in complete source Cartesian product")
        local_order = np.lexsort((kk[original] % patches_per_frame, jj[original]))
        canonical = original[local_order]
        if not np.array_equal(jj[canonical], np.repeat(targets, patches_per_frame)):
            raise ValueError("Incomplete target membership")
        expected_patches = np.arange(int(source) * patches_per_frame,
                                     (int(source) + 1) * patches_per_frame, dtype=np.int64)
        if not np.array_equal(kk[canonical].reshape(degree, patches_per_frame),
                              np.broadcast_to(expected_patches, (degree, patches_per_frame))):
            raise ValueError("Each target must contain every source patch exactly once")
        inverse = np.argsort(local_order)
        if not np.array_equal(canonical[inverse], original):
            raise AssertionError("Internal permutation roundtrip failure")
        plans.append(SourceFramePlan(int(source), patches_per_frame,
                                    _owned_readonly(targets), _owned_readonly(original),
                                    _owned_readonly(canonical), _owned_readonly(inverse)))
    return tuple(plans)


def to_canonical(values, plan):
    """Copy graph row values [B,E,C] to one source's [B,D,M,C] layout."""
    if not isinstance(values, np.ndarray) or values.ndim != 3:
        raise ValueError("Expected NumPy [batch,edges,channels] values")
    if int(plan.canonical_edge_indices.max()) >= values.shape[1]:
        raise ValueError("Plan references an absent global edge")
    degree, patches = plan.tensor_shape_without_batch_channels
    return np.array(values[:, plan.canonical_edge_indices, :].reshape(
        values.shape[0], degree, patches, values.shape[2]), copy=True, order="C")


def to_source_original_order(values, plan):
    """Copy [B,D,M,C] back to source-filtered original order.

    The owner must scatter these rows at plan.original_edge_indices only after
    all source-block outputs have passed validation for the same graph revision.
    """
    if not isinstance(values, np.ndarray) or values.ndim != 4 or values.shape[1:3] != plan.tensor_shape_without_batch_channels:
        raise ValueError("Expected matching [batch,degree,patches,channels] values")
    flattened = values.reshape(values.shape[0], -1, values.shape[3])
    return np.array(flattened[:, plan.original_to_canonical, :], copy=True, order="C")
