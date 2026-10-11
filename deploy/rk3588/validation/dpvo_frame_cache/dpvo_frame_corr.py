"""Immutable per-frame DPVO correlation cache, isolated v36 CPU research.

Derived from DPVO MIT 0ac95b656d1fda91c271d2a106460d19ad966fc7 and
byte-frozen v27 correlation. Copyright (c) 2022 Princeton Vision & Learning
Lab. See LICENSE. No Torch/model/image/BA/device execution is imported.

Each stable frame_id is bound to the SAME read-only C-order feature ndarray
objects for its lifetime. The owner must never mutate their values (including
through another alias) or recycle an input arena while that frame is retained.
Native preparation makes an independent snapshot once. There is no per-query
feature hashing, concatenation or pointer-identity-only cache. A new sequence
requires reset(); changed feature objects under a resident frame_id are errors.

Current retained-frame order is supplied on EVERY call; original ii/jj/kk and
edge order are preserved. Removal and capacity eviction destroy only immutable
frame snapshots. One RLock serializes query, reset, eviction and close; every
returned output is independently owned. No global affinity/OMP state changes.
"""
import collections
import copy
import ctypes as ct
import hashlib
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ABI_VERSION = 1
MEMORY_LIMIT = 512 * 1024 * 1024
PARENT_SOURCE_PINS = {
    "dpvo_corr.cpp": "3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990",
    "dpvo_corr.h": "1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932",
}


class CorrelationError(RuntimeError):
    pass


class NativeStats(ct.Structure):
    _fields_ = [("abi_version", ct.c_uint32), ("reserved", ct.c_uint32),
                ("context_bytes", ct.c_uint64), ("output_elements", ct.c_uint64),
                ("scan_seconds", ct.c_double), ("layout_seconds", ct.c_double),
                ("pool_seconds", ct.c_double), ("kernel_seconds", ct.c_double),
                ("total_seconds", ct.c_double), ("threads", ct.c_int32), ("neon", ct.c_int32)]


def _plain(stats):
    return {key: getattr(stats, key) for key, _ in stats._fields_}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_parent_sources(directory):
    actual = {name: sha256(Path(directory) / name) for name in PARENT_SOURCE_PINS}
    if actual != PARENT_SOURCE_PINS:
        raise CorrelationError("Byte-frozen v27 parent source mismatch")
    return actual


def load_library(path, expected_sha256):
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("Explicit lowercase native library SHA256 required")
    path = Path(path).resolve()
    if sha256(path) != expected_sha256:
        raise CorrelationError("Native library SHA256 mismatch")
    api = ct.CDLL(str(path))
    api.dpvo_corr_v36_abi_version.restype = ct.c_uint32
    api.dpvo_corr_v36_stats_size.restype = ct.c_uint64
    if (api.dpvo_corr_v36_abi_version() != ABI_VERSION
            or api.dpvo_corr_v36_stats_size() != ct.sizeof(NativeStats)):
        raise CorrelationError("Immutable frame correlation ABI mismatch")
    api.dpvo_corr_v36_parent_source_sha256.restype = ct.c_char_p
    if api.dpvo_corr_v36_parent_source_sha256().decode("ascii") != PARENT_SOURCE_PINS["dpvo_corr.cpp"]:
        raise CorrelationError("Native parent provenance mismatch")
    api.dpvo_corr_v36_build_info.restype = ct.c_char_p
    fptr, iptr = ct.POINTER(ct.c_float), ct.POINTER(ct.c_int64)
    stats, error = ct.POINTER(NativeStats), ct.POINTER(ct.c_char)
    api.dpvo_corr_v36_prepare_frame.argtypes = [
        fptr, ct.c_uint64, fptr, ct.c_uint64, *([ct.c_int64] * 6), ct.c_int32,
        ct.POINTER(ct.c_void_p), stats, error, ct.c_size_t]
    api.dpvo_corr_v36_prepare_frame.restype = ct.c_int
    api.dpvo_corr_v36_correlate_frames.argtypes = [
        ct.POINTER(ct.c_void_p), ct.c_uint64, fptr, ct.c_uint64, iptr, iptr,
        ct.c_uint64, ct.c_int64, ct.c_int32, fptr, ct.c_uint64, stats, error, ct.c_size_t]
    api.dpvo_corr_v36_correlate_frames.restype = ct.c_int
    api.dpvo_corr_v36_destroy_frame.argtypes = [ct.POINTER(ct.c_void_p), error, ct.c_size_t]
    api.dpvo_corr_v36_destroy_frame.restype = ct.c_int
    return api


def _pointer(value, dtype):
    return value.ctypes.data_as(ct.POINTER(dtype))


def _frame_key(frame_id):
    if (type(frame_id) not in (str, int, float) or frame_id == ""
            or (type(frame_id) is float and not math.isfinite(frame_id))):
        raise ValueError("Finite scalar stable frame_id required")
    # Typed keys prevent Python's 1 == 1.0 identity ambiguity.
    return type(frame_id), frame_id


def _array(value, dtype, ndim, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D NumPy {np.dtype(dtype)} ndarray")
    if value.nbytes > MEMORY_LIMIT:
        raise ValueError(name + " exceeds 512MiB buffer budget")
    return value


class PerFrameCorrelation:
    """Drop-in correlate callback for the frozen DPVOStream, with bounded cache.

    Default C128, patch3x3 and M48 are the original FAST feature contract.
    Other divisible-by-eight channels/patch shapes are for tiny protocol tests.
    Metrics histories are bounded; cumulative counters retain all calls.
    Construction verifies/loads the library once, outside per-frame queries.
    """
    def __init__(self, library, *, library_sha256, patch_count=48, threads=4,
                 max_frames=64, max_bytes=MEMORY_LIMIT, channels=128,
                 patch_shape=(3, 3), history_limit=4096):
        for name, value, upper in (("patch_count", patch_count, 2**31-1),
                                  ("channels", channels, 2**31-1),
                                  ("threads", threads, 64), ("max_frames", max_frames, 64),
                                  ("max_bytes", max_bytes, MEMORY_LIMIT),
                                  ("history_limit", history_limit, 65536)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(name + " outside bounded positive integer domain")
        if (channels % 8 or not isinstance(patch_shape, (tuple, list)) or len(patch_shape) != 2
                or any(type(n) is not int or not 1 <= n <= 1024 for n in patch_shape)):
            raise ValueError("C divisible by8 and positive bounded patch shape required")
        self._lock = threading.RLock()
        self._api = load_library(library, library_sha256)
        self.library_sha256 = library_sha256
        self.build_info = self._api.dpvo_corr_v36_build_info().decode("ascii")
        self.M, self.channels, self.patch_shape = patch_count, channels, tuple(patch_shape)
        self.threads, self.max_frames, self.max_bytes = threads, max_frames, max_bytes
        self._entries = collections.OrderedDict()
        self._geometry = None
        self._bytes = 0
        self._peak_bytes = 0
        self._closed = False
        self._epoch = 0
        self._counts = dict(attempted_calls=0, completed_calls=0, failed_calls=0,
                            prepared_frames=0, cache_hits=0, evicted_frames=0,
                            destroyed_frames=0, resets=0)
        self._times = dict(total_seconds=0.0, preparation_seconds=0.0,
                           query_seconds=0.0, kernel_seconds=0.0)
        self._calls = collections.deque(maxlen=history_limit)
        self._preparations = collections.deque(maxlen=history_limit)
        self._evictions = collections.deque(maxlen=history_limit)
        self.last_call = None

    def _open(self):
        if self._closed:
            raise CorrelationError("frame correlation cache closed")

    def _describe(self, frames):
        if not isinstance(frames, (tuple, list)) or not 1 <= len(frames) <= 64:
            raise ValueError("1..64 current retained frame records required")
        rows, seen, geometry = [], set(), self._geometry
        for record in frames:
            if not isinstance(record, Mapping):
                raise ValueError("Immutable frame mapping required")
            key = _frame_key(record.get("frame_id"))
            if key in seen:
                raise ValueError("Duplicate current stable frame_id")
            seen.add(key)
            gmap = _array(record.get("gmap"), np.float32, 4, "gmap")
            fmap = _array(record.get("fmap"), np.float32, 3, "fmap")
            if (gmap.shape != (self.M, self.channels, *self.patch_shape)
                    or fmap.shape[0] != self.channels or min(fmap.shape[1:]) < 4
                    or max(fmap.shape) > 2**31-1):
                raise ValueError("Per-frame feature geometry differs")
            if any(a.flags.writeable or not a.flags.c_contiguous for a in (gmap, fmap)):
                raise ValueError("Feature arrays must be immutable C-order FP32 objects")
            if geometry is None:
                geometry = fmap.shape[1:]
            if fmap.shape[1:] != geometry:
                raise ValueError("Feature geometry changed: explicitly reset between sequences")
            entry = self._entries.get(key)
            if entry is not None and (entry["gmap"] is not gmap or entry["fmap"] is not fmap):
                raise ValueError("Resident frame_id replaced feature objects: explicitly reset")
            size = gmap.nbytes + fmap.nbytes + self.channels * (fmap.shape[1] // 4) * (fmap.shape[2] // 4) * 4
            rows.append(dict(key=key, frame_id=record["frame_id"], gmap=gmap, fmap=fmap, bytes=size))
        return rows, geometry

    def _destroy(self, key, reason):
        entry = self._entries[key]
        error = ct.create_string_buffer(512)
        status = self._api.dpvo_corr_v36_destroy_frame(ct.byref(entry["handle"]), error, len(error))
        if status:
            raise CorrelationError(error.value.decode("utf8", errors="replace"))
        del self._entries[key]
        self._bytes -= entry["bytes"]
        self._counts["destroyed_frames"] += 1
        if reason not in ("close", "reset"):
            self._counts["evicted_frames"] += 1
        self._evictions.append(dict(frame_id=entry["frame_id"], reason=reason, bytes=entry["bytes"], epoch=self._epoch))

    def _prepare(self, row):
        begin = time.monotonic()
        handle, stats, error = ct.c_void_p(), NativeStats(), ct.create_string_buffer(512)
        gmap, fmap = row["gmap"], row["fmap"]
        status = self._api.dpvo_corr_v36_prepare_frame(
            _pointer(gmap, ct.c_float), gmap.size, _pointer(fmap, ct.c_float), fmap.size,
            self.M, self.channels, *self.patch_shape, *fmap.shape[1:], self.threads,
            ct.byref(handle), ct.byref(stats), error, len(error))
        if status:
            # Parent prepare is transactional; still release any returned handle.
            if handle.value:
                cleanup_error = ct.create_string_buffer(512)
                cleanup = self._api.dpvo_corr_v36_destroy_frame(ct.byref(handle), cleanup_error, len(cleanup_error))
                if cleanup:
                    raise CorrelationError("Prepare and cleanup failed: " + cleanup_error.value.decode("utf8", errors="replace"))
            raise CorrelationError(error.value.decode("utf8", errors="replace"))
        if stats.context_bytes != row["bytes"] or not handle.value:
            cleanup_error = ct.create_string_buffer(512)
            self._api.dpvo_corr_v36_destroy_frame(ct.byref(handle), cleanup_error, len(cleanup_error))
            raise CorrelationError("Native prepared frame byte/handle contract differs")
        entry = dict(row, handle=handle)
        self._entries[row["key"]] = entry
        self._bytes += row["bytes"]
        self._peak_bytes = max(self._peak_bytes, self._bytes)
        self._counts["prepared_frames"] += 1
        timing = dict(frame_id=row["frame_id"], epoch=self._epoch,
                      total_seconds=time.monotonic()-begin, native=_plain(stats),
                      context_bytes=row["bytes"], feature_snapshot_owned=True)
        self._preparations.append(timing)
        return timing

    def __call__(self, coords, *, ii, jj, kk, frames):
        begin = time.monotonic()
        with self._lock:
            self._counts["attempted_calls"] += 1
            try:
                self._open()
                ii = _array(ii, np.int64, 1, "ii")
                jj = _array(jj, np.int64, 1, "jj")
                kk = _array(kk, np.int64, 1, "kk")
                coords = _array(coords, np.float32, 5, "coords")
                rows, geometry = self._describe(frames)
                edges = kk.size
                if (not edges or ii.shape != jj.shape or jj.shape != kk.shape
                        or coords.shape != (1, edges, 2, *self.patch_shape)):
                    raise ValueError("Nonempty equal edge IDs and original coordinate shape required")
                if (np.any(ii < 0) or np.any(ii >= len(rows)) or np.any(jj < 0)
                        or np.any(jj >= len(rows)) or np.any(kk < 0)
                        or not np.array_equal(ii, kk // self.M)):
                    raise ValueError("Original source ownership or retained-frame index range differs")
                count = edges * 98 * math.prod(self.patch_shape)
                if count * 4 > MEMORY_LIMIT:
                    raise ValueError("Output exceeds 512MiB buffer budget")
                # O(E+F) membership only; do not sort/deduplicate actual edges.
                needed_flags = np.zeros(len(rows), bool)
                needed_flags[ii] = True
                needed_flags[jj] = True
                needed = [rows[int(i)] for i in np.flatnonzero(needed_flags)]
                needed_keys = {r["key"] for r in needed}
                if len(needed) > self.max_frames or sum(r["bytes"] for r in needed) > self.max_bytes:
                    raise ValueError("Referenced immutable frames exceed explicit cache capacity")
                validation_end = time.monotonic()
                retained = {r["key"] for r in rows}
                for key in tuple(self._entries):
                    if key not in retained:
                        self._destroy(key, "not_retained")
                missing = [row for row in needed if row["key"] not in self._entries]
                missing_bytes = sum(row["bytes"] for row in missing)
                # Evict least-recently-used nonreferenced frames only. Every frame
                # referenced by this query must stay alive through native return.
                for key in tuple(self._entries):
                    if (len(self._entries) + len(missing) <= self.max_frames
                            and self._bytes + missing_bytes <= self.max_bytes):
                        break
                    if key not in needed_keys:
                        self._destroy(key, "capacity_unreferenced")
                prep_begin = time.monotonic()
                preparations = [self._prepare(row) for row in missing]
                prep_end = time.monotonic()
                preparation_seconds = prep_end-prep_begin
                self._geometry = geometry
                self._counts["cache_hits"] += len(needed)-len(missing)
                for row in needed:
                    self._entries.move_to_end(row["key"])
                handles = (ct.c_void_p * len(rows))()
                for i, row in enumerate(rows):
                    entry = self._entries.get(row["key"])
                    if entry is not None:
                        handles[i] = entry["handle"].value
                kk, jj, coords = (np.ascontiguousarray(a) for a in (kk, jj, coords))
                output = np.empty((1, edges, 98 * math.prod(self.patch_shape)), np.float32)
                stats, error = NativeStats(), ct.create_string_buffer(512)
                native_begin = time.monotonic()
                status = self._api.dpvo_corr_v36_correlate_frames(
                    handles, len(rows), _pointer(coords, ct.c_float), coords.size,
                    _pointer(kk, ct.c_int64), _pointer(jj, ct.c_int64), edges, edges, 3,
                    _pointer(output, ct.c_float), output.size, ct.byref(stats), error, len(error))
                native_end = time.monotonic()
                if status:
                    raise CorrelationError(error.value.decode("utf8", errors="replace"))
                # Native checks every computed FP32 result for nonfinite/overflow.
                # The stream retains its own independent output-array validation.
                total = time.monotonic()-begin
                query = dict(total_seconds=native_end-prep_end,
                             native_call_seconds=native_end-native_begin, native=_plain(stats),
                             includes_prepare=False, output_finite_validated_by_native=True)
                call = dict(index=self._counts["completed_calls"], epoch=self._epoch, edges=int(edges),
                            prepared_frame_ids=[r["frame_id"] for r in missing],
                            prepare_misses=len(missing), cache_hits=len(needed)-len(missing),
                            current_context_bytes=self._bytes, cached_frames=len(self._entries),
                            total_seconds=total, validation_and_lock_seconds=validation_end-begin,
                            eviction_seconds=prep_begin-validation_end,
                            preparation_seconds=preparation_seconds, native_query=query,
                            preparation=preparations, edge_order_preserved=True,
                            no_historical_feature_concatenation=True)
                self._calls.append(call)
                self.last_call = copy.deepcopy(call)
                self._counts["completed_calls"] += 1
                self._times["total_seconds"] += total
                self._times["preparation_seconds"] += preparation_seconds
                self._times["query_seconds"] += query["total_seconds"]
                self._times["kernel_seconds"] += stats.kernel_seconds
                return output
            except Exception:
                self._counts["failed_calls"] += 1
                raise

    def reset(self):
        """Explicit new-sequence boundary. Wait for query, free all snapshots."""
        with self._lock:
            self._open()
            for key in tuple(self._entries):
                self._destroy(key, "reset")
            self._geometry = None
            self._epoch += 1
            self._counts["resets"] += 1

    def close(self):
        with self._lock:
            if self._closed:
                return
            for key in tuple(self._entries):
                self._destroy(key, "close")
            self._closed = True

    def metrics(self):
        with self._lock:
            result = dict(self._counts, abi_version=ABI_VERSION, library_sha256=self.library_sha256,
                          build_info=self.build_info, parent_source_pins=PARENT_SOURCE_PINS,
                          threads=self.threads, patches_per_frame=self.M, channels=self.channels,
                          patch_shape=list(self.patch_shape), max_frames=self.max_frames,
                          max_bytes=self.max_bytes, current_cached_frames=len(self._entries),
                          current_context_bytes=self._bytes, peak_context_bytes=self._peak_bytes,
                          closed=self._closed, epoch=self._epoch, timing_sums=self._times,
                          calls=list(self._calls), preparations=list(self._preparations),
                          evictions=list(self._evictions), history_limit=self._calls.maxlen,
                          call_history_dropped=max(0, self._counts["completed_calls"]-len(self._calls)),
                          preparation_history_dropped=max(0, self._counts["prepared_frames"]-len(self._preparations)),
                          feature_input_arrays_strongly_retained=True,
                          native_feature_snapshots_owned=True, historical_features_hashed_per_query=False,
                          capacity_bytes_scope="native FP32 frame payload only; shared input references, Python metadata and owned query output separate",
                          Torch_used=False, CUDA_used=False, NPU_used=False,
                          VO_or_BA_implemented=False, ARM_numerical_validation_completed=False,
                          realtime_validated=False)
            return copy.deepcopy(result)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
