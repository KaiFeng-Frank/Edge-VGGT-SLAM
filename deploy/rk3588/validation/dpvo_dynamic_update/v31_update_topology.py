"""Pure NumPy topology inputs for the frozen original DPVO Update adapter.

Derived from Princeton Vision & Learning Lab DPVO, commit
0ac95b656d1fda91c271d2a106460d19ad966fc7, and the frozen v28 external-topology
adapter. Only integer topology is computed here; no model/backend is imported.
Two stable neighbor sorts and one stable sort per group domain replace the
O(groups*edges) member scans. Output edge order is never changed.

Copyright (c) 2022 Princeton Vision & Learning Lab.
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

Callers must keep input values stable during a call. Each returned Topology
owns its arrays; changing returned arrays cannot alter a cached template.
Cache identity is exact ii/jj/kk values plus capacities, never a pointer, shape,
hash, source ID or assumed graph generation. Retained NumPy payload is bounded;
fixed Python bookkeeping overhead is not represented by the payload counter.
"""
from dataclasses import dataclass, fields
from collections.abc import Mapping
import copy
import threading
import time

import numpy as np

ORIGINAL_ADAPTER_SHA256 = '3f755c8e24d092fa55fdf698d87c3d9ba8944487ee95d4e5afc47554d31d1380'
INPUT_NAMES = ('net', 'inp', 'corr', 'previous_indices', 'previous_valid',
               'following_indices', 'following_valid', 'kk_indices', 'kk_valid',
               'kk_inverse', 'ij_indices', 'ij_valid', 'ij_inverse')
TOPOLOGY_NAMES = INPUT_NAMES[3:]
MAX_TABLE_BYTES = 32 << 20
MAX_EDGES = 98304


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _positive_int(value, name):
    _require(type(value) is int and value > 0, name + ' must be a positive Python int')
    return value


def _capacities(group_caps):
    if group_caps is None:
        group_caps = {}
    _require(isinstance(group_caps, Mapping) and set(group_caps).issubset({'kk', 'ij'}),
             'group_caps may contain only kk and ij')
    result = []
    for name in ('kk', 'ij'):
        value = group_caps.get(name)
        if value is not None:
            _require(isinstance(value, (tuple, list)) and len(value) == 2,
                     name + ' capacity must be (groups, slots)')
            value = tuple(_positive_int(v, name + ' capacity') for v in value)
        result.append(value)
    return tuple(result)


def _snapshot(ii, jj, kk, max_edges):
    _positive_int(max_edges, 'max_edges')
    for name, value in zip(('ii', 'jj', 'kk'), (ii, jj, kk)):
        _require(isinstance(value, np.ndarray) and value.dtype == np.int64 and value.ndim == 1,
                 name + ' must be one-dimensional NumPy int64')
    _require(ii.shape == jj.shape == kk.shape and 0 < ii.size <= max_edges,
             'ii/jj/kk must have identical nonempty bounded edge shape')
    return tuple(value.copy(order='C') for value in (ii, jj, kk))


@dataclass(frozen=True)
class Topology:
    previous_indices: np.ndarray
    previous_valid: np.ndarray
    following_indices: np.ndarray
    following_valid: np.ndarray
    kk_indices: np.ndarray
    kk_valid: np.ndarray
    kk_inverse: np.ndarray
    ij_indices: np.ndarray
    ij_valid: np.ndarray
    ij_inverse: np.ndarray
    ii: np.ndarray
    jj: np.ndarray
    kk: np.ndarray
    kk_labels: np.ndarray
    ij_labels: np.ndarray

    def arrays(self):
        """The ten owned NumPy arrays, in frozen INPUT_NAMES[3:] order."""
        return tuple(getattr(self, name) for name in TOPOLOGY_NAMES)

    def numpy_dict(self):
        """Owned copies of all inputs/IDs/labels for artifacts, not a hot path."""
        return {f.name: getattr(self, f.name).copy(order='C') for f in fields(self)}

    def copy(self):
        return Topology(**self.numpy_dict())

    @property
    def nbytes(self):
        # Every field owns a distinct array allocation.
        return sum(getattr(self, f.name).nbytes for f in fields(self))

    def summary(self, *, batch=1, channels=384):
        _positive_int(batch, 'batch'); _positive_int(channels, 'channels')
        groups = {}
        for name in ('kk', 'ij'):
            index, valid = getattr(self, name + '_indices'), getattr(self, name + '_valid')
            count = valid.sum(axis=1, dtype=np.int64)
            groups[name] = dict(capacity=list(index.shape), actual_groups=int(np.count_nonzero(count)),
                                group_sizes=count.tolist(), padding_slots=int(valid.size - count.sum()),
                                single_gathered_FP32_tensor_bytes=batch * index.size * channels * 4,
                                labels=getattr(self, name + '_labels').tolist())
        return dict(kind='external_dynamic_DPVO_Update_topology_NumPy', edge_count=self.ii.size,
                    groups=groups, payload_bytes=self.nbytes,
                    neighbor_rule='stable jj sort inside kk; tied jj retain input edge order',
                    ij_group_rule='original int64 ii*12345+jj, including wraparound/collisions',
                    group_member_rule='original edge-row order within sorted group labels',
                    workspace_bytes_are_not_RSS_or_peak_memory=True)


def _group_info(labels, capacity, name):
    # Stable order preserves original edge-row order inside every group.
    order = np.argsort(labels, kind='stable')
    sorted_labels = labels[order]
    starts = np.r_[np.int64(0), np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1]
    counts = np.diff(np.r_[starts, labels.size])
    unique = sorted_labels[starts].copy(order='C')
    actual_g, actual_l = unique.size, int(counts.max())
    g, l = (actual_g, actual_l) if capacity is None else capacity
    _require(g >= actual_g and l >= actual_l, name + ' group capacity overflow; no truncation')
    rows = np.repeat(np.arange(actual_g, dtype=np.int64), counts)
    inverse = np.empty(labels.size, dtype=np.int64)
    inverse[order] = rows
    slots = np.arange(labels.size, dtype=np.int64) - np.repeat(starts, counts)
    return dict(g=g, l=l, order=order, rows=rows, slots=slots, inverse=inverse, labels=unique)


def _table(info):
    indices = np.zeros((info['g'], info['l']), dtype=np.int64)
    valid = np.zeros((info['g'], info['l']), dtype=np.float32)
    indices[info['rows'], info['slots']] = info['order']
    valid[info['rows'], info['slots']] = np.float32(1)
    return indices, valid, info['inverse'], info['labels']


def _build(snapshots, capacities, max_table_bytes):
    _positive_int(max_table_bytes, 'max_table_bytes')
    ii, jj, kk = snapshots
    with np.errstate(over='ignore'):
        ij_labels = np.add(np.multiply(ii, np.int64(12345), dtype=np.int64), jj, dtype=np.int64)
    kinfo = _group_info(kk, capacities[0], 'kk')
    jinfo = _group_info(ij_labels, capacities[1], 'ij')
    # Python-int products are evaluated before allocating either padded table.
    table_bytes = 12 * (int(kinfo['g']) * int(kinfo['l']) + int(jinfo['g']) * int(jinfo['l']))
    _require(table_bytes <= max_table_bytes, 'Group tables exceed max_table_bytes before allocation')
    ki, kv, kr, kl = _table(kinfo)
    ji, jv, jr, jl = _table(jinfo)
    e = ii.size
    previous, following = np.zeros(e, dtype=np.int64), np.zeros(e, dtype=np.int64)
    pm, nm = np.zeros((1, e, 1), np.float32), np.zeros((1, e, 1), np.float32)
    order = np.argsort(jj, kind='stable')
    order = order[np.argsort(kk[order], kind='stable')]
    same = kk[order[1:]] == kk[order[:-1]]
    left, right = order[:-1][same], order[1:][same]
    previous[right], following[left] = left, right
    pm.reshape(-1)[right] = np.float32(1)
    nm.reshape(-1)[left] = np.float32(1)
    return Topology(previous, pm, following, nm, ki, kv, kr, ji, jv, jr, ii, jj, kk, kl, jl)


def build_topology(ii, jj, kk, *, group_caps=None,
                   max_table_bytes=MAX_TABLE_BYTES, max_edges=MAX_EDGES):
    """Dynamic exact-value topology; guarded padding, no O(groups*edges) scans."""
    capacities = _capacities(group_caps)
    snapshots = _snapshot(ii, jj, kk, max_edges)
    return _build(snapshots, capacities, max_table_bytes)


class TopologyCache:
    """Small LRU of private templates, returning independent owned arrays.

    The payload cap includes templates' ii/jj/kk key arrays; those allocations
    are shared with the template identity fields and counted exactly once.
    Oversized valid topology is built/returned but never retained. Building
    still obeys max_table_bytes and max_edges. Concurrent input mutation by a
    caller is unsupported; the cache itself serializes get/clear/close.
    """
    def __init__(self, *, max_entries=2, max_bytes=32 << 20,
                 max_table_bytes=MAX_TABLE_BYTES, max_edges=MAX_EDGES):
        self.max_entries = _positive_int(max_entries, 'max_entries')
        self.max_bytes = _positive_int(max_bytes, 'max_bytes')
        self.max_table_bytes = _positive_int(max_table_bytes, 'max_table_bytes')
        self.max_edges = _positive_int(max_edges, 'max_edges')
        self._lock = threading.RLock()
        self._entries = []  # (capacity tuple, private Topology), oldest first.
        self._closed = False
        self._counts = dict(requests=0, hits=0, misses=0, builds=0, evictions=0,
                            oversized_not_cached=0, failed_requests=0, clears=0)
        self._last_call = None

    def get(self, ii, jj, kk, *, group_caps=None):
        with self._lock:
            _require(not self._closed, 'Topology cache is closed')
            begin = time.perf_counter()
            timing = dict(snapshot_seconds=0., lookup_seconds=0., build_seconds=0., copy_seconds=0.)
            self._counts['requests'] += 1
            try:
                capacities = _capacities(group_caps)
                snapshots = _snapshot(ii, jj, kk, self.max_edges)
                mark = time.perf_counter(); timing['snapshot_seconds'] = mark - begin
                selected = None
                for position, (caps, value) in enumerate(self._entries):
                    if caps == capacities and all(np.array_equal(a, b) for a, b in
                                                   zip(snapshots, (value.ii, value.jj, value.kk))):
                        selected = self._entries.pop(position)
                        self._entries.append(selected)
                        break
                end = time.perf_counter(); timing['lookup_seconds'] = end - mark
                hit = selected is not None
                if hit:
                    self._counts['hits'] += 1
                    value = selected[1]
                else:
                    self._counts['misses'] += 1
                    value = _build(snapshots, capacities, self.max_table_bytes)
                    self._counts['builds'] += 1
                    timing['build_seconds'] = time.perf_counter() - end
                    if value.nbytes > self.max_bytes:
                        self._counts['oversized_not_cached'] += 1
                    else:
                        while self._entries and (len(self._entries) >= self.max_entries or
                                sum(v.nbytes for _, v in self._entries) + value.nbytes > self.max_bytes):
                            self._entries.pop(0)
                            self._counts['evictions'] += 1
                        self._entries.append((capacities, value))
                copy_begin = time.perf_counter()
                result = value.copy()
                timing['copy_seconds'] = time.perf_counter() - copy_begin
                timing.update(total_seconds=time.perf_counter() - begin, status='completed', cache_hit=hit)
                self._last_call = timing
                return result
            except Exception as exc:
                self._counts['failed_requests'] += 1
                timing.update(total_seconds=time.perf_counter() - begin, status='failed', error=repr(exc))
                self._last_call = timing
                raise

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._counts['clears'] += 1

    def close(self):
        with self._lock:
            if not self._closed:
                self.clear()
                self._closed = True

    def metrics(self):
        with self._lock:
            return copy.deepcopy(dict(self._counts, current_entries=len(self._entries),
                current_payload_bytes=sum(value.nbytes for _, value in self._entries),
                max_entries=self.max_entries, max_bytes=self.max_bytes,
                max_table_bytes=self.max_table_bytes, max_edges=self.max_edges, closed=self._closed,
                key='Exact owned int64 ii/jj/kk values plus capacities, no hashes/pointers',
                returned_arrays_are_independent=True, last_call=self._last_call,
                payload_excludes_Python_bookkeeping_and_temporary_or_caller_owned_arrays=True))
