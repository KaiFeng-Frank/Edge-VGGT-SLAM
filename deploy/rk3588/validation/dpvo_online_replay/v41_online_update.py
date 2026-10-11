"""Diagnostic full-graph M16 SDK callback; real execution is board-owned.

DPVO, Princeton University (2022), MIT. See adjacent LICENSE. The default VGGT
path is unchanged. SDK FP32 rejection is retained: experimental recurrent use
requires an explicit diagnostic flag and never implies model adoption.
"""
import copy
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

MODEL_KEYS = frozenset(f'B{b}_D{d}_M16' for b in (1, 2) for d in range(4, 8))
INPUT_NAMES = ('net', 'inp', 'corr')


def metadata(value):
    raw = np.ascontiguousarray(value)
    return dict(shape=list(raw.shape), dtype=str(raw.dtype), bytes=raw.nbytes,
                tensor_sha256=hashlib.sha256(raw.tobytes()).hexdigest())


class OnlineUpdateError(RuntimeError):
    """No partial fullgraph is returned. Saved diagnostics do not imply commit."""
    def __init__(self, record):
        self.record = copy.deepcopy(record)
        super().__init__(record['error'])


class OnlineM16Update:
    """Serial callback, with explicit startup-probe route set by the owner.

    The owner must set route from the core's actual probe argument, restore it in
    finally, and commit returned values only synchronously. No E-size heuristic
    chooses Torch. Without identity_reader, the serial callback uses a local
    token, not fabricated graph/geometry revisions. A supplied reader returns
    the frozen adapter's SnapshotIdentity and is rechecked before every group
    and final scatter. The input/outputs remain owned after SDK release.
    """
    def __init__(self, runtime, adapter, cache, *, motion_probe,
                 diagnostic_allow_fp32_rejected, failure_dir,
                 identity_reader=None, on_record=None, frame_capacity=64,
                 timeout_ms=60000):
        if diagnostic_allow_fp32_rejected is not True:
            raise ValueError('SDK FP32 rejection requires explicit diagnostic flag')
        if not callable(motion_probe) or (identity_reader is not None and not callable(identity_reader)):
            raise ValueError('Explicit original Torch motion-probe callable required')
        if on_record is not None and not callable(on_record):
            raise ValueError('on_record must be callable or None')
        if type(frame_capacity) is not int or not 1 <= frame_capacity <= 12345:
            raise ValueError('Positive bounded frame capacity required')
        if type(timeout_ms) is not int or not 1 <= timeout_ms < 2**31:
            raise ValueError('Bounded SDK timeout required')
        self.runtime, self.adapter, self.cache = runtime, adapter, cache
        self.motion_probe, self.identity_reader, self.on_record = motion_probe, identity_reader, on_record
        self.failure_dir = Path(failure_dir).resolve(strict=True)
        if not self.failure_dir.is_dir():
            raise ValueError('Existing isolated diagnostic failure directory required')
        if set(cache.specs) != MODEL_KEYS:
            raise ValueError('Exact eight online B1/B2 D4..7 M16 models required')
        self.frame_capacity, self.timeout_ms = frame_capacity, timeout_ms
        self.route = 'full'
        self._lock, self._closed, self._failed = threading.RLock(), False, False
        self._calls = []
        self._counts = dict(requests=0, full_attempts=0, full_completed=0,
            motion_probe_attempts=0, motion_probe_completed=0, failed_calls=0)
        self._shape_histogram = {}

    def _identity(self, index):
        if self.identity_reader is None:
            token = self.adapter.SnapshotIdentity(0, 0, 0, index)
            return token, lambda: token
        token = self.identity_reader()
        if not isinstance(token, self.adapter.SnapshotIdentity):
            raise ValueError('Owner must return adapter SnapshotIdentity')
        return token, self.identity_reader

    def _snapshot(self, values):
        net, inp, corr, ii, jj, kk = values
        if any(not isinstance(v, np.ndarray) for v in values):
            raise ValueError('NumPy input arrays required')
        if any(v.dtype != np.int64 or v.ndim != 1 for v in (ii, jj, kk)):
            raise ValueError('Ordered int64 topology required')
        e = ii.size
        if not e or jj.shape != ii.shape or kk.shape != ii.shape:
            raise ValueError('Nonempty full topology required')
        for value, channels in zip((net, inp, corr), (384, 384, 882)):
            if value.dtype != np.float32 or value.shape != (1, e, channels) or not np.isfinite(value).all():
                raise ValueError('Finite full FP32 input dimensions differ')
        if sum(v.nbytes for v in values) > 256 << 20:
            raise ValueError('Full owned input snapshot exceeds 256MiB bound')
        owned = {}
        for name, value in zip(('net', 'inp', 'corr', 'ii', 'jj', 'kk'), values):
            item = np.array(value, copy=True, order='C')
            item.flags.writeable = False
            owned[name] = item
        return owned

    @staticmethod
    def _outputs(values, edges):
        if not isinstance(values, dict) or set(values) != {'net', 'delta', 'weight'}:
            raise ValueError('Exactly three fullgraph outputs required')
        for name, c in (('net', 384), ('delta', 2), ('weight', 2)):
            value = values[name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (1, edges, c) or not np.isfinite(value).all():
                raise ValueError('Malformed/nonfinite fullgraph output '+name)
        if np.any(values['weight'] < 0) or np.any(values['weight'] > 1):
            raise ValueError('Original weight [0,1] domain violated')
        return {name: np.array(value, copy=True, order='C') for name, value in values.items()}

    def _save_failure(self, record, original, error):
        """Failure I/O is outside normal inference; keep original error visible."""
        if original is not None:
            arrays = dict(original)
            for key, value in getattr(error, 'raw_outputs', {}).items():
                if isinstance(value, np.ndarray):
                    arrays['failed_group_'+key] = value
            path = self.failure_dir/f'update-{record["index"]:06d}-failed-inputs.npz'
            if path.exists():
                raise RuntimeError('Refuse to overwrite a prior diagnostic failure')
            np.savez(path, **arrays)
            record['failure_artifact'] = dict(file=path.name, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                arrays={name: metadata(v) for name, v in arrays.items()})
        path = self.failure_dir/f'update-{record["index"]:06d}-failed.json'
        if path.exists():
            raise RuntimeError('Refuse to overwrite a prior diagnostic failure')
        path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')

    def __call__(self, net, inp, corr, ii, jj, kk):
        # Route is owner-controlled from _neural(probe), never inferred from E.
        begin = time.monotonic()
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError('Online Update callback is closed or failed')
            index = self._counts['requests']; self._counts['requests'] += 1
            record = dict(index=index, route=self.route, status='starting',
                candidate_accepted=False, SDK_FP32_equivalence_validated=False,
                historical_SDK_strict_FP32_rejected=True,
                diagnostic_recurrent_use=True, default_VGGT_unchanged=True,
                owner_revisions_available=self.identity_reader is not None,
                fullgraph_returned=False, outputs_committed_by_callback=False)
            original = None
            try:
                original = self._snapshot((net, inp, corr, ii, jj, kk))
                record['edges'] = original['ii'].size
                token, reader = self._identity(index)
                record['snapshot_identity'] = dict(graph_revision=token.graph_revision,
                    geometry_revision=token.geometry_revision, feature_generation=token.feature_generation,
                    recurrent_revision=token.recurrent_revision)
                plan = self.adapter.build_degree_plan(original['ii'], original['jj'], original['kk'],
                    patches_per_frame=16, frame_capacity=self.frame_capacity,
                    max_sources_per_batch=2)
                record['degree_plan'] = plan.metadata()
                if self.route == 'motion_probe':
                    self._counts['motion_probe_attempts'] += 1
                    if (len(plan.batches) != 1 or len(plan.batches[0].source_slots) != 1
                            or plan.batches[0].degree != 1 or plan.edges != 16
                            or plan.batches[0].source_slots[0] == plan.batches[0].target_slots[0][0]):
                        raise ValueError('Explicit startup probe must be one complete D1 source to another frame')
                    values = self.motion_probe(*(np.array(original[k], copy=True) for k in
                        ('net', 'inp', 'corr', 'ii', 'jj', 'kk')))
                    if reader() != token:
                        raise RuntimeError('Stale owner snapshot after startup probe')
                    output = self._outputs(values, plan.edges)
                    record.update(execution_backend='original_Torch_startup_probe',
                        NPU_used=False, fullgraph_returned=False,
                        Torch_fallback=False, explicit_startup_route=True)
                elif self.route == 'full':
                    self._counts['full_attempts'] += 1
                    keys = [f'B{len(b.source_slots)}_D{b.degree}_M{b.patches}' for b in plan.batches]
                    record['model_keys'] = keys
                    missing = sorted(set(keys)-MODEL_KEYS)
                    if missing:
                        raise ValueError('Unknown complete online shape(s): '+', '.join(missing))
                    # A diagnostic SDK error cannot trigger original Torch.
                    output, details = self.runtime.execute_full_graph(self.adapter, plan,
                        {k: original[k] for k in INPUT_NAMES}, self.cache,
                        snapshot_identity=token, current_identity=reader,
                        reference=None, timeout_ms=self.timeout_ms)
                    output = self._outputs(output, plan.edges)
                    state = self.cache.metrics()
                    used = {v['model_key']: v['metrics'].get('npu_used') is True
                            for v in state.get('histories', [])}
                    used.update({key: v.get('npu_used') is True for key, v in state.get('sessions', {}).items()})
                    record.update(execution_backend='RKNN_complete_source_groups',
                        NPU_used=all(used.get(key, False) for key in keys),
                        Torch_fallback=False, fullgraph_returned=True, SDK_execution=details,
                        strict_FP32_gate_passed=None,
                        numerical_status='historical_FP32_rejected_diagnostic_uncompared_current_call')
                    for key in keys:
                        self._shape_histogram[key] = self._shape_histogram.get(key, 0)+1
                else:
                    raise ValueError('Owner route must explicitly be full or motion_probe')
                record.update(status='completed_diagnostic_not_adopted',
                    callback_seconds=time.monotonic()-begin,
                    timing_scope='Lock wait, owned snapshot, exact topology plan, all pack/cache/SDK/release/scatter, output validation/copy. Metadata hashes and owner callback I/O outside this clock.')
                # Normal callback retains only shape/timing/lifecycle metadata.
                # The board owner saves its predeclared numerical snapshots.
                descriptor = lambda v: dict(shape=list(v.shape), dtype=str(v.dtype), bytes=v.nbytes)
                record['inputs'] = {name: descriptor(value) for name, value in original.items()}
                record['outputs'] = {name: descriptor(value) for name, value in output.items()}
                if self.on_record is not None:
                    self.on_record(copy.deepcopy(record))
                self._calls.append(copy.deepcopy(record))
                self._counts['full_completed' if self.route == 'full' else 'motion_probe_completed'] += 1
                return output
            except BaseException as error:
                self._failed = True; self._counts['failed_calls'] += 1
                record.update(status='failed_no_partial_fullgraph', error=repr(error),
                    fullgraph_returned=False, callback_seconds=time.monotonic()-begin)
                if hasattr(error, 'record'):
                    record['SDK_failure'] = copy.deepcopy(error.record)
                try:
                    self._save_failure(record, original, error)
                except Exception as save_error:
                    record['failure_save_error'] = repr(save_error)
                self._calls.append(copy.deepcopy(record))
                raise OnlineUpdateError(record) from error

    def metrics(self):
        with self._lock:
            return copy.deepcopy(dict(self._counts, closed=self._closed, failed=self._failed,
                candidate_accepted=False, SDK_FP32_equivalence_validated=False,
                historical_SDK_strict_FP32_rejected=True, complete_Update_Torch_fallback=False,
                model_keys=sorted(MODEL_KEYS), max_sources_per_batch=2, patches_per_frame=16,
                shape_histogram=self._shape_histogram, calls=self._calls, cache=self.cache.metrics()))

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.cache.close()
            state = self.cache.metrics()
            if state.get('cleanup_errors') or state.get('resident_contexts') or state.get('current_DMA_bytes') or state.get('current_model_host_bytes'):
                self._failed = True
                raise RuntimeError('Online Update context cleanup failed')
