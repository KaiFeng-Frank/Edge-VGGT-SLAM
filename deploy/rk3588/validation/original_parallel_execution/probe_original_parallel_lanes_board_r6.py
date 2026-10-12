#!/usr/bin/env python3
"""Bounded original pair00 serial/parallel SDK diagnostic; RK3588 only.

Uses the same eight frozen real inputs, unchanged FP16 graph and owned half-I/O
route. Explicit weight-sharing initialization avoids the context-duplication API.
Physical weight sharing and NPU parallel utilization are not assumed. No async flag or system-clock writes.
"""
import argparse
import faulthandler
faulthandler.enable()
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import signal
import threading
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
HELPER_SHA = 'f3801ba5b92c1043aab2d3ebc6c0ad3258b8633c25e216b1298fbb1e0c9c38a0'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
CAPTURE_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
SDK_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
NEON_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
SHAPE = (2, 53, 1024)
BINDINGS = [dict(index=0, name='frame_output', shape=[2, 53, 1024], dtype='float32'),
            dict(index=1, name='global_output', shape=[1, 106, 1024], dtype='float32')]
ORDER = ('S', 'P', 'P', 'S')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrameAPI:
    """Per-lane frame-ID evidence without modifying the frozen I/O helper."""
    def __init__(self, api, base):
        self.api, self.base, self.evidence = api, base, {}

    def __getattr__(self, name):
        return getattr(self.api, name)

    def rknn_run(self, context, extend):
        self.evidence['sdk_run_begin'] = time.perf_counter()
        try:
            code = self.api.rknn_run(context, extend)
            self.evidence['run_frame_id'] = int(ct.cast(extend, ct.POINTER(self.base.RunExtend)).contents.frame_id)
            return code
        finally: self.evidence['sdk_run_end'] = time.perf_counter()

    def rknn_outputs_get(self, context, count, outputs, extend):
        code = self.api.rknn_outputs_get(context, count, outputs, extend)
        self.evidence['output_frame_id'] = int(ct.cast(extend, ct.POINTER(self.base.OutputExtend)).contents.frame_id)
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frequency-hz', type=int, choices=(1000000000,), required=True)
    p.add_argument('--repo', type=Path, default=BASE/'src')
    p.add_argument('--manifest', type=Path, default=BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--capture', type=Path, default=BASE/'experiments/vggt-v56-calibration-r1/capture.json')
    a = p.parse_args()
    if platform.machine() != 'aarch64' or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        p.error('Actual SDK execution is permitted only on RK3588')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started + 180
    old_affinity = os.sched_getaffinity(0)
    stop = threading.Event(); lanes = []; threads = []; pins = {}; api = None
    old_signals = {}
    r = dict(schema='v79_original_pair00_parallel_lanes_diagnostic', status='failed',
        script_sha256=sha(__file__), original_model_only=True, candidate_accepted=False,
        full_model_or_pose_or_trajectory_or_realtime_validated=False, order=list(ORDER),
        requests_per_phase=8, expected_total_requests=32, expected_outputs=64,
        requested_frequency_hz=a.frequency_hz, normal_init_flags=0, async_flag_used=False,
        duplicate_context_API=None, independent_context_initialization=False, explicit_share_weight_flag_used=True, physical_weight_sharing_verified=False,
        observed_physical_core_utilization_verified=False, system_clocks_changed=False,
        maximum_temperature_c=75, internal_deadline_seconds=180, hidden_warmup_calls=0,
        phases=[], calls=[], queries=[], contexts_created=0, contexts_destroyed=0,
        cleanup_errors=[], initialization_excluded_from_batch_wall=True,
        timing_scope='Batch-ready to all owned outputs and platform checks; per-request queue and service recorded; hashes/comparison/NPZ/JSON excluded from batch wall. SDK route includes conversion, submission, run, get, owned copy, release and finite checks.')

    def persist():
        temporary = a.output_dir/'result.json.tmp'
        temporary.write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')
        temporary.replace(a.output_dir/'result.json')

    def interrupt(signum, frame):
        stop.set()
        raise InterruptedError('Termination signal '+str(signum))

    def bounded():
        require(not stop.is_set(), 'Execution cancelled')
        require(time.monotonic() < deadline, '180 second deadline reached')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_signals[sig] = signal.signal(sig, interrupt)
        os.sched_setaffinity(0, {4, 5, 6, 7})
        r['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        print("stage: import numpy", flush=True)
        import numpy as np
        helper_path = a.repo/'deploy/rk3588/validation/original_pair_group/probe_original_pair_group.py'
        require(sha(helper_path) == HELPER_SHA, 'Frozen I/O helper differs')
        h = load_source(helper_path, '_v79_original_pair_IO')
        pin = lambda path, digest: h.pin(path, digest, pins)
        artifact = lambda root, row: h.artifact(root, row, pins)
        pin(__file__, r['script_sha256']); pin(helper_path, HELPER_SHA)
        parent_path = pin(a.manifest, PARENT_SHA); parent = json.loads(parent_path.read_text())
        capture_path = pin(a.capture, CAPTURE_SHA); capture = json.loads(capture_path.read_text())
        expected_checkpoint = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
        require(parent['checkpoint'] == capture['checkpoint'] == expected_checkpoint,
                'Only pinned original checkpoint lineage supported')
        require(parent['status'] == 'compiled_not_quality_accepted' and len(parent['blocks']) == 24,
                'Frozen original 24-pair manifest required')
        row = parent['blocks'][0]
        require(row['index'] == 0 and row['role'] == 'pair' and row['sha256'] == GRAPH_SHA
            and row['input_name'] == 'input' and row['input_shape'] == list(SHAPE)
            and row['output_bindings'] == BINDINGS
            and row['all_36_original_parameter_values_preserved_by_parent_composition'] is True,
            'Only unchanged original pair00 and its two outputs supported')
        artifact(parent_path.parent, row); model_path = artifact(parent_path.parent, row['rknn'])
        for relative, digest in parent['source_hashes'].items():
            path = (a.repo/relative).resolve(strict=True)
            require(path.is_relative_to(a.repo.resolve()), 'Original source escaped repository')
            pin(path, digest)
        require(capture['original_model_only'] and capture['full_forward_completed'] == 8
            and capture['cleanup_errors'] == [] and capture['model_state_unchanged']
            and capture['hooks_removed'] and capture['source_and_images_unchanged'],
            'Successful original eight-pair capture required')
        groups = capture['groups']['aggregator']
        group = next(g for g in groups if g['index'] == 0)
        require(group['input_shape'] == list(SHAPE) and len(group['samples']) == 8, 'Eight original frame0 inputs required')
        cases = []
        expected_pairs = [(96*i, 96*i+48) for i in range(8)]
        for i, sample in enumerate(group['samples']):
            require(sample['sample_id'] == f'calib{i:02d}' and sample['source_pair_indices'] == list(expected_pairs[i])
                and sample['origin_batch_shape'] == list(SHAPE) and sample['origin_batch_row'] is None,
                'Original calibration sample order/role differs')
            source = artifact(capture_path.parent, sample['artifact'])
            x = np.load(source, allow_pickle=False)
            require(x.shape == SHAPE and x.dtype == np.float32 and x.flags.c_contiguous
                and np.isfinite(x).all() and hashlib.sha256(x.tobytes()).hexdigest() == sample['artifact']['raw_sha256'],
                'Captured input shape/dtype/finite/raw SHA differs')
            x.flags.writeable = False
            cases.append((sample, x))
        base = load_source(pin(a.repo/'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA), '_v79_typed_ABI')
        pin(a.repo/'deploy/rk3588/reference/rknn_api_v2.3.2.h', HEADER_SHA)
        class InitExtend(ct.Structure):
            _fields_ = [('ctx', base.CONTEXT), ('real_model_offset', ct.c_int32), ('real_model_size', ct.c_uint32), ('reserved', ct.c_uint8 * 120)]
        require(ct.sizeof(InitExtend) == 136 and InitExtend.ctx.offset == 0 and InitExtend.reserved.offset == 16, 'InitExtend ABI differs')
        r['board'] = base.require_board()
        api = base.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so', SDK_SHA))))
        neon = ct.CDLL(str(pin(BASE/'experiments/vggt-v56-neon-build-r1/fp16_neon.so', NEON_SHA)))
        for name in ('f32_to_f16', 'f16_to_f32'):
            fn = getattr(neon, name); fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]; fn.restype = None
        r.update(parent_manifest_sha256=PARENT_SHA, original_graph_sha256=GRAPH_SHA,
            model_sha256=row['rknn']['sha256'], capture_sha256=CAPTURE_SHA,
            helper_sha256=HELPER_SHA, typed_ABI_sha256=ABI_SHA, header_sha256=HEADER_SHA,
            SDK_library_sha256=SDK_SHA, NEON_library_sha256=NEON_SHA,
            inputs=[s for s, x in cases])

        def checked_sample():
            bounded(); value = h.sample()
            require(value['actual_frequency_hz'] == a.frequency_hz, 'Unexpected sampled NPU frequency')
            require(value['temperatures_c'] and max(value['temperatures_c'].values()) < 75,
                    'Temperature reached75C; stop')
            return value

        def descriptor(attr, binding):
            value = base.attr_record(attr); count = math.prod(binding['shape'])
            identity = value['quantization_type'] == 2 and value['zero_point'] == 0 and value['scale'] == 1.
            require(value['index'] == binding['index'] and value['name'] == binding['name']
                and value['shape'] == binding['shape'] and value['type'] == 1 and value['format'] == 3
                and value['pass_through'] == 0 and value['n_elems'] == count and value['size'] == count*2
                and (value['quantization_type'] == 0 or identity), 'Dense original FP16 descriptor differs')
            return value

        print("stage: inputs and API bound", flush=True)
        checked_sample()
        model_blobs = []
        for ordinal, mask in enumerate((7, 1, 2, 4)):
            bounded()
            lane = dict(name='serial' if ordinal == 0 else f'lane{ordinal-1}', mask=mask,
                context=base.CONTEXT(), queue=queue.Queue(maxsize=1),
                counters=dict(sdk_run_attempts=0, sdk_runs_succeeded=0, outputs_acquired=0, outputs_released=0),
                cleanup_errors=[], alive=False, completed_requests=0)
            lanes.append(lane)
            r['stage'] = f'context_init_{ordinal}'; persist()
            print(r['stage'], flush=True)
            filename = ct.create_string_buffer(os.fsencode(model_path))
            extend = InitExtend(ctx=lanes[0]['context'].value) if ordinal else None
            flags = 0x20 if ordinal else 0
            r['pending_init_flags'] = flags; persist()
            model_bytes = model_path.stat().st_size if ordinal else 0
            argument_blob = ct.create_string_buffer(model_path.read_bytes()) if ordinal else filename
            if ordinal: model_blobs.append(argument_blob)
            r['child_model_input'] = 'owned_CPU_buffers_retained_until_all_contexts_closed'; persist()
            code = api.rknn_init(ct.byref(lane['context']), ct.cast(argument_blob, ct.c_void_p), model_bytes, flags, ct.byref(extend) if extend else None)
            operation = 'rknn_init_share_weight' if ordinal else 'rknn_init'
            r['contexts_created'] += int(bool(lane['context'].value))
            query = dict(lane=lane['name'], context_operation=operation, context_operation_code=code,
                         core_mask_requested=mask); r['queries'].append(query)
            base.check_call(code, operation); require(bool(lane['context'].value), 'Empty SDK context')
            r['stage'] = f'context_mask_{ordinal}'; persist()
            print(r['stage'], flush=True)
            code = api.rknn_set_core_mask(lane['context'], mask); query['set_core_mask_code'] = code
            base.check_call(code, 'rknn_set_core_mask')
            version = base.query(api, lane['context'], 5, base.SDKVersion())
            require(re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()), 'SDK2.3.2 required')
            counts = base.query(api, lane['context'], 0, base.InputOutputNum())
            require((counts.n_input, counts.n_output) == (1, 2), 'One input/two outputs required')
            query['input'] = descriptor(base.query(api, lane['context'], 1, base.TensorAttr(index=0)),
                                       dict(index=0, name='input', shape=list(SHAPE)))
            query['outputs'] = [descriptor(base.query(api, lane['context'], 2, base.TensorAttr(index=i)), b)
                                for i, b in enumerate(BINDINGS)]
            memory = base.query(api, lane['context'], 6, base.MemSize())
            query.update(api_version=bytes(version.api_version).decode(), driver_version=bytes(version.drv_version).decode(),
                SDK_weight_bytes=int(memory.total_weight_size), SDK_internal_bytes=int(memory.total_internal_size),
                SDK_DMA_bytes=int(memory.total_dma_allocated_size))
            lane['proxy'] = FrameAPI(api, base)
        require(len({lane['context'].value for lane in lanes}) == 4, 'Four distinct SDK contexts required')

        def execute_job(lane, job):
            start = time.perf_counter(); job['started'] = start; job['lane'] = lane['name']
            job['first_for_context'] = lane['completed_requests'] == 0
            lane['alive'] = True; lane['proxy'].evidence = {}
            try:
                job['platform_before'] = checked_sample()
                y, timing = h.execute(base, lane['proxy'], neon, np, lane['context'], cases[job['input_index']][1],
                                      BINDINGS, lane['counters'], lane['cleanup_errors'])
                job['values'], job['route'] = y, timing
                job['platform_after'] = checked_sample()
                job['status'] = 'completed'; lane['completed_requests'] += 1
            except BaseException as exc:
                job['status'], job['error'] = 'failed', f'{type(exc).__name__}: {exc}'
                stop.set()
            finally:
                job['frame_evidence'] = dict(lane['proxy'].evidence)
                job['finished'] = time.perf_counter(); job['service_seconds'] = job['finished']-start
                job['queue_seconds'] = start-job['ready']; lane['alive'] = False; job['event'].set()

        def worker(lane):
            while True:
                job = lane['queue'].get()
                try:
                    if job is None: return
                    if stop.is_set(): job['status'] = 'cancelled'; job['event'].set()
                    else: execute_job(lane, job)
                finally: lane['queue'].task_done()

        for lane in lanes[1:]:
            thread = threading.Thread(target=worker, args=(lane,), name=lane['name'], daemon=True)
            lane['thread'] = thread; threads.append(thread); thread.start()
        r['initialization_seconds'] = time.monotonic()-started
        refs = {}; parallel_phase = 0
        for phase_index, mode in enumerate(ORDER):
            bounded(); phase = dict(index=phase_index, mode=mode, status='running')
            r['phases'].append(phase); persist()
            if mode == 'S':
                code = api.rknn_set_core_mask(lanes[0]['context'], 7); phase['set_core_mask_code'] = code
                base.check_call(code, 'serial rknn_set_core_mask7')
            begin = time.perf_counter()
            jobs = [dict(phase=phase_index, mode=mode, input_index=i, sample_id=cases[i][0]['sample_id'],
                         ready=begin, status='pending', event=threading.Event()) for i in range(8)]
            if mode == 'S':
                for job in jobs:
                    bounded(); execute_job(lanes[0], job)
                    if stop.is_set(): break
            else:
                for job in jobs:
                    lane = lanes[1+(job['input_index']+parallel_phase)%3]
                    while True:
                        bounded()
                        try: lane['queue'].put(job, timeout=.1); break
                        except queue.Full: pass
                parallel_phase += 1
                for job in jobs:
                    while not job['event'].wait(.1): bounded()
            end = time.perf_counter()
            phase.update(batch_started=begin, batch_finished=end, batch_wall_seconds=end-begin,
                completed_requests=sum(j['status'] == 'completed' for j in jobs))
            for job in jobs:
                entry = {k:v for k,v in job.items() if k not in ('event','values')}
                if 'values' in job:
                    y = job['values']; key = job['sample_id']
                    if phase_index == 0: refs[key] = y
                    entry['versus_serial_FP16'] = [h.compare(np, x, ref) for x, ref in zip(y, refs[key])]
                    path = a.output_dir/f'phase{phase_index}-{key}-{job["lane"]}.npz'
                    with path.open('xb') as stream: np.savez(stream, output0=y[0], output1=y[1])
                    entry['artifact'] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))
                    entry['outputs'] = [dict(index=i, shape=list(x.shape), dtype=str(x.dtype), owned=bool(x.flags.owndata),
                        finite=bool(np.isfinite(x).all()), raw_sha256=hashlib.sha256(x.tobytes()).hexdigest()) for i,x in enumerate(y)]
                r['calls'].append(entry)
            phase['status'] = 'completed' if phase['completed_requests'] == 8 else 'failed'
            require(phase['status'] == 'completed' and not stop.is_set(), 'Incomplete phase; no performance acceptance')
            require(all(all(c['bitwise_equal'] for c in entry['versus_serial_FP16'])
                for entry in r['calls'] if entry['phase'] == phase_index),
                'Output differs from same-input original serial FP16')
            phase['requests_per_second'] = 8/phase['batch_wall_seconds']; persist()
        r['summary'] = {mode:dict(phases=2, requests=16,
            batch_wall_seconds=sum(x['batch_wall_seconds'] for x in r['phases'] if x['mode']==mode),
            requests_per_second=16/sum(x['batch_wall_seconds'] for x in r['phases'] if x['mode']==mode)) for mode in ('S','P')}
        r['summary']['parallel_to_serial_throughput_ratio'] = r['summary']['P']['requests_per_second']/r['summary']['S']['requests_per_second']
        r['status'] = 'completed_component_parallel_diagnostic_not_adopted'
    except BaseException as exc:
        stop.set(); r.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        stop.set()
        for lane in lanes[1:]:
            if 'thread' not in lane: continue
            while True:
                try:
                    pending = lane['queue'].get_nowait()
                    if pending is not None: pending['status']='cancelled'; pending['event'].set()
                    lane['queue'].task_done()
                except queue.Empty: break
            lane['queue'].put_nowait(None)
        for thread in threads: thread.join(max(0., deadline-time.monotonic()))
        alive = [lane['name'] for lane in lanes if 'thread' in lane and lane['thread'].is_alive()]
        r['threads_still_alive'] = alive; r['threads_joined'] = len(threads)-len(alive)
        for lane in reversed(lanes):
            context = lane['context']
            if not context.value or api is None: continue
            if lane['name'] in alive or (lane is lanes[0] and any(l['context'].value for l in lanes[1:])):
                r['cleanup_errors'].append('Preserved active/dependent context: '+lane['name']); continue
            try:
                code = api.rknn_destroy(context); lane['destroy_code'] = code
                base.check_call(code, 'rknn_destroy'); context.value=0; r['contexts_destroyed']+=1
            except BaseException as exc: r['cleanup_errors'].append(lane['name']+': '+str(exc))
        r['live_contexts'] = sum(bool(l['context'].value) for l in lanes)
        r['lane_resources'] = [dict(lane=l['name'], core_mask_requested=l['mask'], counters=l['counters'],
            completed_requests=l['completed_requests'], live_context=bool(l['context'].value),
            destroy_code=l.get('destroy_code'), cleanup_errors=l['cleanup_errors']) for l in lanes]
        for lane in lanes: r['cleanup_errors'].extend(lane['cleanup_errors'])
        try:
            r['source_and_inputs_unchanged'] = bool(pins) and all(sha(path)==digest for path,digest in pins.items())
        except BaseException as exc:
            r['source_and_inputs_unchanged']=False; r['cleanup_errors'].append('Identity: '+str(exc))
        try:
            os.sched_setaffinity(0, old_affinity); r['affinity_restored'] = os.sched_getaffinity(0)==old_affinity
        except BaseException as exc:
            r['affinity_restored']=False; r['cleanup_errors'].append('Affinity: '+str(exc))
        for sig, handler in old_signals.items(): signal.signal(sig, handler)
        if r['status'].startswith('completed'):
            totals = {k:sum(l['counters'][k] for l in lanes) for k in lanes[0]['counters']}
            r['resource_totals'] = totals
            if totals != dict(sdk_run_attempts=32, sdk_runs_succeeded=32, outputs_acquired=64, outputs_released=64):
                r['cleanup_errors'].append('Expected32 calls and64 released outputs')
        if r['cleanup_errors'] or r['live_contexts'] or not r.get('source_and_inputs_unchanged') or not r.get('affinity_restored'):
            r['status']='failed_cleanup_or_identity'
        r['seconds'] = time.monotonic()-started
        persist()
        print(json.dumps({k:r.get(k) for k in ('status','summary','error','contexts_created','contexts_destroyed','live_contexts','cleanup_errors')}), flush=True)
    return 0 if r['status']=='completed_component_parallel_diagnostic_not_adopted' else 1


if __name__ == '__main__': raise SystemExit(main())
