#!/usr/bin/env python3
"""Board-only original native518 ABBA: sequential versus rolling group preload.

Every call computes both original DINO24 chains, all48 aggregator blocks,
four camera refinements, both DPT frames and the original codec. References
are compared after execution, never fed into the model. Repeated real pixels
are a latency experiment, not fresh-observation throughput or SLAM acceptance.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import sys
import threading
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
SRC, E = BASE/'src', BASE/'experiments'
EXECUTOR = E/'vggt-v110-source-r1/v110_native518_executor.py'
EXECUTOR_SHA = 'e153a5f71ef0b10119207b2e43589b628a2e85ab1abeaeb72d2df2a7dd6fb638'
CONFIG = E/'vggt-v100-source-r1/full-native-sequential-config-r1.json'
CONFIG_SHA = '7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
REFERENCE = Path('/dev/shm/edge-vggt-v111-native/batch0-r1/result.json')
REFERENCE_SHA = '2eb6f63d2c832c92631a55e527f4b49e3bf0b197685f3bdea8f76008aada1081'
SOURCE_PINS = {
    'vggt_slam/model_execution.py': 'dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87',
    'third_party/vggt/vggt/utils/load_fn.py': 'b0fe762f65497dca9a22069a3e4a37fd3508398c55383a8c1cb85c403fb3e9dd',
}
OUTPUTS = ('pose_enc', 'extrinsic', 'intrinsic', 'depth', 'depth_conf')
SUCCESS = 'completed_original_full_native518_ABBA_not_realtime_or_quality_accepted'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def raw(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def memory_sample():
    row = dict(monotonic_seconds=time.monotonic())
    fields = {'MemTotal', 'MemAvailable', 'MemFree', 'Shmem', 'Cached', 'AnonPages', 'CmaTotal', 'CmaFree'}
    row['meminfo_KiB'] = {k: int(v.split()[0]) for line in Path('/proc/meminfo').read_text().splitlines()
                        for k, v in [line.split(':', 1)] if k in fields}
    zones, current = {}, None
    for line in Path('/proc/zoneinfo').read_text().splitlines():
        if line.startswith('Node '):
            current = line.strip()
        elif current and line.strip().startswith('pages free'):
            zones[current] = int(line.split()[-1]) * os.sysconf('SC_PAGE_SIZE')
    row['zone_free_bytes'] = zones
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'Actual original model execution requires RK3588/S')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); pins = {}; executor = None; sampling = None
    stop_sample = threading.Event(); memory_rows = []; sampling_errors = []
    oldenv, oldaff, oldpath = dict(os.environ), os.sched_getaffinity(0), sys.path[:]
    handlers = {}
    r = dict(schema='v123_original_native518_full_model_rolling_preload_ABBA', status='running',
             schedule='ABBA', original_model_only=True, layers_tokens_weights_and_resolution_unchanged=True,
             reference_intermediate_substituted=False, previous_tail_cache_enabled=False,
             shape=[2,3,392,518], core_mask=7, source_indices=[32,56],
             DINO_layers_per_frame=24, aggregator_blocks_per_pair=48, camera_refinements=4, DPT_frames=2,
             original_pose_codec=True, max_loops=0, SL4_backend_executed=False,
             deployment_defaults_changed=False, candidate_accepted=False,
             timestamp_paced=False, fresh_observation_rate_or_P95_measured=False,
             precision_trajectory_map_30min_or_8GB_accepted=False,
             system_clock_driver_governor_writes=False, calls=[], cleanup_errors=[],
             timing_scope='Each call includes original PNG read/preprocess and complete two-image model, checks, all group hashes/init/run/destroy and output ownership. Saving/comparison and startup are separate. Overlapped stage durations must not be summed as wall time.')

    def pin(path, digest):
        path = Path(path).resolve(strict=True)
        require(sha(path) == digest, 'Pinned source/input differs: '+str(path))
        pins[path] = digest
        return path

    def progress(stage, **fields):
        r.update(stage=stage, elapsed_seconds=time.monotonic()-start)
        atomic(out/'result.json', r)
        print(json.dumps(dict(stage=stage, **fields)), flush=True)
        require(time.monotonic()-start < 170, 'Inner bounded 170 second deadline reached')

    def sampler():
        # A process-level SDK fault cannot run finally; retain each observation.
        try:
            with (out/'memory-samples.jsonl').open('x', buffering=1) as stream:
                while not stop_sample.is_set():
                    try:
                        row=memory_sample(); memory_rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                    except BaseException as error: sampling_errors.append(repr(error))
                    stop_sample.wait(.25)
        except BaseException as error: sampling_errors.append(repr(error))

    try:
        os.sched_setaffinity(0, {4,5,6,7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        def stop(number, frame):
            raise InterruptedError('External bounded controller stopped original model experiment')
        for number in (signal.SIGINT, signal.SIGTERM): handlers[number] = signal.signal(number, stop)
        manifest = json.loads(pin(a.source_manifest, a.source_manifest_sha256).read_text())
        require(set(manifest) == {'run_rolling_full_model_board.py', 'rolling_native_aggregator.py'}, 'Exact R123 source bundle required')
        for name, meta in manifest.items():
            path = pin(a.source_manifest.parent/name, meta['sha256'])
            require(path.stat().st_size == meta['bytes'], 'Source length differs')
        require(Path(__file__).resolve() == (a.source_manifest.parent/'run_rolling_full_model_board.py').resolve(), 'Executed runner differs from manifest')
        r.update(source_bundle=manifest, source_manifest_sha256=a.source_manifest_sha256)
        pin(EXECUTOR, EXECUTOR_SHA); pin(CONFIG, CONFIG_SHA)
        for name, digest in SOURCE_PINS.items(): pin(SRC/name, digest)
        reference = json.loads(pin(REFERENCE, REFERENCE_SHA).read_text())
        require(reference['status'].startswith('completed_actual_native518_four_pair_batch') and not reference['cleanup_errors'], 'Completed original full-model reference required')
        ref = reference['pairs'][0]
        require(ref['source_indices'] == [32,56] and not ref['executor_call']['tail_cache_hit'], 'Exact same cold original pair required')
        paths = []
        for row in ref['source_images']: paths.append(str(pin(row['file'], row['sha256'])))
        r.update(reference_receipt_sha256=REFERENCE_SHA, source_images=ref['source_images'],
                 reference_output_metadata=ref['outputs'], reference_input_raw_sha256=ref['prepared_pair_raw_sha256'])
        sys.path[:0] = [str(SRC), str(SRC/'third_party/vggt')]
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from vggt_slam.model_execution import execute, prepare_rgb_images
        original = load_module(EXECUTOR, '_v123_frozen_original_executor')
        candidate = load_module(a.source_manifest.parent/'rolling_native_aggregator.py', '_v123_rolling_original_aggregator')
        class TracedRollingGroups(candidate.NativeRollingGroups):
            def _event(self, worker, phase, **extra):
                # Preserve the original event and mirror it to the controller log.
                with self._lock:
                    super()._event(worker, phase, **extra)
                    event=copy.deepcopy(self._timeline[-1])
                print(json.dumps(dict(rolling_event=event), allow_nan=False), flush=True)

        relocation=Path('/tmp/vggt-v123-export-storage/relocation.json')
        r['storage_relocation']=dict(receipt_sha256=sha(relocation), receipt=json.loads(relocation.read_text()))
        r['diagnostics']=dict(persisted_memory_file='memory-samples.jsonl',
            rolling_events_in_worker_log=True, python_faulthandler=True,
            original_rolling_runtime_unchanged=True,
            note='Diagnostic logging is included in B wall time; does not change model scheduling or computation')
        sampling = threading.Thread(target=sampler, name='v123-memory-observer', daemon=False); sampling.start()
        progress('full_original_executor_initializing'); tick=time.monotonic()
        executor = original.Native518Executor(config=CONFIG, config_sha256=CONFIG_SHA, frozen_source_dir=CONFIG.parent,
            capture_intermediates=False, source_root=SRC, expected_npu_hz=1000000000, reuse_previous_tail=False, core_mask=7)
        sequential = executor.NativeSequentialGroups
        r.update(initialization_seconds=time.monotonic()-tick, frozen_original_initialization=copy.deepcopy(executor.initialization),
                 aggregation_override='B selects rolling runtime; frozen executor initialization describes original artifacts, not B scheduling')

        def save(name, value):
            value = np.ascontiguousarray(value); path = out/(name+'.npy')
            with path.open('xb') as f: np.save(f, value, allow_pickle=False)
            return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), raw_sha256=raw(value),
                        shape=list(value.shape), dtype=str(value.dtype), finite=bool(np.isfinite(value).all()))

        for index, route in enumerate('ABBA'):
            executor.NativeSequentialGroups = sequential if route == 'A' else TracedRollingGroups
            progress('complete_original_prediction_start', index=index, route=route)
            begin = time.monotonic(); images=prepare_rgb_images(paths,518); prepared=time.monotonic()
            require(images.shape==(2,3,392,518) and raw(images)==ref['prepared_pair_raw_sha256'], 'Original native pixels differ')
            prediction=execute(executor,images,compute_similarity=False); finished=time.monotonic()
            call=copy.deepcopy(executor.last_call)
            require(call['status']=='completed' and call['complete_expected_execution'] and call['resources_balanced']
                    and call['source_and_inputs_unchanged'] and not call['tail_cache_hit']
                    and call['requested_core_mask']==7, 'Complete unchanged original model call required')
            outputs={key:save('call'+str(index)+'-'+key,prediction[key]) for key in OUTPUTS}
            equal={key:outputs[key]['raw_sha256']==ref['outputs'][key]['raw_sha256'] for key in OUTPUTS}
            row=dict(index=index,route=route,preprocessing_seconds=prepared-begin,
                     complete_prediction_seconds=finished-prepared,image_read_to_prediction_seconds=finished-begin,
                     original_input_raw_sha256=raw(images), outputs=outputs, byte_equal_frozen_full_model=equal,
                     executor_call=call, save_and_compare_seconds=time.monotonic()-finished)
            r['calls'].append(row); progress('complete_original_prediction_saved', index=index, route=route,
                                             seconds=row['complete_prediction_seconds'], all_outputs_byte_equal=all(equal.values()))
            require(all(equal.values()), 'Rolling/full baseline output changed from pinned original NPU model')
            del prediction, images
        r['all_call_means_seconds']={route:statistics.mean(c['complete_prediction_seconds'] for c in r['calls'] if c['route']==route) for route in 'AB'}
        r['status']=SUCCESS
    except BaseException as error:
        executor=getattr(error,'native_executor',executor)
        r.update(status='failed', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        if executor is not None:
            try: executor.close()
            except BaseException as error: r['cleanup_errors'].append('Executor close: '+repr(error))
            try:
                metrics=executor.metrics()
                r['frozen_executor_reported_maximum_contexts']=metrics.pop('maximum_simultaneously_live_SDK_contexts')
                metrics['maximum_contexts_field_scope']='Frozen scalar omitted; per-call aggregator timeline establishes actual overlap; DINO/heads execute before/after it'
                r['executor_resources']=metrics
                if r['status']==SUCCESS:
                    require(metrics['closed'] and metrics['resources_balanced'] and not metrics['cleanup_errors']
                            and not metrics['live_contexts'] and not metrics['live_output_buffers']
                            and metrics['SDK_totals']==dict(contexts_created=40,contexts_destroyed=40,runs_completed=72,
                                output_buffers_acquired=92,output_buffers_released=92), 'Complete ABBA resource accounting differs')
            except BaseException as error: r['cleanup_errors'].append('Resource audit: '+repr(error))
        stop_sample.set()
        if sampling is not None:
            sampling.join(timeout=3)
            if sampling.is_alive(): r['cleanup_errors'].append('Memory sampler did not join')
        r.update(memory_samples=memory_rows, memory_sampling_errors=sampling_errors,
                 memory_scope='Host RSS, SDK per-context fields and system/zone availability are separate observations, not additive unique physical bytes or 8GB acceptance')
        for number, handler in handlers.items(): signal.signal(number, handler)
        try:
            for path,digest in pins.items(): require(sha(path)==digest,'Pinned source/input changed: '+str(path))
            r['source_and_inputs_unchanged']=True
        except BaseException as error:
            r['source_and_inputs_unchanged']=False
            r['cleanup_errors'].append('Pin validation: '+repr(error))
        try:
            os.sched_setaffinity(0,oldaff); os.environ.clear(); os.environ.update(oldenv); sys.path[:]=oldpath
            r.update(affinity_restored=os.sched_getaffinity(0)==oldaff,
                     environment_restored=dict(os.environ)==oldenv,sys_path_restored=sys.path==oldpath)
        except BaseException as error: r['cleanup_errors'].append('Restore validation: '+repr(error))
        r.update(total_wall_seconds=time.monotonic()-start, peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if r['cleanup_errors']: r['status']='failed_cleanup'
        atomic(out/'result.json',r)
        print(json.dumps({k:r.get(k) for k in ('status','error','cleanup_errors','all_call_means_seconds')}),flush=True)
    return 0 if r['status']==SUCCESS else 1


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
