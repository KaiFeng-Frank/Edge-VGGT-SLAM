#!/usr/bin/env python3
"""Board-only complete original VGGT: serial mask7 versus three private threaded lanes sharing RKNN weights.

This is an offline full-inference throughput diagnostic, NOT SLAM/local poses.
Workers own independent original models, caches, contexts and immutable sessions.
Only RKNN weights are shared by the explicit SDK registry in this one process.
Torch thread settings are configured once by the parent. Initialization is serialized.
"""
import argparse
import faulthandler
import hashlib
import json
import threading
import os
from pathlib import Path
import queue
import signal
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
CHECKPOINT = dict(bytes=5026874952,sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
MANIFESTS = {
    'retained': ('vggt-v67-retained-groups-r1/compiled.json','771a9b82007d2feefd65ebbf3f4b0fb27070b79ff7f6d0faab1a8c866da90e43'),
    'dino': ('vggt-v58-dino-groups-r1/compiled.json','a602bcdaab3f2e1ab51d49feb300ff3af76c07fe334489c983d647859f3f9493'),
    'depth': ('vggt-v54-depth-graphs-r1/compiled.json','9a6e5435df543b8484a4fc50da4b065995f32153233548829234dd149fbc7135')}
NEON = BASE/'experiments/vggt-v56-neon-build-r1/fp16_neon.so'
NEON_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
KEYS = ('pose_enc','extrinsic','intrinsic','depth','depth_conf')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4 << 20),b''): h.update(part)
    return h.hexdigest()


def write(path,value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def log(worker,stage):
    print(f'[{time.monotonic():.6f}] worker={worker} {stage}',flush=True)


def terminate(signum,frame):
    raise InterruptedError(f'Received signal {signum}')


def memory():
    rows = dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return dict(scope='process_wide_all_worker_threads',**{name+'_MiB':int(rows[name].split()[0])/1024 for name in ('VmRSS','VmHWM')})


def available_bytes():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))


def worker(index,mask,commands,events,output,overlay):
    """This thread owns its separate model and all its backend API calls."""
    source = BASE/'src'
    os.sched_setaffinity(0,{4,5,6,7})
    folder = Path(output)/f'worker-{index}'; folder.mkdir()
    handles,stored,session,cache = {},{},None,None
    receipt = dict(worker=index,status='initializing',initial_core_mask=mask,actual_core_mask=mask,
        jobs=[],pageout_stages=[],mask_switches=[],cleanup_errors=[],cpu_affinity=sorted(os.sched_getaffinity(0)),
        ownership='independent_model_cache_session_owned_by_this_thread',
        memory_scope='process_wide_all_worker_threads')
    begin = time.monotonic()
    try:
        log(index,'imports')
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        from vggt_slam.model_execution import prepare_rgb_images,as_executor,execute
        from vggt_slam.patch_cache import install_patch_cache
        from vggt_slam.immutable_execution_session import install_immutable_session
        from vggt_slam.checkpoint_residency import checkpoint_stat,pageout_checkpoint
        from vggt_slam.rknn_retained_aggregator import install_retained_aggregator
        from vggt_slam.rknn_camera_head import install_rknn_camera_head
        from vggt_slam.rknn_grouped_dino import install_grouped_dino
        from vggt_slam.rknn_depth_head import install_rknn_depth_head
        receipt['overlay_sources'] = {name:sha(Path(overlay)/'vggt_slam'/name) for name in
            ('rknn_retained_aggregator.py','rknn_camera_head.py','rknn_grouped_dino.py','rknn_depth_head.py')}
        loaded = checkpoint_stat(BASE/'weights/model.pt')
        log(index,'load_model original checkpoint')
        model = load_model(BASE/'weights/model.pt',device='cpu',image_size=112,prune_unused_heads=True,offline=True)
        if checkpoint_stat(BASE/'weights/model.pt') != loaded: raise RuntimeError('Checkpoint identity changed across load')
        def install(name,function,*args,**kwargs):
            log(index,'install '+name)
            handles[name] = function(model,*args,core_mask=mask,**kwargs)
            stage = name+'_installed'; path = folder/('checkpoint-pageout-'+stage+'.json')
            item = dict(stage=stage,file=path.relative_to(output).as_posix(),status='calling')
            receipt['pageout_stages'].append(item); write(folder/'worker.json',receipt)
            page = pageout_checkpoint(model,BASE/'weights/model.pt',loaded,path,source_root=source,stage=stage)
            item.update(status=page['status'],sha256=sha(path)); write(folder/'worker.json',receipt)
        path,digest = MANIFESTS['retained']
        install('aggregator',install_retained_aggregator,BASE/'experiments'/path,digest,
                half_io_library=NEON,half_io_library_sha256=NEON_SHA)
        install('camera',install_rknn_camera_head,split_fc1=True)
        path,digest = MANIFESTS['dino']
        install('DINO',install_grouped_dino,BASE/'experiments'/path,digest,half_io_library=NEON,half_io_library_sha256=NEON_SHA)
        path,digest = MANIFESTS['depth']
        install('depth',install_rknn_depth_head,BASE/'experiments'/path,digest)
        cache = install_patch_cache(model)
        session = install_immutable_session(model,cache,handles['aggregator'],handles['DINO'],contract='exclusive_immutable_model')
        executor = as_executor(model)
        receipt.update(status='ready',initialization_seconds=time.monotonic()-begin,memory_after_init=memory())
        write(folder/'worker.json',receipt); events.put(dict(kind='ready',worker=index,receipt=receipt))
        log(index,'ready')
        while True:
            command = commands.get(); action = command['action']
            if action == 'stop': break
            if action == 'mask':
                if cache._context is not None: raise RuntimeError('Mask switch requires idle owner')
                session.close(); receipt['baseline_immutable'] = session.metrics()
                cache.close(); receipt['baseline_cache'] = cache.metrics()
                newmask = command['mask']; changed = []
                for name,handle in handles.items():
                    contexts = handle.contexts if hasattr(handle,'contexts') else [handle.context]
                    for ordinal,ctx in enumerate(contexts):
                        code = handle.api.rknn_set_core_mask(ctx,newmask)
                        changed.append(dict(backend=name,context_index=ordinal,code=code))
                        handle.base.check_call(code,'rknn_set_core_mask idle switch')
                    handle.stats['core_mask'] = newmask
                retained = handles['aggregator']; retained.core_mask = newmask
                retained.backend_state = retained._backend_state()
                cache = install_patch_cache(model)
                session = install_immutable_session(model,cache,retained,handles['DINO'],contract='exclusive_immutable_model')
                receipt['mask_switches'].append(dict(old_mask=mask,new_mask=newmask,calls=changed,cache_reset=True,session_closed_and_resealed=True))
                mask = newmask; receipt['actual_core_mask'] = mask
                write(folder/'worker.json',receipt); events.put(dict(kind='mask',worker=index,mask=mask))
                continue
            if action == 'save':
                saved = []
                for (route,case_index),(case,images,pred,job) in stored.items():
                    arrays = {key:np.array(pred[key],copy=True) for key in KEYS}; arrays['images'] = np.array(images,copy=True)
                    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays.values()): raise ValueError('Saved original output is not finite FP32')
                    target = folder/route; target.mkdir(exist_ok=True); path = target/f'pair-{case["pair"]:02d}.npz'
                    np.savez(path,**arrays)
                    metadata = {key:dict(shape=list(value.shape),dtype='float32',raw_sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key,value in arrays.items()}
                    job.update(preprocessed_images_raw_sha256=metadata['images']['raw_sha256'],image_shape=list(images.shape),
                        source_images=[dict(path=path,sha256=sha(path)) for path in case['image_paths']],
                        artifact=dict(file=path.relative_to(output).as_posix(),bytes=path.stat().st_size,sha256=sha(path),arrays=metadata))
                    saved.append(job)
                receipt['jobs'] = saved; write(folder/'worker.json',receipt)
                events.put(dict(kind='saved',worker=index,jobs=saved)); continue
            if action != 'run': raise ValueError('Unknown worker command')
            case,route = command['case'],command['route']; log(index,f'{route} pair {case["pair"]} start mask{mask}')
            start = time.monotonic()
            with torch.inference_mode():
                images = prepare_rgb_images(case['image_paths'],112)
                prepared = time.monotonic(); pred = execute(executor,images)
            end = time.monotonic()
            if list(images.shape) != [2,3,84,112]: raise ValueError('Fixed original image shape differs')
            job = dict(route=route,worker=index,core_mask=mask,case_index=case['case_index'],pair=case['pair'],source_indices=case['source_indices'],
                image_paths=case['image_paths'],timestamps=case['timestamps'],start_monotonic=start,end_monotonic=end,
                preprocessing_seconds=prepared-start,model_and_codec_seconds=end-prepared,complete_inference_seconds=end-start)
            # Keep arrays alive without IPC, hashing or disk work until both timed
            # batches end. Every next execute allocates its own original outputs.
            stored[(route,case['case_index'])] = (case,images,pred,job)
            events.put(dict(kind='done',worker=index,job=job)); log(index,f'{route} pair {case["pair"]} done')
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        traceback.print_exc(); log(index,'failed '+receipt['error'])
    finally:
        for name,handle in [('immutable',session),('cache',cache)]+[(name,handles.get(name)) for name in ('depth','DINO','camera','aggregator')]:
            if handle is not None:
                try: handle.close()
                except BaseException as exc: receipt['cleanup_errors'].append(f'{name}: {type(exc).__name__}: {exc}')
                try: receipt.setdefault('closed_metrics',{})[name] = handle.metrics()
                except BaseException as exc: receipt['cleanup_errors'].append(f'{name} metrics: {exc}')
        if receipt['cleanup_errors']: receipt['status'] = 'failed_cleanup'
        elif receipt['status'] != 'failed': receipt['status'] = 'closed'
        receipt.update(total_seconds=time.monotonic()-begin,memory_after_cleanup=memory())
        write(folder/'worker.json',receipt); log(index,'shutdown '+receipt['status'])
        events.put(dict(kind='closed',worker=index,receipt=receipt))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mask-overlay',type=Path,default=Path(__file__).with_name('mask-shared-overlay'))
    p.add_argument('--timeout',type=float,default=165); a = p.parse_args()
    if not 0 < a.timeout <= 180: p.error('timeout must be within the bounded 180-second experiment')
    if b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes(): p.error('RK3588 board required')
    faulthandler.enable(all_threads=True); signal.signal(signal.SIGTERM,terminate)
    a.output_dir = a.output_dir.resolve(); a.output_dir.mkdir(parents=True,exist_ok=False)
    a.mask_overlay = a.mask_overlay.resolve(strict=True); sys.path.insert(0,str(BASE/'src'))
    sys.path.insert(0,str(BASE/'src/third_party/vggt'))
    import vggt_slam
    vggt_slam.__path__.insert(0,str(a.mask_overlay/'vggt_slam'))
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from vggt_slam import shared_context_registry as registry_module
    registry_snapshot = registry_module.snapshot
    from vggt_slam.timestamp_replay import read_rgb_index
    rows = read_rgb_index(BASE/'datasets/rgbd_dataset_freiburg1_xyz/rgb.txt')
    indices = list(range(32,417,48))
    cases = [dict(case_index=i-1,pair=i,source_indices=indices[i-1:i+1],image_paths=[rows[j]['path'] for j in indices[i-1:i+1]],
        timestamps=[rows[j]['timestamp'] for j in indices[i-1:i+1]]) for i in range(1,len(indices))]
    assert len(cases) == 8 and len({tuple(case['source_indices']) for case in cases}) == 8
    report = dict(schema='v79_original_VGGT_full_parallel_diagnostic',status='running',script_sha256=sha(__file__),original_model_only=True,
        checkpoint=CHECKPOINT,manifests={key:dict(file=path,sha256=digest) for key,(path,digest) in MANIFESTS.items()},
        cases=cases,image_shape=[2,3,84,112],routes={},workers={},cleanup_errors=[],queue_capacity=1,
        separate_original_models=3,process_count=1,worker_thread_count=3,
        runner_execution_mode='same_process_threads_shared_RKNN_weights',
        SDK_weight_sharing_API='rknn_init flags0x20 with master InitExtend.ctx',
        shared_registry_source_sha256=sha(Path(registry_module.__file__)),
        memory_scope='process_wide_all_worker_threads_not_per_model',
        all_original_layers_tokens_heads_preserved=True,immutable_contract='exclusive_immutable_model',
        SL4_executed=False,continuous_local_pose_rate_validated=False,quality_accepted=False,realtime_validated=False,
        throughput_scope='Offline complete original model plus canonical pose codec. Input decode/preprocess included; init/save/hash/array compare excluded. Not timestamp or SLAM latency.',
        cache_scope='Each worker thread has its own original model/cache/session; scheduling can reduce cache hits and is reported, not prewarmed.',
        physical_8GB_acceptance=False)
    events = queue.Queue(maxsize=1); processes,commands,closed = {},{},{}
    deadline = time.monotonic()+a.timeout
    def receive():
        while True:
            if time.monotonic() >= deadline: raise TimeoutError('Parent diagnostic deadline exceeded')
            try: event = events.get(timeout=min(1,max(.01,deadline-time.monotonic())))
            except queue.Empty:
                dead = [i for i,proc in processes.items() if not proc.is_alive() and i not in closed]
                if dead: raise RuntimeError('Worker exited without final receipt: '+str(dead))
                continue
            if event['kind'] == 'closed':
                closed[event['worker']] = event['receipt']
                if event['receipt']['status'] != 'closed': raise RuntimeError('Worker failed: '+str(event['receipt'].get('error',event['receipt']['cleanup_errors'])))
            return event
    def start(index,mask):
        available = available_bytes(); report.setdefault('MemAvailable_before_worker_init_bytes',[]).append(available)
        if available < 4*(1 << 30): raise MemoryError('MemAvailable below 4 GiB before worker initialization')
        commands[index] = queue.Queue(maxsize=1)
        proc = threading.Thread(target=worker,name=f'original-VGGT-worker-{index}',daemon=False,
            args=(index,mask,commands[index],events,str(a.output_dir),str(a.mask_overlay)))
        processes[index] = proc; proc.start(); event = receive()
        if event['kind'] != 'ready' or event['worker'] != index: raise RuntimeError('Sequential initialization event differs')
        report['workers'][str(index)] = event['receipt']
        report['shared_registry_after_worker_init'] = registry_snapshot()
        write(a.output_dir/'result.json',report)
    def batch(route,lanes):
        jobs,pending,nextcase = [],{},0; begin = time.monotonic()
        for lane in lanes:
            commands[lane].put(dict(action='run',route=route,case=cases[nextcase])); pending[lane] = nextcase; nextcase += 1
        while pending:
            event = receive()
            if event['kind'] != 'done' or event['worker'] not in pending or event['job']['case_index'] != pending[event['worker']]: raise RuntimeError('Job identity/event differs')
            lane = event['worker']; jobs.append(event['job']); del pending[lane]
            if nextcase < len(cases):
                commands[lane].put(dict(action='run',route=route,case=cases[nextcase])); pending[lane] = nextcase; nextcase += 1
        end = time.monotonic(); jobs.sort(key=lambda job:job['case_index'])
        assert [job['case_index'] for job in jobs] == list(range(8))
        report['routes'][route] = dict(jobs=jobs,batch_start_monotonic=begin,batch_end_monotonic=end,batch_wall_seconds=end-begin,
            unique_pair_count=8,complete_pairs_per_second=8/(end-begin),includes_preprocessing=True,SL4_executed=False)
        write(a.output_dir/'result.json',report)
    try:
        start(0,7); batch('serial',[0])
        # Preserve baseline evidence even if a later shared initialization fails.
        # Saving and hashes are outside the completed serial batch timer.
        commands[0].put(dict(action='save')); event = receive()
        if event['kind'] != 'saved' or event['worker'] != 0: raise RuntimeError('Baseline save acknowledgement differs')
        for job in event['jobs']:
            report['routes'][job['route']]['jobs'][job['case_index']] = job
        write(a.output_dir/'result.json',report)
        commands[0].put(dict(action='mask',mask=1)); event = receive()
        if event != dict(kind='mask',worker=0,mask=1): raise RuntimeError('Idle mask switch acknowledgement differs')
        start(1,2); start(2,4); batch('parallel',[0,1,2])
        for index in processes:
            commands[index].put(dict(action='save')); event = receive()
            if event['kind'] != 'saved' or event['worker'] != index: raise RuntimeError('Save acknowledgement differs')
            for job in event['jobs']:
                report['routes'][job['route']]['jobs'][job['case_index']] = job
        import numpy as np
        checks = []
        for case in cases:
            left,right = [report['routes'][route]['jobs'][case['case_index']] for route in ('serial','parallel')]
            assert left['preprocessed_images_raw_sha256'] == right['preprocessed_images_raw_sha256'] and left['source_images'] == right['source_images']
            with np.load(a.output_dir/left['artifact']['file'],allow_pickle=False) as baseline, np.load(a.output_dir/right['artifact']['file'],allow_pickle=False) as candidate:
                errors = {}
                for key in KEYS:
                    u,v = baseline[key],candidate[key]; delta = v.astype(np.float64)-u.astype(np.float64)
                    outside = int(np.count_nonzero(np.abs(delta) > 1e-5+1e-4*np.abs(u.astype(np.float64))))
                    errors[key] = dict(byte_equal=u.tobytes() == v.tobytes(),strict_pass=outside == 0,outside_tolerance=outside,max_abs=float(np.max(np.abs(delta))))
                checks.append(dict(case_index=case['case_index'],pair=case['pair'],outputs=errors))
        report.update(comparison=checks,status='completed_not_quality_acceptance',
            all_same_input_outputs_byte_equal=all(x['byte_equal'] for row in checks for x in row['outputs'].values()),
            all_same_input_outputs_strict_passed=all(x['strict_pass'] for row in checks for x in row['outputs'].values()))
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        # Threads cannot be killed safely. Stop children before the master;
        # the external bounded controller owns process termination if a thread
        # cannot finish SDK cleanup. Do not claim release on a missing receipt.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        stop_deadline = time.monotonic()+25
        stop_order = list(reversed(processes)); report['worker_shutdown_order'] = stop_order
        unsafe_child = False
        for index in stop_order:
            proc = processes[index]
            if unsafe_child:
                report['cleanup_errors'].append(f'Worker {index} left alive after dependent thread timeout; SDK release unknown')
            elif proc.is_alive():
                try: commands[index].put(dict(action='stop'),timeout=min(1,max(.01,stop_deadline-time.monotonic())))
                except BaseException as exc: report['cleanup_errors'].append(f'Worker {index} stop: {exc}')
                while proc.is_alive() and time.monotonic() < stop_deadline:
                    try:
                        event = events.get(timeout=min(.5,max(.01,stop_deadline-time.monotonic())))
                        if event['kind'] == 'closed': closed[event['worker']] = event['receipt']
                    except queue.Empty: pass
                    proc.join(timeout=0)
            # A closed event may have been delivered immediately before exit.
            proc.join(timeout=max(0,min(.1,stop_deadline-time.monotonic())))
            while not events.empty():
                try:
                    event = events.get_nowait()
                    if event['kind'] == 'closed': closed[event['worker']] = event['receipt']
                except queue.Empty: break
            if proc.is_alive():
                unsafe_child = True
                report['cleanup_errors'].append(f'Worker {index} still alive; external controller required, SDK release unknown')
            if index not in closed or closed[index]['status'] != 'closed':
                report['cleanup_errors'].append(f'Worker {index} failed or missing final cleanup receipt')
            report['workers'][str(index)] = closed.get(index,dict(status='missing_final_receipt',thread_alive=proc.is_alive()))
        final_registry = registry_snapshot()
        report['shared_context_registry_final'] = final_registry
        if (final_registry['live_contexts'] != 0 or final_registry['cleanup_errors']
                or final_registry['actual_contexts_created'] != final_registry['actual_contexts_destroyed']
                or final_registry['all_registered_contexts_final_zero'] is not True):
            report['cleanup_errors'].append('Shared RKNN registry actual context cleanup incomplete')
        report['all_worker_threads_joined'] = all(not proc.is_alive() for proc in processes.values())
        if report['cleanup_errors']: report['status'] = 'failed_cleanup'
        write(a.output_dir/'result.json',report)
        log('parent','terminal '+report['status'])
    return 0 if report['status'] == 'completed_not_quality_acceptance' else 1


if __name__ == '__main__': raise SystemExit(main())
