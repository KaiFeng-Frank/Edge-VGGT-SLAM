#!/usr/bin/env python3
"""Real original-timestamp replay: three full original VGGT lanes, one SL4.

Each lane owns its model/cache/session and executes all original computation.
Only SDK weights share the validated explicit registry. One latest-frame slot,
one outstanding request per lane, ordered SL4/map commits and EOF drain.
No interpolation, pose reuse or independent pair coordinates are published.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import queue
import resource
import signal
import sys
import threading
import time
import traceback
import full_parallel_shared_board as inherited
from full_parallel_shared_board import BASE,CHECKPOINT,MANIFESTS,NEON,NEON_SHA,KEYS,sha,write,log,memory,available_bytes,terminate

INHERITED_SHA='7081dfe13f0ef83ff23b41d12a891ff838eaaf545da8598777f02578cb3597f4'


def worker(index,mask,commands,events,output,overlay):
    """This thread owns its separate model and all its backend API calls."""
    source = BASE/'src'
    os.sched_setaffinity(0,{4,5,6,7})
    folder = Path(output)/f'worker-{index}'; folder.mkdir()
    handles,session,cache = {},None,None
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
            if action != 'run': raise ValueError('Unknown worker command')
            case,route = command['case'],command['route']; log(index,f'{route} pair {case["pair"]} start mask{mask}')
            start = time.monotonic()
            with torch.inference_mode():
                images = prepare_rgb_images(case['image_paths'],112)
                prepared = time.monotonic(); pred = execute(executor,images)
            predicted = time.monotonic()
            outputs = {key:np.array(pred[key],copy=True) for key in KEYS}
            images = np.array(images,copy=True)
            if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in [images,*outputs.values()]):
                raise ValueError('Original model/transport arrays must be finite FP32')
            end = time.monotonic()
            if list(images.shape) != [2,3,84,112]: raise ValueError('Fixed original image shape differs')
            job = dict(route=route,worker=index,core_mask=mask,case_index=case['case_index'],pair=case['pair'],source_indices=case['source_indices'],
                image_paths=case['image_paths'],timestamps=case['timestamps'],start_monotonic=start,end_monotonic=end,
                preprocessing_seconds=prepared-start,model_and_codec_seconds=predicted-prepared,
                owned_copy_and_validation_seconds=end-predicted,complete_inference_seconds=end-start)
            # In-process owned arrays; bounded queue transport/ordering remain
            # within the source-arrival-to-published latency in the parent.
            events.put(dict(kind='done',worker=index,job=job,images=images,outputs=outputs))
            log(index,f'{route} pair {case["pair"]} done')
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

def publish(event, *, solver, execution, np, output, manifest, executor_type):
    """One original SL4 commit and atomic dense-map/trajectory publication."""
    images,outputs,job = event['images'],event['outputs'],event['job']
    prepared_sha = hashlib.sha256(images.tobytes()).hexdigest()
    executor = executor_type(execution,np,images,outputs,prepared_sha)
    case = job['case']
    prediction = solver.run_predictions(case['image_paths'],executor,max_loops=0,prepared_images=images)
    current = solver.current_working_submap
    if current.frame_ids != case['timestamps'] or executor.calls != 1:
        raise RuntimeError('Original SL4 source identities/executor count differ')
    solver.add_points(prediction); solver.graph.optimize()
    graph_error = float(solver.graph.graph.error(solver.graph.values))
    if not np.isfinite(graph_error): raise RuntimeError('Nonfinite original SL4 graph error')
    pose = np.asarray(current.get_all_poses_world(solver.graph)[-1])
    if pose.shape != (4,4) or not np.isfinite(pose).all(): raise RuntimeError('Nonfinite original world pose')
    optimized = time.monotonic()
    arrays,counts = {},[]
    for i in range(len(current.pointclouds)):
        mask = current.conf_masks[i] > current.conf_threshold
        points = np.asarray(current.pointclouds[i][mask],np.float32)
        colors = np.asarray(current.colors[i][mask],np.uint8)
        if points.shape != (len(points),3) or not np.isfinite(points).all(): raise RuntimeError('Invalid original dense chunk')
        arrays[f'points_{i}'],arrays[f'colors_{i}'] = points,colors
        counts.append(len(points))
    node = current.get_id(); path=output/'map_chunks'/f'submap_{node:06d}.npz'
    np.savez(path,**arrays)
    manifest['chunks'].append(dict(path='map_chunks/'+path.name,submap_id=node,
        nodes=[node+i for i in range(len(counts))],point_counts=counts,bytes=path.stat().st_size,sha256=sha(path)))
    for first,submap in sorted(solver.map.submaps.items()):
        for i in range(len(submap.pointclouds)):
            H=np.asarray(solver.graph.get_homography(first+i),np.float64)
            if H.shape!=(4,4) or not np.isfinite(H).all(): raise RuntimeError('Invalid original SL4 homography')
            manifest['nodes'][str(first+i)] = H.tolist()
    manifest['revision'] += 1
    temporary=output/'poses.tum.tmp'
    solver.map.write_poses_to_file(str(temporary),solver.graph,kitti_format=False)
    temporary.replace(output/'poses.tum'); write(output/'map.json',manifest)
    return dict(optimized_monotonic=optimized,published_monotonic=time.monotonic(),
        graph_error=graph_error,graph_nodes=int(solver.graph.values.size()),pose=pose.tolist(),
        prepared_images_raw_sha256=prepared_sha)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mask-overlay',type=Path,default=Path(__file__).with_name('mask-shared-buffer-overlay'))
    p.add_argument('--timeout',type=float,default=170)
    a=p.parse_args()
    if not 0<a.timeout<=170: p.error('timeout must be within170 seconds; external controller bounds180')
    if b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes(): p.error('RK3588 board required')
    if sha(inherited.__file__)!=INHERITED_SHA: raise RuntimeError('Successful full runner source changed')
    faulthandler.enable(all_threads=True); signal.signal(signal.SIGTERM,terminate)
    setup=time.monotonic(); deadline=setup+a.timeout
    affinity=os.sched_getaffinity(0)
    if not {0,4,5,6,7}<=affinity: raise RuntimeError('Decoder CPU0 and four model CPUs required')
    a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=False)
    a.mask_overlay=a.mask_overlay.resolve(strict=True)
    sys.path[:0]=[str(BASE/'src'),str(BASE/'src/third_party/vggt')]
    import vggt_slam
    vggt_slam.__path__.insert(0,str(a.mask_overlay/'vggt_slam'))
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    from vggt_slam import shared_context_registry as registry_module,model_execution as execution
    from vggt_slam.timestamp_replay import TimestampReplay,read_rgb_index
    from vggt_slam.solver import Solver
    from vggt_slam.edge_mapping import platform_sample
    from commit_parallel_predictions_board import FrozenPredictionExecutor,SOURCE_PINS
    for relative,digest in SOURCE_PINS.items():
        if sha(BASE/'src'/relative)!=digest: raise RuntimeError('Original SL4 source changed: '+relative)
    rows=read_rgb_index(BASE/'datasets/rgbd_dataset_freiburg1_xyz/rgb.txt')
    if len(rows)!=798: raise RuntimeError('Expected all798 original TUM RGB frames')
    report=dict(schema='v79_original_VGGT_parallel_shared_timestamp_replay',status='running',script_sha256=sha(__file__),
        inherited_full_runner_sha256=INHERITED_SHA,original_model_only=True,checkpoint=CHECKPOINT,
        shared_registry_source_sha256=sha(registry_module.__file__),
        saved_executor_source_sha256=sha(Path(__file__).with_name('commit_parallel_predictions_board.py')),
        runtime_source_sha256=SOURCE_PINS,original_frames=len(rows),worker_threads3=True,process_count=1,
        independent_original_model_cache_sessions=3,worker_core_masks=[1,2,4],
        shared_weights_API='rknn_init flags0x20 InitExtend.ctx with owned CPU child model buffer',
        memory_scope='process_wide_all_worker_threads',queue_capacity=1,per_worker_outstanding_capacity=1,
        total_uncommitted_capacity=3,unique_parent_SL4_solver=True,SL4_executed=True,
        fresh_pose_count=0,dispatched_pairs=0,updates=[],workers={},cleanup_errors=[],
        dispatches=[],source_order='anchor is preceding dispatched target; commit in dispatch order',
        all_original_layers_tokens_heads_preserved=True,immutable_contract='exclusive_immutable_model',
        short_replay_not_30min=True,real_time_validated=False,quality_accepted=False,
        physical_8GB_acceptance=False,physical_sensor_latency_validated=False,
        latency_scope='original software arrival through decoder, latest slot, dispatch, complete model, owned-copy/thread queue, ordered SL4, chunk/poses/map atomic publication',
        EOF_tail_drained=False)
    events=queue.Queue(maxsize=1);threads,commands,closed={},{},{}
    replay=None
    def receive(timeout=1):
        if time.monotonic()>=deadline: raise TimeoutError('Parallel replay diagnostic deadline exceeded')
        try: event=events.get(timeout=min(timeout,max(.001,deadline-time.monotonic())))
        except queue.Empty:
            dead=[i for i,t in threads.items() if not t.is_alive() and i not in closed]
            if dead: raise RuntimeError('Worker exited without final receipt: '+str(dead))
            return None
        if event['kind']=='closed':
            closed[event['worker']]=event['receipt']
            raise RuntimeError('Worker closed before replay finished: '+str(event['receipt'].get('error',event['receipt']['status'])))
        return event
    try:
        for index,mask in enumerate((1,2,4)):
            available=available_bytes();report.setdefault('MemAvailable_before_worker_init_bytes',[]).append(available)
            if available<4*(1<<30): raise MemoryError('MemAvailable below4GiB before worker initialization')
            commands[index]=queue.Queue(maxsize=1)
            t=threading.Thread(target=worker,name=f'original-VGGT-worker-{index}',daemon=False,
                args=(index,mask,commands[index],events,str(a.output_dir),str(a.mask_overlay)))
            threads[index]=t;t.start()
            while True:
                event=receive()
                if event is not None:break
            if event['kind']!='ready' or event['worker']!=index:raise RuntimeError('Serialized init event differs')
            report['workers'][str(index)]=event['receipt'];write(a.output_dir/'results.json',report)
        os.sched_setaffinity(0,{4,5,6,7})
        solver=Solver(25.,device='cpu',image_size=112,max_loops=0,offline=True,enable_viewer=False)
        (a.output_dir/'map_chunks').mkdir()
        manifest=dict(format='vggt_camera_local_dense_chunks_with_SL4_graph',units='model_units',
            chunks=[],nodes={},revision=0,metric_scale_validated=False,rectification=None)
        report.update(startup_seconds=time.monotonic()-setup,torch_version=torch.__version__,
            cpu_affinity=sorted(os.sched_getaffinity(0)),decoder_cpu_affinity=[0],platform_start=platform_sample(),
            shared_registry_after_init=registry_module.snapshot(),startup_memory=memory())
        write(a.output_dir/'results.json',report)
        replay=TimestampReplay(rows,cpus={0});replay.start()
        free=[0,1,2];pending={};finished={};anchor=None;next_commit=0
        while True:
            if time.monotonic()>=deadline:raise TimeoutError('Parallel replay deadline exceeded')
            # Each lane stays occupied through its ordered commit. Thus even
            # completion out of order cannot grow an unbounded result buffer.
            if free and not replay.exhausted:
                item=replay.read(timeout=.001);item.pop('frame',None) if item is not None else None
                if item is not None:
                    if anchor is None:
                        anchor=item;report.update(seed_timestamp=item['timestamp'],seed_sequence=item['sequence'])
                    else:
                        if item['timestamp']<=anchor['timestamp']:raise RuntimeError('Target timestamp not newer')
                        ordinal=report['dispatched_pairs'];lane=free.pop(0);dispatch=time.monotonic()
                        case=dict(case_index=ordinal,pair=ordinal+1,source_indices=[anchor['sequence'],item['sequence']],
                            image_paths=[anchor['path'],item['path']],timestamps=[anchor['timestamp'],item['timestamp']])
                        item=dict(item);item['anchor_timestamp']=anchor['timestamp']
                        pending[ordinal]=dict(worker=lane,case=case,item=item,dispatch_monotonic=dispatch)
                        commands[lane].put(dict(action='run',route='timestamp_parallel',case=case))
                        report['dispatches'].append(dict(case=case,worker=lane,dispatch_monotonic=dispatch))
                        report['dispatched_pairs']+=1;anchor=item
            event=receive(timeout=.001) if pending else None
            if event is not None:
                job=event['job'];ordinal=job['case_index'];expected=pending.get(ordinal)
                if event['kind']!='done' or expected is None or event['worker']!=expected['worker']:
                    raise RuntimeError('Completed job identity differs')
                if job['source_indices']!=expected['case']['source_indices'] or job['timestamps']!=expected['case']['timestamps']:
                    raise RuntimeError('Completed source timestamps differ')
                job['case']=expected['case'];event['received_monotonic']=time.monotonic();finished[ordinal]=event
            while next_commit in finished:
                event=finished.pop(next_commit);record=pending[next_commit];job=event['job'];item=record['item']
                begin=time.monotonic()
                result=publish(event,solver=solver,execution=execution,np=np,output=a.output_dir,
                    manifest=manifest,executor_type=FrozenPredictionExecutor)
                published=result['published_monotonic'];optimized=result['optimized_monotonic']
                update=dict(revision=manifest['revision'],pair=next_commit+1,worker=record['worker'],
                    core_mask=job['core_mask'],timestamp=item['timestamp'],sequence=item['sequence'],
                    source_indices=job['source_indices'],image_paths=job['image_paths'],timestamps=job['timestamps'],
                    anchor_timestamp=item['anchor_timestamp'],captured_monotonic=item['captured_monotonic'],
                    dispatch_monotonic=record['dispatch_monotonic'],worker_start_monotonic=job['start_monotonic'],
                    worker_end_monotonic=job['end_monotonic'],result_received_monotonic=event['received_monotonic'],
                    commit_start_monotonic=begin,published_monotonic=published,
                    arrival_to_decode_seconds=item['decode_started_monotonic']-item['captured_monotonic'],
                    decode_seconds=item['ready_monotonic']-item['decode_started_monotonic'],
                    latest_slot_seconds=record['dispatch_monotonic']-item['ready_monotonic'],
                    worker_queue_seconds=job['start_monotonic']-record['dispatch_monotonic'],
                    preprocessing_seconds=job['preprocessing_seconds'],full_model_and_codec_seconds=job['model_and_codec_seconds'],
                    owned_copy_seconds=job['owned_copy_and_validation_seconds'],
                    result_queue_seconds=event['received_monotonic']-job['end_monotonic'],
                    ordered_commit_wait_seconds=begin-event['received_monotonic'],SL4_and_points_seconds=optimized-begin,
                    publication_seconds=published-optimized,input_to_pose_seconds=published-item['captured_monotonic'],
                    graph_error=result['graph_error'],graph_nodes=result['graph_nodes'],pose=result['pose'],
                    prepared_images_raw_sha256=result['prepared_images_raw_sha256'])
                report['updates'].append(update);report['fresh_pose_count']+=1
                free.append(record['worker']);del pending[next_commit];next_commit+=1
                write(a.output_dir/'results.json',report);print(json.dumps(update,allow_nan=False),flush=True)
            if replay.exhausted and not pending:break
        if not report['updates']:raise RuntimeError('No fresh original VGGT+SL4 pose published')
        duration=report['updates'][-1]['published_monotonic']-replay.start_monotonic
        report.update(status='completed',EOF_tail_drained=True,fresh_wall_seconds_including_EOF_drain=duration,
            fresh_pose_hz=report['fresh_pose_count']/duration,
            input_to_pose_p95_seconds=float(np.percentile([u['input_to_pose_seconds'] for u in report['updates']],95)),
            map_chunks=len(manifest['chunks']),graph_nodes=len(manifest['nodes']),active_memory=memory())
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if replay is not None:
            try:replay.close()
            except BaseException as exc:report['cleanup_errors'].append('Decoder close: '+str(exc))
            report['source']=replay.metrics()
        stop_deadline=time.monotonic()+20;unsafe=False
        for index in reversed(threads):
            t=threads[index]
            if unsafe:
                report['cleanup_errors'].append(f'Worker{index} left alive after dependent timeout; release unknown')
            elif t.is_alive():
                try:commands[index].put(dict(action='stop'),timeout=min(1,max(.01,stop_deadline-time.monotonic())))
                except BaseException as exc:report['cleanup_errors'].append(f'Worker{index} stop: '+str(exc))
                while t.is_alive() and time.monotonic()<stop_deadline:
                    try:
                        event=events.get(timeout=min(.5,max(.01,stop_deadline-time.monotonic())))
                        if event['kind']=='closed':closed[event['worker']]=event['receipt']
                    except queue.Empty:pass
                    t.join(timeout=0)
            t.join(timeout=max(0,min(.1,stop_deadline-time.monotonic())))
            while not events.empty():
                try:
                    event=events.get_nowait()
                    if event['kind']=='closed':closed[event['worker']]=event['receipt']
                except queue.Empty:break
            if t.is_alive():unsafe=True;report['cleanup_errors'].append(f'Worker{index} still alive; external controller required')
            if index not in closed or closed[index]['status']!='closed':report['cleanup_errors'].append(f'Worker{index} failed/missing close')
            report['workers'][str(index)]=closed.get(index,dict(status='missing_final_receipt',thread_alive=t.is_alive()))
        final=registry_module.snapshot();report['shared_context_registry_final']=final
        if final['live_contexts'] or final['cleanup_errors'] or final['actual_contexts_created']!=final['actual_contexts_destroyed'] or not final['all_registered_contexts_final_zero']:
            report['cleanup_errors'].append('Shared registry physical cleanup incomplete')
        report['all_worker_threads_joined']=all(not t.is_alive() for t in threads.values())
        try:os.sched_setaffinity(0,affinity)
        except BaseException as exc:report['cleanup_errors'].append('Affinity restore: '+str(exc))
        report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        report['post_cleanup_memory']=memory()
        if report['cleanup_errors']:report['status']='failed_cleanup'
        write(a.output_dir/'results.json',report);log('parent','terminal '+report['status'])
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
