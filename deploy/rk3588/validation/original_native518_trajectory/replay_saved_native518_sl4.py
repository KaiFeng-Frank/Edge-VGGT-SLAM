#!/usr/bin/env python3
"""Replay 16 saved original native518 predictions through the unchanged SL4 solver.

RK3588/S CPU geometry only: no model loading, NN, SDK or GPU execution. Accept
four pinned native batches or one pinned host original-FP32 capture. Original
pixel preprocessing, submap equations, confidence masks and SL4 optimization
are unchanged. Save original poses/H and deterministic unfiltered pixel samples;
this replay is not prediction throughput, trajectory-quality or realtime acceptance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

BASE=Path('/mnt/vggt-data/vggt-slam2');SRC=BASE/'src'
CHECKPOINT=dict(bytes=5026874952,sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
NATIVE_PRODUCER_SHA='78cb60f56df0b98879a987b34db1172c62cd0d51b5364cb1c3c81fd6f3ac1fe9'
CPU_PRODUCER_SHA='23e29fa0679e80755153b055702129309f39e495cf3349a9d080fa3980a5190c'
EXECUTOR_SHA='e153a5f71ef0b10119207b2e43589b628a2e85ab1abeaeb72d2df2a7dd6fb638'
CONFIG_SHA='7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
RGB_SHA='532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3'
SOURCE_PINS={
 'vggt_slam/model_execution.py':'dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87',
 'vggt_slam/solver.py':'33b5b4d816cd08f0cb9857f89c9e3fc770ad0c2ab3c9dc938864f00cefe86905',
 'vggt_slam/graph.py':'dda8fb7c947c3c39baa7d190e2942cd4b9f26fe1557d7605fdc8651f27a5ab48',
 'vggt_slam/submap.py':'fddccafee4e9f32452ce044b80669bd48f5fe5f331a980986a6091e7fce92e63',
 'vggt_slam/map.py':'2d7b374101248519d921e16e272063ff6f7e001c7af0073a7a7c13c14ac79143',
 'vggt_slam/slam_utils.py':'aeba669bcb263f24e6bd93e5a176bc028e2022770499bd5ea1c3eeeebbbd5008',
 'vggt_slam/scale_solver.py':'dc36b11d4d9953222a3fc2f3aef274acc9e9cc6dc61cb22d86b1e6721d9bd691',
 'vggt_slam/frame_overlap.py':'bbb78f549a34b58eaf27556e6dda703468cc0518de4b936f7f886882d151afb8',
 'vggt_slam/runtime.py':'f36bb5b87fbdbf03984ea3e1466e6ee27865a540ae3e5abd4ee07a45538e23f5',
 'vggt_slam/checkpoint_identity.py':'9fc0b927f6a68ac6ca55194616104febb1c488370593d4be60e8dba5bc336461',
 'vggt_slam/timestamp_replay.py':'14b168028a6f7722f4bea036285b4f9cd8119fed59230b7e3e07c4b7af07addc',
 'third_party/vggt/vggt/utils/load_fn.py':'b0fe762f65497dca9a22069a3e4a37fd3508398c55383a8c1cb85c403fb3e9dd',
 'third_party/vggt/vggt/utils/geometry.py':'3e14a94b5c947685c8b332984e7d115d4ef3e9b060137efd07df79ece8ccb7ce',
 'third_party/vggt/vggt/dependency/distortion.py':'d95ab35395ea5887138e0855599298a91d55bb9c49e0c4025af5f2e1879bf5e8',
}
INDICES=[32+24*i for i in range(17)]
SHAPES={'pose_enc':(2,9),'extrinsic':(2,3,4),'intrinsic':(2,3,3),
        'depth':(2,392,518,1),'depth_conf':(2,392,518)}
SUCCESS='completed_saved_original_native518_16_pair_SL4_replay_not_quality_accepted'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def raw_sha(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def atomic(path,row):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-manifests',type=Path,nargs='+',required=True)
    p.add_argument('--manifest-sha256',nargs='+',required=True)
    p.add_argument('--rgb-index',type=Path,default=BASE/'datasets/rgbd_dataset_freiburg1_xyz/rgb.txt')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=float,default=170.)
    a=p.parse_args();sys.dont_write_bytecode=True
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'SL4 saved prediction consumption executes on the RK3588/S board only')
    require(len(a.prediction_manifests)==len(a.manifest_sha256) and len(a.prediction_manifests) in (1,4),
            'Supply one CPU manifest or four native manifests and their exact SHA256 pins')
    require(0<a.timeout_seconds<=180,'Run under the existing bounded board controller')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+a.timeout_seconds
    env,affinity,oldpath=dict(os.environ),os.sched_getaffinity(0),sys.path[:]
    pins,handlers={},{};executor=solver=None
    r=dict(schema='v111_original_native518_saved_prediction_SL4_replay',status='running',script_sha256=sha(__file__),
        source_indices=INDICES,checkpoint=CHECKPOINT,max_loops=0,image_size=518,confidence_threshold_percentile=25.,
        prepared_image_shape=[2,3,392,518],single_original_solver_without_batch_reset=True,
        model_loaded=False,NN_SDK_GPU_executed=False,system_clock_writes=False,saved_prediction_replay=True,
        fresh_prediction_rate_or_latency_measured=False,quality_realtime_P95_or_8GB_accepted=False,
        PLY_exported=False,sample_flat_stride=50,samples_are_not_confidence_filtered=True,
        timing_scope='Board CPU saved-array verification/preprocessing and original SL4 consumption; not NN inference or fresh observation pose throughput.',
        prediction_manifests=[],pairs=[],artifacts={},cleanup_errors=[])
    atomic(out/'result.json',r)

    def pin(path,digest):
        path=Path(path).resolve(strict=True)
        require(sha(path)==digest,'Pinned original source/input differs: '+str(path))
        pins[path]=digest;return path

    def progress(stage,**fields):
        require(time.monotonic()<deadline,'Bounded saved replay deadline reached at '+stage)
        r['stage']=stage;atomic(out/'result.json',r)
        print(json.dumps(dict(stage=stage,**fields)),flush=True)

    try:
        os.sched_setaffinity(0,{4,5,6,7})
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
        for number in (signal.SIGTERM,signal.SIGINT):
            def stop(n,frame):raise InterruptedError('External bounded board controller stopped saved SL4 replay')
            handlers[number]=signal.signal(number,stop)
        pin(__file__,r['script_sha256'])
        for name,digest in SOURCE_PINS.items():pin(SRC/name,digest)
        sys.path[:0]=[str(SRC),str(SRC/'third_party/vggt')]
        import numpy as np
        import torch
        import gtsam
        import scipy
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from vggt_slam.model_execution import ExecutionCapabilities
        from vggt_slam.timestamp_replay import read_rgb_index
        from vggt_slam.solver import Solver
        r['versions']=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,gtsam=str(getattr(gtsam,'__version__','not_exposed')))
        records=read_rgb_index(pin(a.rgb_index,RGB_SHA));selected=[records[i] for i in INDICES]
        sources=[]
        for index,record in zip(INDICES,selected):
            path=Path(record['path']).resolve(strict=True);digest=sha(path);pin(path,digest)
            sources.append(dict(index=index,timestamp=record['timestamp'],file=str(path),bytes=path.stat().st_size,sha256=digest))
        require(len({row['sha256'] for row in sources})==17,'Original 17 distinct observations required')
        r.update(rgb_index_sha256=RGB_SHA,source_images=sources,source_duration_seconds=sources[-1]['timestamp']-sources[0]['timestamp'])

        def logical_source(row):
            return {key:row[key] for key in ('index','timestamp','bytes','sha256')}

        logical_sources=[logical_source(row) for row in sources]
        r['logical_source_selection_sha256']=hashlib.sha256(json.dumps(logical_sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        pair_records={};kinds=set();native_starts=set()
        for manifest_path,digest in zip(a.prediction_manifests,a.manifest_sha256):
            path=pin(manifest_path,digest);receipt=json.loads(path.read_text())
            require(receipt['checkpoint']==CHECKPOINT and receipt['original_model_only'] and receipt['max_loops']==0
                and receipt['source_indices']==INDICES and receipt['rgb_index_sha256']==RGB_SHA
                and receipt['source_and_inputs_unchanged'] and receipt['cleanup_errors']==[],
                'Original complete prediction provenance/cleanup required')
            require([logical_source(row) for row in receipt['source_images']]==logical_sources,
                    'Original source index/timestamp/PNG SHA differs from board dataset')
            if receipt['schema']=='v111_original_native518_trajectory_prediction_batch':
                kind='native';kinds.add(kind);native_starts.add(receipt['start_pair'])
                z=receipt['executor_resources']
                require(receipt['status']=='completed_actual_native518_four_pair_batch_not_quality_or_realtime_accepted'
                    and receipt['script_sha256']==NATIVE_PRODUCER_SHA and receipt['executor_sha256']==EXECUTOR_SHA
                    and receipt['config_sha256']==CONFIG_SHA and receipt['core_mask']==7 and receipt['pair_count']==4
                    and receipt['prepared_shape']==[2,3,392,518] and not receipt['SL4_backend_executed']
                    and receipt['DINO_layers_per_frame']==24 and receipt['aggregator_blocks_per_pair']==48
                    and receipt['camera_refinements_per_pair']==4 and receipt['depth_frames_per_pair']==2
                    and z['closed'] and z['resources_balanced'] and not z['live_contexts'] and not z['live_output_buffers']
                    and not z['cleanup_errors'] and z['SDK_totals']==dict(contexts_created=40,contexts_destroyed=40,
                        runs_completed=60,output_buffers_acquired=80,output_buffers_released=80)
                    and len(receipt['pairs'])==4,'Native full-model batch identity/resource contract differs')
                require([row['index'] for row in receipt['pairs']]==list(range(receipt['start_pair'],receipt['start_pair']+4)),
                        'Native complete four-pair ordering differs')
            elif receipt['schema']=='v111_original_native518_host_CPU_FP32_predictions':
                kind='original_fp32';kinds.add(kind)
                require(receipt['status']=='completed_original_native518_host_FP32_prediction_capture'
                    and receipt['script_sha256']==CPU_PRODUCER_SHA and receipt['execution_device']=='host_cpu'
                    and not receipt['host_GPU_NPU_SDK_ORt_execution'] and not receipt['prune_unused_heads']
                    and not receipt['patch_or_feature_cache_installed'] and not receipt['forward_hooks_installed']
                    and not receipt['original_model_methods_overridden'] and receipt['all_original_heads_retained']
                    and receipt['prepared_image_shape']==[2,3,392,518] and receipt['model_state_unchanged']
                    and receipt['checkpoint_verified_before_deserialization_by_pinned_loader']
                    and receipt['full_original_forward_count']==receipt['full_model_forward_calls_completed']==16
                    and receipt['depth_frames_chunk_size']==1 and len(receipt['pairs'])==16,
                    'Original unchanged CPU FP32 reference contract differs')
            else:raise ValueError('Unrecognized prediction schema: '+str(receipt['schema']))
            r['prediction_manifests'].append(dict(file=str(path),sha256=digest,schema=receipt['schema'],
                producer_source_sha256=receipt['script_sha256'],kind=kind,pair_indices=[row['index'] for row in receipt['pairs']]))
            for row in receipt['pairs']:
                i=row['index'];require(type(i) is int and 0<=i<16 and i not in pair_records,'Repeated/missing/unrecognized pair ID')
                require(row['source_indices']==INDICES[i:i+2] and row['timestamps']==[v['timestamp'] for v in sources[i:i+2]]
                    and [logical_source(v) for v in row['source_images']]==logical_sources[i:i+2]
                    and row['prepared_image_shape']==[2,3,392,518] and set(row['outputs'])==set(SHAPES),
                    'Pair source/pixel-grid/canonical-output roles differ')
                pair_records[i]=dict(row=row,manifest=path,manifest_sha256=digest,kind=kind)
        require(set(pair_records)==set(range(16)) and len(kinds)==1
            and ((kinds=={'native'} and native_starts=={0,4,8,12} and len(a.prediction_manifests)==4)
                 or (kinds=={'original_fp32'} and len(a.prediction_manifests)==1)),
            'Exactly all16 ordered native or original-FP32 pairs required')
        r['prediction_kind']=next(iter(kinds))

        def load_output(bundle,name):
            metadata=bundle['row']['outputs'][name];relative=Path(metadata['file'])
            require(not relative.is_absolute() and '..' not in relative.parts,'Prediction files must be receipt-relative')
            root=bundle['manifest'].parent;path=(root/relative).resolve(strict=True)
            require(path.is_relative_to(root) and path.stat().st_size==metadata['bytes'],'Prediction file path/size differs')
            pin(path,metadata['sha256']);value=np.load(path,allow_pickle=False)
            require(value.dtype==np.float32 and value.shape==SHAPES[name] and list(value.shape)==metadata['shape']
                and str(value.dtype)==metadata['dtype'] and value.nbytes==metadata['tensor_bytes']
                and raw_sha(value)==metadata['raw_sha256'] and metadata['finite'] and np.isfinite(value).all(),
                'Saved original canonical tensor differs: '+name)
            return np.array(value,copy=True,order='C')

        class SavedExecutor:
            capabilities=ExecutionCapabilities('pinned original native518 saved prediction replay; no inference',False)
            device='cpu'
            def __init__(self):self.next_pair=0;self.calls=[];self.closed=False
            def predict(self,images,*,compute_similarity=False):
                require(not self.closed and not compute_similarity and images.shape==(2,3,392,518),
                        'Saved consumption supports only the original ordinary two-frame request')
                bundle=pair_records[self.next_pair];row=bundle['row']
                expected=row.get('prepared_pair_raw_sha256',row.get('preprocessed_images_raw_sha256'))
                actual=raw_sha(images);frames=[raw_sha(value) for value in images]
                require(actual==expected and frames==row['prepared_frame_raw_sha256'],
                        'Board original prepared pixels differ from saved prediction inputs')
                result={name:load_output(bundle,name) for name in SHAPES}
                require(raw_sha(images)==actual,'Saved executor must not mutate caller pixels')
                result.update(images=images,image_match_ratio=None)
                self.calls.append(dict(index=self.next_pair,prepared_pair_raw_sha256=actual,
                    prepared_frame_raw_sha256=frames,manifest_sha256=bundle['manifest_sha256'],
                    outputs={name:row['outputs'][name] for name in SHAPES},owned_output_arrays=True))
                self.next_pair+=1;return result
            def close(self):self.closed=True

        progress('single_original_solver_initializing')
        executor=SavedExecutor();solver=Solver(init_conf_threshold=25.,device='cpu',image_size=518,max_loops=0,offline=True,enable_viewer=False)
        previous_tail=None
        for i in range(16):
            progress('saved_pair_original_SL4_start',index=i)
            paths=[row['file'] for row in sources[i:i+2]];tick=time.monotonic()
            prediction=solver.run_predictions(paths,executor,max_loops=0);loaded=time.monotonic()
            call=executor.calls[-1]
            require(call['index']==i and (previous_tail is None or previous_tail==call['prepared_frame_raw_sha256'][0]),
                    'Prepared overlap differs across prediction batch boundaries')
            previous_tail=call['prepared_frame_raw_sha256'][1]
            solver.add_points(prediction);added=time.monotonic()
            initial_error=float(solver.graph.graph.error(solver.graph.values));solver.graph.optimize();optimized=time.monotonic()
            final_error=float(solver.graph.graph.error(solver.graph.values));H=solver.map.get_all_homographies(solver.graph)
            determinants=np.linalg.det(H)
            require(np.isfinite(initial_error) and np.isfinite(final_error) and H.shape==(2*(i+1),4,4)
                and np.isfinite(H).all() and np.isfinite(determinants).all() and (determinants!=0).all()
                and len(solver.graph.initialized_nodes)==2*(i+1) and solver.graph.graph.size()==2*(i+1)
                and solver.map.get_num_submaps()==i+1 and solver.graph.get_num_loops()==0,
                'Original SL4 finite/invertible graph or complete node/factor/submap counts differ')
            submap=solver.map.get_latest_submap()
            require(submap.get_frame_ids()==[v['timestamp'] for v in sources[i:i+2]]
                and submap.pointclouds.shape==(2,392,518,3) and np.isfinite(submap.pointclouds).all(),
                'Original frame IDs or unprojected point geometry differs')
            r['pairs'].append(dict(index=i,source_indices=INDICES[i:i+2],timestamps=submap.get_frame_ids(),
                submap_id=submap.get_id(),source_manifest_sha256=call['manifest_sha256'],
                prepared_pair_raw_sha256=call['prepared_pair_raw_sha256'],prepared_frame_raw_sha256=call['prepared_frame_raw_sha256'],
                saved_output_raw_sha256={name:call['outputs'][name]['raw_sha256'] for name in SHAPES},
                preprocessing_saved_array_verification_and_owned_copy_seconds=loaded-tick,
                original_add_points_seconds=added-loaded,original_graph_optimization_seconds=optimized-added,
                graph_initial_error=initial_error,graph_final_error=final_error,node_count=2*(i+1),
                factor_count=2*(i+1),submap_count=i+1,confidence_threshold=float(submap.get_conf_threshold()),
                local_unprojected_points_finite=True,homographies_finite_and_invertible=True))
            atomic(out/'result.json',r);del prediction,H
        require(executor.next_pair==16,'Saved execution must consume all16 complete pairs exactly once')
        progress('final_original_pose_and_geometry_export')

        def save(name,value):
            value=np.ascontiguousarray(value);path=out/(name+'.npy')
            require(np.isfinite(value).all(),'Cannot save nonfinite original geometry: '+name)
            with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
            row=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),shape=list(value.shape),
                dtype=str(value.dtype),tensor_bytes=value.nbytes,raw_sha256=raw_sha(value),finite=True)
            r['artifacts'][name]=row;atomic(out/'result.json',r);return row

        pose_path=out/'poses-tum.txt';solver.map.write_poses_to_file(str(pose_path),solver.graph,kitti_format=False)
        poses=np.loadtxt(pose_path);expected_stamps=np.asarray([v['timestamp'] for v in sources],dtype=np.float64)
        require(poses.shape==(17,8) and np.isfinite(poses).all() and np.array_equal(poses[:,0],expected_stamps),
                'Original pose writer must emit17 finite chronological original timestamps')
        r['poses_tum']=dict(file=pose_path.name,bytes=pose_path.stat().st_size,sha256=sha(pose_path),
            rows=17,columns=8,finite=True,original_writer_unchanged=True)
        homographies=solver.map.get_all_homographies(solver.graph);save('homographies',homographies)
        save('poses_tum_rows',poses)
        flat_indices=np.arange(0,392*518,50,dtype=np.int64)
        sampled_points=np.empty((32,len(flat_indices),3),dtype=np.float64)
        sampled_masks=np.empty((32,len(flat_indices)),dtype=np.bool_)
        full_masks=np.empty((32,392,518),dtype=np.bool_)
        node_ids=[];source_indices=[];timestamps=[];frame_metadata=[]
        offset=0
        for pair_index,submap in enumerate(solver.map.ordered_submaps_by_key()):
            points,ids,masks=submap.get_points_list_in_world_frame(solver.graph)
            require(len(points)==len(ids)==len(masks)==2,'Exactly two original world point grids per submap required')
            for frame_in_pair,(world_points,stamp,mask) in enumerate(zip(points,ids,masks)):
                require(world_points.shape==(392,518,3) and np.isfinite(world_points).all()
                    and mask.shape==(392,518) and mask.dtype==np.bool_
                    and stamp==sources[pair_index+frame_in_pair]['timestamp'],
                    'Nonfinite full original world geometry or changed confidence mask/frame association')
                sampled_points[offset]=world_points.reshape(-1,3)[flat_indices]
                sampled_masks[offset]=mask.reshape(-1)[flat_indices];full_masks[offset]=mask
                node_id=submap.get_id()+frame_in_pair;source_index=INDICES[pair_index+frame_in_pair]
                node_ids.append(node_id);source_indices.append(source_index);timestamps.append(stamp)
                frame_metadata.append(dict(occurrence=offset,pair_index=pair_index,frame_in_pair=frame_in_pair,
                    submap_id=submap.get_id(),node_id=node_id,source_index=source_index,timestamp=stamp,
                    full_point_count=392*518,full_confident_point_count=int(mask.sum()),
                    sampled_point_count=len(flat_indices),sampled_confident_point_count=int(sampled_masks[offset].sum()),
                    original_confidence_threshold=float(submap.get_conf_threshold()),all_world_points_finite=True))
                offset+=1
            del points,masks
        require(offset==32 and len(set(timestamps))==17,'Expected32 frame occurrences and17 distinct observations required')
        save('sample_flat_indices',flat_indices);save('sampled_world_points',sampled_points)
        save('sampled_confidence_masks',sampled_masks);save('full_confidence_masks',full_masks)
        save('frame_occurrence_node_ids',np.asarray(node_ids,dtype=np.int64))
        save('frame_occurrence_source_indices',np.asarray(source_indices,dtype=np.int64))
        save('frame_occurrence_timestamps',np.asarray(timestamps,dtype=np.float64))
        determinants=np.linalg.det(homographies)
        r.update(status=SUCCESS,pair_count=16,submap_count=solver.map.get_num_submaps(),node_count=len(solver.graph.initialized_nodes),
            factor_count=solver.graph.graph.size(),loop_closure_count=solver.graph.get_num_loops(),
            distinct_source_observation_count=17,frame_occurrence_count=32,homography_count=len(homographies),
            homography_determinants=determinants.tolist(),all_geometry_finite_and_invertible=True,
            per_frame_geometry=frame_metadata,full_unfiltered_point_count=32*392*518,
            full_confident_point_count=int(full_masks.sum()),sampled_unfiltered_point_count=32*len(flat_indices),
            sampled_confident_point_count=int(sampled_masks.sum()),sampled_world_gauge='Original final optimized SL4 world; no Sim3 alignment or SE3/metric gauge claim.',
            all_input_and_prediction_records_validated=True,saved_executor_calls=len(executor.calls))
    except BaseException as exc:
        r.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());traceback.print_exc()
        if solver is not None:
            r['partial_backend_counts']=dict(submaps=solver.map.get_num_submaps(),nodes=len(solver.graph.initialized_nodes),
                factors=solver.graph.graph.size(),loops=solver.graph.get_num_loops())
    finally:
        if executor is not None:executor.close();r['saved_executor_closed']=executor.closed
        for number,handler in handlers.items():signal.signal(number,handler)
        try:os.sched_setaffinity(0,affinity)
        except BaseException as exc:r['cleanup_errors'].append('Affinity restore: '+repr(exc))
        for path,digest in pins.items():
            try:require(sha(path)==digest,'Pinned source/input/NPY changed: '+str(path))
            except BaseException as exc:r['cleanup_errors'].append(str(exc))
        os.environ.clear();os.environ.update(env);sys.path[:]=oldpath
        r.update(source_and_inputs_unchanged=not r['cleanup_errors'],source_and_input_sha256={str(p):v for p,v in pins.items()},
            environment_restored=dict(os.environ)==env,affinity_restored=os.sched_getaffinity(0)==affinity,
            sys_path_restored=sys.path==oldpath,total_wall_seconds=time.monotonic()-started,
            peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            Torch_threads_process_local_worker_exits_after_replay=True)
        if r['cleanup_errors'] or not r['environment_restored'] or not r['affinity_restored'] or not r['sys_path_restored']:
            r['status']='failed_cleanup'
        atomic(out/'result.json',r);print(json.dumps({k:r.get(k) for k in ('status','error','cleanup_errors')}),flush=True)
    return 0 if r['status']==SUCCESS else 1


if __name__=='__main__':raise SystemExit(main())
