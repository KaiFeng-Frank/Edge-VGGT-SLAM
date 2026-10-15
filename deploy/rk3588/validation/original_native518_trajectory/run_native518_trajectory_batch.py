#!/usr/bin/env python3
"""Four actual native518 pairs for later unchanged original SL4 quality replay.

RK3588/S only. The frozen v110 executor uses original full models, mask7 and
one previous-tail DINO cache within this batch. No SLAM/PLY or reference reset
runs here. Saved model outputs are actual predictions, not a fresh-rate claim.
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
import sys
import time
import traceback

BASE=Path('/mnt/vggt-data/vggt-slam2');SRC=BASE/'src';E=BASE/'experiments'
S100=E/'vggt-v100-source-r1';S110=E/'vggt-v110-source-r1'
EXECUTOR_SHA='e153a5f71ef0b10119207b2e43589b628a2e85ab1abeaeb72d2df2a7dd6fb638'
CONFIG_SHA='7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
SOURCE_PINS={
    'vggt_slam/model_execution.py':'dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87',
    'vggt_slam/timestamp_replay.py':'14b168028a6f7722f4bea036285b4f9cd8119fed59230b7e3e07c4b7af07addc',
    'third_party/vggt/vggt/utils/load_fn.py':'b0fe762f65497dca9a22069a3e4a37fd3508398c55383a8c1cb85c403fb3e9dd',
}
INDICES=[32+24*i for i in range(17)]
OUTPUTS=('pose_enc','extrinsic','intrinsic','depth','depth_conf')
SUCCESS='completed_actual_native518_four_pair_batch_not_quality_or_realtime_accepted'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def atomic(path,row):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start-pair',type=int,choices=(0,4,8,12),required=True)
    p.add_argument('--pair-count',type=int,choices=(4,),default=4)
    p.add_argument('--executor-source',type=Path,default=S110/'v110_native518_executor.py')
    p.add_argument('--config',type=Path,default=S100/'full-native-sequential-config-r1.json')
    p.add_argument('--frozen-source-dir',type=Path,default=S100)
    p.add_argument('--rgb-index',type=Path,default=BASE/'datasets/rgbd_dataset_freiburg1_xyz/rgb.txt')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=float,default=170.)
    a=p.parse_args();sys.dont_write_bytecode=True
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'Deployment execution is RK3588/S board only')
    require(0<a.timeout_seconds<=180,'Existing bounded180s/75C/1GHz-restoring controller required')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+a.timeout_seconds
    oldenv,oldaff,oldpath=dict(os.environ),os.sched_getaffinity(0),sys.path[:]
    handlers,pins={},{};executor=None
    r=dict(schema='v111_original_native518_trajectory_prediction_batch',status='running',script_sha256=sha(__file__),
        executor_sha256=EXECUTOR_SHA,config_sha256=CONFIG_SHA,start_pair=a.start_pair,pair_count=a.pair_count,
        source_indices=INDICES,core_mask=7,max_loops=0,original_model_only=True,prepared_shape=[2,3,392,518],
        DINO_layers_per_frame=24,aggregator_blocks_per_pair=48,camera_refinements_per_pair=4,depth_frames_per_pair=2,
        cache_scope='One original previous tail, within this4pair batch only; clear onfailure/close',
        timestamp_paced=False,SL4_backend_executed=False,PLY_exported=False,
        original_CPU_FP32_reference_or_v109_same_pair_byte_gate_available=False,
        trajectory_quality_realtime_P95_or_8GB_accepted=False,pairs=[],cleanup_errors=[],
        timing_scope='Original file read/preprocessing and actual complete native model; prediction saves and receipt checks are separate. Startup source/checkpoint/compiled-artifact checks are outside per-pair timing.')
    atomic(out/'result.json',r)

    def pin(path,digest):
        path=Path(path).resolve(strict=True)
        require(sha(path)==digest,'Pinned original source/input changed: '+str(path))
        pins[path]=digest;return path

    def progress(stage,**fields):
        require(time.monotonic()<deadline,'Bounded prediction batch deadline reached at '+stage)
        print(json.dumps(dict(stage=stage,**fields)),flush=True)

    try:
        os.sched_setaffinity(0,{4,5,6,7})
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        for number in (signal.SIGINT,signal.SIGTERM):
            def stop(n,frame):raise InterruptedError('External bounded controller stopped native prediction batch')
            handlers[number]=signal.signal(number,stop)
        pin(__file__,r['script_sha256']);source=pin(a.executor_source,EXECUTOR_SHA);pin(a.config,CONFIG_SHA)
        for name,digest in SOURCE_PINS.items():pin(SRC/name,digest)
        sys.path[:0]=[str(SRC),str(SRC/'third_party/vggt')]
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from vggt_slam.timestamp_replay import read_rgb_index
        from vggt_slam.model_execution import execute,prepare_rgb_images
        index=Path(a.rgb_index).resolve(strict=True);index_sha=sha(index)
        records=read_rgb_index(pin(index,index_sha))
        require(len(records)>INDICES[-1],'Original real RGB index must include source416')
        selected=[records[i] for i in INDICES];source_rows=[]
        for position,item in zip(INDICES,selected):
            path=Path(item['path']).resolve(strict=True);digest=sha(path);pin(path,digest)
            source_rows.append(dict(index=position,timestamp=item['timestamp'],file=str(path),
                                    bytes=path.stat().st_size,sha256=digest))
        require(len({v['sha256'] for v in source_rows})==17,'Distinct original source files required')
        selection_sha=hashlib.sha256(json.dumps(source_rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        r.update(rgb_index_sha256=index_sha,source_images=source_rows,source_selection_sha256=selection_sha,
                 selected_source_duration_seconds=source_rows[-1]['timestamp']-source_rows[0]['timestamp'])
        atomic(out/'result.json',r)
        spec=importlib.util.spec_from_file_location('_v111_frozen_original_tail_executor',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        progress('frozen_native_executor_initializing');tick=time.monotonic()
        executor=module.Native518Executor(config=a.config,config_sha256=CONFIG_SHA,frozen_source_dir=a.frozen_source_dir,
            capture_intermediates=False,source_root=SRC,expected_npu_hz=1000000000,reuse_previous_tail=True,core_mask=7)
        r.update(executor_init_seconds=time.monotonic()-tick,startup_seconds=time.monotonic()-started,
                 checkpoint=executor.initialization['checkpoint'],executor_initialization=copy.deepcopy(executor.initialization))
        atomic(out/'result.json',r)
        previous_tail=None;seen=set()

        def save(name,value):
            value=np.ascontiguousarray(value);path=out/(name+'.npy')
            with path.open('xb') as f:np.save(f,value,allow_pickle=False)
            return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),shape=list(value.shape),dtype=str(value.dtype),
                tensor_bytes=value.nbytes,raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(),finite=bool(np.isfinite(value).all()))

        for pair_index in range(a.start_pair,a.start_pair+a.pair_count):
            rows=source_rows[pair_index:pair_index+2];paths=[v['file'] for v in rows]
            progress('actual_fresh_prediction_pair_start',index=pair_index,timestamps=[v['timestamp'] for v in rows])
            begin=time.monotonic();images=prepare_rgb_images(paths,518);prepared=time.monotonic()
            require(images.shape==(2,3,392,518),'Exact original native pixel grid required')
            frame_raw=[hashlib.sha256(value.tobytes()).hexdigest() for value in images]
            pair_raw=hashlib.sha256(images.tobytes()).hexdigest()
            require(frame_raw[0]!=frame_raw[1] and (previous_tail is None or frame_raw[0]==previous_tail) and
                    frame_raw[1] not in seen,'Every pair must introduce a new tail and preserve exact overlap pixels')
            seen.update(frame_raw);previous_tail=frame_raw[1]
            prediction=execute(executor,images,compute_similarity=False);predicted=time.monotonic()
            call=copy.deepcopy(executor.last_call);local_index=pair_index-a.start_pair
            require(call['status']=='completed' and call['source_and_inputs_unchanged'] and call['resources_balanced'] and
                call['complete_expected_execution'] and call['input_raw_sha256']==pair_raw and
                call['input_frame_raw_sha256']==frame_raw and call['tail_cache_hit']==(local_index>0) and
                call['requested_core_mask']==7,'Actual fresh input/cache/resource receipt differs')
            outputs={name:save('pair'+str(pair_index)+'-'+name,prediction[name]) for name in OUTPUTS}
            item=dict(index=pair_index,source_indices=[v['index'] for v in rows],timestamps=[v['timestamp'] for v in rows],
                source_images=rows,prepared_image_shape=list(images.shape),prepared_pair_raw_sha256=pair_raw,
                prepared_frame_raw_sha256=frame_raw,preprocessing_seconds=prepared-begin,
                prediction_seconds=predicted-prepared,image_read_to_canonical_prediction_seconds=predicted-begin,
                save_and_postcheck_seconds=time.monotonic()-predicted,outputs=outputs,executor_call=call)
            r['pairs'].append(item);atomic(out/'result.json',r)
            progress('actual_fresh_prediction_pair_saved',index=pair_index,seconds=item['prediction_seconds'],
                     tail_cache_hit=call['tail_cache_hit'])
            del prediction,images
        r['status']=SUCCESS
    except BaseException as exc:
        executor=getattr(exc,'native_executor',executor)
        r.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        if executor is not None:
            try:executor.close()
            except BaseException as exc:r['cleanup_errors'].append('Executor close: '+repr(exc))
            try:
                z=executor.metrics();r['executor_resources']=z
                if r['status']==SUCCESS:
                    require(z['SDK_totals']==dict(contexts_created=40,contexts_destroyed=40,runs_completed=60,
                        output_buffers_acquired=80,output_buffers_released=80) and z['calls_completed']==4 and
                        z['closed'] and z['resources_balanced'] and not z['live_contexts'] and not z['live_output_buffers'] and
                        not z['cleanup_errors'] and z['previous_tail_cache']['hits']==3 and
                        z['previous_tail_cache']['misses']==1 and not z['previous_tail_cache']['cache_entries'],
                        'Expected4actual predictions/40contexts/60runs/80buffers/3hits and final cacheclear required')
            except BaseException as exc:r['cleanup_errors'].append('Executor resources: '+repr(exc))
        for number,handler in handlers.items():signal.signal(number,handler)
        try:os.sched_setaffinity(0,oldaff)
        except BaseException as exc:r['cleanup_errors'].append('Affinity restore: '+repr(exc))
        os.environ.clear();os.environ.update(oldenv);sys.path[:]=oldpath
        for path,digest in pins.items():
            try:require(sha(path)==digest,'Pinned original source/input changed: '+str(path))
            except BaseException as exc:r['cleanup_errors'].append(str(exc))
        r.update(source_and_inputs_unchanged=not r['cleanup_errors'],source_and_input_sha256={str(p):v for p,v in pins.items()},
            environment_restored=dict(os.environ)==oldenv,affinity_restored=os.sched_getaffinity(0)==oldaff,
            sys_path_restored=sys.path==oldpath,total_wall_seconds=time.monotonic()-started,
            peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if r['cleanup_errors'] or not r['environment_restored'] or not r['affinity_restored'] or not r['sys_path_restored']:
            r['status']='failed_cleanup'
        atomic(out/'result.json',r);print(json.dumps({k:r.get(k) for k in ('status','error','cleanup_errors')}),flush=True)
    return 0 if r['status']==SUCCESS else 1


if __name__=='__main__':raise SystemExit(main())
