#!/usr/bin/env python3
"""Host CPU-only original native518 reference predictions for 16 overlapping pairs.

This is offline numerical-reference capture, never board performance or realtime
acceptance. The pinned original loader/model/executor/codec are unchanged; no
hooks, caches, tracing, accelerator installers or alternative models are used.
"""
import argparse
import faulthandler
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

ROOT = Path(__file__).resolve().parent
P = Path('/dev/shm/edge-vggt-v50/public-snapshot')
NATIVE_SHA = '3cdfa55556896b0f5356f710c65069b8b5e53f37a0f8cb6f550439f6dfb06730'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
EXTRA_PINS = {
    'vggt_slam/model_execution.py': 'dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87',
    'third_party/vggt/vggt/utils/pose_enc.py': '845fc3a46cb851bebf0720b8fc2b9f34334ae8e11f1982376ba59f90ff18f22b',
}
RGB_SHA = '532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3'
RECORDS_SHA = '59d29f9c7afe14c8b11faad515128636e95c4334fdf15058142cf31313557c38'
GT_SHA = 'aac0319a6ef4e1cdf61e779d2152b95aa7e9f7b1749d6d18717b43ddabffede2'
INDICES = [32+24*i for i in range(17)]
SHAPES = {'pose_enc': (2,9), 'extrinsic': (2,3,4), 'intrinsic': (2,3,3),
          'depth': (2,392,518,1), 'depth_conf': (2,392,518)}
SUCCESS = 'completed_original_native518_host_FP32_prediction_capture'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4 << 20), b''): h.update(part)
    return h.hexdigest()


def stat_identity(path):
    s = Path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def atomic(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=P)
    ap.add_argument('--vendor',type=Path,default=ROOT/'original-vggt-source')
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--rgb-index',type=Path,default=ROOT/'data/rgb.txt')
    ap.add_argument('--selected-records',type=Path,default=ROOT/'data/selected-records.json')
    ap.add_argument('--groundtruth',type=Path,default=ROOT/'data/groundtruth.txt')
    ap.add_argument('--native-reference',type=Path,default=Path('/dev/shm/edge-vggt-v92/complete-capture/capture.json'))
    ap.add_argument('--cpu-threads',type=int,choices=(4,8),default=4)
    ap.add_argument('--output-dir',type=Path,required=True)
    a = ap.parse_args()
    require(platform.system()=='Linux' and platform.machine() in ('x86_64','AMD64'), 'This reference runner executes on the authorized host CPU only')
    a.repo,a.vendor,a.checkpoint,a.rgb_index,a.selected_records,a.native_reference = [v.resolve(strict=True) for v in
        (a.repo,a.vendor,a.checkpoint,a.rgb_index,a.selected_records,a.native_reference)]
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    env,affinity,oldpath=dict(os.environ),os.sched_getaffinity(0),sys.path[:]
    handlers,pins={},{};model=executor=None;started=time.monotonic()
    report=dict(schema='v111_original_native518_host_CPU_FP32_predictions',status='running',script_sha256=sha(__file__),
        checkpoint=CHECKPOINT,original_model_only=True,execution_device='host_cpu',host_GPU_NPU_SDK_ORt_execution=False,
        image_size=518,prepared_image_shape=[2,3,392,518],source_indices=INDICES,pairs=[],cleanup_errors=[],
        max_loops=0,prune_unused_heads=False,patch_or_feature_cache_installed=False,forward_hooks_installed=False,
        original_model_methods_overridden=False,all_original_heads_retained=True,point_and_track_not_called_by_original_forward=True,
        full_model_forward_attempts=0,full_model_forward_calls_completed=0,module_forward_counts_instrumented=False,
        expected_original_per_forward=dict(DINO_blocks=24,frame_blocks=24,global_blocks=24,camera_refinements=4,depth_frames=2),
        intra_op_threads=a.cpu_threads,inter_op_threads=1,quality_or_board_performance_or_realtime_accepted=False,
        timing_scope='Offline host CPU full original forward and pose codec; preprocessing, validation/copy/hash/NPY IO separately recorded. Not board latency or a paced fresh stream.')
    atomic(out/'result.json',report);faulthandler.enable(all_threads=True)
    def progress(stage,**fields):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-started,**fields)
        atomic(out/'result.json',report)
        print(json.dumps(dict(stage=stage,elapsed_seconds=report['elapsed_seconds'],completed_pairs=len(report['pairs']),**fields)),flush=True)
    def pin(path,digest):
        path=Path(path).resolve(strict=True);require(sha(path)==digest,'Pinned source/input differs: '+str(path))
        pins[path]=digest;return path
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(a.cpu_threads),MKL_NUM_THREADS=str(a.cpu_threads),OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
        require(len(affinity)>=a.cpu_threads,'Requested bounded host CPU threads unavailable')
        cpus=set(sorted(affinity)[:a.cpu_threads]);os.sched_setaffinity(0,cpus)
        for number in (signal.SIGTERM,signal.SIGINT):
            def stop(n,frame):raise InterruptedError('External host controller stopped FP32 reference')
            handlers[number]=signal.signal(number,stop)
        pin(__file__,report['script_sha256'])
        native=json.loads(pin(a.native_reference,NATIVE_SHA).read_text())
        require(native['checkpoint']==CHECKPOINT and native['full_model_forward_completed'] and native['cleanup_errors']==[]
            and native['image_shape']==[2,3,392,518], 'Pinned complete original native518 reference required')
        expected=dict(native['source_hashes'],**EXTRA_PINS)
        for name,digest in expected.items():
            path=a.vendor/Path(name).relative_to('third_party/vggt') if name.startswith('third_party/vggt/') else a.repo/name
            pin(path,digest)
        # Freeze the additional original source files that the full loaded track
        # head imports, even though upstream forward never calls that head.
        for path in sorted(a.vendor.rglob('*.py')):pin(path,sha(path))
        pin(a.rgb_index,RGB_SHA);pin(a.selected_records,RECORDS_SHA);pin(a.groundtruth,GT_SHA)
        records=[]
        for line in a.rgb_index.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            stamp,relative=line.split(maxsplit=1);stamp=float(stamp)
            require(math.isfinite(stamp) and (not records or stamp>records[-1]['timestamp']), 'Original RGB index must remain chronological')
            records.append(dict(index=len(records),timestamp=stamp,relative_path=relative))
        selected=[records[i] for i in INDICES]
        require(len(records)==798 and selected==json.loads(a.selected_records.read_text()), 'Exact original 17-frame selection differs')
        sources=[]
        for row in selected:
            path=(a.rgb_index.parent/row['relative_path']).resolve(strict=True)
            require(path.is_relative_to(a.rgb_index.parent) and path.suffix=='.png','Original selected source outside dataset')
            digest=sha(path);pin(path,digest);sources.append(dict(row,path=str(path),bytes=path.stat().st_size,sha256=digest))
        require(len({v['sha256'] for v in sources})==17,'Selected observations must be fresh distinct files')
        checkpoint_before=stat_identity(a.checkpoint)
        require(checkpoint_before[2]==CHECKPOINT['bytes'],'Original full checkpoint byte size differs')
        report.update(source_images=sources,source_image_span_seconds=sources[-1]['timestamp']-sources[0]['timestamp'],
            native_reference_sha256=NATIVE_SHA,source_hashes=expected,vendor_source_sha256={str(p.relative_to(a.vendor)):h for p,h in pins.items() if p.is_relative_to(a.vendor)},
            rgb_index_sha256=RGB_SHA,selected_records_sha256=RECORDS_SHA,groundtruth_sha256=GT_SHA,model_CPU_affinity=sorted(cpus))
        sys.path[:0]=[str(a.vendor),str(a.repo)]
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        from vggt_slam.model_execution import TorchVGGTExecutor,prepare_rgb_images,execute
        from vggt.models import vggt as original_model_module
        from vggt.utils import pose_enc as original_codec_module
        require(Path(original_model_module.__file__).resolve()==a.vendor/'vggt/models/vggt.py'
            and Path(original_codec_module.__file__).resolve()==a.vendor/'vggt/utils/pose_enc.py','Wrong original vendor import selected')
        torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1)
        require(torch.get_default_dtype()==torch.float32 and os.environ['CUDA_VISIBLE_DEVICES']=='','CPU FP32/no host GPU required')
        report['versions']=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version())
        progress('original_checkpoint_hash_and_strict_mmap_meta_model_load')
        tick=time.monotonic()
        model=load_model(a.checkpoint,device='cpu',image_size=518,prune_unused_heads=False,offline=True)
        report['model_loading_including_checkpoint_SHA_seconds']=time.monotonic()-tick
        agg,dino=model.aggregator,model.aggregator.patch_embed
        require(len(dino.blocks)==len(agg.frame_blocks)==len(agg.global_blocks)==24 and not dino.chunked_blocks
            and agg.retained_layer_indices is None and agg.aa_order==['frame','global'] and agg.aa_block_size==1
            and agg.patch_size==14 and agg.patch_start_idx==5 and agg.target_layer==21
            and model.depth_frames_chunk_size==1 and model.inference_image_size==518
            and all(getattr(model,k+'_head') is not None for k in ('camera','depth','point','track')),'Original complete architecture/chunk1 differs')
        require(all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad for p in model.parameters())
            and all(not m.training and not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()), 'Original model must remain CPU FP32 eval without hooks')
        def state():
            return [(n,id(v),v.data_ptr(),v._version,str(v.dtype),str(v.device)) for n,v in model.named_parameters()]
        original_state=state();executor=TorchVGGTExecutor(model)
        report.update(checkpoint_verified_before_deserialization_by_pinned_loader=True,parameter_tensors=len(original_state),depth_frames_chunk_size=1)
        previous_tail=None
        for i in range(16):
            pair=sources[i:i+2];paths=[v['path'] for v in pair]
            progress('original_pair_start',pair_index=i,source_pair_indices=INDICES[i:i+2])
            tick=time.monotonic();images=prepare_rgb_images(paths,518);prepared=time.monotonic()
            require(images.shape==(2,3,392,518),'Reject changed original native518 input shape')
            raw=hashlib.sha256(images.tobytes()).hexdigest();frame_raw=[hashlib.sha256(v.tobytes()).hexdigest() for v in images]
            require(frame_raw[0]!=frame_raw[1] and (previous_tail is None or previous_tail==frame_raw[0]), 'Overlapping prepared pixels differ')
            previous_tail=frame_raw[1];report['full_model_forward_attempts']+=1
            atomic(out/'result.json',report);forward_start=time.monotonic()
            with torch.inference_mode():pred=execute(executor,images,compute_similarity=False)
            forward_done=time.monotonic();report['full_model_forward_calls_completed']+=1
            require(len(pred['pose_enc_list'])==4 and hashlib.sha256(images.tobytes()).hexdigest()==raw
                and state()==original_state,'Four original camera refinements/input/model state differ')
            outputs={}
            for name,shape in SHAPES.items():
                value=np.array(pred[name],copy=True,order='C')
                require(value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all(),'Original canonical output differs: '+name)
                path=out/f'pair{i:02d}-{name}.npy';temporary=path.with_suffix('.npy.tmp')
                with temporary.open('xb') as stream:np.save(stream,value,allow_pickle=False)
                temporary.replace(path)
                outputs[name]=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),shape=list(shape),dtype='float32',
                    raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(),tensor_bytes=value.nbytes,finite=True)
                del value
            report['pairs'].append(dict(index=i,source_indices=INDICES[i:i+2],timestamps=[v['timestamp'] for v in pair],image_paths=paths,
                source_images=pair,prepared_image_shape=list(images.shape),preprocessed_images_raw_sha256=raw,
                prepared_frame_raw_sha256=frame_raw,preprocess_seconds=prepared-tick,
                original_full_model_and_pose_codec_seconds=forward_done-forward_start,
                validation_copy_hash_save_seconds=time.monotonic()-forward_done,outputs=outputs,camera_refinement_count=4,
                complete_original_forward=True,input_unchanged=True,model_parameter_storage_and_versions_unchanged=True))
            del pred,images
            progress('original_pair_completed',pair_index=i)
        require(report['full_model_forward_calls_completed']==16 and stat_identity(a.checkpoint)==checkpoint_before,'Incomplete reference/changed checkpoint')
        report.update(status=SUCCESS,full_original_forward_count=16,model_state_unchanged=state()==original_state)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        executor=model=None
        for number,handler in handlers.items():signal.signal(number,handler)
        try:os.sched_setaffinity(0,affinity)
        except BaseException as exc:report['cleanup_errors'].append('CPU affinity: '+repr(exc))
        changed=[str(path) for path,digest in pins.items() if not path.is_file() or sha(path)!=digest]
        report.update(source_and_inputs_unchanged=not changed,changed_sources_or_inputs=changed,
            source_and_input_sha256={str(path):digest for path,digest in pins.items()},peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            total_host_wall_seconds=time.monotonic()-started)
        if changed:report['cleanup_errors'].append('Original source/input identity changed')
        os.environ.clear();os.environ.update(env);sys.path[:]=oldpath
        report.update(environment_restored=dict(os.environ)==env,affinity_restored=os.sched_getaffinity(0)==affinity,sys_path_restored=sys.path==oldpath,
            Torch_threads_are_process_local_and_worker_exits_after_capture=True)
        if report['cleanup_errors']:report['status']='failed_cleanup'
        atomic(out/'result.json',report)
    print(json.dumps(dict(status=report['status'],completed_pairs=len(report['pairs']),error=report.get('error'))),flush=True)
    return 0 if report['status']==SUCCESS else 1


if __name__=='__main__':raise SystemExit(main())
