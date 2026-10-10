#!/usr/bin/env python3
"""Board-only original FP32 hooks for one isolated Co-Me MLP experiment.

One complete unchanged original VGGT forward captures DINO block[14] output
and global block[0] MLP stages. Hooks only clone values and are always removed.
No merging, predictor inference, NPU, GT, ORB or online pipeline runs here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE/'src'
PROFILE_SHA = '6c3d2cb51af75410188c062e83a26eae4d4c5d69c632a6eeae95993eba004820'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CHECKPOINT_BYTES = 5026874952
WEIGHT_KEYS = ('mlp.fc1.weight','mlp.fc1.bias','mlp.fc2.weight','mlp.fc2.bias','ls2.gamma')
PREDICTION_KEYS = ('depth','depth_conf','extrinsic','intrinsic','pose_enc')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):
            h.update(part)
    return h.hexdigest()


def array_info(value):
    import numpy as np
    a = np.asarray(value)
    if (a.dtype not in (np.dtype('float32'),np.dtype('int64')) or not a.size
            or not a.flags.c_contiguous or not np.isfinite(a).all()):
        raise ValueError('Only finite contiguous FP32/int64 arrays may be captured')
    return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def bitwise_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes(order='C')==b.tobytes(order='C')


def board_guard(machine,output):
    if machine.lower() not in ('aarch64','arm64'):
        raise ValueError('Original capture is aarch64 board-only')
    if not Path(output).resolve().is_relative_to(BASE/'experiments'):
        raise ValueError('Output must be under board BASE/experiments')
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError('Output already exists; frozen experiments cannot be overwritten')


def validate_reference(profile,reference):
    if (profile.get('kind')!='original_VGGT_rectified_stage_and_block_capture'
            or profile.get('status')!='completed_original_CPU_profile_and_capture'
            or profile.get('image_size')!=336 or profile.get('original_weights') is not True
            or profile.get('architecture_changed') is not False or profile.get('npu_used') is not False
            or profile.get('ORB_used') is not False):
        raise ValueError('Expected the original CPU FP32 336 profile')
    if (reference.get('status')!='completed_fixed_input_diagnostic'
            or reference.get('image_size')!=336 or reference.get('precision')!='float32'
            or reference.get('device')!='cpu' or reference.get('original_weights') is not True
            or reference.get('npu_used') is not False or reference.get('ORB_used') is not False):
        raise ValueError('Expected the original 336 CPU resolution reference')
    if (profile['checkpoint']['sha256']!=CHECKPOINT_SHA
            or profile['checkpoint']['bytes']!=CHECKPOINT_BYTES
            or reference['checkpoint']['sha256']!=CHECKPOINT_SHA
            or reference['checkpoint']['bytes']!=CHECKPOINT_BYTES):
        raise ValueError('Official original checkpoint identity differs')
    pair,selection = reference['pairs'][0],reference['selected_inputs'][0]
    ids = pair['frame_ids']
    if (pair['revision']!=1 or selection['revision']!=1 or len(ids)!=2
            or not all(type(fid) in (int,float) and math.isfinite(fid) for fid in ids)
            or not ids[0]<ids[1] or selection['frame_ids']!=ids
            or profile['selected_pair']!=selection
            or pair['input_tensor']!=profile['input_tensor']
            or pair['input_tensor']['shape']!=[2,3,252,336]):
        raise ValueError('Exact first TUM pair or actual prepared shape differs')
    for fid,image in zip(ids,selection['images']):
        if (image['frame_id']!=fid or image['timestamp']!=fid
                or image['schedule_token']['frame_id']!=fid
                or image['schedule_token']['timestamp']!=fid
                or image.get('samefile_validated') is not True):
            raise ValueError('Original image/schedule timestamp identity differs')
    if len(selection['images'])!=2:
        raise ValueError('Capture requires exactly two real images')
    return pair,selection


def predictor_features(stage,grid_hw=(18,24),special_tokens=5):
    """Official direct.py stage: no final norm; B1,S,C,Hpatch,Wpatch."""
    import numpy as np
    a = np.asarray(stage)
    h,w = grid_hw
    if (a.dtype!=np.float32 or a.ndim!=3 or a.shape[0]!=2
            or a.shape[1]!=special_tokens+h*w or not np.isfinite(a).all()):
        raise ValueError('DINO block14 must contain view-local special+patch tokens')
    patches = a[:,special_tokens:,:].transpose(0,2,1).reshape(1,2,a.shape[2],h,w)
    return np.ascontiguousarray(patches)


class Captures:
    """Never return replacement inputs/outputs or execute extra model calls."""
    def __init__(self,model):
        self.handles,self.values,self.calls = [],{},{}
        global0 = model.aggregator.global_blocks[0]
        modules = {
            'dino14_output':model.aggregator.patch_embed.blocks[14],
            'norm2':global0.norm2,'mlp_output':global0.mlp,
            'post_ls2':global0.ls2,'block_output':global0,
        }
        try:
            for key,module in modules.items():
                self.calls[key] = 0
                self.handles.append(module.register_forward_hook(self.after(key)))
            self.handles.append(global0.norm2.register_forward_pre_hook(self.before_residual))
            self.handles.append(global0.register_forward_pre_hook(self.before_block,with_kwargs=True))
        except BaseException:
            self.close()
            raise

    def copy(self,key,value):
        if key in self.values or value.device.type!='cpu' or str(value.dtype) not in ('torch.float32','torch.int64'):
            raise ValueError(f'Unexpected invocation or nonoriginal capture dtype: {key}')
        self.values[key] = value.detach().clone().contiguous()

    def after(self,key):
        def hook(module,args,output):
            self.copy(key,output)
            self.calls[key] += 1
        return hook

    def before_residual(self,module,args):
        self.copy('residual1',args[0])

    def before_block(self,module,args,kwargs):
        if kwargs.get('viz_attention',False):
            raise ValueError('Capture may not visualize or alter attention')
        self.copy('global0_input',args[0])
        pos = kwargs.get('pos',args[1] if len(args)>1 else None)
        if pos is None:
            raise ValueError('Original global0 must carry RoPE positions')
        self.copy('global0_pos',pos)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run(args):
    import numpy as np
    report = dict(kind='original_VGGT_CoMe_MLP_stage_capture',status='started',
        script_sha256=sha(__file__),profile_sha256=args.profile_sha256,
        original_weights=True,architecture_changed=False,ORB_used=False,npu_used=False,
        GT_used=False,host_inference_executed=False,inference_on_board_only=True,
        full_CoMe_model_validated=False,realtime_validated=False,quality_validated=False,
        predefined_reference_gate='FP32_C_order_bytes_exact',full_model_forward_count=0,
        cleanup_errors=[],hooks_removed=False)
    started = time.monotonic()
    old_affinity = os.sched_getaffinity(0)
    captures = None
    try:
        cpus = {int(v) for v in args.cpus.split(',')}
        if not cpus or not cpus<=old_affinity or args.threads<1:
            raise ValueError('CPU/threads setting is outside allowed board affinity')
        os.sched_setaffinity(0,cpus)
        os.environ.update(OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
            OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',MPLBACKEND='Agg')
        sys.path[:0] = [str(SOURCE),str(SOURCE/'third_party/vggt')]
        profile_path = args.profile_dir/'profile.json'
        if args.profile_sha256!=PROFILE_SHA or sha(profile_path)!=PROFILE_SHA:
            raise ValueError('Pinned original 336 profile SHA differs')
        profile = json.loads(profile_path.read_text())
        if sha(args.reference_receipt)!=profile['reference_sha256']:
            raise ValueError('Underlying original336 capture receipt SHA differs')
        reference = json.loads(args.reference_receipt.read_text())
        pair,selection = validate_reference(profile,reference)
        sources = {name:sha(SOURCE/name) for name in profile['source_hashes']}
        if sources!=profile['source_hashes']:
            raise ValueError('Original model/preprocessing source changed')
        report['source_hashes'] = sources
        report['selected_pair'] = selection
        report['reference_sha256'] = profile['reference_sha256']
        checkpoint = Path(profile['checkpoint']['path'])
        if checkpoint.stat().st_size!=CHECKPOINT_BYTES or sha(checkpoint)!=CHECKPOINT_SHA:
            raise ValueError('Official checkpoint bytes/SHA differ')
        checkpoint_stat = checkpoint.stat()
        report['checkpoint'] = profile['checkpoint']
        report['checkpoint_hash_before_load_warms_filesystem_cache'] = True
        calibration = Path(reference['calibration']['path'])
        if sha(calibration)!=reference['calibration']['sha256']:
            raise ValueError('Original calibration SHA differs')
        for image in selection['images']:
            if (sha(image['path'])!=image['sha256']
                    or not Path(image['path']).samefile(image['schedule_token']['path'])):
                raise ValueError('Actual source image identity or SHA differs')
        original_file = args.reference_receipt.parent/pair['file']
        if sha(original_file)!=pair['sha256']:
            raise ValueError('Original saved full prediction file SHA differs')
        block_row = profile['blocks']['global0']
        old_block_file = args.profile_dir/block_row['artifacts']['tensors']['file']
        if sha(old_block_file)!=block_row['artifacts']['tensors']['sha256']:
            raise ValueError('Original global0 capture artifact SHA differs')
        import cv2
        import torch
        from vggt_slam.image_rectification import CalibrationRectifier
        from vggt_slam.edge_mapping import prepare_rectified_images
        from vggt_slam.runtime import load_model
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        cv2.setNumThreads(1)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        report['runtime'] = dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
            opencv=cv2.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),
            threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
        rectifier = CalibrationRectifier(json.loads(calibration.read_text()))
        if rectifier.metadata()!=reference['rectification']:
            raise ValueError('Rectification method/metadata changed')
        report['rectification'] = rectifier.metadata()
        begin = time.monotonic()
        images = prepare_rectified_images([im['path'] for im in selection['images']],rectifier,336)
        pixels = images.numpy()
        if (list(pixels.shape)!=[2,3,252,336]
                or hashlib.sha256(pixels.tobytes(order='C')).hexdigest()!=profile['input_tensor']['tensor_sha256']):
            raise ValueError('Prepared pixels are not byte-identical to original profile')
        report['tensor_preparation_seconds'] = time.monotonic()-begin
        np.save(args.output/'images.npy',pixels,allow_pickle=False)
        begin = time.monotonic()
        model = load_model(checkpoint,device='cpu',image_size=336,prune_unused_heads=True,offline=True)
        report['model_load_seconds'] = time.monotonic()-begin
        if (model.training or sum(p.numel() for p in model.parameters())!=1157941492
                or any(p.dtype!=torch.float32 or p.device.type!='cpu' for p in model.parameters())
                or model.aggregator.patch_start_idx!=5):
            raise ValueError('Original eval FP32 architecture or special-token count differs')
        block = model.aggregator.global_blocks[0]
        state = {key:np.ascontiguousarray(value.detach().numpy()) for key,value in block.state_dict().items()}
        if set(state)!=set(block_row['weights']):
            raise ValueError('Original global0 parameter keys differ')
        state_pins = {key:array_info(value) for key,value in state.items()}
        if state_pins!=block_row['weights']:
            raise ValueError('Original global0 parameter bytes differ')
        report['global0_state'] = state_pins
        report['architecture'] = dict(mlp_class=type(block.mlp).__name__,gelu_approximate=block.mlp.act.approximate,
            dropout=block.mlp.drop.p,norm2_eps=block.norm2.eps,ls2_inplace=block.ls2.inplace,
            views=2,grid_hw=[18,24],special_tokens=5,channels=1024,
            dino_hook='aggregator.patch_embed.blocks[14] forward output',
            dino_extra_norm=False,predictor_feature_layout='B,S,C,Hpatch,Wpatch',
            global_flat_layout='B=1,view_contiguous(special5_then_row_major_patches),C')
        if block.mlp.act.approximate!='none' or block.mlp.drop.p!=0.:
            raise ValueError('Original MLP exact GELU/eval dropout differs')
        captures = Captures(model)
        begin = time.monotonic()
        with torch.inference_mode():
            raw = model(images)
        report['hooked_full_forward_seconds'] = time.monotonic()-begin
        report['full_model_forward_count'] = 1
        arrays = {key:np.ascontiguousarray(value.numpy()) for key,value in captures.values.items()}
        if any(count!=1 for count in captures.calls.values()):
            raise ValueError('Each original hooked stage must run exactly once')
        if arrays['dino14_output'].shape!=(2,437,1024) or any(
                arrays[key].shape!=(1,874,1024) for key in ('global0_input','residual1','norm2','mlp_output','post_ls2','block_output')):
            raise ValueError('Actual captured stage layout differs from original336 S2')
        arrays['predictor_features'] = predictor_features(arrays['dino14_output'])
        if not bitwise_equal(arrays['residual1']+arrays['post_ls2'],arrays['block_output']):
            raise ValueError('Full residual1+postLS2 does not exactly reconstruct block output')
        with np.load(old_block_file,allow_pickle=False) as old:
            report['global0_vs_profile'] = {key:bitwise_equal(arrays[new],old[key]) for key,new in (
                ('input','global0_input'),('output','block_output'),('pos','global0_pos'))}
        if not all(report['global0_vs_profile'].values()):
            raise ValueError('Hooked global0 input/position/output changed vs frozen profile')
        extrinsic,intrinsic = pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
        predictions = {key:np.ascontiguousarray(value.detach().numpy()) for key,value in dict(
            depth=raw['depth'],depth_conf=raw['depth_conf'],pose_enc=raw['pose_enc'],
            extrinsic=extrinsic,intrinsic=intrinsic).items()}
        with np.load(original_file,allow_pickle=False) as old:
            report['full_predictions_vs_original'] = {key:bitwise_equal(predictions[key],old[key]) for key in PREDICTION_KEYS}
        if not all(report['full_predictions_vs_original'].values()):
            raise ValueError('Full original prediction bytes changed')
        np.savez(args.output/'activations.npz',**arrays)
        np.savez(args.output/'mlp-state.npz',**{key:state[key] for key in WEIGHT_KEYS})
        np.savez(args.output/'predictions.npz',**predictions,frame_ids=np.asarray(pair['frame_ids'],np.float64))
        report['arrays'] = {key:array_info(value) for key,value in arrays.items()}
        report['mlp_state_keys'] = list(WEIGHT_KEYS)
        report['artifacts'] = {name:dict(sha256=sha(args.output/name),bytes=(args.output/name).stat().st_size)
            for name in ('images.npy','activations.npz','mlp-state.npz','predictions.npz')}
        if {name:sha(SOURCE/name) for name in sources}!=sources:
            raise ValueError('Source changed during capture')
        if (checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)!=(checkpoint_stat.st_size,checkpoint_stat.st_mtime_ns):
            raise ValueError('Checkpoint changed during capture')
        if sha(calibration)!=reference['calibration']['sha256'] or any(
                sha(im['path'])!=im['sha256'] for im in selection['images']):
            raise ValueError('Source images or calibration changed')
        report.update(status='captured_original_FP32_stages_bitwise_bound',sources_and_inputs_unchanged=True)
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if captures is not None:
            try:
                captures.close()
                report['hooks_removed'] = not captures.handles
            except BaseException as exc:
                report['cleanup_errors'].append(f'hooks: {exc}')
        try:
            os.sched_setaffinity(0,old_affinity)
            report['cpu_affinity_restored'] = True
        except OSError as exc:
            report['cleanup_errors'].append(f'affinity: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report.update(total_seconds=time.monotonic()-started,
            peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (args.output/'capture.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['status']=='captured_original_FP32_stages_bitwise_bound' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-dir',type=Path,required=True)
    p.add_argument('--profile-sha256',default=PROFILE_SHA)
    p.add_argument('--reference-receipt',type=Path,required=True)
    p.add_argument('--cpus',default='4,5,6,7')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    try:
        board_guard(platform.machine(),args.output)
        args.output.mkdir(parents=True,exist_ok=False)
    except (ValueError,FileExistsError) as exc:
        p.error(str(exc))
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
