#!/usr/bin/env python3
"""Isolated board-only MLP-only Co-Me approximation on three original pairs.

One original FP32 VGGT model and one separately released BF16 CPU confidence
predictor execute a fixed ABBA/AB/BA sequence. Only the research context changes
57 MLP token layouts. Norms, attention, LayerScale, original full residuals,
CameraHead and DPT remain in their original modules. This is not complete Co-Me,
SLAM, a GT accuracy gate, NPU execution or a realtime benchmark.

Timed forward includes per-scene BF16 conversion, predictor, plan, merge and
split. Context installation/parameter hashing, preprocessing, pose decoding,
array validation and output I/O are outside that timing and recorded separately.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
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
HERE = Path(__file__).resolve().parent
REFERENCE_SHA = '8a5b9e8a7f94757fa906fbcc1b2d37ae3d4751076d69e966dd626a15a6c63598'
STAGE_SHA = '3664f5c932602a3c6538ea6013ea0389625c42bdcb7dcf10a9fbceb9304abeb1'
CAPTURE_SCRIPT_SHA = '59c1b346bdda45fe1130eab0f869f47732755cc95761c16c43d3c0346805bc14'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CHECKPOINT_BYTES = 5026874952
PREDICTOR_SHA = '95b0a004544c96da5c4e55748e8cf4837608627a6414563af06cb3ee25acd29f'
PREDICTION_SHAPES = dict(depth=(1,2,252,336,1),depth_conf=(1,2,252,336),
    intrinsic=(1,2,3,3),extrinsic=(1,2,3,4),pose_enc=(1,2,9))
CALL_ORDER = ((0,'original'),(0,'merged'),(0,'merged'),(0,'original'),
              (1,'original'),(1,'merged'),(2,'merged'),(2,'original'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):
            h.update(part)
    return h.hexdigest()


def array_info(value):
    import numpy as np
    a = np.asarray(value)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('Every numerical artifact must be finite and nonempty')
    return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def bitwise_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes(order='C')==b.tobytes(order='C')


def validate_reference(row):
    if (row.get('kind')!='fixed_rectified_original_VGGT_resolution_pairs'
            or row.get('status')!='completed_fixed_input_diagnostic' or row.get('image_size')!=336
            or row.get('precision')!='float32' or row.get('device')!='cpu'
            or row.get('original_weights') is not True or row.get('ORB_used') is not False
            or row.get('npu_used') is not False or len(row.get('pairs',[]))!=3
            or len(row.get('selected_inputs',[]))!=3
            or row.get('checkpoint',{}).get('sha256')!=CHECKPOINT_SHA
            or row['checkpoint'].get('bytes')!=CHECKPOINT_BYTES):
        raise ValueError('Expected frozen original FP32 336 three-pair reference')
    for index,(pair,selection) in enumerate(zip(row['pairs'],row['selected_inputs'])):
        ids = pair.get('frame_ids',[])
        if (pair.get('revision')!=index+1 or selection.get('revision')!=index+1
                or len(ids)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in ids)
                or not ids[0]<ids[1] or selection.get('frame_ids')!=ids
                or len(selection.get('images',[]))!=2
                or pair.get('input_tensor',{}).get('shape')!=[2,3,252,336]
                or pair['input_tensor'].get('dtype')!='float32'
                or set(pair.get('arrays',{}))!=set(PREDICTION_SHAPES)):
            raise ValueError('Actual pair/order/shape differs from frozen reference')
        for key,shape in PREDICTION_SHAPES.items():
            info = pair['arrays'][key]
            if info.get('shape')!=list(shape) or info.get('dtype')!='float32' or info.get('finite') is not True:
                raise ValueError('Reference predictions must retain original B1 FP32 shape')
        for fid,image in zip(ids,selection['images']):
            token = image.get('schedule_token',{})
            if (image.get('frame_id')!=fid or image.get('timestamp')!=fid
                    or token.get('frame_id')!=fid or token.get('timestamp')!=fid
                    or image.get('samefile_validated') is not True):
                raise ValueError('Exact original image/schedule frame identity differs')
        if index and ids[0]!=row['pairs'][index-1]['frame_ids'][-1]:
            raise ValueError('Reference overlap source identity differs')


def validate_predictions(arrays):
    import numpy as np
    if set(arrays)!=set(PREDICTION_SHAPES):
        raise ValueError('All five original prediction arrays are required')
    for key,shape in PREDICTION_SHAPES.items():
        if arrays[key].dtype!=np.float32 or arrays[key].shape!=shape:
            raise ValueError(f'Original B1 prediction dtype/shape changed: {key}')
        array_info(arrays[key])


def timed_forward(call,clock=time.perf_counter):
    started = clock()
    raw = call()
    return raw,clock()-started


def board_guard(machine,output):
    if machine.lower() not in ('aarch64','arm64'):
        raise ValueError('Actual fullmodel research is aarch64 board-only')
    if not Path(output).resolve().is_relative_to(BASE/'experiments'):
        raise ValueError('Output must be under board BASE/experiments')
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError('Output already exists; experiments cannot be overwritten')


def load_pinned_module(path,digest,name):
    if sha(path)!=digest:
        raise ValueError(f'Explicit implementation SHA mismatch: {path.name}')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_forward_state(model):
    """Instance replacements/hooks only; no tensor hashing inside forward."""
    targets = [block.mlp for block in model.aggregator.patch_embed.blocks[15:24]]
    targets += [block.mlp for block in model.aggregator.frame_blocks]
    targets += [block.mlp for block in model.aggregator.global_blocks]
    hooks = [model.aggregator,model.aggregator.patch_embed.blocks[14]]
    return ([('forward' in mod.__dict__,mod.__dict__.get('forward')) for mod in targets],
            [(tuple(mod._forward_pre_hooks),tuple(mod._forward_hooks)) for mod in hooks])


def save_scene(path,scene):
    import numpy as np
    import torch
    def raw_bf16(value):
        if value.dtype!=torch.bfloat16 or value.device.type!='cpu':
            raise ValueError('Raw BF16 artifact must retain original CPU tensor precision')
        return value.detach().contiguous().view(torch.int16).numpy().view(np.uint16).copy()
    plan = scene['plan']
    confidence,means,median = (scene[key] for key in ('confidence','group_means','confidence_quantile'))
    if tuple(confidence.shape)!=(2,432,1) or tuple(means.shape)!=(2,48) or tuple(median.shape)!=(1,1):
        raise ValueError('Completed scene predictor outputs have unexpected shape')
    recomputed = confidence.reshape(2,6,3,8,3).permute(0,1,3,2,4).reshape(2,48,9).mean(-1)
    expected_median = torch.quantile(recomputed.float().reshape(1,-1),.5,dim=-1,keepdim=True)
    expected_mask = (recomputed.reshape(1,-1)<expected_median).reshape(2,6,8).numpy()
    if (not torch.equal(means,recomputed) or not torch.equal(median,expected_median)
            or not np.array_equal(plan.merge_mask,expected_mask)):
        raise ValueError('Scene did not preserve the BF16 mean/FP32 strict median rule')
    arrays = dict(features=scene['features'].detach().contiguous().numpy(),
        predictor_features_bf16_bits=raw_bf16(scene['predictor_features']),
        confidence=confidence.float().numpy(),confidence_bf16_bits=raw_bf16(confidence),
        group_means=means.float().reshape(2,6,8).numpy(),group_means_bf16_bits=raw_bf16(means.reshape(2,6,8)),
        median_threshold=median.numpy(),merge_mask=plan.merge_mask,
        reverse_indices=plan.reverse_indices,view_offsets=plan.view_offsets,token_weights=plan.token_weights,
        merged_patch_mask=plan.merged_patch_mask,representative_indices=plan.representative_indices,
        mean_input_indices=plan.mean_input_indices,mean_output_indices=plan.mean_output_indices)
    np.savez(path,**arrays)
    return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),
                arrays={key:array_info(value) for key,value in arrays.items() if value.size},
                BF16_storage='uint16 raw bit patterns; FP32 confidence/group means are exact expansion',
                quantile=.5,strict_less_than=True)


def run(args):
    report = dict(kind='isolated_original_weight_MLP_only_CoMe_fullmodel_research',status='started',
        script_sha256=sha(__file__),reference_capture_sha256=args.reference_sha256,
        stage_capture_sha256=args.stage_capture_sha256,image_size=336,device='cpu',precision='float32',
        original_weights=True,original_VGGT_parameter_shapes_preserved=True,
        original_computation_preserved=False,original_VGGT_modules_and_heads_retained=True,
        computation_changed='MLP-only token mean/split in DINO15..23 and frame/global0..23; 57 branches',
        predictor_precision='BF16_CPU',predictor_CUDA_bitwise_equivalence_claim=False,
        full_CoMe_method_implemented=False,npu_used=False,ORB_used=False,GT_used=False,
        inference_on_board_only=True,online_pipeline_run=False,quality_validated=False,realtime_validated=False,
        predefined_original_gate='FP32_C_order_bytes_exact',approximation_outputs_must_be_finite=True,
        call_order=[dict(pair_index=p,method=m) for p,m in CALL_ORDER],call_records=[],
        full_model_forward_attempts=0,full_model_forward_count=0,cleanup_errors=[],
        timing_scope='model(images) only, including each fresh predictor/plan/merge/split; excludes context parameter hashes, preprocessing, codec, array validation and file I/O',
        warmup_forwards=0)
    started = time.monotonic()
    old_affinity = os.sched_getaffinity(0)
    handle = None
    try:
        cpus = {int(v) for v in args.cpus.split(',')}
        if not cpus or not cpus<=old_affinity or args.threads<1:
            raise ValueError('CPU/thread setting is invalid or outside allowed affinity')
        os.sched_setaffinity(0,cpus)
        os.environ.update(OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
            OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',MPLBACKEND='Agg')
        sys.path[:0] = [str(SOURCE),str(SOURCE/'third_party/vggt')]
        import numpy as np
        import torch
        import cv2
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        report['runtime'] = dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,opencv=cv2.__version__,
            cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),opencv_threads=cv2.getNumThreads(),
            environment={key:os.environ[key] for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_WAIT_POLICY','GOMP_SPINCOUNT')})
        if args.reference_sha256!=REFERENCE_SHA or sha(args.reference_receipt)!=REFERENCE_SHA:
            raise ValueError('Frozen original three-pair reference receipt SHA differs')
        reference = json.loads(args.reference_receipt.read_text())
        validate_reference(reference)
        stage_path = args.stage_capture_dir/'capture.json'
        if args.stage_capture_sha256!=STAGE_SHA or sha(stage_path)!=STAGE_SHA:
            raise ValueError('Frozen original stage capture receipt SHA differs')
        stage = json.loads(stage_path.read_text())
        if (stage.get('status')!='captured_original_FP32_stages_bitwise_bound'
                or stage.get('script_sha256')!=CAPTURE_SCRIPT_SHA
                or stage.get('reference_sha256')!=REFERENCE_SHA
                or stage.get('selected_pair')!=reference['selected_inputs'][0]
                or stage.get('sources_and_inputs_unchanged') is not True or stage.get('hooks_removed') is not True
                or stage.get('cleanup_errors') or stage.get('full_model_forward_count')!=1):
            raise ValueError('Original stage capture identity/lifecycle differs')
        merger = load_pinned_module(args.merge_module,args.merge_module_sha256,'_v22_full_driver_merge')
        wrapper = load_pinned_module(args.wrapper_module,args.wrapper_module_sha256,'_v22_full_driver_wrapper')
        if (merger.RELEASED_CHECKPOINT_SHA256!=PREDICTOR_SHA
                or wrapper.MERGE_HELPER_SHA256!=args.merge_module_sha256
                or args.merge_module.resolve()!=args.wrapper_module.with_name('v22_token_merge.py').resolve()):
            raise ValueError('Research wrapper/merger/released predictor binding differs')
        official = {name:sha(args.official_source_dir/name) for name in merger.SOURCE_PINS}
        if official!=merger.SOURCE_PINS:
            raise ValueError('Pinned official Co-Me source differs')
        sources = dict(reference['source_hashes'])
        for name,value in stage['source_hashes'].items():
            if name in sources and sources[name]!=value:
                raise ValueError('Stage/reference original model sources disagree')
            sources[name] = value
        if {name:sha(SOURCE/name) for name in sources}!=sources:
            raise ValueError('Original model/preprocessing source differs')
        report.update(source_hashes=sources,selected_inputs=reference['selected_inputs'],
            rectification=reference['rectification'],checkpoint=reference['checkpoint'],
            calibration=reference['calibration'],rgb_index=reference['rgb_index'],
            preprocessing_provenance=reference['preprocessing_provenance'],
            CoMe_source=dict(commit=merger.SOURCE_COMMIT,files=official,license='CC BY-NC 4.0 research adaptation'),
            research_sources=dict(wrapper_sha256=args.wrapper_module_sha256,merger_sha256=args.merge_module_sha256))
        checkpoint = Path(reference['checkpoint']['path'])
        if checkpoint.stat().st_size!=CHECKPOINT_BYTES or sha(checkpoint)!=CHECKPOINT_SHA:
            raise ValueError('Official checkpoint bytes/SHA differ')
        checkpoint_stat = checkpoint.stat()
        report['checkpoint_hash_before_load_warms_filesystem_cache'] = True
        for row in (reference['calibration'],reference['rgb_index']):
            if sha(row['path'])!=row['sha256']:
                raise ValueError('Original calibration/RGB index SHA differs')
        for selection in reference['selected_inputs']:
            if sha(selection['schedule_file'])!=selection['schedule_sha256']:
                raise ValueError('Original scheduling artifact SHA differs')
            for image in selection['images']:
                if sha(image['path'])!=image['sha256'] or not Path(image['path']).samefile(image['schedule_token']['path']):
                    raise ValueError('Original source image identity/SHA differs')
        originals = []
        for pair in reference['pairs']:
            path = args.reference_receipt.parent/pair['file']
            if sha(path)!=pair['sha256'] or path.stat().st_size!=pair['bytes']:
                raise ValueError('Frozen original prediction artifact differs')
            with np.load(path,allow_pickle=False) as saved:
                arrays = {key:saved[key].copy() for key in PREDICTION_SHAPES}
                if not bitwise_equal(saved['frame_ids'],np.asarray(pair['frame_ids'],np.float64)):
                    raise ValueError('Frozen prediction frame IDs differ')
            validate_predictions(arrays)
            if any(array_info(value)['sha256']!=pair['arrays'][key]['tensor_sha256'] for key,value in arrays.items()):
                raise ValueError('Frozen original prediction raw SHA differs')
            originals.append(arrays)
        if sha(args.stage_capture_dir/'activations.npz')!=stage['artifacts']['activations.npz']['sha256']:
            raise ValueError('Original DINO14 stage artifact SHA differs')
        with np.load(args.stage_capture_dir/'activations.npz',allow_pickle=False) as saved:
            first_features = np.ascontiguousarray(saved['predictor_features'].transpose(0,1,3,4,2))
        from vggt_slam.image_rectification import CalibrationRectifier
        from vggt_slam.edge_mapping import prepare_rectified_images
        from vggt_slam.runtime import load_model
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        rectifier = CalibrationRectifier(json.loads(Path(reference['calibration']['path']).read_text()))
        if rectifier.metadata()!=reference['rectification']:
            raise ValueError('Original shared rectification metadata differs')
        prepared = []
        report['tensor_preparation_seconds'] = []
        for index,selection in enumerate(reference['selected_inputs']):
            begin = time.perf_counter()
            images = prepare_rectified_images([image['path'] for image in selection['images']],rectifier,336)
            report['tensor_preparation_seconds'].append(time.perf_counter()-begin)
            info = array_info(images.numpy())
            if (info['shape']!=[2,3,252,336] or info['dtype']!='float32'
                    or info['sha256']!=reference['pairs'][index]['input_tensor']['tensor_sha256']):
                raise ValueError('Actual prepared RGB is not byte-identical to frozen original input')
            np.save(args.output/f'input-pair-{index}.npy',images.numpy(),allow_pickle=False)
            prepared.append(images)
        begin = time.perf_counter()
        model = load_model(checkpoint,device='cpu',image_size=336,prune_unused_heads=True,offline=True)
        report['model_load_seconds'] = time.perf_counter()-begin
        if (model.training or sum(p.numel() for p in model.parameters())!=1157941492
                or any(p.dtype!=torch.float32 or p.device.type!='cpu' for p in model.parameters())):
            raise ValueError('Original CPU eval FP32 model architecture differs')
        if args.predictor_checkpoint.stat().st_size!=831722 or sha(args.predictor_checkpoint)!=PREDICTOR_SHA:
            raise ValueError('Exact released BF16 predictor checkpoint differs')
        begin = time.perf_counter()
        predictor = merger.load_released_predictor(args.predictor_checkpoint,args.predictor_config,
            expected_checkpoint_sha256=PREDICTOR_SHA,expected_config_sha256=args.predictor_config_sha256)
        report['predictor_load_seconds'] = time.perf_counter()-begin
        report['predictor'] = dict(checkpoint_sha256=PREDICTOR_SHA,config_sha256=args.predictor_config_sha256,
            precision='BF16_CPU',threshold=.5,strict_less_than_median=True,specials_excluded=True,no_final_DINO_norm=True)
        original_forward_state = model_forward_state(model)
        with torch.inference_mode():
            for order_index,(pair_index,method) in enumerate(CALL_ORDER):
                images = prepared[pair_index]
                scene,scene_record,metrics = None,None,None
                installation_seconds = restoration_seconds = 0.
                if method=='merged':
                    handle = wrapper.ComeMLPResearch(model,predictor,threshold=.5,
                        expected_grid_hw=(18,24),expected_views=2,channels=1024)
                    begin = time.perf_counter()
                    handle.__enter__()
                    installation_seconds = time.perf_counter()-begin
                    try:
                        report['full_model_forward_attempts'] += 1
                        raw,forward_seconds = timed_forward(lambda:model(images))
                        report['full_model_forward_count'] += 1
                        scene = handle.last_scene
                        scene_record = handle.scene_record()
                    finally:
                        primary_error = sys.exc_info()[1]
                        begin = time.perf_counter()
                        try:
                            handle.close()
                        except BaseException as exc:
                            report['cleanup_errors'].append(f'context close: {type(exc).__name__}: {exc}')
                            if primary_error is None:
                                raise
                        finally:
                            restoration_seconds = time.perf_counter()-begin
                            try:
                                metrics = handle.metrics()
                                report.setdefault('research_contexts',[]).append(metrics)
                            except BaseException as exc:
                                report['cleanup_errors'].append(f'context metrics: {type(exc).__name__}: {exc}')
                                if primary_error is None:
                                    raise
                    if (metrics['closed'] is not True or metrics['active_scene'] is not False
                            or metrics['original_weight_hashes_unchanged'] is not True or metrics['cleanup_errors']
                            or scene_record['status']!='completed'
                            or scene_record['mlp_calls']!={'dino':9,'frame':24,'global':24}
                            or scene_record.get('synthetic_plan_callback_used') is not False
                            or scene_record.get('synthetic_predictor_callback_used') is not False):
                        raise ValueError('Research context did not execute/restore exactly 57 MLP branches')
                    if pair_index==0 and not bitwise_equal(scene['features'].numpy(),first_features):
                        raise ValueError('First merged scene DINO14 raw features differ from original captured stage')
                else:
                    report['full_model_forward_attempts'] += 1
                    raw,forward_seconds = timed_forward(lambda:model(images))
                    report['full_model_forward_count'] += 1
                if model_forward_state(model)!=original_forward_state:
                    raise ValueError('Original MLP forward identities or hooks were not restored')
                begin = time.perf_counter()
                extrinsic,intrinsic = pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
                arrays = {key:np.ascontiguousarray(value.detach().numpy()) for key,value in dict(
                    depth=raw['depth'],depth_conf=raw['depth_conf'],pose_enc=raw['pose_enc'],
                    extrinsic=extrinsic,intrinsic=intrinsic).items()}
                validate_predictions(arrays)
                reference_equal = {key:bitwise_equal(arrays[key],originals[pair_index][key]) for key in PREDICTION_SHAPES}
                if method=='original' and not all(reference_equal.values()):
                    raise ValueError('Same-input original model did not pass frozen reference byte-exact gate')
                path = args.output/f'call-{order_index:02d}-pair-{pair_index}-{method}.npz'
                np.savez(path,**arrays,frame_ids=np.asarray(reference['pairs'][pair_index]['frame_ids'],np.float64))
                record = dict(pair_index=pair_index,method=method,order_index=order_index,
                    frame_ids=reference['pairs'][pair_index]['frame_ids'],
                    pair_source_images=[dict(path=im['path'],sha256=im['sha256']) for im in reference['selected_inputs'][pair_index]['images']],
                    input_tensor_sha256=reference['pairs'][pair_index]['input_tensor']['tensor_sha256'],
                    forward_seconds=forward_seconds,context_installation_seconds=installation_seconds,
                    context_restoration_seconds=restoration_seconds,
                    arrays={key:array_info(value) for key,value in arrays.items()},
                    original_reference_bitwise_equal=reference_equal,
                    artifact=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)),
                    peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
                if scene is not None:
                    record.update(scene_record=scene_record,research_metrics=metrics,
                                  scene_artifact=save_scene(args.output/f'scene-plan-{order_index:02d}.npz',scene))
                record['post_forward_decode_validate_hash_save_seconds'] = time.perf_counter()-begin
                report['call_records'].append(record)
                print(json.dumps(dict(order_index=order_index,pair_index=pair_index,method=method,
                    forward_seconds=forward_seconds,
                    compact_rows=None if scene_record is None else scene_record['plan']['compact_rows'])),flush=True)
                del raw,arrays,scene,handle
                handle = None
        if report['full_model_forward_count']!=8 or len(report['call_records'])!=8:
            raise ValueError('Fixed eight-call sequence was not completed')
        inputs_before = {}
        for row in (reference['calibration'],reference['rgb_index']):
            inputs_before[row['path']] = row['sha256']
        for selection in reference['selected_inputs']:
            inputs_before[selection['schedule_file']] = selection['schedule_sha256']
            inputs_before.update({im['path']:im['sha256'] for im in selection['images']})
        if ({name:sha(SOURCE/name) for name in sources}!=sources
                or {name:sha(args.official_source_dir/name) for name in official}!=official
                or any(sha(path)!=digest for path,digest in inputs_before.items())
                or sha(args.merge_module)!=args.merge_module_sha256
                or sha(args.wrapper_module)!=args.wrapper_module_sha256
                or sha(args.predictor_config)!=args.predictor_config_sha256
                or sha(args.predictor_checkpoint)!=PREDICTOR_SHA
                or sha(args.reference_receipt)!=REFERENCE_SHA or sha(stage_path)!=STAGE_SHA
                or sha(args.stage_capture_dir/'activations.npz')!=stage['artifacts']['activations.npz']['sha256']):
            raise ValueError('Sources/inputs changed during research')
        if (checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)!=(checkpoint_stat.st_size,checkpoint_stat.st_mtime_ns):
            raise ValueError('Original checkpoint stat changed during research')
        for pair in reference['pairs']:
            if sha(args.reference_receipt.parent/pair['file'])!=pair['sha256']:
                raise ValueError('Frozen original prediction reference changed')
        report.update(status='completed',sources_and_inputs_unchanged=True,all_original_calls_byte_exact=True,
            all_research_contexts_closed=True,original_forward_identities_and_hooks_restored=True,
            actual_research_predictor_calls=sum(row['predictor_calls'] for row in report['research_contexts']),
            actual_research_MLP_calls=sum(sum(row['mlp_calls'].values()) for row in report['research_contexts']),
            checkpoint_after_validation='size/mtime unchanged; initial full SHA verified',
            artifacts={path.name:dict(bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(args.output.iterdir()) if path.is_file()})
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if handle is not None:
            try:
                handle.close()
            except BaseException as exc:
                report['cleanup_errors'].append(f'research context: {type(exc).__name__}: {exc}')
        try:
            os.sched_setaffinity(0,old_affinity)
            report['cpu_affinity_restored'] = True
        except OSError as exc:
            report['cleanup_errors'].append(f'affinity: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report.update(total_seconds=time.monotonic()-started,
            peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (args.output/'fullmodel.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['status']=='completed' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-receipt',type=Path,required=True)
    p.add_argument('--reference-sha256',default=REFERENCE_SHA)
    p.add_argument('--stage-capture-dir',type=Path,required=True)
    p.add_argument('--stage-capture-sha256',default=STAGE_SHA)
    p.add_argument('--merge-module',type=Path,default=HERE/'v22_token_merge.py')
    p.add_argument('--merge-module-sha256',required=True)
    p.add_argument('--wrapper-module',type=Path,default=HERE/'v22_come_fullmodel.py')
    p.add_argument('--wrapper-module-sha256',required=True)
    p.add_argument('--official-source-dir',type=Path,required=True)
    p.add_argument('--predictor-checkpoint',type=Path,required=True)
    p.add_argument('--predictor-config',type=Path,required=True)
    p.add_argument('--predictor-config-sha256',required=True)
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
