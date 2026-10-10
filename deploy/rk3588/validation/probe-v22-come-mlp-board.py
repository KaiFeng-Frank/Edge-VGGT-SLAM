#!/usr/bin/env python3
"""Board-only fixed-scene original-MLP merge/split approximation experiment.

The original global0 norm2 input, residual, weights and outputs come from a
byte-bound original FP32 capture. One released BF16 CPU predictor fixes one
scene mask. Two ABBA cycles make four original/four merged measurements after
one warmup each. Timings include merge/MLP/split/LayerScale/full residual, but
exclude predictor, scene planning, metrics and file I/O (reported separately).
No full compressed model, NPU, GT, ORB, map or realtime claim is made.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
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
CAPTURE_SCRIPT_SHA = '59c1b346bdda45fe1130eab0f869f47732755cc95761c16c43d3c0346805bc14'
PREDICTOR_SHA = '95b0a004544c96da5c4e55748e8cf4837608627a6414563af06cb3ee25acd29f'
MEASURED_ORDER = ('original','merged','merged','original','original','merged','merged','original')
RTOL,ATOL = 1e-4,1e-5


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
        raise ValueError('Every saved numerical value must be finite and nonempty')
    return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def error_stats(actual,reference):
    import numpy as np
    a,b = np.asarray(actual),np.asarray(reference)
    if a.shape!=b.shape or a.dtype!=np.float32 or b.dtype!=np.float32 or not a.size:
        raise ValueError('Error comparison requires matching nonempty FP32 arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Error comparison requires finite arrays')
    d = a.astype(np.float64)-b.astype(np.float64)
    norm = float(np.linalg.norm(b.astype(np.float64)))
    outside = np.abs(d)>(ATOL+RTOL*np.abs(b.astype(np.float64)))
    return dict(elements=int(a.size),max_abs=float(np.max(np.abs(d))),
        rmse=float(np.sqrt(np.mean(d*d))),relative_l2=float(np.linalg.norm(d)/norm) if norm else None,
        bitwise_equal=a.tobytes(order='C')==b.tobytes(order='C'),
        strict_FP32_allclose=bool(not outside.any()),outside_strict_FP32=int(outside.sum()),
        predefined_rtol=RTOL,predefined_atol=ATOL)


def bf16_bits(value):
    """Preserve BF16 bit patterns without relying on NumPy BF16 support."""
    import numpy as np
    import torch
    if value.device.type!='cpu' or value.dtype!=torch.bfloat16:
        raise ValueError('BF16 bit-pattern artifact requires original CPU BF16 tensor')
    return value.detach().contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def validate_capture(row):
    if (row.get('kind')!='original_VGGT_CoMe_MLP_stage_capture'
            or row.get('status')!='captured_original_FP32_stages_bitwise_bound'
            or row.get('script_sha256')!=CAPTURE_SCRIPT_SHA
            or row.get('original_weights') is not True or row.get('architecture_changed') is not False
            or row.get('ORB_used') is not False or row.get('npu_used') is not False
            or row.get('hooks_removed') is not True or row.get('cleanup_errors')
            or row.get('full_model_forward_count')!=1):
        raise ValueError('Expected successful frozen original FP32 stage capture')
    architecture = row['architecture']
    fields = dict(views=2,grid_hw=[18,24],special_tokens=5,channels=1024,
        gelu_approximate='none',dropout=0.,dino_extra_norm=False,
        dino_hook='aggregator.patch_embed.blocks[14] forward output')
    if any(architecture.get(key)!=value for key,value in fields.items()):
        raise ValueError('Original stage/MLP architecture differs')
    if (set(row['global0_vs_profile'])!={'input','output','pos'}
            or set(row['full_predictions_vs_original'])!={'depth','depth_conf','intrinsic','extrinsic','pose_enc'}
            or not all(value is True for value in row['global0_vs_profile'].values())
            or not all(value is True for value in row['full_predictions_vs_original'].values())
            or row.get('sources_and_inputs_unchanged') is not True):
        raise ValueError('Capture is not byte-bound to original reference')


def validate_plan(plan,mask):
    """Independent index checks, including view boundaries and all specials."""
    import numpy as np
    if mask.dtype!=np.bool_ or mask.shape!=(2,6,8):
        raise ValueError('Fixed scene mask must be bool [2,6,8]')
    if plan.views!=2 or plan.tokens_per_view!=437 or plan.compact_rows!=874-8*int(mask.sum()):
        raise ValueError('Returned actual compact count differs from fixed scene')
    reverse = np.asarray(plan.reverse_indices)
    offsets = np.asarray(plan.view_offsets)
    if reverse.shape!=(2,437) or offsets.shape!=(3,) or offsets[0]!=0 or offsets[-1]!=plan.compact_rows:
        raise ValueError('Malformed plan reverse mapping or view boundaries')
    for view in range(2):
        if (not np.array_equal(reverse[view,:5],np.arange(offsets[view],offsets[view]+5))
                or np.any(reverse[view]<offsets[view]) or np.any(reverse[view]>=offsets[view+1])):
            raise ValueError('Plan merged specials or crossed a view boundary')
        patch = reverse[view,5:].reshape(18,24)
        for y in range(6):
            for x in range(8):
                region = patch[y*3:y*3+3,x*3:x*3+3]
                if np.unique(region).size!=(1 if mask[view,y,x] else 9):
                    raise ValueError('Plan does not match exact 3x3 spatial grouping')
    return dict(actual_compact_rows=int(plan.compact_rows),actual_merged_groups=int(mask.sum()),
                special_rows_preserved=10,views_not_crossed=True)


def region_errors(actual,reference,mask):
    """All 9x1024 values of each patch region; no favourable point selection."""
    import numpy as np
    a,b = np.asarray(actual).reshape(2,437,1024),np.asarray(reference).reshape(2,437,1024)
    patch_a,patch_b = a[:,5:].reshape(2,18,24,1024),b[:,5:].reshape(2,18,24,1024)
    pixel_mask = np.repeat(np.repeat(mask,3,axis=1),3,axis=2)
    result = dict(all_tokens=error_stats(a,b),all_patches=error_stats(patch_a,patch_b),
                 specials=error_stats(a[:,:5],b[:,:5]),merged_patches=None,unmerged_patches=None,
                 regions=[])
    if pixel_mask.any():
        result['merged_patches'] = error_stats(patch_a[pixel_mask],patch_b[pixel_mask])
    if (~pixel_mask).any():
        result['unmerged_patches'] = error_stats(patch_a[~pixel_mask],patch_b[~pixel_mask])
    for view in range(2):
        for y in range(6):
            for x in range(8):
                result['regions'].append(dict(view=view,region_y=y,region_x=x,merged=bool(mask[view,y,x]),
                    **error_stats(patch_a[view,y*3:y*3+3,x*3:x*3+3],patch_b[view,y*3:y*3+3,x*3:x*3+3])))
    return result


def board_guard(machine,output):
    if machine.lower() not in ('aarch64','arm64'):
        raise ValueError('Actual probe is aarch64 board-only')
    if not Path(output).resolve().is_relative_to(BASE/'experiments'):
        raise ValueError('Output must be a new board BASE/experiments directory')
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError('Output already exists; original experiments are never overwritten')


def timed_calls(original,merged,on_result,clock=time.perf_counter):
    rows = []
    for index,name in enumerate(MEASURED_ORDER):
        begin = clock()
        values = (original if name=='original' else merged)()
        end = clock()
        rows.append(dict(index=index,mode=name,end_to_end_seconds=end-begin,
                         **on_result(index,name,values)))
    return rows


def run(args):
    report = dict(kind='isolated_original_MLP_CoMe_merge_split_probe',status='started',
        script_sha256=sha(__file__),capture_sha256=args.capture_sha256,
        merge_module_sha256=args.merge_module_sha256,original_VGGT_MLP_weights=True,
        CoMe_confidence_predictor_separate=True,predictor_precision='BF16_CPU',
        original_MLP_precision='FP32_CPU',GT_used=False,ORB_used=False,npu_used=False,
        host_inference_executed=False,full_CoMe_model_validated=False,
        quality_validated=False,realtime_validated=False,predictor_CUDA_bitwise_equivalence_claim=False,
        measured_order=list(MEASURED_ORDER),warmup_counts={'original':0,'merged':0},
        measured_counts={'original':0,'merged':0},cleanup_errors=[],
        actual_MLP_calls={'original':0,'merged':0},
        timing_scope='captured norm2 -> merge -> original MLP -> split -> LayerScale -> original full residual; no norms/attention/other blocks',
        predictor_and_scene_planning_excluded_from_round_timings=True,
        masks_frozen_once_for_this_scene_only=True,MLP_strict_rtol=RTOL,MLP_strict_atol=ATOL)
    started = time.monotonic()
    old_affinity = os.sched_getaffinity(0)
    try:
        cpus = {int(v) for v in args.cpus.split(',')}
        if not cpus or not cpus<=old_affinity or args.threads<1:
            raise ValueError('Requested CPU/threads invalid or outside allowed affinity')
        os.sched_setaffinity(0,cpus)
        os.environ.update(OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
            OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',MPLBACKEND='Agg')
        sys.path[:0] = [str(SOURCE),str(SOURCE/'third_party/vggt')]
        import numpy as np
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        capture_path = args.capture_dir/'capture.json'
        if sha(capture_path)!=args.capture_sha256:
            raise ValueError('Frozen original capture receipt SHA mismatch')
        capture = json.loads(capture_path.read_text())
        validate_capture(capture)
        if sha(args.merge_module)!=args.merge_module_sha256:
            raise ValueError('Explicit research merger implementation SHA mismatch')
        spec = importlib.util.spec_from_file_location('_v22_frozen_research_merge',args.merge_module)
        merger = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = merger
        spec.loader.exec_module(merger)
        if merger.RELEASED_CHECKPOINT_SHA256!=PREDICTOR_SHA:
            raise ValueError('Research merger does not declare the exact released predictor')
        official_sources = {name:sha(args.official_source_dir/name) for name in merger.SOURCE_PINS}
        if official_sources!=merger.SOURCE_PINS:
            raise ValueError('Pinned official CoMe source changed')
        sources = {name:sha(SOURCE/name) for name in capture['source_hashes']}
        if sources!=capture['source_hashes']:
            raise ValueError('Original model/preprocessing source changed')
        if sha(HERE/'capture-v22-come-mlp-board.py')!=CAPTURE_SCRIPT_SHA:
            raise ValueError('Frozen original stage capture implementation changed')
        report['source_hashes'] = sources
        report['CoMe_source'] = dict(commit=merger.SOURCE_COMMIT,files=official_sources,
            license='CC BY-NC 4.0 research adaptation; not the repository BSD license')
        report['runtime'] = dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
            cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads())
        for name,info in capture['artifacts'].items():
            if sha(args.capture_dir/name)!=info['sha256']:
                raise ValueError(f'Original capture artifact SHA mismatch: {name}')
        with np.load(args.capture_dir/'activations.npz',allow_pickle=False) as saved:
            arrays = {key:saved[key].copy() for key in saved.files}
        if {key:array_info(value) for key,value in arrays.items()}!=capture['arrays']:
            raise ValueError('Original activation array metadata/raw SHA mismatch')
        with np.load(args.capture_dir/'mlp-state.npz',allow_pickle=False) as saved:
            state = {key:saved[key].copy() for key in saved.files}
        if set(state)!=set(capture['mlp_state_keys']) or any(
                array_info(value)!=capture['global0_state'][key] for key,value in state.items()):
            raise ValueError('Original captured MLP weight bytes differ')
        if args.predictor_checkpoint.stat().st_size!=831722 or sha(args.predictor_checkpoint)!=PREDICTOR_SHA:
            raise ValueError('Released predictor checkpoint bytes/SHA differ')
        if sha(args.predictor_config)!=args.predictor_config_sha256:
            raise ValueError('Released predictor config SHA mismatch')
        begin = time.perf_counter()
        predictor = merger.load_released_predictor(args.predictor_checkpoint,args.predictor_config,
            expected_checkpoint_sha256=PREDICTOR_SHA,expected_config_sha256=args.predictor_config_sha256)
        report['predictor_load_seconds'] = time.perf_counter()-begin
        begin = time.perf_counter()
        features = torch.from_numpy(arrays['predictor_features']).permute(0,1,3,4,2).to(torch.bfloat16)
        report['predictor_input_BF16_conversion_seconds'] = time.perf_counter()-begin
        begin = time.perf_counter()
        with torch.inference_mode():
            confidence = predictor(features,BPHWC=(1,2,18,24,1024))
        report['predictor_forward_seconds'] = time.perf_counter()-begin
        if confidence.shape!=(2,432,1) or confidence.dtype!=torch.bfloat16 or not torch.isfinite(confidence).all():
            raise ValueError('Released original CPU BF16 confidence output invalid; no fallback')
        begin = time.perf_counter()
        mask = merger.merge_mask_from_confidence(confidence,threshold=.5)
        conf_groups = confidence.reshape(2,6,3,8,3).permute(0,1,3,2,4).reshape(2,48,9)
        means = conf_groups.mean(dim=-1)
        median = torch.quantile(means.float().reshape(1,-1),.5,dim=-1,keepdim=True)
        independent_mask = (means.reshape(1,-1)<median).reshape(2,6,8).numpy()
        if not np.array_equal(mask,independent_mask):
            raise ValueError('Merger mask differs from original BF16-group/FP32-median rule')
        plan = merger.build_plan(mask)
        report['plan_guard'] = validate_plan(plan,mask)
        report['mask_and_plan_seconds'] = time.perf_counter()-begin
        report['plan'] = plan.summary()
        x = torch.from_numpy(arrays['norm2'])
        residual = torch.from_numpy(arrays['residual1'])
        gamma = torch.from_numpy(state['ls2.gamma'])
        from vggt.layers.mlp import Mlp
        mlp = Mlp(1024,hidden_features=4096,out_features=1024,drop=0.,bias=True)
        mlp.load_state_dict({key.removeprefix('mlp.'):torch.from_numpy(value)
            for key,value in state.items() if key.startswith('mlp.')},strict=True)
        mlp.eval().requires_grad_(False)
        no_merge = merger.build_plan(np.zeros((2,6,8),np.bool_))
        if not torch.equal(merger.restore_tokens(merger.merge_tokens(x,no_merge),no_merge,layout='global'),x):
            raise ValueError('All-false merge/split failed to restore original token ordering')
        report['all_false_index_roundtrip_exact'] = True
        report['predictor'] = dict(checkpoint_sha256=PREDICTOR_SHA,
            config_sha256=args.predictor_config_sha256,logical_precision='BF16_CPU',
            confidence_for_artifact_expanded_to_FP32=True,quantile=.5,strict_less_than=True,
            groups=96,merged_groups=int(mask.sum()),tie_groups=int((means.float()==median).sum()),
            special_tokens_excluded=True,no_final_DINO_norm=True,
            raw_BF16_artifact_encoding='uint16 bit patterns, native little endian',
            group_mean_precision_before_quantile='BF16',quantile_precision='FP32')
        np.savez(args.output/'scene-plan.npz',confidence=confidence.float().numpy(),
            confidence_bf16_bits=bf16_bits(confidence),
            group_means_bf16_bits=bf16_bits(means.reshape(2,6,8)),
            group_means=means.float().reshape(2,6,8).numpy(),median_threshold=median.numpy(),merge_mask=mask,
            reverse_indices=plan.reverse_indices,view_offsets=plan.view_offsets,
            token_weights=plan.token_weights,merged_patch_mask=plan.merged_patch_mask,
            representative_indices=plan.representative_indices,
            mean_input_indices=plan.mean_input_indices,mean_output_indices=plan.mean_output_indices)

        def original_call():
            begin = time.perf_counter()
            out = mlp(x)
            report['actual_MLP_calls']['original'] += 1
            mlp_done = time.perf_counter()
            scaled = out*gamma
            full = residual+scaled
            end = time.perf_counter()
            return dict(mlp_output=out,post_ls2=scaled,block_output=full,
                phase_seconds=dict(mlp=mlp_done-begin,ls2_and_full_residual=end-mlp_done))

        def merged_call():
            begin = time.perf_counter()
            compact = merger.merge_tokens(x,plan)
            packed = time.perf_counter()
            compact_output = mlp(compact)
            report['actual_MLP_calls']['merged'] += 1
            mlp_done = time.perf_counter()
            expanded = merger.restore_tokens(compact_output,plan,layout='global')
            restored = time.perf_counter()
            scaled = expanded*gamma
            full = residual+scaled
            end = time.perf_counter()
            return dict(compact_input=compact,compact_output=compact_output,mlp_output=expanded,
                post_ls2=scaled,block_output=full,phase_seconds=dict(merge=packed-begin,
                    mlp=mlp_done-packed,split=restored-mlp_done,ls2_and_full_residual=end-restored))

        last = {}
        def observe(index,name,values):
            outputs = {key:np.ascontiguousarray(value.numpy()) for key,value in values.items() if key!='phase_seconds'}
            if name=='original' and any(not error_stats(outputs[key],arrays[key])['bitwise_equal']
                                       for key in ('mlp_output','post_ls2','block_output')):
                raise ValueError('Same-shape original MLP/LS2/residual failed captured byte-exact gate')
            record = {key:array_info(value) for key,value in outputs.items()}
            previous = last.get(name)
            if previous is not None and any(record[key]!=array_info(previous[key]) for key in outputs):
                raise ValueError('Repeated same-input MLP output is nondeterministic')
            last[name] = outputs
            if index>=0:
                report['measured_counts'][name] += 1
                np.save(args.output/f'round-{index:02d}-{name}-block-output.npy',outputs['block_output'],allow_pickle=False)
            return dict(arrays=record,phase_seconds=values['phase_seconds'])

        with torch.inference_mode():
            for name,call in (('original',original_call),('merged',merged_call)):
                begin = time.perf_counter()
                values = call()
                elapsed = time.perf_counter()-begin
                observe(-1,name,values)
                report['warmup_counts'][name] += 1
                report.setdefault('warmup_seconds',{})[name] = elapsed
            report['rounds'] = timed_calls(original_call,merged_call,observe)
        if report['actual_MLP_calls']!={name:report['warmup_counts'][name]+report['measured_counts'][name]
                                      for name in ('original','merged')}:
            raise ValueError('Measured/warmup records do not cover all executed MLP calls')
        np.savez(args.output/'outputs.npz',**{name+'_'+key:value for name,rows in last.items() for key,value in rows.items()})
        report['errors'] = {key:region_errors(last['merged'][key],arrays[key],mask)
                            for key in ('mlp_output','post_ls2','block_output')}
        for name in ('original','merged'):
            values = [row['end_to_end_seconds'] for row in report['rounds'] if row['mode']==name]
            report.setdefault('timing_summary',{})[name] = dict(measured_calls=len(values),
                mean_seconds=float(np.mean(values)),median_seconds=float(np.median(values)),
                min_seconds=min(values),max_seconds=max(values),warmup_excluded=True)
        report['timing_summary']['speedup_original_over_merged'] = (
            report['timing_summary']['original']['mean_seconds']/report['timing_summary']['merged']['mean_seconds'])
        report['first_scene_prediction_plus_plan_plus_first_measured_merged_seconds'] = (
            report['predictor_input_BF16_conversion_seconds']+
            report['predictor_forward_seconds']+report['mask_and_plan_seconds']+
            next(row['end_to_end_seconds'] for row in report['rounds'] if row['mode']=='merged'))
        report['first_scene_timing_scope'] = 'BF16 input conversion + predictor forward + fixed mask/plan + first measured merge/MLP/split/LS2/residual; excludes cold loads, warmups, hash/metrics/file I/O'
        report['artifacts'] = {path.name:dict(bytes=path.stat().st_size,sha256=sha(path))
                              for path in sorted(args.output.iterdir()) if path.is_file()}
        if ({name:sha(SOURCE/name) for name in sources}!=sources
                or {name:sha(args.official_source_dir/name) for name in official_sources}!=official_sources
                or sha(args.merge_module)!=args.merge_module_sha256
                or sha(capture_path)!=args.capture_sha256
                or sha(args.predictor_checkpoint)!=PREDICTOR_SHA
                or sha(args.predictor_config)!=args.predictor_config_sha256):
            raise ValueError('Sources/inputs changed during isolated probe')
        for name,info in capture['artifacts'].items():
            if sha(args.capture_dir/name)!=info['sha256']:
                raise ValueError('Original capture artifact changed during probe')
        report.update(status='completed_isolated_MLP_approximation_probe',sources_and_inputs_unchanged=True,
            kept_and_special_strict_FP32_gates_reported_not_relaxed=True,
            merged_regions_not_required_to_equal_original=True,
            original_scene_anchor_ids=capture['selected_pair']['frame_ids'])
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        try:
            os.sched_setaffinity(0,old_affinity)
            report['cpu_affinity_restored'] = True
        except OSError as exc:
            report['cleanup_errors'].append(f'affinity: {exc}')
            report['status'] = 'failed'
        report.update(total_seconds=time.monotonic()-started,
            peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (args.output/'probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['status']=='completed_isolated_MLP_approximation_probe' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir',type=Path,required=True)
    p.add_argument('--capture-sha256',required=True)
    p.add_argument('--merge-module',type=Path,default=HERE/'v22_token_merge.py')
    p.add_argument('--merge-module-sha256',required=True)
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
