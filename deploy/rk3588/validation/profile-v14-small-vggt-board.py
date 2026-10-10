#!/usr/bin/env python3
"""Board-only original VGGT small-image CPU/mixed diagnostics, never all-NPU proof."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CHECKPOINT_BYTES = 5026874952
COUNTERS = ('npu_calls', 'npu_run_attempts', 'npu_run_seconds', 'linear_seconds',
            'context_creates', 'context_evictions', 'weight_upload_bytes', 'npu_errors')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def event(name, **values):
    print(json.dumps(dict(event=name, monotonic=time.monotonic(), **values), allow_nan=False), flush=True)


def require_board():
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Actual forward execution requires Linux aarch64 RK3588; host model execution is forbidden')
    for path in (Path('/proc/device-tree/compatible'), Path('/sys/firmware/devicetree/base/compatible')):
        if path.is_file():
            value = path.read_bytes().replace(b'\0', b',').decode(errors='replace')
            if 'rk3588' not in value:
                raise RuntimeError(f'Unexpected board {value}')
            return value
    raise RuntimeError('RK3588 device-tree identity unavailable')


def platform_record():
    result = {'monotonic': time.monotonic(), 'temperatures_mC': {}, 'CPU_frequency_kHz': {}}
    for path in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        if (path/'type').is_file() and (path/'temp').is_file():
            result['temperatures_mC'][(path/'type').read_text().strip()] = int((path/'temp').read_text())
    for path in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
        if (path/'scaling_cur_freq').is_file():
            result['CPU_frequency_kHz'][path.name] = int((path/'scaling_cur_freq').read_text())
    for path in sorted(Path('/sys/class/devfreq').glob('*npu*')):
        if (path/'cur_freq').is_file():
            result['NPU_frequency_hz'] = int((path/'cur_freq').read_text())
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value, unit = line.split()
            result[key.rstrip(':')+'_MiB'] = int(value)/1024
    return result


def counters(handle):
    if handle is None:
        return {key: 0 for key in COUNTERS}
    metrics = handle.metrics()
    return {key: metrics[key] for key in COUNTERS}


def delta(after, before):
    return {key: after[key]-before[key] for key in COUNTERS}


class StageHooks:
    """Value-only inclusive wall timings; no tensor references are retained."""
    def __init__(self, model, handle=None):
        self.handle, self.rows, self.hooks, self.stacks = handle, [], [], defaultdict(list)
        modules = [('model', 'model', model), ('aggregator', 'aggregator', model.aggregator),
                   ('dino_encoder', 'aggregator.patch_embed', model.aggregator.patch_embed),
                   ('camera_head', 'camera_head', model.camera_head), ('depth_head', 'depth_head', model.depth_head)]
        modules += [('dino_blocks', f'dino.block.{i}', block)
                    for i, block in enumerate(model.aggregator.patch_embed.blocks)]
        modules += [('frame_blocks', f'frame.block.{i}', block)
                    for i, block in enumerate(model.aggregator.frame_blocks)]
        modules += [('global_blocks', f'global.block.{i}', block)
                    for i, block in enumerate(model.aggregator.global_blocks)]
        modules += [('camera_trunk', f'camera.trunk.{i}', block)
                    for i, block in enumerate(model.camera_head.trunk)]
        try:
            for group, name, module in modules:
                if module is None:
                    continue
                self.hooks.append(module.register_forward_pre_hook(self._before(group, name)))
                self.hooks.append(module.register_forward_hook(self._after(group, name)))
        except BaseException:
            self.close()
            raise

    def _before(self, group, name):
        def before(module, args):
            if group in {'model', 'dino_encoder', 'camera_head', 'depth_head'}:
                event('stage_begin', group=group)
            self.stacks[name].append((time.perf_counter(), time.monotonic(), counters(self.handle)))
        return before

    def _after(self, group, name):
        def after(module, args, output):
            end, end_monotonic = time.perf_counter(), time.monotonic()
            begin, begin_monotonic, before = self.stacks[name].pop()
            row = dict(group=group, name=name, inclusive_seconds=end-begin,
                       begin_monotonic=begin_monotonic, end_monotonic=end_monotonic,
                       npu_counter_delta=delta(counters(self.handle), before))
            self.rows.append(row)
            if group in {'model', 'dino_encoder', 'camera_head', 'depth_head'}:
                event('stage_end', group=group, inclusive_seconds=row['inclusive_seconds'])
        return after

    def summaries(self):
        sums = {}
        for row in self.rows:
            group = sums.setdefault(row['group'], dict(calls=0, inclusive_seconds=0.0,
                                                      npu_counter_delta={key:0 for key in COUNTERS}))
            group['calls'] += 1
            group['inclusive_seconds'] += row['inclusive_seconds']
            for key in COUNTERS:
                group['npu_counter_delta'][key] += row['npu_counter_delta'][key]
        return sums

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def prediction_values(predictions, torch):
    values = {}
    for key in ('pose_enc', 'depth', 'depth_conf'):
        value = predictions[key]
        if value.dtype != torch.float32 or value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
            raise ValueError(f'{key} must be finite CPU FP32')
        values[key] = value.detach().clone()
    return values


def compare(actual, reference):
    import numpy as np
    comparisons = {}
    for key, value in actual.items():
        a, r = value.numpy().astype(np.float64), reference[key].numpy().astype(np.float64)
        if a.shape != r.shape or not np.isfinite(a).all() or not np.isfinite(r).all():
            raise ValueError(f'Invalid prediction shape/finiteness: {key}')
        d = a-r
        norm = np.linalg.norm(r)
        comparisons[key] = dict(shape=list(a.shape), array_equal=bool(np.array_equal(a,r)),
            max_abs_error=float(np.max(np.abs(d))), rmse=float(np.sqrt(np.mean(d*d))),
            relative_l2=float(np.linalg.norm(d)/norm) if norm else None)
    return comparisons


def run_backend(model, images, torch, label, handle=None):
    if model.training or model.camera_head is None or model.depth_head is None:
        raise ValueError('This diagnostic requires eval mode and both original camera/depth heads')
    input_hash = hashlib.sha256(images.numpy().tobytes()).hexdigest()
    row = dict(backend=label, all_model_operators_on_NPU=False,
               CPU_model_inference=label=='original_CPU_FP32', mixed_CPU_NPU_model_inference=handle is not None,
               unhooked_calls=1, hooked_calls=1, unmeasured_warmup_calls=0,
               platform_before=platform_record(), input_tensor_raw_sha256=input_hash,
               same_input_tensor_for_unhooked_and_hooked=True,
               forward_options=dict(query_points=None, compute_similarity=False,
                   camera_head_enabled=True, depth_head_enabled=True,
                   depth_frames_chunk_size=model.depth_frames_chunk_size))
    event('unhooked_forward_begin', backend=label)
    before = counters(handle)
    with torch.inference_mode():
        start = time.perf_counter()
        predictions = model(images, query_points=None, compute_similarity=False)
        row['unhooked_forward_seconds'] = time.perf_counter()-start
    row['unhooked_npu_counter_delta'] = delta(counters(handle), before)
    reference = prediction_values(predictions, torch)
    del predictions
    row['prediction_shapes'] = {key:list(value.shape) for key,value in reference.items()}
    row['all_pose_depth_conf_finite'] = True
    row['pose_encoding'] = reference['pose_enc'].tolist()
    event('unhooked_forward_end', backend=label, seconds=row['unhooked_forward_seconds'])
    hooks = StageHooks(model, handle)
    try:
        before = counters(handle)
        with torch.inference_mode():
            row['hooked_forward_started_monotonic'] = time.monotonic()
            start = time.perf_counter()
            predictions = model(images, query_points=None, compute_similarity=False)
            row['hooked_forward_seconds_including_hooks_and_stage_logging'] = time.perf_counter()-start
            row['hooked_forward_completed_monotonic'] = time.monotonic()
        row['hooked_npu_counter_delta'] = delta(counters(handle), before)
        hooked = prediction_values(predictions, torch)
        del predictions
        row['hooked_vs_unhooked'] = compare(hooked, reference)
        if not all(value['array_equal'] for value in row['hooked_vs_unhooked'].values()):
            raise ValueError('Stage hooks changed original pose/depth/conf values')
        row['stages'], row['stage_summaries'] = hooks.rows, hooks.summaries()
        camera_end = next(value['end_monotonic'] for value in hooks.rows if value['group']=='camera_head')
        row['hooked_camera_head_completion_offset_seconds'] = camera_end-row['hooked_forward_started_monotonic']
        row['camera_prefix_scope'] = 'Measured internal completion offset in full hooked forward; not an independently benchmarked pose-only pipeline'
        row['incomplete_hook_stacks'] = {name:len(stack) for name,stack in hooks.stacks.items() if stack}
        if row['incomplete_hook_stacks']:
            raise RuntimeError('Incomplete stage hooks')
    finally:
        hooks.close()
        row['hooks_removed'] = not hooks.hooks
    row['platform_after'] = platform_record()
    row['input_tensor_raw_sha256_after'] = hashlib.sha256(images.numpy().tobytes()).hexdigest()
    row['input_unchanged'] = row['input_tensor_raw_sha256_after'] == input_hash
    if not row['input_unchanged']:
        raise ValueError('Input tensor changed during diagnostic')
    return row, reference


def stat_record(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--images', type=Path, nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-size', type=int, choices=(84,112), default=112)
    p.add_argument('--backend', choices=('cpu','both'), default='cpu', help='both runs original CPU then the existing mixed RKNN Linear path')
    p.add_argument('--threads', type=int, choices=(2,4), default=4)
    p.add_argument('--cpus', default='4,5,6,7')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output must be a new JSON file')
    cpus = {int(value) for value in args.cpus.split(',')}
    if len(cpus) != args.threads or not cpus <= {4,5,6,7}:
        p.error('Select exactly --threads big CPU cores from 4,5,6,7')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open('x')
    report = dict(kind='original_VGGT_small_input_CPU_and_optional_mixed_stage_diagnostic', status='failed',
        script_sha256=sha(__file__), arguments={key:str(value) if isinstance(value,Path) else
            [str(v) for v in value] if isinstance(value,list) else value for key,value in vars(args).items()},
        all_model_operators_on_NPU=False, realtime_1Hz_validated=False, accuracy_validated=False,
        source_production_modified=False, ORB_used=False, backends=[], cleanup_errors=[],
        adaptation='New small-input helper uses current runtime directly; old 518 profiler hardpins/timing are not reused',
        timing_semantics=['Unhooked forward excludes loading, preprocessing, prediction copies, summaries and hashes.',
            'Stage timings are inclusive/nested; aggregator and model totals must not be added to child groups.',
            'Hooked forward follows one reported unhooked forward; it includes hooks and stage logging.',
            'CPU measurements and the optional mixed Linear path are diagnostic baselines, not all-NPU success.',
            'One call per boundary does not establish statistical or temperature-controlled performance.'])
    started, allowed, handle, pins, checkpoint_stat = time.monotonic(), None, None, {}, None
    try:
        report['device_tree_compatible'] = require_board()
        allowed = os.sched_getaffinity(0)
        if not cpus <= allowed:
            raise RuntimeError(f'Requested CPU affinity not permitted: {sorted(allowed)}')
        os.sched_setaffinity(0, cpus)
        if os.sched_getaffinity(0) != cpus:
            raise RuntimeError('CPU affinity silently changed')
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = str(args.threads)
        report['cpu_affinity'], report['threads'] = sorted(cpus), args.threads
        checkpoint = args.checkpoint.resolve(strict=True)
        checkpoint_stat = stat_record(checkpoint)
        if checkpoint_stat['bytes'] != CHECKPOINT_BYTES:
            raise ValueError('Only the original official VGGT-1B checkpoint is accepted')
        event('checkpoint_hash_begin')
        checksum_started = time.perf_counter()
        digest = sha(checkpoint)
        if digest != CHECKPOINT_SHA:
            raise ValueError('Official VGGT checkpoint SHA mismatch')
        report['checkpoint'] = dict(path=str(checkpoint), sha256=digest, stat_before=checkpoint_stat,
            hash_before_verified=True, hash_seconds=time.perf_counter()-checksum_started,
            full_hash_rechecked_after=False)
        sources = [Path(__file__).resolve(), REPO/'vggt_slam/runtime.py']
        sources += sorted((REPO/'third_party/vggt/vggt').rglob('*.py'))
        if args.backend == 'both':
            sources += [REPO/'vggt_slam/rknn_linear.py', REPO/'deploy/rk3588/reference/rknn_matmul_api_v2.3.2.h', Path('/usr/lib/librknnrt.so')]
        images_paths = [path.resolve(strict=True) for path in args.images]
        pins = {str(path):sha(path) for path in sources+images_paths}
        report['source_and_image_hashes'] = pins.copy()
        report['input_images'] = [dict(path=str(path),sha256=pins[str(path)]) for path in images_paths]
        sys.path[:0] = [str(REPO),str(REPO/'third_party/vggt')]
        import torch
        from vggt_slam.runtime import load_model
        from vggt.utils.load_fn import load_and_preprocess_images
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        report['torch_version'] = torch.__version__
        report['platform_before_model_load'] = platform_record()
        start = time.perf_counter()
        model = load_model(checkpoint, device='cpu', image_size=args.image_size, prune_unused_heads=True, offline=True)
        report['model_load_seconds'] = time.perf_counter()-start
        report['unused_point_and_track_heads_pruned'] = True
        start = time.perf_counter()
        images = load_and_preprocess_images([str(path) for path in images_paths],image_size=args.image_size)
        report['preprocessing_seconds'] = time.perf_counter()-start
        report['input_shape'] = list(images.shape)
        report['input_dtype'] = str(images.dtype)
        report['input_tensor_raw_sha256'] = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        if images.shape[0] != 2 or images.dtype != torch.float32 or not bool(torch.isfinite(images).all()):
            raise ValueError('Expected two real finite FP32 images')
        report['tokens_per_image'] = (images.shape[-2]//14)*(images.shape[-1]//14)+5
        cpu_row, cpu_reference = run_backend(model,images,torch,'original_CPU_FP32')
        report['backends'].append(cpu_row)
        if args.backend == 'both':
            from vggt_slam.rknn_linear import install_rknn_linear
            event('mixed_RKNN_linear_install_begin')
            start = time.perf_counter()
            handle = install_rknn_linear(model,weight_layout='native',core_mask=1,max_contexts=8,
                max_cache_bytes=256*1024*1024,max_rows=1024,predicate=lambda name,module:name.startswith('aggregator.'))
            report['mixed_install_seconds'] = time.perf_counter()-start
            mixed_row, mixed_predictions = run_backend(model,images,torch,'mixed_RKNN_linear_CPU_rest',handle)
            mixed_row['versus_original_CPU'] = compare(mixed_predictions,cpu_reference)
            report['backends'].append(mixed_row)
            report['npu_before_cleanup'] = handle.metrics()
            if handle.metrics()['npu_calls'] <= 0 or handle.metrics()['npu_errors']:
                raise RuntimeError('Mixed diagnostic did not execute successful real NPU Linear calls')
        report['status'] = 'completed'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['error_traceback'] = traceback.format_exc()
    finally:
        if handle is not None:
            try:
                handle.close()
                report['npu_after_cleanup'] = handle.metrics()
            except BaseException as exc:
                report['cleanup_errors'].append(f'NPU cleanup: {type(exc).__name__}: {exc}')
        if pins:
            try:
                report['source_and_image_hashes_after'] = {path:sha(path) for path in pins}
                report['source_and_images_unchanged'] = report['source_and_image_hashes_after'] == pins
                if not report['source_and_images_unchanged']:
                    raise RuntimeError('Source or image changed during diagnostic')
            except BaseException as exc:
                report['cleanup_errors'].append(f'Hash recheck: {type(exc).__name__}: {exc}')
        if checkpoint_stat is not None:
            try:
                report['checkpoint_stat_unchanged_after'] = stat_record(args.checkpoint) == checkpoint_stat
                if not report['checkpoint_stat_unchanged_after']:
                    raise RuntimeError('Checkpoint stat changed during diagnostic')
            except BaseException as exc:
                report['cleanup_errors'].append(f'Checkpoint stat recheck: {type(exc).__name__}: {exc}')
        if allowed is not None:
            try:
                os.sched_setaffinity(0,allowed)
                report['cpu_affinity_restored'] = os.sched_getaffinity(0) == allowed
                if not report['cpu_affinity_restored']:
                    raise RuntimeError('CPU affinity restore mismatch')
            except BaseException as exc:
                report['cleanup_errors'].append(f'Affinity restore: {type(exc).__name__}: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['total_seconds_including_hash_load_preprocess_checks'] = time.monotonic()-started
        try:
            json.dump(report,stream,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        finally:
            stream.close()
    event('finished',status=report['status'],output=str(args.output),error=report.get('error'))
    return 0 if report['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
