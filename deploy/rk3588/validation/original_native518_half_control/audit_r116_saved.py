#!/usr/bin/env python3
"""Saved NumPy/JSON audit only; never import Torch, ORT, SDK or device tools."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SOURCE_SHA = '9be7e141e7246b684d69d9d0ff4dba865ba6bf6904be391bf5068c1d869d4a56'
RESULT_SHA = '7144d5189ed4703e6be7649fcde09d159c753c87d023adf42a745315f2394355'
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
NPU_SHA = '1f89eac69b85ec84ea76f9e8ca0a50ca7d4c5f19620c4159f41eaf6f3661225a'
POINTS = ('norm1', 'attn.qkv', 'attention', 'attn.proj.input', 'output')
ROUTES = ('original_FP32', 'storage_rounding_only_FP32', 'CPU_float16')
SHAPE = (1, 1041, 1024)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def raw(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def compare(value, reference):
    a, b = value.astype(np.float64), reference.astype(np.float64)
    delta = a - b
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    bad = np.abs(delta) > 1e-5 + 1e-4 * np.abs(b)
    return dict(relative_l2=float(np.linalg.norm(delta)) / bn if bn else None,
        reference_l2=bn, max_abs=float(np.abs(delta).max()),
        cosine=float(np.sum(a * b) / (an * bn)) if an and bn else None,
        strict_rtol=1e-4, strict_atol=1e-5, strict_pass=not bool(bad.any()),
        outside_count=int(bad.sum()), byte_equal=value.dtype == reference.dtype and value.tobytes() == reference.tobytes())


def same_metrics(actual, recorded):
    require(set(actual) == set(recorded), 'Metric fields differ')
    for name, value in actual.items():
        old = recorded[name]
        if isinstance(value, float):
            require(isinstance(old, (int, float)) and math.isclose(value, old, rel_tol=1e-11, abs_tol=1e-14), 'Metric differs: ' + name)
        else:
            require(value == old, 'Exact metric differs: ' + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args(); root = a.root.resolve(strict=True); watched = {}
    def pin(path, digest=None, size=None):
        path = path.resolve(strict=True); found = sha(path)
        require((digest is None or found == digest) and (size is None or path.stat().st_size == size), 'File identity differs: ' + str(path))
        watched[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=found)
        return path
    result = json.loads(pin(root / 'half-control-r1/result.json', RESULT_SHA).read_text())
    capture = json.loads(pin(root / 'reference-r100/capture.json', CAPTURE_SHA).read_text())
    tap = json.loads(pin(root / 'reference-r100/npu-result.json', NPU_SHA).read_text())
    pin(root / 'run_dino0_half_control_board.py', SOURCE_SHA)
    require(result['status'] == 'completed_original_DINO0_CPU_precision_controls_not_quality_accepted' and
        result['script_sha256'] == SOURCE_SHA and result['capture_sha256'] == CAPTURE_SHA and result['tap_result_sha256'] == NPU_SHA and
        result['original_FP32_control_all_points_byte_equal'] and not result['cleanup_errors'], 'Completed pinned control required')
    for flag in ('hooks_removed', 'original_SDPA_function_restored', 'source_and_inputs_unchanged',
        'all_instantiated_parameter_states_unchanged', 'affinity_restored', 'environment_restored', 'sys_path_restored'):
        require(result[flag] is True, 'Reported cleanup/state flag failed: ' + flag)
    require(result['FP16_failure_fallback_enabled'] is False and result['SDK_ORT_NPU_GPU_executed'] is False and
        result['original_parameters'] == capture['original_parameters'] and len(result['original_parameters']) == 14 and
        result['checkpoint'] == capture['checkpoint'] and tap['final_baseline_byte_equal_all_calls'] and
        tap['resources_balanced'] and not tap['cleanup_errors'], 'Original-model/no-fallback/NPU-baseline provenance differs')
    def load(folder, metadata, dtype, shape):
        path = Path(metadata['file']); require(path.name == str(path), 'Direct saved filename required')
        value = np.load(pin(folder / path, metadata['sha256'], metadata['bytes']), allow_pickle=False)
        require(value.dtype == dtype and value.shape == shape and metadata['shape'] == list(shape) and
            metadata['dtype'] == str(dtype) and value.nbytes == metadata['tensor_bytes'] and metadata['finite'] and
            value.flags.c_contiguous and np.isfinite(value).all() and raw(value) == metadata['raw_sha256'], 'Array identity differs: ' + str(path))
        return value
    refs = {name: load(root / 'reference-r100', capture['artifacts'][name], np.dtype('float32'),
        (1, 1041, 3072) if name == 'attn.qkv' else SHAPE) for name in ('input', 'norm1', 'attn.qkv', 'attn.proj.input', 'output')}
    x = refs.pop('input')
    refs['attention'] = np.ascontiguousarray(refs['attn.proj.input'].reshape(1, 1041, 16, 64).transpose(0, 2, 1, 3))
    payload = tap['calls'][0]['artifact']; pin(root / 'reference-r100/npu-call0.npz', payload['sha256'], payload['bytes'])
    with np.load(root / 'reference-r100/npu-call0.npz', allow_pickle=False) as archive:
        require(set(archive.files) == set(payload['arrays']) == {'output', 'norm1', 'attn.qkv'}, 'Complete NPU payload required')
        npu = {name: np.array(archive[name], copy=True, order='C') for name in archive.files}
    for name, value in npu.items():
        meta = payload['arrays'][name]
        require(value.dtype == np.float32 and list(value.shape) == meta['shape'] and meta['dtype'] == 'float32' and
            np.isfinite(value).all() and raw(value) == meta['raw_sha256'], 'NPU tensor identity differs')
    require(result['actual_NPU_reference']['payload'] == payload, 'Actual NPU reference alias differs')
    checks, controls, saved = 0, {}, 0
    require([row['route'] for row in result['calls']] == list(ROUTES), 'Three distinct controls required')
    for call in result['calls']:
        route = call['route']; dtype = np.dtype('float16' if route == 'CPU_float16' else 'float32')
        expected_input = x if route == 'original_FP32' else x.astype(np.float16).astype(dtype)
        require(call['input_raw_sha256'] == raw(expected_input) and call['forward_attempts'] == 1 and call['forward_completed'] and
            call['model_parameter_state_unchanged'] and call['input_unchanged'] and set(call['observations']) == set(POINTS) and
            call['tensor_dtype'] == 'torch.' + str(dtype), 'Declared input/dtype/full forward differs')
        require(set(call['parameter_metadata']) == set(result['original_parameters']) and all(
            meta['dtype'] == call['tensor_dtype'] and meta['shape'] == result['original_parameters'][name]['shape']
            for name, meta in call['parameter_metadata'].items()), 'Complete declared14 parameter dtype differs')
        sdpa = call['SDPA']; require(sdpa['tensor_dtype'] == call['tensor_dtype'] and sdpa['QKV_shapes'] == [[1,16,1041,64]] * 3 and
            sdpa['kwargs'] == {'dropout_p': 0.0} and sdpa['mask'] is None and sdpa['is_causal'] is False and
            sdpa['default_scale'] == .125 and sdpa['arguments_forwarded_unchanged'] and sdpa['original_return_preserved'], 'Original SDPA call differs')
        arrays, point_metrics = {}, {}
        for name in POINTS:
            meta = call['observations'][name]
            if route == 'original_FP32':
                value = refs[name]
                require(meta['saved'] is False and meta['raw_sha256'] == raw(value) and meta['tensor_bytes'] == value.nbytes and
                    meta['dtype'] == 'float32' and meta['shape'] == list(value.shape), 'Observed FP32 byte-gate metadata differs')
            else:
                require(meta['saved'] is True, 'Actual rounded/Half array must be saved')
                value = load(root / 'half-control-r1', meta, dtype, refs[name].shape); saved += meta['bytes']
            metrics = compare(value, refs[name]); same_metrics(metrics, call['comparisons'][name]); checks += 1
            arrays[name] = value; point_metrics[name] = {k: metrics[k] for k in ('relative_l2', 'max_abs', 'strict_pass', 'outside_count', 'byte_equal')}
        require(np.array_equal(arrays['attention'].transpose(0,2,1,3).reshape(SHAPE), arrays['attn.proj.input']), 'Actual attention consumer layout changed')
        y = arrays['output']; update = y.astype(np.float64) - x
        extra = dict(output_vs_actual_NPU=compare(y, npu['output']),
            residual_update_same_original_input_vs_CPU=compare(update, refs['output'].astype(np.float64) - x),
            residual_update_same_original_input_vs_actual_NPU=compare(update, npu['output'].astype(np.float64) - x))
        for name, value in extra.items():
            same_metrics(value, call[name]); checks += 1
        controls[route] = dict(points=point_metrics, residual_update_vs_CPU=extra['residual_update_same_original_input_vs_CPU'])
    npu_metrics = {name: compare(value, refs[name]) for name, value in npu.items()}
    same_metrics(npu_metrics['output'], result['actual_NPU_vs_original_CPU']); checks += 1
    npu_update = compare(npu['output'].astype(np.float64)-x, refs['output'].astype(np.float64)-x)
    same_metrics(npu_update, result['actual_NPU_residual_update_vs_original_CPU']); checks += 1
    control = root / 'cpu-control-r1'
    for file in control.iterdir():
        if file.is_file(): pin(file)
    require((control/'child-returncode.txt').read_text().strip() == '0' and
        (control/'npu-hz-before.txt').read_text().strip() == (control/'npu-hz-after.txt').read_text().strip() == '300000000', 'CPU exit/baseline NPU frequency differs')
    require(saved == result['saved_array_bytes'] == 44772608 and saved < result['payload_byte_limit'] and
        result['completed_forwards_by_control'] == result['forward_attempts_by_control'] == dict.fromkeys(ROUTES, 1), 'Payload/call counts differ')
    require(all(sha(root/name) == meta['sha256'] for name, meta in watched.items()), 'Saved source or evidence changed during audit')
    output = dict(schema='v116_independent_saved_original_DINO0_precision_audit', status='passed', auditor_sha256=sha(__file__),
        execution='Saved NumPy/JSON only; no Torch, ORT, SDK, GPU or board calls', comparison_records_recomputed=checks,
        checked_control_NPY=10, checked_original_reference_NPY=5, checked_NPU_NPZ=1, checked_NPU_tensors=3,
        saved_control_bytes=saved, original_FP32_observed_metadata_matches_all_pinned_reference_bytes=True,
        actual_reference_attention_layout_and_both_control_consumer_layouts_byte_equal=True,
        controls=controls, actual_NPU_points_vs_original=npu_metrics, actual_NPU_update_vs_original=npu_update,
        reported_hooks_state_sources_and_process_cleanup_passed=True, child_returncode=0, NPU_before_after_hz=300000000,
        board_wall_clock_not_validated_as_real_UTC=True, temperature_sampled=False,
        observed_worker_monotonic_seconds=result['elapsed_seconds'], observed_peak_process_RSS_MiB=result['peak_rss_mib'],
        metric_receipt_recompute_tolerance=dict(relative=1e-11, absolute=1e-14, fixed_tensor_thresholds_unchanged=True),
        limits=['Original FP32 forward arrays were not resaved; its recorded observed raw hashes match the pinned references.',
            'Half parameter payloads were not independently resaved; assigned-value checks/state are enforced by pinned executed source.',
            'One original DINO0 input only; tensor Half dtype does not establish internal accumulation precision or a NPU root cause.',
            'No deployment speedup, trajectory, whole-model, realtime, thermal or8GB acceptance.'], files=watched)
    (root/'saved-array-audit.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed', checks=checks, saved_bytes=saved)), flush=True)


if __name__ == '__main__':
    main()
