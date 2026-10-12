#!/usr/bin/env python3
"""Recompute the pinned v75 folded saved-array checks using NumPy only."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

PINS = {
    'compile-folded-r1/result.json': '083f279af1d98bfa9857ad053842e46ad28cd8667cd4fdf49851e783f5837fef',
    'compile-folded-r1/check.json': 'da50dc65e6104944455e63fd0536ed0ff6dbc481c507215077ac36fc0362dc27',
    'normal-folded-r1/result.json': '9ad66d1b10f4c84401c16faebce19bd3983215398fe3bd469ee2903cd900a6f6',
    'profile-folded-r1/result.json': '2483ef8b5adbdd806e6bebe1274e58fa7f1c303153c121708fe763a6dc03f4b4',
}
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
BUILDER_SHA = '1e3b1a7b4aaf8b240865ead4eccaa2255f2e31b709245ad7dfc521d2346fa7e6'
REWRITE_SHA = '2bc71dcb1b7c277681cb6121250d7255cb18c499ab7d2bd5e6c7a7a35c8608e3'
PROBE_SHA = 'd0a1d18ec1a184383c1f6a7d494dec33f7d3184f5984a40f764835d93b9e775e'
SHAPES = ((2, 53, 1024), (1, 106, 1024))
NAMES = ('frame_output', 'global_output')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compare(value, reference):
    require(value.shape == reference.shape, 'Comparison shape differs')
    a, b = value.astype(np.float64), reference.astype(np.float64)
    delta = a - b
    failed = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(b)))
    return dict(bitwise_equal=value.tobytes() == reference.tobytes(),
                elements=int(value.size), strict_rtol=1e-4, strict_atol=1e-5,
                failed_elements=failed, strict_pass=failed == 0,
                max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.sqrt(np.sum(delta * delta))) /
                max(float(np.sqrt(np.sum(b * b))), np.finfo(float).tiny))


def match(actual, recorded):
    require(isinstance(recorded, dict), 'Recorded comparison missing')
    for key, value in recorded.items():
        observed = actual['failed_elements' if key == 'outside_tolerance' else key]
        if key in ('max_abs', 'relative_l2'):
            require(math.isclose(observed, value, rel_tol=2e-12, abs_tol=1e-15), key + ' differs')
        else:
            require(observed == value, key + ' differs')


def audit(root):
    ledger, array_count = {}, 0

    def file(directory, row):
        path = (directory / row['file']).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), 'Artifact escapes saved root')
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'File identity differs: ' + relative)
        ledger[relative] = dict(bytes=len(data), sha256=sha(data))
        return relative, data

    def array(value, shape):
        nonlocal array_count
        require(value.shape == tuple(shape) and value.dtype == np.float32 and value.flags.c_contiguous
                and np.isfinite(value).all(), 'Array shape/dtype/contiguity/finite differs')
        array_count += 1
        return sha(value.tobytes())

    def npy(directory, row, shape):
        relative, data = file(directory, row)
        value = np.load(io.BytesIO(data), allow_pickle=False)
        digest = array(value, shape)
        require(row['shape'] == list(shape) and row['dtype'] == 'float32'
                and row['tensor_bytes'] == value.nbytes and row['raw_sha256'] == digest,
                'NPY raw identity differs: ' + relative)
        ledger[relative]['raw_sha256'] = digest
        return value

    receipts = {}
    for relative, expected in PINS.items():
        data = (root / relative).read_bytes()
        require(sha(data) == expected, 'Pinned receipt differs: ' + relative)
        receipts[relative] = json.loads(data)
    built, checked = receipts['compile-folded-r1/result.json'], receipts['compile-folded-r1/check.json']
    require(built['original_parent_manifest_sha256'] == checked['original_parent_manifest_sha256'] == PARENT_SHA
            and built['original_pair_onnx_sha256'] == checked['original_pair_onnx_sha256'] == GRAPH_SHA
            and built['script_sha256'] == checked['script_sha256'] == BUILDER_SHA
            and built['rewrite_helper_sha256'] == checked['rewrite_helper_sha256'] == REWRITE_SHA
            and built['check_receipt_sha256'] == PINS['compile-folded-r1/check.json']
            and checked['candidate_onnx_sha256'] == built['onnx']['sha256']
            and built['cases'] == checked['cases'], 'Array provenance differs')
    cases = checked['cases']
    require([c['tag'] for c in cases] == ['v53', 'v56-calib00']
            and cases[0]['original_outputs'] and cases[1]['original_outputs'] is None,
            'Original Torch reference coverage differs')
    require(cases[0]['input']['raw_sha256'] != cases[1]['input']['raw_sha256'], 'Distinct real inputs required')
    refs = [npy(root / 'original-reference/v53', row, shape)
            for row, shape in zip(cases[0]['original_outputs'], SHAPES)]
    require(len(refs) == 2, 'Two original Torch outputs required')
    fp32_checks = []
    for case in cases:
        inp = npy(root / 'compile-folded-r1', case['saved_input'], SHAPES[0])
        require(case['saved_input']['raw_sha256'] == case['input']['raw_sha256']
                and inp.nbytes == case['input']['tensor_bytes'], 'Captured input raw identity differs')
        require(len(case['baseline_outputs']) == len(case['candidate_outputs']) == len(case['checks']) == 2,
                'Two saved FP32 outputs required')
        for i, name in enumerate(NAMES):
            baseline = npy(root / 'compile-folded-r1', case['baseline_outputs'][i], SHAPES[i])
            candidate = npy(root / 'compile-folded-r1', case['candidate_outputs'][i], SHAPES[i])
            recorded = case['checks'][i]
            result = compare(candidate, baseline)
            match(result, recorded['versus_baseline_FP32_ONNX'])
            require(recorded['name'] == name and result['bitwise_equal']
                    and recorded['byte_equal_to_baseline'], 'FP32 baseline/candidate bytes differ')
            quality = None
            if case['tag'] == 'v53':
                quality = {label: compare(value, refs[i]) for label, value in
                           [('baseline', baseline), ('candidate', candidate)]}
                for label, comparison in quality.items():
                    match(comparison, recorded['versus_original_FP32'][label])
                    require(comparison['strict_pass'], 'Saved FP32 original Torch strict gate failed')
            else:
                require(recorded['versus_original_FP32'] is None, 'Missing Torch reference mislabeled')
            fp32_checks.append(dict(input=case['tag'], output=name,
                                    versus_baseline_FP32_ONNX=result, versus_original_Torch_FP32=quality))
    require(checked['all_candidate_vs_baseline_FP32_ONNX_byte_equal'] is True
            and checked['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] is True
            and checked['original_Torch_gate_passed_where_reference_available'] is True
            and checked['original_Torch_reference_case_count'] == 1
            and checked['original_Torch_references_cover_all_cases'] is False, 'Aggregate FP32 claims differ')

    normal_first, run_summaries, all_npu_quality = {}, [], []
    npz_count, repeat_count, cross_count = 0, 0, 0
    for run_name, expected_calls in [('normal-folded-r1', 80), ('profile-folded-r1', 4)]:
        run = receipts[run_name + '/result.json']
        require(run['compile_receipt_sha256'] == PINS['compile-folded-r1/result.json']
                and run['check_receipt_sha256'] == PINS['compile-folded-r1/check.json']
                and run['script_sha256'] == PROBE_SHA and run['input_cases'] == cases
                and len(run['calls']) == expected_calls, 'Run array provenance differs')
        first, first_quality = {}, []
        for call in run['calls']:
            tag, role = call['input'], call['role']
            require(tag in ('v53', 'v56-calib00') and role in ('A', 'B'), 'Unknown input/route')
            relative, data = file(root / run_name, call['artifact'])
            with np.load(io.BytesIO(data), allow_pickle=False) as saved:
                require(set(saved.files) == {'output0', 'output1'}, 'Saved NPZ bindings differ')
                outputs = [saved['output0'], saved['output1']]
            hashes = [array(value, shape) for value, shape in zip(outputs, SHAPES)]
            ledger[relative]['output_raw_sha256'] = hashes
            npz_count += 1
            cold = (tag, role) not in first
            require(call['first_for_input_role'] == cold, 'Saved first-route marker differs')
            if cold:
                first[tag, role] = outputs
            require((tag, 'A') in first, 'Same-input A reference missing')
            for i, value in enumerate(outputs):
                baseline = compare(value, first[tag, 'A'][i])
                repeated = compare(value, first[tag, role][i])
                match(baseline, call['same_input_first_A'][i])
                match(repeated, call['repeat_first_role'][i])
                require(repeated['bitwise_equal'], 'Same route does not reproduce its first output')
                repeat_count += int(not cold)
                quality = None
                if tag == 'v53':
                    quality = compare(value, refs[i])
                    match(quality, call['original_FP32'][i])
                    all_npu_quality.append(quality['strict_pass'])
                else:
                    require(call['original_FP32'] is None, 'Second input lacks original Torch references')
                if cold:
                    first_quality.append(dict(input=tag, route=role, output=NAMES[i], raw_sha256=hashes[i],
                                              versus_same_input_A=baseline, versus_original_Torch_FP32=quality))
                if run_name == 'profile-folded-r1':
                    require(value.tobytes() == normal_first[tag, role][i].tobytes(), 'Profile/normal arrays differ')
                    cross_count += 1
        require(set(first) == {(tag, role) for tag in ('v53', 'v56-calib00') for role in ('A', 'B')},
                'Not all four input/route combinations saved')
        if run_name == 'normal-folded-r1':
            normal_first = first
        run_summaries.append(dict(run=run_name, calls=len(run['calls']), first_output_comparisons=first_quality))
    require(len(ledger) == 96 and array_count == 180 and npz_count == 84, 'Saved artifact count differs')
    compact_ledger = json.dumps(ledger, sort_keys=True, separators=(',', ':')).encode()
    return dict(schema='v75_original_QKV_head_major_folded_saved_array_audit', status='passed_saved_array_audit',
                saved_data_only=True, model_OR_ORT_OR_SDK_executed=False, original_model_only=True,
                candidate_accepted=False, timing_and_lifecycle_audited_here=False,
                receipt_sha256=PINS, source_sha256=dict(builder=BUILDER_SHA, rewrite=REWRITE_SHA, probe=PROBE_SHA),
                original_parent_manifest_sha256=PARENT_SHA, original_pair_onnx_sha256=GRAPH_SHA,
                fixed_strict_rtol=1e-4, fixed_strict_atol=1e-5,
                artifact_files_verified=len(ledger), NPY_files_with_recorded_raw_SHA_verified=12,
                NPZ_files_with_recorded_file_SHA_verified=npz_count, arrays_shape_dtype_finite_verified=array_count,
                NPZ_output_raw_SHA_recomputed=True, NPZ_individual_raw_SHA_present_in_runtime_receipt=False,
                artifact_identity_ledger_sha256=sha(compact_ledger), FP32_checks=fp32_checks,
                NPU_runs=run_summaries, nonfirst_repeat_arrays_byte_equal=repeat_count,
                profile_vs_normal_arrays_byte_equal=cross_count,
                original_Torch_NPU_strict_checks=len(all_npu_quality),
                original_Torch_NPU_strict_checks_passed=sum(all_npu_quality),
                all_NPU_first_outputs_byte_equal_to_same_input_A=all(
                    item['versus_same_input_A']['bitwise_equal'] for run in run_summaries
                    for item in run['first_output_comparisons']),
                original_Torch_reference_case_count=1, original_Torch_references_cover_all_cases=False,
                limitations=['Only saved numerical outputs are checked; component timings and SDK lifecycle are audited separately.',
                             'The second real input has no original Torch reference; NPU strict failures remain failures.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(strict=True))
    result['auditor_sha256'] = sha(Path(__file__).read_bytes())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'artifact_files_verified',
          'arrays_shape_dtype_finite_verified', 'all_NPU_first_outputs_byte_equal_to_same_input_A',
          'original_Torch_NPU_strict_checks', 'original_Torch_NPU_strict_checks_passed')}))


if __name__ == '__main__':
    main()
