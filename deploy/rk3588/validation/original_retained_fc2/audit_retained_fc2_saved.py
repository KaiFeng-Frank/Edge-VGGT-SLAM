#!/usr/bin/env python3
"""Host-only NumPy audit of saved original-VGGT retained-group FC2 checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

GROUPS = (list(range(5)), list(range(5, 12)), list(range(12, 18)), list(range(18, 24)))
FRAME = [2, 53, 1024]
GLOBAL = [1, 106, 1024]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def read_json(path, expected_sha):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected_sha, 'Receipt SHA differs: ' + str(path))
    return json.loads(path.read_text())


def load(root, item, shape, ledger):
    root = Path(root).resolve(strict=True)
    relative = Path(item['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative tensor artifact required')
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.stat().st_size == item['bytes']
            and sha(path) == item['sha256'], 'Tensor file identity differs')
    value = np.load(path, allow_pickle=False)
    require(isinstance(value, np.ndarray) and value.dtype == np.float32
            and list(value.shape) == shape and value.flags.c_contiguous
            and np.isfinite(value).all(), 'FP32 shape/finite contract differs')
    raw = hashlib.sha256(memoryview(value).cast('B')).hexdigest()
    require(item['dtype'] == 'float32' and item['shape'] == shape
            and item['tensor_bytes'] == value.nbytes and item['raw_sha256'] == raw,
            'Tensor raw identity differs')
    ledger.append(dict(file=item['file'], bytes=item['bytes'], sha256=item['sha256'],
                       raw_sha256=raw, shape=shape))
    return value


def compare(actual, reference):
    # Same fixed FP32 rule as check_compile_pair_group.compare; no tolerance changes.
    delta, expected = actual.astype(np.float64) - reference.astype(np.float64), reference.astype(np.float64)
    outside = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(expected)))
    return dict(strict_rtol=1e-4, strict_atol=1e-5, strict_pass=outside == 0,
                outside_tolerance=outside, elements=int(actual.size), max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-30)))


def match(recorded, actual):
    for key, value in actual.items():
        require(key in recorded, 'Comparison field missing: ' + key)
        if key in ('max_abs', 'relative_l2'):
            require(math.isclose(recorded[key], value, rel_tol=2e-12, abs_tol=2e-12),
                    'Comparison metric differs: ' + key)
        else:
            require(recorded[key] == value, 'Strict count/threshold differs: ' + key)


def audit_arrays(check, directory):
    index = check['group_index']
    require(type(index) is int and index in range(4) and check['indices'] == GROUPS[index],
            'Complete original group indices differ')
    names = ['frame_output', 'global_output'] + (['target_global_output'] if index == 3 else [])
    shapes = [FRAME, GLOBAL] + ([GLOBAL] if index == 3 else [])
    bindings = [dict(index=i, name=name, shape=shape, dtype='float32')
                for i, (name, shape) in enumerate(zip(names, shapes))]
    require(check['output_bindings'] == bindings, 'Original retained output ABI differs')
    ledger, comparisons = [], []
    require([case['tag'] for case in check['cases']] == ['v53', 'v56-calib00'],
            'Two original real cases required')
    for case in check['cases']:
        x = load(directory, case['saved_input'], FRAME, ledger)
        require(case['saved_input']['raw_sha256'] == case['input']['raw_sha256'],
                'Saved input differs from original capture')
        del x
        require(len(case['baseline_outputs']) == len(case['candidate_outputs'])
                == len(case['checks']) == len(names), 'Retained output count differs')
        references = case['original_reference_outputs']
        require((references is None) == (case['tag'] == 'v56-calib00'),
                'Original Torch reference coverage differs')
        if references is not None:
            require(len(references) == len(names) == len(case['original_outputs']),
                    'Original Torch reference count differs')
        for i, (name, shape) in enumerate(zip(names, shapes)):
            baseline = load(directory, case['baseline_outputs'][i], shape, ledger)
            candidate = load(directory, case['candidate_outputs'][i], shape, ledger)
            recorded = case['checks'][i]
            require(recorded['name'] == name, 'Retained output name differs')
            versus = compare(candidate, baseline)
            match(recorded['versus_baseline_FP32_ONNX'], versus)
            bitwise = candidate.tobytes() == baseline.tobytes()
            require(recorded['byte_equal_to_baseline'] is bitwise, 'Byte comparison differs')
            original = None
            if references is not None:
                require(references[i]['raw_sha256'] == case['original_outputs'][i]['raw_sha256'],
                        'Saved Torch reference differs from original capture')
                reference = load(directory, references[i], shape, ledger)
                original = {role: compare(value, reference)
                            for role, value in [('baseline', baseline), ('candidate', candidate)]}
                for role, row in original.items():
                    match(recorded['versus_original_FP32'][role], row)
            else:
                require(recorded['versus_original_FP32'] is None, 'Missing Torch reference fabricated')
            comparisons.append(dict(input=case['tag'], output=name, byte_equal=bitwise,
                                     versus_baseline_FP32_ONNX=versus, original_Torch=original))
    passed = all(row['versus_baseline_FP32_ONNX']['strict_pass'] for row in comparisons)
    original_passed = all(row['original_Torch'][role]['strict_pass']
                          for row in comparisons if row['original_Torch'] is not None
                          for role in ('baseline', 'candidate'))
    require(check['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] is passed,
            'Aggregate candidate FP32 strict flag differs')
    require(check['original_Torch_gate_passed_where_reference_available'] is original_passed,
            'Aggregate original Torch gate differs')
    return dict(arrays_verified=len(ledger), artifacts=ledger, comparisons=comparisons,
                candidate_strict_passed=passed, original_Torch_strict_passed=original_passed,
                candidate_failed_elements=sum(row['versus_baseline_FP32_ONNX']['outside_tolerance']
                                               for row in comparisons),
                byte_equal_outputs=sum(row['byte_equal'] for row in comparisons),
                original_Torch_reference_case_count=1,
                last_group_global20_checked=index == 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', nargs=2, metavar=('JSON', 'SHA256'), required=True)
    parser.add_argument('--build', nargs=2, metavar=('JSON', 'SHA256'), required=True)
    parser.add_argument('--builder-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check = read_json(*args.check)
    build = read_json(*args.build)
    require(check['schema'] == 'v73_original_VGGT_retained_FC2_K4_check'
            and build['schema'] == 'v73_original_VGGT_retained_FC2_K4_build', 'v73 receipt schema differs')
    for row in (check, build):
        require(row['script_sha256'] == args.builder_sha256 and row['chunks'] == 4
                and row['baseline_manifest_sha256'] == '771a9b82007d2feefd65ebbf3f4b0fb27070b79ff7f6d0faab1a8c866da90e43'
                and row['original_parent_manifest_sha256'] == '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
                and row['FP32_reduction_order_changed'] is True
                and row['partial_sum_order'] == 'left_associative_in_increasing_K_chunk_order'
                and row['candidate_accepted'] is False and row['cleanup_errors'] == [], 'Original identity/policy differs')
    for key in ('group_index', 'indices', 'output_bindings', 'public_output_sources', 'baseline_graph_sha256'):
        require(check[key] == build[key], 'Check/build identity differs: ' + key)
    require(check['candidate_onnx_sha256'] == build['onnx']['sha256'], 'Check is not bound to candidate graph')
    require(build['source_and_inputs_unchanged'] is True and build['environment_restored'] is True
            and build['source_and_input_hashes'] == build['source_and_input_hashes_after'], 'Source/cleanup identity differs')
    if 'check_receipt_sha256' in build:
        require(build['check_receipt_sha256'] == args.check[1], 'Build check SHA differs')
    index = check['group_index']
    require(type(index) is int and index in range(4), 'Group index differs')
    sources = [dict(index=i, name=name, pair_index=GROUPS[index][-1], role=role)
               for i, (name, role) in enumerate((('frame_output', 'frame'), ('global_output', 'global')))]
    if index == 3:
        sources.append(dict(index=2, name='target_global_output', pair_index=20, role='global'))
    require(check['public_output_sources'] == sources, 'Retained/global20 source contract differs')
    result = audit_arrays(check, Path(args.check[0]).resolve(strict=True).parent)
    passed = result['candidate_strict_passed']
    require(check['status'] == ('passed_original_FP32_ONNX_equivalence_not_quality_accepted' if passed else 'failed')
            and check['sessions_dropped'] is True, 'Check status/session evidence differs')
    if not passed:
        require(build['status'] == 'failed' and build['operations'] == [] and build.get('rknn') is None,
                'Failed FP32 candidate must not compile')
    result.update(schema='v73_retained_FC2_K4_saved_numpy_audit', status='passed_saved_evidence_audit',
                  group_index=index, indices=GROUPS[index], check_status=check['status'], build_status=build['status'],
                  check_sha256=args.check[1], build_sha256=args.build[1], builder_sha256=args.builder_sha256,
                  auditor_sha256=sha(__file__), candidate_accepted=False,
                  limits=['Saved arrays/receipts only; no model, ORT, SDK, compiler or board execution.',
                          'Candidate strict pass and original Torch strict pass are separate; original references cover v53 only.',
                          'K partition changes floating-point reduction order; byte identity is not required or assumed.',
                          'Graph/weight preservation is bound to pinned receipts and source; host graph binaries were not read.',
                          'No hardware/full-model performance or quality acceptance is established by this audit.'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'group_index', 'arrays_verified',
                      'candidate_strict_passed', 'candidate_failed_elements', 'original_Torch_strict_passed')}))


if __name__ == '__main__':
    main()
