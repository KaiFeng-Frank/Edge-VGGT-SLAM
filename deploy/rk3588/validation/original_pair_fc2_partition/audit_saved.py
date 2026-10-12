#!/usr/bin/env python3
"""Audit saved v72 FC2 K-split arrays and receipts; never import a model or SDK."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PINS = {'normal-r1/result.json': '44682fef1a9a47551038841ce157a57fc5ef04c89bcefb464b0d050190197dc3', 'profile-r1/result.json': 'c4a534059642b588775f520c6d9a84ec64c5b343b683d73646b1e7ce627fee18', 'split2-r1/result.json': '82a7d3b21511b6f6ec8b2fe053b245aeeff7f4b052d3edb3dd84314d922f5e0e', 'split2-r1/check.json': '25714014bce0172b04d71a7e8be08ce63ca26b3b22d2310eabf19bab48116e27', 'split4-r1/result.json': '71641275e5501bcf3967c62a96e09706589e55c7088ed43f8587708b287c0622', 'split4-r1/check.json': '38ab16cb877f6696194559c3f9055f1179f7e8ebf5fc8ddf46ee7d621ed2dceb', 'build_pair_fc2_k_splits_board.py': 'a339f234e6fae5bd3ce92627342c2805a41d0018bd2dd978b48ba1722827040b', 'probe_pair_fc2_k_splits_board.py': 'dd6db7a844a9b3f7edb8bf45bb3ddb1bfd7a2433183deb7a2e61d99a447d9c46', 'vggt-v72-compile2-control-r1.json': 'a141634503579a6b59859099a2c6aa9a24cd6cf14784eb2709386c54483b696e', 'vggt-v72-compile4-control-r1.json': 'c5587bea3685f5446b5834323937b31dc3d198f33b45fd4f5071fd6e114450c4', 'vggt-v72-normal-control-r1.json': '9de1eb12541866feae2fc1c2844aa3818287be1eb4524e6b8dda0682dcb3059a', 'vggt-v72-normalretry-control-r1.json': '7c71874a1c62ca0fb764d8b856aa817587585f2322dc08acde2880a44b916a3c', 'vggt-v72-profile-control-r1.json': '6b916c0d1919a2af515a5bd738f419a2566c320205cf6f0c40faeddd139b0b0b'}
PARENT = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
ONNX = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
SHAPES = [(2, 53, 1024), (1, 106, 1024)]
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
              mean_values=None, std_values=None, quant_img_RGB2BGR=False,
              model_pruning=False, remove_reshape=False, quantize_weight=False,
              enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compare(value, reference):
    a, b = value.astype(np.float64), reference.astype(np.float64)
    delta = a - b
    bad = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(b)))
    denom = float(np.sqrt(np.sum(b * b)))
    return dict(bitwise_equal=value.tobytes() == reference.tobytes(),
                max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.sqrt(np.sum(delta * delta))) / max(denom, np.finfo(float).tiny),
                strict_rtol=1e-4, strict_atol=1e-5, failed_elements=bad,
                strict_pass=bad == 0)


def match(actual, recorded):
    for key, value in recorded.items():
        if key == 'elements':
            require(value == 108544, 'Wrong element count')
            continue
        observed = actual['failed_elements' if key == 'outside_tolerance' else key]
        if key in ('max_abs', 'relative_l2'):
            require(math.isclose(observed, value, rel_tol=2e-12, abs_tol=1e-15), key + ' summary differs')
        else:
            require(observed == value, key + ' differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, default=Path(__file__).resolve().parent.parent / 'edge-vggt-v70')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    ledger, count, first_normal = {}, 0, {}
    def file(path, row):
        require(path.resolve().is_relative_to(root) and path.is_file(), 'Artifact outside audit root')
        data = path.read_bytes()
        require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'File identity differs: ' + str(path))
        ledger[str(path.relative_to(root))] = dict(bytes=len(data), sha256=sha(data))
    def array(value, shape):
        nonlocal count
        require(value.shape == tuple(shape) and value.dtype == np.float32 and value.flags.c_contiguous
                and np.isfinite(value).all(), 'Array shape/dtype/finite differs')
        count += 1
        return sha(value.tobytes())
    def npy(directory, row):
        path = directory / row['file']; file(path, row)
        value = np.load(path, allow_pickle=False)
        digest = array(value, row['shape'])
        require(row['dtype'] == 'float32' and digest == row['raw_sha256']
                and value.nbytes == row['tensor_bytes'], 'NPY raw identity differs')
        ledger[str(path.relative_to(root))]['raw_sha256'] = digest
        return value
    receipts = {}
    for name, digest in PINS.items():
        data = (root / name).read_bytes()
        require(sha(data) == digest, 'Frozen input changed: ' + name)
        if name.endswith('.json'):
            receipts[name] = json.loads(data)
    compiled = [receipts[f'split{n}-r1/result.json'] for n in (2, 4)]
    refs = [npy(root / 'original-reference', row) for row in compiled[0]['cases'][0]['original_outputs']]
    old_root = args.original_root.resolve(strict=True)
    old_receipt = old_root / 'normal-r2/result.json'
    old_data = old_receipt.read_bytes()
    require(sha(old_data) == '2ace34b958b0bd062f36887b6975e12569cd87895709117884a073173a78d5ef', 'Original v70 confirmation receipt changed')
    old_run = json.loads(old_data); old_A = {}; old_A_artifacts = []
    for call in old_run['calls']:
        if call['role'] != 'A' or not call['first_for_input_role']: continue
        path = (old_root / 'normal-r2' / call['artifact']['file']).resolve(strict=True)
        require(path.is_relative_to(old_root), 'Old baseline artifact escaped root')
        raw = path.read_bytes(); require(len(raw) == call['artifact']['bytes'] and sha(raw) == call['artifact']['sha256'], 'Old baseline artifact changed')
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == {'output0', 'output1'}, 'Old baseline output set differs')
            old_A[call['input']] = [saved['output0'], saved['output1']]
        for value, shape in zip(old_A[call['input']], SHAPES): array(value, shape)
        old_A_artifacts.append(dict(input=call['input'], artifact=call['artifact']))
    require(set(old_A) == {'v53', 'v56-calib00'}, 'Two old original FP16 inputs required')
    require([c['saved_input']['raw_sha256'] for c in old_run['input_cases']] == [c['saved_input']['raw_sha256'] for c in compiled[0]['cases']], 'Original FP16 baseline inputs differ')
    checks = []
    for chunks, built in zip((2, 4), compiled):
        checked = receipts[f'split{chunks}-r1/check.json']
        require(built['status'] == 'compiled_not_quality_accepted' and built['chunks'] == chunks
                and built['configuration'] == CONFIG and built['build_options'] == BUILD
                and built['original_parent_manifest_sha256'] == PARENT and built['original_pair_onnx_sha256'] == ONNX
                and built['check_receipt_sha256'] == PINS[f'split{chunks}-r1/check.json']
                and built['script_sha256'] == PINS['build_pair_fc2_k_splits_board.py'], 'Compile provenance differs')
        require(built['schema'] == 'v72_original_VGGT_pair00_FC2_K_split_build'
            and built['FP32_reduction_order_changed'] is True
            and built['partial_sum_order'] == 'left_associative_in_increasing_K_chunk_order', 'K-partition scope differs')
        proof=built['graph_proof']
        for key in ('all_other_original_nodes_protobuf_unchanged','all_other_original_initializers_protobuf_unchanged',
            'original_FC1_nodes_protobuf_unchanged','original_GELU_nodes_protobuf_unchanged','original_FC2_bias_Add_nodes_protobuf_unchanged',
            'original_LayerScale_and_residual_nodes_protobuf_unchanged','original_IO_protobuf_unchanged',
            'all_original_FC2_parameter_values_preserved','FC2_weights_not_duplicated','FC2_bias_added_once_after_partial_sum'):
            require(proof[key] is True, 'Missing original parameter/node proof: '+key)
        require(built['released'] and built['release_code'] in (None, 0) and not built['cleanup_errors']
                and built['environment_restored'] and built['source_and_inputs_unchanged']
                and built['source_and_input_hashes'] == built['source_and_input_hashes_after'], 'Compile cleanup failed')
        require([x['name'] for x in built['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
                and all(x['status'] == 'completed' and x['code'] in (None, 0) for x in built['operations']), 'Compile incomplete')
        require(checked['schema'] == 'v72_original_VGGT_pair00_FC2_K_split_ONNX_check' and checked['status'] == 'passed_original_FP32_ONNX_equivalence_not_quality_accepted'
                and checked['cases'] == built['cases'] and checked['candidate_onnx_sha256'] == built['onnx']['sha256']
                and checked['optimization'] == 'ORT_DISABLE_ALL' and checked['sessions_dropped']
                and not checked['cleanup_errors'] and checked['original_Torch_references_cover_all_cases'] is False,
                'Saved ONNX check provenance differs')
        for case in built['cases']:
            directory = root / f'split{chunks}-r1'
            input_array = npy(directory, case['saved_input'])
            require(case['saved_input']['raw_sha256'] == case['input']['raw_sha256']
                    and input_array.shape == SHAPES[0], 'Real input differs')
            ref_available = case['tag'] == 'v53'
            require((case['original_outputs'] is not None) == ref_available, 'Torch reference coverage differs')
            if ref_available:
                require(case['original_outputs'] == compiled[0]['cases'][0]['original_outputs'], 'Torch references differ')
            for i, recorded in enumerate(case['checks']):
                baseline = npy(directory, case['baseline_outputs'][i])
                candidate = npy(directory, case['candidate_outputs'][i])
                require(baseline.shape == SHAPES[i] and candidate.shape == SHAPES[i], 'Output binding shape differs')
                c = compare(candidate, baseline); match(c, recorded['versus_baseline_FP32_ONNX'])
                require(c['bitwise_equal'] == recorded['byte_equal_to_baseline'] and c['strict_pass'], 'Fixed original FP32 ONNX gate failed or byte-equality misreported')
                quality = {role: compare(v, refs[i]) for role, v in [('baseline', baseline), ('candidate', candidate)]} if ref_available else None
                if quality:
                    for role, result in quality.items():
                        match(result, recorded['versus_original_FP32'][role]); require(result['strict_pass'], 'Torch gate failed')
                else:
                    require(recorded['versus_original_FP32'] is None, 'Missing reference mislabeled')
                checks.append(dict(chunks=chunks, input=case['tag'], output=i, baseline_byte_equal=c['bitwise_equal'], versus_baseline_FP32_ONNX=c, original_Torch=quality))
    runs, sdk_strict, repeats, cross, platforms = {}, [], 0, 0, []
    model_shas = {'A': '309a95a7467cce404161db3ed674b1ff1ca8cf7806776c40b21b195b2619cbd9',
                  'B': compiled[0]['rknn']['sha256'], 'C': compiled[1]['rknn']['sha256']}
    for name, rounds in [('normal-r1', 10), ('profile-r1', 0)]:
        run = receipts[name + '/result.json']; profiling = name == 'profile-r1'
        expected = list('ABC' if profiling else 'ABCCBA' * rounds)
        calls, first, route_quality = run['calls'], {}, {}
        require(run['schedule'] == expected and len(calls) == len(expected) * 2 and run['hidden_warmup_calls'] == 0,
                'Actual call order/denominator differs')
        require(run['input_cases'] == compiled[0]['cases'] and run['parent_manifest_sha256'] == PARENT
                and run['script_sha256'] == PINS['probe_pair_fc2_k_splits_board.py']
                and run['status'] == 'completed_component_diagnostic_not_quality_accepted'
                and run['schema'] == 'v72_original_pair00_FC2_K_splits_FP16_IO_probe'
                and run['FP32_reduction_order_changed'] is True and run['partial_sum_order'] == 'left_associative_in_increasing_K_chunk_order'
                and run['profile'] == profiling and run['init_flags'] == (8 if profiling else 0)
                and run['normal_benchmark_timing'] == (not profiling) and run['core_mask'] == 7
                and run['cpu_affinity'] == [4, 5, 6, 7] and not run['system_clocks_changed']
                and run['original_model_only'] and not run['candidate_accepted'] and not run['full_model_or_realtime_validated'],
                'Run identity/scope differs')
        require(run['contexts_created'] == run['contexts_destroyed'] == 3 and run['live_contexts'] == 0
                and not run['cleanup_errors'] and run['source_and_inputs_unchanged'] and run['affinity_restored']
                and run['counters'] == dict(sdk_run_attempts=len(calls), sdk_runs_succeeded=len(calls),
                    outputs_acquired=2*len(calls), outputs_released=2*len(calls)), 'SDK lifecycle failed')
        for query in run['queries']:
            require(query['model_sha256'] == model_shas[query['role']], 'Compiled model identity differs')
            for desc, shape in zip([query['input']] + query['outputs'], [SHAPES[0]] + SHAPES):
                require(desc['shape'] == list(shape) and desc['type'] == 1 and desc['format'] == 3
                        and desc['quantization_type'] in (0, 2) and desc['zero_point'] == 0 and desc['scale'] == 1
                        and desc['pass_through'] == 0 and desc['size'] == desc['size_with_stride'] == 217088,
                        'Native FP16 ABI differs')
        for position, call in enumerate(calls):
            tag = ('v53', 'v56-calib00')[position // len(expected)]; role = expected[position % len(expected)]
            require(call['input'] == tag and call['sequence'] == position % len(expected) and call['role'] == role,
                    'Saved call order differs')
            path = root / name / call['artifact']['file']; file(path, call['artifact'])
            with np.load(path, allow_pickle=False) as saved:
                require(set(saved.files) == {'output0', 'output1'}, 'Unexpected saved outputs')
                outputs = [saved['output0'], saved['output1']]
            hashes = [array(value, shape) for value, shape in zip(outputs, SHAPES)]
            if role == 'A': require(all(v.tobytes() == b.tobytes() for v, b in zip(outputs, old_A[tag])), 'Original FP16 baseline changed since v70')
            ledger[str(path.relative_to(root))]['output_raw_sha256'] = hashes
            cold = (tag, role) not in first
            require(call['first_for_input_role'] == cold, 'First-call exclusion differs')
            if cold:
                first[tag, role] = outputs
            for i, value in enumerate(outputs):
                a = compare(value, first[tag, 'A'][i]); match(a, call['same_input_first_A'][i])
                repeat = compare(value, first[tag, role][i]); match(repeat, call['repeat_first_role'][i])
                require(repeat['bitwise_equal'], 'Repeated output differs'); repeats += int(not cold)
                if tag == 'v53':
                    q = compare(value, refs[i]); match(q, call['original_FP32'][i]); sdk_strict.append(q['strict_pass'])
                else:
                    require(call['original_FP32'] is None, 'Second input has no Torch reference'); q = None
                if cold:
                    route_quality.setdefault(tag, {})[role + str(i)] = dict(versus_first_A=a, original_Torch=q, raw_sha256=hashes[i])
                if name != 'normal-r1':
                    require(value.tobytes() == first_normal[tag, role][i].tobytes(), 'Normal/profile/confirmation output differs')
                    cross += 1
            timing = call['timing']; require(call['component_seconds'] == timing['route_seconds'] > 0
                and timing['output_count'] == 2 and timing['input_api_type'] == 1 and timing['output_want_float'] == 0
                and timing['outputs_released'], 'Route timing/output release differs')
            phase_values = [timing[k] for k in ('inputs_set_seconds', 'run_seconds', 'outputs_get_seconds', 'outputs_release_seconds')]
            require(all(math.isfinite(v) and v >= 0 for v in phase_values) and sum(phase_values) <= timing['route_seconds'], 'Invalid timing phases')
            platforms.extend([call['platform_before'], call['platform_after']])
            if profiling:
                detail = call['profile']; raw = detail['raw_text'].encode()
                require(detail['query_command'] == 3 and sha(raw) == detail['raw_sha256']
                        and len(raw) == detail['bytes'], 'Saved profile text identity differs')
        if name == 'normal-r1':
            first_normal = first
        summaries = {}
        if not profiling:
            for role in 'ABC':
                warm = [c for c in calls if c['role'] == role and not c['first_for_input_role']]
                mean = float(np.mean([c['component_seconds'] for c in warm])); saved = run['summary'][role]
                require(len(warm) == 4*rounds-2 == saved['warm_count'] and mean == saved['mean_component_seconds'], 'Warm mean differs')
                per_input = {tag: float(np.mean([c['component_seconds'] for c in warm if c['input'] == tag])) for tag in ('v53', 'v56-calib00')}
                require(per_input == saved['per_input'], 'Per-input warm mean differs')
                summaries[role] = dict(warm_count=len(warm), mean_seconds=mean, per_input=per_input,
                    SDK_phase_mean_seconds={k: float(np.mean([c['timing'][k] for c in warm])) for k in ('inputs_set_seconds', 'run_seconds', 'outputs_get_seconds', 'outputs_release_seconds')})
            for role in 'BC':
                summaries[role]['reduction_vs_A_percent'] = 100*(1-summaries[role]['mean_seconds']/summaries['A']['mean_seconds'])
        runs[name] = dict(calls=len(calls), outputs=2*len(calls), summaries=summaries,
                         route_quality=route_quality, lifecycle=run['counters'], contexts_closed=3)
    require(all(p['actual_frequency_hz'] == 300000000 and p['governor'] == 'userspace' for p in platforms), 'Call frequency differs')
    controls = {}
    for name, control in receipts.items():
        if '-control-' not in name:
            continue
        failed_launch = control['stage'] == 'normal'
        if failed_launch:
            require(control['status'] == 'failed' and control.get('returncode') is None and 'child_pid' not in control
                and control.get('error', '').startswith('FileNotFoundError:') and control['command'][0].endswith('/python3.11.17'), 'Unexpected failed-launch evidence')
        else:
            require(control['status'] == 'completed' and control['returncode'] == 0, 'Actual process did not complete')
        require(control['frequency_written'] is False and not control['cleanup_errors']
                and all(s['npu_hz'] == 300000000 for s in control['samples']), 'Control lifecycle/frequency differs')
        temperatures = [v for s in control['samples'] for v in s['temperatures_c'].values()]
        require(all(math.isfinite(v) for v in temperatures), 'Nonfinite thermal samples')
        controls[name] = dict(status=control['status'], returncode=control.get('returncode'), child_started='child_pid' in control,
                             failed_before_child_start=failed_launch, inference_run_count_not_derived_from_monitor=True,
                             samples=len(control['samples']), npu_hz=300000000,
                             min_temperature_c=min(temperatures), max_temperature_c=max(temperatures), no_clock_write=True)
    require(count == 278 and len(ledger) == 148 and len(sdk_strict) == 126, 'Array/file accounting differs')
    report = dict(schema='v72_original_pair00_FC2_K_saved_numpy_audit', status='passed_saved_evidence_audit',
        auditor_sha256=sha(Path(__file__).read_bytes()), input_pins=PINS,
        arrays_verified=count, artifact_files_verified=len(ledger), file_hashes=ledger,
        SDK_runs=126, SDK_outputs_released=252, contexts_closed=6,
        original_FP16_A_unchanged_since_v70=dict(receipt_sha256=sha(old_data), original_artifacts=old_A_artifacts, compared_outputs=84, bitwise_equal=True),
        FP32_reduction_order_changed=True, FP32_ONNX_byte_equivalence_required=False,
        nonfirst_repeat_outputs_byte_equal=repeats, profile_and_confirmation_outputs_byte_equal_to_normal=cross,
        original_Torch_SDK_strict=dict(covered_outputs=len(sdk_strict), passed=sum(sdk_strict), reference_case='v53', missing_case='v56-calib00'),
        compiled_saved_checks=checks, runs=runs, monitors=controls,
        call_frequency_samples=len(platforms), all_sampled_call_NPU_hz=300000000,
        limits=['Saved-array/receipt audit only; no actual model, ONNX, ORT, SDK or board execution.',
                'Original Torch references cover v53 only; v56-calib00 has no original Torch output reference.',
                'FP32 export checks and SDK FP16 strict gates are distinct; component success is not quality acceptance.',
                'Warm timing excludes first call separately per input and route; profile timing is not normal timing.',
                'NPU frequency/temperature samples do not prove frequency or temperatures between samples.',
                'K-partition changes FP32 reduction order: strict equivalence is required, byte equivalence is recorded rather than assumed.',
                'The normal launcher failure occurred before child creation; it contributes no SDK inference calls.',
                'Graphs and RKNN binaries are not present in the saved audit root; graph/model identities are bound through the pinned build/check receipts.',
                'No full-model speed, ATE/RPE, fresh pose rate or realtime acceptance demonstrated.'])
    with args.output.open('x') as stream: stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'arrays_verified', 'artifact_files_verified', 'SDK_runs', 'SDK_outputs_released', 'contexts_closed', 'original_Torch_SDK_strict')}))


if __name__ == '__main__':
    main()
