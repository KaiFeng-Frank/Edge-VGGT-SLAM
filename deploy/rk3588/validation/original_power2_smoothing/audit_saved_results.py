#!/usr/bin/env python3
"""Read saved v78 JSON/NumPy only; never import Torch, ONNX, ORT or SDK."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SHAPES = ([2, 53, 1024], [1, 106, 1024])
files_checked = set()
arrays_checked = 0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def artifact(directory, row):
    relative = Path(row['file'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = (directory / relative).resolve(strict=True)
    assert path.is_relative_to(directory.resolve())
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    files_checked.add(path)
    return path


def array(value, shape, digest=None):
    global arrays_checked
    assert value.dtype == np.float32 and list(value.shape) == shape
    assert value.flags.c_contiguous and np.isfinite(value).all()
    if digest is not None:
        assert raw(value) == digest
    arrays_checked += 1
    return value


def tensor(directory, row):
    value = array(np.load(artifact(directory, row), allow_pickle=False), row['shape'], row['raw_sha256'])
    assert row['dtype'] == 'float32' and row['tensor_bytes'] == value.nbytes
    return value


def compare(value, reference):
    delta = value.astype(np.float64) - reference.astype(np.float64)
    threshold = 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))
    count = int(np.count_nonzero(np.abs(delta) > threshold))
    return dict(strict_pass=count == 0, outside_tolerance=count, elements=value.size,
                max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta.ravel()) /
                                  max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-30)))


def same_report(actual, recorded):
    for key in ('strict_pass', 'outside_tolerance', 'elements'):
        assert actual[key] == recorded[key], (key, actual, recorded)
    for key in ('max_abs', 'relative_l2'):
        assert math.isclose(actual[key], recorded[key], rel_tol=1e-11, abs_tol=1e-12)
    assert recorded['strict_rtol'] == 1e-4 and recorded['strict_atol'] == 1e-5


def stats(values):
    assert values and all(math.isfinite(x) and x >= 0 for x in values)
    return dict(count=len(values), mean_ms=float(np.mean(values) * 1000),
                median_ms=float(np.median(values) * 1000),
                p95_ms=float(np.percentile(values, 95) * 1000),
                min_ms=min(values) * 1000, max_ms=max(values) * 1000)


def main():
    checked = load_json(ROOT / 'check-r1/result.json')
    check = load_json(ROOT / 'check-r1/check.json')
    compiled = load_json(ROOT / 'compile.json')
    assert checked['script_sha256'] == sha(ROOT / 'build_pair_power2_smooth_board.py')
    assert compiled['script_sha256'] == sha(ROOT / 'compile_smoothed_pair_board.py')
    assert compiled['checked_sha256'] == sha(ROOT / 'check-r1/result.json')
    assert compiled['check_receipt_sha256'] == checked['check_receipt_sha256'] == sha(ROOT / 'check-r1/check.json')
    assert checked['status'] == 'checked_original_FP32_equivalence_not_quality_accepted'
    assert checked['source_and_inputs_unchanged'] and checked['cleanup_errors'] == []
    assert compiled['status'] == 'compiled_not_quality_accepted'
    assert compiled['released'] and compiled['release_code'] in (None, 0)
    assert compiled['cleanup_errors'] == [] and compiled['environment_restored'] and compiled['source_and_inputs_unchanged']
    assert [x['name'] for x in compiled['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
    assert all(x['code'] in (0, None) for x in compiled['operations'])
    assert compiled['candidate_graph_sha256'] == checked['onnx']['sha256'] == check['candidate_onnx_sha256']
    assert compiled['check_source_sha256'] == checked['script_sha256']
    references, torchrefs, inputs, fp32_checks = {}, {}, {}, []
    assert check['cases'] == checked['cases'] and len(check['cases']) == 2
    for case in check['cases']:
        tag = case['tag']
        x = tensor(ROOT / 'check-r1', case['saved_input'])
        assert list(x.shape) == SHAPES[0] and raw(x) == case['input']['raw_sha256']
        inputs[tag] = raw(x)
        references[tag] = [tensor(ROOT / 'check-r1', row) for row in case['baseline_outputs']]
        candidate = [tensor(ROOT / 'check-r1', row) for row in case['candidate_outputs']]
        assert len(references[tag]) == len(candidate) == len(case['checks']) == 2
        if case['original_outputs']:
            directory = Path('/dev/shm/edge-vggt-v75/original-reference/v53')
            torchrefs[tag] = [tensor(directory, row) for row in case['original_outputs']]
        for index, (base, changed, row) in enumerate(zip(references[tag], candidate, case['checks'])):
            eq = base.tobytes() == changed.tobytes()
            assert eq == row['byte_equal_to_baseline']
            same_report(compare(changed, base), row['versus_baseline_FP32_ONNX'])
            assert row['versus_baseline_FP32_ONNX']['strict_pass']
            if tag in torchrefs:
                for role, value in [('baseline', base), ('candidate', changed)]:
                    same_report(compare(value, torchrefs[tag][index]), row['versus_original_FP32'][role])
            fp32_checks.append(dict(input=tag, output=index, byte_equal=eq, strict_pass=True))
    maxima = {name: np.zeros(1024, dtype=np.float64) for name in ['frame0/norm1', 'frame0/norm2', 'global0/norm1', 'global0/norm2']}
    assert len(checked['calibration']) == 8
    for sample in checked['calibration']:
        with np.load(artifact(ROOT / 'check-r1', sample['artifact']), allow_pickle=False) as saved:
            assert set(saved.files) == set(maxima)
            for row in sample['norms']:
                y = array(saved[row['name']], row['shape'], row['raw_sha256'])
                maxima[row['name']] = np.maximum(maxima[row['name']], np.max(np.abs(y.astype(np.float64)), axis=(0, 1)))
    with np.load(artifact(ROOT / 'check-r1', checked['norm_absmax']), allow_pickle=False) as saved:
        assert set(saved.files) == set(maxima)
        assert all(saved[name].dtype == np.float64 and saved[name].shape == (1024,) and
                   np.isfinite(saved[name]).all() and saved[name].tobytes() == maxima[name].tobytes() for name in maxima)
    summaries, firsts, stage_reports, profile_checks = {}, {}, {}, []
    for stage, expected_calls in [('normal', 72), ('profile', 6)]:
        directory = ROOT / (stage + '-r1')
        report = load_json(directory / 'result.json')
        assert report['script_sha256'] == sha(ROOT / 'probe_smoothed_pair_board.py')
        assert report['checked_sha256'] == sha(ROOT / 'check-r1/result.json') and report['compiled_sha256'] == sha(ROOT / 'compile.json')
        assert report['cases'] == check['cases'] and report['status'] == 'completed_component_diagnostic_not_quality_accepted'
        assert report['contexts_created'] == report['contexts_destroyed'] == 3 and report['live_contexts'] == 0
        assert report['cleanup_errors'] == [] and not report['candidate_accepted'] and not report['realtime_validated']
        assert report['profile'] == (stage == 'profile') and report['init_flags'] == (8 if stage == 'profile' else 0)
        assert report['core_mask'] == 7 and report['frequency_hz'] == 1000000000
        assert report['counters'] == dict(run_attempts=expected_calls, runs_succeeded=expected_calls,
                                          outputs_acquired=expected_calls * 2, outputs_released=expected_calls * 2)
        assert len(report['calls']) == expected_calls and report['schedule'] == list('ABC' if stage == 'profile' else 'ABCCBA' * 6)
        counts, seen, temperatures, freq_count = {}, {}, [], 0
        for case in check['cases']:
            tag = case['tag']; rows = [row for row in report['calls'] if row['input'] == tag]
            assert [x['sequence'] for x in rows] == list(range(len(report['schedule'])))
            assert [x['role'] for x in rows] == report['schedule']
        for row in report['calls']:
            key = (row['input'], row['role'])
            assert row['first_for_input_role'] == (key not in seen)
            with np.load(artifact(directory, row['artifact']), allow_pickle=False) as saved:
                assert set(saved.files) == {'output0', 'output1'}
                ys = [array(saved['output' + str(i)], SHAPES[i], row['raw_sha256'][i]).copy() for i in range(2)]
            if key not in seen: seen[key] = ys
            eq = [a.tobytes() == b.tobytes() for a, b in zip(ys, seen[key])]
            assert eq == row['repeat_first_role_byte_equal'] and all(eq)
            for i, y in enumerate(ys):
                same_report(compare(y, references[key[0]][i]), row['versus_original_ONNX'][i])
                if key[0] in torchrefs:
                    same_report(compare(y, torchrefs[key[0]][i]), row['versus_original_Torch'][i])
                else: assert row['versus_original_Torch'] is None
            assert row['component_seconds'] == row['timing']['route_seconds']
            items = [v for k, v in row['timing'].items() if k != 'route_seconds']
            assert all(math.isfinite(v) and v >= 0 for v in items)
            assert sum(items) <= row['component_seconds'] + 1e-9
            for when in ('platform_before', 'platform_after'):
                sample = row[when]; assert sample['actual_frequency_hz'] == 1000000000
                assert len(sample['temperatures_c']) == 7 and max(sample['temperatures_c'].values()) < 75
                temperatures.extend(sample['temperatures_c'].values()); freq_count += 1
            counts[key] = counts.get(key, 0) + 1
            if stage == 'profile':
                assert all(a.tobytes() == b.tobytes() for a, b in zip(ys, firsts[key]))
                p = row['profile']; data = p['raw_text'].encode('utf-8')
                assert len(data) == p['bytes'] and hashlib.sha256(data).hexdigest() == p['raw_sha256']
                profile_checks.append(dict(input=key[0], role=key[1], raw_sha256=p['raw_sha256'], bytes=len(data), outputs_byte_equal_to_normal=True))
        if stage == 'normal':
            firsts = seen
            for role in 'ABC':
                warm = [x['component_seconds'] for x in report['calls'] if x['role'] == role and not x['first_for_input_role']]
                summaries[role] = stats(warm)
                assert len(warm) == report['summary'][role]['warm_count'] == 22
                assert float(np.mean(warm)) == report['summary'][role]['mean_component_seconds']
                summaries[role]['per_input'] = {tag: stats([x['component_seconds'] for x in report['calls'] if x['role'] == role and x['input'] == tag and not x['first_for_input_role']]) for tag in inputs}
        stage_reports[stage] = dict(result_sha256=sha(directory / 'result.json'), calls=expected_calls,
            outputs=expected_calls * 2, contexts_created=3, contexts_destroyed=3, live_contexts=0,
            repetitions_bitwise=True, sampled_frequency_count=freq_count, sampled_frequency_hz=1000000000,
            max_call_temperature_c=max(temperatures), cleanup_errors=[])
    errors = []
    for (tag, role), ys in firsts.items():
        errors.append(dict(input=tag, role=role,
            versus_original_FP32_ONNX=[compare(y, z) for y, z in zip(ys, references[tag])],
            versus_original_Torch=[compare(y, z) for y, z in zip(ys, torchrefs[tag])] if tag in torchrefs else None))
    controls = []
    for rel in ['vggt-v78-smoothcheck-control-r1.json', 'vggt-v78-smoothcompile-control-r1.json',
                'normal-control-r1/control.json', 'profile-control-r1/control.json']:
        path = ROOT / rel; ctl = load_json(path); clocked = 'observations' in ctl
        samples = ctl['observations'] if clocked else ctl['samples']
        hz = 1000000000 if clocked else 300000000
        assert samples and all(x['actual_frequency_hz' if clocked else 'npu_hz'] == hz for x in samples)
        assert all(len(x['temperatures_c']) == 7 and max(x['temperatures_c'].values()) < 75 for x in samples)
        assert ctl['child_returncode' if clocked else 'returncode'] == 0
        assert ctl['restore_errors' if clocked else 'cleanup_errors'] == []
        if clocked:
            assert ctl['frequency_governor_limits_restored'] and ctl['baseline'] == ctl['restored']
            assert ctl['restored']['cur_freq'] == ctl['restored']['userspace/set_freq'] == '300000000'
        else: assert ctl['frequency_written'] is False and ctl['status'] == 'completed'
        controls.append(dict(stage=rel.split('/')[0], sha256=sha(path), samples=len(samples), sampled_hz=hz,
            max_temperature_c=max(max(x['temperatures_c'].values()) for x in samples), child_returncode=0,
            restored_frequency_hz=300000000 if clocked else None, cleanup_errors=[]))
    out = dict(schema='v78_independent_saved_original_pair_smoothing_audit', status='passed_saved_evidence_audit',
        auditor_sha256=sha(Path(__file__)), model_Torch_ORT_SDK_compiler_board_executed_by_auditor=False,
        checkpoint_and_graph_math_quality_accepted=False, arrays_checked=arrays_checked,
        array_files_checked=len(files_checked), shapes_finite_file_raw_SHA_all_passed=True,
        check_result_sha256=sha(ROOT / 'check-r1/result.json'), check_sha256=sha(ROOT / 'check-r1/check.json'),
        compile_sha256=sha(ROOT / 'compile.json'), candidate_model_sha256=compiled['rknn']['sha256'],
        original_FP32_candidate_vs_original_gate=fp32_checks, calibration_norm_arrays_checked=32,
        four_saved_norm_channel_maxima_bitwise_recomputed=True, input_raw_sha256=inputs,
        sources=dict(check=checked['script_sha256'], compiler=compiled['script_sha256'], probe=sha(ROOT / 'probe_smoothed_pair_board.py')),
        stage_audits=stage_reports, normal_warm_route_timings=summaries, first_output_errors=errors,
        profile_raw_and_normal_equivalence=profile_checks, controls=controls,
        limits=['Partial outputs_get failure accounting can record released>acquired; this frozen limitation was not exercised. All actual calls succeeded and buffers balanced.',
                'The second case calib00 is one of eight calibration inputs and not held out; v53 image SHA overlap with calibration is empty.',
                'Component diagnostic only, not a whole-model speed or quality acceptance. Original FP32 gates remain fixed; all actual NPU routes fail them.',
                'Profile timings remain separate from normal; sampled 1GHz does not prove unsampled continuous frequency.',
                'Host audit does not independently reconstruct the 12 derived initializer inverse transforms; the pinned actual source/check receipt enforce these, and private weight graphs are not loaded.'])
    assert checked['v53_image_SHA_overlap_with_calibration'] == []
    assert all(not x['strict_pass'] for row in errors for x in row['versus_original_FP32_ONNX'])
    path = ROOT / 'saved_result_audit.json'
    with path.open('x') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=out['status'], audit_sha256=sha(path), array_files_checked=len(files_checked),
                         arrays_checked=arrays_checked, timings={k: v['mean_ms'] for k, v in summaries.items()},
                         controller_sample_counts=[x['samples'] for x in controls]), ensure_ascii=False))


if __name__ == '__main__':
    main()
