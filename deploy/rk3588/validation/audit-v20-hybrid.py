"""Independent saved-array audit. Never imports Torch, ORT or RKNN."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(actual, expected):
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape and actual.size > 0
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    ref = expected.astype(np.float64)
    diff = actual.astype(np.float64) - ref
    bad = np.abs(diff) > (1e-5 + 1e-4 * np.abs(ref))
    return dict(array_equal=bool(np.array_equal(actual, expected)),
                max_abs_error=float(np.abs(diff).max()), rmse=float(np.sqrt(np.mean(diff**2))),
                relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(ref), 1e-30)),
                outside_tolerance=int(bad.sum()), elementwise_gate_passed=not bool(bad.any()),
                output_all_zero=not bool(np.any(actual)), rtol=1e-4, atol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'profile-dir', 'capture-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    directory, output = a.run_dir, a.output
    report_path = directory / 'run.json'
    r = json.loads(report_path.read_text())
    helper = Path(__file__).with_name('run-v20-aggregator-attention-board.py')
    assert r['block'] == 'global0', 'This audit binds the original global0 experiment'
    assert r['status'] == 'completed' and r['script_sha256'] == sha(helper)
    assert r['source_and_inputs_unchanged'] and r['source_and_input_hashes'] == r['source_and_input_hashes_after']
    assert r['sdpa_restored'] and r['native_streams_restored'] and r['cpu_affinity_restored']
    assert r['workers_joined'] and not r['cleanup_errors']
    assert len(r['forwards']) == 8 and len(r['output_arrays']) == 16
    assert r['round_order'] == [['cpu', 'hybrid'], ['hybrid', 'cpu']]*2
    expected_order = [(i, mode) for i in range(4) for mode in r['round_order'][i]]
    assert [(row['round_index'], row['mode']) for row in r['forwards']] == expected_order
    count = 3 if r['mode'] == 'parallel' else 1
    assert len(r['contexts']) == count
    assert r['npu_run_api_call_count'] == r['successful_synchronous_call_count'] == r['output_release_attempt_count'] == count*4
    for context in r['contexts']:
        assert context['context_destroyed'] and context['destroy_return_code'] == 0
    log = directory/'sdk.log'
    assert not re.search(r'E RKNN|normalize error|unsupported src layout', log.read_text())
    assert r['native_log']['error_line_count'] == 0 and r['native_log']['sha256'] == sha(log)
    profile, capture = a.profile_dir, a.capture_dir
    assert r['profile_sha256'] == sha(profile/'profile.json')
    assert r['capture_sha256'] == sha(capture/'capture.json')
    profile_receipt = json.loads((profile/'profile.json').read_text())
    capture_receipt = json.loads((capture/'capture.json').read_text())
    assert sha(profile/'global0-tensors.npz') == profile_receipt['blocks']['global0']['artifacts']['tensors']['sha256']
    assert sha(capture/'attention.npz') == capture_receipt['artifacts']['attention.npz']['sha256']
    with np.load(profile/'global0-tensors.npz', allow_pickle=False) as source:
        expected_block = source['output'].copy()
    with np.load(capture/'attention.npz', allow_pickle=False) as source:
        expected_attention = source['output'].copy()
    references = {'block': expected_block, 'attention': expected_attention}
    records, values = [], {}
    for entry in r['output_arrays']:
        file = directory/entry['file']
        assert file.stat().st_size == entry['bytes'] and sha(file) == entry['sha256']
        value = np.load(file, allow_pickle=False)
        meta = entry['array']
        raw_sha = hashlib.sha256(memoryview(value).cast('B')).hexdigest()
        assert list(value.shape) == meta['shape'] and str(value.dtype) == meta['dtype']
        assert value.nbytes == meta['bytes'] and raw_sha == meta['sha256']
        key = (entry['mode'], entry['round_index'], entry['role'])
        assert key not in values
        values[key] = value
        result = metrics(value, references[entry['role']])
        forward = next(x for x in r['forwards'] if (x['mode'], x['round_index']) == key[:2])
        actual_receipt = forward['versus_original_CPU_block' if key[2] == 'block' else 'attention_versus_original_CPU_SDPA']
        for name, number in result.items():
            if name in ('rtol', 'atol'):
                continue
            if isinstance(number, float):
                assert math.isclose(number, actual_receipt[name], rel_tol=1e-9, abs_tol=1e-12), (key, name)
            else:
                assert number == actual_receipt[name], (key, name)
        if entry['mode'] == 'cpu':
            assert result['array_equal']
        else:
            assert not result['output_all_zero']
            if entry['role'] == 'attention':
                assert raw_sha == '86890b4947cc4edc675fff4e9cadeb49ac9e25bbf089e7aea0565473b2cae403'
        records.append({**entry, 'independently_recomputed_metrics': result})
    for mode in ('cpu', 'hybrid'):
        for role in ('block', 'attention'):
            assert all(np.array_equal(values[(mode, 0, role)], values[(mode, i, role)]) for i in range(4))
        rows = [x for x in r['forwards'] if x['mode'] == mode]
        for field in ('whole_block_seconds', 'attention_seconds'):
            assert all(math.isfinite(x[field]) and x[field] > 0 for x in rows)
        assert math.isclose(np.mean([x['whole_block_seconds'] for x in rows[1:]]),
                            r[mode+'_timing']['subsequent_mean_seconds'], rel_tol=1e-12)
        assert math.isclose(np.mean([x['attention_seconds'] for x in rows[1:]]),
                            r[mode+'_timing']['attention_subsequent_mean_seconds'], rel_tol=1e-12)
    for row in r['forwards']:
        assert row['sdpa_calls'] == 1
        if row['mode'] == 'hybrid':
            native = row['native_attention']
            assert len(native['calls']) == native['completed_synchronous_calls'] == count
            for call in native['calls']:
                assert call['run_api_succeeded'] and call['output_buffer_released'] and call['outputs_release_code'] == 0
                assert call['output_copied_before_release'] and call['runtime_frame_id'] == call['received_frame_id']
    summary = dict(kind='independent_saved_array_audit_original_global_block_with_NPU_attention',
        status='verified_execution_and_evidence_consistency_quality_gate_separate',
        run_name=directory.name, run_receipt_sha256=sha(report_path), script_sha256=r['script_sha256'],
        analysis_script_sha256=sha(Path(__file__)), native_log_sha256=sha(log),
        mode=r['mode'], profile_sha256=r['profile_sha256'], capture_sha256=r['capture_sha256'],
        original_checkpoint_sha256=r['checkpoint']['sha256'], original_block_shape=r['block_shape'],
        source_and_input_hashes_unchanged=True, original_weights=True, architecture_changed=False,
        contexts=r['core_masks'], original_block_parameter_metadata=r['block_parameter_metadata'],
        normal_benchmark_timing=r['normal_benchmark_timing'],
        cpu_timing=r['cpu_timing'], hybrid_timing=r['hybrid_timing'], forwards=r['forwards'],
        outputs=records, context_destroy_return_codes=[c['destroy_return_code'] for c in r['contexts']],
        successful_synchronous_call_count=r['successful_synchronous_call_count'],
        output_release_attempt_count=r['output_release_attempt_count'],
        CPU_outputs_bitwise_equal_original_reference=True,
        hybrid_attention_bitwise_equal_v19_standalone_NPU=True,
        within_mode_outputs_bitwise_stable=True,
        hybrid_strict_FP32_block_gate_all_rounds_passed=r['hybrid_strict_FP32_block_gate_all_rounds_passed'],
        SDPA_workers_native_streams_affinity_contexts_restored=True,
        endpoint=r['whole_block_endpoint'], timing_excludes=r['timing_excludes'], timing_limits=r['timing_limits'],
        full_model_run=False, realtime_validated=False, whole_model_quality_validated=False,
        ORB_used=False, inference_on_board_only=True, host_model_inference_executed=False)
    assert not output.exists()
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': summary['status'], 'cpu':r['cpu_timing'], 'hybrid':r['hybrid_timing'],
                      'FP32_gate':summary['hybrid_strict_FP32_block_gate_all_rounds_passed'],
                      'output_sha256':sha(output)}, allow_nan=False))


if __name__ == '__main__':
    main()
