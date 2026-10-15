#!/usr/bin/env python3
"""Independent NumPy-only audit of saved FP32 pair endpoints; no NN/SDK."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CHECKER_SHA = 'b0f4bd489d1e6ec8eb28869f609d7e48469aa9d19b2120d7906cd9284059bc4c'
CAPTURE_SHA = 'ed500643ef333dd824ccab81a126e87ff8616c3cb3ee09e3ec59f2a4b7bbf7e9'
CAPTURES = ['cc2dd7fbddef8ee12de00b7a5cc88a16ba9d531783ab09ae8e66abe50df43408',
            '34ee3b510014f55c34525c6a96a3527603ba6350c896fae7498ae5493987df69',
            '5560b34755bfbca54cc179c119991792b63d6135e839ad15ac8db95e16965660']
CHECKS = ['70caf6de746831cf7950fb4971359a1341d2dd4d845f3f3eb5a1da0e1b32072c',
          'b975a0eb82fa3d2ce275a8e487aed583e37bfc16bfb2afc5b6e42f7fc0402618',
          '17f9aec6d7c93189c7ba74c33c3101160c4170f0a38e2b6ae91d77eeb4fdc250']
EXPORT_SHA = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
ORIGINAL_CHECK_SHA = '48a7d5034b9eea0afa139e4a69cdb5b393c694dcec2b8e874a1c1f380f5f42b1'
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def raw(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def read_json(path, expected):
    require(sha(path) == expected, 'Receipt hash differs: ' + str(path))
    return json.loads(path.read_text())


def array(path, row, shape):
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'NPY file hash/size differs')
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float32 and row['dtype'] == 'float32' and list(value.shape) == row['shape'] == shape and
            value.flags.c_contiguous and np.isfinite(value).all() and row['finite'] and
            value.nbytes == row['tensor_bytes'] and raw(value) == row['raw_sha256'], 'Complete logical FP32 NPY differs')
    return value


def compare(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64); delta = a - b
    norm = float(np.linalg.norm(b.ravel()))
    relative = float(np.linalg.norm(delta.ravel())) / norm if norm > 0 else None
    maximum = float(np.abs(delta).max())
    outside = int(np.count_nonzero(np.abs(delta) > GATE['atol'] + GATE['rtol'] * np.abs(b)))
    numeric = relative is not None and relative <= GATE['relative_l2_max'] and maximum <= GATE['max_abs_max']
    return dict(relative_l2=relative, reference_norm=norm, max_abs=maximum, outside_count=outside,
                strict_pass=outside == 0, byte_equal=actual.tobytes(order='C') == reference.tobytes(order='C'),
                fixed_numeric_gate_passed=numeric, fixed_gate_passed=numeric, zero_reference_norm=norm == 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); root = a.root.resolve(); output = a.output or root / 'saved-pair-FP32-audit.json'
    require(sha(root / 'check_captured_pair_fp32_board.py') == CHECKER_SHA and
            sha(root / 'capture_frame0_inputs_board.py') == CAPTURE_SHA, 'Executed sources differ')
    batches, cases, independent_numeric = [], [], True
    for batch in range(3):
        capdir, directory = root / ('capture-batch%d-r1' % batch), root / ('fp32-batch%d-r1' % batch)
        capture = read_json(capdir / 'result.json', CAPTURES[batch])
        r = read_json(directory / 'result.json', CHECKS[batch])
        require(r['schema'] == 'v128_original_native518_captured_pair_FP32_check' and
                r['status'] == 'checked_captured_pair_fixed_FP32_gate_passed' and r['script_sha256'] == CHECKER_SHA and
                r['capture_receipt']['sha256'] == CAPTURES[batch] and r['capture_receipt']['script_sha256'] == CAPTURE_SHA and
                r['records'] == capture['records'] and r['batch_index'] == batch and r['fixed_gate'] == GATE and
                r['strict_allclose_is_diagnostic'], 'Actual original same-input FP32 gate contract differs')
        require(r['export_receipt']['sha256'] == EXPORT_SHA and r['original_check_receipt']['sha256'] == ORIGINAL_CHECK_SHA and
                r['source_hashes'] == capture['source_hashes'] and r['capture_completion_and_cleanup_verified'] and
                r['all36_original_learned_values_verified'] and all(len(v) == 18 for v in r['original_parameters'].values()),
                'Original graph/source/parameter lineage differs')
        require(r['all6_calls_completed'] and r['FP32_gate_passed'] and r['canonical_outputs_restored'] and
                r['original_numeric_gate_unchanged'] and r['sessions_created'] == r['sessions_dropped'] == 6 and
                r['ORT_sessions_dropped'] and r['saved_array_count'] == 12 and r['cleanup_errors'] == [] and
                r['affinity_restored'] and r['environment_restored'] and r['source_and_inputs_unchanged'] and
                r['execution_provider'] == 'CPUExecutionProvider' and r['maximum_simultaneous_ORT_sessions'] == 1 and
                r['intra_op_num_threads'] == r['inter_op_num_threads'] == 1, 'Actual completed CPU sessions/cleanup differ')
        require([(row['input_id'], row['variant']) for row in r['checks']] ==
                [(record['id'], v) for record in capture['records'] for v in ('A', 'F', 'B')], 'Six ordered original CPU calls required')
        file_names, byte_count = [], 0
        for record in capture['records']:
            input_row = record['input']; x = array(capdir / input_row['file'], input_row, FRAME)
            baseline, actual_case = None, dict(id=record['id'], split=record['split'], input_sha256=input_row['sha256'],
                input_raw_sha256=raw(x), source_pair_indices=record['source_pair_indices'], variants=[])
            for row in (v for v in r['checks'] if v['input_id'] == record['id']):
                require(row['complete'] and row['input_unchanged'] and row['source_pair_indices'] == record['source_pair_indices'] and
                        row['split'] == record['split'] and len(row['outputs']) == 2 and row['CPU_forward_seconds'] > 0,
                        'Complete same-input endpoint call required')
                values = []
                for artifact, name, shape in zip(row['outputs'], ('frame_output', 'global_output'), (FRAME, GLOBAL)):
                    require(artifact['name'] == name and Path(artifact['file']).name == artifact['file'], 'Canonical endpoint name/path differs')
                    values.append(array(directory / artifact['file'], artifact, shape))
                    file_names.append(artifact['file']); byte_count += artifact['bytes']
                if row['variant'] == 'A':
                    baseline = values
                metrics = {name: compare(value, expected) for name, value, expected in
                           zip(('frame_output', 'global_output'), values, baseline)}
                require(set(row['comparisons_vs_A']) == set(metrics), 'Original comparison set differs')
                for name, value in metrics.items():
                    saved = row['comparisons_vs_A'][name]
                    for key in ('relative_l2', 'reference_norm', 'max_abs'):
                        require(value[key] is not None and np.isclose(value[key], saved[key], rtol=1e-10, atol=1e-13), 'Reported numerical metric differs')
                    for key in ('outside_count', 'strict_pass', 'byte_equal', 'fixed_numeric_gate_passed', 'fixed_gate_passed', 'zero_reference_norm'):
                        require(value[key] == saved[key], 'Reported diagnostic/gate differs')
                numeric = all(v['fixed_gate_passed'] for v in metrics.values())
                require(numeric == row['fixed_gate_passed'], 'Per-call hard gate differs'); independent_numeric &= numeric
                actual_case['variants'].append(dict(variant=row['variant'], outputs=row['outputs'], independently_recomputed=metrics))
            cases.append(actual_case)
        require(len(set(file_names)) == 12 and set(file_names) == {p.name for p in directory.glob('*.npy')} and
                not list(directory.glob('*.tmp')) and byte_count == r['saved_array_bytes'], 'Missing, duplicate or unlisted endpoint payloads')
        batches.append(dict(batch=batch, receipt_sha256=CHECKS[batch], capture_sha256=CAPTURES[batch],
                            saved_array_count=12, saved_array_bytes=byte_count, six_CPU_sessions_references_dropped=True))
    require(len(cases) == 6 and len({v['input_raw_sha256'] for v in cases}) == 6 and independent_numeric, 'Distinct six inputs/hard numerical gate failed')
    report = dict(schema='v128_independent_saved_pair_FP32_audit', status='saved_FP32_endpoints_and_original_numeric_gate_verified',
        auditor_source_sha256=sha(__file__), checker_source_sha256=CHECKER_SHA, capture_source_sha256=CAPTURE_SHA,
        batches=batches, cases=cases, actual_saved_output_NPY_count=36, complete_CPU_pair_call_count=18,
        fixed_gate=GATE, fixed_numeric_gate_passed_independently_recomputed=independent_numeric,
        strict_allclose_is_independent_diagnostic=True, same_captured_input_A_reference_per_case=True,
        zero_denominator_rule='unavailable, never add a floor', candidate_accepted=False,
        limitations=['Independent stored-file/raw/shape/finiteness and same-case A comparisons only; no NN/SDK/board or graph execution.',
                     'A is the pinned original two-block CPU ONNX reference on actual NPU DINO prefix inputs; not matched complete CPU Torch predictions or trajectory quality.',
                     'Frame-to-global continuity is established by the pinned single-graph source and proof; no intermediate recomputation or teacher inputs.',
                     'CPU forward times are diagnostic and not NPU, complete-model, realtime or heldout deployment acceptance.'])
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temp.replace(output)
    print(json.dumps(dict(file=str(output), sha256=sha(output), output_NPY_count=36, fixed_numeric_gate_passed=independent_numeric)))


if __name__ == '__main__':
    main()
