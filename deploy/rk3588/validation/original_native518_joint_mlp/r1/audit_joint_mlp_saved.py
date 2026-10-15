#!/usr/bin/env python3
"""Independent saved NumPy/JSON audit of R129; no ONNX, NN, SDK or board calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SOURCE_SHA = '82f35353bf57b662b95b89cf5472e18e50e832519e8545ab7237994e35e06c07'
EXPORT_SHA = 'f8f935e43db97ffb5cee683d8eee2a8fe4ceb728a96f10c37ad8b6772a76ea49'
CHECK_SHAS = ['4df223f1fcf6808b3642eb5015ad2e0515dc1f1df1fd5f6966baada0e91a29b7',
    '177f8cd85c3306d23effb2e68358006e817e8f001b2a7030bb1ea10baa7f1566',
    '666908dfbf47c6f9641f1d3d74588bb1abe5298e09dc83dc2205f209bd516490']
CAP_SHAS = ['cc2dd7fbddef8ee12de00b7a5cc88a16ba9d531783ab09ae8e66abe50df43408',
    '34ee3b510014f55c34525c6a96a3527603ba6350c896fae7498ae5493987df69',
    '5560b34755bfbca54cc179c119991792b63d6135e839ad15ac8db95e16965660']
REF_SHAS = ['70caf6de746831cf7950fb4971359a1341d2dd4d845f3f3eb5a1da0e1b32072c',
    'b975a0eb82fa3d2ce275a8e487aed583e37bfc16bfb2afc5b6e42f7fc0402618',
    '17f9aec6d7c93189c7ba74c33c3101160c4170f0a38e2b6ae91d77eeb4fdc250']
PARENT_EXPORT_SHA = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
PARENT_GRAPH_SHA = '57b60feceea0edb57ca4cea38200d6ea47d4e4ee51e4a546357267cbc9f676bb'
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
SHAPES = [[2, 1041, 1024], [1, 2082, 1024]]
NAMES = ['frame_output', 'global_output']


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''): h.update(part)
    return h.hexdigest()


def receipt(path, digest):
    require(sha(path) == digest, 'Receipt/source SHA mismatch: ' + Path(path).name)
    return json.loads(Path(path).read_text())


def array(folder, row, shape):
    name = Path(row['file']); require(not name.is_absolute() and '..' not in name.parts, 'Relative artifact required')
    path = folder / name
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Array file identity differs')
    a = np.load(path, allow_pickle=False)
    require(a.dtype == np.float32 and list(a.shape) == shape == row['shape'] and row['dtype'] == 'float32' and
        a.flags.c_contiguous and np.isfinite(a).all() and row['finite'] and a.nbytes == row['tensor_bytes'] and
        hashlib.sha256(a.tobytes(order='C')).hexdigest() == row['raw_sha256'], 'Array raw/dtype/shape/finite differs')
    return a


def compare(a, b):
    x, y = a.astype(np.float64), b.astype(np.float64); d = x - y
    rn = float(np.linalg.norm(y)); rel = float(np.linalg.norm(d)) / rn if rn else None
    maximum = float(np.abs(d).max()); outside = int(np.count_nonzero(np.abs(d) > GATE['atol'] + GATE['rtol'] * np.abs(y)))
    passed = rel is not None and rel <= GATE['relative_l2_max'] and maximum <= GATE['max_abs_max']
    return dict(relative_l2=rel, reference_norm=rn, zero_reference_norm=rn == 0, max_abs=maximum,
        outside_count=outside, strict_pass=outside == 0, byte_equal=a.tobytes(order='C') == b.tobytes(order='C'),
        fixed_numeric_gate_passed=passed, fixed_gate_passed=passed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--parent-export', type=Path, default=Path('/path/to/research-artifacts/v126-basis-recovery-work/export-r2/result.json'))
    p.add_argument('--output', type=Path)
    a = p.parse_args(); w = a.workdir; parent = w.parent
    authority = receipt(a.parent_export, PARENT_EXPORT_SHA)
    require(sha(w / 'export_joint_mlp_board.py') == SOURCE_SHA, 'Executed source identity differs')
    exported = receipt(w / 'export-r1/result.json', EXPORT_SHA); proof = exported['proof']
    require(exported['schema'] == 'v129_original_native518_joint_MLP_pair_ONNX_export' and exported['script_sha256'] == SOURCE_SHA and
        exported['status'] == 'exported_original_joint_MLP_pair_not_FP32_or_NPU_accepted' and exported['parent_export_receipt']['sha256'] == PARENT_EXPORT_SHA and
        exported['parent_graph']['sha256'] == proof['inverse_parent_ModelProto_sha256'] == PARENT_GRAPH_SHA and
        exported['original_parameters'] == authority['original_parameters'] and exported['source_hashes'] == authority['source_hashes'], 'Original export lineage differs')
    flags = ['all4096_original_hidden_neurons_covered_once', 'original_FC1_weight_columns_and_bias_reconstructed_bitwise',
        'original_FC2_K4_weights_protobuf_unchanged', 'all36_original_learned_values_recovered_and_verified',
        'all_frame0_nodes_and_initializers_protobuf_unchanged', 'all_untouched_nodes_and_initializers_protobuf_unchanged',
        'original_three_partial_sum_Add_nodes_and_one_bias_Add_node_protobuf_unchanged', 'original_GELU_operator_and_operand_order_per_branch_unchanged',
        'original_canonical_IO_protobuf_unchanged', 'whole_parent_inverse_ModelProto_byte_equal']
    require(all(proof[k] is True for k in flags) and proof['candidate_nodes'] == proof['original_nodes'] - proof['removed_node_count'] + proof['added_node_count'], 'Incomplete executed parameter/inverse proof')
    original_params = authority['original_parameters']; verified = proof['original_36_learned_values_verified']
    require(set(verified) == {'frame', 'global'} and all(len(verified[k]) == len(original_params[k]) == 18 for k in verified), '36 original parameter identities required')
    for role, rows in verified.items():
        for name, row in rows.items():
            require(row['finite'] and all(row[k] == original_params[role][name][k] for k in ('shape', 'dtype', 'tensor_bytes', 'raw_sha256')), 'Parameter proof metadata differs')
    expected_order = []
    for i in range(4):
        require(proof['branches'][i]['index'] == i and proof['branches'][i]['original_hidden_channel_range'] == [i * 1024, (i + 1) * 1024], 'Hidden channels incomplete/overlapping')
        prefix = 'global0/joint_mlp_k4/part' + str(i) + '/'
        expected_order.extend(prefix + s for s in ('fc1_matmul', 'fc1', 'gelu_scaled_input', 'gelu_erf', 'gelu_erf_plus_one', 'gelu_before_half', 'gelu'))
        expected_order.append('global0/mlp.fc2.matmul.part' + str(i))
        if i: expected_order.append('global0/mlp.fc2.matmul.sum' + str(i))
    expected_order.append('global0/mlp.fc2')
    require(proof['explicit_source_consumer_chain_order'] == expected_order and proof['source_topological_interleaving_is_not_an_enforced_SDK_schedule'] and
        not proof['actual_SDK_schedule_or_memory_lifetime_validated'] and not proof['dense_MACs_reduced'], 'Source-only scheduling scope differs')
    model = exported['models'][0]; graph = w / 'export-r1' / model['onnx']['file']
    require(len(exported['models']) == 1 and graph.stat().st_size == model['onnx']['bytes'] and sha(graph) == model['onnx']['sha256'], 'Candidate graph bytes differ')
    rows, controls = [], []
    for batch in range(3):
        cap = receipt(parent / f'capture-batch{batch}-r1/result.json', CAP_SHAS[batch])
        ref = receipt(parent / f'fp32-batch{batch}-r1/result.json', REF_SHAS[batch])
        check = receipt(w / f'check{batch}-r1/result.json', CHECK_SHAS[batch])
        require(check['schema'] == 'v129_original_native518_joint_MLP_pair_FP32_check' and check['script_sha256'] == SOURCE_SHA and
            check['status'] == 'checked_original_joint_MLP_pair_fixed_FP32_gate_passed_not_NPU_accepted' and check['fixed_gate'] == GATE and
            check['strict_allclose_is_independent_diagnostic'] and check['batch_index'] == batch and check['models'] == exported['models'] and check['proof'] == proof and
            check['original_parameters'] == original_params and check['export_receipt']['sha256'] == EXPORT_SHA and
            check['capture_receipt']['sha256'] == CAP_SHAS[batch] and check['original_FP32_reference_receipt']['sha256'] == REF_SHAS[batch] and
            check['all2_calls_completed'] and check['FP32_gate_passed'] and check['ORT_sessions_created'] == check['ORT_sessions_dropped_count'] == 2,
            'Complete same-source/matched-input two-call check required')
        require(ref['records'] == cap['records'] and ref['original_parameters'] == original_params and ref['source_hashes'] == check['source_hashes'] == cap['source_hashes'] and
            [r['input_id'] for r in check['checks']] == [r['id'] for r in cap['records']], 'Same actual inputs/source lineage differs')
        files = set()
        for call, record in zip(check['checks'], cap['records']):
            base = next(r for r in ref['checks'] if r['input_id'] == record['id'] and r['variant'] == 'A')
            require(call['complete'] and call['input_unchanged'] and call['fixed_gate_passed'] and call['split'] == record['split'] and
                call['source_pair_indices'] == record['source_pair_indices'] and len(call['outputs']) == len(base['outputs']) == 2 and
                [r['name'] for r in call['outputs']] == [r['name'] for r in base['outputs']] == NAMES, 'Same-input complete canonical endpoints required')
            endpoints = []
            for outrow, refrow, shape in zip(call['outputs'], base['outputs'], SHAPES):
                x = array(w / f'check{batch}-r1', outrow, shape); y = array(parent / f'fp32-batch{batch}-r1', refrow, shape)
                actual = compare(x, y); reported = call['comparisons_vs_original_A'][outrow['name']]
                matches = [math.isclose(value, reported[k], rel_tol=1e-11, abs_tol=1e-12) if isinstance(value, float) else value == reported[k] for k, value in actual.items()]
                require(all(matches) and actual['fixed_gate_passed'], 'Independent saved FP32 comparison/report differs')
                files.add(outrow['file']); endpoints.append(dict(artifact=outrow, original_A_reference=dict(file=refrow['file'], sha256=refrow['sha256'], raw_sha256=refrow['raw_sha256']), comparison=actual))
            rows.append(dict(input_id=record['id'], split=record['split'], source_pair_indices=record['source_pair_indices'],
                input_file_sha256=record['input']['sha256'], input_raw_sha256=record['input']['raw_sha256'], endpoints=endpoints))
        require({p.name for p in (w / f'check{batch}-r1').glob('*.npy')} == files, 'Missing/extra output NPY')
    for stage in ('export', 'check0', 'check1', 'check2'):
        path = w / f'vggt-v83-native518joint129{stage}r1-control-r1.json'; control = json.loads(path.read_text()); group = control['process_group_cleanup']
        require(control['status'] == 'completed' and control['returncode'] == 0 and not control['cleanup_errors'] and control['frequency_written'] is False and
            group['live_group_gone'] and group['leader_reaped'] and not group['remaining_live_members'] and not group['errors'], 'CPU controller cleanup failed')
        samples = control['samples']; temperatures = [v for s in samples for v in s['temperatures_c'].values()]
        require(samples and all(s['npu_hz'] == control['expected_observed_npu_hz'] for s in samples) and temperatures and max(temperatures) < 75, 'Recorded thermal/frequency guard differs')
        controls.append(dict(stage=stage, receipt_sha256=sha(path), returncode=0, seconds=control['seconds'],
            max_observed_temperature_c=max(temperatures), observed_npu_hz=control['expected_observed_npu_hz'], frequency_written=False,
            frequency_restore_scope='Not applicable: CPU controller observes unchanged NPU frequency; no clock writes.', process_group_gone=True, leader_reaped=True))
    for r in [exported] + [json.loads((w / f'check{i}-r1/result.json').read_text()) for i in range(3)]:
        require(not r['cleanup_errors'] and all(r[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'ORT_sessions_dropped')), 'Executed source/session cleanup incomplete')
    report = dict(schema='v129_joint_MLP_independent_saved_FP32_audit', status='passed_saved_FP32_component_audit_not_NPU_or_quality_accepted',
        auditor_sha256=sha(__file__), executed_source_sha256=SOURCE_SHA, export_receipt_sha256=EXPORT_SHA, check_receipt_sha256=CHECK_SHAS,
        parent_export_sha256=PARENT_EXPORT_SHA, candidate_graph=model['onnx'], fixed_numeric_gate=GATE,
        strict_allclose_is_independent_diagnostic=True, original_parameter_value_count=36, original_parameter_proof_metadata_matches_parent=True,
        proof_scope='Executed exporter proof and original 36 metadata identities checked; inverse/weight operations not rerun by this saved-only audit.',
        graph_counts={k: proof[k] for k in ('original_nodes', 'candidate_nodes', 'removed_node_count', 'added_node_count', 'original_initializers', 'candidate_initializers')},
        source_consumer_chain_order_verified=True, source_consumer_chain_order=expected_order,
        liveness_hypothesis=dict(full_hidden_width=4096, branch_hidden_width=1024, full_FP32_hidden_tensor_bytes=2082 * 4096 * 4,
            one_FP32_branch_tensor_bytes=2082 * 1024 * 4, global_MLP_dense_MACs_unchanged=2082 * (1024 * 4096 + 4096 * 1024),
            compiled_schedule_memory_lifetime_or_speed_established=False,
            scope='Four ordered FC1/Erf/FC2 source chains can consume a 1024-wide branch without a full 4096-wide hidden node. Compiler may reorder/retain all branches; this is no physical-memory or speed proof.'),
        cases=rows, controls=controls, audited_output_count=12, same_input_reference_output_count=12,
        all12_outputs_byte_equal_original_A=all(e['comparison']['byte_equal'] for r in rows for e in r['endpoints']),
        all12_numeric_FP32_gates_passed=True, ORT_sessions_created=6, ORT_sessions_dropped=6,
        NN_SDK_ONNX_or_board_operations_by_auditor=False, performance_or_trajectory_quality_or_realtime_accepted=False)
    output = a.output or w / 'joint-MLP-FP32-saved-audit.json'
    with output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(output=output.name, bytes=output.stat().st_size, sha256=sha(output), all12_byte_equal=report['all12_outputs_byte_equal_original_A'])))


if __name__ == '__main__': main()
