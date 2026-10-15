#!/usr/bin/env python3
"""Saved-only original DPT basis audit; no NN, ONNX or SDK imports/execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

WORKER_SHA = '57d888a469581af4c41af58ee858835ba216c98e9c26154a014812e8aeb9bdf7'
EXPORT_SHA = '0040819ec2418235f4f77b49f13943c53aa8a2704ef403bfb63d6b0a8ce6623f'
CHECK_SHAS = ['f9df012758e6f59c6972ef19b3579cd7e36bdf93dac4eb0c8ebed5649dc998d8',
              '53e2362dbac6ad6bddfde7ac82c754afa4a98c7c26de7a9689505a7f3e287e49']
PARENT_EXPORT_SHA = '8e3f465242924445348a589d5b589e4176fabde9e272ad0fe7d49950886f6033'
CAPTURE_SHA = '969ab6f5611efa62d81b5d62625a15815248668e1e079db17ac4d9ed9c96932b'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
SOURCE_PINS = {
    'run_sampled_cpu_stage_r126.py': '700ec5b12b6f15cf9907cfacbff3a63258ef70a14cf5efe06ba5470991de1b89',
    'execute_depth_basis_host.py': '8c8639485b0822ac062df09d84d96e9b8699b0dd7c4f42c3a8b5b3ff3ea1f7b5',
    'verify_generated_basis_consumer_algebra.py': 'cdf7e3032aaa8153e4f3ea50f5a61f595d7376eed6bb1ae5140027cbb5fbdc2c',
    'generated-basis-consumer-algebra.json': 'e31ff44070f9ae63a6dcb7a3630dde85ea4526a2979f0fd9b67ab903bbf13ddf',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path, expected=None):
    digest = sha(path)
    require(expected is None or expected == digest, 'Receipt/source identity differs: ' + path.name)
    return json.loads(path.read_text()), dict(file=path.name, bytes=path.stat().st_size, sha256=digest)


def comparison(x, y):
    dx, dy = x.astype(np.float64), y.astype(np.float64)
    delta = dx - dy
    norm = float(np.linalg.norm(dy.reshape(-1)))
    relative = float(np.linalg.norm(delta.reshape(-1)) / norm) if norm > 0 else None
    maximum = float(np.max(np.abs(delta)))
    outside = int(np.count_nonzero(np.abs(delta) > GATE['atol'] + GATE['rtol'] * np.abs(dy)))
    passed = relative is not None and relative <= GATE['relative_l2_max'] and maximum <= GATE['max_abs_max']
    return dict(relative_l2=relative, reference_norm=norm, zero_reference_norm=norm == 0, max_abs=maximum,
                outside_count=outside, strict_pass=outside == 0, fixed_numeric_gate_passed=passed,
                fixed_gate_passed=passed, byte_equal=x.tobytes() == y.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--original-metadata-dir', type=Path, required=True)
    a = p.parse_args()
    w, d = a.workdir, a.original_metadata_dir
    require(sha(w / 'export_depth_basis_board.py') == WORKER_SHA, 'Executed worker source differs')
    original, original_id = load(d / 'depth-export.json', PARENT_EXPORT_SHA)
    capture, capture_id = load(d / 'head-reference.json', CAPTURE_SHA)
    exported, export_id = load(w / 'depth-basis-export-r1/result.json', EXPORT_SHA)
    for row in [exported]:
        require(row['status'] == 'exported_original_DPT_basis_consumer_not_numerically_accepted', 'Completed export required')
    require(original['checkpoint'] == capture['checkpoint'] == exported['checkpoint'] == CHECKPOINT and
            len(original['original_parameter_metadata']) == 62 and
            exported['original_parameter_metadata'] == original['original_parameter_metadata'] == capture['depth_original_parameter_metadata'] and
            exported['source_hashes'] == original['source_hashes'] == capture['source_hashes'], 'Original checkpoint/source/62-value lineage differs')
    require(exported['parent_export']['sha256'] == PARENT_EXPORT_SHA and exported['head_reference']['sha256'] == CAPTURE_SHA and
            exported['parent_graph']['sha256'] == original['graph']['sha256'] and
            original['native_reference_chunks'] == capture['depth_chunks'], 'Pinned parent/reference linkage differs')
    changed = ['norm.weight', 'norm.bias'] + [f'projects.{i}.{kind}' for i in range(4) for kind in ('weight', 'bias')]
    proof = exported['proof']
    require(proof['changed_initializer_names'] == changed and proof['node_count'] == 79 and proof['initializer_count'] == 76 and
            proof['inverse_model_sha256'] == original['graph']['sha256'] and all(proof[key] is True for key in
                ('all79_original_node_protos_byte_identical', 'all66_untouched_initializer_protos_byte_identical',
                 'all52_untouched_original_learned_tensors_byte_identical', 'all5_original_positions_byte_identical',
                 'all_original_resize_relu_alias_and_fusion_nodes_byte_identical', 'original_initializer_order_preserved',
                 'inverse_entire_model_proto_byte_equal')), 'Executed graph proof differs')
    graph = w / 'depth-basis-export-r1' / exported['graph']['file']
    require(graph.stat().st_size == exported['graph']['bytes'] == 246272013 and sha(graph) == exported['graph']['sha256'],
            'Saved complete derived graph bytes differ')
    report = dict(schema='v130_original_native518_DPT_basis_saved_audit',
        status='saved_raw_logits_verified_not_deployment_or_quality_acceptance',
        script_sha256=sha(Path(__file__)), worker_sha256=WORKER_SHA, host_saved_numpy_only=True,
        NN_ONNX_Torch_SDK_GPU_or_board_execution=False, model_weights_or_graph_protos_deserialized=False,
        checkpoint_metadata_lineage=CHECKPOINT, checkpoint_rehashed_or_deserialized_by_auditor=False,
        original_parent_receipt=original_id, original_reference_receipt=capture_id, export_receipt=export_id,
        graph=dict(bytes=graph.stat().st_size, sha256=sha(graph)), executed_board_graph_proof=proof,
        graph_proof_scope='Verified frozen executed-source and board proof linkage; graph file hash checked without host protobuf/weight deserialization',
        executed_controller_transport_and_generated_proof_pins=SOURCE_PINS,
        original62_parameter_metadata_matched=True, fixed_gate=GATE, strict_allclose_is_independent_diagnostic=True,
        frames=[], controllers=[], files_verified=0, original_activation_not_executed_by_auditor=True,
        NPU_performance_full_model_trajectory_realtime_or_8GB_accepted=False)
    for stage in ('export', 'check0', 'check1'):
        control, cid = load(w / f'vggt-v83-native518depthbasis130{stage}r1-control-r1.json')
        require(control['status'] == 'completed' and control['returncode'] == 0 and not control['frequency_written'] and
                not control['cleanup_errors'] and control['process_group_cleanup']['live_group_gone'] and
                control['expected_observed_npu_hz'] == 1000000000 and control['timeout_seconds'] == 600 and
                all(v['npu_hz'] == 1000000000 for v in control['samples']), 'Original CPU controller or observed clock differs')
        maximum = max(max(v['temperatures_c'].values()) for v in control['samples'])
        require(maximum < 75, 'CPU stage crossed hard thermal limit')
        require(sha(w / f'depth-basis-{stage}-r1-source/export_depth_basis_board.py') == WORKER_SHA, 'Staged executed source differs')
        require(all(sha(w / f'depth-basis-{stage}-r1-source' / name) == digest for name, digest in SOURCE_PINS.items()),
                'Staged controller/transport/generated proof source differs')
        report['controllers'].append(dict(stage=stage, **cid, seconds=control['seconds'], maximum_temperature_c=maximum,
            frequency_written=False, observed_npu_hz=1000000000, process_group_gone=True, cleanup_errors=[]))
    for frame in (0, 1):
        folder = w / f'depth-basis-check{frame}-r1'
        row, rid = load(folder / 'result.json', CHECK_SHAS[frame])
        require(row['schema'] == 'v130_original_native518_DPT_basis_consumer_check' and
                row['status'] == 'checked_original_DPT_basis_consumer_fixed_FP32_gate_passed' and row['frame_index'] == frame and
                row['script_sha256'] == WORKER_SHA and row['fixed_gate'] == GATE and row['proof'] == proof and
                row['graph'] == exported['graph'] and row['export_receipt']['sha256'] == EXPORT_SHA and
                row['sessions_created'] == row['sessions_dropped'] == row['calls_completed'] == 2 and
                row['source_and_inputs_unchanged'] and row['affinity_restored'] and row['environment_restored'] and
                row['ORT_sessions_dropped'] and not row['cleanup_errors'], 'Original frame/check/cleanup linkage differs')
        chunk = capture['depth_chunks'][frame]
        require(chunk['frame_index'] == frame and row['input_artifacts'] == chunk['inputs'] and
                row['original_parameter_metadata'] == original['original_parameter_metadata'], 'Same original frame input identity differs')
        arrays, artifacts = {}, []
        for output in row['outputs']:
            path = folder / output['file']
            require(path.parent == folder and path.stat().st_size == output['bytes'] and sha(path) == output['sha256'], 'Saved array file identity differs')
            value = np.load(path, allow_pickle=False)
            raw = hashlib.sha256(value.tobytes()).hexdigest()
            require(list(value.shape) == output['shape'] == [1, 2, 392, 518] and value.dtype == np.float32 and
                    value.flags.c_contiguous and value.nbytes == output['tensor_bytes'] == 1624448 and
                    raw == output['raw_sha256'] and bool(np.isfinite(value).all()) == output['finite'] is True,
                    'Saved raw-logit shape/dtype/bytes/finiteness differs')
            require(output['route'] not in arrays, 'Duplicate output route')
            arrays[output['route']] = value
            artifacts.append({k: output[k] for k in ('route', 'file', 'bytes', 'sha256', 'raw_sha256', 'shape', 'dtype', 'finite')})
            report['files_verified'] += 1
        require(set(arrays) == {'original_reference', 'baseline', 'folded'}, 'All three original raw-logit routes required')
        ref = next(v for v in row['outputs'] if v['route'] == 'original_reference')
        require(all(ref[k] == chunk['raw_logits'][k] for k in ('bytes', 'sha256', 'raw_sha256', 'shape', 'dtype', 'tensor_bytes', 'finite')),
                'Original captured Torch reference differs')
        comparisons = dict(folded_vs_baseline=comparison(arrays['folded'], arrays['baseline']),
            baseline_vs_original_Torch_capture=comparison(arrays['baseline'], arrays['original_reference']),
            folded_vs_original_Torch_capture=comparison(arrays['folded'], arrays['original_reference']))
        for key, recomputed in comparisons.items():
            saved = row['comparisons'][key]
            for name, value in recomputed.items():
                require((np.isclose(value, saved[name], rtol=1e-12, atol=1e-15) if isinstance(value, float) else value == saved[name]),
                        'Saved metric does not reproduce: ' + key + '/' + name)
        require(comparisons['folded_vs_baseline']['fixed_numeric_gate_passed'] == row['FP32_gate_passed'] is True,
                'Original numeric consumer gate differs')
        report['frames'].append(dict(frame_index=frame, check_receipt=rid, artifacts=artifacts, comparisons=comparisons,
            original_input_raw_sha256s={k: v['raw_sha256'] for k, v in chunk['inputs'].items()},
            main_numeric_gate_passed=True, main_strict_diagnostic_passed=comparisons['folded_vs_baseline']['strict_pass'],
            original_Torch_reference_strict_outliers_reported_separately=True, two_sequential_ORT_sessions_closed=True))
    report['all6_saved_raw_arrays_verified'] = report['files_verified'] == 6
    report['all4_actual_CPU_ORT_calls_completed_and_sessions_dropped'] = True
    report['all_main_fixed_numeric_and_strict_diagnostic_gates_passed'] = all(v['main_numeric_gate_passed'] and v['main_strict_diagnostic_passed'] for v in report['frames'])
    report['worst_folded_vs_baseline_relative_l2'] = max(v['comparisons']['folded_vs_baseline']['relative_l2'] for v in report['frames'])
    report['worst_folded_vs_baseline_max_abs'] = max(v['comparisons']['folded_vs_baseline']['max_abs'] for v in report['frames'])
    target = w / 'depth-basis-saved-audit.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'files_verified', 'all_main_fixed_numeric_and_strict_diagnostic_gates_passed',
        'worst_folded_vs_baseline_relative_l2', 'worst_folded_vs_baseline_max_abs')}))


if __name__ == '__main__':
    main()
