#!/usr/bin/env python3
"""Independently audit saved p0/p1/p2 CameraHead evidence with NumPy only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/dev/shm/edge-vggt-v113-camera-precision')
PRIVATE = Path('/path/to/research-artifacts')
PINS = {
    'diagnostic': '22869179086da3473d2b945b61a452791f53c50a6e83545dfc5a3ecaba31c914',
    'source': 'e16e13894f08f89ce9a52a4cdfff4472c3f38e23d3e4e647b8c0d2fb1f0114b2',
    'cpu': '969ab6f5611efa62d81b5d62625a15815248668e1e079db17ac4d9ed9c96932b',
    'npu': '4af7b9c4cbd38dbfd54c14b1308dd6035da25941f98f858a5ad13b781c3c8af6',
    'camera': '1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(4 << 20), b''):
            digest.update(piece)
    return digest.hexdigest()


def raw_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def comparison(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    require(a.shape == b.shape, 'Comparison shape mismatch')
    difference, norm = a - b, float(np.linalg.norm(b))
    outside = int(np.count_nonzero(np.abs(difference) > 1e-5 + 1e-4 * np.abs(b)))
    return {'array_equal': bool(np.array_equal(actual, reference)),
            'raw_sha256_equal': raw_sha(actual) == raw_sha(reference),
            'l2_error': float(np.linalg.norm(difference)),
            'reference_l2_norm': norm,
            'relative_l2_error': float(np.linalg.norm(difference) / norm) if norm else None,
            'max_abs_error': float(np.abs(difference).max()),
            'outside_fixed_element_gate': outside,
            'strict_rtol_1e_4_atol_1e_5_pass': outside == 0}


def decomposition(p0, p1, p2):
    p0, p1, p2 = [np.asarray(value, dtype=np.float64) for value in (p0, p1, p2)]
    upstream, head, total = p1 - p0, p2 - p1, p2 - p0
    u, h, t = [value.ravel() for value in (upstream, head, total)]
    nu, nh, nt = [float(np.linalg.norm(value)) for value in (u, h, t)]
    dot = float(u @ h)
    return {'upstream_propagated_difference_p1_minus_p0': upstream.tolist(),
            'head_execution_difference_p2_minus_p1': head.tolist(),
            'total_difference_p2_minus_p0': total.tolist(),
            'upstream_l2': nu, 'head_l2': nh, 'total_l2': nt,
            'upstream_head_dot_product': dot,
            'upstream_head_cosine': dot / (nu * nh) if nu and nh else None,
            'negative_dot_product_cancellation': dot < 0,
            'vector_sum_closure_max_abs': float(np.abs(total - upstream - head).max()),
            'squared_norm_identity_residual': float(nt * nt - (nu * nu + nh * nh + 2 * dot)),
            'norms_are_not_additive_error_contribution_percentages': True}


def geometry(extrinsics):
    e = np.eye(4, dtype=np.float64)[None].repeat(2, axis=0)
    e[:, :3] = np.asarray(extrinsics, dtype=np.float64).reshape(2, 3, 4)
    inverse = np.linalg.inv(e)
    return {'world_to_camera': e, 'camera_centers': inverse[:, :3, 3],
            'camera1_from_camera0': e[1] @ inverse[0]}


def rotation_angle_degrees(matrix):
    r = matrix[:3, :3]
    sine = np.linalg.norm([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    cosine = (np.trace(r) - 1) / 2
    return float(np.degrees(np.arctan2(sine, cosine)))


def codec_formula_reference(pose):
    """FP64 validation arithmetic from pinned pose_enc.py and rotation.py."""
    pose = pose.astype(np.float64)
    i, j, k, r = [pose[..., axis] for axis in (3, 4, 5, 6)]
    squared = i*i + j*j + k*k + r*r
    require(np.all(squared > 0), 'Nonzero saved quaternion required')
    s = 2 / squared
    rotation = np.stack((1-s*(j*j+k*k), s*(i*j-k*r), s*(i*k+j*r),
                         s*(i*j+k*r), 1-s*(i*i+k*k), s*(j*k-i*r),
                         s*(i*k-j*r), s*(j*k+i*r), 1-s*(i*i+j*j)), axis=-1).reshape(pose.shape[:-1] + (3, 3))
    extrinsics = np.concatenate((rotation, pose[..., :3, None]), axis=-1)
    intrinsics = np.zeros(pose.shape[:-1] + (3, 3), dtype=np.float64)
    intrinsics[..., 0, 0] = 259 / np.tan(pose[..., 8] / 2)
    intrinsics[..., 1, 1] = 196 / np.tan(pose[..., 7] / 2)
    intrinsics[..., 0, 2], intrinsics[..., 1, 2], intrinsics[..., 2, 2] = 259, 196, 1
    require(np.isfinite(extrinsics).all() and np.isfinite(intrinsics).all(), 'Finite codec validation required')
    return extrinsics, intrinsics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir', type=Path, default=ROOT / 'board-r1')
    parser.add_argument('--cpu-reference', type=Path,
                        default=PRIVATE / 'v99-native518-components-review-r1/head-reference.json')
    parser.add_argument('--camera-metadata', type=Path,
                        default=Path('/dev/shm/edge-vggt-v50/public-snapshot/deploy/rk3588/validation/vggt-v15-camera-capture112.json'))
    parser.add_argument('--npu-dir', type=Path,
                        default=PRIVATE / 'v100-native518-integration-r1/full-model-sequential-core1-r1')
    parser.add_argument('--source-image-dir', type=Path,
                        default=PRIVATE / 'v109-native-slam-review-r1/source-images')
    parser.add_argument('--output', type=Path, default=ROOT / 'camera-precision-saved-audit.json')
    args = parser.parse_args()
    ledger = []

    def pin(path, expected, size=None):
        require((size is None or path.stat().st_size == size) and sha(path) == expected,
                'File identity mismatch: ' + str(path))
        ledger.append({'file': str(path), 'bytes': path.stat().st_size, 'sha256': expected})
        return path

    def read(path, expected):
        return json.loads(pin(path, expected).read_text())

    diagnostic = read(args.diagnostic_dir / 'result.json', PINS['diagnostic'])
    cpu = read(args.cpu_reference, PINS['cpu'])
    npu = read(args.npu_dir / 'result.json', PINS['npu'])
    camera = read(args.camera_metadata, PINS['camera'])
    pin(ROOT / 'capture_camera_precision_board.py', PINS['source'])
    require(diagnostic['schema'] == 'v113_original_native518_CameraHead_precision_diagnosis' and
            diagnostic['completed'] and diagnostic['status'] ==
            'completed_original_CameraHead_precision_diagnosis_not_quality_accepted' and
            diagnostic['script_sha256'] == PINS['source'], 'Successful frozen component diagnosis required')
    require(diagnostic['counts'] == {'camera_calls': 2, 'camera_refinements': 8, 'pose_codec_calls': 8},
            'Actual component call ledger differs')
    for key in ('hooks_removed', 'saved_inputs_unchanged', 'pinned_source_and_inputs_unchanged',
                'affinity_restored', 'environment_restored', 'original_parameter_storage_identity_unchanged'):
        require(diagnostic[key] is True, 'Failed recorded invariant: ' + key)
    require(not diagnostic['cleanup_errors'] and not diagnostic['SDK_ORT_NPU_GPU_executed'] and
            not diagnostic['full_model_forward_completed'] and
            not diagnostic['method_override_or_output_substitution'], 'Component scope/cleanup differs')
    require(diagnostic['checkpoint'] == npu['checkpoint'] == cpu['checkpoint'] and
            diagnostic['source_images'] == npu['source_images'] == cpu['source_images'] and
            diagnostic['source_pair_indices'] == [32, 40] and diagnostic['image_size_hw'] == [392, 518],
            'Same original checkpoint/native source-pair provenance required')
    parameter_rows = camera['artifacts']['camera-state.npz']['tensors']
    require(len(parameter_rows) == 69 and 'empty_pose_tokens' in parameter_rows and
            parameter_rows == cpu['camera_original_parameter_metadata'] == diagnostic['original_parameter_metadata'],
            'Original 69-parameter metadata including empty seed differs')
    for row in diagnostic['source_images']:
        pin(args.source_image_dir / Path(row['file']).name, row['sha256'], row['bytes'])

    def array(folder, row):
        require(Path(row['file']).name == row['file'], 'Array path must be basename')
        value = np.load(pin(folder / row['file'], row['sha256'], row['bytes']), allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == row['shape'] and
                np.isfinite(value).all() and raw_sha(value) == row['raw_sha256'], 'Saved array contract differs')
        if 'tensor_bytes' in row:
            require(value.nbytes == row['tensor_bytes'] and row['dtype'] == 'float32' and row['finite'],
                    'Saved typed array metadata differs')
        ledger[-1].update(shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes,
                          raw_sha256=raw_sha(value), finite=True)
        return value

    saved = {name: array(args.diagnostic_dir, row) for name, row in diagnostic['artifacts'].items()}
    require(len(saved) == 16 and set(path.name for path in args.diagnostic_dir.glob('*.npy')) ==
            {row['file'] for row in diagnostic['artifacts'].values()} and
            not list(args.diagnostic_dir.glob('*.tmp')), 'Saved component array set differs')
    require(sum(row['bytes'] for row in diagnostic['artifacts'].values()) == diagnostic['saved_array_bytes'],
            'Saved array byte ledger differs')
    controls = {}
    for name, ref_name in [('cpu_control_input', 'camera_input'),
                           ('cpu_control_normalized_input', 'camera_normalized_input')] + [
                               ('cpu_control_pose' + str(i), 'camera_pose_iteration_' + str(i)) for i in range(4)]:
        row = diagnostic['artifacts'][name]
        original = cpu['artifacts'][ref_name]
        require(row['sha256'] == original['sha256'] and row['raw_sha256'] == original['raw_sha256'] and
                row['shape'] == original['shape'] and row['bytes'] == original['bytes'],
                'Original CPU control bytes differ: ' + name)
        controls[name] = {'file_and_raw_sha256_equal_to_pinned_original_reference': True}
    frame, global_tokens = [array(args.npu_dir, row) for row in npu['calls'][0]['actual_raw_group_outputs'][3][:2]]
    require(frame.shape == (2, 1041, 1024) and global_tokens.shape == (1, 2082, 1024),
            'Actual final NPU group output dimensions differ')
    reconstructed = np.concatenate((frame.reshape(1, 2, 1041, 1024)[:, :, 0],
                                    global_tokens.reshape(1, 2, 1041, 1024)[:, :, 0]), axis=-1)
    require(np.array_equal(reconstructed, saved['npu_tokens_cpu_head_input']) and
            raw_sha(reconstructed) == '703a919bab1661985a4029498ccd2d5ce62043db97f6aa58a74cfe781b077d43',
            'Actual NPU camera input reconstruction differs')
    p2 = {key: array(args.npu_dir, npu['calls'][0]['artifacts'][key])
          for key in ('pose_enc', 'extrinsics', 'intrinsics')}
    repeats = {}
    for key in p2:
        repeated = array(args.npu_dir, npu['calls'][1]['artifacts'][key])
        repeats[key] = comparison(repeated, p2[key])
        require(repeats[key]['raw_sha256_equal'], 'Saved actual NPU head repeat differs')
    values = {
        'pose_enc': [saved['cpu_control_pose3'], saved['npu_tokens_cpu_head_pose3'], p2['pose_enc']],
        'extrinsics': [saved['cpu_control_extrinsics'][-1], saved['npu_tokens_cpu_head_extrinsics'][-1], p2['extrinsics']],
        'intrinsics': [saved['cpu_control_intrinsics'][-1], saved['npu_tokens_cpu_head_intrinsics'][-1], p2['intrinsics']],
    }
    codec_checks = {}
    for index, name in enumerate(('p0', 'p1', 'p2')):
        expected = codec_formula_reference(values['pose_enc'][index])
        codec_checks[name] = {key: comparison(values[key][index], formula)
                             for key, formula in zip(('extrinsics', 'intrinsics'), expected)}
        require(all(row['strict_rtol_1e_4_atol_1e_5_pass'] for row in codec_checks[name].values()),
                'Saved final codec disagrees with original formula')
    pairs = {key: {name: comparison(a[j], a[i]) for name, i, j in
                   [('p1_vs_p0', 0, 1), ('p2_vs_p1', 1, 2), ('p2_vs_p0', 0, 2)]}
             for key, a in values.items()}
    iterations = [comparison(saved['npu_tokens_cpu_head_pose' + str(i)], saved['cpu_control_pose' + str(i)])
                  for i in range(4)]
    for key, actual in pairs['pose_enc']['p1_vs_p0'].items():
        if key in diagnostic['cases'][1]['pose_vs_original_CPU'][-1] and isinstance(actual, (int, float)):
            require(np.isclose(actual, diagnostic['cases'][1]['pose_vs_original_CPU'][-1][key], rtol=1e-13, atol=1e-15),
                    'Reported pose metrics differ from independent recomputation')
    decoded = [geometry(value) for value in values['extrinsics']]
    geometry_checks = {}
    for name, i, j in [('p1_vs_p0', 0, 1), ('p2_vs_p1', 1, 2), ('p2_vs_p0', 0, 2)]:
        a, b = decoded[i]['camera1_from_camera0'], decoded[j]['camera1_from_camera0']
        delta = b @ np.linalg.inv(a)
        geometry_checks[name] = {
            'relative_rotation_difference_degrees': rotation_angle_degrees(delta),
            'relative_translation_vector_difference_model_gauge': (b[:3, 3] - a[:3, 3]).tolist(),
            'relative_translation_difference_l2_model_gauge': float(np.linalg.norm(b[:3, 3] - a[:3, 3])),
            'camera_center_differences_l2_model_gauge': np.linalg.norm(
                decoded[j]['camera_centers'] - decoded[i]['camera_centers'], axis=-1).tolist(),
            'per_camera_rotation_difference_degrees': [rotation_angle_degrees(
                decoded[j]['world_to_camera'][f] @ np.linalg.inv(decoded[i]['world_to_camera'][f])) for f in range(2)]}
    log = ROOT / 'board-r1.log'
    progress = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
    require(progress[-1]['stage'] == 'finished' and progress[-1]['counts'] == diagnostic['counts'] and
            log.read_text().count('CameraHead trunk_fn: B=1, S=2, C=2048') == 2,
            'Saved worker progress/call ledger differs')
    ledger.append({'file': str(log), 'bytes': log.stat().st_size, 'sha256': sha(log)})
    launch_path = ROOT / 'root-launch-provenance.json'
    launch = json.loads(launch_path.read_text())
    require(launch['schema'] == 'v113_root_launch_provenance' and launch['timeout_seconds'] == 180 and
            launch['outer_adb_session_exit_code'] == 0 and launch['source_sha256'] == PINS['source'] and
            launch['receipt_sha256'] == PINS['diagnostic'] and launch['log_sha256'] == sha(log) and
            not launch['NPU_clock_policy_modified'] and not launch['NPU_thermal_controller_used'] and
            not launch['temperature_samples_collected'], 'Root launch provenance differs')
    ledger.append({'file': str(launch_path), 'bytes': launch_path.stat().st_size, 'sha256': sha(launch_path)})
    report = {
        'schema': 'v113_original_CameraHead_independent_saved_precision_audit',
        'status': 'passed_saved_evidence_audit_not_quality_or_deployment_acceptance',
        'audit_script_sha256': sha(__file__), 'neural_SDK_ORT_GPU_or_board_calls_executed': False,
        'definition': {'p0': 'Original CPU CameraHead(original CPU upstream tokens)',
                       'p1': 'Same original CPU CameraHead(actual saved NPU upstream tokens)',
                       'p2': 'Actual saved NPU CameraHead(actual same NPU upstream tokens)'},
        'file_ledger': ledger, 'diagnostic_arrays_checked': 16, 'selected_NPU_arrays_checked': 8,
        'original_CPU_control': controls, 'actual_NPU_raw_token_reconstruction_byte_equal': True,
        'actual_NPU_head_saved_repeats': repeats, 'final_iteration_comparisons': pairs,
        'all_four_refinements_p1_vs_p0': iterations,
        'saved_final_pose_codec_FP64_formula_consistency': codec_checks,
        'token_input_comparison': comparison(saved['npu_tokens_cpu_head_input'], saved['cpu_control_input']),
        'token_norm_comparison': comparison(saved['npu_tokens_cpu_head_normalized_input'], saved['cpu_control_normalized_input']),
        'vector_decompositions': {key: decomposition(*value) for key, value in values.items()},
        'pose_component_decompositions': {key: decomposition(*[p[..., cut] for p in values['pose_enc']])
                                         for key, cut in [('translation_model_gauge', slice(0, 3)),
                                                          ('quaternion_coefficients_xyzw', slice(3, 7)), ('FoV_radians', slice(7, 9))]},
        'relative_two_frame_geometry': {
            'convention': 'Saved extrinsics E_i are camera-from-world; camera1_from_camera0 = E_1 @ inv(E_0). Camera centers are inv(E_i)[:3,3].',
            'translation_units': 'Original model gauge; no metric scale, GT alignment, SL4 scale or trajectory inference applied',
            'rotation_metric': 'Stable atan2 from the relative rotation matrix, without projecting or modifying saved matrices',
            'routes': {name: {key: value.tolist() for key, value in route.items()}
                       for name, route in zip(('p0', 'p1', 'p2'), decoded)},
            'comparisons': geometry_checks,
            'relative_matrix_decomposition': decomposition(*[route['camera1_from_camera0'][:3] for route in decoded])},
        'actual_component_counts_and_cleanup': {key: diagnostic[key] for key in
            ('counts', 'hooks_removed', 'original_parameter_storage_identity_unchanged', 'saved_inputs_unchanged',
             'affinity_restored', 'environment_restored', 'cleanup_errors', 'pinned_source_and_inputs_unchanged')},
        'execution_controls': {'root_launch_provenance': launch,
                               'external_thermal_or_NPU_controller': False, 'external_temperature_history_available': False,
                               'root_reported_post_run_NPU_hz': 300000000,
                               'NPU_frequency_evidence': 'Readonly post-run root check; not a historical sample during this CPU-only run'},
        'recorded_CPU_timing_only': {key: diagnostic[key] for key in
                                   ('wall_seconds', 'process_user_seconds', 'process_system_seconds', 'peak_rss_mib')},
        'limits': [
            'One original native518 pair [32,40], not the 16-pair R111 trajectory; no definitive explanation of its trajectory gap.',
            'p1-p0 describes input perturbations propagated through the original CPU head, not raw input error alone. p2-p1 compares head execution on identical tokens.',
            'Difference vectors sum; their norms and quality effects do not. Negative dot products reveal cancellation; no additive attribution percentages are assigned.',
            'Raw pose-vector norms mix translation, quaternion and FoV units; separate components and gauge-invariant relative rotations are reported.',
            'The 69-parameter original-state identity is checked against pinned metadata and board verification; the 864 MB NPZ is not independently rehashed on this host.',
            'The audit loads only saved arrays and performs small NumPy statistics/geometry; no neural forward or original model operator executes.',
            'No deployment, full-model timing, ATE/RPE, realtime, map quality, stability or 8 GB acceptance follows.'],
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'final_pose_comparisons': pairs['pose_enc'],
                      'relative_geometry': geometry_checks}, indent=2))


if __name__ == '__main__':
    main()
