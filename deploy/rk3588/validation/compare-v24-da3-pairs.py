#!/usr/bin/env python3
"""Saved-array DA3-SMALL/VGGT pair diagnostics; no model or image decoding.

Three preselected RGB pairs are compared with frozen original VGGT 336/518
predictions and post-experiment TUM pose GT. Depth is camera Z in arbitrary
model units. A single positive scale is fitted and scored on the same complete
two-view pixel intersection, so it is descriptive, not a held-out quality gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
COMMIT = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
REFERENCE_336_SHA = '8a5b9e8a7f94757fa906fbcc1b2d37ae3d4751076d69e966dd626a15a6c63598'
REFERENCE_518_SHA = '81ce7e6b5005c60db97b129b2849e65f614b75dd978015849309bea82810c438'
INPUT_MANIFEST_SHA = '3a104c1f30cf3ed97294852a1d8cabccaf4c8a38d2b8297086eb1503c79a65a2'
WEIGHTS_SHA = '364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf'
CONFIG_SHA = 'a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2'
COMPARISON_SHA = '0f740d3a35c472745eccb3f674b411209a2cef6fcd56ba66e44e6274e224979b'
GEOMETRY_SHA = 'c6f9a296797c58c7bfc234461278335b644b4ab0f27c1b96c781d6731412cb69'
GT_SHA = 'aac0319a6ef4e1cdf61e779d2152b95aa7e9f7b1749d6d18717b43ddabffede2'
NORMALIZATION = dict(mean=[.485, .456, .406], std=[.229, .224, .225])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_sha(path, digest):
    require(sha(path) == digest, 'Pinned file SHA mismatch: ' + Path(path).name)


def strict_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def tensor_sha(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def validate_tensor(a, info):
    require(a.dtype == np.float32 and np.isfinite(a).all(), 'Finite FP32 tensor required')
    digests = [info[key] for key in ('sha256', 'tensor_sha256') if key in info]
    require(bool(digests) and all(value == tensor_sha(a) for value in digests), 'Recorded raw tensor SHA differs')
    require(info['shape'] == list(a.shape) and info['dtype'] == 'float32' and info['bytes'] == a.nbytes,
            'Recorded tensor header differs')


def numerical_helper(root):
    path = root / 'compare-v17-resolution-pairs.py'
    require_sha(path, COMPARISON_SHA)
    require_sha(root.parents[2] / 'scripts/evaluate_edge_trajectory.py', GEOMETRY_SHA)
    spec = importlib.util.spec_from_file_location('_v24_frozen_pair_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_references(args, helper):
    for directory, digest in ((args.reference_336, REFERENCE_336_SHA), (args.reference_518, REFERENCE_518_SHA)):
        require_sha(directory / 'capture.json', digest)
    a, rows336, pins336 = helper.load_capture(args.reference_336, 336, helper.V17_CAPTURE_SHA)
    b, rows518, pins518 = helper.load_capture(args.reference_518, 518, helper.V16_CAPTURE_SHA)
    identity = helper.verify_identity(a, b)
    require_sha(args.input_manifest, INPUT_MANIFEST_SHA)
    inputs = strict_json(args.input_manifest)
    require(inputs['kind'] == 'v24_same_rectified_original_VGGT_RGB_FP32_inputs'
            and inputs['reference_sha256'] == REFERENCE_336_SHA and len(inputs['pairs']) == 3,
            'Input manifest is not the three fixed original RGB pairs')
    input_pins = {}
    for i, (row, ref) in enumerate(zip(inputs['pairs'], rows336)):
        require(row['pair_index'] == i and row['revision'] == i + 1 and row['frame_ids'] == ref['record']['frame_ids']
                and row['source_images'] == ref['selected']['images'], 'Exact source/sequence identity differs')
        name = Path(row['images']).name
        require(name == 'input-pair-' + str(i) + '.npy', 'Unexpected fixed input filename')
        path = args.raw_input_dir / name
        require_sha(path, row['sha256'])
        require(path.stat().st_size == row['bytes'], 'Raw input file bytes differ')
        raw = np.load(path, allow_pickle=False)
        validate_tensor(raw, ref['record']['input_tensor'])
        require(raw.shape == (2, 3, 252, 336) and raw.min() >= 0 and raw.max() <= 1,
                'Original RGB must be two fixed 252x336 views in [0,1]')
        input_pins[name] = row['sha256']
    return a, rows336, rows518, inputs, dict(VGGT336=pins336, VGGT518=pins518,
                                            raw_RGB=input_pins, source_identity=identity)


def load_candidate(args, reference, rows336, inputs, helper):
    path = args.candidate / 'capture.json'
    require_sha(path, args.candidate_sha256)
    c = strict_json(path)
    require(c['status'] == 'completed_fixed_input_diagnostic' and c['model_name'] == 'da3-small'
            and c['source_code_sha256']['run-da3-small-board.py'] == args.capture_script_sha256
            and c['official_commit'] == COMMIT,
            'Candidate is not the pinned official DA3-SMALL capture')
    require(c['device'] == 'cpu' and c['precision'] == 'float32' and c['inference_on_board_only'] is True
            and c['ORB_used'] is False and c['npu_used'] is False and c['GT_used'] is False
            and c['graph_run'] is False and c['online_pipeline_run'] is False,
            'This comparison requires unconditioned board FP32 model output')
    require(c['checkpoint']['sha256'] == WEIGHTS_SHA and c['checkpoint']['bytes'] == 137248940
            and c['config_sha256'] == CONFIG_SHA and c['input_manifest_sha256'] == INPUT_MANIFEST_SHA,
            'Official weights/config/input manifest differ')
    require(c['camera_conditions'] == dict(extrinsics=None, intrinsics=None)
            and c['use_ray_pose'] is False, 'GT/camera conditioning or alternative decoder is not allowed')
    norm = c['normalization']
    require(all(norm.get(key) == value for key, value in NORMALIZATION.items())
            and norm.get('applications') == 1 and norm.get('uint8_roundtrip') is False
            and c['geometric_preprocessing_changed'] is False,
            'Exactly one official normalization, without new resize/crop, is required')
    require(c['selected_inputs'] == inputs['pairs'], 'Original source identities differ from the pinned input manifest')
    require(c['reference_capture_sha256'] == REFERENCE_336_SHA
            and c['rectification'].get('reference_capture_sha256') == REFERENCE_336_SHA
            and c['rectification'].get('performed_in_this_probe') is False,
            'Rectification must be inherited from the byte-exact frozen original336 RGB inputs')
    require(len(c['records']) == 3, 'Exactly three preselected DA3 pair records required')
    result, hashes = [], {'capture.json': args.candidate_sha256}
    for i, (record, ref) in enumerate(zip(c['records'], rows336)):
        require(record['pair_index'] == i and record['frame_ids'] == ref['record']['frame_ids'],
                'DA3 raw RGB tensor identity differs from VGGT336')
        old_input, new_input = ref['record']['input_tensor'], record['input_tensor']
        require(all(new_input.get(key) == old_input.get(key)
                    for key in ('shape', 'dtype', 'bytes', 'tensor_sha256')),
                'DA3 raw RGB tensor byte identity differs from VGGT336')
        normalized = record['normalized_input_tensor']
        require(normalized['shape'] == [1, 2, 3, 252, 336]
                and normalized['dtype'] == 'float32'
                and normalized['bytes'] == 2 * 3 * 252 * 336 * 4,
                'DA3 normalized input must preserve the fixed RGB geometry')
        normalized_path = args.candidate / normalized['file']
        require(Path(normalized['file']).name == normalized['file'], 'Simple normalized-input filename required')
        require_sha(normalized_path, normalized['file_sha256'])
        normalized_array = np.load(normalized_path, allow_pickle=False)
        validate_tensor(normalized_array, normalized)
        raw = np.load(args.raw_input_dir / Path(inputs['pairs'][i]['images']).name, allow_pickle=False)
        mean = np.asarray(NORMALIZATION['mean'], np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(NORMALIZATION['std'], np.float32).reshape(1, 3, 1, 1)
        require(normalized_array.tobytes(order='C') == ((raw - mean) / std)[None].tobytes(order='C'),
                'Recorded normalized pixels are not one FP32 official mean/std application')
        meta = record['artifact']
        require(Path(meta['file']).name == meta['file'], 'Simple local NPZ filename required')
        path = args.candidate / meta['file']
        require_sha(path, meta['sha256'])
        require(path.stat().st_size == meta['bytes'], 'DA3 artifact byte size differs')
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == {'depth', 'depth_conf', 'extrinsic', 'intrinsic', 'pose_enc', 'frame_ids'},
                    'DA3 canonical NPZ schema differs')
            require(saved['frame_ids'].dtype == np.float64 and np.array_equal(saved['frame_ids'], record['frame_ids']),
                    'DA3 exact numeric frame IDs differ')
            arrays = {key: saved[key].copy() for key in record['arrays'] if key != 'frame_ids'}
            ids = saved['frame_ids'].copy()
            id_meta = record['arrays']['frame_ids']
            require(id_meta['shape'] == [2] and id_meta['dtype'] == 'float64'
                    and id_meta['bytes'] == ids.nbytes and id_meta['tensor_sha256'] == tensor_sha(ids),
                    'DA3 frame ID raw metadata differs')
        shapes = dict(depth=(1, 2, 252, 336, 1), depth_conf=(1, 2, 252, 336),
                      extrinsic=(1, 2, 3, 4), intrinsic=(1, 2, 3, 3), pose_enc=(1, 2, 9))
        require(set(arrays) == set(shapes), 'Canonical DA3 output fields differ')
        for key, value in arrays.items():
            validate_tensor(value, record['arrays'][key])
            require(value.shape == shapes[key], 'DA3 output shape differs: ' + key)
        require(np.all(arrays['depth'] > 0) and np.all(arrays['depth_conf'] >= 1), 'Positive Z depth and exp+1 confidence required')
        poses = []
        for value in arrays['extrinsic'][0]:
            T = np.eye(4); T[:3] = value
            poses.append(helper.geometry.rigid(T, 'DA3 official world-to-camera extrinsic'))
        require(np.isfinite(record['forward_seconds']) and record['forward_seconds'] > 0,
                'Positive timed complete forward required')
        semantic_record = dict(record, model_inference_seconds=record['forward_seconds'])
        result.append(dict(record=semantic_record, arrays=arrays, A=ref['A'], mapping=ref['mapping'],
                           relative=poses[1] @ helper.geometry.inverse(poses[0])))
        hashes[meta['file']] = meta['sha256']
        hashes[normalized['file']] = normalized['file_sha256']
    return c, result, hashes


def depth_comparison(candidate, reference, label, pair_index, output, helper):
    a = candidate['arrays']['depth'][0, ..., 0].astype(np.float64)
    b = reference['arrays']['depth'][0, ..., 0].astype(np.float64)
    h, w = a.shape[1:]
    y, x = np.indices((h, w))
    points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    same_grid = np.array_equal(candidate['A'], reference['A']) and a.shape == b.shape
    mapped = points @ (reference['A'] @ np.linalg.inv(candidate['A'])).T
    masks, collected, frame_stats = [], [], []
    for i in range(2):
        aa = a[i].ravel()
        bb = b[i].ravel() if same_grid else helper.bilinear_sample(b[i], mapped[:, :2])
        mask = np.isfinite(aa) & (aa > 0) & np.isfinite(bb) & (bb > 0)
        masks.append(mask.reshape(h, w))
        collected.append((aa[mask], bb[mask]))
        frame_stats.append(dict(frame_id=candidate['record']['frame_ids'][i],
                                raw_DA3=helper.positive_stats(a[i]), raw_reference=helper.positive_stats(b[i]),
                                matched_pixels=int(mask.sum()), excluded_pixels=int(mask.size - mask.sum())))
    aa, bb = (np.concatenate([p[i] for p in collected]) for i in range(2))
    require(len(aa) > 0, 'No common positive depth values')
    scale = float(np.dot(aa, bb) / np.dot(aa, aa))
    require(np.isfinite(scale) and scale > 0, 'Positive descriptive depth scale unavailable')
    path = output / ('depth-mask-pair-' + str(pair_index) + '-' + label + '.npz')
    np.savez_compressed(path, positive_intersection=np.stack(masks))
    return dict(status='descriptive_not_accuracy_gate', reference=label, frames=frame_stats,
                units='independent_arbitrary_model_camera_Z_units',
                sampling='same-grid exact values including all borders' if same_grid else
                         'float64 four-neighbour positive bilinear reference at matching rectified pixel centres',
                sampling_grid='all DA3 252x336 pixel centres in both views',
                confidence_and_rectification_border_not_filtered=True,
                mask=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)),
                unaligned=helper.residual_stats(aa, bb), positive_scale_DA3_to_reference=scale,
                scale_aligned=helper.residual_stats(aa, bb, scale),
                scale_rule='one positive least-squares-through-origin scalar over both views',
                scale_fit_and_score_use_same_pixels=True, holdout_used=False, depth_GT_used=False)


def analyze(args, output):
    helper = numerical_helper(args.helpers_root)
    reference, rows336, rows518, inputs, provenance = load_references(args, helper)
    require_sha(args.groundtruth, GT_SHA)
    times, poses, gt_info = helper.geometry.read_gt(args.groundtruth)
    report = dict(kind='v24_DA3_SMALL_three_fixed_pair_saved_array_diagnostic', status='reference_plan_verified',
                  analysis_script_sha256=sha(__file__), frozen_comparison_dependency_sha256=COMPARISON_SHA,
                  frozen_geometry_dependency_sha256=GEOMETRY_SHA, input_manifest_sha256=INPUT_MANIFEST_SHA,
                  raw_inputs_modified=False, model_inference_executed=False, image_decode_executed=False,
                  board_operations_executed=False, full_dataset_ATE_RPE_validated=False,
                  accuracy_gate_validated=False, metric_scale_validated=False, real_time_validated=False,
                  provenance=provenance, groundtruth=dict(sha256=GT_SHA, **gt_info, used_after_capture_only=True,
                    nearest_tolerance_seconds=.02, minimum_GT_translation_for_direction_m=.01,
                    convention='T_world_from_camera, XYZW quaternion; same nearest samples as historical diagnostics'),
                  pair_selection='three earliest saved revisions, without quality filtering', pairs=[])
    candidate = candidate_rows = candidate_pins = None
    if not args.plan_only:
        candidate, candidate_rows, candidate_pins = load_candidate(args, reference, rows336, inputs, helper)
        report.update(status='completed_saved_array_diagnostic', candidate_capture_sha256=args.candidate_sha256,
                      candidate_producer_sha256=args.capture_script_sha256, checkpoint_sha256=WEIGHTS_SHA,
                      config_sha256=CONFIG_SHA, official_commit=COMMIT,
                      camera_conditions_used=False, original_RGB_exact_same_as_VGGT336=True)
    for i, (r336, r518) in enumerate(zip(rows336, rows518)):
        ids = r336['record']['frame_ids']
        indices = [helper.geometry.nearest(times, timestamp, .02) for timestamp in ids]
        gt = None if any(j is None for j in indices) else helper.geometry.inverse(poses[indices[1]]) @ poses[indices[0]]
        pair = dict(pair_index=i, frame_ids=ids,
                    source_images=[dict(basename=Path(s['path']).name, sha256=s['sha256'], frame_id=s['frame_id'])
                                   for s in r336['selected']['images']],
                    raw_RGB_tensor_sha256=r336['record']['input_tensor']['tensor_sha256'],
                    GT_matches=[dict(frame_id=t, timestamp=None if j is None else float(times[j]),
                                     difference_seconds=None if j is None else float(times[j] - t)) for t, j in zip(ids, indices)],
                    relative_pose_convention='T_second_from_first=Tcw[1]@inverse(Tcw[0]); translation in destination camera',
                    predictions={}, depth_comparisons={})
        rows = [('VGGT336', r336), ('VGGT518', r518)]
        if candidate_rows is not None:
            c = candidate_rows[i]
            rows.append(('DA3_SMALL', c))
            for label, ref in (('VGGT336', r336), ('VGGT518', r518)):
                pair['depth_comparisons'][label] = depth_comparison(c, ref, label, i, output, helper)
            pair['DA3_relative_pose_difference_to_VGGT'] = {
                label: helper.geometry.errors(c['relative'], ref['relative'], 1e-12)
                for label, ref in (('VGGT336', r336), ('VGGT518', r518))}
        calibrated_K = np.asarray(reference['rectification']['output_K'], np.float64)
        for label, row in rows:
            pair['predictions'][label] = dict(actual_input_hw=row['mapping']['actual_input_hw'],
                relative_T_second_from_first=row['relative'].tolist(),
                relative_translation_norm_arbitrary_units=float(np.linalg.norm(row['relative'][:3, 3])),
                GT_pair_diagnostic=None if gt is None else helper.geometry.errors(row['relative'], gt, .01),
                GT_unavailable_reason='no_original_GT_within_20ms' if gt is None else None,
                forward_seconds=row['record']['model_inference_seconds'],
                K_original_rectified_pixels=[helper.camera_stats(K, row['A'], calibrated_K) for K in row['arrays']['intrinsic'][0]])
        report['pairs'].append(pair)
    if candidate_pins is not None:
        require(all(sha(args.candidate / name) == digest for name, digest in candidate_pins.items()),
                'DA3 artifacts changed during comparison')
        report['provenance']['DA3'] = candidate_pins
    for directory, pins in ((args.reference_336, provenance['VGGT336']), (args.reference_518, provenance['VGGT518'])):
        require(all(sha(directory / name) == digest for name, digest in pins.items()), 'Original reference changed during comparison')
    require_sha(args.groundtruth, GT_SHA)
    report['limitations'] = ['Three fixed pairs, not complete-dataset ATE/RPE or an online timing test.',
        'DA3-SMALL replaces the model; this is not an acceleration result for original VGGT.',
        'Depth scale fit uses all compared pixels without held-out data and is not a metric depth accuracy gate.',
        'VGGT518 uses the same physical sources/rectification at another resolution, not byte-identical processed RGB.',
        'GT is not passed into either model; only post-experiment relative rotation and direction are evaluated.']
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--helpers-root', type=Path, default=Path('/dev/shm/vggt-v15-public-snapshot/deploy/rk3588/validation'))
    p.add_argument('--reference-336', type=Path, required=True)
    p.add_argument('--reference-518', type=Path, required=True)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--raw-input-dir', type=Path, required=True)
    p.add_argument('--groundtruth', type=Path, required=True)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--candidate-sha256')
    p.add_argument('--capture-script-sha256')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    if not args.plan_only and not all((args.candidate, args.candidate_sha256, args.capture_script_sha256)):
        p.error('Actual comparison requires candidate directory and explicit capture/producer SHA pins')
    if args.output_dir.exists():
        p.error('Output directory must be new')
    args.output_dir.mkdir(parents=True)
    try:
        report = analyze(args, args.output_dir)
    except Exception as error:
        report = dict(kind='v24_DA3_saved_array_diagnostic', status='failed', analysis_script_sha256=sha(__file__),
                      error=type(error).__name__ + ': ' + str(error), model_inference_executed=False,
                      image_decode_executed=False, board_operations_executed=False)
    path = args.output_dir / 'comparison.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], output=path.name, sha256=sha(path))))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
