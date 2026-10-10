#!/usr/bin/env python3
"""Compare saved original CPU FP32 224/336 pairs to the fixed 518 reference.

Only saved numerical tensors and post-experiment GT are read. There is no image
decoding, tracking, inference or board operation. The three earliest pairs do
not establish complete-dataset ATE/RPE, metric scale or realtime performance.
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
HERE = Path(__file__).resolve().parent
V16_COMPARISON_SHA = '527f4efb0467e1fa0f859e6327e74f72651270480816a5503283536fbe7e2ee7'
V16_CAPTURE_SHA = '64adbcbe7f6e013aa738b8bed57d21ef8fde10c4b63bd7c647e9fa2137a79d7c'
V17_CAPTURE_SHA = '3706712dd3a7451fcbce98e079d4018989eca6f64eb98704a81b8f4a9acc03cb'
V16_EDGE_MAPPING_SHA = '5fd6567c0fc7c0b9ddd8a5c6a904248294ee173c31d3c9b5d16149aec848ba57'
EDGE_MAPPING = 'vggt_slam/edge_mapping.py'


def pinned_module(name, filename, digest):
    path = HERE / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Frozen numerical dependency changed: {filename}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v16 = pinned_module('resolution_comparison_v16', 'compare-v16-resolution-pairs.py', V16_COMPARISON_SHA)
capture = pinned_module('resolution_capture_v17', 'capture-v17-resolution-pairs-board.py', V17_CAPTURE_SHA)
geometry = v16.geometry
pixel_affine = v16.pixel_affine
tensor_sha = v16.tensor_sha
positive_stats = v16.positive_stats
bilinear_sample = v16.bilinear_sample
camera_stats = v16.camera_stats


def load_capture(path, image_size, producer_sha):
    path = Path(path)
    receipt_file = path / 'capture.json'
    receipt = geometry.strict_json(receipt_file.read_text())
    if (receipt['status'] != 'completed_fixed_input_diagnostic' or receipt['image_size'] != image_size
            or receipt['max_pairs'] != 3 or receipt['script_sha256'] != producer_sha
            or receipt['device'] != 'cpu' or receipt['precision'] != 'float32'
            or receipt['ORB_used'] is not False or receipt['npu_used'] is not False
            or receipt['checkpoint']['sha256'] != v16.CHECKPOINT_SHA
            or receipt['checkpoint']['bytes'] != 5026874952
            or receipt['source_hashes'].get('third_party/vggt/vggt/utils/load_fn.py') != v16.LOAD_FN_SHA
            or len(receipt['pairs']) != 3 or len(receipt['selected_inputs']) != 3):
        raise ValueError('Capture is not the pinned three-pair original CPU FP32 diagnostic')
    if set(receipt['source_hashes']) != set(capture.SOURCE_FILES):
        raise ValueError('Capture source manifest is incomplete or contains untracked sources')
    if receipt['rectification']['R'] != np.eye(3).tolist() or receipt['rectification']['crop'] is not None:
        raise ValueError('GT comparison requires unchanged optical axes and no rectification crop')
    if producer_sha == V17_CAPTURE_SHA:
        expected = dict(source=EDGE_MAPPING, function='prepare_rectified_images',
                        sha256=capture.PREPROCESSING_SHA, hash_method='ast_get_source_segment_utf8_sha256',
                        original_v16_function_unchanged=True)
        if receipt.get('preprocessing_provenance') != expected:
            raise ValueError('Candidate preprocessing provenance differs from the original lossless bridge')
    result, hashes = [], {'capture.json': geometry.sha(receipt_file)}
    for i, (record, selected) in enumerate(zip(receipt['pairs'], receipt['selected_inputs']), 1):
        if (record['revision'] != i or selected['revision'] != i
                or record['frame_ids'] != selected['frame_ids'] or len(record['frame_ids']) != 2
                or record['file'] != f'predictions_{i:04d}.npz'):
            raise ValueError('Capture pair revision/identity is inconsistent')
        file = path / record['file']
        digest = geometry.sha(file)
        if digest != record['sha256']:
            raise ValueError('Saved prediction file SHA differs from its capture receipt')
        with np.load(file, allow_pickle=False) as saved:
            if set(saved.files) != {'depth', 'depth_conf', 'intrinsic', 'extrinsic', 'pose_enc', 'frame_ids'}:
                raise ValueError('Unexpected saved original prediction keys')
            if not np.array_equal(saved['frame_ids'], record['frame_ids']):
                raise ValueError('NPZ frame IDs differ from the exact recorded pair')
            arrays = {name: saved[name].copy() for name in record['arrays']}
        for name, value in arrays.items():
            info = record['arrays'][name]
            if (value.dtype != np.float32 or list(value.shape) != info['shape']
                    or value.nbytes != info['bytes'] or not np.isfinite(value).all()
                    or tensor_sha(value) != info['tensor_sha256']):
                raise ValueError('Saved tensor values/header differ from their capture receipt')
        A, mapping = pixel_affine(receipt['rectification']['image_hw'], record['input_tensor']['shape'], image_size)
        h, w = mapping['actual_input_hw']
        expected = {'depth': (1, 2, h, w, 1), 'depth_conf': (1, 2, h, w),
                    'intrinsic': (1, 2, 3, 3), 'extrinsic': (1, 2, 3, 4), 'pose_enc': (1, 2, 9)}
        if set(arrays) != set(expected) or any(arrays[name].shape != shape for name, shape in expected.items()):
            raise ValueError('Unexpected original prediction tensor shapes')
        extrinsics = []
        for matrix in arrays['extrinsic'][0]:
            T = np.eye(4)
            T[:3] = matrix
            extrinsics.append(geometry.rigid(T, 'Original VGGT extrinsic'))
        result.append(dict(record=record, selected=selected, arrays=arrays, A=A, mapping=mapping,
                           relative=extrinsics[1] @ geometry.inverse(extrinsics[0])))
        hashes[record['file']] = digest
    return receipt, result, hashes


def verify_identity(candidate, reference):
    for key in ('rectification', 'model'):
        if candidate[key] != reference[key]:
            raise ValueError(f'Candidate and reference {key} differ')
    for key in ('rgb_index', 'calibration', 'checkpoint'):
        if candidate[key]['sha256'] != reference[key]['sha256']:
            raise ValueError(f'Candidate and reference {key} SHA differ')
    if candidate['schedule'] != reference['schedule']:
        raise ValueError('Candidate and reference schedules differ')
    if candidate['selected_inputs'] != reference['selected_inputs']:
        raise ValueError('Candidate and reference exact source identities/SHA/schedule tokens differ')
    for pair in candidate['selected_inputs']:
        if len(pair['images']) != 2:
            raise ValueError('Recorded original source pair has the wrong count')
        for frame_id, image in zip(pair['frame_ids'], pair['images']):
            if (image['frame_id'] != frame_id or image['timestamp'] != frame_id
                    or image['schedule_token']['frame_id'] != frame_id
                    or image['schedule_token']['timestamp'] != frame_id
                    or image['samefile_validated'] is not True):
                raise ValueError('Original source identity was not exactly validated')
    cs, rs = candidate['source_hashes'], reference['source_hashes']
    if set(cs) != set(rs):
        raise ValueError('Candidate and reference source manifest keys differ')
    differences = {name: dict(candidate=cs[name], reference518=rs[name]) for name in cs if cs[name] != rs[name]}
    if differences and set(differences) != {EDGE_MAPPING}:
        raise ValueError('Candidate and reference original model/preprocessing sources differ')
    if differences and rs[EDGE_MAPPING] != V16_EDGE_MAPPING_SHA:
        raise ValueError('Unrecognized reference mapper source cannot justify the authorized peripheral difference')
    return dict(source_hashes_exact_match=not differences, original_model_sources_exact_match=True,
                lossless_preprocessing_function_unchanged=True, allowed_source_differences=differences,
                preprocessing_function_sha256=capture.PREPROCESSING_SHA,
                difference_scope='only authorized mapper scheduling/NPU code outside the pinned image bridge',
                reference_bridge_evidence='frozen v16 mapper file SHA' if differences else 'same whole mapper file SHA',
                candidate_bridge_evidence='pinned v17 producer validates and records exact function source SHA')


def residual_stats(candidate, reference, scale=1.):
    result = v16.residual_stats(candidate, reference, scale)
    result['median_ratio_reference_over_candidate'] = result.pop('median_ratio_518_over_112')
    return result


def compare_depth(candidate, reference):
    dc = candidate['arrays']['depth'][0, ..., 0].astype(np.float64)
    dr = reference['arrays']['depth'][0, ..., 0].astype(np.float64)
    y, x = np.indices(dc.shape[1:])
    points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    mapped = points @ (reference['A'] @ np.linalg.inv(candidate['A'])).T
    values, frames = [], []
    for i in range(2):
        c, r = dc[i].ravel(), bilinear_sample(dr[i], mapped[:, :2])
        valid = np.isfinite(c) & (c > 0) & np.isfinite(r) & (r > 0)
        values.append((c[valid], r[valid]))
        frames.append(dict(frame_id=candidate['record']['frame_ids'][i], raw_candidate=positive_stats(dc[i]),
                           raw_reference518=positive_stats(dr[i]), matched_pixels=int(valid.sum()),
                           excluded_from_comparison=int(len(c) - valid.sum())))
    c = np.concatenate([v[0] for v in values])
    r = np.concatenate([v[1] for v in values])
    if not len(c):
        return dict(status='no_common_positive_depth', frames=frames)
    scale = float(np.dot(c, r) / np.dot(c, c))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Cannot obtain a finite positive descriptive depth scale')
    return dict(status='descriptive_only', units='arbitrary_VGGT_batch_Z_units', frames=frames,
                sampling='518 float64 bilinear at the same rectified original pixel centres as the candidate grid',
                rectification_border_and_confidence_not_filtered=True, unaligned=residual_stats(c, r),
                positive_scale_candidate_to_reference518=scale,
                scale_scope='one scalar for both frames of this independent prediction batch',
                scale_rule='positive least squares through origin on all matched positive pixels',
                scale_fit_and_score_use_same_pixels=True, depth_GT_used=False,
                scale_aligned=residual_stats(c, r, scale))


def analyze(candidate_dir, reference_dir, groundtruth, candidate_size):
    if candidate_size not in (224, 336):
        raise ValueError('Candidate size must be 224 or 336; the reference is fixed at 518')
    candidate, cp, ch = load_capture(candidate_dir, candidate_size, V17_CAPTURE_SHA)
    reference, rp, rh = load_capture(reference_dir, 518, V16_CAPTURE_SHA)
    source_identity = verify_identity(candidate, reference)
    gt_sha = geometry.sha(groundtruth)
    times, poses, gt_info = geometry.read_gt(groundtruth)
    K_calibrated = np.asarray(candidate['rectification']['output_K'], np.float64)
    result = dict(kind='v17_three_fixed_pair_resolution_numerical_diagnostic', status='completed',
                  candidate_image_size=candidate_size, reference_image_size=518,
                  analyzer_sha256=geometry.sha(__file__), frozen_v16_numerical_dependency_sha256=V16_COMPARISON_SHA,
                  geometry_dependency_sha256=geometry.sha(geometry.__file__),
                  model_inference_executed=False, image_decode_executed=False, tracking_executed=False,
                  board_operations_executed=False, input_identity_verified=True, source_identity=source_identity,
                  raw_inputs_modified=False, full_dataset_ATE_RPE_validated=False, accuracy_gate_validated=False,
                  metric_scale_validated=False, real_time_validated=False,
                  provenance=dict(candidate_capture=ch, reference518_capture=rh,
                    candidate_producer_sha256=V17_CAPTURE_SHA, reference_producer_sha256=V16_CAPTURE_SHA,
                    checkpoint_sha256=v16.CHECKPOINT_SHA, rgb_index_sha256=candidate['rgb_index']['sha256'],
                    calibration_sha256=candidate['calibration']['sha256'],
                    candidate_source_hashes=candidate['source_hashes'], reference_source_hashes=reference['source_hashes'],
                    schedule_metrics_sha256=candidate['schedule']['metrics_sha256'],
                    schedule_events_sha256=candidate['schedule']['events_sha256']),
                  groundtruth=dict(sha256=gt_sha, **gt_info, used_after_capture_only=True,
                    pose_convention='T_world_from_camera; qx qy qz qw; R=I rectification preserves axes',
                    association='same nearest original GT sample for both resolutions',
                    maximum_time_difference_seconds=v16.GT_TOLERANCE_SECONDS,
                    direction_frame='destination camera, relative T_second_from_first',
                    minimum_translation_for_direction_m=v16.MIN_GT_TRANSLATION_METRES),
                  sample_selection=candidate['schedule']['selection'], pairs=[])
    for c, r in zip(cp, rp):
        ids = c['record']['frame_ids']
        indices = [geometry.nearest(times, t, v16.GT_TOLERANCE_SECONDS) for t in ids]
        gt_relative = None if any(i is None for i in indices) else geometry.inverse(poses[indices[1]]) @ poses[indices[0]]
        matches = [dict(frame_id=t, GT_timestamp=None if i is None else float(times[i]),
                        difference_seconds=None if i is None else float(times[i] - t)) for t, i in zip(ids, indices)]
        pair = dict(revision=c['record']['revision'], frame_ids=ids, GT_matches=matches,
                    source_images=[dict(frame_id=image['frame_id'], basename=Path(image['path']).name,
                                        sha256=image['sha256']) for image in c['selected']['images']],
                    mapping_candidate=c['mapping'], mapping_reference518=r['mapping'],
                    relative_pose_convention='T_second_from_first=Tcw[1]@inverse(Tcw[0])',
                    resolution_rotation_difference_degrees=geometry.rotation_degrees(c['relative'][:3, :3] @ r['relative'][:3, :3].T),
                    resolution_translation_direction_difference_degrees=geometry.errors(c['relative'], r['relative'], 1e-12)['translation_direction_error_degrees'],
                    depth=compare_depth(c, r), predictions={})
        for label, row, size in (('candidate', c, candidate_size), ('reference518', r, 518)):
            pair['predictions'][label] = dict(image_size=size, relative_Tcw=row['relative'].tolist(),
                relative_translation_norm_arbitrary_units=float(np.linalg.norm(row['relative'][:3, 3])),
                GT_pair_diagnostic=None if gt_relative is None else geometry.errors(row['relative'], gt_relative, v16.MIN_GT_TRANSLATION_METRES),
                GT_unavailable_reason='no_original_GT_within_20ms' if gt_relative is None else None,
                model_inference_seconds=row['record']['model_inference_seconds'],
                K_original_pixels=[camera_stats(K, row['A'], K_calibrated) for K in row['arrays']['intrinsic'][0]])
        pair['K_candidate_minus_reference518_original_pixels'] = [(np.linalg.inv(c['A']) @ a - np.linalg.inv(r['A']) @ b).tolist()
            for a, b in zip(c['arrays']['intrinsic'][0], r['arrays']['intrinsic'][0])]
        result['pairs'].append(pair)
    for run, hashes in ((candidate_dir, ch), (reference_dir, rh)):
        if any(geometry.sha(Path(run) / name) != digest for name, digest in hashes.items()):
            raise ValueError('Saved capture input changed during analysis')
    if geometry.sha(groundtruth) != gt_sha:
        raise ValueError('Ground truth changed during analysis')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-size', type=int, choices=(224, 336), required=True)
    parser.add_argument('--reference-518', type=Path, required=True)
    parser.add_argument('--groundtruth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(argv)
    if a.output.exists():
        parser.error('Output must be a new JSON file')
    result = analyze(a.candidate, a.reference_518, a.groundtruth, a.candidate_size)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], candidate_image_size=a.candidate_size,
                          pairs=len(result['pairs']), output=a.output.name, sha256=geometry.sha(a.output))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
