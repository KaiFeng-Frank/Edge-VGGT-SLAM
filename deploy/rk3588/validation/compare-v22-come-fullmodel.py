#!/usr/bin/env python3
"""Compare saved original/merged VGGT outputs, without inference or images.

The complete three-pair cohort is fixed before scoring. All depth pixels in
both views contribute, including borders and low-confidence/merged regions.
One descriptive positive least-squares depth scale is fitted and scored on
the same two-view pixels. Pose differences are against original VGGT outputs,
not GT, ATE/RPE, a metric trajectory, or a realtime acceptance test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
REFERENCE_SHA = '8a5b9e8a7f94757fa906fbcc1b2d37ae3d4751076d69e966dd626a15a6c63598'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CHECKPOINT_BYTES = 5026874952
KEYS = ('depth', 'depth_conf', 'extrinsic', 'intrinsic', 'pose_enc')
ORDER = ((0, 'original'), (0, 'merged'), (0, 'merged'), (0, 'original'),
         (1, 'original'), (1, 'merged'), (2, 'merged'), (2, 'original'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order='C')).hexdigest()


def digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a lowercase SHA256')
    return value


def strict_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def artifact_path(directory, name):
    if not isinstance(name, str) or not name or Path(name).name != name or Path(name).suffix != '.npz':
        raise ValueError('Prediction artifacts must be local NPZ basenames')
    path = Path(directory) / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != Path(directory).resolve():
        raise ValueError('Prediction artifact must be an owned regular file in the capture directory')
    return path


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and tensor_sha(a) == tensor_sha(b)


def metadata_sha(metadata):
    keys = [key for key in ('sha256', 'tensor_sha256', 'rawsha') if key in metadata]
    if not keys:
        raise ValueError('Missing raw tensor SHA256')
    values = [digest(metadata[key], 'raw tensor SHA') for key in keys]
    if len(set(values)) != 1:
        raise ValueError('Conflicting raw tensor SHA256 fields')
    return values[0]


def validate_frame_ids(values):
    if (not isinstance(values, list) or len(values) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
            or not values[0] < values[1]):
        raise ValueError('Two distinct increasing finite original frame IDs are required')


def check_rigid(matrix):
    R = np.asarray(matrix, dtype=np.float64)[:3, :3]
    if (not np.isfinite(matrix).all()
            or not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=5e-5)
            or not np.isclose(np.linalg.det(R), 1, rtol=0, atol=5e-5)):
        raise ValueError('Saved extrinsic rotation must be a finite proper rotation')


def load_arrays(directory, artifact, metadata, frame_ids, input_shape):
    path = artifact_path(directory, artifact['file'])
    expected = digest(artifact['sha256'], 'artifact SHA')
    if sha(path) != expected or path.stat().st_size != artifact['bytes']:
        raise ValueError('Prediction artifact file SHA/size differs from its receipt')
    if (len(input_shape) != 4 or input_shape[:2] != [2, 3]
            or any(type(value) is not int or value <= 0 for value in input_shape)):
        raise ValueError('Invalid original prepared RGB input shape')
    h, w = input_shape[2:]
    shapes = {'depth': (1, 2, h, w, 1), 'depth_conf': (1, 2, h, w),
              'extrinsic': (1, 2, 3, 4), 'intrinsic': (1, 2, 3, 3), 'pose_enc': (1, 2, 9)}
    if set(metadata) != set(KEYS):
        raise ValueError('Exactly five canonical prediction metadata keys are required')
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(KEYS) | {'frame_ids'}:
            raise ValueError('Exactly five canonical arrays and frame_ids are required')
        ids = saved['frame_ids']
        if ids.dtype != np.float64 or not exact(ids, np.asarray(frame_ids, dtype=np.float64)):
            raise ValueError('Saved Float64 frame_ids differ from the exact source pair')
        arrays = {key: saved[key] for key in KEYS}
    for key, value in arrays.items():
        info = metadata[key]
        if (value.dtype != np.float32 or value.shape != shapes[key]
                or list(value.shape) != info['shape'] or info.get('dtype') != 'float32'
                or value.nbytes != info['bytes'] or not np.isfinite(value).all()
                or tensor_sha(value) != metadata_sha(info)):
            raise ValueError(f'Saved {key} shape/dtype/finite/raw SHA differs from its receipt')
    for E in arrays['extrinsic'][0]:
        check_rigid(E)
    for K in arrays['intrinsic'][0]:
        if K[0, 0] <= 0 or K[1, 1] <= 0 or np.linalg.det(K.astype(np.float64)) <= 0:
            raise ValueError('Saved intrinsic must be invertible with positive focal lengths')
    return arrays, path, expected


def numerical_error(actual, reference):
    a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Comparison requires finite equally shaped saved arrays')
    difference = a - b
    denominator = float(np.linalg.norm(b.ravel()))
    numerator = float(np.linalg.norm(difference.ravel()))
    if not math.isfinite(denominator) or not math.isfinite(numerator):
        raise ValueError('Nonfinite numerical norm')
    return dict(elements=int(a.size), array_equal=bool(np.array_equal(actual, reference)),
                bitwise_equal=exact(np.asarray(actual), np.asarray(reference)),
                max_absolute=float(np.max(np.abs(difference))),
                mean_absolute=float(np.mean(np.abs(difference))),
                rmse=float(np.sqrt(np.mean(difference * difference))),
                relative_l2=None if denominator == 0 else numerator / denominator,
                relative_l2_unavailable_reason='zero_reference_norm' if denominator == 0 else None)


def compare_depth(merged, original):
    """Use every pixel; one shared positive scale for the two-frame batch."""
    actual, reference = np.asarray(merged, np.float64), np.asarray(original, np.float64)
    if actual.shape != reference.shape or actual.ndim != 5 or actual.shape[:2] != (1, 2) or actual.shape[-1] != 1:
        raise ValueError('Depth must preserve the original B1, two-view, Z-depth shape')
    raw = numerical_error(merged, original)
    source, target = actual.ravel(), reference.ravel()
    denominator = float(np.dot(source, source))
    numerator = float(np.dot(source, target))
    scale = numerator / denominator if denominator > 0 else None
    reason = None
    if scale is None or not math.isfinite(scale) or scale <= 0:
        reason, scale = 'no_finite_strictly_positive_unconstrained_least_squares_scale', None
    aligned = None if scale is None else numerical_error(actual * scale, reference)
    for result, values in ((raw, actual), (aligned, None if scale is None else actual * scale)):
        if result is not None:
            result['mean_absolute_relative'] = (float(np.mean(np.abs(values - reference) / reference))
                if bool((reference > 0).all()) else None)
            result['mean_absolute_relative_unavailable_reason'] = None if bool((reference > 0).all()) else 'nonpositive_original_depth'
    return dict(units='arbitrary_original_VGGT_batch_camera_Z_units', pixels=int(source.size),
                excluded_pixels=0, all_pixels_in_both_views=True,
                confidence_border_and_merge_mask_filtering=False,
                original_nonpositive_pixels=int(np.count_nonzero(target <= 0)),
                merged_nonpositive_pixels=int(np.count_nonzero(source <= 0)),
                raw=raw, positive_scale_merged_to_original=scale,
                scale_unavailable_reason=reason, scale_aligned=aligned,
                scale_equation='s=dot(merged_depth,original_depth)/dot(merged_depth,merged_depth); require s>0',
                scale_scope='one scalar shared by both views of this prediction batch',
                scale_fit_and_score_use_same_pixels=True,
                scale_alignment_is_descriptive_not_independent_validation=True,
                depth_GT_used=False)


def relative_pose(extrinsics):
    if np.asarray(extrinsics).shape != (1, 2, 3, 4):
        raise ValueError('Two OpenCV world-to-camera extrinsics are required')
    first, second = np.asarray(extrinsics, np.float64)[0]
    check_rigid(first)
    check_rigid(second)
    R = second[:, :3] @ first[:, :3].T
    t = second[:, 3] - R @ first[:, 3]
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = R, t
    return result


def compare_pose(merged, original):
    actual, reference = relative_pose(merged), relative_pose(original)
    error_R = actual[:3, :3] @ reference[:3, :3].T
    sin_angle = np.linalg.norm(np.array([error_R[2, 1] - error_R[1, 2],
                                        error_R[0, 2] - error_R[2, 0],
                                        error_R[1, 0] - error_R[0, 1]])) / 2
    cos_angle = np.clip((np.trace(error_R) - 1) / 2, -1., 1.)
    rotation = float(np.degrees(np.arctan2(sin_angle, cos_angle)))
    ta, tb = actual[:3, 3], reference[:3, 3]
    na, nb = float(np.linalg.norm(ta)), float(np.linalg.norm(tb))
    direction = None
    if na > 1e-12 and nb > 1e-12:
        direction = float(np.degrees(np.arccos(np.clip(np.dot(ta, tb) / (na * nb), -1., 1.))))
    return dict(convention='T_second_from_first: R=R1@R0.T; t=t1-R@t0; destination-camera coordinates',
                merged_T_second_from_first=actual.tolist(), original_T_second_from_first=reference.tolist(),
                rotation_difference_degrees=rotation, translation_direction_difference_degrees=direction,
                translation_direction_unavailable_reason=None if direction is not None else 'near_zero_saved_relative_translation',
                near_zero_translation_threshold_arbitrary_units=1e-12,
                merged_translation_norm_arbitrary_units=na, original_translation_norm_arbitrary_units=nb,
                GT_used=False, trajectory_ATE_RPE_validated=False, metric_scale_validated=False)


def load_reference(directory, expected_sha):
    path = Path(directory) / 'capture.json'
    if sha(path) != digest(expected_sha, 'original reference receipt SHA'):
        raise ValueError('Frozen original 336 reference receipt SHA differs')
    receipt = strict_json(path)
    if (receipt.get('status') != 'completed_fixed_input_diagnostic' or receipt.get('image_size') != 336
            or receipt.get('max_pairs') != 3 or receipt.get('original_weights') is not True
            or receipt.get('device') != 'cpu' or receipt.get('precision') != 'float32'
            or receipt.get('ORB_used') is not False or receipt.get('npu_used') is not False
            or receipt['checkpoint']['sha256'] != CHECKPOINT_SHA or receipt['checkpoint']['bytes'] != CHECKPOINT_BYTES
            or len(receipt['pairs']) != 3 or len(receipt['selected_inputs']) != 3):
        raise ValueError('Expected the fixed original CPU FP32 three-pair 336 reference')
    records, hashes = [], {'capture.json': expected_sha}
    for index, (record, selected) in enumerate(zip(receipt['pairs'], receipt['selected_inputs'])):
        ids = record['frame_ids']
        validate_frame_ids(ids)
        if (record['revision'] != index + 1 or selected['revision'] != index + 1
                or selected['frame_ids'] != ids or len(selected['images']) != 2):
            raise ValueError('Frozen reference pair revision/identity differs')
        for fid, image in zip(ids, selected['images']):
            if (image['frame_id'] != fid or image['timestamp'] != fid
                    or image['schedule_token']['frame_id'] != fid or image['schedule_token']['timestamp'] != fid
                    or image.get('samefile_validated') is not True):
                raise ValueError('Frozen reference lacks exact source/timestamp/samefile admission')
            digest(image['sha256'], 'source image SHA')
        arrays, artifact, actual_sha = load_arrays(directory, record, record['arrays'], ids, record['input_tensor']['shape'])
        records.append(dict(record=record, selected=selected, arrays=arrays))
        hashes[artifact.name] = actual_sha
    return receipt, records, hashes


def analyze(fullmodel_dir, reference_dir, expected_fullmodel_sha256, *, expected_reference_sha256=REFERENCE_SHA):
    """The CLI pins the real reference; injection is only for synthetic tests."""
    receipt_path = Path(fullmodel_dir) / 'fullmodel.json'
    receipt_sha = digest(expected_fullmodel_sha256, 'fullmodel receipt SHA')
    if sha(receipt_path) != receipt_sha:
        raise ValueError('Fullmodel receipt SHA differs from the supplied measurement pin')
    receipt = strict_json(receipt_path)
    reference, references, reference_hashes = load_reference(reference_dir, expected_reference_sha256)
    if (receipt.get('kind') != 'isolated_original_weight_MLP_only_CoMe_fullmodel_research'
            or receipt.get('status') != 'completed' or receipt.get('device') != 'cpu'
            or receipt.get('precision') != 'float32' or receipt.get('original_weights') is not True
            or receipt.get('npu_used') is not False or receipt.get('ORB_used') is not False
            or receipt.get('GT_used') is not False or receipt.get('inference_on_board_only') is not True
            or receipt.get('full_CoMe_method_implemented') is not False
            or receipt.get('full_model_forward_attempts') != 8 or receipt.get('full_model_forward_count') != 8
            or receipt.get('warmup_forwards') != 0 or receipt.get('cleanup_errors') != []
            or receipt.get('sources_and_inputs_unchanged') is not True
            or receipt.get('cpu_affinity_restored') is not True
            or receipt.get('all_research_contexts_closed') is not True
            or receipt.get('original_forward_identities_and_hooks_restored') is not True
            or receipt.get('actual_research_predictor_calls') != 4
            or receipt.get('actual_research_MLP_calls') != 228):
        raise ValueError('Expected a completed board-only CPU eight-call MLP-only experiment with complete cleanup')
    reference_pin = receipt.get('original_reference_sha256', receipt.get('reference_capture_sha256'))
    if reference_pin != expected_reference_sha256:
        raise ValueError('Fullmodel original reference binding differs')
    if (receipt.get('image_size') != 336 or receipt['checkpoint']['sha256'] != CHECKPOINT_SHA
            or receipt['checkpoint']['bytes'] != CHECKPOINT_BYTES
            or receipt['selected_inputs'] != reference['selected_inputs']
            or receipt['rectification'] != reference['rectification']):
        raise ValueError('Fullmodel source cohort/checkpoint/rectification differs from the original reference')
    for relative, expected in reference['source_hashes'].items():
        if receipt['source_hashes'].get(relative) != expected:
            raise ValueError('Original model source SHA differs from the reference')
    rows = receipt['call_records']
    if (len(rows) != len(ORDER)
            or [(row['pair_index'], row['method']) for row in rows] != list(ORDER)
            or [row['order_index'] for row in rows] != list(range(len(ORDER)))):
        raise ValueError('All eight original/merged calls must preserve the predefined three-pair ABBA/AB/BA order')
    file_hashes = {'fullmodel.json': receipt_sha}
    comparisons, grouped = [], {(i, method): [] for i in range(3) for method in ('original', 'merged')}
    for row in rows:
        i, method = row['pair_index'], row['method']
        ref = references[i]
        ids = row['frame_ids']
        validate_frame_ids(ids)
        if ids != ref['record']['frame_ids'] or row['input_tensor_sha256'] != ref['record']['input_tensor']['tensor_sha256']:
            raise ValueError('Exact source frame IDs/normalized input tensor SHA differs')
        images = row['pair_source_images']
        if len(images) != 2 or any(actual['path'] != expected['path'] or actual['sha256'] != expected['sha256']
                                  for actual, expected in zip(images, ref['selected']['images'])):
            raise ValueError('Call source image path/SHA differs from the admitted pair')
        if type(row['forward_seconds']) not in (float, int) or not math.isfinite(row['forward_seconds']) or row['forward_seconds'] <= 0:
            raise ValueError('Every call must record a finite positive board forward duration')
        arrays, file, actual_sha = load_arrays(fullmodel_dir, row['artifact'], row['arrays'], ids, ref['record']['input_tensor']['shape'])
        if file.name in file_hashes:
            raise ValueError('Every measured call must have a distinct saved prediction artifact')
        file_hashes[file.name] = actual_sha
        if method == 'original' and not all(exact(arrays[key], ref['arrays'][key]) for key in KEYS):
            raise ValueError('An original call differs bitwise from the frozen original 336 reference')
        if method == 'merged':
            metrics, scene = row['research_metrics'], row['scene_record']
            if (metrics.get('closed') is not True or metrics.get('active_scene') is not False
                    or metrics.get('cleanup_errors') != [] or metrics.get('original_weight_hashes_unchanged') is not True
                    or metrics.get('scenes_started') != 1 or metrics.get('scenes_completed') != 1
                    or metrics.get('scenes_failed') != 0 or metrics.get('predictor_calls') != 1
                    or metrics.get('plan_builds') != 1 or metrics.get('mlp_calls') != {'dino': 9, 'frame': 24, 'global': 24}
                    or scene.get('status') != 'completed' or scene.get('mlp_calls') != metrics['mlp_calls']
                    or scene.get('synthetic_plan_callback_used') is not False
                    or scene.get('synthetic_predictor_callback_used') is not False):
                raise ValueError('Merged call lacks one fresh real predictor/57-MLP scene and restored original weights')
        result = dict(pair_index=i, order_index=row['order_index'], method=method, frame_ids=ids,
                      artifact=dict(file=file.name, sha256=actual_sha, bytes=file.stat().st_size),
                      input_tensor_sha256=row['input_tensor_sha256'], forward_seconds=float(row['forward_seconds']),
                      errors_against_frozen_original={key: numerical_error(arrays[key], ref['arrays'][key]) for key in KEYS},
                      all_pixel_depth=compare_depth(arrays['depth'], ref['arrays']['depth']),
                      relative_pose=compare_pose(arrays['extrinsic'], ref['arrays']['extrinsic']))
        comparisons.append(result)
        grouped[(i, method)].append((row['order_index'], arrays, float(row['forward_seconds'])))
    if receipt.get('research_contexts') != [row['research_metrics'] for row in rows if row['method'] == 'merged']:
        raise ValueError('Terminal research contexts differ from the four ordered merged-call records')
    pairs = []
    for i, reference_pair in enumerate(references):
        repeats = {}
        timings = {}
        for method in ('original', 'merged'):
            calls = grouped[(i, method)]
            repeats[method] = dict(call_indices=[item[0] for item in calls],
                compared_repeat_calls=max(0, len(calls) - 1),
                arrays_bitwise_equal={key: all(exact(item[1][key], calls[0][1][key]) for item in calls[1:]) for key in KEYS},
                repeated_stability_exercised=len(calls) > 1)
            values = [item[2] for item in calls]
            timings[method] = dict(calls=len(values), seconds=values, mean_seconds=float(np.mean(values)))
        pairs.append(dict(pair_index=i, frame_ids=reference_pair['record']['frame_ids'],
                          source_images=[dict(basename=Path(image['path']).name, sha256=image['sha256'])
                                         for image in reference_pair['selected']['images']],
                          repeats=repeats, recorded_board_forward=timings))
    for directory, hashes in ((fullmodel_dir, file_hashes), (reference_dir, reference_hashes)):
        if any(sha(Path(directory) / file) != expected for file, expected in hashes.items()):
            raise ValueError('Saved input changed during numerical comparison')
    return dict(kind='three_fixed_pair_CoMe_isolated_fullmodel_saved_output_comparison', status='completed',
                comparator_sha256=sha(__file__),
                scope=dict(saved_array_math_only=True, host_model_inference_executed=False,
                           image_decoding_executed=False, board_operations_executed=False,
                           GT_used=False, metric_scale_validated=False, full_dataset_ATE_RPE_validated=False,
                           complete_CoMe_algorithm_validated=False, accuracy_acceptance_gate_applied=False,
                           realtime_validated=False),
                provenance=dict(fullmodel_receipt_sha256=receipt_sha, original_336_receipt_sha256=expected_reference_sha256,
                                checkpoint_sha256=CHECKPOINT_SHA, source_sha256=receipt['source_hashes'],
                                fullmodel_artifacts=file_hashes, original_336_artifacts=reference_hashes),
                cohort=dict(pairs=3, calls=8, complete_predeclared_call_order_preserved=True,
                            order=[dict(pair_index=i, method=method) for i, method in ORDER],
                            no_quality_based_pair_or_call_selection=True,
                            original_calls_bitwise_bound_to_frozen_reference=True),
                timing_scope=receipt.get('timing_scope'),
                timing_limitations='Recorded board calls are ordered in one experiment; thermal and clock conditions are not controlled. No whole-pipeline or causal speedup claim is made.',
                numerical_limitations='One scale is fitted and scored on the same complete two-view depth grid. This is descriptive agreement with original VGGT, not heldout accuracy or GT. Pair-relative pose differences are not trajectory ATE/RPE.',
                pairs=pairs, calls=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fullmodel-dir', type=Path, required=True)
    parser.add_argument('--fullmodel-sha256', required=True)
    parser.add_argument('--original-336-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(argv)
    if a.output.exists():
        parser.error('Output must be a new JSON file')
    result = analyze(a.fullmodel_dir, a.original_336_dir, a.fullmodel_sha256)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], pairs=len(result['pairs']), calls=len(result['calls']),
                          output=a.output.name, sha256=sha(a.output))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
