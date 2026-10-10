#!/usr/bin/env python3
"""Independent NumPy-only audit of saved four-scene MLP-merge plans.

This script does not import Torch, the predictor, a model or the merge helper.
It checks bit encodings, arithmetic and index identities in saved arrays.
It neither measures inference nor sets a reconstruction-quality threshold.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CAPTURE_SHA = '3664f5c932602a3c6538ea6013ea0389625c42bdcb7dcf10a9fbceb9304abeb1'
PROBE_SHA = '5abe4899111361fa3f0ba4b1fab78e3c15e984dbe358e2de05a001e0e8936f91'
REFERENCE_SHA = '8a5b9e8a7f94757fa906fbcc1b2d37ae3d4751076d69e966dd626a15a6c63598'
DRIVER_SHA = '47cffa63bfd33ec6fecc66fe62093aa1119882a9bf13ae6cf4691f46f62af3bc'
WRAPPER_SHA = '67c6c720ee2b1a7b698f9b7e0d8a5599115f6b06e638659fdcba5c2008db8716'
MERGER_SHA = '7acb792a94a5e4cb55bed25674e1e30ecf8692296e6e7fa4d7e98bb9da26bf5d'
ORDER = ((0, 'original'), (0, 'merged'), (0, 'merged'), (0, 'original'),
         (1, 'original'), (1, 'merged'), (2, 'merged'), (2, 'original'))
KEYS = {'features', 'predictor_features_bf16_bits', 'confidence', 'confidence_bf16_bits',
        'group_means', 'group_means_bf16_bits', 'median_threshold', 'merge_mask',
        'reverse_indices', 'view_offsets', 'token_weights', 'merged_patch_mask',
        'representative_indices', 'mean_input_indices', 'mean_output_indices'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def raw_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order='C')).hexdigest()


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and raw_sha(a) == raw_sha(b)


def expand_bf16(bits):
    if np.asarray(bits).dtype != np.uint16:
        raise ValueError('BF16 bit storage must be uint16')
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def fp32_to_bf16_bits(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Only finite saved FP32 values can be rounded')
    raw = value.view(np.uint32)
    return ((raw + np.uint32(0x7fff) + ((raw >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def reference_plan(mask):
    """Enumerate original patch rows explicitly; no implementation import."""
    if mask.dtype != np.bool_ or mask.shape != (2, 6, 8):
        raise ValueError('Fixed two-view 18x24 patch grid requires bool [2,6,8]')
    reverse = np.empty((2, 437), dtype=np.int64)
    representatives, means, outputs, weights, offsets = [], [], [], [], [0]
    cursor = 0
    for view in range(2):
        for special in range(5):
            reverse[view, special] = cursor
            representatives.append(view * 437 + special)
            weights.append(1.)
            cursor += 1
        for group_y in range(6):
            for group_x in range(8):
                source = [5 + (group_y * 3 + row) * 24 + group_x * 3 + column
                          for row in range(3) for column in range(3)]
                if mask[view, group_y, group_x]:
                    reverse[view, source] = cursor
                    representatives.append(view * 437 + source[0])
                    means.append([view * 437 + token for token in source])
                    outputs.append(cursor)
                    weights.append(9.)
                    cursor += 1
                else:
                    for token in source:
                        reverse[view, token] = cursor
                        representatives.append(view * 437 + token)
                        weights.append(1.)
                        cursor += 1
        offsets.append(cursor)
    patch_mask = np.empty((2, 18, 24), np.bool_)
    for view in range(2):
        for row in range(18):
            for column in range(24):
                patch_mask[view, row, column] = mask[view, row // 3, column // 3]
    return dict(reverse_indices=reverse, representative_indices=np.asarray(representatives, np.int64),
                mean_input_indices=np.asarray(means, np.int64).reshape(-1, 9),
                mean_output_indices=np.asarray(outputs, np.int64),
                token_weights=np.asarray(weights, np.float32), view_offsets=np.asarray(offsets, np.int64),
                merged_patch_mask=patch_mask)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def audit(run_dir, fullmodel_sha, capture_dir, probe_dir):
    checks = []
    report = dict(kind='independent_four_scene_saved_BF16_and_token_layout_audit',
                  script_sha256=sha(__file__), status='started', checks=checks, scenes=[],
                  host_execution='NumPy saved-array arithmetic only; no Torch, predictor, model, image decoder, board or GT',
                  no_quality_gate_applied=True, complete_CoMe_model_validated=False,
                  realtime_validated=False)

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise ValueError(name)

    def pinned_file(directory, record, name=None):
        filename = name if name is not None else record['file']
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError('Artifact must have a local basename')
        path = Path(directory) / filename
        check('regular_file_' + filename, path.is_file() and not path.is_symlink())
        check('file_SHA_and_size_' + filename, sha(path) == record['sha256'] and path.stat().st_size == record['bytes'])
        return path

    try:
        full_path = Path(run_dir) / 'fullmodel.json'
        check('fullmodel_receipt_supplied_pin', len(fullmodel_sha) == 64 and sha(full_path) == fullmodel_sha)
        row = json.loads(full_path.read_text())
        check('capture_and_probe_receipt_pins', sha(Path(capture_dir) / 'capture.json') == CAPTURE_SHA and sha(Path(probe_dir) / 'probe.json') == PROBE_SHA)
        capture = json.loads((Path(capture_dir) / 'capture.json').read_text())
        probe = json.loads((Path(probe_dir) / 'probe.json').read_text())
        check('completed_original_weight_MLP_only_measurement', row['status'] == 'completed' and row['reference_capture_sha256'] == REFERENCE_SHA and row['stage_capture_sha256'] == CAPTURE_SHA and row['script_sha256'] == DRIVER_SHA and row['research_sources'] == dict(wrapper_sha256=WRAPPER_SHA, merger_sha256=MERGER_SHA) and row['npu_used'] is False and row['ORB_used'] is False and row['GT_used'] is False and row['full_CoMe_method_implemented'] is False)
        calls = row['call_records']
        check('complete_ordered_eight_calls', len(calls) == 8 and [(v['pair_index'], v['method']) for v in calls] == list(ORDER) and [v['order_index'] for v in calls] == list(range(8)))
        check('complete_forward_and_fresh_predictor_counts', row['full_model_forward_attempts'] == 8 and row['full_model_forward_count'] == 8 and row['actual_research_predictor_calls'] == 4 and row['actual_research_MLP_calls'] == 228)
        check('terminal_cleanup_source_guards', row['cleanup_errors'] == [] and row['sources_and_inputs_unchanged'] is True and row['cpu_affinity_restored'] is True and row['all_research_contexts_closed'] is True and row['original_forward_identities_and_hooks_restored'] is True)
        capture_values = load_npz(pinned_file(capture_dir, capture['artifacts']['activations.npz'], 'activations.npz'))
        isolated = load_npz(pinned_file(probe_dir, probe['artifacts']['scene-plan.npz'], 'scene-plan.npz'))
        expected_first_features = np.ascontiguousarray(capture_values['predictor_features'].transpose(0, 1, 3, 4, 2))
        expected_first_raw = capture_values['dino14_output'][:, 5:].reshape(1, 2, 18, 24, 1024)
        check('original_capture_feature_layout_crosscheck', same(expected_first_features, expected_first_raw))
        scenes = {}
        prediction_files = set()
        expected_layers = {('dino', layer) for layer in range(15, 24)} | {('frame', layer) for layer in range(24)} | {('global', layer) for layer in range(24)}
        for call in calls:
            index = call['order_index']
            prefix = f'call_{index}_'
            path = pinned_file(run_dir, call['artifact'])
            check(prefix + 'distinct_canonical_output', path.name not in prediction_files)
            prediction_files.add(path.name)
            check(prefix + 'source_pair_identity', call['frame_ids'] == row['selected_inputs'][call['pair_index']]['frame_ids'] and call['pair_source_images'] == [dict(path=image['path'], sha256=image['sha256']) for image in row['selected_inputs'][call['pair_index']]['images']])
            if call['method'] == 'original':
                check(prefix + 'no_merge_scene_on_original', 'scene_artifact' not in call and 'scene_record' not in call and 'research_metrics' not in call)
                continue
            path = pinned_file(run_dir, call['scene_artifact'])
            values = load_npz(path)
            check(prefix + 'all_scene_keys', set(values) == KEYS)
            metadata = call['scene_artifact']['arrays']
            check(prefix + 'all_nonempty_array_metadata', set(metadata) == {key for key, value in values.items() if value.size})
            for key, value in values.items():
                check(prefix + key + '_finite', bool(np.isfinite(value).all()))
                if value.size:
                    info = metadata[key]
                    check(prefix + key + '_raw_SHA_dtype_shape', list(value.shape) == info['shape'] and str(value.dtype) == info['dtype'] and value.nbytes == info['bytes'] and raw_sha(value) == info['sha256'])
            features = values['features']
            check(prefix + 'raw_DINO14_feature_shape', features.shape == (1, 2, 18, 24, 1024) and features.dtype == np.float32)
            bits = values['predictor_features_bf16_bits']
            check(prefix + 'FP32_to_BF16_RNE_predictor_input', same(bits, fp32_to_bf16_bits(features)))
            check(prefix + 'BF16_predictor_input_finite', bool(np.isfinite(expand_bf16(bits)).all()))
            for key, shape in [('confidence', (2, 432, 1)), ('group_means', (2, 6, 8))]:
                actual, raw_bits = values[key], values[key + '_bf16_bits']
                check(prefix + key + '_BF16_exact_expansion', actual.shape == shape and actual.dtype == np.float32 and raw_bits.shape == shape and same(expand_bf16(raw_bits), actual))
            check(prefix + 'expp1_confidence_range', bool((values['confidence'] >= 1).all()))
            groups = np.empty((2, 6, 8), dtype=np.float64)
            confidence = values['confidence'].reshape(2, 18, 24)
            for view in range(2):
                for gy in range(6):
                    for gx in range(8):
                        region = confidence[view, gy * 3:gy * 3 + 3, gx * 3:gx * 3 + 3]
                        groups[view, gy, gx] = np.sum(region.astype(np.float64)) / 9
            rounded = fp32_to_bf16_bits(groups.astype(np.float32))
            check(prefix + 'independent_3x3_BF16_group_mean_RNE', same(rounded, values['group_means_bf16_bits']))
            sorted_means = np.sort(values['group_means'].reshape(-1).astype(np.float64))
            median = np.asarray([[(sorted_means[47] + sorted_means[48]) / 2]], dtype=np.float32)
            check(prefix + 'cross_view_FP32_linear_median', same(median, values['median_threshold']))
            mask = values['group_means'] < median.item()
            check(prefix + 'strict_less_than_median_mask_no_forced_quota', same(mask, values['merge_mask']) and int(mask.sum()) <= 48)
            expected = reference_plan(mask)
            for key, value in expected.items():
                check(prefix + 'independent_' + key, same(values[key], value))
            reverse, offsets, reps = expected['reverse_indices'], expected['view_offsets'], expected['representative_indices']
            compact = len(reps)
            for view in range(2):
                selected = reverse[view]
                check(prefix + f'view_{view}_range_and_specials', bool(((selected >= offsets[view]) & (selected < offsets[view + 1])).all()) and np.array_equal(selected[:5], np.arange(offsets[view], offsets[view] + 5)))
                check(prefix + f'view_{view}_representatives_do_not_cross', bool((reps[offsets[view]:offsets[view + 1]] // 437 == view).all()))
            mean_ids = expected['mean_input_indices']
            check(prefix + 'no_special_averaged_or_cross_view_group', bool((mean_ids % 437 >= 5).all()) and bool(((mean_ids // 437) == (mean_ids[:, :1] // 437)).all()))
            multiplicities = np.bincount(reverse.reshape(-1), minlength=compact).astype(np.float32)
            check(prefix + 'compact_multiplicity_weights_and_total_rows', same(multiplicities, values['token_weights']) and float(multiplicities.sum()) == 874 and compact == 874 - 8 * int(mask.sum()))
            scene, metrics = call['scene_record'], call['research_metrics']
            plan = scene['plan']
            check(prefix + 'plan_row_and_region_counts', plan['original_rows'] == 874 and plan['compact_rows'] == compact and plan['merged_groups'] == int(mask.sum()) and plan['total_groups'] == 96 and plan['merged_patch_tokens'] == 9 * int(mask.sum()) and plan['unmerged_patch_tokens'] == 864 - 9 * int(mask.sum()) and plan['view_offsets'] == offsets.tolist() and plan['special_tokens_per_view'] == 5 and plan['grid_hw'] == [18, 24] and plan['group_hw'] == [3, 3] and plan['views'] == 2 and plan['residual_merged'] is False and plan['attention_merged'] is False and plan['token_weights_applied_to_MLP'] is False)
            check(prefix + 'fresh_57_original_MLP_branches', scene['mlp_calls'] == metrics['mlp_calls'] == {'dino': 9, 'frame': 24, 'global': 24} and len(scene['called_layers']) == 57 and {tuple(item) for item in scene['called_layers']} == expected_layers and metrics['full_mlp_rows'] == 874 * 57 and metrics['compact_mlp_rows'] == compact * 57)
            check(prefix + 'one_actual_predictor_and_closed_context', metrics['predictor_calls'] == 1 and metrics['plan_builds'] == 1 and metrics['scenes_started'] == metrics['scenes_completed'] == 1 and metrics['scenes_failed'] == 0 and metrics['closed'] is True and metrics['active_scene'] is False and metrics['cleanup_errors'] == [] and metrics['original_weight_hashes_unchanged'] is True and scene['synthetic_plan_callback_used'] is False and scene['synthetic_predictor_callback_used'] is False)
            bindings = {'features': (features, 'torch.float32'), 'predictor_features': (bits, 'torch.bfloat16'),
                        'confidence': (values['confidence_bf16_bits'], 'torch.bfloat16'),
                        'group_means': (values['group_means_bf16_bits'].reshape(2, 48), 'torch.bfloat16'),
                        'confidence_quantile': (values['median_threshold'], 'torch.float32')}
            check(prefix + 'complete_scene_record_tensor_pins', set(scene['arrays']) == set(bindings))
            for key, (value, dtype) in bindings.items():
                info = scene['arrays'][key]
                check(prefix + key + '_scene_record_pin', list(value.shape) == info['shape'] and dtype == info['dtype'] and value.nbytes == info['bytes'] and raw_sha(value) == info['sha256'])
            if call['pair_index'] == 0:
                check(prefix + 'first_pair_DINO14_features_equal_original', same(features, expected_first_features))
                for key in isolated:
                    check(prefix + 'first_pair_equals_isolated_' + key, same(values[key], isolated[key]))
            scenes[index] = values
            report['scenes'].append(dict(order_index=index, pair_index=call['pair_index'], frame_ids=call['frame_ids'],
                scene_file=path.name, scene_file_sha256=call['scene_artifact']['sha256'],
                features_sha256=raw_sha(features), confidence_sha256=raw_sha(values['confidence_bf16_bits']),
                mask_sha256=raw_sha(mask), merged_groups=int(mask.sum()), compact_rows=compact,
                MLP_calls=57, full_MLP_rows=874 * 57, compact_MLP_rows=compact * 57,
                all_specials_individual=True, no_cross_view_groups=True))
        check('all_four_ordered_merged_scenes', sorted(scenes) == [1, 2, 5, 6])
        for key in KEYS:
            check('first_pair_merged_repeats_exact_' + key, same(scenes[1][key], scenes[2][key]))
        check('terminal_ordered_context_records', row['research_contexts'] == [call['research_metrics'] for call in calls if call['method'] == 'merged'])
        check('receipts_unchanged_after_saved_array_analysis', sha(full_path) == fullmodel_sha and sha(Path(capture_dir) / 'capture.json') == CAPTURE_SHA and sha(Path(probe_dir) / 'probe.json') == PROBE_SHA)
        report.update(status='passed', fullmodel_receipt_sha256=fullmodel_sha, capture_receipt_sha256=CAPTURE_SHA,
                      isolated_probe_receipt_sha256=PROBE_SHA, four_fresh_predictor_calls=4,
                      original_MLP_branch_calls=228, canonical_output_artifacts_hash_verified=8,
                      first_pair_features_and_plan_equal_frozen_isolated_probe=True,
                      all_first_pair_repeated_scene_arrays_bitwise_equal=True)
    except Exception as exc:
        report.update(status='failed', error=str(exc))
    report['checks_passed'] = sum(item['passed'] for item in checks)
    report['checks_failed'] = sum(not item['passed'] for item in checks)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--fullmodel-sha256', required=True)
    parser.add_argument('--capture-dir', type=Path, default=Path('/dev/shm/vggt-v22-come-global0-capture-r1'))
    parser.add_argument('--probe-dir', type=Path, default=Path('/dev/shm/vggt-v22-come-global0-probe-r1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new receipt')
    result = audit(args.run_dir, args.fullmodel_sha256, args.capture_dir, args.probe_dir)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], passed=result['checks_passed'], failed=result['checks_failed'],
                         receipt=args.output.name, sha256=sha(args.output), error=result.get('error'))))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
