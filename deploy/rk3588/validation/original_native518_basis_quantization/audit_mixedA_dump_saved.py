#!/usr/bin/env python3
"""Independently audit saved untouched mixedA dump; NumPy/JSON only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

RESULT_SHA = 'ecd2d726c8b1edba0ffa6b5fc453c5e181e2983fa17a373ff51126984bb581f2'
CONTROL_SHA = '1ea35861e313659f7d803fa84ac9ed43f87fd98eaccc696f9cf1e77a968924a1'
PROFILE_SHA = '4013651f463a39475560b7c2e0544b3ce0fb813ece5a73e343cae9299e84d91b'
CAPTURE_SHA = 'cc2dd7fbddef8ee12de00b7a5cc88a16ba9d531783ab09ae8e66abe50df43408'
MODEL_SHA = '48621d9d6db93779215d35856f58d8db378a0779e359b3ea0063e28bc141b6d8'
SOURCES = {
    'dump_mixedA_calibration_board.py': 'a365611b48844170c2685c18ac31b8e96b05e2b5ae199ecb8298116ef79480ba',
    'profile_owned_context_r2.py': '5aa75a7630ecb9793773553da1f7351a97ca5bac728d4f01f9b74d83d3ca2426',
    'run_hybrid_pair_board.py': 'd3157c5d1cecc4de541e9ff9a0d2adbf02e4a0269018892f24cddecbe47fce30',
    'execute_mixedA_dump_host.py': '9d7e7ea48adb7eb52da8f04f75db921c50867991ebd16ff9a5a08cae8a152800',
    'npu_frequency_test.py': 'cb991cf03abc872e3590dd92f26fd0f5ac548d620c5ca791f9354537a9f793b9',
}
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, digest, size=None):
    path = Path(path)
    require(sha(path) == digest and (size is None or path.stat().st_size == size), 'File identity differs: ' + str(path))
    return path


def array(folder, row):
    name = Path(row['file'])
    require(not name.is_absolute() and '..' not in name.parts, 'Owned relative artifact required')
    a = np.load(pin(folder / name, row['sha256'], row['bytes']), allow_pickle=False)
    require(a.dtype == np.float32 and str(a.dtype) == row['dtype'] and list(a.shape) == row['shape'] and a.nbytes == row['tensor_bytes'], 'Actual dtype/shape/byte count differs')
    require(hashlib.sha256(a.tobytes(order='C')).hexdigest() == row['raw_sha256'], 'Actual C-order payload differs')
    finite = np.isfinite(a)
    stats = dict(finite=bool(finite.all()), finite_count=int(finite.sum()), nan_count=int(np.isnan(a).sum()),
        positive_inf_count=int(np.isposinf(a).sum()), negative_inf_count=int(np.isneginf(a).sum()))
    require(all(row[k] == v for k, v in stats.items()), 'Reported finite/NaN/Inf counts differ')
    require(sum(stats[k] for k in ('finite_count', 'nan_count', 'positive_inf_count', 'negative_inf_count')) == a.size, 'Element accounting differs')
    z = a[finite]
    return a, dict(file=row['file'], bytes=row['bytes'], sha256=row['sha256'], raw_sha256=row['raw_sha256'],
        shape=list(a.shape), dtype=str(a.dtype), tensor_bytes=a.nbytes, c_contiguous=bool(a.flags.c_contiguous),
        finite_value_range=[float(z.min()), float(z.max())] if z.size else None, **stats)


def bytes_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C')


def representation_relation(a, b):
    require(a.shape == b.shape and a.dtype == b.dtype == np.float32, 'Matched logical layouts required')
    x, y = np.ascontiguousarray(a), np.ascontiguousarray(b)
    bitdiff = x.view(np.uint32) != y.view(np.uint32)
    both_finite, both_nan = np.isfinite(x) & np.isfinite(y), np.isnan(x) & np.isnan(y)
    return dict(raw_byte_equal=bytes_equal(x, y), finite_value_unequal_count=int(np.count_nonzero(x[both_finite] != y[both_finite])),
        finite_pair_max_abs=float(np.max(np.abs(x[both_finite].astype(np.float64) - y[both_finite]))) if both_finite.any() else None,
        NaN_mask_equal=bool(np.array_equal(np.isnan(x), np.isnan(y))),
        positive_Inf_mask_equal=bool(np.array_equal(np.isposinf(x), np.isposinf(y))),
        negative_Inf_mask_equal=bool(np.array_equal(np.isneginf(x), np.isneginf(y))),
        raw_uint32_bit_unequal_count=int(bitdiff.sum()), both_NaN_bit_unequal_count=int(np.count_nonzero(bitdiff & both_nan)),
        non_both_NaN_bit_unequal_count=int(np.count_nonzero(bitdiff & ~both_nan)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); work = args.work_dir.resolve(); folder = work / 'mixedA-dump-r1'
    r = json.loads(pin(folder / 'result.json', RESULT_SHA).read_text())
    prior = json.loads(pin(work / 'calibration-profile-r2/result.json', PROFILE_SHA).read_text())
    control = json.loads(pin(work / 'mixedA-dump-control-r1/control.json', CONTROL_SHA).read_text())
    require(r['schema'] == 'v128_original_mixedA_calibration_pair_untouched_RKNN_debug_dump' and r['status'] == 'completed_mixedA_debug_dump_final_byte_equal_not_root_cause_or_quality_accepted', 'Wrong actual dump receipt')
    require(r['script_sha256'] == SOURCES['dump_mixedA_calibration_board.py'] and r['model_sha256'] == MODEL_SHA and r['profile_receipt']['sha256'] == PROFILE_SHA and r['capture_receipt']['sha256'] == CAPTURE_SHA, 'Untouched source/model/input provenance differs')
    require(r['input_id'] == 'pair0' and r['input_split'] == 'calibration' and r['init_flags'] == 0 and r['core_mask'] == 1 and not r['heldout_arrays_accessed'], 'Calibration-only flags0/core1 scope differs')
    require(not r['candidate_accepted'] and not r['numerical_acceptance'] and not r['graph_recompiled_or_outputs_appended'] and not r['production_finite_guard_modified'] and not r['nonfinite_values_rewritten'], 'Numeric diagnostic scope differs')
    require(not r['cleanup_errors'] and all(r[k] for k in ('source_and_inputs_unchanged', 'environment_restored', 'affinity_restored', 'file_size_limit_restored', 'default_dump_absent_before', 'default_dump_absent_after')), 'Incomplete worker restoration')
    source_rows = []
    for name, digest in SOURCES.items():
        path = pin(work / 'executed-mixedA-dump-source-r1' / name, digest)
        source_rows.append(dict(file=name, bytes=path.stat().st_size, sha256=digest))
    cap = json.loads(pin(work / 'capture-batch0-r1/result.json', CAPTURE_SHA).read_text())
    row = next(v for v in cap['records'] if v['id'] == 'pair0')
    require(row['input'] == r['input'] == prior['input'] and row['source_pair_indices'] == [0, 8] and row['split'] == 'calibration', 'Actual captured input differs')
    # Capture metadata lacks per-class counts; add no new numerical gate.
    input_row = dict(row['input']); x = np.load(work / 'capture-batch0-r1' / input_row['file'], allow_pickle=False)
    input_row.update(finite_count=x.size, nan_count=0, positive_inf_count=0, negative_inf_count=0)
    x, input_audit = array(work / 'capture-batch0-r1', input_row); require(input_audit['finite'], 'Captured input must be finite'); del x
    model = json.loads((work / 'step2-A-r2/result.json').read_text())['rknn']
    pin(work / 'step2-A-r2' / model['file'], MODEL_SHA, model['bytes'])
    require(r['model']['sha256'] == prior['models']['mixedA']['sha256'] == MODEL_SHA and r['original_parameters'] == prior['original_parameters'] and r['checkpoint'] == prior['checkpoint'], 'Same compiled graph/36-value provenance differs')
    original = next(c for c in prior['calls'] if c['route'] == 'mixedA' and c['init_flags'] == 0)
    final, final_rows = {}, []
    require([v['name'] for v in r['outputs']] == ['frame_output', 'global_output'], 'Complete final pair outputs required')
    for current, baseline, shape in zip(r['outputs'], original['outputs'], (FRAME, GLOBAL)):
        a, info = array(folder, current); b, _ = array(work / 'calibration-profile-r2', baseline)
        require(list(a.shape) == shape and bytes_equal(a, b) and current['byte_equal_r2_flags0'], 'Final same-model byte gate differs')
        info['byte_equal_r2_flags0'] = True; final[current['name']] = a; final_rows.append(info)
    require(r['final_baseline_byte_equal'] and r['original_output_guard_raised'] and r['numerical_guard_error'] == 'Nonfinite owned SDK output' and r['invalid_numeric'], 'Original finite guard/invalid payload evidence differs')
    selected = set(range(52, 60)) | {61, 76, 88, 89, 90, 91}
    retained, inventory, names = {}, [], set()
    for item in r['dump_files']:
        a, info = array(folder, item)
        require(item['file'] not in names and not item['unexpected_default_destination'], 'Duplicate/unexpected dump destination')
        names.add(item['file']); match = re.match(r'^(\d+)_([^_]+)_(.*?)_shape_', Path(item['file']).name)
        require(match is not None and int(match[1]) == item['filename_operator_index'] and match[2] == item['filename_op_type'], 'Emitted filename metadata differs')
        i = int(match[1]); info.update(emitted_operator_index=i, emitted_op_type=match[2], emitted_tensor_label=match[3])
        inventory.append(info)
        if i in selected:
            require(i not in retained, 'Selected dump index has multiple bindings'); retained[i] = a
        del a
    require(names == {str(p.relative_to(folder)) for p in (folder / 'dumps').rglob('*.npy')} and len(names) == r['valid_NPY_dump_count'] == 102, 'Complete NPY dump inventory differs')
    total = sum(v['bytes'] for v in inventory); require(total == r['dump_total_file_bytes'] == 758990304, 'Dump total bytes differ')
    invalid = sorted((v for v in inventory if not v['finite']), key=lambda v: (v['emitted_operator_index'], v['file']))
    require(invalid and invalid[0]['emitted_operator_index'] == 57 and all(v['finite'] for v in inventory if v['emitted_operator_index'] < 57), 'First saved nonfinite localization differs')
    f, g = final['frame_output'], final['global_output']
    pure = dict(dump052_transpose_equals053=bytes_equal(retained[52].transpose(2, 3, 0, 1), retained[53]),
        dump053_reshape_equals054=bytes_equal(retained[53].reshape(f.shape), retained[54]),
        dump055_equals_canonical_frame=bytes_equal(retained[55], f), dump056_equals_canonical_frame=bytes_equal(retained[56], f),
        dump054_reshape_equals057=bytes_equal(retained[54].reshape(retained[57].shape), retained[57]),
        canonical_frame_reshape_equals057=bytes_equal(f.reshape(retained[57].shape), retained[57]))
    require(all(pure[k] for k in list(pure)[:4]) and not pure['dump054_reshape_equals057'] and not pure['canonical_frame_reshape_equals057'], 'Finite predecessor/canonical relationships differ')
    transpose = representation_relation(retained[57], retained[58].transpose(0, 3, 2, 1))
    require(transpose['finite_value_unequal_count'] == transpose['non_both_NaN_bit_unequal_count'] == 0 and all(transpose[k] for k in ('NaN_mask_equal', 'positive_Inf_mask_equal', 'negative_Inf_mask_equal')), 'Transpose changes more than NaN payload representations')
    mask_rows = []
    first = retained[57].reshape(g.shape)
    for i in (76, 88, 89, 90, 91):
        value = retained[i]
        if value.shape == (1, 1024, 1, 2082):
            value = value.transpose(0, 3, 2, 1)
        value = value.reshape(g.shape)
        mask_rows.append(dict(emitted_operator_index=i, NaN_mask_equal_to_057=bool(np.array_equal(np.isnan(first), np.isnan(value))),
            negative_Inf_mask_equal_to_057=bool(np.array_equal(np.isneginf(first), np.isneginf(value))),
            positive_Inf_mask_equal_to_057=bool(np.array_equal(np.isposinf(first), np.isposinf(value))),
            raw_byte_equal_to_canonical_global=bytes_equal(value, g)))
    require(all(v['NaN_mask_equal_to_057'] and v['negative_Inf_mask_equal_to_057'] and v['positive_Inf_mask_equal_to_057'] for v in mask_rows), 'Residual/canonical invalid coordinate masks differ')
    state = r['resources']; expected = dict(contexts_created=1, contexts_destroyed=1, runs_completed=1, output_buffers_acquired=2, output_buffers_released=2)
    require({k: state[k] for k in expected} == expected and state['closed'] and state['failed'] and state['resource_counts_balanced'] and state['live_contexts'] == state['live_output_buffers'] == 0 and not state['cleanup_errors'], 'Invalid numeric owner cleanup ledger differs')
    require(r['resource_counts_verified'] and r['forward_calls'] == 1 and r['attribution_preconditions_met'], 'Worker completion ledger differs')
    require(control['child_returncode'] == 0 and not control['restore_errors'] and control['frequency_governor_limits_restored'] and control['baseline'] == control['restored'], 'Controller restoration differs')
    observations = control['observations']; temperatures = [v for o in observations for v in o['temperatures_c'].values()]
    require(observations and all(o['actual_frequency_hz'] == 1000000000 for o in observations) and max(temperatures) < control['temperature_limit_c'] == 75, 'Recorded controller bounds differ')
    report = dict(schema='v128_independent_saved_mixedA_untouched_dump_audit', status='saved_dump_verified_frame_to_global_boundary_localized_not_root_cause_or_quality_accepted',
        auditor_sha256=sha(__file__), result_sha256=RESULT_SHA, profile_r2_sha256=PROFILE_SHA, control_sha256=CONTROL_SHA,
        model_sha256=MODEL_SHA, source_files=source_rows, input_id='pair0', source_pair_indices=[0, 8], split='calibration', input=input_audit,
        final_outputs=final_rows, final_outputs_both_raw_byte_equal_r2=True, dump_array_count=102, dump_file_bytes=total,
        actual_dump_dtype_counts=dict(Counter(v['dtype'] for v in inventory)), all_dump_file_raw_shape_dtype_and_nonfinite_count_checks_passed=True,
        inventory=inventory, first_nonfinite_by_emitted_index=invalid[0], nonfinite_emitted_indices=[v['emitted_operator_index'] for v in invalid],
        finite_predecessor_and_canonical_relationships=pure, dump057_to058_transpose_relation=transpose,
        residual_and_canonical_nonfinite_coordinate_masks=mask_rows, resource_counts=expected, all_owned_outputs_released_context_closed=True,
        controller=dict(child_returncode=0, seconds=control['seconds'], observation_count=len(observations), maximum_recorded_temperature_c=max(temperatures),
            baseline=control['baseline'], restored=control['restored'], frequency_governor_limits_restored=True),
        numerical_acceptance=False, candidate_accepted=False, host_NN_ORt_SDK_GPU_or_board_calls=False,
        strongest_supported_localization='All saved snapshots preceding emitted index057 are finite. The finite frame producer and canonical SDK frame output are verified; the first saved invalid values occur at the derived frame-to-global input boundary before the named global norm/attention region. The same invalid coordinate masks recur at both global residual Adds and the final output.',
        limitations=[
            'File indices identify emitted dump order, not independently measured execution timestamps or causal instruction order.',
            'Every saved dump here is FP32. Names containing int8 and profile dtype labels do not establish the raw backing storage or conversion performed by the dumper.',
            '057 to058 preserves every non-NaN bit pattern and the NaN/Inf coordinate masks; only NaN payload bits differ. Raw-byte equality is not claimed for this transpose.',
            'The first invalid dump cannot be a pure mathematical reshape of either verified finite predecessor. This localizes the observation, but does not prove dtype bitcast, alias overwrite, uninitialized storage, or which SDK component is defective.',
            'Finite later update-path dumps do not demonstrate update accuracy. Under the saved-snapshot interpretation, later global attention/MLP arithmetic cannot alone explain the earlier invalid snapshot.',
            'Flags0 final raw parity supports a faithful untouched-model diagnostic; it does not validate every dump semantic or accept nonfinite outputs.',
            'Only calibration pair0 was examined. No heldout/full-model/trajectory accuracy or normal performance claim is made.'
        ])
    output = args.output or work / 'mixedA-dump-saved-audit.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=output.name, sha256=sha(output), dump_arrays=102, bytes=total, first_nonfinite_index=57, final_byte_parity=True)))


if __name__ == '__main__':
    main()
