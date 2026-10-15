#!/usr/bin/env python3
"""Independent saved NumPy/JSON audit; no SDK, network, or model computation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_SHA = 'ed500643ef333dd824ccab81a126e87ff8616c3cb3ee09e3ec59f2a4b7bbf7e9'
PAIRS_SHA = '18288334f5309b59c6eacdf8a95c8141fae6a453cf05305d8d7d8f88b13d81f2'
LINEAGE_SHA = 'f46edc183442c69af3dd314c9a1e096dca42f0590881d51c6fd8c7390df9617e'
CONFIG_SHA = '7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
RECEIPTS = ['cc2dd7fbddef8ee12de00b7a5cc88a16ba9d531783ab09ae8e66abe50df43408',
            '34ee3b510014f55c34525c6a96a3527603ba6350c896fae7498ae5493987df69',
            '5560b34755bfbca54cc179c119991792b63d6135e839ad15ac8db95e16965660']
COUNTERS = ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')
EXPECTED = dict(zip(COUNTERS, (4, 4, 16, 16, 16)))
FRAME, DINO = [2, 1041, 1024], [1, 1041, 1024]


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
    require(sha(path) == digest and (size is None or path.stat().st_size == size), 'File hash/bytes differ: ' + str(path))
    return path


def raw(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def tensor_metadata(row, shape):
    require(row['shape'] == shape and row['dtype'] == 'float32' and row['tensor_bytes'] == int(np.prod(shape)) * 4 and
            row['finite'] and len(row['raw_sha256']) == 64, 'Captured tensor metadata differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--png-dir', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); root = a.root.resolve(); output = a.output or root / 'saved-frame0-inputs-audit.json'
    pin(root / 'capture_frame0_inputs_board.py', SOURCE_SHA)
    pairs = json.loads(pin(root / 'real-image-pair-pins.json', PAIRS_SHA).read_text())
    lineage = json.loads(pin(root / 'dino-board-preflight.json', LINEAGE_SHA).read_text())
    config = json.loads(pin(root / 'original-full-config.json', CONFIG_SHA).read_text())
    require([r['indices'] for r in pairs['pairs']] == [[0, 8], [16, 24], [48, 56], [64, 72], [96, 104], [112, 120]] and
            [r['split'] for r in pairs['pairs']] == ['calibration'] * 4 + ['heldout'] * 2, 'Declared six-pair split differs')
    pngs = [v for row in pairs['pairs'] for v in row['images']]
    require(len(pngs) == 12 and all(len({v[k] for v in pngs}) == 12 for k in ('index', 'file', 'sha256')), 'PNG identities overlap')
    calibration_pngs = {v['sha256'] for r in pairs['pairs'][:4] for v in r['images']}
    heldout_pngs = {v['sha256'] for r in pairs['pairs'][4:] for v in r['images']}
    require(not calibration_pngs & heldout_pngs, 'Calibration/heldout PNG split overlaps')
    if a.png_dir is not None:
        for row in pngs:
            pin(a.png_dir / Path(row['file']).name, row['sha256'], row['bytes'])
    require(lineage['config_sha256'] == CONFIG_SHA and lineage['all_four_DINO_models_present_and_SHA_match'] and
            [row['group'] for row in lineage['rows']] == list(range(4)) and
            [row['compile'] for row in lineage['rows']] == config['dino'], 'Four original ordered group identities differ')
    for index, row in enumerate(lineage['rows']):
        c = json.loads(pin(root / 'dino-original-lineage' / ('group%d-compile.json' % index), row['compile']['sha256']).read_text())
        e = json.loads(pin(root / 'dino-original-lineage' / ('group%d-export.json' % index), row['export']['sha256']).read_text())
        require(c['rknn']['sha256'] == row['model']['sha256'] and c['rknn']['bytes'] == row['model']['bytes'] and
                c['onnx']['sha256'] == e['graph']['sha256'] and e['indices'] == list(range(index * 6, index * 6 + 6)) and
                e['proof']['original_parameter_count'] == 84 and e['proof']['namespace_inverse_protobuf_checks_passed'],
                'Original six-layer parameter/compiled graph lineage differs')
    batches, samples, role_parameters, source_hashes = [], [], None, None
    for index, expected_sha in enumerate(RECEIPTS):
        directory = root / ('capture-batch%d-r1' % index)
        r = json.loads(pin(directory / 'result.json', expected_sha).read_text())
        require(r['schema'] == 'v128_original_native518_frame0_input_capture' and
                r['status'] == 'completed_original_native518_frame0_input_subbatch_capture' and
                r['script_sha256'] == SOURCE_SHA and r['batch_index'] == index and r['stage'] == 'finished', 'Complete pinned subbatch required')
        require(r['NPU_core_mask'] == 7 and r['NPU_init_flags'] == 0 and r['maximum_live_NPU_contexts'] == 1 and
                r['original_DINO_layers_per_image'] == 24 and r['original_DINO_parameter_count'] == 336 and
                r['original_aggregator_blocks_executed'] == r['CPU_DINO_block_forwards_executed'] == 0 and
                r['camera_or_DPT_executed'] is False and r['reuse_previous_tail'] is False, 'Capture mathematical/execution scope differs')
        require(r['expected_SDK_totals'] == r['SDK_totals'] == EXPECTED and r['cleanup_errors'] == [] and
                all(r[k] for k in ('all16_SDK_calls_completed', 'all_NPU_contexts_closed', 'model_state_unchanged',
                                  'all_inputs_before_frame0', 'two_pair_subbatch_complete', 'checkpoint_verified_before_deserialization',
                                  'checkpoint_final_SHA_verified', 'full_six_pair_PNG_partitions_disjoint',
                                  'affinity_restored', 'environment_restored', 'sys_path_restored', 'source_and_inputs_unchanged')),
                'Original state, SDK balance, identity or restoration differs')
        require(r['checkpoint_final_sha256'] == r['checkpoint']['sha256'] ==
                'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0' and
                r['checkpoint']['bytes'] == 5026874952 and r['pinned_pair_manifest_sha256'] == PAIRS_SHA and
                r['pinned_dino_lineage_sha256'] == LINEAGE_SHA and r['upstream_models'] == lineage['rows'], 'Checkpoint/source/model pins differ')
        if role_parameters is None:
            role_parameters, source_hashes = r['original_role_token_parameters'], r['source_hashes']
        require(r['original_role_token_parameters'] == role_parameters and r['source_hashes'] == source_hashes,
                'Learned role values/source authority differ between batches')
        tensor_metadata(role_parameters['camera_token'], [1, 2, 1, 1024])
        tensor_metadata(role_parameters['register_token'], [1, 2, 4, 1024])
        require(all(r['pinned_files'][v['file']] == v['sha256'] for v in pngs), 'Worker did not pin all12 real PNGs')
        selected = pairs['pairs'][index * 2:index * 2 + 2]
        require([(v['id'], v['source_pair_indices'], v['split'], v['source_images']) for v in r['records']] ==
                [(v['id'], v['indices'], v['split'], v['images']) for v in selected], 'Fresh pair identities/order differ')
        expected_calls = [(g, pair['id'], f) for g in range(4) for pair in selected for f in range(2)]
        require([(v['group'], v['pair_id'], v['frame']) for v in r['SDK_calls']] == expected_calls and len(r['owner_metrics']) == 4,
                'Actual group-major SDK call order differs')
        sums = {key: sum(v[key] for v in r['owner_metrics']) for key in COUNTERS}
        require(sums == EXPECTED, 'Per-owner SDK counters differ from total')
        for group, owner in enumerate(r['owner_metrics']):
            require(owner['model_sha256'] == lineage['rows'][group]['model']['sha256'] and
                    {key: owner[key] for key in COUNTERS} == dict(zip(COUNTERS, (1, 1, 4, 4, 4))) and
                    owner['closed'] and not owner['failed'] and owner['live_contexts'] == owner['live_output_buffers'] == 0 and
                    owner['resource_counts_balanced'] and owner['cleanup_errors'] == [] and owner['core_mask'] == 7 and owner['init_flags'] == 0,
                    'Original model owner did not close cleanly')
        file_names, byte_count = [], 0
        for record in r['records']:
            require(record['original_final_norm_and_role_concat_executed'] and record['input_is_before_original_frame0'] and
                    record['actual_four_DINO_groups_executed'], 'Before-frame0 boundary not reported')
            tensor_metadata(record['preprocessed_pixels'], [2, 3, 392, 518])
            require(len(record['prepared_DINO_inputs']) == 2, 'Both images required')
            for prepared in record['prepared_DINO_inputs']: tensor_metadata(prepared, DINO)
            require([v['group'] for v in record['DINO_group_outputs']] == list(range(4)), 'Complete24 ordered DINO layers required')
            for group in record['DINO_group_outputs']:
                require(group['indices'] == list(range(group['group'] * 6, group['group'] * 6 + 6)) and len(group['frames']) == 2,
                        'Original six-layer range differs')
                for frame, value in enumerate(group['frames']):
                    tensor_metadata(value, DINO)
                    call = next(v for v in r['SDK_calls'] if (v['group'], v['pair_id'], v['frame']) == (group['group'], record['id'], frame))
                    require(call['output'] == value, 'SDK output metadata differs from per-record graph chain')
            row = record['input']; tensor_metadata(row, FRAME)
            require(Path(row['file']).name == row['file'], 'Input filename must be relative basename')
            path = pin(directory / row['file'], row['sha256'], row['bytes']); x = np.load(path, allow_pickle=False)
            require(x.dtype == np.float32 and list(x.shape) == FRAME and x.flags.c_contiguous and np.isfinite(x).all() and
                    x.nbytes == row['tensor_bytes'] and raw(x) == row['raw_sha256'], 'Actual saved frame0 bytes/dtype/finite differ')
            camera, register = np.ascontiguousarray(x[:, 0:1]), np.ascontiguousarray(x[:, 1:5])
            require(raw(camera) == role_parameters['camera_token']['raw_sha256'] and
                    raw(register) == role_parameters['register_token']['raw_sha256'], 'Original pair-role token concatenation byte identity failed')
            samples.append(dict(id=record['id'], split=record['split'], source_pair_indices=record['source_pair_indices'],
                input=row, finite_independently_checked=True, min=float(x.min()), max=float(x.max()),
                camera_role_raw_sha256=raw(camera), register_role_raw_sha256=raw(register),
                camera_and_register_match_original_receipt_bytes=True))
            file_names.append(row['file']); byte_count += row['bytes']
        require(set(file_names) == {p.name for p in directory.glob('*.npy')} and not list(directory.glob('*.tmp')) and
                len(file_names) == r['saved_array_count'] == 2 and byte_count == r['saved_array_bytes'] and
                r['original_final_norm_calls'] == 4, 'Saved payload inventory/final norm count differs')
        control_path = root / ('capture-control-batch%d-r1' % index) / 'control.json'
        control = json.loads(control_path.read_text())
        require(control['child_returncode'] == 0 and control['restore_errors'] == [] and
                control['frequency_governor_limits_restored'] and control['baseline'] == control['restored'] and
                control['requested_frequency_hz'] == 1000000000, 'Controller clock/exit restoration differs')
        maximum = max(max(v['temperatures_c'].values()) for v in control['observations'])
        require(maximum < 75 and all(v['actual_frequency_hz'] == 1000000000 for v in control['observations']), 'Controller actual thermal/frequency samples differ')
        batches.append(dict(batch=index, receipt_sha256=expected_sha, SDK_totals=sums,
            control_sha256=sha(control_path), control_seconds=control['seconds'], max_temperature_c=maximum,
            baseline_clock=control['baseline'], restored_clock=control['restored'], worker_cleanup_and_clock_restoration_consistent=True))
    require(len(samples) == 6 and len({v['input']['raw_sha256'] for v in samples}) == 6 and
            [v['split'] for v in samples] == ['calibration'] * 4 + ['heldout'] * 2, 'Six distinct actual input payloads required')
    report = dict(schema='v128_independent_saved_frame0_input_audit', status='saved_data_verified_not_model_quality_acceptance',
        auditor_source_sha256=sha(__file__), capture_source_sha256=SOURCE_SHA, pair_manifest_sha256=PAIRS_SHA,
        DINO_lineage_manifest_sha256=LINEAGE_SHA, batches=batches, samples=samples,
        total_saved_NPY_count=6, total_saved_NPY_bytes=sum(v['input']['bytes'] for v in samples),
        SDK_totals={key: sum(v['SDK_totals'][key] for v in batches) for key in COUNTERS},
        six_distinct_frame0_raw_hashes=True, calibration_pair_count=4, heldout_pair_count=2,
        twelve_distinct_PNG_ids_and_SHA_pins=True, calibration_heldout_PNG_sets_disjoint=True,
        original_pair_role_token_bytes_independently_verified=True,
        PNG_files_independently_rehashed=a.png_dir is not None,
        chain_evidence_scope='Per-record four groups0..23 exactly match48 ordered SDK output metadata rows and pinned original compiled lineage. Intermediate payloads are not saved; continuity follows pinned reviewed capture source, not host recomputation.',
        limitations=['Actual frame0 inputs are independently hash/shape/finite/role-slice checked. No DINO, normalization, patch projection or model-forward recomputation.',
                     'Without --png-dir, PNG identity is checked against manifests/worker hash ledger, not independent host image-file hashing.',
                     'Original NPU FP16 prefix data for one sequence; no complete original FP32 model accuracy, trajectory, realtime or statistical-independence claim.'])
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temp.replace(output)
    print(json.dumps(dict(file=str(output), sha256=sha(output), verified_inputs=6, SDK_totals=report['SDK_totals'])))


if __name__ == '__main__':
    main()
