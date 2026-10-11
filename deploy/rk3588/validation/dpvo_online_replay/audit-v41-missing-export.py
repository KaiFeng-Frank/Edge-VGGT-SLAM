#!/usr/bin/env python3
"""Saved-file NumPy audit; never imports Torch, ONNX, ORT, RKNN or a model."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
EXPORT_SHA = '2f5b3dd89b5863963bdfa52ffa213af02f08b6654a7012f5980ea3189d7ba47f'
PARENT_SHA = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
RUN_SHA = 'dd9f37524b999c1e586182a92c226481fcfc7f4da851d5e88d62d82be1029ecd'
SOURCE_HASHES = {
    'export-v41-m16-missing-source-board.py': '350294c45f2a78f77daa376b3e24cac7443a11d1cb1d83b6a51b6f3bde6669a9',
    'compile-v41-m16-online-update.py': '4ddab438d294172380876d82ae1e6fd7a4f52ed797f901248fec492d331f2ff1',
}
CONTRACT_SHA = '3533d791978085216c4a28cfae45d2c8ce9e2d62a8862efbf66db02d34be5e21'
OLD_COMPILER_SHA = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'
OLD_CONTRACT_SHA = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
M16_CONTRACT_SHA = '7be6bf750dabb8a3d88464fd1502bde3bf7690596a39791710cf41a9f8cb62fe'
NAMES = ('net', 'delta', 'weight')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def compare(actual, reference):
    if actual.dtype != np.float32 or actual.shape != reference.shape or reference.dtype != np.float32:
        raise ValueError('Same FP32 array shape required')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Finite saved neural arrays required')
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = a-b; failed = int(np.count_nonzero(np.abs(error) > 1e-5+1e-4*np.abs(b)))
    norm = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, gate_passed=failed == 0, failed_elements=failed,
                max_abs=float(np.max(np.abs(error), initial=0)),
                relative_L2=float(np.linalg.norm(error.ravel())/norm) if norm else None,
                bitwise_equal=actual.tobytes() == reference.tobytes(), rtol=1e-4, atol=1e-5)


def audit(args):
    checks = 0
    def check(value, message):
        nonlocal checks
        checks += 1
        if not value: raise ValueError(message)
    receipts = {}
    def read(path, expected=None):
        actual = sha(path)
        if expected is not None: check(actual == expected, 'Receipt SHA: '+Path(path).name)
        receipts[Path(path).parent.name+'/'+Path(path).name] = actual
        return json.loads(Path(path).read_text())
    def file(directory, row):
        check(Path(row['file']).name == row['file'], 'Artifact basename required')
        path = directory/row['file']
        check(not path.is_symlink() and path.stat().st_size == row['bytes']
              and sha(path) == row['sha256'], 'Artifact bytes/SHA: '+row['file'])
        return path
    def arrays(directory, row):
        with np.load(file(directory, row), allow_pickle=False) as bundle:
            check(set(bundle.files) == set(row['arrays']), 'NPZ keys differ')
            values = {name: np.array(bundle[name], copy=True, order='C') for name in bundle.files}
        for name, value in values.items():
            check(meta(value) == row['arrays'][name] and np.isfinite(value).all(), 'Raw metadata/finite: '+name)
        return values
    export = read(args.export_dir/'manifest.json', EXPORT_SHA)
    parent = read(args.parent_export_dir/'manifest.json', PARENT_SHA)
    run = read(args.run_dir/'run.json', RUN_SHA)
    for name, expected in SOURCE_HASHES.items(): check(sha(args.source_dir/name) == expected, 'Frozen source '+name)
    check(export['schema'] == 'v41_missing_M16_single_source_Update_CPU_ONNX_diagnostic'
          and export['status'] == 'completed_missing_M16_source_numerical_accepted_not_adopted'
          and export['script_sha256'] == SOURCE_HASHES['export-v41-m16-missing-source-board.py']
          and export['source_M16_manifest_sha256'] == PARENT_SHA and export['original_run_sha256'] == RUN_SHA,
          'Export identity differs')
    check(export['original_weights_before'] == export['original_weights_after'] == parent['original_weights_before']
          == run['original_weights_before'] == run['original_weights_after']
          and len(export['original_weights_before']) == 94, 'Original 94 state identities')
    parameters = export['original_Update_parameters']
    check(len(parameters) == 50 and parameters == parent['original_Update_parameters']
          and {'update.'+key: value for key, value in parameters.items()}
          == {key: value for key, value in run['original_weights_before'].items() if key.startswith('update.')},
          'All 50 original Update parameter identities')
    check(export['cleanup_errors'] == [] and export['original_parameters_unchanged'] is True
          and export['ORT_session_references_released'] is True and export['affinity_restored'] is True
          and export['sources_and_inputs_unchanged'] is True
          and export['source_pins'] == export['source_hashes_after'] and export['candidate_accepted'] is False,
          'Export source/lifecycle/research scope')
    keys = ('B1_D4_M16', 'B1_D6_M16')
    check(set(export['models']) == set(keys) and [v['model_key'] for v in export['cases']] == list(keys), 'Exactly two sources')
    parent_case = next(case for case in parent['cases'] if case['observation_id'] == 34)
    summaries = {}; comparison_count = 0
    for case in export['cases']:
        key = case['model_key']; model = export['models'][key]; d = model['degree']; source = case['source_binding']
        index, group = next((i, g) for i, g in enumerate(parent_case['groups']) if g['model_key'] == f'B2_D{d}_M16')
        check(model['batch'] == 1 and model['patches'] == 16 and d in (4, 6)
              and case['edges'] == 16*d and case['source_binding'] == model['source_binding']
              and source['parent_group_index'] == index and source['parent_model_key'] == group['model_key']
              and source['parent_manifest_sha256'] == PARENT_SHA
              and source['selected_source_slot'] == group['topology']['source_slots'][0]
              and source['selected_target_slots'] == group['topology']['target_slots'][0]
              and source['observation_id'] == source['original_call_index'] == 34, 'Predeclared complete first source')
        full = arrays(args.run_dir, case['original_full_inputs_artifact'])
        gold = arrays(args.run_dir, case['original_full_gold_artifact'])
        indices = arrays(args.export_dir, case['source_indices_artifact'])['canonical_edge_indices']
        check(indices.dtype == np.int64 and indices.shape == (1, d, 16) and meta(indices) == source['canonical_edge_indices']
              and (indices >= 0).all() and (indices < 704).all(), 'Complete source index metadata/domain')
        selected_source = source['selected_source_slot']; flat = indices.ravel()
        check(np.array_equal(full['ii'], full['kk']//16)
              and np.array_equal(np.sort(flat), np.flatnonzero(full['ii'] == selected_source))
              and np.array_equal(full['jj'][indices], np.broadcast_to(np.asarray(source['selected_target_slots'])[None,:,None], indices.shape))
              and np.array_equal(full['kk'][indices], np.broadcast_to((selected_source*16+np.arange(16))[None,None,:], indices.shape)),
              'Every original target/patch exactly once; no neighbor split or edge loss')
        feeds = arrays(args.export_dir, case['inputs_artifact'])
        saved_gold = arrays(args.export_dir, case['original_gold_artifact'])
        parent_feed = arrays(args.parent_export_dir, group['inputs_artifact'])
        for name in ('net', 'inp', 'corr'):
            check(np.array_equal(feeds[name], full[name][0,indices,:].transpose(0,3,1,2))
                  and np.array_equal(feeds[name], parent_feed[name][:1]), 'Complete-source feed '+name)
        for name in NAMES: check(np.array_equal(saved_gold[name], gold[name][0,indices,:].transpose(0,3,1,2)), 'Same-call original gold '+name)
        actual = arrays(args.export_dir, case['Torch_artifact']); ort = arrays(args.export_dir, case['ORT_artifact'])
        stats = {}
        for label, a, b in (('Torch_vs_original_group_gold', actual, saved_gold),
                            ('ORT_vs_original_group_gold', ort, saved_gold), ('ORT_vs_grouped_Torch', ort, actual)):
            stats[label] = {}
            for name in NAMES:
                result = compare(a[name], b[name]); recorded = case[label][name]; comparison_count += 1
                check(result['gate_passed'] and all(result[k] == recorded[k] for k in result if k != 'relative_L2')
                      and (result['relative_L2'] is recorded['relative_L2'] or np.isclose(result['relative_L2'], recorded['relative_L2'], rtol=1e-10, atol=1e-15)),
                      'Independent fixed element gate/receipt '+label+'/'+name)
                stats[label][name] = result
        check(all(np.all((v['weight'] >= 0)&(v['weight'] <= 1)) for v in (gold, saved_gold, actual, ort)), 'Unclipped weight domain')
        file(args.export_dir, model['model'])
        covered = set()
        for entry in model['parameter_coverage']['mapping']:
            for match in entry['original_matches']:
                expected = deepcopy(parameters[match['parameter']]); shape = expected['shape']
                if match['transform'] == 'conv1x1_reshape': expected['shape'] = shape+[1,1]
                elif match['transform'] == 'channel_affine_reshape': expected['shape'] = [1,shape[0],1,1]
                else: check(match['transform'] == 'identity', 'Unknown parameter transform')
                check(expected == entry['array'], 'Initializer metadata original byte binding')
                covered.add(match['parameter'])
        check(covered == set(parameters) and model['parameter_coverage']['all_original_parameters_covered'] is True,
              'All fifty parameter coverage metadata')
        compiled_dir = args.compile_root/key; compiled = read(compiled_dir/'compile.json')
        check(compiled['schema'] == 'v41_online_M16_Update_RKNN_compilation_diagnostic'
              and compiled['status'] == 'compiled_group_diagnostic_not_runtime_validated'
              and compiled['manifest_sha256'] == EXPORT_SHA and compiled['model_key'] == key
              and compiled['script_sha256'] == SOURCE_HASHES['compile-v41-m16-online-update.py']
              and compiled['contracts_sha256'] == CONTRACT_SHA and compiled['parent_compiler_sha256'] == OLD_COMPILER_SHA
              and compiled['parent_contracts_sha256'] == OLD_CONTRACT_SHA and compiled['m16_contracts_sha256'] == M16_CONTRACT_SHA
              and compiled['input'] == model['model'] and compiled['logical_inputs'] == model['inputs']
              and compiled['logical_outputs'] == model['outputs'], 'Compiler source/export/model/IO chain')
        check(compiled['configuration'] == dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False)
              and [r['name'] for r in compiled['operations']] == ['config','load_onnx','build','export_rknn']
              and all(r['code'] in (0,None) and np.isfinite(r['seconds']) and r['seconds'] >= 0 for r in compiled['operations'])
              and compiled['versions']['rknn-toolkit2'] == '2.3.2', 'Original nonquantized SDK operations/config')
        check(compiled['released'] is True and not compiled.get('release_error')
              and compiled['sources_and_inputs_unchanged'] is True and compiled['source_pins'] == compiled['source_hashes_after']
              and compiled['export_strict_FP32_gate_passed'] is True and compiled['candidate_accepted'] is False
              and compiled['model_inference_executed'] is False and compiled['NPU_execution_or_placement_validated'] is False,
              'Compile cleanup/provenance/no-runtime-claim')
        file(compiled_dir, compiled['output'])
        summaries[key] = dict(source=source,comparison=stats,compiled_model=compiled['output'],compile_operations=compiled['operations'])
    check(comparison_count == export['strict_comparison_count'] == 18 and export['strict_FP32_gate_passed'] is True
          and export['strict_FP32_thresholds'] == dict(rtol=1e-4,atol=1e-5)
          and export['strict_FP32_thresholds_relaxed'] is False and export['output_clipping'] is False,
          'All eighteen fixed gates; no threshold relaxation')
    return dict(status='passed_saved_missing_source_identity_not_adopted',checks=dict(count=checks,failures=0),
        auditor_sha256=sha(__file__),receipt_sha256=receipts,numerical_comparisons=comparison_count,
        sources=summaries,export_driver_seconds=export['actual_driver_seconds'],candidate_accepted=False,
        limits=['NumPy saved-array recomputation only, no neural/ORT/SDK forward.',
                'Initializer and original state byte bindings are checked as frozen metadata; graphs are file-hashed, not executed or reparsed.',
                'Board source before/after and compiler lifecycle are receipt declarations; original board absolute files are not rehashed here.',
                'Successful nonquantized compilation does not establish SDK numerical quality, operator placement, online VO or 20Hz acceptance.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export-dir','parent-export-dir','run-dir','compile-root','source-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): raise ValueError('New output required')
    try: report = audit(args)
    except Exception as error: report = dict(status='failed_saved_audit',error=repr(error),auditor_sha256=sha(__file__))
    with args.output.open('x') as stream: stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','checks','error')}))
    return int(report['status'] == 'failed_saved_audit')


if __name__ == '__main__':raise SystemExit(main())
