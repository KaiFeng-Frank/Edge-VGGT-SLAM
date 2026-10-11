#!/usr/bin/env python3
"""NumPy saved-file audit of the EOF D3 export; no neural execution or SDK imports."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
EXPORT_SHA = '6837252e28ee109fc67680ed197825e403ec727ce3e7ad391a3090ea85ff234f'
EXPORTER_SHA = 'ae036a3fdfb0f9dc1c5cf0dd2dae37bd59ab313974904c20611936d42f9eca52'
PARENT_SHA = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
RUN_SHA = '78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b'
FAILURE_SHA = 'f7fecc2da09ee74da72da4d30af99bb874149a5e6f6961a199a28c29f91cb072'
INPUT_SHA = 'b0954b8dd84118d9749e7baeca15df83c59985059a0407cf1970521e8acb18ca'
SNAPSHOT_SHA = 'd879eece461d1c20cd52388bcde97ba29f196b468b9398aac49eec6a9a85313e'
MODEL_KEY = 'B1_D3_M16'
NAMES = ('net', 'delta', 'weight')
TEACHER_KIND = 'new_original_Torch_on_saved_NPU_feedback_input'
OLD_KEYS = tuple(f'B{b}_D{d}_M16' for b in (1, 2) for d in range(4, 8))
OLD_INDEX_SHA = '8c3f1c3b92e14a6742a984f750a5eb4c1bbf6397ca31440b96b0768333dc21fa'
NEW_INDEX_SHA = 'eeb9241c9eab52039aa64fcf6546ce4624d1388756a9e919f42c22de4083a85c'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def compare(actual, reference):
    if actual.dtype != np.float32 or reference.dtype != np.float32 or actual.shape != reference.shape:
        raise ValueError('Same FP32 array shape required')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Finite saved arrays required')
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = a-b
    failed = int(np.count_nonzero(np.abs(error) > 1e-5+1e-4*np.abs(b)))
    norm = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, gate_passed=failed == 0, failed_elements=failed,
                max_abs=float(np.max(np.abs(error), initial=0)),
                relative_L2=float(np.linalg.norm(error.ravel())/norm) if norm else None,
                bitwise_equal=actual.tobytes() == reference.tobytes(), rtol=1e-4, atol=1e-5)


def derive_source_indices(inputs, source=11, targets=(9, 10, 11), patches=16):
    """Independent target-major/patch-major selection, retaining every source edge."""
    ii, jj, kk = (inputs[name] for name in ('ii', 'jj', 'kk'))
    if any(v.dtype != np.int64 or v.ndim != 1 for v in (ii, jj, kk)) or not (ii.shape == jj.shape == kk.shape):
        raise ValueError('Same int64 topology vectors required')
    if not np.array_equal(ii, kk//patches): raise ValueError('Original source/patch mapping differs')
    chosen = np.flatnonzero(ii == source)
    if chosen.size != len(targets)*patches: raise ValueError('Complete source degree required')
    result = []
    for target in targets:
        for patch in range(patches):
            edge = np.flatnonzero((ii == source)&(jj == target)&(kk == source*patches+patch))
            if edge.size != 1: raise ValueError('Each original target/patch must occur once')
            result.append(int(edge[0]))
    value = np.asarray(result, dtype=np.int64).reshape(1, len(targets), patches)
    if not np.array_equal(np.sort(value.ravel()), chosen): raise ValueError('Source edges lost or added')
    return value


def validate_index_retention(old, new):
    if (set(old['models']) != set(OLD_KEYS) or set(new['models']) != set(OLD_KEYS+(MODEL_KEY,))
            or any(old['models'][k] != new['models'][k] for k in OLD_KEYS)):
        raise ValueError('The eight complete historical model entries must remain exactly identical')
    if (old['candidate_accepted'] is not False or new['candidate_accepted'] is not False
            or new['old_online_compile_index_sha256'] != OLD_INDEX_SHA
            or new['parent_manifest_sha256'] != old['parent_manifest_sha256']
            or new['additions_manifest_sha256'] != old['additions_manifest_sha256']
            or new['eof_manifest_sha256'] != EXPORT_SHA):
        raise ValueError('Index ancestry/research scope differs')


def audit(args):
    count = 0
    file_hashes = {}
    def check(value, reason):
        nonlocal count
        count += 1
        if not value: raise ValueError(reason)
    def bind(path, expected, label):
        path = Path(path)
        check(path.is_file() and not path.is_symlink() and sha(path) == expected, 'File SHA: '+label)
        file_hashes[label] = expected
        return path
    def read(path, expected, label):
        return json.loads(bind(path, expected, label).read_text())
    def artifact(directory, row, label):
        check(Path(row['file']).name == row['file'], 'Artifact basename: '+label)
        path = bind(directory/row['file'], row['sha256'], label)
        check(path.stat().st_size == row['bytes'], 'Artifact size: '+label)
        return path
    def arrays(directory, row, label):
        with np.load(artifact(directory, row, label), allow_pickle=False) as bundle:
            check(set(bundle.files) == set(row['arrays']), 'NPZ keys: '+label)
            values = {name: np.array(bundle[name], copy=True, order='C') for name in bundle.files}
        for name, value in values.items():
            check(meta(value) == row['arrays'][name] and np.isfinite(value).all(), 'Raw metadata/finite: '+label+'/'+name)
        return values
    export = read(args.export_dir/'manifest.json', EXPORT_SHA, 'export/manifest.json')
    parent = read(args.parent_export_dir/'manifest.json', PARENT_SHA, 'parent/manifest.json')
    run = read(args.run_dir/'run.json', RUN_SHA, 'failed_run/run.json')
    failure = read(args.run_dir/'online-sdk/update-000203-failed.json', FAILURE_SHA, 'failed_run/failure.json')
    bind(args.exporter, EXPORTER_SHA, 'exporter.py')
    check(export['schema'] == 'v42_EOF_M16_D3_single_source_Update_CPU_ONNX_diagnostic'
          and export['status'] in ('completed_EOF_M16_D3_numerical_accepted_not_adopted', 'completed_EOF_M16_D3_numerical_rejected')
          and export['script_sha256'] == EXPORTER_SHA and export['source_M16_manifest_sha256'] == PARENT_SHA
          and export['source_run_sha256'] == export['original_run_sha256'] == RUN_SHA
          and export['source_failure_sha256'] == FAILURE_SHA and export['source_failed_input_sha256'] == INPUT_SHA
          and export['source_before_termination_sha256'] == SNAPSHOT_SHA, 'Export/source receipt identity')
    check(run['status'] == 'failed' and run['frames_completed'] == 192 and run['active_stage'] == 'termination_offline'
          and run['online_metrics']['calls'][-1] == failure and failure['index'] == 203
          and failure['route'] == 'full' and failure['edges'] == 640 and 'B1_D3_M16' in failure['error']
          and failure['status'] == 'failed_no_partial_fullgraph' and 'SDK_execution' not in failure
          and failure['fullgraph_returned'] is False and failure['outputs_committed_by_callback'] is False,
          'Historical EOF failure retained without partial SDK/VO output')
    check(export['original_full_Update_reexecuted'] is True and export['original_full_Update_forward_calls'] == 1
          and export['teacher_reference_kind'] == TEACHER_KIND and export['original_CPU_trajectory_gold_used'] is False
          and export['historical_SDK_strict_FP32_rejected'] is True and export['diagnostic_allow_fp32_rejected'] is True,
          'New same-input original Torch teacher; not CPU trajectory gold')
    weights = export['original_weights_before']
    check(len(weights) == 94 and weights == export['original_weights_after'] == parent['original_weights_before']
          == parent['original_weights_after'] == run['original_weights_before'] == run['original_weights_after'],
          'Original 94 state metadata unchanged across ancestors')
    parameters = export['original_Update_parameters']
    check(len(parameters) == 50 and parameters == parent['original_Update_parameters']
          and {'update.'+k: v for k, v in parameters.items()} == {k: v for k, v in weights.items() if k.startswith('update.')},
          'All 50 original Update state metadata bind original 94')
    check(export['source_pins'] == export['source_hashes_after'] and run['source_pins'] == run['source_hashes_after']
          and all(export['source_pins'].get(p) == h for p, h in run['source_pins'].items())
          and export['cleanup_errors'] == [] and run['cleanup_errors'] == []
          and export['sources_and_inputs_unchanged'] is True and export['original_parameters_unchanged'] is True
          and export['ORT_session_references_released'] is True and export['affinity_restored'] is True
          and run['affinity_restored'] is True and export['candidate_accepted'] is False
          and export['NPU_used'] is False and export['BA_executed'] is False and export['VO_state_committed'] is False
          and export['realtime_validated'] is False, 'Sources/lifecycle/research scope')
    check(set(export['models']) == {MODEL_KEY} and len(export['cases']) == 1, 'Only one new actual B1D3 graph')
    case, model = export['cases'][0], export['models'][MODEL_KEY]
    check(case['model_key'] == MODEL_KEY and case['observation_id'] == 191 and case['original_call_index'] == 203
          and case['edges'] == 48 and model['batch'] == 1 and model['degree'] == 3 and model['patches'] == 16
          and case['source_binding'] == model['source_binding'], 'Case/source shape identity')
    check(case['original_full_inputs_artifact'] == export['source_failure_artifact'] == failure['failure_artifact']
          and failure['failure_artifact']['sha256'] == INPUT_SHA
          and export['source_before_termination_artifact'] == run['snapshot_before_termination']['artifact']
          and export['source_before_termination_artifact']['sha256'] == SNAPSHOT_SHA
          and export['source_failure_snapshot_identity'] == failure['snapshot_identity']
          and export['full_source_degree_plan'] == failure['degree_plan']
          and case['full_original_teacher_artifact'] == export['full_teacher_artifact'], 'Full teacher owning input/snapshot bindings')
    full = arrays(args.run_dir/'online-sdk', case['original_full_inputs_artifact'], 'failed_run/full_inputs.npz')
    snapshot = arrays(args.run_dir, export['source_before_termination_artifact'], 'failed_run/before_termination.npz')
    teacher = arrays(args.export_dir, export['full_teacher_artifact'], 'export/full_teacher.npz')
    check(set(full) == {'net','inp','corr','ii','jj','kk'} and set(teacher) == set(NAMES), 'Full640 input/teacher keys')
    for name, channels in zip(('net','inp','corr'), (384,384,882)):
        check(full[name].shape == (1,640,channels) and full[name].dtype == np.float32, 'Full640 input shape: '+name)
    for name, channels in zip(NAMES, (384,2,2)):
        check(teacher[name].shape == (1,640,channels) and teacher[name].dtype == np.float32, 'Full640 teacher shape: '+name)
    for name in ('net','ii','jj','kk'):
        check(np.array_equal(full[name], snapshot[name]), 'Failed input equals pre-EOF snapshot: '+name)
    check(snapshot['poses'].shape == (1,12,7) and snapshot['patches'].shape == (1,192,3,3,3), 'Original retained snapshot population')
    bind(args.run_dir/'input-manifest.json', run['input_manifest_sha256'], 'failed_run/input-manifest.json')
    check(export['input_manifest_sha256'] == run['input_manifest_sha256'] and export['policy_sha256'] == run['policy_sha256'],
          'Original source/RNG workload identities')
    expected_indices = derive_source_indices(full)
    indices = arrays(args.export_dir, case['source_indices_artifact'], 'export/source_indices.npz')['canonical_edge_indices']
    source = case['source_binding']
    check(np.array_equal(indices, expected_indices) and meta(indices) == source['canonical_edge_indices']
          and source['parent_run_sha256'] == RUN_SHA and source['parent_failure_sha256'] == FAILURE_SHA
          and source['before_termination_sha256'] == SNAPSHOT_SHA and source['failed_update_index'] == 203
          and source['selected_source_slot'] == 11 and source['selected_target_slots'] == [9,10,11]
          and source['teacher_reference_kind'] == TEACHER_KIND and source['original_CPU_trajectory_gold_used'] is False,
          'Independent complete target/patch/source selection')
    check(case['topology'] == dict(source_slots=[11],target_slots=[[9,10,11]],sources=1,degree=3,patches=16,edges=48,
          padding=False,dependency_groups_split=False,physical_layout='source_batch,channels,target,patch'), 'No neighbor split/padding')
    feeds = arrays(args.export_dir, case['inputs_artifact'], 'export/source_inputs.npz')
    gold = arrays(args.export_dir, case['original_gold_artifact'], 'export/source_teacher.npz')
    actual = arrays(args.export_dir, case['Torch_artifact'], 'export/Torch.npz')
    ort = arrays(args.export_dir, case['ORT_artifact'], 'export/ORT.npz')
    check(set(feeds) == {'net','inp','corr'} and set(gold) == set(actual) == set(ort) == set(NAMES), 'Source feed/output keys')
    for name in ('net','inp','corr'):
        expected = full[name][0,indices,:].transpose(0,3,1,2)
        check(np.array_equal(feeds[name], expected), 'CF source input is complete original gather: '+name)
    for name in NAMES:
        expected = teacher[name][0,indices,:].transpose(0,3,1,2)
        check(np.array_equal(gold[name], expected), 'Source teacher is exact same-input full640 gather: '+name)
    numerical = {}; gates = []
    for label, a, b in (('Torch_vs_original_group_gold',actual,gold),
                        ('ORT_vs_original_group_gold',ort,gold), ('ORT_vs_grouped_Torch',ort,actual)):
        numerical[label] = {}
        for name in NAMES:
            result = compare(a[name], b[name]); recorded = case[label][name]
            check(all(result[k] == recorded[k] for k in result if k != 'relative_L2')
                  and (result['relative_L2'] is recorded['relative_L2'] or np.isclose(result['relative_L2'],recorded['relative_L2'],rtol=1e-10,atol=1e-15)),
                  'Saved fixed numerical comparison matches: '+label+'/'+name)
            numerical[label][name] = result; gates.append(result['gate_passed'])
    check(all(((v['weight'] >= 0)&(v['weight'] <= 1)).all() for v in (teacher,gold,actual,ort))
          and case['Torch_domain'] == case['ORT_domain'] == dict(finite=True,weight_domain_valid=True), 'Unclipped teacher/Torch/ORT weight domain')
    check(len(gates) == export['strict_comparison_count'] == 9 and all(gates) == export['strict_FP32_gate_passed']
          and export['status'] == ('completed_EOF_M16_D3_numerical_accepted_not_adopted' if all(gates) else 'completed_EOF_M16_D3_numerical_rejected')
          and export['strict_FP32_thresholds'] == dict(rtol=1e-4,atol=1e-5)
          and export['strict_FP32_thresholds_relaxed'] is False and export['output_clipping'] is False, 'Nine original fixed gates/status')
    artifact(args.export_dir, model['model'], 'export/B1_D3_M16.onnx')
    check(model['inputs'] == [dict(name=n,dtype='float32',shape=[1,c,3,16]) for n,c in zip(('net','inp','corr'),(384,384,882))]
          and model['outputs'] == [dict(name=n,dtype='float32',shape=[1,c,3,16]) for n,c in zip(('net_out','delta','weight'),(384,2,2))]
          and model['padding'] is False and model['external_data_used'] is False and model['numerical_validation_completed'] is True,
          'Only actual fixed CF model contract')
    covered = set()
    for entry in model['parameter_coverage']['mapping']:
        for match in entry['original_matches']:
            expected = deepcopy(parameters[match['parameter']]); shape = expected['shape']
            if match['transform'] == 'conv1x1_reshape': expected['shape'] = shape+[1,1]
            elif match['transform'] == 'channel_affine_reshape': expected['shape'] = [1,shape[0],1,1]
            else: check(match['transform'] == 'identity', 'Unknown original parameter transform')
            check(expected == entry['array'], 'Original initializer raw metadata binding')
            covered.add(match['parameter'])
    check(covered == set(parameters) and model['parameter_coverage']['all_original_parameters_covered'] is True,
          'Original fifty parameter coverage declarations')
    old_models = {}
    roots = {p.name: p for p in args.old_model_root}
    for key in OLD_KEYS:
        rows = [(Path(p), h) for p,h in run['source_pins'].items() if Path(p).parent.name == key and Path(p).name == 'block.rknn']
        check(len(rows) == 1, 'Exactly one historical compiled model: '+key)
        board_model, digest = rows[0]
        check(board_model.parent.parent.name in roots, 'Explicit old model root required: '+board_model.parent.parent.name)
        local = roots[board_model.parent.parent.name]/key
        old_compile_sha = run['source_pins'][str(board_model.parent/'compile.json')]
        old_compile = read(local/'compile.json', old_compile_sha, 'old8/'+key+'/compile.json')
        check(old_compile['model_key'] == key and old_compile['output']['sha256'] == digest
              and old_compile['output']['file'] == 'block.rknn', 'Historical model/compile byte identity: '+key)
        artifact(local, old_compile['output'], 'old8/'+key+'/block.rknn')
        old_models[key] = dict(model_sha256=digest,model_bytes=old_compile['output']['bytes'],compile_sha256=old_compile_sha)
    for key in OLD_KEYS:
        row = next(Path(p) for p in run['source_pins'] if Path(p).parent.name == key and Path(p).name == 'block.rknn')
        local = roots[row.parent.parent.name]/key
        check(sha(local/'block.rknn') == old_models[key]['model_sha256'] and sha(local/'compile.json') == old_models[key]['compile_sha256'],
              'Old model/compile remained unchanged during audit: '+key)
    index_retention = dict(performed=False)
    check(bool(args.old_index) == bool(args.new_index), 'Old/new index options must be supplied together')
    if args.new_index:
        old = read(args.old_index, OLD_INDEX_SHA, 'old8/index.json')
        new = read(args.new_index, NEW_INDEX_SHA, 'new9/index.json')
        validate_index_retention(old, new)
        check(any(Path(p).name == 'index.json' and h == OLD_INDEX_SHA for p,h in run['source_pins'].items()), 'Old index bound by original run')
        for key in OLD_KEYS:
            row = old['models'][key]
            check(row['model']['sha256'] == old_models[key]['model_sha256']
                  and row['model']['bytes'] == old_models[key]['model_bytes']
                  and row['compile_sha256'] == old_models[key]['compile_sha256'], 'Index matches independently hashed old bytes: '+key)
        index_retention = dict(performed=True,old_index_sha256=OLD_INDEX_SHA,new_index_sha256=NEW_INDEX_SHA,
                               eight_complete_model_entries_exact=True,only_new_key=MODEL_KEY)
    check(export['ordinary_Torch_forward_calls'] == export['ORT_calls'] == 1 and export['original_full_Update_forward_calls'] == 1,
          'Separate full teacher/group/ORT execution counts')
    return dict(schema='v42_saved_EOF_D3_export_independent_review',status='passed_saved_EOF_D3_identity_not_adopted',
        auditor_sha256=sha(__file__),checks=dict(count=count,failures=0),file_sha256=file_hashes,
        selected_source=dict(source_slot=11,targets=[9,10,11],patches=16,edges=48,full_input_edges=640,indices=meta(indices)),
        teacher_reference_kind=TEACHER_KIND,original_CPU_trajectory_gold_used=False,
        original_state_count=94,original_Update_parameter_count=50,numerical_comparisons=9,
        strict_FP32_gate_passed=all(gates),numerical=numerical,old_eight_compiled_bytes=old_models,index_retention=index_retention,
        export_driver_seconds=export['actual_driver_seconds'],candidate_accepted=False,
        historical_SDK_strict_FP32_rejected=True,
        limits=['Saved NumPy arrays and file hashes only; no Torch/ORT/SDK/model/BA execution.',
                '94/50 state and ONNX initializer raw byte bindings are frozen metadata checked against ancestors; the graph is file-hashed, not reparsed or executed.',
                'Same-input teacher execution and board source before/after/lifecycle are receipt declarations; host checks its saved arrays and lineage.',
                'Eight historical compiled model and compile-receipt files are independently rehashed on host; exporter retention of their board pins is checked separately.',
                'CPU/ORT component gates do not clear historical SDK rejection or establish online feedback, trajectory or 20Hz acceptance.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export-dir','parent-export-dir','run-dir','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--exporter',type=Path,default=Path(__file__).with_name('export-v42-m16-eof-source-board.py'))
    parser.add_argument('--old-model-root',type=Path,action='append',required=True)
    parser.add_argument('--old-index',type=Path)
    parser.add_argument('--new-index',type=Path)
    args = parser.parse_args(argv)
    if args.output.exists(): raise ValueError('New output required')
    try: result = audit(args)
    except Exception as error: result = dict(status='failed_saved_audit',error=repr(error),auditor_sha256=sha(__file__))
    with args.output.open('x') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','checks','strict_FP32_gate_passed','error')}))
    return int(result['status'] == 'failed_saved_audit')


if __name__ == '__main__': raise SystemExit(main())
