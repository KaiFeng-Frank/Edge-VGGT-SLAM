#!/usr/bin/env python3
"""Portable NumPy-only audit of saved v36 correlation AB arrays and receipts.

No native library, Torch, SDK, model, geometry or device execution is imported.
Later repeated calls have only raw-array hash metadata; that evidence is labelled
separately from the eight independently recomputed representative arrays.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
CAPTURE_SHA = 'e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088'
MOTION_SHA = '55a7db19d1a9bfbf986da62ed999fc8b122add0fca84f01681e6ad0b82ba4e86'
AB_SHA = '531777cd463d4ace4299960a7034cfe42b108db68391c3241cbc6d616a560c9d'
BUILD_SHA = 'b1f2553aa0e7786dd1343288ea67dd05d5b6e9f470c88693a6134d5d82756d2d'
CORE_PINS = {
    'dpvo_frame_corr.cpp': 'a04ab5a4b86cfd189ef194893b9b07c06742e21141056cc94a120b42c9d2899d',
    'dpvo_frame_corr.h': '55a39130998285aaf50d0e2f5b03660f5ad19f629907cea3ef850f552eb10e76',
    'dpvo_frame_corr.py': '9743962325ba1c4ada689769eaf800a617ba1f46f4602062a6dab3b23b5bdf48',
}
PARENT_PINS = {
    'dpvo_corr.cpp': '3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990',
    'dpvo_corr.h': '1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932',
    'dpvo_corr.py': 'a6c37bf4b840b492736d5c5916be6e2c0e8680693d01d0fe3a5fe62702ecb779',
}
OLD_LIBRARY_SHA = '829d1b42fa24d28b1e280b882c0a9e8556b0e7cdbe02f7489eeef15ba3581bca'
M = 96
PER_FRAME_BYTES = 10034176
CASE_NAMES = ('first_motion_two_frames', 'grow_to_three_frames',
              'reorder_slots_same_three_frames', 'delete_middle_slot_preserve_edge_order')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def comparison(actual, reference):
    if actual.dtype != np.float32 or reference.dtype != np.float32 or actual.shape != reference.shape:
        raise ValueError('Equal FP32 array shapes required')
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        return dict(finite=False, failed_elements=None, gate_passed=False, max_abs=None,
                    relative_L2=None, bitwise_equal=False, rtol=1e-4, atol=1e-5)
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = a-b
    failed = int(np.count_nonzero(np.abs(error) > 1e-5+1e-4*np.abs(b)))
    norm = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, failed_elements=failed, gate_passed=failed == 0,
                max_abs=float(np.max(np.abs(error), initial=0)),
                relative_L2=float(np.linalg.norm(error.ravel())/norm) if norm else None,
                bitwise_equal=actual.tobytes() == reference.tobytes(), rtol=1e-4, atol=1e-5)


class Checks:
    def __init__(self):
        self.count = 0

    def require(self, condition, message):
        self.count += 1
        if not condition:
            raise ValueError(message)

    def seconds(self, value, message):
        self.require(type(value) in (float, int) and math.isfinite(value) and value >= 0, message)

    def close(self, actual, expected, message, *, atol=1e-12):
        self.require(type(actual) in (float, int) and math.isfinite(actual)
                     and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=atol), message)


def load_artifact(folder, item, checks, files, *, namespace):
    name = item['file']
    checks.require(isinstance(name, str) and Path(name).name == name, 'Safe artifact basename required')
    path = Path(folder)/name
    digest = sha(path)
    checks.require(digest == item['sha256'] and path.stat().st_size == item['bytes'], 'Artifact SHA/bytes: '+name)
    files[namespace+'/'+name] = digest
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    checks.require(set(values) == set(item['arrays']), 'Artifact array key set: '+name)
    for key, value in values.items():
        checks.require(metadata(value) == item['arrays'][key], 'Raw array metadata: '+name+'/'+key)
        checks.require(np.isfinite(value).all(), 'Nonfinite saved array: '+name+'/'+key)
    return values


def check_comparison(recorded, actual, checks, label):
    for key in ('finite', 'failed_elements', 'gate_passed', 'max_abs', 'bitwise_equal', 'rtol', 'atol'):
        checks.require(recorded[key] == actual[key], label+' comparison '+key)
    # ARM/x86 norm reduction rounding is descriptive, never a gate adjustment.
    if actual['relative_L2'] is None:
        checks.require(recorded['relative_L2'] is None, label+' zero-reference norm')
    else:
        checks.close(recorded['relative_L2'], actual['relative_L2'], label+' relative norm', atol=1e-15)


def expected_cases(first, full, first_reference, full_reference):
    """Independent current-slot rewrite; select edges in their original order."""
    result = [(first, first_reference, (0, 1))]
    for order in ((0, 1, 2), (2, 0, 1), (0, 2)):
        slot = {original: current for current, original in enumerate(order)}
        selected = np.array([edge for edge, (i, j) in enumerate(zip(full['ii'], full['jj']))
                             if int(i) in slot and int(j) in slot], dtype=np.int64)
        i = np.array([slot[int(full['ii'][edge])] for edge in selected], dtype=np.int64)
        j = np.array([slot[int(full['jj'][edge])] for edge in selected], dtype=np.int64)
        k = i*M + full['kk'][selected] % M
        graph = dict(ii=i, jj=j, kk=k, coords=full['coords'][:, selected])
        result.append((graph, full_reference[:, selected], order))
    return result


def check_repeat(call, representative, checks):
    checks.require(call['array'] == representative and call['repeat_bitwise_equal'] is True,
                   'Repeat raw-array metadata/hash differs')


def unchanged(record, checks, label):
    before = record['source_pins']
    checks.require(bool(before) and record['sources_and_inputs_unchanged'] is True
                   and before == record['source_hashes_after'], label+' before/after pins differ')
    checks.require(all(isinstance(key, str) and isinstance(value, str) and len(value) == 64
                       and all(c in '0123456789abcdef' for c in value) for key, value in before.items()),
                   label+' malformed source hash')


def audit(run_dir, capture_dir, motion_dir, candidate_build, source_dir, parent_dir):
    checks, files = Checks(), {}
    result = dict(schema='v36_saved_array_correlation_independent_review', status='failed',
                  script_sha256=sha(__file__), candidate_accepted=False, realtime_validated=False)
    try:
        run_file, capture_file = Path(run_dir)/'run.json', Path(capture_dir)/'capture.json'
        motion_file, build_file = Path(motion_dir)/'motion.json', Path(candidate_build)
        raw = {'run': run_file, 'capture': capture_file, 'motion': motion_file, 'build': build_file}
        files.update({role+'/'+path.name: sha(path) for role, path in raw.items()})
        run, capture, motion, build = [json.loads(path.read_text()) for path in raw.values()]
        checks.require(files['capture/capture.json'] == CAPTURE_SHA and files['motion/motion.json'] == MOTION_SHA,
                       'Frozen original capture/motion receipt differs')
        checks.require(capture['status'] == 'completed_encoder_cpu_probe_not_VO'
                       and motion['status'] == 'completed_dynamic_neural_probe_not_VO'
                       and motion['capture_sha256'] == CAPTURE_SHA, 'Original capture chain/status differs')
        checks.require(capture['sources_and_inputs_unchanged'] is True and motion['sources_and_inputs_unchanged'] is True,
                       'Original saved reference declares changed source')
        checks.require(run['schema'] == 'v36_saved_array_frame_correlation_AB_probe'
                       and run['script_sha256'] == AB_SHA and run['capture_sha256'] == CAPTURE_SHA
                       and run['motion_sha256'] == MOTION_SHA, 'AB receipt source/recipe chain differs')
        checks.require(run['candidate_accepted'] is False and run['realtime_validated'] is False
                       and run['actual_board_execution'] is True and run['neural_forward_executed'] is False
                       and run['BA_executed'] is False and run['ORB_LK_PnP_used'] is False, 'Research scope differs')
        unchanged(run, checks, 'AB')
        unchanged(build, checks, 'build')
        checks.require(build['schema'] == 'v36_immutable_frame_correlation_native_build'
                       and build['status'] == 'compiled_and_synthetic_smoke_passed'
                       and build['script_sha256'] == BUILD_SHA and build['cleanup_errors'] == [], 'Successful build required')
        checks.require(build['candidate_source_sha256'] == CORE_PINS
                       and build['parent_source_sha256'] == {k: PARENT_PINS[k] for k in ('dpvo_corr.cpp', 'dpvo_corr.h')},
                       'Frozen native source chain differs')
        checks.require(run['candidate_build_sha256'] == files['build/build.json']
                       and run['candidate_library_sha256'] == build['output']['sha256']
                       and run['candidate_helper_sha256'] == CORE_PINS['dpvo_frame_corr.py']
                       and run['parent_library_sha256'] == OLD_LIBRARY_SHA, 'Build/library/helper chain differs')
        for folder, pins in ((Path(source_dir), dict(CORE_PINS, **{'run-v36-frame-corr-ab-board.py': AB_SHA,
                                                               'build-v36-frame-corr-board.py': BUILD_SHA})),
                             (Path(parent_dir), PARENT_PINS)):
            for name, expected in pins.items():
                digest = sha(folder/name)
                checks.require(digest == expected and digest in (*run['source_pins'].values(), *build['source_pins'].values()),
                               'Local source pin: '+name)
                files['source/'+name] = digest
        checks.require(run['cleanup_errors'] == [] and run['affinity_restored'] is True
                       and run['cpu_affinity'] == [4, 5, 6, 7] and run['native_threads'] == 4, 'Cleanup/thread evidence differs')
        # Before/after board paths are metadata evidence on a portable host. We
        # separately rehash all supplied source/reference/output artifacts here.
        patches = load_artifact(motion_dir, motion['patches'], checks, files, namespace='motion')['gmap']
        checks.require(patches.shape == (1, 288, 128, 3, 3), 'Original 3x96 source patch contract')
        checks.require(len(run['source_frames']) == len(motion['frame_ids']) == 3, 'Three source frame identities')
        for index, row in enumerate(run['source_frames']):
            original = next(c for c in capture['calls'] if c['frame_index'] == index)
            value = load_artifact(capture_dir, dict(original['artifact'], arrays=original['arrays']),
                                  checks, files, namespace='capture')['fmap']
            checks.require(value.shape == (1, 1, 128, 116, 152) and row['frame_id'] == motion['frame_ids'][index],
                           'Captured source frame/feature shape differs')
            checks.require(row['fmap'] == metadata(value[0, 0])
                           and row['gmap'] == metadata(patches[0, index*M:(index+1)*M]), 'Per-frame source raw SHA differs')
        scenarios = {}
        for scenario in motion['scenarios']:
            graph = load_artifact(motion_dir, scenario['graph'], checks, files, namespace='motion')
            reference = load_artifact(motion_dir, scenario['calls'][0]['artifact'], checks, files, namespace='motion')['corr']
            checks.require(graph['ii'].dtype == graph['jj'].dtype == graph['kk'].dtype == np.int64
                           and graph['coords'].dtype == reference.dtype == np.float32
                           and np.array_equal(graph['ii'], graph['kk']//M), 'Original source ownership differs')
            scenarios[graph['kk'].size] = ({k: graph[k] for k in ('ii', 'jj', 'kk', 'coords')}, reference)
        first, first_reference = scenarios[96]
        full, full_reference = scenarios[864]
        checks.require(np.all(first['ii'] == 0) and np.all(first['jj'] == 1), 'Original first-motion graph differs')
        expected = expected_cases(first, full, first_reference, full_reference)
        checks.require([case['name'] for case in run['cases']] == list(CASE_NAMES), 'Four actual case order differs')
        summaries, gates, bitwise, new_records, old_records = [], [], [], [], []
        prior_end = -math.inf
        for case_index, (case, (graph, reference, order)) in enumerate(zip(run['cases'], expected)):
            name = case['name']
            actual_graph = load_artifact(run_dir, case['graph'], checks, files, namespace='AB')
            actual_ref = load_artifact(run_dir, case['original_CPU_reference'], checks, files, namespace='AB')['corr']
            checks.require(set(actual_graph) == set(graph), name+' graph keys')
            for key, value in graph.items():
                checks.require(metadata(actual_graph[key]) == metadata(value), name+' original edge order/slots/xy '+key)
            checks.require(metadata(actual_ref) == metadata(reference), name+' independent original CPU reference association')
            checks.require(case['frame_ids_current'] == [motion['frame_ids'][slot] for slot in order], name+' frame order')
            prepared, resident = (2 if case_index == 0 else 3), len(order)
            checks.require(case['expected_prepared_frames'] == prepared and case['expected_resident_frames'] == resident,
                           name+' prepare/residency declaration')
            calls = case['calls']
            schedule = [(round_index, mode) for round_index in range(4)
                        for mode in (('old', 'new') if round_index % 2 == 0 else ('new', 'old'))]
            checks.require([(call['round'], call['mode']) for call in calls] == schedule, name+' ABBA eight-call schedule')
            saved, stats = {}, {}
            for mode in ('old', 'new'):
                first_call = next(call for call in calls if call['mode'] == mode)
                value = load_artifact(run_dir, first_call['artifact'], checks, files, namespace='AB')
                checks.require(set(value) == {'corr'} and value['corr'].shape == (1, graph['kk'].size, 882), name+' output contract')
                saved[mode] = value['corr']
                stats[mode] = comparison(value['corr'], reference)
                gates.append(stats[mode]['gate_passed'])
            paired = comparison(saved['new'], saved['old'])
            gates.append(paired['gate_passed'])
            bitwise.append(paired['bitwise_equal'])
            for call_index, call in enumerate(calls):
                mode, round_index = call['mode'], call['round']
                checks.require(call['timing_role'] == ('cold_graph' if round_index == 0 else 'warm_same_graph'), name+' timing role')
                for key in ('begin_monotonic', 'end_monotonic', 'full_callback_seconds'):
                    checks.seconds(call[key], name+' '+key)
                checks.require(call['begin_monotonic'] >= prior_end, 'Interleaved calls must not overlap')
                prior_end = call['end_monotonic']
                checks.close(call['full_callback_seconds'], call['end_monotonic']-call['begin_monotonic'], name+' callback elapsed')
                check_repeat(call, metadata(saved[mode]), checks)
                checks.require(('artifact' in call) == (round_index == 0), name+' representative saving scope')
                check_comparison(call['versus_original_CPU'], stats[mode], checks, name+'/'+mode)
                check_comparison(call['old_new_comparison'], paired, checks, name+'/paired')
                internal = call['internal_callback_record']
                (new_records if mode == 'new' else old_records).append(internal)
                query, native = internal['native_query'], internal['native_query']['native']
                for duration in (internal['total_seconds'], query['total_seconds'], query['native_call_seconds'], native['kernel_seconds']):
                    checks.seconds(duration, name+' nested native duration')
                checks.require(native['neon'] == 1 and native['threads'] == 4 and native['output_elements'] == saved[mode].size,
                               name+' ARM native shape/thread evidence')
                checks.require(native['kernel_seconds'] <= query['native_call_seconds'] <= query['total_seconds'] + 1e-9
                               and query['total_seconds'] <= internal['total_seconds'] + 1e-9
                               and internal['total_seconds'] <= call['full_callback_seconds'] + 1e-9
                               and query['includes_prepare'] is False, name+' nested timing boundaries')
                if mode == 'new':
                    missing = [motion['frame_ids'][s] for s in ((0, 1) if case_index == 0 else (2,) if case_index == 1 else ())] if round_index == 0 else []
                    checks.require(internal['prepared_frame_ids'] == missing and internal['prepare_misses'] == len(missing)
                                   and internal['cache_hits'] == resident-len(missing), name+' incremental prepare identity/count')
                    checks.require(call['prepared_frames'] == prepared and call['resident_frames'] == resident
                                   and call['resident_bytes'] == resident*PER_FRAME_BYTES
                                   and internal['current_context_bytes'] == native['context_bytes'] == resident*PER_FRAME_BYTES,
                                   name+' bounded cache payload')
                    checks.require(internal['edge_order_preserved'] is True and internal['no_historical_feature_concatenation'] is True,
                                   name+' original query order declaration')
                    checks.require([p['frame_id'] for p in internal['preparation']] == missing, name+' preparation rows')
                    for preparation in internal['preparation']:
                        checks.require(preparation['context_bytes'] == PER_FRAME_BYTES and preparation['feature_snapshot_owned'] is True,
                                       name+' owned native frame payload')
                    subtotal = sum(internal[k] for k in ('validation_and_lock_seconds', 'eviction_seconds', 'preparation_seconds')) + query['total_seconds']
                    checks.require(subtotal <= internal['total_seconds'] + 1e-9, name+' disjoint internal times not double counted')
            warm = {mode: sum(call['full_callback_seconds'] for call in calls if call['mode'] == mode and call['round'] > 0)/3
                    for mode in ('old', 'new')}
            for mode in ('old', 'new'):
                checks.close(case['warm_means_seconds'][mode], warm[mode], name+' warm mean '+mode)
            initial = {mode: next(call['full_callback_seconds'] for call in calls if call['mode'] == mode and call['round'] == 0)
                       for mode in ('old', 'new')}
            summaries.append(dict(name=name, edges=int(graph['kk'].size), first_callback_seconds=initial,
                                  last_three_warm_means_seconds=warm, old_new_bitwise_equal=paired['bitwise_equal'],
                                  versus_original_CPU=stats, prepared_frames_cumulative=prepared,
                                  resident_frames=resident, resident_payload_bytes=resident*PER_FRAME_BYTES))
        lifecycle = run['cache_lifecycle']
        new_before, new_after = lifecycle['new']['before_close'], lifecycle['new']['after_close']
        for field, value in {'attempted_calls': 16, 'completed_calls': 16, 'failed_calls': 0, 'prepared_frames': 3,
                             'cache_hits': 37, 'evicted_frames': 1, 'resets': 0, 'peak_context_bytes': 3*PER_FRAME_BYTES}.items():
            checks.require(new_before[field] == new_after[field] == value, 'Final native counter '+field)
        checks.require(new_before['calls'] == new_after['calls'] == new_records
                       and lifecycle['old']['after_close']['calls'] == old_records, 'Final call histories match actual schedule')
        checks.require(new_before['destroyed_frames'] == 1 and new_after['destroyed_frames'] == 3
                       and new_before['current_cached_frames'] == 2 and new_before['current_context_bytes'] == 2*PER_FRAME_BYTES
                       and new_after['closed'] is True and new_after['current_cached_frames'] == new_after['current_context_bytes'] == 0,
                       'New cache partial eviction and final close')
        checks.require([p['frame_id'] for p in new_after['preparations']] == motion['frame_ids'], 'Exactly three preparations')
        checks.require([(p['frame_id'], p['reason']) for p in new_after['evictions']] ==
                       [(motion['frame_ids'][1], 'not_retained'), (motion['frame_ids'][0], 'close'), (motion['frame_ids'][2], 'close')],
                       'One removed frame then all remaining native handles destroyed')
        old_after = lifecycle['old']['after_close']
        checks.require(old_after['attempted_calls'] == old_after['completed_calls'] == 16
                       and old_after['current_context_open'] is False and len(old_after['generations']) == len(old_after['closed_contexts']) == 4
                       and all(context['closed'] and context['current_context_bytes'] == 0 for context in old_after['closed_contexts']),
                       'All four old graph-generation contexts closed')
        for mode in ('old', 'new'):
            checks.seconds(lifecycle[mode]['close_seconds'], mode+' final close time')
            checks.seconds(run['constructor_seconds'][mode], mode+' library construction time')
        checks.require(run['all_fixed_FP32_gates_passed'] is all(gates)
                       and run['old_new_bitwise_equal'] is all(bitwise)
                       and run['incremental_preparation_checks_passed'] is True, 'Reported numerical/lifecycle gates differ')
        expected_status = 'completed_saved_array_correlation_AB_probe' if all(gates) else 'completed_numerical_rejected'
        checks.require(run['status'] == expected_status, 'Execution/numerical status mismatch')
        result.update(status='passed', raw_receipt_sha256={role: sha(path) for role, path in raw.items()},
                      file_sha256=files, calls=32, representative_arrays_recomputed=8,
                      later_repeat_array_metadata_checked=24, all_fixed_FP32_gates_passed=all(gates),
                      old_new_bitwise_equal=all(bitwise), cases=summaries,
                      native_lifecycle=dict(prepared_frames=3, cache_hits=37, evicted_frames=1, destroyed_frames=3,
                                            max_resident_payload_bytes=3*PER_FRAME_BYTES, closed=True, retained_payload_bytes=0,
                                            original_contexts_closed=4),
                      timing=dict(constructor_seconds=run['constructor_seconds'],
                                  close_seconds={mode: lifecycle[mode]['close_seconds'] for mode in ('old', 'new')},
                                  new_prepare_total_seconds=new_after['timing_sums']['preparation_seconds'],
                                  scope='Full callback includes miss preparation, current-order packing/query and Python bookkeeping; hashes/statistics/saves outside. First transition callback and last-three same-graph means are separate.'),
                      limitations=['Eight representative arrays independently checked; later repeats have raw-array hash/metadata evidence only.',
                                   'Board absolute source paths are not opened on this host; supplied local sources/references and all 16 AB NPZ files are rehashed.',
                                   'Three original captured frames with M96, short interleaved CPU correlation only; no new model/Update/BA or full M48 replay.',
                                   'No temperature/frequency control or sustained throughput evidence; no full VO, NPU, original VGGT or 20 Hz acceptance.'])
    except Exception as error:
        result.update(status='failed', error=str(error), file_sha256=files)
    result['checks'] = dict(count=checks.count, failed=0 if result['status'] == 'passed' else 1)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'capture-dir', 'motion-dir', 'candidate-build', 'source-dir', 'parent-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('New output receipt required; failed evidence is never overwritten')
    result = audit(args.run_dir, args.capture_dir, args.motion_dir, args.candidate_build, args.source_dir, args.parent_dir)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'checks', 'error', 'all_fixed_FP32_gates_passed', 'old_new_bitwise_equal')}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
