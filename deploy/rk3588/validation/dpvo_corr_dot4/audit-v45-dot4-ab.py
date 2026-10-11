#!/usr/bin/env python3
"""Saved NumPy-only dot4 AB audit. No native/model/ORT/SDK/BA execution.

DPVO-derived research: Copyright (c) 2022 Princeton University; MIT LICENSE.
Eight representative outputs are reopened; later repeats have saved metadata.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RUN_SHA = 'd5ee830cc74553223871c67ed97f07988a77630c095ea06a553b09209c58a773'
DRIVER_SHA = 'b6fc741c2342e3cb684d1e643e5aa23f37b2827a41c3bc032422c8b562e7522e'
BUILD_SHA = 'bf74468f6f365204c0ed7e23a8bd7d2f95c9d1316de4c78f21c68aa01e353e2e'
BUILDER_SHA = 'aea69d5e614efeb7b80ab309c04dc821379b6eb96c048be7bc590a5d63e0472f'
CPP_SHA = '8e174eb55e0b114a89b6d61aef9e56e32312633d0b0be44d3ffbca9df67c15e9'
PARENT_CPP_SHA = '3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990'
HELPER_SHA = '9743962325ba1c4ada689769eaf800a617ba1f46f4602062a6dab3b23b5bdf48'
LIBRARIES = dict(A='23ffa5b4b3e9251b8a41e149da2610c9db5818fe5fe148ab18f8b45b41486582',
    B='bd324375294722faeaf02b71645082e9b11e46f64bbdedee7eb52e185d0bb9cc')
CASE_NAMES = ('first_two_frames', 'grow_three_frames', 'reorder_slots', 'delete_middle')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
        tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def load_artifact(directory, item, files):
    name = item['file']
    require(isinstance(name, str) and Path(name).name == name, 'Safe artifact basename required')
    path = Path(directory) / name
    require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
        'Saved artifact file SHA/size differs: ' + name)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy(order='C') for key in archive.files}
    require(set(values) == set(item['arrays']), 'Saved artifact key set differs')
    for key, value in values.items():
        require(metadata(value) == item['arrays'][key], 'Saved array raw metadata differs: ' + name + ':' + key)
        require(value.dtype in (np.dtype('float32'), np.dtype('int64')) and np.isfinite(value).all(),
            'Finite FP32/int64 saved values required')
    files[name] = dict(bytes=item['bytes'], sha256=item['sha256'], arrays=item['arrays'])
    return values


def comparison(actual, reference):
    require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32,
        'Equal FP32 shape/type required')
    require(np.isfinite(actual).all() and np.isfinite(reference).all(), 'Nonfinite saved comparison')
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = a - b
    failed = int(np.count_nonzero(np.abs(error) > 1e-5 + 1e-4 * np.abs(b)))
    denominator = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, failed_elements=failed, gate_passed=failed == 0,
        max_abs=float(np.abs(error).max(initial=0)),
        relative_L2=float(np.linalg.norm(error.ravel()) / denominator) if denominator else None,
        bitwise_equal=actual.tobytes(order='C') == reference.tobytes(order='C'), rtol=1e-4, atol=1e-5)


def check_comparison(recorded, recomputed):
    for key in recomputed:
        if key == 'relative_L2' and recomputed[key] is not None:
            # Descriptive FP64 ARM/x86 reductions may round differently. This
            # tolerance never changes the original elementwise FP32 gate.
            require(type(recorded[key]) in (int, float) and math.isclose(recorded[key],
                recomputed[key], rel_tol=1e-9, abs_tol=1e-15), 'Recorded descriptive relative L2 differs')
        else:
            require(recorded[key] == recomputed[key], 'Recorded comparison differs: ' + key)


def duration(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Finite nonnegative duration required')
    return value


def schedule(calls):
    expected = [(m, r, r == 0) for r in range(4) for m in (('A', 'B') if r % 2 == 0 else ('B', 'A'))]
    require([(c['method'], c['round'], c['cold']) for c in calls] == expected,
        'Exactly two ABBA cycles/cold first round required')


def audit(directory, run, build):
    require(run['schema'] == 'v45_saved_features_cached_corr_dot4_ABBA'
        and run['status'] == 'completed_dot4_AB_bitwise_not_adopted' and run['script_sha256'] == DRIVER_SHA,
        'Exact completed dot4 AB source/schema required')
    require(run['candidate_accepted'] is False and run['realtime_validated'] is False
        and run['neural_forward_executed'] is False and run['Update_or_BA_executed'] is False
        and run['actual_board_native_correlation'] is True, 'Component-only research scope required')
    require(run['cleanup_errors'] == [] and run['affinity_restored'] is True
        and run['sources_and_inputs_unchanged'] is True
        and run['source_pins'] == run['source_hashes_after'], 'Source/affinity/cleanup declarations differ')
    require(run['cpu_affinity'] == [4, 5, 6, 7] and run['native_threads'] == 4
        and run['OMP_WAIT_POLICY'] == 'PASSIVE' and run['GOMP_SPINCOUNT'] == '0', 'Recorded CPU settings differ')
    require(run['candidate_build_sha256'] == BUILD_SHA and run['candidate_library_sha256'] == LIBRARIES['B']
        and run['parent_library_sha256'] == LIBRARIES['A'], 'Actual build/library ancestry differs')
    expected_pins = {'run-v45-dot4-corr-ab-board.py': DRIVER_SHA,
        'run-v36-frame-corr-ab-board.py': '531777cd463d4ace4299960a7034cfe42b108db68391c3241cbc6d616a560c9d',
        'dpvo_frame_corr.py': HELPER_SHA, 'libdpvo_frame_corr.so': LIBRARIES['A'],
        'libdpvo_frame_corr_dot4.so': LIBRARIES['B'],
        'capture.json': 'e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088',
        'motion.json': '55a7db19d1a9bfbf986da62ed999fc8b122add0fca84f01681e6ad0b82ba4e86'}
    pairs = {(Path(p).name, digest) for p, digest in run['source_pins'].items()}
    require(set(expected_pins.items()) <= pairs and ('build.json', BUILD_SHA) in pairs
        and ('build.json', run['parent_build_sha256']) in pairs, 'Exact source/capture/build pins absent')
    require(build['schema'] == 'v45_horizontal_dot4_native_build'
        and build['status'] == 'compiled_dot4_candidate_not_numerically_validated'
        and build['script_sha256'] == BUILDER_SHA and build['output']['sha256'] == LIBRARIES['B']
        and build['source_sha256']['dpvo_frame_corr_dot4.cpp'] == CPP_SHA
        and build['source_sha256']['dpvo_corr.cpp'] == PARENT_CPP_SHA
        and build['source_sha256']['dpvo_frame_corr.py'] == HELPER_SHA
        and build['source_pins'] == build['source_hashes_after'] and build['cleanup_errors'] == []
        and build['sources_and_inputs_unchanged'] is True, 'Candidate build identity differs')
    require('-fno-fast-math' in build['command'] and '-ffp-contract=off' in build['command']
        and not set(build['command']) & {'-ffast-math', '-Ofast'}, 'Frozen strict floating-point flags required')
    require([c['name'] for c in run['cases']] == list(CASE_NAMES), 'Exactly four original cases required')
    files, summaries, graphs, refs = {}, [], {}, {}
    histories = dict(A=[], B=[])
    frame_bytes = 128 * (96 * 9 + 116 * 152 + 29 * 38) * 4
    for index, case in enumerate(run['cases']):
        name, edges = case['name'], case['edges']
        require(edges == (96, 864, 864, 384)[index] and len(case['frame_ids']) == (2, 3, 3, 2)[index],
            'Actual graph/frame count differs')
        graph = load_artifact(directory, case['graph'], files)
        reference = load_artifact(directory, case['original_CPU_reference'], files)
        require(set(graph) == {'ii', 'jj', 'kk', 'coords'} and set(reference) == {'corr'}, 'Graph/ref key set differs')
        for key in ('ii', 'jj', 'kk'):
            require(graph[key].dtype == np.int64 and graph[key].shape == (edges,), 'Index shape/dtype differs')
        require(graph['coords'].dtype == np.float32 and graph['coords'].shape == (1, edges, 2, 3, 3)
            and np.max(np.abs(graph['coords']), initial=0) < 2**30, 'Coordinate shape/native safe domain differs')
        require(np.array_equal(graph['ii'], graph['kk'] // 96) and np.all(graph['kk'] >= 0)
            and np.all(graph['kk'] < len(case['frame_ids']) * 96) and np.all(graph['jj'] >= 0)
            and np.all(graph['jj'] < len(case['frame_ids'])), 'Frame/patch source ownership differs')
        require(reference['corr'].dtype == np.float32 and reference['corr'].shape == (1, edges, 882), 'CPU ref layout differs')
        graphs[name], refs[name] = graph, reference['corr']
        schedule(case['calls'])
        output, checks = {}, {}
        for method in ('A', 'B'):
            first = next(c for c in case['calls'] if c['method'] == method)
            saved = load_artifact(directory, first['artifact'], files)
            require(set(saved) == {'corr'}, 'Output key set differs')
            output[method] = saved['corr']
            checks[method] = comparison(saved['corr'], reference['corr'])
            require(checks[method]['gate_passed'], 'Saved original CPU FP32 gate failed')
        ab = comparison(output['B'], output['A'])
        require(ab['bitwise_equal'] and ab['gate_passed'], 'Saved A/B bit pattern differs')
        warm = {}
        for call in case['calls']:
            m, r = call['method'], call['round']
            require(call['array'] == metadata(output[m]) and call['repeat_bitwise_equal'] is True,
                'All repeat metadata must match representative saved bytes')
            require(('artifact' in call) == (r == 0), 'Only the first representative per method is saved')
            check_comparison(call['versus_original_CPU'], checks[m]); check_comparison(call['B_vs_A'], ab)
            wall = duration(call['full_callback_seconds']); inner = call['internal_callback_record']
            require(inner['index'] == len(histories[m]) and inner['epoch'] == 0 and inner['edges'] == edges
                and inner['edge_order_preserved'] is True and inner['no_historical_feature_concatenation'] is True,
                'Internal callback identity/edge order differs')
            histories[m].append(inner)
            missing = (2, 1, 0, 0)[index] if r == 0 else 0
            require(inner['prepare_misses'] == missing and inner['cache_hits'] == len(case['frame_ids']) - missing
                and inner['cached_frames'] == len(case['frame_ids'])
                and inner['current_context_bytes'] == len(case['frame_ids']) * frame_bytes,
                'Incremental preparation/resident allocation differs')
            expect_ids = case['frame_ids'] if index == 0 and r == 0 else (case['frame_ids'][2:] if index == 1 and r == 0 else [])
            require(inner['prepared_frame_ids'] == expect_ids and len(inner['preparation']) == missing,
                'Prepared immutable frame IDs differ')
            query = inner['native_query']; native = query['native']
            require(query['includes_prepare'] is False and query['output_finite_validated_by_native'] is True
                and native['abi_version'] == 1 and native['threads'] == 4 and native['neon'] == 1
                and native['context_bytes'] == len(case['frame_ids']) * frame_bytes
                and native['output_elements'] == edges * 882, 'Actual native stats differ')
            chain = [native['kernel_seconds'], native['total_seconds'], query['native_call_seconds'],
                query['total_seconds'], inner['total_seconds'], wall]
            require(all(duration(a) <= duration(b) + 1e-9 for a, b in zip(chain, chain[1:])), 'Nested clock intervals differ')
            for key in ('validation_and_lock_seconds', 'eviction_seconds', 'preparation_seconds'):
                require(duration(inner[key]) <= inner['total_seconds'], 'Nested setup interval exceeds callback')
            require(native['layout_seconds'] == native['pool_seconds'] == 0, 'Query unexpectedly includes feature preparation')
        for m in ('A', 'B'):
            samples = [duration(c['full_callback_seconds']) for c in case['calls'] if c['method'] == m and not c['cold']]
            mean = sum(samples) / len(samples)
            require(len(samples) == 3 and math.isclose(mean, case['warm_mean_seconds'][m], rel_tol=1e-12, abs_tol=1e-15),
                'Warm mean/cold exclusion differs')
            warm[m] = dict(count=3, samples_seconds=samples, mean_seconds=mean)
        summaries.append(dict(name=name, edges=edges, saved_B_vs_A=ab, saved_vs_original_CPU=checks,
            warm_callback=warm, descriptive_wall_reduction_fraction=1 - warm['B']['mean_seconds'] / warm['A']['mean_seconds']))
    full, full_ref = graphs['grow_three_frames'], refs['grow_three_frames']
    full_ids = run['cases'][1]['frame_ids']
    require(run['cases'][0]['frame_ids'] == full_ids[:2], 'Initial frame identity differs')
    for name, permutation in (('reorder_slots', (2, 0, 1)), ('delete_middle', (0, 2))):
        inverse = np.full(3, -1, dtype=np.int64); inverse[list(permutation)] = np.arange(len(permutation))
        keep = (inverse[full['ii']] >= 0) & (inverse[full['jj']] >= 0)
        transformed = graphs[name]
        require(np.array_equal(transformed['ii'], inverse[full['ii'][keep]])
            and np.array_equal(transformed['jj'], inverse[full['jj'][keep]])
            and np.array_equal(transformed['kk'], inverse[full['ii'][keep]] * 96 + full['kk'][keep] % 96)
            and transformed['coords'].tobytes() == np.ascontiguousarray(full['coords'][:, keep]).tobytes()
            and refs[name].tobytes() == np.ascontiguousarray(full_ref[:, keep]).tobytes(),
            'Slot reorder/delete must preserve selected original edge order/xy/reference')
        require(run['cases'][CASE_NAMES.index(name)]['frame_ids'] == [full_ids[i] for i in permutation], 'Slot frame IDs differ')
    lifecycle = {}
    for m in ('A', 'B'):
        before, after = (run['lifecycle'][m][key] for key in ('before_close', 'after_close'))
        for row in (before, after):
            require(row['library_sha256'] == LIBRARIES[m] and row['attempted_calls'] == row['completed_calls'] == 16
                and row['failed_calls'] == 0 and row['prepared_frames'] == 3 and row['cache_hits'] == 37
                and row['evicted_frames'] == 1 and row['resets'] == 0 and row['calls'] == histories[m]
                and row['peak_context_bytes'] == 3 * frame_bytes and row['threads'] == 4
                and row['patches_per_frame'] == 96 and row['channels'] == 128 and row['patch_shape'] == [3, 3],
                'Cache histories/counter/memory identity differs')
            for k, field in (('total_seconds', 'total_seconds'), ('preparation_seconds', 'preparation_seconds')):
                require(math.isclose(row['timing_sums'][k], sum(c[field] for c in histories[m]), rel_tol=1e-12, abs_tol=1e-15),
                    'Cache timing sum differs')
            require(math.isclose(row['timing_sums']['query_seconds'], sum(c['native_query']['total_seconds'] for c in histories[m]), rel_tol=1e-12), 'Query timing sum differs')
            require(math.isclose(row['timing_sums']['kernel_seconds'], sum(c['native_query']['native']['kernel_seconds'] for c in histories[m]), rel_tol=1e-12), 'Kernel timing sum differs')
        require(before['closed'] is False and before['current_cached_frames'] == 2
            and before['current_context_bytes'] == 2 * frame_bytes and before['destroyed_frames'] == 1
            and after['closed'] is True and after['current_cached_frames'] == after['current_context_bytes'] == 0
            and after['destroyed_frames'] == 3, 'Cache close must destroy all owned snapshots')
        require([(p['frame_id'], p['epoch'], p['context_bytes'], p['feature_snapshot_owned']) for p in after['preparations']]
            == [(i, 0, frame_bytes, True) for i in full_ids], 'Exactly three owned preparations required')
        require([(e['frame_id'], e['reason'], e['bytes']) for e in after['evictions']]
            == [(full_ids[1], 'not_retained', frame_bytes), (full_ids[0], 'close', frame_bytes), (full_ids[2], 'close', frame_bytes)],
            'Deletion and final context destruction history differs')
        lifecycle[m] = dict(calls=16, prepared_frames=3, cache_hits=37, evicted_frames=1,
            destroyed_frames=3, peak_context_bytes=3 * frame_bytes, closed=True, current_context_bytes=0)
    require(len(files) == 16 and run['all_FP32_gates_passed'] is True and run['A_B_bitwise_equal'] is True,
        'Final gate/count declarations differ')
    return dict(schema='v45_dot4_saved_AB_independent_review',
        status='passed_saved_AB_bitwise_and_fixed_CPU_gate_not_realtime_acceptance',
        candidate_accepted=False, independent_saved_AB_pairs=4, independent_saved_CPU_comparisons=8,
        representative_output_files=8, graph_files=4, CPU_reference_files=4, recorded_calls=32,
        fixed_gate=dict(rtol=1e-4, atol=1e-5), correlation_domain='finite signed real; not probability/weight',
        cases=summaries, lifecycle=lifecycle, raw_artifacts=files, source_declarations_unchanged=True,
        affinity_restored=True, cleanup_errors=[], limits=[
            'Only saved NumPy/JSON/source bytes; no native/model/ORT/SDK/BA execution.',
            'Eight representative outputs independently reopened. Later repeats checked via raw metadata/history, not separately saved raw arrays.',
            'Original CPU references are the saved representative arrays with declared v26 source hashes; original feature payloads/board binaries are not independently reopened by this audit.',
            'Round-zero cold labels exclude first calls per case; reorder/delete reuse already prepared frames. Nested kernel/setup clocks cannot be added again to callback.',
            'Four unpaced same-feature scenarios, n3 warm per method, no controlled thermal/frequency or complete online replay/20Hz claim.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-sha256', default=RUN_SHA)
    parser.add_argument('--candidate-build', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'New audit output required')
    try:
        require(args.run_sha256 == RUN_SHA and sha(args.run_dir / 'run.json') == RUN_SHA, 'Exact actual run SHA required')
        require(sha(args.candidate_build) == BUILD_SHA, 'Actual candidate build SHA differs')
        source = {'run-v45-dot4-corr-ab-board.py': DRIVER_SHA, 'build-v45-frame-corr-dot4-board.py': BUILDER_SHA,
            'dpvo_frame_corr_dot4.cpp': CPP_SHA, 'dpvo_corr.cpp': PARENT_CPP_SHA, 'dpvo_frame_corr.py': HELPER_SHA}
        for name, expected in source.items():
            require(sha(args.source_dir / name) == expected, 'Frozen source bytes differ: ' + name)
        result = audit(args.run_dir, json.loads((args.run_dir / 'run.json').read_text()), json.loads(args.candidate_build.read_text()))
        result.update(actual_run_sha256=RUN_SHA, candidate_build_sha256=BUILD_SHA,
            source_sha256=source, auditor_sha256=sha(__file__))
    except Exception as error:
        result = dict(status='failed_saved_dot4_AB_audit', error=repr(error), candidate_accepted=False, auditor_sha256=sha(__file__))
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result.get(key) for key in ('status', 'error', 'recorded_calls', 'independent_saved_AB_pairs')}))
    return int(result['status'] == 'failed_saved_dot4_AB_audit')


if __name__ == '__main__':
    raise SystemExit(main())
