#!/usr/bin/env python3
"""Read saved v41 JSON clocks/topology only; never import an executor or model."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

FROZEN_RUNS = {
    'vggt-v41-online-cpuencoder-r1': '33fb611edf250ae250f06c804507d2a7a180855c401e5457ab8a5ebf217b7f36',
    'vggt-v41-online-bothnpu-r2': '78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    ordered = sorted(values)
    assert ordered and all(math.isfinite(v) and v >= 0 for v in ordered)
    return dict(count=len(ordered), mean_ms=statistics.mean(ordered)*1000,
                median_ms=statistics.median(ordered)*1000,
                p95_ms=ordered[math.ceil(len(ordered)*.95)-1]*1000,
                max_ms=ordered[-1]*1000)


def shape_rows(shapes):
    return [dict(model_key=f'B{b}_D{d}_M16', batch=b, degree=d, patches=16,
                 occurrences=n, net_shape=[b,384,d,16], corr_shape=[b,882,d,16],
                 output_net_shape=[b,384,d,16], output_delta_weight_shape=[b,2,d,16])
            for (b,d),n in sorted(shapes.items(), key=lambda v:(v[0][1],v[0][0]))]


def topology(rows, records):
    shapes = Counter(); capped = unbounded = 0
    for row in rows:
        degrees = row['source_degrees']; slots = row['source_slots']
        assert len(degrees) == len(slots) == len(row['source_patch_counts'])
        assert len(slots) == len(set(slots))
        assert all(type(d) is int and d > 0 for d in degrees)
        assert row['source_patch_counts'] == [d*16 for d in degrees]
        assert sum(degrees)*16 == row['edges']
        groups = Counter(degrees)
        plan = records[row['SDK_record_index']]['degree_plan']
        source_groups = Counter({d: n for d,n in groups.items()})
        plan_groups = Counter()
        for batch in plan['batches']:
            assert batch['patches'] == 16 and batch['sources'] in (1,2)
            plan_groups[batch['degree']] += batch['sources']
        assert source_groups == plan_groups
        count = sum((n+1)//2 for n in groups.values())
        assert count == plan['neural_calls'] == len(plan['batches'])
        shapes.update((n,d) for d,n in groups.items())
        capped += count; unbounded += len(groups)
    return dict(successful_full_calls=len(rows), cap_B2_groups=capped,
                independent_equal_degree_groups=unbounded,
                groups_eliminated=capped-unbounded,
                group_count_reduction_fraction=(capped-unbounded)/capped,
                required_shapes=shape_rows(shapes),
                mathematical_work_reduced=False, input_elements_reduced=False,
                measured_large_batch_latency=False)


def analyze(path, expected):
    assert sha(path) == expected, 'Frozen run SHA differs'
    run = json.loads(path.read_text())
    assert run['schema'] == 'v41_cached_undistort_online_RKNN_research_replay'
    assert run['status'] == 'failed' and run['frames_completed'] == 192
    assert run['candidate_accepted'] is False and run['SDK_FP32_equivalence_validated'] is False
    records = run['online_update_records']
    rows = [c for c in run['update_calls'] if c['route'] == 'full']
    steady = [c for c in rows if c['stage'] == 'stream' and 35 <= c['frame_index'] <= 191]
    assert len(steady) == 157 and {c['frame_index'] for c in steady} == set(range(35,192))
    clock = Counter()
    timings = {}
    for row in steady:
        record = records[row['SDK_record_index']]
        assert record['status'] == 'completed_diagnostic_not_adopted'
        sdk = record['SDK_execution']; groups = sdk['groups']
        assert len(groups) == sdk['sequential_groups'] == record['degree_plan']['neural_calls']
        group_total = sum(g['total_execute_seconds'] for g in groups)
        group_cache = sum(g['cache_call_seconds'] for g in groups)
        forward = sum(g['timing']['forward_seconds'] for g in groups)
        values = dict(callback=record['callback_seconds'], full_graph=sdk['full_graph_seconds'],
            driver_enclosing_update=row['seconds'], SDK_forward_sum=forward,
            callback_outside_full_graph=record['callback_seconds']-sdk['full_graph_seconds'],
            full_graph_outside_cache_calls=sdk['full_graph_seconds']-group_cache,
            cache_calls_outside_session_execute=group_cache-group_total,
            session_execute_outside_SDK_forward=group_total-forward,
            driver_capture_and_metadata=row['capture_and_metadata_seconds'])
        for key in ('pack_seconds','input_set_seconds','run_api_seconds','outputs_get_seconds',
                    'owned_copy_seconds','outputs_release_seconds'):
            values['SDK_'+key+'_sum'] = sum(g['timing'][key] for g in groups)
        for key,value in values.items(): timings.setdefault(key,[]).append(value)
    last = run['online_metrics']['calls'][-1]
    assert last['status'] == 'failed_no_partial_fullgraph'
    plan = last['degree_plan']; eof = Counter()
    for batch in plan['batches']: eof[batch['degree']] += batch['sources']
    cache = run['online_metrics_before_close']['cache']
    return dict(evidence_id=path.parent.name, file=path.name, sha256=expected, original_status=run['status'],
        frames_completed=192, fresh_pose_count=sum(f['result']['fresh_pose_valid'] for f in run['frames']),
        timing_scope='157 successful stream full Updates on observation indices35..191; nested saved clocks, not a new benchmark.',
        steady_clocks={k:summary(v) for k,v in timings.items()},
        all_successful_full_topology=topology(rows,records),
        steady_topology=topology(steady,records),
        first_EOF_failure=dict(status=last['status'], error=last['error'], edges=plan['edges'],
            cap_B2_groups=plan['neural_calls'], independent_equal_degree_groups=len(eof),
            required_shapes=shape_rows(Counter({(n,d):1 for d,n in eof.items()})),
            no_fullgraph_returned=last['fullgraph_returned'] is False),
        cache_state_reported_before_close=dict(created=cache['created'], hits=cache['hits'],
            evicted=cache['evicted'], resident_contexts=cache['resident_contexts'],
            history_count=len(cache['histories'])), SDK_activity_reported=run['SDK_activity'],
        context_close_and_weight_stability='Reported saved metadata only; no runtime/source/weight rerun here.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=dict(schema='v42_v41_saved_Update_performance_analysis', candidate_accepted=False,
        models_or_SDK_executed=False, clocks_newly_measured=False,
        runs={name:analyze(a.runs_dir/name/'run.json',digest) for name,digest in FROZEN_RUNS.items()},
        interpretation=[
            'Removing the callback full-cache metrics copy preserves only activity reporting; all full metrics and close checks remain.',
            'Callback minus full_graph is an upper ceiling on all work outside the full graph, not a measured cache.metrics cost.',
            'Independent equal-degree batching preserves all source/edge inputs and MACs. Fewer groups do not imply proportional run/inputset savings.',
            'An intentionally optimistic proportional fixed-per-group scenario can scale invocation overhead only; the existing inputset/run clocks cannot establish such a cost model.',
            'CPUencoder and bothNPU times came from separate feedback runs with uncontrolled temperature/frequency; they are not a controlled encoder comparison.',
            'Both saved runs retain failed first-EOF status and historical strict FP32 rejection. No adoption or 20Hz validation is claimed.'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,r in result['runs'].items():
        t=r['all_successful_full_topology'];s=r['steady_topology']
        print(name, 'all groups',t['cap_B2_groups'],'->',t['independent_equal_degree_groups'],
              'steady groups',s['cap_B2_groups'],'->',s['independent_equal_degree_groups'])
    print('output_sha256',sha(a.output))


if __name__=='__main__':main()
