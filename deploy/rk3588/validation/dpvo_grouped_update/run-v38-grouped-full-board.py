#!/usr/bin/env python3
"""Three frozen full M48 Update graphs, true-B SDK groups; diagnostic only.

No Torch or ORT forwards; neural execution uses board RKNN only.
No recurrent commit, BA, trajectory or adoption.
All original inputs are immutable; SDK group arrays remain owned until their
hashes are collected outside the timed full-graph call. DPVO MIT attribution.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

MANIFEST_SHA = '0e182dcb42825525fe1e7e6b84ffd5af50c3dc02a33f183087bff3963a653331'
RUNTIME_SHA = '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'
CONTRACTS_SHA = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
COMPILER_SHA = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'
OBSERVATIONS = (13, 127, 191)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name, expected):
    if sha(path) != expected:
        raise ValueError('Pinned source hash differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raw_metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
        tensor_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        finite_elements=int(np.isfinite(value).sum()), nonfinite_elements=int((~np.isfinite(value)).sum()))


def arrays(c, directory, row, pins):
    path = c.artifact(directory, row, pins)
    with np.load(path, allow_pickle=False) as archive:
        c.require(set(archive.files) == set(row['arrays']), 'Original artifact array names differ')
        result = {name: np.array(archive[name], copy=True, order='C') for name in archive.files}
    for name, value in result.items():
        actual, expected = raw_metadata(value), row['arrays'][name]
        c.require(all(actual[key] == expected[key] for key in ('shape','dtype','bytes','tensor_sha256')),
                  'Original artifact array binding differs: '+name)
        value.flags.writeable = False
    return result


def artifact(path, values):
    np.savez(path, **values)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                arrays={name:raw_metadata(value) for name,value in values.items()})


def validate_compile_index(c, path, expected_sha, manifest, sdk_dir, *, verify_artifacts=True):
    """Metadata-only mode is for host preflight; board main always verifies files."""
    pins = {}; path = c.pin(path, expected_sha, pins)
    index = json.loads(path.read_text())
    c.require(index['schema'] == 'v38_grouped_Update_compile_index'
        and index['status'] == 'staged_all_actual_shapes' and index['manifest_sha256'] == MANIFEST_SHA
        and index['compiler_sha256'] == COMPILER_SHA and index['contracts_sha256'] == CONTRACTS_SHA
        and index['candidate_accepted'] is False and index['host_model_inference_executed'] is False
        and index['host_calibration_executed'] is False, 'Compile index ancestry/scope differs')
    c.require(set(index['models']) == set(manifest['models']) and len(index['models']) == 14,
              'All fourteen actual shape models required')
    specs, compiler_rows = {}, {}
    for key, row in index['models'].items():
        selected = manifest['models'][key]
        c.require(c.model_key(selected['inputs'], selected['outputs']) == key, 'Index shape key differs')
        directory = Path(row['compile_dir'])
        c.require(directory.is_absolute() and row['model']['file'] == 'block.rknn'
                  and type(row['model']['bytes']) is int and row['model']['bytes'] > 0,
                  'Explicit absolute staged compilation directory/model required')
        if verify_artifacts:
            compiled, model, extra = c.validate_compilation(directory/'compile.json', row['compile_sha256'],
                manifest, MANIFEST_SHA, key, compiler_path=sdk_dir/'compile-v37-grouped-update.py',
                compiler_sha256=COMPILER_SHA, contracts_path=sdk_dir/'v37_grouped_contracts.py',
                contracts_sha256=CONTRACTS_SHA)
            c.require(compiled['output'] == row['model'], 'Index model bytes/SHA differ from compilation')
            pins.update(extra); compiler_rows[key] = compiled
        else:
            model = directory/row['model']['file']
        specs[key] = dict(inputs=selected['inputs'], outputs=selected['outputs'], model_bytes=row['model']['bytes'],
                         model_path=str(model), model_sha256=row['model']['sha256'], compile_sha256=row['compile_sha256'])
    c.require(sum(row['model_bytes']+1 for row in specs.values()) <= 256 << 20,
              'All actual model host buffers exceed requested 256MiB bound')
    return index, specs, compiler_rows, pins


def schedule(profile):
    return [(observation, number, 'profile' if profile else ('cold' if number == 0 else 'warm'))
            for observation in OBSERVATIONS for number in range(1 if profile else 3)]


def prepare_case(c, adapter, case, capture_dir, export_dir, pins):
    original = arrays(c, capture_dir, case['original_inputs_artifact'], pins)
    references = dict(original_Torch=arrays(c, capture_dir, case['original_gold_artifact'], pins),
        grouped_Torch=arrays(c, export_dir, case['grouped_Torch_artifact'], pins),
        grouped_ORT=arrays(c, export_dir, case['grouped_ORT_artifact'], pins))
    plan = adapter.build_degree_plan(original['ii'], original['jj'], original['kk'])
    c.require(plan.metadata() == case['degree_plan'] and plan.edges == case['edges'], 'Full original topology differs')
    c.require(len(case['groups']) == len(plan.batches), 'Full group count differs')
    for batch, group in zip(plan.batches, case['groups']):
        key = f'B{len(batch.source_slots)}_D{batch.degree}_M{batch.patches}'
        c.require(key == group['model_key'], 'Actual group order/key differs')
        feed = arrays(c, export_dir, group['inputs_artifact'], pins)
        for name in ('net','inp','corr'):
            c.require(np.array_equal(adapter.pack_batch(original[name], batch), feed[name]), 'Original complete group feed differs')
    recorded = case['snapshot_identity']
    identity = adapter.SnapshotIdentity(recorded['graph']['revision'], recorded['geometry']['revision'],
        recorded['features']['revision'], recorded['recurrent_revision'])
    return original, references, plan, identity


class RecordingCache:
    """Retain owned output references only; no hashes/statistics inside SDK path."""
    def __init__(self, cache):
        self.cache, self.calls = cache, []
    def execute(self, key, feed, reference=None, *, timeout_ms):
        values, record = self.cache.execute(key, feed, reference, timeout_ms=timeout_ms)
        self.calls.append((key, values, record))
        return values, record


def full_call(runtime, adapter, plan, original, cache, identity, *, timeout_ms=60000):
    recorder = RecordingCache(cache)
    begin = time.monotonic()
    try:
        outputs, row = runtime.execute_full_graph(adapter, plan, original, recorder,
            snapshot_identity=identity, current_identity=lambda: identity, reference=None, timeout_ms=timeout_ms)
    except BaseException as error:
        # Caller saves raw invalid group output and receipt before stopping; no
        # complete result can be returned after a late group failure.
        error.completed_group_records = recorder.calls
        error.full_wall_seconds = time.monotonic()-begin
        raise
    row['full_wall_seconds'] = time.monotonic()-begin
    return outputs, row, recorder.calls


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('manifest','compile-index','capture-dir','sdk-dir','adapter','abi-helper','header','log-helper','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--compile-index-sha256',required=True)
    parser.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'))
    parser.add_argument('--collect-perf-detail',action='store_true')
    parser.add_argument('--diagnostic-allow-fp32-rejected',action='store_true')
    parser.add_argument('--max-contexts',type=int,choices=(4,14),default=14)
    parser.add_argument('--timeout-ms',type=int,default=60000)
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    if not args.diagnostic_allow_fp32_rejected:
        raise ValueError('Explicit rejected-parent diagnostic flag required before side effects')
    c = load(args.sdk_dir/'v37_grouped_contracts.py','_v38_frozen_contracts',CONTRACTS_SHA)
    abi = load(args.abi_helper,'_v38_frozen_ABI',c.ABI_SHA256)
    abi.require_board()  # Before output writes, native library or SDK context.
    c.require(type(args.timeout_ms) is int and 1 <= args.timeout_ms < 2**31, 'Invalid SDK timeout')
    out = args.output_dir.resolve()
    c.require(out.is_relative_to('/mnt/vggt-data/vggt-slam2/experiments') and not out.exists(), 'New board output directory required')
    c.require(all(not out.is_relative_to(path.resolve()) for path in (args.capture_dir,args.sdk_dir,args.manifest.parent)),
              'Output must be outside frozen capture/export/source directories')
    out.mkdir(parents=True)
    affinity = os.sched_getaffinity(0)
    report = dict(schema='v38_three_complete_M48_grouped_Update_RKNN_diagnostic',status='failed',
        script_sha256=sha(__file__),candidate_accepted=False,VO_state_committed=False,production_changed=False,
        original_VGGT_default_unchanged=True,diagnostic_allow_fp32_rejected=True,collect_perf_detail=args.collect_perf_detail,
        actual_Torch_OR_ORT_calls=0,host_inference_executed=False,calls=[],cleanup_errors=[],
        scope='Three fixed captured graphs, no recurrent feedback/BA/trajectory/20Hz acceptance.',
        cache_bounds=dict(max_contexts=args.max_contexts,max_model_bytes=256<<20,max_dma_bytes=512<<20),
        input_immutability='Owned readonly captured arrays; frozen adapter makes a readonly full input snapshot per call. Fixed capture-derived revisions, no mutable VO owner.')
    pins, cache, initialization_events, error = {}, None, [], None
    try:
        c.require({4,5,6,7} <= affinity, 'Requested CPU affinity unavailable')
        os.sched_setaffinity(0,{4,5,6,7})
        c.pin(__file__,report['script_sha256'],pins)
        manifest, extra = c.validate_export_manifest(args.manifest,MANIFEST_SHA,
            allow_fp32_rejected_diagnostic=True,verify_source_files=True);pins.update(extra)
        index, specs, compiled, extra = validate_compile_index(c,args.compile_index,args.compile_index_sha256,
            manifest,args.sdk_dir,verify_artifacts=True);pins.update(extra)
        runtime = load(args.sdk_dir/'v37_grouped_rknn.py','_v38_frozen_runtime',RUNTIME_SHA)
        c.pin(args.sdk_dir/'v37_grouped_rknn.py',RUNTIME_SHA,pins)
        adapter = load(args.adapter,'_v38_frozen_adapter',c.ADAPTER_SHA256)
        c.pin(args.adapter,c.ADAPTER_SHA256,pins)
        c.pin(args.adapter.with_name('source_frame_plan.py'),adapter.PLAN_SOURCE_SHA256,pins)
        c.pin(args.abi_helper,c.ABI_SHA256,pins);c.pin(args.header,c.HEADER_SHA256,pins)
        c.pin(args.library,c.LIBRARY_SHA256,pins);c.pin(args.log_helper,c.LOG_HELPER_SHA256,pins)
        c.pin(args.log_helper.with_name('v33_rknn_source_frame.py'),c.LOG_PARENT_SHA256,pins)
        report.update(export_manifest_sha256=MANIFEST_SHA,compile_index_sha256=args.compile_index_sha256,
            runtime_sha256=RUNTIME_SHA,inherited_CPU_ORT_FP32_gate_passed=manifest['strict_FP32_gate_passed'],
            compiled_models=index['models'],total_compiled_model_bytes=sum(v['model_bytes'] for v in specs.values()),
            initialization_events=initialization_events)
        def factory(spec):
            started = time.monotonic()
            event = dict(model_key=c.model_key(spec['inputs'],spec['outputs']),status='starting')
            initialization_events.append(event)
            try:
                session = runtime.GroupedSession.open_board(spec['model_path'],spec['model_sha256'],spec['inputs'],spec['outputs'],
                    abi_path=args.abi_helper,header_path=args.header,library_path=args.library,log_helper_path=args.log_helper,
                    export_strict_FP32_gate_passed=manifest['strict_FP32_gate_passed'],diagnostic_allow_fp32_rejected=True,
                    collect_perf_detail=args.collect_perf_detail,log_directory=out)
                event.update(status='initialized',memory=session.metrics()['memory'])
                return session
            except BaseException as exc:
                event.update(status='failed',error=repr(exc),failure_record=getattr(exc,'record',None))
                raise
            finally:
                event['initialization_seconds'] = time.monotonic()-started
        cache = runtime.BoundedContextCache(specs,factory,max_contexts=args.max_contexts,
            max_model_bytes=256<<20,max_dma_bytes=512<<20)
        for case in manifest['cases']:
            prepare_start = time.monotonic()
            original, references, plan, identity = prepare_case(c,adapter,case,args.capture_dir,args.manifest.parent,pins)
            preparation_seconds = time.monotonic()-prepare_start
            for _, number, label in [row for row in schedule(args.collect_perf_detail) if row[0] == case['observation_id']]:
                row = dict(observation_id=case['observation_id'],repeat_index=number,label=label,status='starting',
                    candidate_accepted=False,VO_state_committed=False,edges=plan.edges,degree_plan=plan.metadata(),
                    snapshot_identity=case['snapshot_identity'],preparation_seconds=preparation_seconds if number==0 else 0)
                report['calls'].append(row)
                before = cache.metrics(); init_offset = len(initialization_events)
                try:
                    outputs, call_record, recorded = full_call(runtime,adapter,plan,original,cache,identity,timeout_ms=args.timeout_ms)
                except BaseException as exc:
                    row.update(status='execution_or_domain_failed',error=repr(exc),failure_record=getattr(exc,'record',None),
                        full_wall_seconds=getattr(exc,'full_wall_seconds',None))
                    row['groups'] = []
                    for group_number,(key,values,group_record) in enumerate(getattr(exc,'completed_group_records',[])):
                        entry = dict(model_key=key,record=group_record)
                        if values:
                            entry['raw_failure_artifact'] = artifact(out/f'obs-{case["observation_id"]}-call-{number}-failed-group-{group_number}.npz',values)
                        row['groups'].append(entry)
                    raise
                row.update(call_record,status='completed_diagnostic_not_adopted')
                statistics_begin = time.monotonic()
                row['groups'] = [dict(model_key=key,record=record,raw_outputs={name:raw_metadata(value) for name,value in values.items()})
                                 for key,values,record in recorded]
                row['versus_saved_references'] = {kind:{name:runtime.compare(value,reference[name]) for name,value in outputs.items()}
                                                 for kind,reference in references.items()}
                row['strict_FP32_gate_passed'] = all(v['gate_passed'] for v in row['versus_saved_references']['original_Torch'].values())
                row['outputs_finite'] = all(bool(np.isfinite(v).all()) for v in outputs.values())
                row['weight_domain_valid'] = bool(((outputs['weight'] >= 0)&(outputs['weight'] <= 1)).all())
                row['cache_before'],row['cache_after'] = before,cache.metrics()
                row['new_initializations'] = initialization_events[init_offset:]
                row['summed_nested_SDK_forward_seconds'] = sum(record['timing']['forward_seconds'] for _,_,record in recorded)
                row['summed_nested_cache_call_seconds'] = sum(record['cache_call_seconds'] for _,_,record in recorded)
                row['outside_forward_statistics_seconds'] = time.monotonic()-statistics_begin
                saving_begin = time.monotonic()
                row['full_order_artifact'] = artifact(out/f'obs-{case["observation_id"]}-call-{number}-full.npz',outputs)
                row['outside_forward_save_seconds'] = time.monotonic()-saving_begin
                del outputs,recorded
            del original,references,plan,identity
        report['status'] = 'completed_three_full_graphs_diagnostic_not_VO'
        report['actual_original_FP32_gate_passed'] = all(row['strict_FP32_gate_passed'] for row in report['calls'])
        report['timing_summary'] = {str(observation):dict(
            full_wall_seconds=[row['full_wall_seconds'] for row in report['calls'] if row['observation_id']==observation],
            warm_mean_full_wall_seconds=(sum(row['full_wall_seconds'] for row in report['calls']
                if row['observation_id']==observation and row['label']=='warm')/2) if not args.collect_perf_detail else None)
            for observation in OBSERVATIONS}
        report['timing_scope'] = 'Full wall includes immutable input snapshot/validation, source-group packing, context cold initialization/eviction when needed, native log handling, SDK I/O/release, domain checks and original-E scatter. Nested group/cache/init times are not additive. Hashes/reference statistics/output saves are outside full wall; profile is a separate diagnostic.'
    except BaseException as exc:
        error = exc;report.update(error=repr(exc),traceback=traceback.format_exc())
        if hasattr(exc,'record'):report['failure_record'] = exc.record
    finally:
        close_begin = time.monotonic()
        if cache is not None:
            cache.close();report['cache'] = cache.metrics()
            report['cleanup_errors'].extend(report['cache']['cleanup_errors'])
        report['cache_close_seconds'] = time.monotonic()-close_begin
        try:
            verification_begin = time.monotonic()
            report['source_pins'] = pins;report['source_hashes_after'] = c.recheck_pins(pins)
            report['sources_and_inputs_unchanged'] = True
            report['outside_forward_final_verification_seconds'] = time.monotonic()-verification_begin
        except Exception as exc:
            report['sources_and_inputs_unchanged'] = False;report['cleanup_errors'].append(repr(exc))
        try:
            os.sched_setaffinity(0,affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            c.require(report['affinity_restored'],'CPU affinity restoration failed')
        except Exception as exc:report['cleanup_errors'].append(repr(exc))
        if error or report['cleanup_errors']:report['status'] = 'failed'
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (out/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:report.get(key) for key in ('status','error','actual_original_FP32_gate_passed','timing_summary','cleanup_errors')}),flush=True)
    return int(report['status']=='failed')


if __name__ == '__main__':
    raise SystemExit(main())
