#!/usr/bin/env python3
"""Board-only B8/D8/M48 complete initial Update diagnostic; no VO commit.

Reads the already captured original full state and frozen grouped references.
It does not rerun Torch/ORT, feed GT, alter weights, or change numerical gates.
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

CONTRACTS_SHA = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
COMPILER_SHA = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'
MANIFEST_SHA = '0e182dcb42825525fe1e7e6b84ffd5af50c3dc02a33f183087bff3963a653331'
KEY = 'B8_D8_M48'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name, expected):
    if sha(path) != expected:
        raise ValueError('Source hash differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arrays(c, directory, row, pins):
    path = c.artifact(directory, row, pins)
    with np.load(path, allow_pickle=False) as archive:
        c.require(set(archive.files) == set(row['arrays']), 'Artifact array names differ')
        result = {name: np.array(archive[name], copy=True, order='C') for name in archive.files}
    for name, value in result.items():
        expected = row['arrays'][name]
        c.require(list(value.shape) == expected['shape'] and str(value.dtype) == expected['dtype']
                  and value.nbytes == expected['bytes']
                  and hashlib.sha256(value.tobytes()).hexdigest() == expected['tensor_sha256'],
                  'Artifact array contract differs: ' + name)
        value.flags.writeable = False
    return result


def artifact(path, values):
    np.savez(path, **values)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), arrays={
        name: dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                   tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name, value in values.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('manifest', 'capture-dir', 'compile-dir', 'sdk-dir', 'adapter', 'abi-helper',
                 'header', 'log-helper', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--compile-sha256', required=True)
    p.add_argument('--runtime-sha256', required=True)
    p.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--collect-perf-detail', action='store_true')
    p.add_argument('--diagnostic-allow-fp32-rejected', action='store_true')
    args = p.parse_args(argv)
    if not args.diagnostic_allow_fp32_rejected:
        raise ValueError('Explicit rejected-parent diagnostic flag required')
    c = load(args.sdk_dir/'v37_grouped_contracts.py', '_v37_driver_contract', CONTRACTS_SHA)
    abi = load(args.abi_helper, '_v37_driver_ABI', c.ABI_SHA256)
    abi.require_board()
    out = args.output_dir.resolve()
    c.require(out.is_relative_to('/mnt/vggt-data/vggt-slam2/experiments') and not out.exists(),
              'New board experiment directory required')
    out.mkdir(parents=True)
    original_affinity = os.sched_getaffinity(0)
    report = dict(schema='v37_complete_initial_M48_grouped_Update_RKNN_diagnostic', status='failed',
        script_sha256=sha(__file__), candidate_accepted=False, VO_state_committed=False,
        production_changed=False, original_VGGT_default_unchanged=True, model_key=KEY,
        diagnostic_allow_fp32_rejected=True, host_inference_executed=False,
        actual_Torch_OR_ORT_calls=0, collect_perf_detail=args.collect_perf_detail,
        calls=[], cleanup_errors=[], scope='One complete captured initial graph: obs13/E3072/B8D8M48. No evolving graph, recurrent feedback, BA, trajectory or realtime acceptance.')
    session, error, pins = None, None, {}
    try:
        os.sched_setaffinity(0, {4, 5, 6, 7})
        c.pin(__file__, report['script_sha256'], pins)
        manifest, manifest_pins = c.validate_export_manifest(args.manifest, MANIFEST_SHA,
            allow_fp32_rejected_diagnostic=True, verify_source_files=True)
        pins.update(manifest_pins)
        compile_report, model, compile_pins = c.validate_compilation(args.compile_dir/'compile.json',
            args.compile_sha256, manifest, MANIFEST_SHA, KEY,
            compiler_path=args.sdk_dir/'compile-v37-grouped-update.py', compiler_sha256=COMPILER_SHA,
            contracts_path=args.sdk_dir/'v37_grouped_contracts.py', contracts_sha256=CONTRACTS_SHA)
        pins.update(compile_pins)
        c.pin(args.sdk_dir/'v37_grouped_rknn.py', args.runtime_sha256, pins)
        runtime = load(args.sdk_dir/'v37_grouped_rknn.py', '_v37_driver_runtime', args.runtime_sha256)
        c.pin(args.adapter, c.ADAPTER_SHA256, pins)
        adapter = load(args.adapter, '_v37_driver_adapter', c.ADAPTER_SHA256)
        c.pin(args.adapter.with_name('source_frame_plan.py'), adapter.PLAN_SOURCE_SHA256, pins)
        c.pin(args.abi_helper, c.ABI_SHA256, pins)
        c.pin(args.header, c.HEADER_SHA256, pins)
        c.pin(args.library, c.LIBRARY_SHA256, pins)
        case = manifest['cases'][0]
        c.require(case['observation_id'] == 13 and case['edges'] == 3072 and len(case['groups']) == 1
                  and case['groups'][0]['model_key'] == KEY, 'Complete initial graph contract differs')
        original = arrays(c, args.capture_dir, case['original_inputs_artifact'], pins)
        references = {'original_Torch': arrays(c, args.capture_dir, case['original_gold_artifact'], pins),
            'grouped_Torch': arrays(c, args.manifest.parent, case['grouped_Torch_artifact'], pins),
            'grouped_ORT': arrays(c, args.manifest.parent, case['grouped_ORT_artifact'], pins)}
        feed = arrays(c, args.manifest.parent, case['groups'][0]['inputs_artifact'], pins)
        plan = adapter.build_degree_plan(original['ii'], original['jj'], original['kk'])
        c.require(plan.metadata() == case['degree_plan'] and len(plan.batches) == 1, 'Full topology changed')
        batch = plan.batches[0]
        for name in c.INPUT_NAMES:
            c.require(np.array_equal(adapter.pack_batch(original[name], batch), feed[name]), 'Captured CF feed differs')
        packed_references = {kind: {sdk: adapter.pack_batch(values[canonical], batch)
            for sdk, canonical in zip(c.OUTPUT_NAMES, ('net', 'delta', 'weight'))}
            for kind, values in references.items()}
        selected = manifest['models'][KEY]
        report.update(export_manifest_sha256=MANIFEST_SHA, compile_sha256=args.compile_sha256,
            runtime_sha256=args.runtime_sha256, model_sha256=compile_report['output']['sha256'],
            inherited_CPU_ORT_FP32_gate_passed=manifest['strict_FP32_gate_passed'],
            case=dict(observation_id=13, edges=3072, degree_plan=plan.metadata(), snapshot_identity=case['snapshot_identity']))
        session = runtime.GroupedSession.open_board(model, compile_report['output']['sha256'],
            selected['inputs'], selected['outputs'], abi_path=args.abi_helper, header_path=args.header,
            library_path=args.library, log_helper_path=args.log_helper,
            export_strict_FP32_gate_passed=manifest['strict_FP32_gate_passed'],
            diagnostic_allow_fp32_rejected=True, collect_perf_detail=args.collect_perf_detail, log_directory=out)
        for index in range(1 if args.collect_perf_detail else 4):
            values, row = session.execute(feed, packed_references['original_Torch'])
            row.update(call_index=index, label='profile' if args.collect_perf_detail else ('cold' if index == 0 else 'warm'))
            if values:
                row['CF_artifact'] = artifact(out/f'call-{index:02d}-CF.npz', values)
            report['calls'].append(row)
            c.require(row['execution_status'] == 'completed', 'Actual SDK execution failed')
            c.require(row['outputs_finite'] and row['weight_domain_valid'] and not row['all_outputs_zero'],
                      'Actual numerical domain failed')
            row['versus_saved_references'] = {kind: {name: runtime.compare(value, reference[name])
                for name, value in values.items()} for kind, reference in packed_references.items()}
            begin = time.monotonic()
            restored = {}
            for sdk, canonical in zip(c.OUTPUT_NAMES, ('net', 'delta', 'weight')):
                target = np.empty_like(references['original_Torch'][canonical])
                target[0, batch.canonical_edge_indices, :] = values[sdk].transpose(0, 2, 3, 1)
                restored[canonical] = target
            row['restore_edge_order_seconds'] = time.monotonic()-begin
            row['full_order_artifact'] = artifact(out/f'call-{index:02d}-full.npz', restored)
            row['full_versus_original'] = {name: runtime.compare(value, references['original_Torch'][name])
                for name, value in restored.items()}
        report['status'] = 'completed_initial_group_RKNN_diagnostic_not_VO'
        report['actual_original_FP32_gate_passed'] = all(row['strict_FP32_gate_passed'] for row in report['calls'])
        if not args.collect_perf_detail:
            warm = [row['timing']['forward_seconds'] for row in report['calls'][1:]]
            report['timing_summary'] = dict(warm_forward_seconds=warm, warm_mean_forward_seconds=sum(warm)/len(warm),
                scope='Complete B8 SDK call incl finite input scan/pack/set/run/get/owned CF copy/release. Excludes full topology planning, file loading/saving, original-order scatter, reference checks and init. Not full VO latency.')
    except BaseException as exc:
        error = exc
        report.update(error=repr(exc), traceback=traceback.format_exc())
        if hasattr(exc, 'record'):
            report['failure_record'] = exc.record
    finally:
        if session is not None:
            session.close()
            report['session'] = session.metrics()
            report['cleanup_errors'].extend(report['session']['cleanup_errors'])
        try:
            report['source_pins'] = pins
            report['source_hashes_after'] = c.recheck_pins(pins)
            report['sources_and_inputs_unchanged'] = True
        except Exception as exc:
            report['sources_and_inputs_unchanged'] = False
            report['cleanup_errors'].append(repr(exc))
        try:
            os.sched_setaffinity(0, original_affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
        except Exception as exc:
            report['cleanup_errors'].append(repr(exc))
        if error or report['cleanup_errors']:
            report['status'] = 'failed'
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (out/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'error', 'actual_original_FP32_gate_passed', 'timing_summary', 'cleanup_errors')}), flush=True)
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
