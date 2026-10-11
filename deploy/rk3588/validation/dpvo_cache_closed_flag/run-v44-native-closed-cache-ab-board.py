#!/usr/bin/env python3
"""Same native input transport, original versus closed-query cache, board ABBA.

Only B's frozen-cache live-hit closed query changes. Both methods retain the
same metadata callback, nine models, native pack/SDK code and complete original
edge order. Three immutable saved states, ABBA twice each; no recurrent
feedback, CPU teacher, Torch/ORT/BA forward or adoption. DPVO Princeton (2022),
MIT; see adjacent LICENSE. Real SDK execution is restricted to RK3588 ARM.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

sys.dont_write_bytecode = True
NATIVE_HARNESS_SHA256 = '4c056f07828db7b5fa74914148e5a196840af2275988eb4f4104a459488fa9c2'
CLOSED_FLAG_HELPER_SHA256 = '014685c3b7736070bf3b1a1e3a333a928468e013cf454e4acac4ee53da1354b2'
PARENT_RUNTIME_SHA256 = '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'
SCHEMA = 'v44_native_cache_closed_flag_three_saved_state_ABBA_diagnostic'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_native_harness(path, expected):
    require(expected == NATIVE_HARNESS_SHA256 and sha(path) == expected,
            'Frozen native full-state AB harness SHA differs')
    name = '_v44_native_AB_parent_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def bind_pair(owners, native, closed_helper, G, H, config, pins, directories):
    """Bind only unused caches, with identical source payloads in both methods."""
    require(set(owners) == set(directories) == {'A', 'B'}, 'Two independent owners required')
    require(owners['A'] is not owners['B'] and owners['A'].cache is not owners['B'].cache,
            'Owners and original empty caches must be independent')
    parent = H.pin(config['parent_runtime'], PARENT_RUNTIME_SHA256, pins)
    candidate = closed_helper.bind_runtime(native, parent_runtime_path=parent,
        diagnostic_allow_closed_flag=True)
    require(candidate.GroupedSession is native.GroupedSession
            and candidate.execute_full_graph is native.execute_full_graph
            and candidate.pack_inputs is native.pack_inputs
            and candidate.BoundedContextCache is not native.BoundedContextCache,
            'Only the cache class may differ between native methods')
    runtimes = dict(A=native, B=candidate)
    # Gather every dependency before creating either cache factory. Session
    # reporting then copies equal-size/equal-content pin dictionaries in A/B.
    for owner in owners.values():
        for path, digest in owner.source_pins.items():
            H.pin(path, digest, pins)
    for runtime in runtimes.values():
        for path, digest in runtime.source_pins.items():
            H.pin(path, digest, pins)
    for method in ('A', 'B'):
        G.replace_empty_cache(owners[method], runtimes[method], config, pins, directories[method])
        owners[method].catalogue_provenance = dict(owners[method].catalogue_provenance,
            cache_closed_flag_only=method == 'B', candidate_accepted=False)
    require(owners['A'].source_pins == owners['B'].source_pins,
            'Session source-pin reporting payload must be identical across methods')
    require(owners['A'].runtime.GroupedSession is owners['B'].runtime.GroupedSession,
            'Same native session implementation required')
    return copy.deepcopy(candidate.closed_flag_source_proof)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('native-harness', 'parent-harness', 'closed-flag-helper', 'factory',
                 'config', 'transport', 'packer', 'packing-library', 'query',
                 'build-receipt', 'conversion-receipt', 'single-ab-receipt'):
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--inputs-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--diagnostic-allow-fp32-rejected', action='store_true')
    p.add_argument('--diagnostic-allow-native-fp16', action='store_true')
    p.add_argument('--diagnostic-allow-closed-flag', action='store_true')
    a = p.parse_args(argv)
    require(a.diagnostic_allow_fp32_rejected and a.diagnostic_allow_native_fp16
            and a.diagnostic_allow_closed_flag, 'All three explicit diagnostic flags required')
    G = load_native_harness(a.native_harness, a.native_harness_sha256)
    H = G.load_parent(a.parent_harness, a.parent_harness_sha256)
    H.board_guard()  # Before SDK, packing-library load or output-directory write.
    out = a.output_dir.resolve()
    require(out.is_relative_to(Path('/mnt/vggt-data/vggt-slam2/experiments')) and not out.exists(),
            'New isolated board experiments output required')
    out.mkdir(parents=True, exist_ok=False)
    started, prior = time.monotonic(), os.sched_getaffinity(0)
    report = dict(schema=SCHEMA, status='failed_execution', script_sha256=sha(__file__),
        native_harness_sha256=NATIVE_HARNESS_SHA256,
        parent_harness_sha256=G.PARENT_HARNESS_SHA256,
        closed_flag_helper_sha256=CLOSED_FLAG_HELPER_SHA256,
        parent_runtime_sha256=PARENT_RUNTIME_SHA256,
        source_run_sha256=H.RUN_SHA256, compile_index_sha256=G.INDEX_SHA256,
        factory_sha256=G.FACTORY_SHA256,
        callback_sha256=dict(A=G.CALLBACK_SHA256, B=G.CALLBACK_SHA256),
        same_callback_for_both_methods=True, same_native_input_transport=True,
        only_cache_live_hit_closed_query_changed=True, candidate_accepted=False,
        historical_SDK_strict_FP32_rejected=True, strict_FP32_equivalence_validated=False,
        CPU_teacher_used=False, recurrent_feedback=False, BA_executed=False,
        VO_state_committed=False, realtime_validated=False, collect_perf_detail=False,
        cleanup_errors=[], calls=[], schedule=H.schedule(),
        cold_definition='Exclude first call per method per state; not every excluded call initializes a context.',
        methods=dict(A='native_FP16_original_cache', B='native_FP16_closed_flag_cache'),
        limits=['Three fixed saved NPU-feedback states; no original CPU-gold equivalence or new observation/trajectory.',
                'Same native pack, SDK, metadata callback, nine models and source-pin reporting payload in A/B.',
                'Only the locked live-hit closed query differs; cold allocation, budgets, history and full metrics remain original.',
                'Actual callback/fullgraph/SDK clocks are nested and must not be added.',
                'Warm n=3, ABBA and observed health do not establish controlled thermal conditions or statistical significance.',
                'Historical FP32/quality rejection and default VGGT remain unchanged.'])
    pins, owners, records = {}, {}, dict(A=[], B=[])

    def save(name, values):
        path = out / name
        with path.open('xb') as stream:
            np.savez(stream, **values)
        return dict(file=name, bytes=path.stat().st_size, sha256=H.sha(path),
                    arrays={n: H.metadata(v) for n, v in values.items()})

    def no_motion(*_args):
        raise RuntimeError('No startup motion probe or CPU fallback in saved full-graph AB')

    try:
        require(a.factory_sha256 == G.FACTORY_SHA256 and a.config_sha256 == G.CONFIG_SHA256
                and a.transport_sha256 == G.TRANSPORT_SHA256
                and a.single_ab_receipt_sha256 == G.SINGLE_AB_SHA256
                and a.closed_flag_helper_sha256 == CLOSED_FLAG_HELPER_SHA256,
                'Exact frozen factory/config/native transport/cache helper/prerequisite required')
        H.pin(__file__, report['script_sha256'], pins)
        for name in ('native_harness', 'parent_harness', 'closed_flag_helper', 'factory',
                     'config', 'transport', 'packer', 'packing_library', 'query',
                     'build_receipt', 'conversion_receipt', 'single_ab_receipt'):
            H.pin(getattr(a, name), getattr(a, name + '_sha256'), pins)
        single = json.loads(a.single_ab_receipt.read_text())
        G.validate_single_prerequisite(single, a.packing_library_sha256)
        report['single_AB_prerequisite_sha256'] = a.single_ab_receipt_sha256
        build = json.loads(a.build_receipt.read_text())
        conversion = json.loads(a.conversion_receipt.read_text())
        require(build['status'] == 'compiled_ABI_loaded_conversion_not_yet_verified'
                and build['output']['sha256'] == a.packing_library_sha256
                and build['sources_unchanged'] is True
                and conversion['status'] == 'passed_conversion_reference_and_ownership_only'
                and conversion['library_sha256'] == a.packing_library_sha256,
                'Packing build/conversion prerequisite differs')
        config = json.loads(a.config.read_text())
        require(config['compile_index_sha256'] == G.INDEX_SHA256 and config['max_contexts'] == 9
                and config['callback_sha256'] == G.CALLBACK_SHA256,
                'Same frozen nine-shape metadata callback configuration required')
        factory = H.load_module(a.factory, G.FACTORY_SHA256, 'v44_factory', pins)
        transport = H.load_module(a.transport, G.TRANSPORT_SHA256, 'v44_native_transport', pins)
        closed_helper = H.load_module(a.closed_flag_helper, CLOSED_FLAG_HELPER_SHA256, 'v44_closed_flag', pins)
        _, states = H.load_inputs(a.inputs_dir, pins)
        report['input_captures'] = [{k: v for k, v in s.items() if k not in ('values', 'before')} for s in states]
        require({4, 5, 6, 7} <= prior, 'Four A76 CPUs required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        directories = {}
        for method in ('A', 'B'):
            directory = out / ('callback-' + method)
            directory.mkdir()
            directories[method] = directory
            owner = factory.open_board_online_from_index(**config, failure_dir=directory,
                motion_probe=no_motion, diagnostic_allow_fp32_rejected=True,
                on_record=records[method].append)
            owners[method] = owner
            require(set(owner.cache.specs) == G.MODEL_KEYS and owner.route == 'full',
                    'Validated nine-model full callback required')
        native = transport.load_runtime(config['parent_runtime'], config['contracts'],
            config['contracts_sha256'], config['parent_contracts'], runtime_bridge=config['runtime_bridge'],
            packer_path=a.packer, packer_sha256=a.packer_sha256,
            packing_library_path=a.packing_library, packing_library_sha256=a.packing_library_sha256,
            packing_source_pins={str(a.build_receipt): a.build_receipt_sha256},
            native_query_receipt=a.query, native_query_sha256=a.query_sha256,
            diagnostic_allow_native_fp16=True)
        report['closed_flag_source_proof'] = bind_pair(owners, native, closed_helper, G, H,
            config, pins, directories)
        report['same_session_source_pins_for_both_methods'] = owners['A'].source_pins == owners['B'].source_pins
        H.run_calls(owners, records, states, save, report, H.platform_sample)
        # All SDK/fullgraph clocks remain inherited; this supplements the same
        # means retained by the v43 harness, with no extra measured work.
        for row in report['warm_statistics']:
            for method in ('A', 'B'):
                selected = [v for v in report['calls'] if v['observation_id'] == row['observation_id']
                            and v['method'] == method and not v['excluded_first_state_method_call']]
                row['methods'][method]['sum_SDK_interval_means'] = {field: float(np.mean([
                    v['timing']['sum_SDK_intervals'][field] for v in selected]))
                    for field in selected[0]['timing']['sum_SDK_intervals']}
        report.update(status='completed_AB_bitwise_equal_diagnostic_not_adopted',
                      closed_flag_bitwise_for_three_saved_states_only=True)
    except BaseException as error:
        report.update(status='failed_execution', error=repr(error))
        if hasattr(error, 'record'):
            report['callback_failure_record'] = copy.deepcopy(error.record)
    finally:
        report['final_callbacks'], report['lifecycle'] = {}, {}
        for method, owner in owners.items():
            try:
                owner.close()
            except BaseException as error:
                report['cleanup_errors'].append(method + ': ' + repr(error))
            try:
                state = owner.metrics()
                report['final_callbacks'][method] = state
                expected = sum(v['timing']['SDK_calls'] for v in report['calls']
                    if v['method'] == method) if report['status'].startswith('completed_') else None
                report['lifecycle'][method] = H.closed_lifecycle(state, expected)
            except BaseException as error:
                report['cleanup_errors'].append(method + ': ' + repr(error))
        try:
            after = {path: H.sha(path) for path in pins}
            report.update(source_pins=pins, source_hashes_after=after,
                          sources_and_inputs_unchanged=after == pins)
            require(after == pins, 'Frozen source/input/model changed')
        except BaseException as error:
            report['cleanup_errors'].append(repr(error))
        try:
            os.sched_setaffinity(0, prior)
            report['affinity_restored'] = os.sched_getaffinity(0) == prior
            require(report['affinity_restored'], 'Affinity restore failed')
        except BaseException as error:
            report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']:
            report['status'] = 'failed_execution'
        report['actual_experiment_seconds'] = time.monotonic() - started
        with (out / 'run.json').open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report.get(k) for k in
        ('status', 'A_B_all_saved_outputs_bitwise_equal', 'error', 'cleanup_errors')}), flush=True)
    return int(not report['status'].startswith('completed_'))


if __name__ == '__main__':
    raise SystemExit(main())
