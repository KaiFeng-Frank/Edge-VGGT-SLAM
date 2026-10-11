#!/usr/bin/env python3
"""Three complete saved Update states, identical metadata callback, native-pack AB.

Neural execution uses board RKNN only. No Torch/ORT forward, recurrent feedback,
BA, trajectory or candidate adoption. A retains FP32 NHWC SDK conversion; B uses
explicit native FP16 NC1HWC2. Frozen v42 helpers retain input identity, ABBA,
complete-edge output checks, nested clocks, saved arrays and close validation.
DPVO attribution: Copyright (c) 2022 Princeton University, MIT.
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

PARENT_HARNESS_SHA256 = '434e7d8bbd13c14833368cbc7c42da5266c48353936b7a0c741dfe8a0b81fea2'
FACTORY_SHA256 = 'e15ff495ac626e7a1a7271a918b39854b4cc45989b77e334db56d8234ea5f300'
CONFIG_SHA256 = '7cb492b839a8529698687e6224868cf3a3d4de5e0546f945747963282290b622'
CALLBACK_SHA256 = '4f4a5457dbdec5aebe1ddd6204a37c9073896ac99ed90abd3fbfe3d1f78db566'
TRANSPORT_SHA256 = '222c254361a50d9180725ea780116a2302a29be8396136b15db91e7d11e912ca'
SINGLE_AB_SHA256 = '28185202278d0446b62525721338345ce279a88773969d45e9efb8f3f9fbcba2'
INDEX_SHA256 = 'eeb9241c9eab52039aa64fcf6546ce4624d1388756a9e919f42c22de4083a85c'
MODEL_KEYS = frozenset([f'B{b}_D{d}_M16' for b in (1, 2) for d in range(4, 8)] + ['B1_D3_M16'])


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


def load_parent(path, expected):
    require(expected == PARENT_HARNESS_SHA256 and sha(path) == expected, 'Frozen AB harness SHA differs')
    name = '_v43_full_AB_parent_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def validate_single_prerequisite(row, library_sha256):
    require(row['schema'] == 'v43_native_single_B2D6_saved_state_ABBA'
            and row['status'] == 'completed_single_B2D6_AB_bitwise_equal_diagnostic_not_adopted'
            and row['model_key'] == 'B2_D6_M16' and row['candidate_accepted'] is False
            and row['A_B_all_outputs_bitwise_equal'] is True
            and row['SDK_native_transport_equivalence_for_this_feed_only'] is True
            and row['cleanup_errors'] == [] and row['sources_unchanged'] is True,
            'Actual single-group SDK bitwise prerequisite required')
    require([v['method'] for v in row['calls']] == list(('A', 'B', 'B', 'A')*2)
            and all(v['bitwise_equal_first_A'] is True for v in row['calls']), 'Complete actual single-group ABBA required')
    require(any(Path(p).name == 'libv43_native_fp16_pack.so' and digest == library_sha256
                for p, digest in row['source_pins'].items()), 'Single-group packing library differs')
    for method in ('A', 'B'):
        m = row['closed_contexts'][method]
        require(m['closed'] is True and m['context_destroyed'] is True and m['cleanup_errors'] == []
                and m['sdk_run_api_calls'] == m['completed_calls'] == m['SDK_outputs_release_attempts'] == m['SDK_outputs_releases'] == 4
                and m['current_DMA_bytes'] == m['current_model_host_bytes'] == 0,
                'Single-group context/release prerequisite failed')


def replace_empty_cache(owner, runtime, config, pins, failure_dir):
    """Replace no live contexts; all original validated specs and callback remain."""
    old = owner.cache
    m = old.metrics()
    require(set(old.specs) == MODEL_KEYS and owner.route == 'full' and m['created'] == 0
            and m['resident_contexts'] == m['current_DMA_bytes'] == m['current_model_host_bytes'] == 0
            and m['cleanup_errors'] == [] and not m['closed'], 'Only an unused validated nine-shape cache may be replaced')
    specs = copy.deepcopy(old.specs)
    old.close()
    require(old.metrics()['closed'] and old.metrics()['resident_contexts'] == 0, 'Empty original cache close failed')
    sources = dict(owner.source_pins); sources.update(runtime.source_pins); sources.update(pins)
    def factory(spec):
        return runtime.GroupedSession.open_board(spec['model_path'], spec['model_sha256'],
            spec['inputs'], spec['outputs'],
            **{k: config[k] for k in ('abi_path', 'header_path', 'library_path', 'log_helper_path')},
            export_strict_FP32_gate_passed=spec['export_strict_FP32_gate_passed'],
            diagnostic_allow_fp32_rejected=True, diagnostic_allow_native_fp16=True,
            pins=sources, collect_perf_detail=False, log_directory=failure_dir)
    new = runtime.BoundedContextCache(specs, factory, max_contexts=config['max_contexts'],
        max_model_bytes=config['max_model_bytes'], max_dma_bytes=config['max_dma_bytes'])
    owner.runtime, owner.cache, owner.source_pins = runtime, new, sources
    owner.catalogue_provenance = dict(owner.catalogue_provenance,
        native_input_transport_added=True, input_transport=runtime.input_transport,
        transport_equivalence_validated=False, candidate_accepted=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('parent-harness', 'factory', 'config', 'transport', 'packer', 'packing-library',
                 'query', 'build-receipt', 'conversion-receipt', 'single-ab-receipt'):
        p.add_argument('--'+name, type=Path, required=True)
        p.add_argument('--'+name+'-sha256', required=True)
    p.add_argument('--inputs-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--diagnostic-allow-fp32-rejected', action='store_true')
    p.add_argument('--diagnostic-allow-native-fp16', action='store_true')
    a = p.parse_args(argv)
    require(a.diagnostic_allow_fp32_rejected and a.diagnostic_allow_native_fp16, 'Both explicit diagnostic flags required')
    H = load_parent(a.parent_harness, a.parent_harness_sha256)
    H.board_guard()  # Before SDK, packing library or any output-directory creation.
    out = a.output_dir.resolve()
    require(out.is_relative_to(Path('/mnt/vggt-data/vggt-slam2/experiments')) and not out.exists(), 'New board experiments output required')
    out.mkdir(parents=True, exist_ok=False)
    started, prior = time.monotonic(), os.sched_getaffinity(0)
    report = dict(schema='v43_native_three_complete_saved_state_ABBA_diagnostic', status='failed_execution',
        script_sha256=sha(__file__), parent_harness_sha256=PARENT_HARNESS_SHA256,
        source_run_sha256=H.RUN_SHA256, compile_index_sha256=INDEX_SHA256,
        factory_sha256=FACTORY_SHA256, callback_sha256=dict(A=CALLBACK_SHA256, B=CALLBACK_SHA256),
        same_callback_for_both_methods=True, only_input_transport_changed=True,
        candidate_accepted=False, historical_SDK_strict_FP32_rejected=True,
        strict_FP32_equivalence_validated=False, CPU_teacher_used=False,
        recurrent_feedback=False, BA_executed=False, VO_state_committed=False,
        realtime_validated=False, collect_perf_detail=False, cleanup_errors=[], calls=[],
        schedule=H.schedule(),
        cold_definition='Exclude first call per method per state; not every excluded call initializes a context.',
        methods=dict(A='original_FP32_NHWC_pass_through_0', B='native_FP16_NC1HWC2_C2_8_pass_through_1'),
        limits=['Three fixed saved NPU-feedback states; no original CPU-gold equivalence or new observation/trajectory.',
                'Callback and all nine compiled model bytes are identical; only B input transport changes.',
                'Actual callback/fullgraph/SDK clocks are nested and must not be added.',
                'Warm n=3, ABBA and observed health do not establish controlled thermal conditions or statistical significance.',
                'Historical FP32/quality rejection and default VGGT remain unchanged.'])
    pins, owners, records = {}, {}, dict(A=[], B=[])
    def save(name, values):
        path = out/name
        with path.open('xb') as stream:
            np.savez(stream, **values)
        return dict(file=name, bytes=path.stat().st_size, sha256=H.sha(path),
                    arrays={n:H.metadata(v) for n, v in values.items()})
    def no_motion(*_args):
        raise RuntimeError('No startup motion probe or CPU fallback in saved full-graph AB')
    try:
        require(a.factory_sha256 == FACTORY_SHA256 and a.config_sha256 == CONFIG_SHA256
                and a.transport_sha256 == TRANSPORT_SHA256 and a.single_ab_receipt_sha256 == SINGLE_AB_SHA256,
                'Exact frozen factory/config/transport and actual single-group prerequisite required')
        H.pin(__file__, report['script_sha256'], pins)
        for name in ('parent_harness', 'factory', 'config', 'transport', 'packer', 'packing_library',
                     'query', 'build_receipt', 'conversion_receipt', 'single_ab_receipt'):
            H.pin(getattr(a, name), getattr(a, name+'_sha256'), pins)
        single = json.loads(a.single_ab_receipt.read_text())
        validate_single_prerequisite(single, a.packing_library_sha256)
        report['single_AB_prerequisite_sha256'] = a.single_ab_receipt_sha256
        build = json.loads(a.build_receipt.read_text())
        conversion = json.loads(a.conversion_receipt.read_text())
        require(build['status'] == 'compiled_ABI_loaded_conversion_not_yet_verified'
                and build['output']['sha256'] == a.packing_library_sha256 and build['sources_unchanged'] is True
                and conversion['status'] == 'passed_conversion_reference_and_ownership_only'
                and conversion['library_sha256'] == a.packing_library_sha256,
                'Packing build/conversion prerequisite differs')
        config = json.loads(a.config.read_text())
        require(config['compile_index_sha256'] == INDEX_SHA256 and config['max_contexts'] == 9
                and config['callback_sha256'] == CALLBACK_SHA256, 'Same frozen nine-shape metadata callback configuration required')
        factory = H.load_module(a.factory, FACTORY_SHA256, 'v43_factory', pins)
        transport = H.load_module(a.transport, TRANSPORT_SHA256, 'native_transport', pins)
        _, states = H.load_inputs(a.inputs_dir, pins)
        report['input_captures'] = [{k:v for k,v in s.items() if k not in ('values', 'before')} for s in states]
        require({4, 5, 6, 7} <= prior, 'Four A76 CPUs required')
        os.sched_setaffinity(0, {4, 5, 6, 7}); report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        for method in ('A', 'B'):
            directory = out/('callback-'+method); directory.mkdir()
            owner = factory.open_board_online_from_index(**config, failure_dir=directory,
                motion_probe=no_motion, diagnostic_allow_fp32_rejected=True, on_record=records[method].append)
            owners[method] = owner
            require(set(owner.cache.specs) == MODEL_KEYS and owner.route == 'full', 'Validated nine-model full callback required')
            if method == 'B':
                runtime = transport.load_runtime(config['parent_runtime'], config['contracts'], config['contracts_sha256'],
                    config['parent_contracts'], runtime_bridge=config['runtime_bridge'],
                    packer_path=a.packer, packer_sha256=a.packer_sha256,
                    packing_library_path=a.packing_library, packing_library_sha256=a.packing_library_sha256,
                    packing_source_pins={str(a.build_receipt): a.build_receipt_sha256},
                    native_query_receipt=a.query, native_query_sha256=a.query_sha256,
                    diagnostic_allow_native_fp16=True)
                replace_empty_cache(owner, runtime, config, pins, directory)
            for path, digest in owner.source_pins.items(): H.pin(path, digest, pins)
        require(owners['A'].cache is not owners['B'].cache, 'Independent context caches required')
        H.run_calls(owners, records, states, save, report, H.platform_sample)
        # Parent helpers strictly require the original eight shapes used by
        # these three saved states; the validated nine-model caches retain D3.
        for row in report['warm_statistics']:
            for method in ('A', 'B'):
                selected = [v for v in report['calls'] if v['observation_id'] == row['observation_id']
                            and v['method'] == method and not v['excluded_first_state_method_call']]
                row['methods'][method]['sum_SDK_interval_means'] = {field:float(np.mean([
                    v['timing']['sum_SDK_intervals'][field] for v in selected]))
                    for field in selected[0]['timing']['sum_SDK_intervals']}
        report.update(status='completed_AB_bitwise_equal_diagnostic_not_adopted',
                      native_transport_bitwise_for_three_saved_states_only=True)
    except BaseException as error:
        report.update(status='failed_execution', error=repr(error))
        if hasattr(error, 'record'): report['callback_failure_record'] = copy.deepcopy(error.record)
    finally:
        report['final_callbacks'], report['lifecycle'] = {}, {}
        for method, owner in owners.items():
            try: owner.close()
            except BaseException as error: report['cleanup_errors'].append(method+': '+repr(error))
            try:
                state = owner.metrics(); report['final_callbacks'][method] = state
                expected = sum(v['timing']['SDK_calls'] for v in report['calls'] if v['method'] == method) if report['status'].startswith('completed_') else None
                report['lifecycle'][method] = H.closed_lifecycle(state, expected)
            except BaseException as error: report['cleanup_errors'].append(method+': '+repr(error))
        try:
            after = {path:H.sha(path) for path in pins}
            report.update(source_pins=pins, source_hashes_after=after, sources_and_inputs_unchanged=after == pins)
            require(after == pins, 'Frozen source/input/model changed')
        except BaseException as error: report['cleanup_errors'].append(repr(error))
        try:
            os.sched_setaffinity(0, prior); report['affinity_restored'] = os.sched_getaffinity(0) == prior
            require(report['affinity_restored'], 'Affinity restore failed')
        except BaseException as error: report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']: report['status'] = 'failed_execution'
        report['actual_experiment_seconds'] = time.monotonic()-started
        with (out/'run.json').open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status', 'A_B_all_saved_outputs_bitwise_equal', 'error', 'cleanup_errors')}), flush=True)
    return int(not report['status'].startswith('completed_'))


if __name__ == '__main__':
    raise SystemExit(main())
