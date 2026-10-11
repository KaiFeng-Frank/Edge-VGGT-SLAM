#!/usr/bin/env python3
"""Board-only original pair hybrid compile; one frozen fc1/GELU diagnostic policy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
B = BASE / 'experiments'
CONTRACTS_SHA = '4d599517db9cbc438c1b095bb97c0dd74ea923cc6eaf613a8d32cd594b9a9101'
QUANT_SHA = '47237b1a1e1614822c75180efd15be3e5c708b610f587f698fe9b37fcf1bc649'
ENV = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2'}


def source(path, expected):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen original hybrid source SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v50_step2_' + path.stem.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    defaults = {'quant-contracts': B / 'vggt-v49-quant-source/v49_quant_contracts.py',
                'pair-contracts': B / 'vggt-v48-pair-source/v47_pair_contracts.py', 'pair-dir': B / 'vggt-v48-pair',
                'capture-contracts': B / 'vggt-v49-capture-source/v49_capture_contracts.py',
                'capture-dir': B / 'vggt-v49-capture-r1',
                'layout-contracts': B / 'vggt-v50-layout-source/v50_original_mlp_scope_contracts.py',
                'step1-dir': B / 'vggt-v50-hybrid-step1-r1',
                'precision-policy': B / 'vggt-v50-hybrid-policy-r2/policy-r2.json',
                'failed-r1-receipt': B / 'vggt-v50-hybrid-step2-r1/compile.json',
                'toolchain-prefix': B / 'vggt-v49-toolchain-venv-r1'}
    for name, path in defaults.items():
        p.add_argument('--' + name, type=Path, default=path)
    for name in ('output-dir', 'work-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(argv)
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        p.error('Actual hybrid compilation requires Linux RK3588S ARM64')
    C = source(args.quant_contracts, QUANT_SHA)
    compatible = C.require_board()
    output, work = args.output_dir.resolve(), args.work_dir.resolve()
    if (Path.cwd().resolve() != BASE or output == B or not output.is_relative_to(B)
            or work == Path('/dev/shm') or not work.is_relative_to('/dev/shm')
            or Path(sys.prefix).resolve() != args.toolchain_prefix.resolve()
            or not args.step1_dir.resolve().is_relative_to(B)
            or not args.precision_policy.resolve().is_relative_to(B)):
        p.error('Use board cwd, isolated interpreter and confined new output/work plus board-generated inputs')
    S = source(Path(__file__).with_name('v50_hybrid_step2_r2_contracts.py'), CONTRACTS_SHA)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': S.SCHEMA, 'status': 'failed', 'revision': S.REVISION,
              'r1_contracts_sha256': S.R1_CONTRACTS_SHA, 'failed_r1_receipt_sha256': S.FAILED_R1_SHA,
              'failed_r1_rejection_retained': False, 'selector_scope_proof_sha256': S.SELECTOR_PROOF_SHA,
              'external_IO_initializer_and_hidden_attention_precision_unproven': True, 'script_sha256': C.sha(__file__),
              'contracts_sha256': CONTRACTS_SHA, 'quant_contracts_sha256': QUANT_SHA,
              'pair_contracts_sha256': S.PAIR_CONTRACTS_SHA, 'layout_contracts_sha256': S.LAYOUT_CONTRACTS_SHA,
              'capture_contracts_sha256': S.CAPTURE_CONTRACTS_SHA, 'pair_export_sha256': S.PAIR_EXPORT_SHA,
              'pair_graph_sha256': S.PAIR_GRAPH_SHA, 'capture_manifest_sha256': S.CAPTURE_SHA,
              'capture_script_sha256': S.CAPTURE_SOURCE_SHA, 'hybrid_api_sha256': S.HYBRID_API_SHA,
              'step1_receipt_sha256': S.STEP1_SHA, 'step1_script_sha256': S.STEP1_SOURCE_SHA,
              'precision_policy_sha256': S.POLICY_SHA, 'SDK_API_source_sha256': S.SDK_API_SHA,
              'board': {'machine': platform.machine(), 'device_tree_compatible': compatible, 'python': sys.version},
              'toolchain_prefix': str(args.toolchain_prefix.resolve()), 'configuration': dict(S.CONFIGURATION),
              'input_transform': dict(S.INPUT_TRANSFORM), 'model_inference_on_host': False,
              'calibration_executed_on_board_only': True, 'new_calibration_dataset_supplied_to_step2': False,
              'runtime_inference_API_called': False, 'precision_policy_approved': False,
              'candidate_accepted': False, 'historical_FP32_rejections_retained': True,
              'fixed_FP32_gate': {'rtol': 1e-4, 'atol': 1e-5},
              'original_Torch_FP32_reference_is_sole_numerical_reference': True,
              'whole_model_quality_validated': False, 'real_time_validated': False,
              'diagnostic_requested_scope': S.DIAGNOSTIC_SCOPE,
              'final_fc1_weight_or_arithmetic_precision_proven': False,
              'operations': [], 'released': None, 'cleanup_errors': []}
    pins = {str(Path(__file__).resolve()): report['script_sha256'],
            str(Path(__file__).with_name('v50_hybrid_step2_r2_contracts.py').resolve()): CONTRACTS_SHA,
            str(args.quant_contracts.resolve()): QUANT_SHA, str(S.R1_SOURCE): S.R1_CONTRACTS_SHA}
    original_cwd = Path.cwd()
    environment_keys = tuple(ENV) + ('PATH', 'TMPDIR')
    before_env = {key: os.environ.get(key) for key in environment_keys}
    toolkit, H = None, None
    started = time.monotonic_ns()
    try:
        work.mkdir(parents=True, exist_ok=False)
        os.environ.update(ENV)
        os.environ['TMPDIR'] = str(work)
        os.environ['PATH'] = str(args.toolchain_prefix.resolve() / 'bin') + os.pathsep + os.environ.get('PATH', '')
        report['versions'] = {name: importlib.metadata.version(name) for name in C.EXPECTED_VERSIONS}
        C.check_versions(report['versions'], sys.version_info, sys.prefix, args.toolchain_prefix)
        P = C.load_source(args.pair_contracts, S.PAIR_CONTRACTS_SHA, pins)
        pair, arrays, graph, bound = P.load_pair(args.pair_dir, S.PAIR_EXPORT_SHA)
        pins.update(bound)
        del arrays
        L = C.load_source(args.layout_contracts, S.LAYOUT_CONTRACTS_SHA, pins)
        L.pin_graph(graph)
        K = C.load_source(args.capture_contracts, S.CAPTURE_CONTRACTS_SHA, pins)
        captured, samples, bound = K.load_capture(args.capture_dir, S.CAPTURE_SHA)
        pins.update(bound)
        step1, raw_paths = S.load_step1(args.step1_dir, S.STEP1_SHA, pins)
        S.load_failed_r1(args.failed_r1_receipt, pins)
        report['failed_r1_rejection_retained'] = True
        dataset, dataset_row = C.prepare_dataset(args.capture_dir, captured, pair, output, pins)
        if S.canonical(dataset_row) != S.canonical(step1['dataset']):
            raise ValueError('Original board calibration input provenance differs from successful step1')
        import ruamel.yaml
        yaml_source = Path(ruamel.yaml.__file__).resolve()
        if ruamel.yaml.__version__ != '0.18.6' or not yaml_source.is_relative_to(args.toolchain_prefix.resolve()):
            raise ValueError('Use the original isolated ruamel.yaml 0.18.6 parser')
        pins[str(yaml_source)] = C.sha(yaml_source)
        report['yaml_parser'] = {'version': ruamel.yaml.__version__, 'path': str(yaml_source),
                                'sha256': pins[str(yaml_source)], 'safe': True, 'duplicate_keys_allowed': False}
        policy, approved_cfg, gate = S.load_policy(args.precision_policy, S.POLICY_SHA,
                                                  raw_paths['configuration'], L, pins)
        report.update(original_checkpoint=pair['checkpoint'], capture_samples=samples, dataset=dataset_row,
                      raw_generated_artifacts=S.RAW_ARTIFACTS, approved_configuration=policy['approved_configuration'],
                      approved_configuration_sha256=policy['approved_configuration']['sha256'],
                      precision_edit_gate=gate, precision_policy_approved=True)
        H = C.load_source(Path(__file__).with_name('v50_hybrid_api.py'), S.HYBRID_API_SHA, pins)
        # Stage exact private artifacts so SDK work cannot overwrite the frozen step1 outputs.
        staged = {}
        for role, original in {'model': raw_paths['model'], 'data': raw_paths['data'],
                               'configuration': approved_cfg}.items():
            destination = work / original.name
            expected = C.sha(original)
            shutil.copyfile(original, destination)
            C.pin(destination, expected, pins)
            staged[role] = destination
        report['step2_options'] = {'model_input': str(staged['model']), 'data_input': str(staged['data']),
                                  'model_quantization_cfg': str(staged['configuration'])}
        from rknn.api import RKNN
        sdk_source = Path(inspect.getfile(RKNN)).resolve()
        if not sdk_source.is_relative_to(args.toolchain_prefix.resolve()):
            raise ValueError('SDK leaked from outside the selected isolated prefix')
        C.pin(sdk_source, S.SDK_API_SHA, pins)
        os.chdir(work)
        toolkit = RKNN(verbose=True, verbose_file=str(output / 'sdk.log'))
        exported = output / 'pair.rknn'
        H.step2(toolkit, staged['model'], staged['data'], staged['configuration'], exported, report, C)
        if exported.is_symlink() or not exported.is_file() or exported.stat().st_size <= 0:
            raise ValueError('SDK exported no nonempty regular original pair diagnostic')
        report['output'] = {'file': exported.name, 'bytes': exported.stat().st_size, 'sha256': C.sha(exported)}
        report['status'] = 'compiled_not_runtime_validated'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if toolkit is not None:
            H.release(toolkit, report)
        try:
            after, unchanged = C.recheck(pins)
            report.update(source_and_input_hashes=pins, source_and_input_hashes_after=after,
                          source_and_inputs_unchanged=unchanged)
            if not unchanged:
                report['cleanup_errors'].append('Frozen source/input/staged artifact bytes changed')
        except Exception as exc:
            report['cleanup_errors'].append(f'Integrity cleanup: {type(exc).__name__}: {exc}')
        try:
            os.chdir(original_cwd)
            report['cwd_restored'] = Path.cwd() == original_cwd
        except Exception as exc:
            report['cleanup_errors'].append(f'Cwd cleanup: {type(exc).__name__}: {exc}')
        try:
            for key, value in before_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            report['environment_restored'] = {key: os.environ.get(key) for key in environment_keys} == before_env
        except Exception as exc:
            report['cleanup_errors'].append(f'Environment cleanup: {type(exc).__name__}: {exc}')
        try:
            if 'output' in report:
                report['output_after'] = {'file': 'pair.rknn', 'bytes': (output / 'pair.rknn').stat().st_size,
                                          'sha256': C.sha(output / 'pair.rknn')}
                report['output_unchanged'] = report['output_after'] == report['output']
                if not report['output_unchanged']:
                    report['cleanup_errors'].append('Exported diagnostic bytes changed after release')
            if report.get('cwd_restored') is not True or report.get('environment_restored') is not True:
                report['cleanup_errors'].append('Process restoration mismatch')
            if report['cleanup_errors'] or (report['status'] == 'compiled_not_runtime_validated' and report['released'] is not True):
                report['status'] = 'failed'
            if report['status'] == 'compiled_not_runtime_validated':
                report['compile_receipt_gate'] = S.check_step2_compile(
                    report, report['script_sha256'], CONTRACTS_SHA, S.POLICY_SHA, S.STEP1_SHA, report['output']['sha256'])
        except Exception as exc:
            report['cleanup_errors'].append(f'Final receipt/output gate: {type(exc).__name__}: {exc}')
            report['status'] = 'failed'
        report['total_seconds'] = (time.monotonic_ns() - started) / 1e9
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report['peak_rss_scope'] = 'step2 compiler process, not whole VGGT/SLAM memory validation'
        with (output / 'compile.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'receipt_sha256': C.sha(output / 'compile.json'),
                      'error': report.get('error')}), flush=True)
    return 0 if report['status'] == 'compiled_not_runtime_validated' else 2


if __name__ == '__main__':
    raise SystemExit(main())
