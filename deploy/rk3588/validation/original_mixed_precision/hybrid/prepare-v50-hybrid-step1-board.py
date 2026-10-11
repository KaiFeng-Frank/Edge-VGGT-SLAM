#!/usr/bin/env python3
"""Original pair hybrid step1 on RK3588S only; preserve raw SDK config bytes."""
from __future__ import annotations

import argparse
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
QUANT_SHA = '47237b1a1e1614822c75180efd15be3e5c708b610f587f698fe9b37fcf1bc649'
CAPTURE_CONTRACTS_SHA = '6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8'
CAPTURE_SHA = 'bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc'
SDK_API_SHA = '63e4506fb851214e79f933f9de0d27dab04ecf93f6f46959c07c5787433fd05e'
HYBRID_API_SHA = '9781b778c999ccadb43ee612a88c973e982dcd9f3977b15e526ddcdff3899b3d'
ENV = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2'}


def source(path, expected=None):
    import hashlib
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Frozen step1 dependency SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v50_' + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, digest


def collect_intermediates(work, output, C):
    work, output = Path(work).resolve(), Path(output).resolve()
    artifacts = {}
    stems = set()
    for role, suffix in (('model', '.model'), ('data', '.data'), ('configuration', '.quantization.cfg')):
        found = [path for path in work.glob('*' + suffix) if path.is_file() and path.stat().st_size > 0]
        if len(found) != 1:
            raise ValueError(f'Exactly one nonempty SDK-generated {suffix} artifact required')
        original = found[0]
        if original.is_symlink() or original.resolve().parent != work:
            raise ValueError('SDK-generated intermediate must be a regular confined file')
        stems.add(original.name[:-len(suffix)])
        destination = output / original.name
        if destination.exists():
            raise ValueError('Generated intermediate output must not already exist')
        digest = C.sha(original)
        shutil.copyfile(original, destination)
        if C.sha(original) != digest or C.sha(destination) != digest:
            raise ValueError('SDK intermediate bytes changed during preservation')
        artifacts[role] = {'file': destination.name, 'bytes': destination.stat().st_size,
                           'sha256': digest, 'SDK_generated_path': str(original)}
    if len(stems) != 1:
        raise ValueError('Generated model/data/configuration do not share a single original model name')
    return artifacts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    defaults = {'contracts': B / 'vggt-v49-quant-source/v49_quant_contracts.py',
                'pair-contracts': B / 'vggt-v48-pair-source/v47_pair_contracts.py',
                'pair-dir': B / 'vggt-v48-pair',
                'capture-contracts': B / 'vggt-v49-capture-source/v49_capture_contracts.py',
                'capture-dir': B / 'vggt-v49-capture-r1',
                'toolchain-prefix': B / 'vggt-v49-toolchain-venv-r1'}
    for name, path in defaults.items():
        p.add_argument('--' + name, type=Path, default=path)
    for name in ('output-dir', 'work-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(argv)
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        p.error('Actual hybrid calibration requires Linux RK3588S ARM64')
    C, _ = source(args.contracts, QUANT_SHA)
    compatible = C.require_board()
    output, work = args.output_dir.resolve(), args.work_dir.resolve()
    if (Path.cwd().resolve() != BASE or output == B or not output.is_relative_to(B)
            or work == Path('/dev/shm') or not work.is_relative_to('/dev/shm')
            or Path(sys.prefix).resolve() != args.toolchain_prefix.resolve()):
        p.error('Use the board project cwd, isolated interpreter and new confined output/work directories')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v50_original_VGGT_pair_board_hybrid_step1', 'status': 'failed',
              'script_sha256': C.sha(__file__), 'quant_contracts_sha256': QUANT_SHA,
              'capture_manifest_sha256': CAPTURE_SHA, 'capture_contracts_sha256': CAPTURE_CONTRACTS_SHA,
              'pair_export_sha256': C.PAIR_EXPORT_SHA, 'pair_graph_sha256': C.PAIR_GRAPH_SHA,
              'board': {'machine': platform.machine(), 'device_tree_compatible': compatible, 'python': sys.version},
              'calibration_on_board_only': True, 'model_inference_on_host': False,
              'calibration_api_invoked': False, 'calibration_completed_on_board': False,
              'configuration': {key: value for key, value in C.CONFIGURATION.items() if key != 'do_quantization'},
              'input_transform': {'mean_values': None, 'std_values': None, 'quant_img_RGB2BGR': False,
                                  'transpose': False, 'extra_batch_dimension': False},
              'precision_policy_approved': False, 'candidate_accepted': False,
              'historical_FP32_rejections_retained': True, 'fixed_FP32_gate': {'rtol': 1e-4, 'atol': 1e-5},
              'SDK_runtime_or_inference_API_called': False, 'whole_model_quality_validated': False,
              'real_time_validated': False, 'operations': [], 'released': None, 'cleanup_errors': []}
    pins = {str(Path(__file__).resolve()): report['script_sha256'], str(args.contracts.resolve()): QUANT_SHA}
    original_cwd = Path.cwd()
    environment_keys = tuple(ENV) + ('PATH', 'TMPDIR')
    before_env = {key: os.environ.get(key) for key in environment_keys}
    toolkit = None
    started = time.monotonic_ns()
    try:
        work.mkdir(parents=True, exist_ok=False)
        os.environ.update(ENV)
        os.environ['TMPDIR'] = str(work)
        os.environ['PATH'] = str(args.toolchain_prefix.resolve() / 'bin') + os.pathsep + os.environ.get('PATH', '')
        report['versions'] = {name: importlib.metadata.version(name) for name in C.EXPECTED_VERSIONS}
        C.check_versions(report['versions'], sys.version_info, sys.prefix, args.toolchain_prefix)
        P = C.load_source(args.pair_contracts, C.PAIR_CONTRACTS_SHA, pins)
        pair, arrays, graph, bound = P.load_pair(args.pair_dir, C.PAIR_EXPORT_SHA)
        pins.update(bound)
        del arrays
        K = C.load_source(args.capture_contracts, CAPTURE_CONTRACTS_SHA, pins)
        captured, samples, bound = K.load_capture(args.capture_dir, CAPTURE_SHA)
        pins.update(bound)
        dataset, dataset_row = C.prepare_dataset(args.capture_dir, captured, pair, output, pins)
        H, hybrid_source_sha = source(Path(__file__).with_name('v50_hybrid_api.py'), HYBRID_API_SHA)
        pins[str(Path(__file__).with_name('v50_hybrid_api.py').resolve())] = hybrid_source_sha
        report.update(hybrid_api_sha256=hybrid_source_sha, checkpoint=pair['checkpoint'], dataset=dataset_row,
                      capture_script_sha256=captured['script_sha256'], step1_options=dict(H.STEP1_OPTIONS),
                      toolchain_prefix=str(args.toolchain_prefix.resolve()))
        from rknn.api import RKNN
        sdk_source = Path(inspect.getfile(RKNN)).resolve()
        if not sdk_source.is_relative_to(args.toolchain_prefix.resolve()):
            raise ValueError('Actual hybrid SDK leaked from outside the selected isolated prefix')
        C.pin(sdk_source, SDK_API_SHA, pins)
        report['SDK_API_source_sha256'] = SDK_API_SHA
        staged_graph = work / graph.name
        shutil.copyfile(graph, staged_graph)
        C.pin(staged_graph, C.PAIR_GRAPH_SHA, pins)
        if C.sha(graph) != C.PAIR_GRAPH_SHA:
            raise ValueError('Frozen original graph changed during confined staging')
        report['SDK_graph_input'] = {'original_file': str(graph), 'staged_file': str(staged_graph),
                                     'sha256': C.PAIR_GRAPH_SHA,
                                     'bytes': staged_graph.stat().st_size,
                                     'exact_original_graph_bytes': True}
        os.chdir(work)
        toolkit = RKNN(verbose=True, verbose_file=str(output / 'sdk.log'))
        H.step1(toolkit, staged_graph, dataset, report, C)
        report['generated_artifacts'] = collect_intermediates(work, output, C)
        report['status'] = 'hybrid_intermediates_generated_not_precision_approved'
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
                report['cleanup_errors'].append('Source/input bytes changed')
        except Exception as exc:
            report['cleanup_errors'].append(f'Integrity cleanup: {type(exc).__name__}: {exc}')
        try:
            os.chdir(original_cwd)
            report['cwd_restored'] = Path.cwd() == original_cwd
            if not report['cwd_restored']:
                report['cleanup_errors'].append('Process cwd restoration mismatch')
        except Exception as exc:
            report['cleanup_errors'].append(f'Process cwd cleanup: {type(exc).__name__}: {exc}')
        try:
            for key, value in before_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            report['environment_restored'] = {key: os.environ.get(key) for key in environment_keys} == before_env
            if not report['environment_restored']:
                report['cleanup_errors'].append('Process environment restoration mismatch')
        except Exception as exc:
            report['cleanup_errors'].append(f'Process environment cleanup: {type(exc).__name__}: {exc}')
        try:
            if 'generated_artifacts' in report:
                report['generated_artifacts_after'] = {
                    role: {'sha256': C.sha(output / item['file']),
                           'bytes': (output / item['file']).stat().st_size}
                    for role, item in report['generated_artifacts'].items()}
                report['generated_artifacts_unchanged'] = all(
                    report['generated_artifacts_after'][role] ==
                    {'sha256': item['sha256'], 'bytes': item['bytes']}
                    for role, item in report['generated_artifacts'].items())
                if not report['generated_artifacts_unchanged']:
                    report['cleanup_errors'].append('Preserved SDK-generated intermediate bytes changed')
        except Exception as exc:
            report['cleanup_errors'].append(f'Generated artifact cleanup: {type(exc).__name__}: {exc}')
        if report['cleanup_errors'] or (report['calibration_completed_on_board'] and report['released'] is not True):
            report['status'] = 'failed'
        report['total_seconds'] = (time.monotonic_ns() - started) / 1e9
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report['peak_rss_scope'] = 'hybrid step1 compiler process, not full-model or SLAM memory'
        with (output / 'step1.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'receipt_sha256': C.sha(output / 'step1.json')}), flush=True)
    return 0 if report['status'] == 'hybrid_intermediates_generated_not_precision_approved' else 2


if __name__ == '__main__':
    raise SystemExit(main())
