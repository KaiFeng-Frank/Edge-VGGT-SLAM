#!/usr/bin/env python3
"""RK3588-only compile of the frozen original pair group with weight compression."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

EXPORT_SHA = '57c5dd1bcf1cad4d258d1b9c91ccdf23e4e19c71f58975022b48ad3ca8c6cf55'
CHECK_SHA = '5d86ce49dad23858b21e1e8c35ebfaf5d8f3e23af00dfad6daf2b67ae84f3603'
BASELINE_SHA = '8943f56ee991893d2b1bdb726e9d499d3c3e91145ff35672a8d0324755c93935'
GRAPH_SHA = 'fa41f8743f0d03615bd1189c15402661e046bad2ac388aff4e5508bf778dead1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, expected):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, 'Pinned file changed: ' + str(path))
    return path


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def board_guard():
    tree = Path('/proc/device-tree/compatible')
    require(platform.machine() == 'aarch64' and tree.exists()
        and b'rockchip,rk3588' in tree.read_bytes(), 'Compilation is RK3588 board-only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'check-report', 'baseline-compile', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    board_guard()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    old_cwd, old_tmp, old_affinity = Path.cwd(), os.environ.get('TMPDIR'), os.sched_getaffinity(0)
    started, sdk, watched, pins = time.perf_counter(), None, [], {}
    report = dict(schema='v65_original_VGGT_pair0_pair1_weight_compression_build', mode='compile', status='running',
        script_sha256=sha(__file__), manifest_sha256=EXPORT_SHA, check_receipt_sha256=CHECK_SHA,
        baseline_compile_sha256=BASELINE_SHA, graph_sha256=GRAPH_SHA, candidate_accepted=False,
        quality_accepted=False, realtime_validated=False, weight_only_arithmetic_type_claimed=False,
        original_model_only=True, ORT_or_model_or_calibration_executed=False,
        only_configuration_difference={'quantize_weight': {'before': False, 'after': True}},
        operations=[], released=False, cleanup_errors=[], timing_scope='Compiler elapsed time, not inference performance')
    try:
        manifest_path = pin(args.manifest, EXPORT_SHA)
        check_path = pin(args.check_report, CHECK_SHA)
        baseline_path = pin(args.baseline_compile, BASELINE_SHA)
        manifest, check, baseline = [json.loads(p.read_text()) for p in (manifest_path, check_path, baseline_path)]
        row = manifest['blocks'][0]
        require(manifest['schema'] == 'v64_original_VGGT_pair0_pair1_group_ONNX_manifest'
            and len(manifest['blocks']) == 1 and row['indices'] == [0, 1] and row['sha256'] == GRAPH_SHA, 'Frozen original group differs')
        relative = Path(row['file'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Relative frozen graph required')
        graph = pin(manifest_path.parent / relative, GRAPH_SHA)
        require(graph.is_relative_to(manifest_path.parent) and graph.stat().st_size == row['bytes'], 'Graph ownership/size differs')
        for item, status in ((check, 'checked_separate_ONNX_equivalence_not_quality_accepted'), (baseline, 'compiled_not_quality_accepted')):
            require(item['status'] == status and item['manifest_sha256'] == EXPORT_SHA and item['graph_sha256'] == GRAPH_SHA
                and item['source_and_inputs_unchanged'] is True and item['cleanup_errors'] == [], 'Original successful check/build required')
        require(check['all_separate_ONNX_checks_byte_equal'] is True and len(check['cases']) == 2
            and all(len(c['checks']) == 4 and all(v['byte_equal_to_two_separate_FP32_ONNX'] is True for v in c['checks']) for c in check['cases'])
            and baseline['check_receipt_sha256'] == CHECK_SHA and baseline['released'] is True, 'Frozen two-case/four-output check lineage differs')
        configuration, build = dict(baseline['configuration']), dict(baseline['build_options'])
        require(configuration['target_platform'] == 'rk3588' and configuration['float_dtype'] == 'float16'
            and configuration['quantize_weight'] is False and build == dict(do_quantization=False, dataset=None, auto_hybrid=False), 'Nonquantized FP16 baseline required')
        configuration['quantize_weight'] = True
        require({k for k in configuration if configuration[k] != baseline['configuration'][k]} == {'quantize_weight'}, 'Only weight-compression option may change')
        report.update(configuration=configuration, build_options=build, cases=check['cases'], output_bindings=row['output_bindings'],
            original_FP32_gate_passed_where_reference_available=check['original_FP32_gate_passed_where_reference_available'],
            original_FP32_references_cover_all_cases=check['original_FP32_references_cover_all_cases'], baseline_rknn=baseline['rknn'])
        watched = [Path(__file__).resolve(), manifest_path, check_path, baseline_path, graph]
        pins = {str(p): sha(p) for p in watched}
        report['source_and_input_hashes'] = pins
        version = importlib.metadata.version('rknn-toolkit2')
        require(version.split('+')[0] == '2.3.2' and version == baseline['toolkit_version'], 'Same RKNN Toolkit2 2.3.2 required')
        from rknn.api import RKNN
        source = Path(sys.modules[RKNN.__module__].__file__).resolve()
        require(sha(source) == baseline['toolkit_source_sha256'], 'Baseline compiler source changed')
        report.update(toolkit_version=version, toolkit_source_sha256=sha(source))
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.chdir(output)
        os.environ['TMPDIR'] = str(output)
        sdk = RKNN(verbose=False, verbose_file=str(output / 'sdk.log'))
        for name, call in [('config', lambda: sdk.config(**configuration)), ('load_onnx', lambda: sdk.load_onnx(model=str(graph))),
                ('build', lambda: sdk.build(**build)), ('export_rknn', lambda: sdk.export_rknn(str(output / 'group.rknn')))]:
            operation = dict(name=name, status='attempted')
            report['operations'].append(operation)
            tick = time.perf_counter()
            try:
                code = call()
                operation.update(code=code, status='returned')
                require(code in (None, 0), f'{name} returned {code}')
            except BaseException as exc:
                operation.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                operation['seconds'] = time.perf_counter() - tick
                write(output / 'result.json', report)
        model = output / 'group.rknn'
        report.update(rknn=dict(file='group.rknn', bytes=model.stat().st_size, sha256=sha(model)), status='compiled_not_quality_accepted')
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if sdk is not None:
            tick = time.perf_counter()
            try:
                code = sdk.release()
                report['release_code'] = code
                require(code in (None, 0), f'release returned {code}')
                report['released'] = True
            except BaseException as exc:
                report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
                report['status'] = 'failed_cleanup'
            finally:
                report['release_seconds'] = time.perf_counter() - tick
        try:
            os.chdir(old_cwd)
            if old_tmp is None:
                os.environ.pop('TMPDIR', None)
            else:
                os.environ['TMPDIR'] = old_tmp
            os.sched_setaffinity(0, old_affinity)
            after = {str(p): sha(p) for p in watched}
            report.update(source_and_input_hashes_after=after, source_and_inputs_unchanged=bool(pins) and after == pins,
                environment_restored=Path.cwd() == old_cwd and os.environ.get('TMPDIR') == old_tmp,
                affinity_restored=os.sched_getaffinity(0) == old_affinity)
            if pins and after != pins:
                report.update(status='failed', error='Frozen inputs or sources changed during compilation')
        except BaseException as exc:
            report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
            report['status'] = 'failed_cleanup'
        report['seconds'] = time.perf_counter() - started
        write(output / 'result.json', report)
    require(report['status'] == 'compiled_not_quality_accepted' and not report['cleanup_errors'], 'Compile/source/cleanup check failed')
    print(json.dumps({k: report.get(k) for k in ('status', 'seconds', 'manifest_sha256', 'check_receipt_sha256', 'rknn')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
