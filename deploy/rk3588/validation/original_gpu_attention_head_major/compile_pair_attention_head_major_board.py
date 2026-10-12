#!/usr/bin/env python3
"""Board-only FP16 compile; register two original-tail conversion references.

No GPU callback or runtime inference is run here.  The SDK may invoke the exact
original CPU-ORT fragments during conversion, so this command is board-only.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
EXPORTER_SHA = 'deaa04cccf731c98db987bec26e7d69bd73c29d46d1e5f463c997208c379b225'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def artifact(directory, row):
    relative = Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Artifact path escaped')
    path = (directory / relative).resolve(strict=True)
    require(path.is_relative_to(directory.resolve()) and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'Pinned custom export artifact changed')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-sha256', required=True)
    p.add_argument('--builder-source', type=Path, required=True)
    p.add_argument('--builder-source-sha256', required=True)
    p.add_argument('--registration-form', choices=('instance', 'class'), default='instance')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    compatible = next((x.read_bytes() for x in (Path('/proc/device-tree/compatible'),
        Path('/sys/firmware/devicetree/base/compatible')) if x.is_file()), b'')
    require(sys.platform == 'linux' and platform.machine().lower() in ('aarch64', 'arm64')
            and b'rk3588' in compatible, 'Actual Toolkit/ORT is RK3588/S board-only')
    export = a.export.resolve(strict=True)
    source = a.builder_source.resolve(strict=True)
    require(sha(export) == a.export_sha256 and sha(source) == a.builder_source_sha256,
            'Frozen export/source identity differs')
    row = json.loads(export.read_text())
    require(row['schema'] == 'v84_original_pair_attention_head_major_custom_ONNX_export'
            and row['status'] == 'exported_not_compiled_or_runtime_validated'
            and row['script_sha256'] == EXPORTER_SHA
            and row['original_pair_onnx_sha256'] == GRAPH_SHA
            and row['original_parent_manifest_sha256'] == PARENT_SHA
            and row['candidate_accepted'] is False, 'Original custom export contract differs')
    proof = row['proof']
    for key in ('all_original_initializers_protobuf_unchanged', 'all_unrelated_nodes_protobuf_unchanged',
                'original_IO_protobuf_unchanged', 'original_pair_recovered_bytewise_by_restoring_only_two_tail_cuts',
                'all_original_attention_math_delegated_not_omitted'):
        require(proof[key] is True, 'Original graph preservation proof missing: ' + key)
    require(proof['original_parameter_tensors_verified'] == 36
            and proof['original_initializer_count'] == 63
            and len(proof['tails']) == 2
            and proof['candidate_node_count'] == 55
            and proof['original_two_consumer_layout_nodes_per_role_retained'] is True
            and proof['custom_output_is_original_head_major_attention_values'] is True
            and all(t['original_node_count'] == 17 and t['output_shape'] == t['input_shapes'][0]
                    for t in proof['tails']), 'Complete original head-major pair values must be retained')
    require(a.builder_source_sha256 == EXPORTER_SHA, 'Frozen head-major conversion source required')
    graph = artifact(export.parent, row['onnx'])
    paths = [Path(__file__).resolve(), source, export, graph]
    paths.extend(artifact(export.parent, row['reference_tails'][role]) for role in ('frame', 'global'))
    pins = {str(path): sha(path) for path in paths}
    a.output_dir = a.output_dir.resolve()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v84_original_pair_attention_head_major_custom_FP16_compile', status='running',
        script_sha256=sha(__file__), builder_source_sha256=a.builder_source_sha256,
        export_builder_source_sha256=EXPORTER_SHA,
        export_manifest_sha256=a.export_sha256, original_pair_onnx_sha256=GRAPH_SHA,
        original_parent_manifest_sha256=PARENT_SHA, candidate_graph_sha256=sha(graph),
        configuration=CONFIG, build_options=BUILD, registration_form=a.registration_form,
        original_model_only=True, candidate_accepted=False, conversion_reference='exact_original_tail_board_CPU_ORT',
        SDK_runtime_inference_or_GPU_kernels_executed=False,
        conversion_reference_may_execute_board_ORT=True,
        runtime_custom_op_registration_still_required=True, custom_ops=proof['tails'],
        operations=[], released=False, cleanup_errors=[], source_hashes=pins)
    sdk = None
    classes = {}
    cwd, temp, affinity = Path.cwd(), os.environ.get('TMPDIR'), os.sched_getaffinity(0)
    started = time.monotonic()
    try:
        require(importlib.metadata.version('rknn-toolkit2') == '2.3.2', 'Only original pinned Toolkit 2.3.2 accepted')
        require({4, 6}.issubset(affinity), 'Original big cores unavailable')
        os.sched_setaffinity(0, {4, 6})
        report['compile_cpus'] = sorted(os.sched_getaffinity(0))
        spec = importlib.util.spec_from_file_location('v84_original_head_major_builder', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from rknn.api import RKNN
        os.chdir(a.output_dir)
        os.environ['TMPDIR'] = str(a.output_dir)
        sdk = RKNN(verbose=False, verbose_file=str(a.output_dir / 'sdk.log'))

        def operation(name, function):
            record = dict(name=name, status='started')
            report['operations'].append(record)
            begin = time.monotonic()
            try:
                code = function()
                record.update(code=code, status='completed', seconds=time.monotonic()-begin)
                require(code in (0, None), 'SDK ' + name + ' returned ' + str(code))
            except BaseException as exc:
                record.update(status='failed', seconds=time.monotonic()-begin,
                              error=f'{type(exc).__name__}: {exc}')
                raise

        operation('config', lambda: sdk.config(**CONFIG))
        for role in ('frame', 'global'):
            cls = module.make_toolkit_custom_op(export.parent, role)
            classes[role] = cls
            registration = cls() if a.registration_form == 'instance' else cls
            operation('reg_custom_op_' + role, lambda value=registration: sdk.reg_custom_op(value))
        operation('load_onnx', lambda: sdk.load_onnx(model=str(graph)))
        operation('build', lambda: sdk.build(**BUILD))
        target = a.output_dir / 'pair.rknn'
        operation('export_rknn', lambda: sdk.export_rknn(str(target)))
        report.update(status='compiled_custom_callbacks_not_runtime_validated',
                      rknn=dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target)))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        report['conversion_compute_calls_by_role'] = {role: dict(
            attempted=cls.conversion_compute_attempted, completed=cls.conversion_compute_completed)
            for role, cls in classes.items()}
        report['conversion_input_observations_by_role'] = {
            role: cls.conversion_input_observations for role, cls in classes.items()}
        if sdk is not None:
            try:
                code = sdk.release()
                report.update(release_code=code, released=code in (0, None))
                if code not in (0, None):
                    report['cleanup_errors'].append('SDK release returned ' + str(code))
            except BaseException as exc:
                report['cleanup_errors'].append('SDK release: ' + repr(exc))
        try:
            os.chdir(cwd)
            if temp is None:
                os.environ.pop('TMPDIR', None)
            else:
                os.environ['TMPDIR'] = temp
            report['environment_restored'] = Path.cwd() == cwd and os.environ.get('TMPDIR') == temp
        except BaseException as exc:
            report['environment_restored'] = False
            report['cleanup_errors'].append('Environment restore: ' + repr(exc))
        try:
            os.sched_setaffinity(0, affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
        except BaseException as exc:
            report['affinity_restored'] = False
            report['cleanup_errors'].append('Affinity restore: ' + repr(exc))
        report['source_and_inputs_unchanged'] = True
        for path, digest in pins.items():
            try:
                require(sha(path) == digest, 'Pinned source/input changed: ' + path)
            except BaseException as exc:
                report['source_and_inputs_unchanged'] = False
                report['cleanup_errors'].append(str(exc))
        report['seconds'] = time.monotonic() - started
        if (report['cleanup_errors'] or not report['source_and_inputs_unchanged']
                or not report['environment_restored'] or not report['affinity_restored']):
            report['status'] = 'failed_cleanup_or_identity'
        (a.output_dir / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: report.get(k) for k in ('status', 'error', 'seconds', 'rknn', 'cleanup_errors')}), flush=True)
    return 0 if report['status'] == 'compiled_custom_callbacks_not_runtime_validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
