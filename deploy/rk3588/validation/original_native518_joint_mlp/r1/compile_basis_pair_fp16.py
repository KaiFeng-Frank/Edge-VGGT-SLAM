#!/usr/bin/env python3
"""Board-only nonquantizing compilation of one gated native frame0/global0 pair.

No runtime inference or calibration is invoked. A/F/B retain the same CONFIG;
their numerical/trajectory acceptance requires separate actual measurements.
The caller supplies the bounded CPU/thermal/process-group controller.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

EXPORT_SHA = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
CHECK_SHA = '48a7d5034b9eea0afa139e4a69cdb5b393c694dcec2b8e874a1c1f380f5f42b1'
PAIR_SOURCE_SHA = '73affeb0b78a794ed214cadcf0307d164ef74f79d52b779bc579e906a1089b70'
CONFIG_SHA = '442e22d5f8aaa033604610c92d5ce6a8b889005601f3d9f1cf78394c4f41842a'
CONFIG_AUTHORITY_SHA = 'ade8f932c267bdc2a9ff5854e305d87456984630025aa34c0c448499f5a6a81b'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['A', 'F', 'B'], required=True)
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--export-receipt', type=Path, required=True)
    p.add_argument('--check-receipt', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cpu-affinity', nargs='+', type=int, default=[4, 6])
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and
            b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'RK3588S board-only compiler; host execution is forbidden for this worker')
    graph_path, export_path, check_path = (v.resolve(strict=True) for v in
                                         (a.onnx, a.export_receipt, a.check_receipt))
    source = Path(__file__).resolve(strict=True)
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out / 'tmp').mkdir()
    old_env, old_cwd, old_affinity = dict(os.environ), Path.cwd(), set(os.sched_getaffinity(0))
    handlers, watched, sdk = {}, {source: sha(source)}, None
    began = time.monotonic()
    report = dict(schema='v127_original_native518_basis_pair_FP16_compile', status='running',
        script_sha256=watched[source], variant=a.variant, configuration=CONFIG, build_options=BUILD,
        configuration_authority_source_sha256=CONFIG_AUTHORITY_SHA,
        actual_execution_on_board_only=True, original_ONNX_modified_by_compiler=False,
        runtime_inference_invoked_by_this_worker=False, calibration_invoked=False,
        system_or_frequency_writes=False, candidate_accepted=False,
        full_model_quality_performance_realtime_or_8GB_accepted=False,
        calls=[], cleanup_errors=[], sdk_constructor_attempted=False, sdk_initialized=False,
        sdk_release_called=False, sdk_released=False)

    def progress(stage):
        report.update(stage=stage, total_seconds=time.monotonic() - began,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        temporary = out / 'result.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(out / 'result.json')
        print(json.dumps(dict(stage=stage, status=report['status'])), flush=True)

    def pin(path, expected, size=None):
        require(len(expected) == 64 and sha(path) == expected and
                (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
        watched[path] = expected

    def interrupted(number, frame):
        raise InterruptedError('Controller termination signal ' + str(number))

    progress('preflight')
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.signal(number, interrupted)
        require(a.cpu_affinity and set(a.cpu_affinity) <= old_affinity & {4, 5, 6, 7}, 'Requested A76 affinity unavailable')
        os.sched_setaffinity(0, set(a.cpu_affinity))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        os.environ.update(CUDA_VISIBLE_DEVICES='', TMPDIR=str(out / 'tmp'))
        os.chdir(out)
        pin(export_path, EXPORT_SHA); pin(check_path, CHECK_SHA)
        exported, checked = json.loads(export_path.read_text()), json.loads(check_path.read_text())
        require(exported['schema'] == 'v126_original_native518_basis_pair_export' and
                exported['status'] == 'exported_original_basis_pair_not_numerically_accepted' and
                exported['all36_original_learned_values_recovered_and_verified'], 'Gated original36-parameter export required')
        require(checked['schema'] == 'v126_original_native518_basis_pair_check' and
                checked['status'] == 'checked_original_basis_pair_fixed_FP32_gate_passed' and
                checked['FP32_gate_passed'] and checked['all6_calls_completed'] and
                checked['ORT_sessions_dropped'] and checked['ORT_session_dropped'] and
                checked['export_receipt']['sha256'] == EXPORT_SHA and checked['models'] == exported['models'],
                'Actual completed R126 R2 six-call FP32 gate required before SDK import')
        for prior in (exported, checked):
            require(prior['script_sha256'] == PAIR_SOURCE_SHA and prior['config_sha256'] == CONFIG_SHA and
                    prior['fixed_gate'] == GATE and prior['strict_elementwise_allclose_is_independent_diagnostic'] and
                    prior['source_and_inputs_unchanged'] and prior['affinity_restored'] and
                    prior['environment_restored'] and prior['cleanup_errors'] == [], 'Frozen proof/gate/source cleanup differs')
        require([(r['input_id'], r['variant']) for r in checked['checks']] ==
                [(case, variant) for case in ('cpu_prefix', 'npu_prefix') for variant in ('A', 'F', 'B')], 'All six complete input/variant calls required')
        for row in checked['checks']:
            require(row['fixed_gate_passed'] and row['session_dropped'] and len(row['outputs']) == 2 and row['comparisons'], 'Incomplete FP32 call')
            for metric in row['comparisons'].values():
                require(metric['fixed_gate_passed'] and math.isfinite(metric['relative_l2']) and
                        math.isfinite(metric['max_abs']) and metric['relative_l2'] <= GATE['relative_l2_max'] and
                        metric['max_abs'] <= GATE['max_abs_max'], 'Original numeric FP32 gate did not pass')
        model = next(row for row in exported['models'] if row['key'] == a.variant)
        require(model['canonical_outputs_restored'] and model['original36_learned_values_verified'] and
                model['original_K4_partition_and_sum_node_order_preserved'] and
                model['basis_persistent_frame_to_global'] == (a.variant == 'B'), 'Original continuous pair contract differs')
        pin(graph_path, model['onnx']['sha256'], model['onnx']['bytes'])
        report.update(export_receipt=dict(file=str(export_path), sha256=EXPORT_SHA),
            check_receipt=dict(file=str(check_path), sha256=CHECK_SHA), fixed_FP32_gate=GATE,
            six_original_FP32_calls_passed=True, original_parameter_value_count=36,
            checkpoint=exported['checkpoint'], source_hashes=exported['source_hashes'],
            input_scope=exported['input_case_scope'], onnx=dict(file=str(graph_path), **{k: model['onnx'][k] for k in ('bytes', 'sha256')}),
            basis_persistent_frame_to_global=model['basis_persistent_frame_to_global'],
            cpu_affinity=sorted(os.sched_getaffinity(0)))
        progress('static_binding_validation')
        import onnx
        graph = onnx.load_model(str(graph_path), load_external_data=False)
        require(all(t.data_location != onnx.TensorProto.EXTERNAL for t in graph.graph.initializer), 'Self-contained graph required')
        onnx.checker.check_model(graph, full_check=True)
        def binding(value, index):
            t = value.type.tensor_type
            require(t.elem_type == onnx.TensorProto.FLOAT and all(d.HasField('dim_value') and d.dim_value > 0 for d in t.shape.dim), 'Static logical FP32 binding required')
            return dict(index=index, name=value.name, shape=[d.dim_value for d in t.shape.dim], dtype='float32')
        inputs = [v for v in graph.graph.input if v.name not in {t.name for t in graph.graph.initializer}]
        input_rows = [binding(v, i) for i, v in enumerate(inputs)]
        output_rows = [binding(v, i) for i, v in enumerate(graph.graph.output)]
        require(input_rows == model['input_bindings'] == [dict(index=0, name='input', shape=FRAME, dtype='float32')] and
                output_rows == model['output_bindings'] == [dict(index=0, name='frame_output', shape=FRAME, dtype='float32'),
                                                            dict(index=1, name='global_output', shape=GLOBAL, dtype='float32')], 'One input/two canonical endpoints required')
        report.update(input_bindings=input_rows, output_bindings=output_rows,
                      graph_node_count=len(graph.graph.node), graph_initializer_count=len(graph.graph.initializer))
        del graph, inputs; gc.collect()
        report['versions'] = {name: importlib.metadata.version(name) for name in ('rknn-toolkit2', 'onnx', 'numpy')}
        require(report['versions']['rknn-toolkit2'].split('+')[0] == '2.3.2', 'RKNN toolkit2.3.2 required')
        progress('SDK_import')
        import rknn.api as api
        api_path = Path(api.__file__).resolve()
        report['SDK_api_module'] = dict(file=str(api_path), bytes=api_path.stat().st_size, sha256=sha(api_path))
        watched[api_path] = report['SDK_api_module']['sha256']
        report['sdk_constructor_attempted'] = True; progress('SDK_constructor')
        sdk = api.RKNN(verbose=False, verbose_file=str(out / 'sdk.log')); report['sdk_initialized'] = True
        output = out / ('pair-' + a.variant + '.rknn')
        operations = [('config', lambda: sdk.config(**CONFIG)), ('load_onnx', lambda: sdk.load_onnx(model=str(graph_path))),
                      ('build', lambda: sdk.build(**BUILD)), ('export_rknn', lambda: sdk.export_rknn(str(output)))]
        for name, operation in operations:
            entry = dict(operation=name, status='running'); report['calls'].append(entry); progress(name + '_start')
            tick = time.monotonic(); code = operation()
            entry.update(status='completed', returncode=code, seconds=time.monotonic() - tick)
            require(code in (0, None), 'SDK ' + name + ' failed: ' + str(code)); progress(name + '_completed')
        require(output.stat().st_size > 0, 'Empty RKNN artifact')
        report.update(rknn=dict(file=output.name, bytes=output.stat().st_size, sha256=sha(output)),
                      all_operations_succeeded=True, status='compiled_not_runtime_accepted')
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if sdk is not None:
            report['sdk_release_called'] = True
            try:
                progress('SDK_release_start')
            except BaseException as error:
                report['cleanup_errors'].append('Pre-release receipt: ' + str(error))
            try:
                code = sdk.release(); report['sdk_release_code'] = code
                require(code in (0, None), 'SDK release failed: ' + str(code)); report['sdk_released'] = True
            except BaseException as error:
                report['cleanup_errors'].append('SDK release: ' + str(error))
        for label, restore in (('cwd', lambda: os.chdir(old_cwd)), ('affinity', lambda: os.sched_setaffinity(0, old_affinity))):
            try:
                restore()
            except BaseException as error:
                report['cleanup_errors'].append(label + ': ' + str(error))
        os.environ.clear(); os.environ.update(old_env)
        for number, handler in handlers.items():
            signal.signal(number, handler)
        report.update(cwd_restored=Path.cwd() == old_cwd, affinity_restored=set(os.sched_getaffinity(0)) == old_affinity,
                      environment_restored=dict(os.environ) == old_env,
                      source_and_inputs_unchanged=all(p.is_file() and sha(p) == expected for p, expected in watched.items()),
                      input_and_source_SHA_pins={str(p): expected for p, expected in watched.items()})
        if report['cleanup_errors'] or not all(report[k] for k in ('cwd_restored', 'affinity_restored', 'environment_restored', 'source_and_inputs_unchanged')):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == 'compiled_not_runtime_accepted' and report['sdk_released'] else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
