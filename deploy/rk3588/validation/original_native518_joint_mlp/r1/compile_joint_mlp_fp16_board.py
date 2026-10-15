#!/usr/bin/env python3
"""Board-only FP16 compilation after all six real-input joint-MLP FP32 gates.

No calibration or runtime forward is invoked. Source interleaving is not proof
of the compiled schedule, scratch lifetime, precision, placement or speed.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time
import traceback

EXPORT_SHA = 'f8f935e43db97ffb5cee683d8eee2a8fe4ceb728a96f10c37ad8b6772a76ea49'
CHECK_SHAS = ('4df223f1fcf6808b3642eb5015ad2e0515dc1f1df1fd5f6966baada0e91a29b7',
    '177f8cd85c3306d23effb2e68358006e817e8f001b2a7030bb1ea10baa7f1566',
    '666908dfbf47c6f9641f1d3d74588bb1abe5298e09dc83dc2205f209bd516490')
SOURCE_SHA = '82f35353bf57b662b95b89cf5472e18e50e832519e8545ab7237994e35e06c07'
CONFIG_SOURCE_SHA = '2bbf180a9a70444ba592b7898fa23b96b11f36da3859f07cc5b435f0a25acb11'
PARENT_EXPORT_SHA = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
PARENT_CHECK_SHA = '48a7d5034b9eea0afa139e4a69cdb5b393c694dcec2b8e874a1c1f380f5f42b1'
PARENT_GRAPH_SHA = '57b60feceea0edb57ca4cea38200d6ea47d4e4ee51e4a546357267cbc9f676bb'
REFERENCE_SHAS = ('70caf6de746831cf7950fb4971359a1341d2dd4d845f3f3eb5a1da0e1b32072c',
    'b975a0eb82fa3d2ce275a8e487aed583e37bfc16bfb2afc5b6e42f7fc0402618',
    '17f9aec6d7c93189c7ba74c33c3101160c4170f0a38e2b6ae91d77eeb4fdc250')
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)
SUCCESS = 'compiled_original_joint_MLP_pair_not_runtime_or_quality_accepted'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def validate_metadata(exported, checks):
    require(exported['schema'] == 'v129_original_native518_joint_MLP_pair_ONNX_export' and
        exported['status'] == 'exported_original_joint_MLP_pair_not_FP32_or_NPU_accepted' and
        exported['script_sha256'] == SOURCE_SHA and exported['parent_export_receipt']['sha256'] == PARENT_EXPORT_SHA and
        exported['parent_check_receipt']['sha256'] == PARENT_CHECK_SHA and exported['parent_graph']['sha256'] == PARENT_GRAPH_SHA,
        'Actual original-A joint-MLP export lineage differs')
    proof = exported['proof']
    flags = ('all4096_original_hidden_neurons_covered_once', 'original_FC1_weight_columns_and_bias_reconstructed_bitwise',
        'original_FC2_K4_weights_protobuf_unchanged', 'all36_original_learned_values_recovered_and_verified',
        'all_frame0_nodes_and_initializers_protobuf_unchanged', 'all_untouched_nodes_and_initializers_protobuf_unchanged',
        'original_three_partial_sum_Add_nodes_and_one_bias_Add_node_protobuf_unchanged',
        'original_GELU_operator_and_operand_order_per_branch_unchanged', 'original_LayerScale_residual_norm_and_attention_unchanged',
        'original_all1041_frame_tokens_and2082_global_tokens_preserved', 'original_canonical_IO_protobuf_unchanged',
        'whole_parent_inverse_ModelProto_byte_equal')
    require(all(proof[k] for k in flags) and proof['inverse_parent_ModelProto_sha256'] == PARENT_GRAPH_SHA and
        proof['candidate_nodes'] == 160 and proof['candidate_initializers'] == 75 and
        set(exported['original_parameters']) == {'frame', 'global'} and
        all(len(exported['original_parameters'][k]) == 18 for k in ('frame', 'global')), 'Complete original math/parameter proof differs')
    require(len(exported['models']) == 1 and exported['models'][0]['key'] == 'joint_mlp', 'One complete pair graph required')
    require(len(checks) == 3, 'All three two-input checks required')
    for i, checked in enumerate(checks):
        require(checked['schema'] == 'v129_original_native518_joint_MLP_pair_FP32_check' and
            checked['status'] == 'checked_original_joint_MLP_pair_fixed_FP32_gate_passed_not_NPU_accepted' and
            checked['script_sha256'] == SOURCE_SHA and checked['export_receipt']['sha256'] == EXPORT_SHA and
            checked['batch_index'] == i and checked['models'] == exported['models'] and checked['proof'] == proof and
            checked['checkpoint'] == exported['checkpoint'] and checked['original_parameters'] == exported['original_parameters'] and
            checked['source_hashes'] == exported['source_hashes'] and checked['original_FP32_reference_receipt']['sha256'] == REFERENCE_SHAS[i] and
            checked['all2_calls_completed'] and checked['FP32_gate_passed'] and checked['ORT_sessions_dropped'] and
            checked['ORT_sessions_created'] == checked['ORT_sessions_dropped_count'] == 2 and
            [r['input_id'] for r in checked['checks']] == ['pair' + str(2 * i), 'pair' + str(2 * i + 1)], 'Actual six-case completed FP32 check differs')
        for row in checked['checks']:
            require(row['complete'] and row['input_unchanged'] and row['fixed_gate_passed'] and
                [r['name'] for r in row['outputs']] == ['frame_output', 'global_output'], 'Incomplete original canonical pair outputs')
            require(set(row['comparisons_vs_original_A']) == {'frame_output', 'global_output'}, 'Complete endpoint comparisons required')
            for metric in row['comparisons_vs_original_A'].values():
                require(metric['fixed_gate_passed'] and math.isfinite(metric['relative_l2']) and math.isfinite(metric['max_abs']) and
                    metric['relative_l2'] <= GATE['relative_l2_max'] and metric['max_abs'] <= GATE['max_abs_max'], 'Original numeric FP32 gate failed')
    for prior in [exported, *checks]:
        require(prior['fixed_gate'] == GATE and prior['strict_allclose_is_independent_diagnostic'] and
            all(prior[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored')) and not prior['cleanup_errors'], 'Original clean fixed-gate receipt required')
    return exported['models'][0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', type=Path, required=True); p.add_argument('--export-receipt', type=Path, required=True)
    p.add_argument('--check-receipts', type=Path, nargs=3, required=True)
    p.add_argument('--configuration-source', type=Path, default=Path(__file__).with_name('compile_basis_pair_fp16.py'))
    p.add_argument('--output-dir', type=Path, required=True); p.add_argument('--minimum-free-mib', type=int, default=1024)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S board-only compiler')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False); (out / 'tmp').mkdir()
    old_env, old_cwd, old_affinity = dict(os.environ), Path.cwd(), set(os.sched_getaffinity(0))
    source = Path(__file__).resolve(); watched, handlers, sdk = {source: sha(source)}, {}, None; began = time.monotonic()
    report = dict(schema='v129_original_native518_joint_MLP_pair_FP16_compile', status='running', script_sha256=watched[source],
        configuration=CONFIG, build_options=BUILD, configuration_source_sha256=CONFIG_SOURCE_SHA, SDK_verbose=True,
        actual_execution_on_board_only=True, calibration_or_runtime_forward_invoked=False, candidate_accepted=False,
        original_ONNX_modified=False, system_or_frequency_writes=False,
        actual_SDK_schedule_memory_lifetime_or_precision_established=False, calls=[], cleanup_errors=[], sdk_initialized=False, sdk_released=False)

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - began, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        tmp = out / 'result.tmp'; tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); tmp.replace(out / 'result.json'); print(stage, flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path)); watched[path] = digest; return path

    def relative(folder, row):
        file = Path(row['file']); require(not file.is_absolute() and '..' not in file.parts, 'Receipt-relative artifact required')
        return pin(folder / file, row['sha256'], row['bytes'])

    progress('preflight')
    try:
        def interrupted(number, frame): raise InterruptedError('Controller signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require({4, 5, 6, 7} <= old_affinity, 'A76 affinity required'); os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', TMPDIR=str(out / 'tmp')); os.chdir(out)
        ep = pin(a.export_receipt, EXPORT_SHA); cps = [pin(path, digest) for path, digest in zip(a.check_receipts, CHECK_SHAS)]
        exported, checks = json.loads(ep.read_text()), [json.loads(path.read_text()) for path in cps]; model = validate_metadata(exported, checks)
        graph_path = pin(a.onnx, model['onnx']['sha256'], model['onnx']['bytes'])
        for cp, checked in zip(cps, checks):
            for row in checked['checks']:
                for artifact in row['outputs']: relative(cp.parent, artifact)
        config_path = pin(a.configuration_source, CONFIG_SOURCE_SHA)
        spec = importlib.util.spec_from_file_location('v129_original_configuration', config_path); authority = importlib.util.module_from_spec(spec); spec.loader.exec_module(authority)
        require(authority.CONFIG == CONFIG and authority.BUILD == BUILD, 'Frozen R127 FP16 configuration/build differs')
        import onnx
        graph = onnx.load_model(graph_path, load_external_data=False); onnx.checker.check_model(graph, full_check=True)
        require(len(graph.graph.node) == 160, 'Candidate node count differs')
        require(len(graph.graph.initializer) == 75 and all(t.data_location != onnx.TensorProto.EXTERNAL for t in graph.graph.initializer), 'Self-contained full graph required')
        def binding(value, index):
            t = value.type.tensor_type; require(t.elem_type == onnx.TensorProto.FLOAT and all(d.HasField('dim_value') and d.dim_value > 0 for d in t.shape.dim), 'Static FP32 binding required')
            return dict(index=index, name=value.name, shape=[d.dim_value for d in t.shape.dim], dtype='float32')
        init_names = {t.name for t in graph.graph.initializer}; input_rows = [binding(v, i) for i, v in enumerate(v for v in graph.graph.input if v.name not in init_names)]
        output_rows = [binding(v, i) for i, v in enumerate(graph.graph.output)]
        require(input_rows == model['input_bindings'] == [dict(index=0, name='input', shape=FRAME, dtype='float32')] and
            output_rows == model['output_bindings'] == [dict(index=0, name='frame_output', shape=FRAME, dtype='float32'), dict(index=1, name='global_output', shape=GLOBAL, dtype='float32')], 'Original fair canonical pair IO differs')
        del graph; gc.collect(); free = shutil.disk_usage(out).free
        require(a.minimum_free_mib > 0 and free >= a.minimum_free_mib * (1 << 20), 'Output free-space gate failed')
        report.update(export_receipt=dict(file=str(ep), sha256=EXPORT_SHA), check_receipts=[dict(file=str(p), sha256=s) for p, s in zip(cps, CHECK_SHAS)],
            onnx=dict(file=str(graph_path), bytes=model['onnx']['bytes'], sha256=model['onnx']['sha256']), input_bindings=input_rows, output_bindings=output_rows,
            checkpoint=exported['checkpoint'], original_parameters=exported['original_parameters'], source_hashes=exported['source_hashes'], proof=exported['proof'],
            six_new_original_FP32_gates_passed=True, fixed_gate=GATE, pre_SDK_free_bytes=free,
            versions={name: importlib.metadata.version(name) for name in ('rknn-toolkit2', 'onnx')})
        require(report['versions']['rknn-toolkit2'].split('+')[0] == '2.3.2', 'Toolkit2.3.2 required'); progress('verified_before_SDK_import')
        import rknn.api as api
        sdk = api.RKNN(verbose=True, verbose_file=str(out / 'sdk.log')); report['sdk_initialized'] = True; output = out / 'pair-joint-mlp.rknn'
        for name, action in [('config', lambda: sdk.config(**CONFIG)), ('load_onnx', lambda: sdk.load_onnx(model=str(graph_path))),
            ('build', lambda: sdk.build(**BUILD)), ('export_rknn', lambda: sdk.export_rknn(str(output)))]:
            call = dict(operation=name, status='running'); report['calls'].append(call); progress(name + '_start'); tick = time.monotonic(); code = action()
            call.update(returncode=code, seconds=time.monotonic() - tick, status='returned'); require(code in (0, None), 'SDK ' + name + ' failed: ' + str(code))
        require(output.stat().st_size > 0, 'Empty RKNN'); report.update(status=SUCCESS, all_operations_succeeded=True, rknn=dict(file=output.name, bytes=output.stat().st_size, sha256=sha(output)))
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        if sdk is not None:
            report['sdk_release_called'] = True
            try: progress('SDK_release_start')
            except BaseException as error: report['cleanup_errors'].append('Pre-release receipt: ' + str(error))
            try:
                code = sdk.release(); report['sdk_release_code'] = code; require(code in (0, None), 'SDK release failed'); report['sdk_released'] = True
            except BaseException as error: report['cleanup_errors'].append('SDK release: ' + str(error))
        try: report['generated_files'] = [dict(file=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('result.json', 'result.tmp')]
        except BaseException as error: report['cleanup_errors'].append('Generated file metadata: ' + str(error))
        for label, action in [('cwd', lambda: os.chdir(old_cwd)), ('affinity', lambda: os.sched_setaffinity(0, old_affinity))]:
            try: action()
            except BaseException as error: report['cleanup_errors'].append(label + ': ' + str(error))
        os.environ.clear(); os.environ.update(old_env)
        for number, handler in handlers.items(): signal.signal(number, handler)
        try: report['source_and_inputs_unchanged'] = all(p.is_file() and sha(p) == digest for p, digest in watched.items())
        except BaseException as error: report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity: ' + str(error))
        report.update(cwd_restored=Path.cwd() == old_cwd, affinity_restored=set(os.sched_getaffinity(0)) == old_affinity, environment_restored=dict(os.environ) == old_env,
            pinned_files={str(p): digest for p, digest in watched.items()})
        if report['cleanup_errors'] or not all(report[k] for k in ('cwd_restored', 'affinity_restored', 'environment_restored', 'source_and_inputs_unchanged')) or (sdk is not None and not report['sdk_released']): report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS and report['sdk_released'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
