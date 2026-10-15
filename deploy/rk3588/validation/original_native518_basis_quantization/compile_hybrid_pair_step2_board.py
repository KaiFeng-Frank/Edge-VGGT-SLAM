#!/usr/bin/env python3
"""Board-only step2/export of a real-generated manual precision policy.

The policy must be produced by inspecting the actual step1 model/config, not
by assuming public tensor names are internal selectors. Accepted config edits
do not prove accelerator precision placement. No runtime inference is called.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time
import traceback

STEP1_SOURCE_SHA = 'a9b6f6b591e7ec7fef39f04ece167115445f445eecd87f996cdb5921035a20f6'
SELECTOR_SOURCE_SHA = '25db1ca6c66fcea6d84b921c3f8f8382d0d16eeaa9d8f6f75d7ae5b9b804da72'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3, mean_values=None,
    std_values=None, quant_img_RGB2BGR=False, model_pruning=False, remove_reshape=False,
    quantize_weight=False, enable_flash_attention=False, quantized_dtype='w8a8',
    quantized_algorithm='normal', quantized_method='channel')
SUCCESS = 'compiled_manual_hybrid_pair_not_runtime_or_quality_accepted'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step1-receipt', type=Path, required=True)
    p.add_argument('--step1-sha256', required=True)
    p.add_argument('--selector-policy', type=Path, required=True)
    p.add_argument('--selector-policy-sha256', required=True)
    p.add_argument('--selector-source', type=Path, default=Path(__file__).with_name('select_hybrid_precision_policy.py'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--minimum-free-mib', type=int, default=1024)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S-board-only SDK worker')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False); (out / 'tmp').mkdir()
    old_env, old_cwd, old_affinity = dict(os.environ), Path.cwd(), set(os.sched_getaffinity(0))
    source = Path(__file__).resolve()
    watched, handlers, sdk = {source: sha(source)}, {}, None; started = time.monotonic()
    report = dict(schema='v128_original_native518_basis_pair_manual_hybrid_step2', status='running', script_sha256=watched[source],
        requested_policy_only=True, protected_precision_placement_established=False, eligible_interior_INT8_established=False,
        runtime_inference_invoked=False, additional_calibration_invoked=False, candidate_accepted=False,
        original_ONNX_modified=False, system_or_frequency_writes=False, calls=[], cleanup_errors=[], sdk_released=False)

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        t = out / 'result.tmp'; t.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); t.replace(out / 'result.json')
        print(stage, flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(len(digest) == 64 and sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
        watched[path] = digest; return path

    def artifact(folder, row):
        require(not Path(row['file']).is_absolute() and '..' not in Path(row['file']).parts, 'Owned relative artifact required')
        return pin(folder / row['file'], row['sha256'], row['bytes'])

    progress('preflight')
    try:
        def interrupted(number, frame): raise InterruptedError('Controller signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require({4, 5, 6, 7} <= old_affinity, 'A76 affinity required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', TMPDIR=str(out / 'tmp'))
        os.chdir(out)
        sfile, pfile = pin(a.step1_receipt, a.step1_sha256), pin(a.selector_policy, a.selector_policy_sha256)
        mapper_file = pin(a.selector_source, SELECTOR_SOURCE_SHA)
        step1, policy = json.loads(sfile.read_text()), json.loads(pfile.read_text())
        require(step1['schema'] == 'v128_original_native518_basis_pair_manual_hybrid_step1' and
                step1['status'] == 'generated_manual_hybrid_step1_not_compiled_or_accepted' and step1['script_sha256'] == STEP1_SOURCE_SHA and
                step1['all_SDK_operations_succeeded'] and step1['sdk_released'] and step1['sdk_release_code'] in (0, None) and
                all(step1[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'cwd_restored',
                    'all18_new_FP32_calls_passed', 'calibration_evaluation_disjoint', 'heldouts_excluded_from_dataset')) and
                not step1['cleanup_errors'], 'Actual clean original step1/gates required')
        require(step1['configuration'] == CONFIG and step1['variant'] in ('A', 'F', 'B') and
            set(step1['original_parameters']) == {'frame', 'global'} and
            all(len(step1['original_parameters'][role]) == 18 for role in ('frame', 'global')) and
            len(step1['three_clean_capture_check_receipts']) == 3 and
            [r['id'] for r in step1['calibration_records']] == ['pair0', 'pair1', 'pair2', 'pair3'] and
            [r['id'] for r in step1['heldout_records']] == ['pair4', 'pair5'] and step1['dataset']['line_count'] == 4,
            'Original configuration, full parameters or actual split differs')
        require(policy['schema'] == 'v128_real_generated_hybrid_selector_policy' and policy['variant'] == step1['variant'] and
                policy['step1_receipt_sha256'] == a.step1_sha256 and policy['graph_sha256'] == step1['graph']['sha256'] and
                policy['script_sha256'] == SELECTOR_SOURCE_SHA and policy['selectors_verified_against_actual_generated_model'] and
                policy['only_custom_quantize_layers_changed'], 'Actual generated semantic policy required')
        generated = policy['generated_inputs']
        paths = {name: artifact(sfile.parent if name != 'edited_cfg' else pfile.parent, generated[name])
                 for name in ('model', 'data', 'original_cfg', 'edited_cfg')}
        for name in ('model', 'data', 'original_cfg'):
            require(generated[name] in step1['generated_files'], 'Real SDK-generated input identity differs')
        require(policy['requested_float16_layers'] and all(dtype == 'float16' for dtype in policy['requested_float16_layers'].values()), 'Documented FLOAT16 selectors required')
        import yaml
        original = yaml.safe_load(paths['original_cfg'].read_text()); edited = yaml.safe_load(paths['edited_cfg'].read_text())
        require(isinstance(original, dict) and isinstance(edited, dict) and
                {k: v for k, v in original.items() if k != 'custom_quantize_layers'} ==
                {k: v for k, v in edited.items() if k != 'custom_quantize_layers'} and
                edited['custom_quantize_layers'] == policy['requested_float16_layers'], 'Calibration parameters/config changed beyond selectors')
        require(original.get('custom_quantize_layers') == {}, 'Edited original generated config is not allowed')
        original_bytes, edited_bytes = paths['original_cfg'].read_bytes(), paths['edited_cfg'].read_bytes()
        header, separator, quantization_section = original_bytes.partition(b'\n')
        require(header == b'custom_quantize_layers: {}' and separator and quantization_section.startswith(b'quantize_parameters:') and
            edited_bytes.endswith(quantization_section) and policy['original_quantization_section_byte_preserved'] and
            hashlib.sha256(quantization_section).hexdigest() == policy['original_quantization_section_sha256'], 'Original calibration text changed')
        sem_path = artifact(sfile.parent, step1['public_semantic_index']); semantic = json.loads(sem_path.read_text())
        require(policy['public_semantic_index'] == step1['public_semantic_index'] and
            semantic['variant'] == step1['variant'] and semantic['graph_sha256'] == step1['graph']['sha256'], 'Original semantic index differs')
        spec = importlib.util.spec_from_file_location('v128_actual_selector_mapping', mapper_file)
        mapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(mapper)
        import onnx
        metadata_model = onnx.load_model(paths['model'], load_external_data=False)
        actual_mapping = mapper.derive_mapping(metadata_model, original, semantic); del metadata_model
        require(all(policy.get(k) == value for k, value in actual_mapping.items()), 'Selectors or roles do not match actual generated graph/catalogue')
        report.update(variant=step1['variant'], configuration=step1['configuration'], checkpoint=step1['checkpoint'],
            original_parameters=step1['original_parameters'], graph=step1['graph'], input_bindings=step1['input_bindings'], output_bindings=step1['output_bindings'],
            step1_receipt=dict(file=str(sfile), sha256=a.step1_sha256, script_sha256=STEP1_SOURCE_SHA),
            selector_policy=dict(file=str(pfile), sha256=a.selector_policy_sha256), generated_inputs=generated,
            selector_source_sha256=SELECTOR_SOURCE_SHA, selector_count=len(actual_mapping['requested_float16_layers']),
            actual_generated_selector_mapping_recomputed=True, original_quantization_section_byte_preserved=True,
            residual_role_mapping=actual_mapping['residual_role_mapping'], basis_role_mapping=actual_mapping['basis_role_mapping'],
            requested_float16_layers=actual_mapping['requested_float16_layers'], update_branch_inputs_selected=False,
            three_clean_capture_check_receipts=step1['three_clean_capture_check_receipts'],
            placement_evidence_required='Actual compiled log/model dump or operator dtype plus logical descriptors; config acceptance alone is insufficient.')
        require(importlib.metadata.version('rknn-toolkit2').split('+')[0] == '2.3.2', 'Toolkit2.3.2 required')
        report['versions'] = dict(python=platform.python_version(), toolkit=importlib.metadata.version('rknn-toolkit2'), onnx=onnx.__version__)
        free = shutil.disk_usage(out).free
        require(a.minimum_free_mib > 0 and free >= a.minimum_free_mib * (1 << 20), 'Private output free-space gate failed')
        report.update(pre_SDK_free_bytes=free, minimum_free_bytes=a.minimum_free_mib * (1 << 20),
            SDK_verbose=True, verbose_log_file='sdk.log', verbose_is_diagnostic_logging_only=True)
        progress('actual_policy_and_files_verified_before_SDK_import')
        import rknn.api as api
        sdk = api.RKNN(verbose=True, verbose_file=str(out / 'sdk.log'))
        model = out / ('pair-' + step1['variant'] + '-hybrid.rknn')
        for name, action in [('config', lambda: sdk.config(**step1['configuration'])),
            ('hybrid_quantization_step2', lambda: sdk.hybrid_quantization_step2(model_input=str(paths['model']), data_input=str(paths['data']), model_quantization_cfg=str(paths['edited_cfg']))),
            ('export_rknn', lambda: sdk.export_rknn(str(model)))]:
            call = dict(operation=name, status='running'); report['calls'].append(call); progress(name + '_start')
            tick = time.monotonic(); code = action(); call.update(returncode=code, seconds=time.monotonic() - tick, status='returned')
            require(code in (0, None), 'SDK ' + name + ' failed: ' + str(code))
        require(model.stat().st_size > 0, 'Empty compiled RKNN')
        report.update(status=SUCCESS, all_SDK_operations_succeeded=True, rknn=dict(file=model.name, bytes=model.stat().st_size, sha256=sha(model)))
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        if sdk is not None:
            report['sdk_release_called'] = True
            try: progress('SDK_release_start')
            except BaseException as error: report['cleanup_errors'].append('Pre-release receipt: ' + str(error))
            try:
                code = sdk.release(); report['sdk_release_code'] = code
                require(code in (0, None), 'SDK release failed'); report['sdk_released'] = True
            except BaseException as error: report['cleanup_errors'].append('SDK release: ' + str(error))
        try:
            report['generated_files'] = [dict(file=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha(path))
                for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ('result.json', 'result.tmp')]
            report['actual_precision_log_or_dump_available'] = any(row['file'] == 'sdk.log' for row in report['generated_files'])
        except BaseException as error: report['cleanup_errors'].append('Generated file metadata: ' + str(error))
        for label, action in [('cwd', lambda: os.chdir(old_cwd)), ('affinity', lambda: os.sched_setaffinity(0, old_affinity))]:
            try: action()
            except BaseException as error: report['cleanup_errors'].append(label + ': ' + str(error))
        os.environ.clear(); os.environ.update(old_env)
        for number, handler in handlers.items(): signal.signal(number, handler)
        try: report['source_and_inputs_unchanged'] = all(p.is_file() and sha(p) == digest for p, digest in watched.items())
        except BaseException as error:
            report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity: ' + str(error))
        report.update(cwd_restored=Path.cwd() == old_cwd, affinity_restored=set(os.sched_getaffinity(0)) == old_affinity,
            environment_restored=dict(os.environ) == old_env, pinned_files={str(p): digest for p, digest in watched.items()})
        if report['cleanup_errors'] or not all(report[k] for k in ('cwd_restored', 'affinity_restored', 'environment_restored', 'source_and_inputs_unchanged')) or (sdk is not None and not report['sdk_released']):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS and report['sdk_released'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
