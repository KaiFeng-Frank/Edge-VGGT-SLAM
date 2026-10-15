#!/usr/bin/env python3
"""Board-only original native DINO0 compile; sole CONFIG change: level 3 -> 0.

Reuse the pinned completed FP32 graph gate. No Torch/ORT/runtime inference.
The unchanged compiler owns SDK build/release and process-setting restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

BASE_SHA = 'ade8f932c267bdc2a9ff5854e305d87456984630025aa34c0c448499f5a6a81b'
ONNX_SHA = 'd8ce26c2f9cb3018b9a24da9f722937569067a5cffde650288f0073743e547e8'
EXPORT_SHA = 'bd78874ccc7373121696dea581ad88288425cd3878aaadf02a8ae01fd563d532'
CHECK_SHA = '4bd4d2935c8db2f1eb5ac604c39d46d323fcbdd85c4907141e91679ef90c8cf8'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
OUTPUT_DIR = Path('/tmp/vggt-v117-opt0-compile-r1')
BASE_CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)
ENV_KEYS = ('TMPDIR', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def pin(path, digest):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and sha(path) == digest, 'Pinned file differs: ' + str(path))
    return path


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-source', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--export-receipt', type=Path, required=True)
    parser.add_argument('--fp32-check', type=Path, required=True,
        help='Exact frozen v99 JSON; a byte-identical metadata copy is allowed')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    require(platform.machine() == 'aarch64'
        and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
        'Only the actual RK3588 board may compile this candidate')
    out = args.output_dir.resolve()
    require(out == OUTPUT_DIR, 'Keep every compile artifact in the fixed private /tmp directory')
    out.mkdir(parents=True, exist_ok=False)
    wrapper_source = Path(__file__).resolve(strict=True)
    result_path = out / 'result.json'
    result = dict(schema='v117_original_native518_DINO0_OPT0_FP16_compile', status='running',
        script_sha256=sha(wrapper_source), base_source_sha256=BASE_SHA,
        original_onnx_sha256=ONNX_SHA, export_receipt_sha256=EXPORT_SHA,
        reused_FP32_check_sha256=CHECK_SHA, configuration_delta={'optimization_level': {'before': 3, 'after': 0}},
        original_ONNX_modified=False, original_model_only=True, original_layers_or_tokens_omitted=False,
        Torch_ORT_runtime_inference_calls=0, calibration_executed=False, SDK_compile_delegated=False,
        NPU_runtime_executed=False, component_only=True, candidate_accepted=False,
        full_model_or_SLAM_or_realtime_validated=False, system_or_frequency_writes=False,
        output_dir=str(out), cleanup_errors=[], pins_unchanged=False,
        cwd_restored=False, environment_restored=False)
    save(result_path, result)
    old_cwd = Path.cwd()
    old_env = {name: os.environ.get(name) for name in ENV_KEYS}
    module_name = '_v117_original_frozen_native_compiler'
    helper = None
    module_loaded = False
    original_config = None
    started = time.monotonic()
    try:
        base = pin(args.base_source, BASE_SHA)
        graph = pin(args.onnx, ONNX_SHA)
        export_path = pin(args.export_receipt, EXPORT_SHA)
        check_path = pin(args.fp32_check, CHECK_SHA)
        pins = {str(base): BASE_SHA, str(graph): ONNX_SHA,
            str(export_path): EXPORT_SHA, str(check_path): CHECK_SHA, str(wrapper_source): result['script_sha256']}
        export = json.loads(export_path.read_text())
        check = json.loads(check_path.read_text())
        require(export['schema'] == 'v99_native518_original_DINO0_ONNX_export'
            and export['status'] == 'exported_original_native518_DINO0_not_FP32_accepted'
            and export['checkpoint'] == {'sha256': CHECKPOINT_SHA, 'bytes': 5026874952}
            and export['role'] == 'dino' and export['index'] == 0
            and export['logical_shape'] == [1, 1041, 1024]
            and export['onnx']['sha256'] == ONNX_SHA and export['onnx']['bytes'] == graph.stat().st_size
            and export['parameter_count'] == 14 and export['all_original_parameter_values_preserved'] is True
            and export['source_and_inputs_unchanged'] is True and export['verification_errors'] == [],
            'Original complete native DINO0 export lineage differs')
        require(check['status'] == 'passed_complete_block_FP32_numerical_gate'
            and check['role'] == 'dino' and check['onnx_sha256'] == ONNX_SHA
            and check['capture_sha256'] == export['capture_sha256'] and check['whole_block'] is True
            and check['shape'] == [1, 1041, 1024]
            and check['thresholds'] == {'relative_l2': 1e-5, 'max_abs': 1e-3}
            and math.isfinite(check['relative_l2']) and 0 <= check['relative_l2'] <= 1e-5
            and math.isfinite(check['max_abs']) and 0 <= check['max_abs'] <= 1e-3,
            'Pinned complete original FP32 graph gate did not pass')
        result.update(pinned_files=pins, original_parameters=export['blocks'][0]['parameters'],
            checkpoint=export['checkpoint'], source_hashes=export['source_hashes'],
            original_FP32_gate_reused=True, reused_FP32_check=check,
            original_FP32_gate_rerun=False, graph_bytes=graph.stat().st_size,
            free_bytes_before_compile=shutil.disk_usage(out).free)
        require(result['free_bytes_before_compile'] >= 1 << 30, 'At least 1 GiB free /tmp filesystem space required')
        require(module_name not in sys.modules, 'Private helper module already loaded')
        spec = importlib.util.spec_from_file_location(module_name, base)
        require(spec is not None and spec.loader is not None, 'Cannot load pinned compiler')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = helper
        module_loaded = True
        spec.loader.exec_module(helper)
        require(helper.CONFIG == BASE_CONFIG and helper.SHAPES['dino'] == [1, 1041, 1024],
            'Frozen compiler configuration or shape differs')
        original_config = helper.CONFIG
        helper.CONFIG = dict(original_config, optimization_level=0)
        require({key for key in BASE_CONFIG if helper.CONFIG[key] != BASE_CONFIG[key]} == {'optimization_level'},
            'Exactly one compiler setting must change')
        result.update(configuration_before=BASE_CONFIG, configuration=helper.CONFIG,
            build_options=BUILD, private_compile_cwd=str(out), private_TMPDIR=str(out),
            delegated_verbose_setting='unchanged original RKNN(verbose=True, verbose_file=build/sdk.log)')
        os.chdir(out)
        os.environ['TMPDIR'] = str(out)
        result['SDK_compile_delegated'] = True
        save(result_path, result)
        code = helper.main(['--onnx', str(graph), '--onnx-sha256', ONNX_SHA, '--role', 'dino',
            '--output-dir', str(out / 'build'), '--cpu-affinity', '4', '6'])
        child_path = out / 'build' / 'result.json'
        child = json.loads(child_path.read_text())
        result.update(delegated_returncode=code, returned_receipt=child,
            returned_receipt_file='build/result.json', returned_receipt_sha256=sha(child_path))
        require(code == 0 and child['status'] == 'compiled_not_quality_accepted'
            and child['script_sha256'] == BASE_SHA and child['configuration'] == helper.CONFIG
            and child['build_options'] == BUILD and child['original_ONNX_modified'] is False
            and child['onnx']['sha256'] == ONNX_SHA and child['role'] == 'dino'
            and child['logical_shape'] == [1, 1041, 1024]
            and child['graph_node_count'] == 31 and child['graph_initializer_count'] == 23
            and child['sdk_released'] is True and child['cleanup_errors'] == []
            and child['pins_unchanged'] is True and child['affinity_restored'] is True
            and child['environment_restored'] is True, 'Delegated compile or cleanup failed')
        model = child_path.parent / child['rknn']['file']
        require(model.resolve().parent == out / 'build' and model.stat().st_size == child['rknn']['bytes'],
            'Compiled model location or byte count differs')
        pin(model, child['rknn']['sha256'])
        result['rknn'] = dict(child['rknn'], file=str(model.relative_to(out)))
        require(all(sha(path) == digest for path, digest in pins.items()), 'Frozen inputs changed during compile')
        result.update(pins_unchanged=True, status='compiled_original_DINO0_OPT0_not_quality_accepted')
    except BaseException as error:
        result.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
    finally:
        if helper is not None and original_config is not None:
            helper.CONFIG = original_config
        if module_loaded:
            sys.modules.pop(module_name, None)
        try:
            os.chdir(old_cwd)
            result['cwd_restored'] = Path.cwd() == old_cwd
        except BaseException as error:
            result['cleanup_errors'].append('cwd restore: ' + str(error))
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        result['environment_restored'] = all(os.environ.get(key) == value for key, value in old_env.items())
        if result['cleanup_errors'] or not result['cwd_restored'] or not result['environment_restored']:
            result['status'] = 'failed_cleanup'
        result['total_seconds'] = time.monotonic() - started
        save(result_path, result)
        print(json.dumps({key: result.get(key) for key in ('status', 'error', 'rknn', 'cleanup_errors')}), flush=True)
    return 0 if result['status'] == 'compiled_original_DINO0_OPT0_not_quality_accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
