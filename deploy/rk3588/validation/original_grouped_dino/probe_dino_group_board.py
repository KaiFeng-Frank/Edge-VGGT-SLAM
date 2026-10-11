#!/usr/bin/env python3
"""Board-only four original FP16 DINO blocks versus their original group graph.

No model load, calibration, compiler or whole-model/trajectory acceptance. All
ABBAABBA calls, including first calls, use the frozen public NEON FP16 IO route.
Measured intervals include route, finite checks and owned copies; hashing and
NPY writes are outside those intervals. Captured arrays remain private.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
PARENT_SHA = 'd68d05f984688e2c793f2e9d8320531202a4f9871790d7c2d96c74b5f41c9325'
EXPORT_SHA = 'c4e3970ba332f18026670734413450c4bec7dd03d31ebe83d66e03813ad6236c'
CAPTURE_SHA = 'ca981ff8d95d2e2c37c94fde2a6f7d1a30c2f5d8568b127eb77f2bff18a6dbe1'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
NEON_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
SOURCE_PINS = {
    'vggt_slam/rknn_half_io_dino.py': 'e39469432aa5663f0a15756be0067ec92a0fb7127e6702a794298c3e902f3cd7',
    'vggt_slam/rknn_dino_blocks.py': '6cf230ea3ae6c65b9a3c7ec0b31e9abb346edb2fc0413b6b77a3a3b510b495de',
    'vggt_slam/rknn_half_io_pairs.py': '53b573f494f2598183659ba4e8dcb7503c770d9e2ac7b95cae9016279805d9a4',
    'vggt_slam/rknn_aggregator_pairs.py': '08270da911213d0d22c21c9a7f001aaa7e16d7f7b32220f8a2b0c1ddc4b09b88',
    'vggt_slam/rknn_aggregator_blocks.py': '4fea665334e9d12a3d55839c3150680e269bab9794074d698bed2ad0a6e41c93',
    'vggt_slam/native/fp16_neon.c': '77707b735a6291c417a997ed64acbe4e1cd60cc5f9dd150d83fe1836868bd5e3',
    'deploy/rk3588/validation/run-v12-rknn-block-board.py': '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608',
    'deploy/rk3588/reference/rknn_api_v2.3.2.h': 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a',
}
SHAPE = (1, 53, 1024)
ORDER = ('A', 'B', 'B', 'A', 'A', 'B', 'B', 'A')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''): h.update(part)
    return h.hexdigest()


def pin(path, expected):
    path = Path(path).resolve(strict=True)
    if not re.fullmatch('[0-9a-f]{64}', expected) or sha(path) != expected:
        raise ValueError(f'Frozen artifact SHA256 differs: {path}')
    return path


def artifact(directory, row):
    directory = Path(directory).resolve(strict=True)
    path = (directory / row['file']).resolve(strict=True)
    if not path.is_relative_to(directory) or path.stat().st_size != row['bytes']:
        raise ValueError('Artifact path/size differs')
    return pin(path, row['sha256'])


def array_info(np, value):
    if value.shape != SHAPE or value.dtype != np.float32 or not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValueError('Dense finite original [1,53,1024] FP32 tensor required')
    return dict(shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes,
                raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest(), finite=True)


def load_array(np, directory, row):
    value = np.load(artifact(directory, row), allow_pickle=False)
    info = array_info(np, value)
    if any(info[k] != row[k] for k in ('shape', 'dtype', 'tensor_bytes', 'raw_sha256')):
        raise ValueError('Captured tensor raw identity differs')
    return value


def compare(np, actual, reference):
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    delta = a - r; absolute = np.abs(delta); threshold = 1e-5 + 1e-4 * np.abs(r)
    denominator = float(np.linalg.norm(r)); error = float(np.linalg.norm(delta))
    return dict(bitwise_equal=actual.tobytes() == reference.tobytes(), max_abs=float(absolute.max()),
        relative_l2=error / denominator if denominator else (0. if error == 0 else None),
        reference_l2=denominator, strict_rtol=1e-4, strict_atol=1e-5,
        failed_elements=int((absolute > threshold).sum()), strict_pass=not bool((absolute > threshold).any()))


def route_stats():
    names = ('input_conversion', 'output_conversion', 'owned_half_copy', 'sdk_api', 'route',
             'rknn_inputs_set', 'rknn_run', 'rknn_outputs_get', 'rknn_outputs_release')
    half = {name + '_seconds': 0. for name in names}
    half.update(calls_attempted=0, calls_completed=0, input_conversion_calls=0, output_conversion_calls=0)
    return dict(sdk_run_attempts=0, sdk_runs_succeeded=0, sdk_io_seconds=0., cleanup_errors=[], half_io=half)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--parent-manifest', type=Path, default=BASE / 'experiments/vggt-v54-dino-graphs-r1/compiled.json')
    for name in ('group-export', 'group-compile', 'half-io-library', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--group-compile-sha256', required=True)
    args = p.parse_args()
    if (sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64')
            or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes()):
        p.error('Actual SDK execution requires the Linux RK3588S board')
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=False)
    affinity, old_path = os.sched_getaffinity(0), sys.path[:]
    contexts, routes, pins, api, base = [], {}, {}, None, None
    report = dict(schema='v58_original_DINO_four_blocks_vs_group_half_IO_probe', status='failed',
        script_sha256=sha(__file__), original_model_only=True, actual_execution_on_board_only=True,
        full_model_or_trajectory_or_realtime_validated=False, candidate_accepted=False,
        original_indices=[0, 1, 2, 3], shape=list(SHAPE), init_flags=0, core_mask=7,
        timeout_ms=30000, order=list(ORDER), hidden_warmup_calls=0, calls=[], context_queries=[],
        contexts_created=0, contexts_destroyed=0, cleanup_errors=[],
        timing_scope='four-block/group component route including conversion/SDK/finite checks/owned copies; excludes hashes/NPY IO')
    try:
        if 6 not in affinity: raise ValueError('CPU 6 required')
        os.sched_setaffinity(0, {6}); report['cpu_affinity'] = [6]
        repo = args.repo.resolve(strict=True)
        pins.update({repo / name: digest for name, digest in SOURCE_PINS.items()})
        for path, digest in pins.items(): pin(path, digest)
        spec = importlib.util.spec_from_file_location('_v58_group_ABI', repo / 'deploy/rk3588/validation/run-v12-rknn-block-board.py')
        base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
        report['board'] = base.require_board()
        sys.path.insert(0, str(repo))
        from vggt_slam.rknn_half_io_dino import RKNNHalfIOOriginalDino
        from vggt_slam.rknn_half_io_pairs import require_half_descriptor
        import numpy as np
        for name, digest in SOURCE_PINS.items():
            if name.endswith('.py') and name.startswith('vggt_slam/'):
                module = sys.modules[name[:-3].replace('/', '.')]
                if Path(module.__file__).resolve() != repo / name: raise ValueError('Runtime import escaped frozen source root')
        parent_path = pin(args.parent_manifest, PARENT_SHA); parent = json.loads(parent_path.read_text())
        export_path = pin(args.group_export, EXPORT_SHA); export = json.loads(export_path.read_text())
        compile_path = pin(args.group_compile, args.group_compile_sha256); compiled = json.loads(compile_path.read_text())
        capture_dir = Path(parent['original_reference_capture_directory']).resolve(strict=True)
        capture_path = pin(capture_dir / 'capture.json', CAPTURE_SHA); capture = json.loads(capture_path.read_text())
        checkpoint = dict(sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0', bytes=5026874952)
        if (parent['schema'] != 'v54_original_VGGT_24_DINO_block_RKNN_manifest' or parent['checkpoint'] != checkpoint
                or len(parent['blocks']) != 24 or [b['index'] for b in parent['blocks']] != list(range(24))
                or parent['capture_sha256'] != CAPTURE_SHA or capture['checkpoint'] != checkpoint
                or capture['status'] != 'completed_original_FP32_24_DINO_block_capture' or capture['cleanup_errors']):
            raise ValueError('Frozen original parent/capture contract differs')
        group = export['blocks'][0]
        if (export['schema'] != 'v58_original_VGGT_4_DINO_block_group_ONNX_manifest' or len(export['blocks']) != 1
                or export['original_parent_manifest_sha256'] != PARENT_SHA or export['capture_sha256'] != CAPTURE_SHA
                or export['checkpoint'] != checkpoint or export['source_hashes'] != parent['source_hashes']
                or group['indices'] != [0, 1, 2, 3] or group['boundary_mode'] != 'fp32'
                or group['parents'] != parent['blocks'][:4] or group['input_shape'] != list(SHAPE)
                or group['output_shape'] != list(SHAPE) or group['all_original_parent_nodes_attributes_and_initializer_payloads_retained'] is not True
                or group['original_FP32_parent_graph_math_preserved'] is not True):
            raise ValueError('Complete original four-block graph lineage differs')
        expected_refs = {tag: dict(input=parent['blocks'][0]['original_capture_references'][tag]['input'],
            output=parent['blocks'][3]['original_capture_references'][tag]['output']) for tag in ('single0', 'single1')}
        if any(group['original_capture_references'][tag] != row for tag, row in expected_refs.items()):
            raise ValueError('Original single-frame input/layer3 references differ')
        configuration = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
            mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
            remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
        if (compiled['status'] != 'compiled_not_quality_accepted' or compiled['mode'] != 'compile'
                or compiled['manifest_sha256'] != EXPORT_SHA or compiled['graph_sha256'] != group['sha256']
                or compiled['configuration'] != configuration or compiled['cleanup_errors']
                or [row['name'] for row in compiled['operations']] != ['config', 'load_onnx', 'build', 'export_rknn']
                or any(row['code'] != 0 for row in compiled['operations'])):
            raise ValueError('Ordinary unchanged nonquantized compile provenance differs')
        for relative, expected in parent['source_hashes'].items():
            path = (repo / relative).resolve(strict=True)
            if not path.is_relative_to(repo): raise ValueError('Original source path escapes selected root')
            pins[path] = expected; pin(path, expected)
        pins.update({parent_path: PARENT_SHA, export_path: EXPORT_SHA, compile_path: args.group_compile_sha256,
                     capture_path: CAPTURE_SHA})
        artifact(export_path.parent, group)
        model_paths = [('A' + str(i), artifact(parent_path.parent, parent['blocks'][i]['rknn'])) for i in range(4)]
        model_paths.append(('B', artifact(compile_path.parent, compiled['rknn'])))
        neon_path = pin(args.half_io_library, NEON_SHA); sdk_path = pin('/usr/lib/librknnrt.so', LIBRARY_SHA)
        pins.update({neon_path: NEON_SHA, sdk_path: LIBRARY_SHA})
        api, neon = base.bind_api(ct.CDLL(str(sdk_path))), ct.CDLL(str(neon_path))
        for name in ('f32_to_f16', 'f16_to_f32'):
            function = getattr(neon, name); function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]; function.restype = None
        report.update(checkpoint=checkpoint, parent_manifest_sha256=PARENT_SHA, export_manifest_sha256=EXPORT_SHA,
            compile_manifest_sha256=args.group_compile_sha256, capture_sha256=CAPTURE_SHA,
            runtime_source_sha256=SOURCE_PINS, library_sha256=LIBRARY_SHA, neon_library_sha256=NEON_SHA,
            original_reference_pins=expected_refs, boundary_rounding_preserved_by_group_not_assumed=True,
            group_compile_original_FP32_gate_passed=compiled.get('original_FP32_gate_passed'),
            group_compile_check_receipt_sha256=compiled.get('check_receipt_sha256'))
        for name, path in model_paths:
            context = base.CONTEXT(); contexts.append((name, context))
            filename = ct.create_string_buffer(str(path).encode())
            code = api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
            report['contexts_created'] += int(bool(context.value)); base.check_call(code, 'rknn_init')
            if not context.value: raise RuntimeError('Empty RKNN context')
            base.check_call(api.rknn_set_core_mask(context, 7), 'rknn_set_core_mask')
            version = base.query(api, context, 5, base.SDKVersion())
            if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()): raise ValueError('Runtime must remain 2.3.2')
            counts = base.query(api, context, 0, base.InputOutputNum())
            if (counts.n_input, counts.n_output) != (1, 1): raise ValueError('Exactly one original input/output required')
            attrs = [base.attr_record(base.query(api, context, command, base.TensorAttr(index=0))) for command in (1, 2)]
            for row, binding in zip(attrs, ('input', 'output')): require_half_descriptor(row, SHAPE, binding, 0)
            stats = route_stats(); routes[name] = SimpleNamespace(base=base, api=api, np=np, neon=neon, timeout=30000, stats=stats, context=context)
            report['context_queries'].append(dict(role=name, model_sha256=sha(path), model_bytes=path.stat().st_size,
                api_version=bytes(version.api_version).decode(), driver_version=bytes(version.drv_version).decode(), input=attrs[0], output=attrs[1]))
        for tag, references in expected_refs.items():
            original = load_array(np, capture_dir, references['input']); cpu = load_array(np, capture_dir, references['output'])
            first_a, seen = None, set()
            for sequence, role in enumerate(ORDER):
                value = original; begin = time.perf_counter()
                for key in ([f'A{i}' for i in range(4)] if role == 'A' else ['B']):
                    route = routes[key]; value = RKNNHalfIOOriginalDino._run(route, route.context, value, 3, SHAPE)
                    if value.shape != SHAPE or value.dtype != np.float32 or not value.flags.owndata or not np.isfinite(value).all():
                        raise ValueError('Owned finite original FP32 route output required')
                elapsed = time.perf_counter() - begin
                if first_a is None:
                    if role != 'A': raise RuntimeError('ABBA must start with A')
                    first_a = value.copy()
                file = output_dir / f'{tag}-{sequence:02d}-{role}.npy'
                with file.open('xb') as stream: np.save(stream, value, allow_pickle=False)
                info = array_info(np, value)
                report['calls'].append(dict(input=tag, sequence=sequence, role=role, first_for_input_role=role not in seen,
                    component_seconds=elapsed, sdk_calls=4 if role == 'A' else 1,
                    original_FP32_reference=compare(np, value, cpu), same_input_first_A=compare(np, value, first_a),
                    artifact=dict(file=file.name, bytes=file.stat().st_size, sha256=sha(file), **info)))
                seen.add(role)
                if array_info(np, original)['raw_sha256'] != references['input']['raw_sha256']:
                    raise ValueError('Frozen original input changed')
        report['subsequent_summary'] = {}
        for role in ('A', 'B'):
            rows = [row for row in report['calls'] if row['role'] == role and not row['first_for_input_role']]
            if len(rows) != 6: raise ValueError('Exactly six subsequent calls per role required')
            total = 0.
            for row in rows: total += row['component_seconds']
            report['subsequent_summary'][role] = dict(count=6, mean_component_seconds=total / 6)
        if sum(route.stats['sdk_runs_succeeded'] for route in routes.values()) != 40:
            raise ValueError('Exactly 40 successful SDK runs required')
        report['status'] = 'completed_component_diagnostic_not_adopted'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['route_stats'] = {name: route.stats for name, route in routes.items()}
        for name, context in reversed(contexts):
            if context.value and api is not None:
                try:
                    base.check_call(api.rknn_destroy(context), 'rknn_destroy'); report['contexts_destroyed'] += 1
                except BaseException as exc: report['cleanup_errors'].append(f'{name}: {type(exc).__name__}: {exc}')
                finally: context.value = 0
        report['live_contexts'] = sum(bool(context.value) for _, context in contexts)
        report['cleanup_errors'] += [error for route in routes.values() for error in route.stats['cleanup_errors']]
        try:
            if any(sha(path) != digest for path, digest in pins.items()): raise ValueError('Frozen source/input identity changed')
            report['frozen_pins_unchanged'] = True
        except BaseException as exc: report['cleanup_errors'].append(f'identity: {exc}')
        try:
            os.sched_setaffinity(0, affinity); report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            if not report['affinity_restored']: raise ValueError('Affinity restore failed')
        except BaseException as exc: report['cleanup_errors'].append(f'affinity: {exc}')
        sys.path[:] = old_path
        if report['cleanup_errors']: report['status'] = 'failed'
        (output_dir / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], result_sha256=sha(output_dir / 'result.json'))), flush=True)
    return 0 if report['status'] == 'completed_component_diagnostic_not_adopted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
