#!/usr/bin/env python3
"""Board-only ORT CPU check of a captured original VGGT aggregator block.

The explicit exporter/profile/export hashes bind the original CPU reference,
weights, positions and RoPE tables. Run from /mnt/vggt-data/vggt-slam2 using the
isolated ARM ORT environment. Four calls (one cold, three warm) save their actual
outputs. This validates one exported graph/input, not SLAM accuracy or NPU use.
No Torch model, image decoder or full VGGT inference is invoked here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
HERE = Path(__file__).resolve().parent
EXPORTERS = ('export-v17-aggregator-block.py', 'export-v17-aggregator-block-r2.py',
             'export-v17-aggregator-block-r3.py')
RTOL = 1e-4
ATOL = 1e-5
CALLS = 4
ARCHITECTURE_FIELDS = ('num_heads', 'head_dim', 'norm1_eps', 'norm2_eps',
                       'q_norm_eps', 'k_norm_eps', 'gelu_approximate',
                       'rope_frequency', 'rope_scaling_factor')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha_argument(value):
    if not re.fullmatch('[0-9a-f]{64}', value):
        raise argparse.ArgumentTypeError('Supply a complete lowercase SHA256')
    return value


def require_sha(path, expected):
    sha_argument(expected)
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Explicit SHA mismatch: {Path(path).name}')
    return actual


def require_board():
    if (platform.system() != 'Linux' or platform.machine() != 'aarch64'
            or Path.cwd().resolve() != BASE
            or Path(__file__).resolve().parent != BASE / 'src/deploy/rk3588/validation'):
        raise ValueError('Run the deployed script only on the RK3588S board from /mnt/vggt-data/vggt-slam2')


def load_exporter(filename, expected_sha):
    if filename not in EXPORTERS:
        raise ValueError('Only the independently recorded r1/r2/r3 aggregator exporters are allowed')
    path = HERE / filename
    require_sha(path, expected_sha)
    spec = importlib.util.spec_from_file_location('_v17_ort_capture_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This helper imports only NumPy/stdlib: ONNX construction is not called.
    return module


def check_array(array, label, shape, np):
    if (not isinstance(array, np.ndarray) or array.shape != shape
            or array.dtype != np.float32):
        raise ValueError(f'{label} must have fixed float32 shape {shape}')
    if not array.flags.c_contiguous or not np.isfinite(array).all():
        raise ValueError(f'{label} must be C-contiguous and finite')


def raw_sha(array):
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def compare(actual, reference, np):
    """Same predefined FP32 gate/metrics as v12, with this captured shape."""
    shape = tuple(reference.shape)
    check_array(reference, 'Original CPU reference', shape, np)
    check_array(actual, 'ORT output', shape, np)
    a = actual.astype(np.float64)
    r = reference.astype(np.float64)
    difference = a - r
    absolute = np.abs(difference)
    tolerance = ATOL + RTOL * np.abs(r)
    an, rn = float(np.linalg.norm(a)), float(np.linalg.norm(r))
    return dict(finite=True, shape=list(shape), dtype='float32',
                max_abs=float(absolute.max()), mean_abs=float(absolute.mean()),
                rmse=float(np.sqrt(np.mean(difference * difference))),
                relative_l2=float(np.linalg.norm(difference) / max(rn, 1e-30)),
                cosine=float(np.sum(a * r) / (an * rn)) if an > 0 and rn > 0 else None,
                allclose=bool(np.all(absolute <= tolerance)),
                outside_tolerance_elements=int(np.count_nonzero(absolute > tolerance)),
                max_tolerance_ratio=float(np.max(absolute / tolerance)))


def check_export(export, exporter, args, profile, row, shape):
    required = dict(status='exported_and_structurally_checked',
                    script_sha256=args.exporter_sha256, profile_sha256=args.profile_sha256,
                    block=args.block, input_shape=list(shape), input_name='input', output_name='output',
                    original_weights=True, parameter_count=18, position_and_tables_fixed=True,
                    host_trigonometry_recomputed=False, host_model_inference_executed=False,
                    host_ORT_executed=False, external_data=False, external_data_used=False)
    if any(export.get(key) != value for key, value in required.items()):
        raise ValueError('Export receipt does not bind the original capture/static single-file graph')
    if (export.get('capture_artifacts') != row['artifacts']
            or export.get('captured_array_metadata') != row['arrays']
            or export.get('captured_parameter_metadata') != row['weights']
            or export.get('source_pins') != exporter.SOURCE_PINS
            or export.get('frozen_helpers') != exporter.GRAPH_HELPERS
            or export.get('architecture') != {key: row[key] for key in ARCHITECTURE_FIELDS}
            or export.get('selected_pair') != profile['selected_pair']
            or export.get('checkpoint') != profile['checkpoint']):
        raise ValueError('Export weights/positions/architecture/source/input identity differs from capture')
    if set(export.get('artifacts', {})) != {'input.npy', 'output.npy'}:
        raise ValueError('Exactly the two bound input/output NPY artifacts are required')
    for key in ('input', 'output'):
        entry = export['artifacts'][key + '.npy']
        if (entry.get('file') != key + '.npy' or entry.get('shape') != list(shape)
                or entry.get('dtype') != 'float32'
                or entry.get('raw_sha256') != row['arrays'][key]['sha256']):
            raise ValueError('Export input/reference metadata differs from original CPU capture')
    if export.get('onnx', {}).get('file') != 'block.onnx':
        raise ValueError('Expected the bound single-file block.onnx artifact')


def prepare_inputs(args, np):
    exporter = load_exporter(args.exporter_name, args.exporter_sha256)
    profile, row, arrays, weights, pins = exporter.load_capture(
        args.capture_dir, args.profile_sha256, args.block)
    del weights  # ORT needs only the graph and the exact captured input/reference.
    pins = dict(pins)
    pins[str((HERE / args.exporter_name).resolve())] = args.exporter_sha256
    pins[str(Path(__file__).resolve())] = sha(__file__)
    require_sha(args.export_receipt, args.export_sha256)
    pins[str(args.export_receipt.resolve())] = args.export_sha256
    export = json.loads(args.export_receipt.read_text())
    shape = tuple(arrays['input'].shape)
    check_array(arrays['input'], 'Captured block input', shape, np)
    check_array(arrays['output'], 'Captured CPU reference', shape, np)
    check_export(export, exporter, args, profile, row, shape)
    for name, expected in exporter.GRAPH_HELPERS.items():
        path = HERE / name
        require_sha(path, expected)
        pins[str(path.resolve())] = expected
    directory = args.export_receipt.parent
    exported = {}
    for key in ('input', 'output'):
        entry = export['artifacts'][key + '.npy']
        path = directory / entry['file']
        require_sha(path, entry['sha256'])
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Export NPY file size mismatch')
        value = np.load(path, allow_pickle=False)
        check_array(value, 'Exported ' + key, shape, np)
        if (raw_sha(value) != entry['raw_sha256']
                or not np.array_equal(value, arrays[key])):
            raise ValueError('Export NPY values differ from original CPU capture')
        exported[key] = value
        pins[str(path.resolve())] = entry['sha256']
    model = directory / export['onnx']['file']
    require_sha(model, export['onnx']['sha256'])
    model_bytes = model.read_bytes()
    if (len(model_bytes) != export['onnx']['bytes']
            or hashlib.sha256(model_bytes).hexdigest() != export['onnx']['sha256']):
        raise ValueError('ONNX size/hash changed before session construction')
    pins[str(model.resolve())] = export['onnx']['sha256']
    evidence = dict(profile_sha256=args.profile_sha256, exporter_name=args.exporter_name,
                    exporter_sha256=args.exporter_sha256,
                    export_sha256=args.export_sha256, block=args.block, input_shape=list(shape),
                    original_input_and_reference_exact=True, onnx=export['onnx'],
                    checkpoint=profile['checkpoint'], selected_pair=profile['selected_pair'],
                    architecture=export['architecture'], source_pins=export['source_pins'],
                    frozen_helpers=export['frozen_helpers'], capture_artifacts=row['artifacts'],
                    captured_array_metadata=row['arrays'], export_artifacts=export['artifacts'],
                    no_external_data=True,
                    external_data_check='SHA-bound exporter declaration and byte-only ORT model construction')
    return exported['input'], exported['output'], model_bytes, pins, evidence


def configure_cpu():
    allowed = set(os.sched_getaffinity(0))
    if not {6, 7}.issubset(allowed):
        raise ValueError(f'CPU 6,7 must be allowed; actual {sorted(allowed)}')
    os.sched_setaffinity(0, {6, 7})
    enforced = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2'}
    before = {key: os.environ.get(key) for key in enforced}
    os.environ.update(enforced)
    return before, dict(enforced)


def check_session(session, shape):
    session.disable_fallback()
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Expected only CPUExecutionProvider')
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('Exactly one graph input and one output required')
    result = {}
    for name, node in (('input', inputs[0]), ('output', outputs[0])):
        if (node.name != name or node.type != 'tensor(float)' or node.shape != list(shape)
                or any(type(n) is not int or n <= 0 for n in node.shape)):
            raise ValueError(f'Graph {name} must have exact name and fixed FP32 shape {shape}')
        result[name] = dict(name=node.name, type=node.type, shape=node.shape)
    return result


def execute_runs(session, tokens, reference, directory, receipt, np):
    previous = None
    for index in range(CALLS):
        timer = time.perf_counter()
        result = session.run(['output'], {'input': tokens})
        seconds = time.perf_counter() - timer
        receipt['successful_session_run_count'] += 1
        if not isinstance(result, list) or len(result) != 1:
            raise ValueError('ORT returned an unexpected output count/type')
        actual = result[0]
        if not isinstance(actual, np.ndarray) or actual.dtype.hasobject:
            raise ValueError('ORT output must be a numeric NumPy array')
        path = directory / f'actual_{index:02d}.npy'
        with path.open('xb') as stream:
            np.save(stream, actual, allow_pickle=False)
        row = dict(run_index=index, phase='cold_first_run' if index == 0 else 'warm',
                   seconds=seconds, actual=dict(file=path.name, bytes=path.stat().st_size,
                   sha256=sha(path), shape=list(actual.shape), dtype=str(actual.dtype)))
        receipt['runs'].append(row)
        row['comparison'] = compare(actual, reference, np)
        row['actual']['raw_sha256'] = raw_sha(actual)
        row['elementwise_equal_to_previous_ORT'] = (
            None if previous is None else bool(np.array_equal(actual, previous)))
        previous = actual.copy()
    receipt['cold_seconds'] = receipt['runs'][0]['seconds']
    receipt['warm_seconds'] = [row['seconds'] for row in receipt['runs'][1:]]
    receipt['warm_mean_seconds'] = sum(receipt['warm_seconds']) / len(receipt['warm_seconds'])
    receipt['all_runs_within_tolerance'] = all(row['comparison']['allclose'] for row in receipt['runs'])
    receipt['repeated_ORT_elementwise_equal'] = all(
        row['elementwise_equal_to_previous_ORT'] is True for row in receipt['runs'][1:])
    if not receipt['all_runs_within_tolerance']:
        raise ValueError('ORT differs from the original CPU reference beyond the predefined FP32 gate')
    if not receipt['repeated_ORT_elementwise_equal']:
        raise ValueError('Repeated fixed-input ORT outputs are not elementwise identical')


def run(args):
    require_board()
    receipt = dict(kind='original_VGGT_aggregator_block_CPU_ORT_check', status='started',
                   script_sha256=sha(__file__), block=args.block,
                   profile_sha256=args.profile_sha256, export_sha256=args.export_sha256,
                   exporter_name=args.exporter_name, exporter_sha256=args.exporter_sha256,
                   ORB_used=False, npu_used=False,
                   Torch_model_executed=False, full_VGGT_executed=False,
                   inference_on_board_only=True, graph_numerical_check_only=True,
                   accuracy_validated=False, real_time_validated=False,
                   tolerance=dict(rtol=RTOL, atol=ATOL, predefined=True,
                        formula='abs(ORT - original CPU reference) <= atol + rtol * abs(original CPU reference)'),
                   timing_scope='session.run only; excludes session creation/load/comparison/output I/O',
                   requested_calls=CALLS, successful_session_run_count=0, runs=[])
    started = time.monotonic()
    session = None
    try:
        before, environment = configure_cpu()
        import numpy as np
        import onnxruntime as ort
        receipt['runtime'] = dict(python=sys.version, platform=platform.platform(),
                    onnxruntime=ort.__version__, numpy=np.__version__,
                    cpu_affinity=sorted(os.sched_getaffinity(0)), intra_op_threads=2,
                    inter_op_threads=1, environment_before=before, enforced_environment=environment,
                    wait_environment={key: os.environ.get(key) for key in ('OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')})
        timer = time.perf_counter()
        tokens, reference, model_bytes, pins, identity = prepare_inputs(args, np)
        receipt['artifact_validation_seconds'] = time.perf_counter() - timer
        receipt['identity'] = identity
        receipt['input_pins'] = pins
        input_raw_sha = raw_sha(tokens)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        timer = time.perf_counter()
        # No filename is supplied: external initializer file resolution is disabled.
        session = ort.InferenceSession(model_bytes, sess_options=options,
                                       providers=['CPUExecutionProvider'])
        receipt['session_creation_seconds'] = time.perf_counter() - timer
        receipt['model_io'] = check_session(session, tuple(tokens.shape))
        execute_runs(session, tokens, reference, args.output, receipt, np)
        receipt['input_tensor_unchanged'] = raw_sha(tokens) == input_raw_sha
        receipt['input_files_unchanged'] = all(sha(path) == expected for path, expected in pins.items())
        if not receipt['input_tensor_unchanged'] or not receipt['input_files_unchanged']:
            raise ValueError('Bound model/capture/source/input changed during validation')
        receipt['status'] = 'passed'
    except Exception as exc:
        receipt['status'] = 'failed'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        receipt['traceback'] = traceback.format_exc()
    finally:
        session = None
        receipt['session_released'] = True
        receipt['total_seconds_including_load_comparison_output_io'] = time.monotonic() - started
        receipt['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        receipt['rss_scope'] = 'process high-water mark, including inherited pre-exec history if any'
        with (args.output / 'check.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture-dir', 'export-receipt', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    for key in ('profile-sha256', 'export-sha256', 'exporter-sha256'):
        parser.add_argument('--' + key, required=True, type=sha_argument)
    parser.add_argument('--exporter-name', choices=EXPORTERS, default=EXPORTERS[0])
    parser.add_argument('--block', choices=('frame0', 'global0'), required=True)
    args = parser.parse_args(argv)
    try:
        require_board()
        if os.path.lexists(args.output):
            raise ValueError('Refusing an existing output directory or symlink')
        args.output = args.output.resolve()
        if (not args.output.is_relative_to(BASE / 'experiments')
                or args.output == BASE / 'experiments'):
            raise ValueError('New output directory must be below board experiments')
        args.capture_dir = args.capture_dir.resolve()
        args.export_receipt = args.export_receipt.resolve()
        for path in (args.capture_dir, args.export_receipt):
            if not path.is_relative_to(BASE):
                raise ValueError('Capture/export inputs must be under the board project directory')
        if not args.capture_dir.is_dir() or not args.export_receipt.is_file():
            raise ValueError('Existing capture directory and export JSON required')
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = run(args)
    path = args.output / 'check.json'
    print(json.dumps(dict(status=receipt['status'], receipt=str(path), sha256=sha(path),
                          error=receipt.get('error'))))
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
