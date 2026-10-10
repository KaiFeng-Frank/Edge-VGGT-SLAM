#!/usr/bin/env python3
"""Capture original aggregator SDPA operands on the RK3588S board only.

Rebuild one frozen frame0/global0 block, never the full VGGT model. The original
Torch SDPA call is intercepted only to copy its post-QK-norm/RoPE inputs and
output; its arguments, dispatch and returned tensor are unchanged. This is a
diagnostic capture, not a performance or whole-model accuracy benchmark.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
HERE = Path(__file__).resolve().parent
LOADER_NAME = 'export-v17-aggregator-block.py'
LOADER_SHA = '13bd46a6b6ee6e4700ddb3742059c1ba80c0d517b874c3cc2c56028220618ff8'
RTOL, ATOL = 1e-4, 1e-5
ARRAY_KEYS = ('query', 'key', 'value', 'output')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            result.update(data)
    return result.hexdigest()


def require_sha(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)
            or sha(path) != expected):
        raise ValueError(f'Frozen SHA256 mismatch: {path}')


def load_helper():
    path = HERE / LOADER_NAME
    require_sha(path, LOADER_SHA)
    spec = importlib.util.spec_from_file_location('_v19_frozen_capture_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(actual, expected, np):
    if (actual.dtype != np.float32 or expected.dtype != np.float32
            or actual.shape != expected.shape or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError('Finite same-shape FP32 comparison required')
    reference = expected.astype(np.float64)
    difference = actual.astype(np.float64) - reference
    absolute = np.abs(difference)
    within = absolute <= ATOL + RTOL * np.abs(reference)
    return dict(array_equal=bool(np.array_equal(actual, expected)), finite=True,
                max_abs=float(absolute.max()), rmse=float(np.sqrt(np.mean(difference * difference))),
                relative_l2=float(np.linalg.norm(difference) / max(float(np.linalg.norm(reference)), 1e-30)),
                elementwise_gate_passed=bool(within.all()), outside_tolerance=int((~within).sum()),
                rtol=RTOL, atol=ATOL)


class SDPAInterceptor:
    """Single-call copy observer. Always restore the original callable on exit."""

    def __init__(self, functional, snapshot):
        self.functional, self.snapshot = functional, snapshot
        self.original = functional.scaled_dot_product_attention
        self.calls = 0
        self.restored = False
        self.semantics = None
        self.wrapper = self.intercept

    def __enter__(self):
        if self.functional.scaled_dot_product_attention is not self.original:
            raise RuntimeError('SDPA callable changed before capture')
        self.functional.scaled_dot_product_attention = self.wrapper
        return self

    def intercept(self, query, key, value, *args, **kwargs):
        if self.calls:
            raise ValueError('Expected exactly one original SDPA call')
        # The pinned Attention.forward supplies only dropout_p=0.0 in eval.
        # Refuse any added mask, causal path, explicit scale or GQA extension.
        if args or set(kwargs) != {'dropout_p'} or kwargs['dropout_p'] != 0.0:
            raise ValueError('Pinned unmasked eval SDPA call signature changed')
        self.calls += 1
        self.semantics = dict(attn_mask=None, dropout_p=0.0, is_causal=False,
                              scale=0.125, scale_origin='original SDPA default 1/sqrt(64)',
                              enable_gqa=False, post_qk_norm_and_rope=True)
        for name, tensor in zip(ARRAY_KEYS, (query, key, value)):
            self.snapshot(name, tensor)
        # Pass the original objects and kwargs through unchanged.
        result = self.original(query, key, value, *args, **kwargs)
        self.snapshot('output', result)
        return result

    def __exit__(self, exc_type, exc, tb):
        self.functional.scaled_dot_product_attention = self.original
        self.restored = self.functional.scaled_dot_product_attention is self.original
        return False


def require_board(args):
    if platform.machine() != 'aarch64' or Path.cwd().resolve() != BASE:
        raise ValueError('Run only on the board from /mnt/vggt-data/vggt-slam2')
    if Path(__file__).resolve().parents[3] != SOURCE:
        raise ValueError('Use the deployed helper below board src/deploy/rk3588/validation')
    capture_dir = args.capture_dir.resolve()
    if not capture_dir.is_relative_to(BASE / 'experiments'):
        raise ValueError('The frozen capture must be below the board experiments directory')
    if os.path.lexists(args.output):
        raise ValueError('Refusing an existing output path or symlink')
    output = args.output.resolve()
    if output == BASE / 'experiments' or not output.is_relative_to(BASE / 'experiments'):
        raise ValueError('New output must be strictly below the board experiments directory')
    return capture_dir, output


def capture(args):
    """Called only after the CLI board/path guard and new output creation."""
    report = dict(kind='original_VGGT_attention_QKV_CPU_capture', status='started',
                  script_sha256=sha(__file__), block=args.block,
                  profile_sha256=args.profile_sha256, frozen_loader_sha256=LOADER_SHA,
                  inference_on_board_only=True, original_weights=True, architecture_changed=False,
                  full_VGGT_constructed=False, only_original_block_weights_loaded=True,
                  ORB_used=False, npu_used=False, host_model_inference_executed=False,
                  original_SDPA_dispatch_preserved=True, attention_NxN_saved=False,
                  performance_benchmark=False, full_model_quality_validated=False,
                  realtime_validated=False, predeclared_block_reference_gate=dict(rtol=RTOL, atol=ATOL),
                  cpu_affinity_requested=[6, 7], torch_threads_requested=2,
                  artifacts={}, cleanup_errors=[])
    started, interceptor = time.monotonic(), None
    allowed = os.sched_getaffinity(0)
    try:
        if not {6, 7}.issubset(allowed):
            raise ValueError('CPU 6,7 must be in the process allowed CPU set')
        os.sched_setaffinity(0, {6, 7})
        enforced = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        report['environment_before'] = {key: os.environ.get(key) for key in enforced}
        os.environ.update(enforced)
        helper = load_helper()  # Pure NumPy loader; no ONNX or inference.
        profile, row, arrays, weights, pins = helper.load_capture(
            args.capture_dir, args.profile_sha256, args.block)
        pins[str(Path(__file__).resolve())] = report['script_sha256']
        report.update(checkpoint=profile['checkpoint'], selected_pair=profile['selected_pair'],
                      image_size=profile['image_size'], source_pins=helper.SOURCE_PINS,
                      original_block_array_metadata=row['arrays'], original_block_parameter_metadata=row['weights'],
                      original_capture_artifacts=row['artifacts'], source_and_input_hashes=dict(pins))
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/vggt')]
        import numpy as np
        import torch
        from torch.nn import functional as functional
        from vggt.layers.block import Block
        from vggt.layers.rope import RotaryPositionEmbedding2D

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        report['runtime'] = dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version,
                                 cpu_affinity=sorted(os.sched_getaffinity(0)),
                                 torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                                 environment={key: os.environ.get(key) for key in
                                              (*enforced, 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')})
        rope = RotaryPositionEmbedding2D(frequency=row['rope_frequency'], scaling_factor=row['rope_scaling_factor'])
        # Use the actual original tables instead of recomputing trigonometry.
        rope.frequency_cache[(row['head_dim'] // 2, int(arrays['pos'].max()) + 1,
                              torch.device('cpu'), torch.float32)] = (
                                  torch.from_numpy(arrays['rope_cos']), torch.from_numpy(arrays['rope_sin']))
        with torch.device('meta'):
            block = Block(dim=1024, num_heads=16, mlp_ratio=4., qkv_bias=True,
                          proj_bias=True, ffn_bias=True, drop=0., attn_drop=0.,
                          init_values=0.01, drop_path=0., act_layer=torch.nn.GELU,
                          norm_layer=partial(torch.nn.LayerNorm, eps=row['norm1_eps']),
                          qk_norm=True, fused_attn=True, rope=rope)
        block.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()},
                              strict=True, assign=True)
        block.eval()
        if (block.training or block.attn.training or block.attn.attn_drop.p != 0.0
                or block.attn.proj_drop.p != 0.0 or block.sample_drop_ratio != 0.0
                or block.mlp.act.approximate != 'none' or block.ls1.inplace or block.ls2.inplace
                or any(value.is_meta or value.device.type != 'cpu' or value.dtype != torch.float32
                       for value in (*block.parameters(), *block.buffers()))):
            raise ValueError('Reconstructed block is not the original eval CPU FP32 architecture')
        if {name: helper.array_record(np.ascontiguousarray(value.detach().numpy()))
                for name, value in block.state_dict().items()} != row['weights']:
            raise ValueError('Assigned original 18-parameter state differs from frozen capture')
        report['architecture'] = {name: row[name] for name in (
            'module_class', 'num_heads', 'head_dim', 'norm1_eps', 'norm2_eps',
            'q_norm_eps', 'k_norm_eps', 'gelu_approximate', 'rope_frequency', 'rope_scaling_factor')}
        report['position_and_rope_tables_from_original_capture'] = True
        tokens, positions = torch.from_numpy(arrays['input']), torch.from_numpy(arrays['pos'])
        with torch.inference_mode():
            timer = time.perf_counter()
            plain = block(tokens, pos=positions)
            report['unhooked_block_seconds'] = time.perf_counter() - timer
        report['unhooked_vs_frozen_reference'] = compare(plain.numpy(), arrays['output'], np)
        if not report['unhooked_vs_frozen_reference']['elementwise_gate_passed']:
            raise ValueError('Original reconstructed block fails predefined frozen-reference FP32 gate')

        captured = {}
        expected_shape = (arrays['input'].shape[0], 16, arrays['input'].shape[1], 64)

        def snapshot(name, tensor):
            if (name in captured or tuple(tensor.shape) != expected_shape
                    or tensor.device.type != 'cpu' or tensor.dtype != torch.float32):
                raise ValueError(f'Unexpected original SDPA operand: {name}')
            value = tensor.detach().contiguous().numpy().copy()
            helper.array_record(value)
            captured[name] = value

        interceptor = SDPAInterceptor(functional, snapshot)
        with interceptor, torch.inference_mode():
            timer = time.perf_counter()
            observed = block(tokens, pos=positions)
            report['captured_block_seconds_including_operand_copies'] = time.perf_counter() - timer
        if interceptor.calls != 1 or tuple(captured) != ARRAY_KEYS or not interceptor.restored:
            raise ValueError('One complete original SDPA call and restoration are required')
        report['captured_vs_unhooked'] = compare(observed.numpy(), plain.numpy(), np)
        report['captured_vs_frozen_reference'] = compare(observed.numpy(), arrays['output'], np)
        if (not report['captured_vs_unhooked']['array_equal']
                or not report['captured_vs_frozen_reference']['elementwise_gate_passed']):
            raise ValueError('SDPA observer changed the original CPU block output')
        report['sdpa_call_count'] = interceptor.calls
        report['attention_semantics'] = interceptor.semantics
        report['attention_shape'] = list(expected_shape)
        report['flattened_shape'] = [1, expected_shape[0] * expected_shape[1], expected_shape[2], expected_shape[3]]
        report['flattening_order'] = 'C-order batch-major then head-major; no token/key partition'
        attention_path = args.output / 'attention.npz'
        with attention_path.open('xb') as stream:
            np.savez(stream, **captured)
        report['artifacts']['attention.npz'] = dict(file=attention_path.name, sha256=sha(attention_path),
            bytes=attention_path.stat().st_size, arrays={name: helper.array_record(value) for name, value in captured.items()})
        output_path = args.output / 'block-output.npy'
        with output_path.open('xb') as stream:
            np.save(stream, np.ascontiguousarray(observed.numpy()), allow_pickle=False)
        report['artifacts']['block-output.npy'] = dict(file=output_path.name, sha256=sha(output_path),
            bytes=output_path.stat().st_size, array=helper.array_record(np.ascontiguousarray(observed.numpy())))
        after = {path: sha(path) for path in pins}
        report['source_and_input_hashes_after'] = after
        report['source_and_inputs_unchanged'] = after == pins
        if not report['source_and_inputs_unchanged']:
            raise ValueError('Frozen source or input changed during capture')
        report['status'] = 'completed_original_CPU_attention_capture'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if interceptor is not None and not interceptor.restored:
            try:
                interceptor.__exit__(None, None, None)
            except BaseException as exc:
                report['cleanup_errors'].append(f'SDPA restoration: {exc}')
        report['sdpa_restored'] = interceptor is None or interceptor.restored
        report['hooks_installed'] = False
        try:
            os.sched_setaffinity(0, allowed)
            report['cpu_affinity_restored'] = os.sched_getaffinity(0) == allowed
        except BaseException as exc:
            report['cleanup_errors'].append(f'CPU affinity restoration: {exc}')
        if report['cleanup_errors'] or not report['sdpa_restored']:
            report['status'] = 'failed'
        report['total_seconds_including_validation_load_capture_save'] = time.monotonic() - started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        report['rss_scope'] = 'process high-water mark; may include inherited pre-exec history'
        with (args.output / 'capture.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--profile-sha256', required=True)
    parser.add_argument('--block', choices=('frame0', 'global0'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.capture_dir, args.output = require_board(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    report = capture(args)
    print(json.dumps(dict(status=report['status'], receipt=str(args.output / 'capture.json'),
                          sha256=sha(args.output / 'capture.json'), error=report.get('error'))))
    return 0 if report['status'] == 'completed_original_CPU_attention_capture' else 1


if __name__ == '__main__':
    sys.exit(main())
