#!/usr/bin/env python3
"""Pure ONNX export of full and three independent-head attention graphs.

Reads completed board-only v19 operand captures. No Torch, ORT, model forward,
image decoding or NPU runtime is used. Every graph keeps the complete key/token
domain, original scale 1/8 and last-key-axis softmax. Only (batch, head) is
flattened and partitioned; the original ordering is recorded and reversible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAPTURE_HELPER_NAME = 'capture-v19-attention-board.py'
CAPTURE_HELPER_SHA = '7d6a606c050ecdc0fd67cb27a87d0c6a7924751e79e48fd1fa0745886089a0dc'
LOADER_NAME = 'export-v17-aggregator-block.py'
LOADER_SHA = '13bd46a6b6ee6e4700ddb3742059c1ba80c0d517b874c3cc2c56028220618ff8'
ARRAY_KEYS = ('query', 'key', 'value', 'output')
SCALE = np.float32(0.125)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def require_sha(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected) or sha(path) != expected):
        raise ValueError(f'Required SHA256 mismatch: {path}')


def array_record(array):
    if (array.dtype != np.float32 or not array.size or not array.flags.c_contiguous
            or not np.isfinite(array).all()):
        raise ValueError('Finite nonempty C-contiguous FP32 array required')
    return dict(shape=list(array.shape), dtype=str(array.dtype), bytes=array.nbytes,
                sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())


def head_plan(original_shape):
    if (len(original_shape) != 4 or any(type(n) is not int or n <= 0 for n in original_shape)
            or original_shape[3] != 64 or original_shape[0] * original_shape[1] < 3):
        raise ValueError('Static [batch,heads,tokens,64] with at least three independent heads required')
    batch, heads, tokens, dim = original_shape
    total = batch * heads
    quotient, remainder = divmod(total, 3)
    begin, plan = 0, []
    for index in range(3):
        end = begin + quotient + int(index < remainder)
        plan.append(dict(index=index, begin=begin, end=end, independent_heads=end-begin,
                         input_shape=[1, end-begin, tokens, dim],
                         original_head_indices=[dict(batch=i // heads, head=i % heads) for i in range(begin, end)]))
        begin = end
    return plan


def flatten_heads(array):
    array_record(array)
    head_plan(tuple(array.shape))
    return array.reshape(1, array.shape[0] * array.shape[1], array.shape[2], array.shape[3])


def restore_heads(shards, original_shape):
    """Pure layout inverse; never creates or interpolates an attention result."""
    plan = head_plan(tuple(original_shape))
    if len(shards) != 3:
        raise ValueError('Exactly three ordered head shards required')
    for value, row in zip(shards, plan):
        array_record(value)
        if list(value.shape) != row['input_shape']:
            raise ValueError('Shard shape or original head ordering mismatch')
    return np.concatenate(shards, axis=1).reshape(original_shape)


def frozen_loader():
    path = HERE / LOADER_NAME
    require_sha(path, LOADER_SHA)
    spec = importlib.util.spec_from_file_location('_v19_original_source_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_operands(directory, expected_capture_sha):
    directory = Path(directory).resolve()
    receipt_path = directory / 'capture.json'
    require_sha(receipt_path, expected_capture_sha)
    receipt = json.loads(receipt_path.read_text())
    required = dict(kind='original_VGGT_attention_QKV_CPU_capture',
                    status='completed_original_CPU_attention_capture', script_sha256=CAPTURE_HELPER_SHA,
                    frozen_loader_sha256=LOADER_SHA, inference_on_board_only=True,
                    original_weights=True, architecture_changed=False, full_VGGT_constructed=False,
                    only_original_block_weights_loaded=True, ORB_used=False, npu_used=False,
                    original_SDPA_dispatch_preserved=True, sdpa_call_count=1, sdpa_restored=True,
                    hooks_installed=False, cpu_affinity_restored=True, source_and_inputs_unchanged=True,
                    position_and_rope_tables_from_original_capture=True,
                    flattening_order='C-order batch-major then head-major; no token/key partition')
    if any(receipt.get(key) != value for key, value in required.items()) or receipt.get('cleanup_errors'):
        raise ValueError('Completed frozen original operand capture and cleanup are required')
    if receipt.get('predeclared_block_reference_gate') != dict(rtol=1e-4, atol=1e-5):
        raise ValueError('Original predeclared block-reference gate changed')
    for key in ('unhooked_vs_frozen_reference', 'captured_vs_frozen_reference', 'captured_vs_unhooked'):
        check = receipt.get(key, {})
        if (check.get('finite') is not True or check.get('elementwise_gate_passed') is not True
                or check.get('rtol') != 1e-4 or check.get('atol') != 1e-5
                or check.get('outside_tolerance') != 0
                or (key == 'captured_vs_unhooked' and check.get('array_equal') is not True)):
            raise ValueError('Original CPU block reference or unchanged observer gate was not passed')
    semantics = dict(attn_mask=None, dropout_p=0.0, is_causal=False, scale=0.125,
                     scale_origin='original SDPA default 1/sqrt(64)', enable_gqa=False,
                     post_qk_norm_and_rope=True)
    if receipt.get('attention_semantics') != semantics:
        raise ValueError('Original complete-key unmasked SDPA semantics mismatch')
    loader = frozen_loader()
    checkpoint = receipt.get('checkpoint', {})
    if checkpoint.get('sha256') != loader.CHECKPOINT_SHA or checkpoint.get('bytes') != 5026874952:
        raise ValueError('Official checkpoint provenance mismatch')
    if (receipt.get('source_pins') != loader.SOURCE_PINS
            or set(receipt.get('original_block_parameter_metadata', {})) != set(loader.STATE_KEYS)):
        raise ValueError('Original sources or 18-parameter provenance mismatch')
    architecture = dict(module_class='Block', num_heads=16, head_dim=64, norm1_eps=1e-5,
                        norm2_eps=1e-5, q_norm_eps=1e-5, k_norm_eps=1e-5, gelu_approximate='none',
                        rope_frequency=100, rope_scaling_factor=1.0)
    if receipt.get('architecture') != architecture:
        raise ValueError('Original QK normalization/RoPE architecture changed')
    original = receipt.get('source_and_input_hashes', {})
    if not original or original != receipt.get('source_and_input_hashes_after'):
        raise ValueError('Capture did not retain an unchanged frozen-input hash chain')
    needed = [receipt.get('profile_sha256'), CAPTURE_HELPER_SHA, LOADER_SHA,
              *loader.SOURCE_PINS.values()]
    needed.extend(item.get('sha256') for item in receipt.get('original_capture_artifacts', {}).values())
    if (not {'tensors', 'weights'} <= set(receipt.get('original_capture_artifacts', {}))
            or any(not isinstance(value, str) or value not in original.values() for value in needed)):
        raise ValueError('Frozen profile, sources, parameter and position artifacts are not hash-bound')
    pair = receipt.get('selected_pair', {})
    if (len(pair.get('frame_ids', [])) != 2 or len(pair.get('images', [])) != 2
            or any(item.get('samefile_validated') is not True or item.get('frame_id') != frame_id
                   for item, frame_id in zip(pair['images'], pair['frame_ids']))):
        raise ValueError('Original two-frame source identity mismatch')
    block, shape = receipt.get('block'), receipt.get('attention_shape', [])
    if block not in ('frame0', 'global0') or len(shape) != 4 or shape[:2] != [2 if block == 'frame0' else 1, 16]:
        raise ValueError('Only original frame0/global0 head layout is supported')
    head_plan(tuple(shape))
    input_meta = receipt.get('original_block_array_metadata', {}).get('input', {})
    if (input_meta.get('shape') != [shape[0], shape[2], 1024]
            or input_meta.get('dtype') != 'float32'
            or receipt.get('flattened_shape') != [1, shape[0] * 16, shape[2], 64]):
        raise ValueError('Original block/token layout is not bound to the captured attention')
    pins = {str(receipt_path): expected_capture_sha, str(Path(__file__).resolve()): sha(__file__)}
    for name, expected in ((CAPTURE_HELPER_NAME, CAPTURE_HELPER_SHA), (LOADER_NAME, LOADER_SHA)):
        path = HERE / name
        require_sha(path, expected)
        pins[str(path)] = expected
    for name, expected in loader.SOURCE_PINS.items():
        path = REPO / name
        require_sha(path, expected)
        pins[str(path)] = expected
    artifacts = receipt.get('artifacts', {})
    if set(artifacts) != {'attention.npz', 'block-output.npy'}:
        raise ValueError('Exact original capture artifact set required')
    for filename, entry in artifacts.items():
        path = directory / filename
        if entry.get('file') != filename or path.stat().st_size != entry.get('bytes'):
            raise ValueError('Capture artifact file/byte identity mismatch')
        require_sha(path, entry.get('sha256'))
        pins[str(path)] = entry['sha256']
    with np.load(directory / 'attention.npz', allow_pickle=False) as saved:
        if set(saved.files) != set(ARRAY_KEYS):
            raise ValueError('Exact original Q/K/V/output keys required')
        arrays = {name: saved[name] for name in ARRAY_KEYS}
    metadata = artifacts['attention.npz'].get('arrays', {})
    if set(metadata) != set(ARRAY_KEYS):
        raise ValueError('Exact operand raw-byte metadata required')
    for name, value in arrays.items():
        if list(value.shape) != shape or array_record(value) != metadata[name]:
            raise ValueError(f'Original operand raw hash/type/shape mismatch: {name}')
    full_output = np.load(directory / 'block-output.npy', allow_pickle=False)
    if (array_record(full_output) != artifacts['block-output.npy'].get('array')
            or list(full_output.shape) != [shape[0], shape[2], 1024]):
        raise ValueError('Actual original block output proof mismatch')
    return receipt, arrays, pins


def build_attention_onnx(independent_heads, tokens):
    """Only construct the original attention equation; no forward execution."""
    import onnx
    from onnx import checker, helper, numpy_helper, shape_inference

    if any(type(value) is not int or value <= 0 for value in (independent_heads, tokens)):
        raise ValueError('Positive static independent-head and complete-token counts required')
    shape = [1, independent_heads, tokens, 64]
    nodes = [
        helper.make_node('Transpose', ['key'], ['key_transpose'], name='key_transpose', perm=[0, 1, 3, 2]),
        helper.make_node('MatMul', ['query', 'key_transpose'], ['attention_scores'], name='attention_scores'),
        helper.make_node('Mul', ['attention_scores', 'attention_scale'], ['attention_scaled_scores'], name='attention_scaled_scores'),
        helper.make_node('Softmax', ['attention_scaled_scores'], ['attention_probabilities'], name='attention_probabilities', axis=3),
        helper.make_node('MatMul', ['attention_probabilities', 'value'], ['output'], name='attention_output'),
    ]
    graph = helper.make_graph(nodes, 'original_complete_key_attention',
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name in ARRAY_KEYS[:3]],
        [helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, shape)],
        initializer=[numpy_helper.from_array(np.array(SCALE, dtype=np.float32), 'attention_scale')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)],
                             producer_name='vggt-slam2-rk3588s/v19-independent-head-attention')
    model.ir_version = 8
    model.doc_string = ('Original post-QK-norm/RoPE attention: softmax(Q K^T / 8, complete key axis) V. '
                        'Only independent batch/head indexing is changed; no weights, masks or approximations added. '
                        'Algebraic equivalence does not establish FP16 numerical equivalence or NPU speed.')
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True, check_type=True, data_prop=True)
    checker.check_model(model, full_check=True)
    return model


def export(directory, expected_capture_sha, output):
    report = dict(kind='original_VGGT_independent_head_attention_ONNX_export', status='started',
                  script_sha256=sha(__file__), capture_sha256=expected_capture_sha,
                  host_model_inference_executed=False, host_ORT_executed=False,
                  production_changed=False, npu_used=False, learned_parameter_count=0,
                  all_operators_on_npu_verified=False, realtime_validated=False,
                  whole_model_quality_validated=False, models=[])
    try:
        receipt, arrays, pins = load_operands(directory, expected_capture_sha)
        shape = tuple(arrays['query'].shape)
        flat = {name: flatten_heads(value) for name, value in arrays.items()}
        plan = head_plan(shape)
        report.update(block=receipt['block'], original_shape=list(shape),
            flattened_shape=list(flat['query'].shape), flattening_order=receipt['flattening_order'],
            attention_semantics=receipt['attention_semantics'], head_shard_plan=plan,
            inverse_order='concatenate shard outputs along axis 1 in indices 0,1,2, then C-order reshape to original_shape',
            checkpoint=receipt['checkpoint'], selected_pair=receipt['selected_pair'],
            profile_sha256=receipt['profile_sha256'], original_parameter_metadata=receipt['original_block_parameter_metadata'],
            original_array_metadata=receipt['original_block_array_metadata'], source_pins=receipt['source_pins'],
            frozen_helpers={CAPTURE_HELPER_NAME: CAPTURE_HELPER_SHA, LOADER_NAME: LOADER_SHA},
            input_pins=pins, block_reference_checks={key: receipt[key] for key in (
                'unhooked_vs_frozen_reference', 'captured_vs_unhooked', 'captured_vs_frozen_reference')})
        specs = [dict(name='baseline', kind='full_independent_head_baseline', begin=0,
                      end=flat['query'].shape[1], independent_heads=flat['query'].shape[1], index=None),
                 *[dict(name=f'shard-{row["index"]:02d}', kind='head_shard', **row) for row in plan]]
        for spec in specs:
            folder = output / spec['name']
            folder.mkdir()
            values = {name: np.ascontiguousarray(value[:, spec['begin']:spec['end']]) for name, value in flat.items()}
            graph = build_attention_onnx(spec['independent_heads'], shape[2])
            model_path = folder / 'attention.onnx'
            with model_path.open('xb') as stream:
                stream.write(graph.SerializeToString())
            artifacts = {}
            for name, value in values.items():
                path = folder / f'{name}.npy'
                with path.open('xb') as stream:
                    np.save(stream, value, allow_pickle=False)
                artifacts[path.name] = dict(file=str(path.relative_to(output)), bytes=path.stat().st_size,
                                             sha256=sha(path), array=array_record(value))
            report['models'].append(dict(**{key: value for key, value in spec.items() if key != 'input_shape'},
                input_names=list(ARRAY_KEYS[:3]), output_names=['output'],
                input_shape=list(values['query'].shape), output_shape=list(values['output'].shape),
                onnx=dict(file=str(model_path.relative_to(output)), bytes=model_path.stat().st_size, sha256=sha(model_path)),
                artifacts=artifacts, complete_key_count=shape[2], token_or_key_partitioned=False,
                node_count=len(graph.graph.node), op_counts=dict(Counter(node.op_type for node in graph.graph.node)),
                external_data_used=False))
        restored = restore_heads([flat['output'][:, row['begin']:row['end']].copy() for row in plan], shape)
        if not np.array_equal(restored, arrays['output']):
            raise ValueError('Head partition inverse did not reproduce original CPU reference bytes')
        for path, expected in pins.items():
            require_sha(path, expected)
        report.update(status='exported_and_structurally_checked',
                      saved_reference_partition_inverse_bitwise_equal=True, source_and_inputs_unchanged=True,
                      graph_inference_executed=False, unsharded_and_sharded_board_ORT_checks_pending=True,
                      NPU_concurrency_speed_and_precision_checks_pending=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    with (output / 'export.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.output):
        parser.error('Refusing an existing output directory or symlink')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = export(args.capture_dir, args.capture_sha256, output)
    print(json.dumps(dict(status=report['status'], receipt=str(output / 'export.json'),
                          sha256=sha(output / 'export.json'), error=report.get('error'))))
    return 0 if report['status'] == 'exported_and_structurally_checked' else 1


if __name__ == '__main__':
    sys.exit(main())
