"""Static complete original VGGT DINO block graphs, with no model execution."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

CHECKPOINT = {'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
              'bytes': 5026874952}
IMAGE_SHAPE = (1, 3, 84, 112)
SHAPE = (1, 53, 1024)
DEPTH, HEADS, DIM = 24, 16, 1024
PARAMETER_NAMES = (
    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight',
    'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight',
    'mlp.fc2.bias', 'ls2.gamma')
TRANSPOSED_PARAMETERS = {'attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight'}
OPERATOR_BUILDERS = {
    'export-v12-dino-block.py': '2759fce1551b2a64da140a7be0b16c9d233241e0bb427b4d52c58d2243d6bf07',
    'export-v12-dino-block-gelu.py': '59f5ea1be93812571947f7afb71ea9d908057cf277c3cbc2cfb9bbaaf10e5a78'}
_BUILDERS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def save_array(directory, name, value):
    value = np.ascontiguousarray(value)
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('Finite original CPU FP32 captured tensor required')
    path = Path(directory) / name
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    return dict(file=name, bytes=path.stat().st_size, sha256=sha(path),
                shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
                raw_sha256=raw_sha(value))


def load_builders(directory):
    directory = Path(directory).resolve(strict=True)
    if str(directory) not in _BUILDERS:
        loaded = []
        for index, (filename, expected) in enumerate(OPERATOR_BUILDERS.items()):
            path = directory / filename
            if sha(path) != expected:
                raise ValueError('Validated original static operator source changed: ' + filename)
            spec = importlib.util.spec_from_file_location(f'_v54_dino_static_{index}', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(module)
        _BUILDERS[str(directory)] = tuple(loaded)
    return _BUILDERS[str(directory)]


def build_original_dino_block(weights, *, index, metadata, operator_directory):
    if type(index) is not int or not 0 <= index < DEPTH or set(weights) != set(PARAMETER_NAMES):
        raise ValueError('All 14 original parameters for DINO layer 0..23 required')
    if (metadata['heads'] != HEADS or metadata['epsilons'] != {'norm1': 1e-6, 'norm2': 1e-6}
            or metadata['gelu_approximate'] != 'none' or metadata['qk_norm'] is not False
            or metadata['rope'] is not None):
        raise ValueError('Original DINO attention/normalization/exact GELU contract differs')
    original, gelu = load_builders(operator_directory)
    model = original.build_block_onnx(weights, SHAPE, heads=HEADS, layer_norm='standard')
    model = gelu.rewrite_gelu(model)
    from onnx import numpy_helper
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    for name in PARAMETER_NAMES:
        tensor_name = name + '_transposed' if name in TRANSPOSED_PARAMETERS else name
        expected = weights[name].T if name in TRANSPOSED_PARAMETERS else weights[name]
        actual = numpy_helper.to_array(tensors[tensor_name])
        if actual.dtype != np.float32 or actual.shape != expected.shape or raw_sha(actual) != raw_sha(expected):
            raise ValueError('Original DINO learned tensor changed during static construction: ' + name)
    model.graph.name = f'original_VGGT_DINO_block_{index:02d}'
    model.producer_name = 'Edge-VGGT-SLAM/original-v54-static-DINO'
    model.doc_string = ('Complete original VGGT internal DINO block. All 14 original parameters preserved. '
        'Original 53 tokens, SDPA mathematics, exact Erf GELU, LayerScale and residuals retained. '
        'GELU fusion pattern changes FP32 operation ordering; board numerical validation is separate.')
    return model
