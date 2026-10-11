"""Fixed original VGGT depth-head metadata, without model or SDK imports."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CHECKPOINT = {'bytes': 5026874952,
              'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'}
SOURCE_HASHES = {
    'third_party/vggt/vggt/heads/dpt_head.py': '72e9ed26196113eb71e26b641b4262606b5aacc797accafc6c975f5db0d2ace2',
    'third_party/vggt/vggt/heads/utils.py': '2dce4c0dca1974a5250cc5d5d95bd3417e3ba9494978437a12d5ab6bbff4d4df',
    'third_party/vggt/vggt/heads/head_act.py': '428bdb087f457b809a66e22d7d01147c1cb4dab43bb0d98bc61c7f05d8383a1e',
}
LAYERS = (4, 11, 17, 23)
INPUT_NAMES = tuple(f'tokens_{index}' for index in LAYERS)
INPUT_SHAPE = (1, 48, 2048)
OUTPUT_SHAPE = (1, 2, 84, 112)
POSITION_SHAPES = {'project_0': (1, 256, 6, 8), 'project_1': (1, 512, 6, 8),
                   'project_2': (1, 1024, 6, 8), 'project_3': (1, 1024, 6, 8),
                   'final': (1, 128, 84, 112)}


def parameter_shapes():
    shapes = {'norm.weight': (2048,), 'norm.bias': (2048,)}
    def conv(name, output_channels, input_channels, kernel, *, bias=True):
        shapes[name+'.weight'] = (output_channels, input_channels, kernel, kernel)
        if bias:
            shapes[name+'.bias'] = (output_channels,)
    for index, channels in enumerate((256, 512, 1024, 1024)):
        conv(f'projects.{index}', channels, 2048, 1)
        conv(f'scratch.layer{index+1}_rn', 256, channels, 3, bias=False)
    conv('resize_layers.0', 256, 256, 4)
    conv('resize_layers.1', 512, 512, 2)
    conv('resize_layers.3', 1024, 1024, 3)
    for index in range(1, 5):
        prefix = f'scratch.refinenet{index}'
        conv(prefix+'.out_conv', 256, 256, 1)
        for unit in ((1, 2) if index < 4 else (2,)):
            for number in (1, 2):
                conv(f'{prefix}.resConfUnit{unit}.conv{number}', 256, 256, 3)
    conv('scratch.output_conv1', 128, 256, 3)
    conv('scratch.output_conv2.0', 32, 128, 3)
    conv('scratch.output_conv2.2', 2, 32, 1)
    return shapes


PARAMETER_SHAPES = parameter_shapes()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def raw_sha256(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def array_info(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'tensor_bytes': value.nbytes, 'raw_sha256': raw_sha256(value)}


def verify_arrays(arrays, shapes):
    if set(arrays) != set(shapes):
        raise ValueError('All original depth-head tensor identities are required')
    for name, shape in shapes.items():
        value = arrays[name]
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                or value.shape != shape or not value.flags.c_contiguous or not np.isfinite(value).all()):
            raise ValueError('Original contiguous finite FP32 tensor shape differs: '+name)


def head_contract(head):
    if (type(head).__name__ != 'DPTHead' or head.patch_size != 14 or head.activation != 'exp'
            or head.conf_activation != 'expp1' or head.pos_embed is not True
            or head.feature_only is not False or head.down_ratio != 1
            or tuple(head.intermediate_layer_idx) != LAYERS
            or tuple(head.norm.normalized_shape) != (2048,) or head.norm.eps != 1e-5
            or len(head.projects) != 4 or len(head.resize_layers) != 4
            or type(head.resize_layers[2]).__name__ != 'Identity'
            or head.scratch.stem_transpose is not None):
        raise ValueError('Unchanged original depth-only DPT configuration required')
    def check_conv(module, name, transposed=False, stride=1, padding=0):
        expected = PARAMETER_SHAPES[name+'.weight']
        if (type(module).__name__ != ('ConvTranspose2d' if transposed else 'Conv2d')
                or tuple(module.weight.shape) != expected or module.groups != 1
                or tuple(module.stride) != (stride, stride) or tuple(module.padding) != (padding, padding)
                or tuple(module.dilation) != (1, 1) or module.padding_mode != 'zeros'
                or (module.bias is not None) != (name+'.bias' in PARAMETER_SHAPES)
                or (transposed and tuple(module.output_padding) != (0, 0))):
            raise ValueError('Original convolution configuration changed: '+name)
    for index, project in enumerate(head.projects):
        check_conv(project, f'projects.{index}')
        check_conv(getattr(head.scratch, f'layer{index+1}_rn'), f'scratch.layer{index+1}_rn', padding=1)
    check_conv(head.resize_layers[0], 'resize_layers.0', transposed=True, stride=4)
    check_conv(head.resize_layers[1], 'resize_layers.1', transposed=True, stride=2)
    check_conv(head.resize_layers[3], 'resize_layers.3', stride=2, padding=1)
    for index in range(1, 5):
        block = getattr(head.scratch, f'refinenet{index}')
        prefix = f'scratch.refinenet{index}'
        if (block.has_residual is not (index < 4) or block.align_corners is not True
                or block.expand is not False or block.groups != 1 or block.deconv is not False or block.size is not None):
            raise ValueError('Original fusion configuration changed: '+prefix)
        check_conv(block.out_conv, prefix+'.out_conv')
        for unit in ((1, 2) if index < 4 else (2,)):
            residual = getattr(block, f'resConfUnit{unit}')
            if (residual.bn is not False or residual.groups != 1 or residual.norm1 is not None
                    or residual.norm2 is not None or type(residual.activation).__name__ != 'ReLU'):
                raise ValueError('Original residual configuration changed')
            for number in (1, 2):
                check_conv(getattr(residual, f'conv{number}'), f'{prefix}.resConfUnit{unit}.conv{number}', padding=1)
    check_conv(head.scratch.output_conv1, 'scratch.output_conv1', padding=1)
    check_conv(head.scratch.output_conv2[0], 'scratch.output_conv2.0', padding=1)
    check_conv(head.scratch.output_conv2[2], 'scratch.output_conv2.2')
    if len(head.scratch.output_conv2) != 3 or type(head.scratch.output_conv2[1]).__name__ != 'ReLU':
        raise ValueError('Original output activation changed')
    return {'layers': list(LAYERS), 'patch_size': 14, 'patch_start_idx': 5,
            'norm_epsilon': head.norm.eps, 'pos_embed': True, 'position_ratio': 0.1,
            'activation': 'exp', 'conf_activation': 'expp1', 'down_ratio': 1,
            'feature_only': False, 'frames_chunk_size': 1,
            'input_shape': list(INPUT_SHAPE), 'output_shape': list(OUTPUT_SHAPE),
            'preserved_original_parameter_count': len(PARAMETER_SHAPES)}
