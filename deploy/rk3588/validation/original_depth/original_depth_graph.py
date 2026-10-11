"""Static complete original DPT logits graph; no Torch/model/ORT/SDK imports.

The caller supplies all 62 captured original FP32 tensors and five additive
position constants computed by the original board utilities. One frame uses
four original patch-token views; the runtime retains the original two chunks
and activate_head outside this graph. In-place ReLU changes the residual skip
to the first ReLU's output; this alias effect is preserved explicitly.
"""
from __future__ import annotations

import numpy as np
import depth_contract as C


def build_original_depth_logits_onnx(state, *, norm_epsilon, positional_embeddings):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    C.verify_arrays(state, C.PARAMETER_SHAPES)
    C.verify_arrays(positional_embeddings, C.POSITION_SHAPES)
    if norm_epsilon != 1e-5:
        raise ValueError('Original DPT LayerNorm epsilon 1e-5 required')
    nodes = []
    initializers = [numpy_helper.from_array(value, name) for name, value in state.items()]
    initializers += [numpy_helper.from_array(value, 'position.'+name)
                     for name, value in positional_embeddings.items()]

    def node(kind, inputs, name, **attributes):
        output = name+'.output'
        nodes.append(helper.make_node(kind, inputs, [output], name=name, **attributes))
        return output

    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))
        return name

    def conv(x, name, *, transposed=False, stride=1, padding=0):
        inputs = [x, name+'.weight']
        if name+'.bias' in state:
            inputs.append(name+'.bias')
        return node('ConvTranspose' if transposed else 'Conv', inputs, name,
                    kernel_shape=list(state[name+'.weight'].shape[-2:]),
                    strides=[stride, stride], pads=[padding]*4, dilations=[1, 1], group=1)

    def resize(x, name, height, width, channels):
        sizes = constant(name+'.sizes', [1, channels, height, width])
        return node('Resize', [x, '', '', sizes], name, mode='linear',
                    coordinate_transformation_mode='align_corners')

    def residual(x, name):
        # Original nn.ReLU(inplace=True) overwrites the skip tensor on entry.
        skip = node('Relu', [x], name+'.relu1')
        out = conv(skip, name+'.conv1', padding=1)
        out = node('Relu', [out], name+'.relu2')
        out = conv(out, name+'.conv2', padding=1)
        return node('Add', [out, skip], name+'.skip')

    projected = []
    for index, input_name in enumerate(C.INPUT_NAMES):
        prefix = f'branch{index}'
        x = node('LayerNormalization', [input_name, 'norm.weight', 'norm.bias'], prefix+'.norm',
                 axis=-1, epsilon=norm_epsilon, stash_type=1)
        x = node('Transpose', [x], prefix+'.transpose', perm=[0, 2, 1])
        shape = constant(prefix+'.shape', [1, 2048, 6, 8])
        x = node('Reshape', [x, shape], prefix+'.reshape', allowzero=0)
        x = conv(x, f'projects.{index}')
        x = node('Add', [x, f'position.project_{index}'], prefix+'.position')
        if index in (0, 1):
            stride = 4 if index == 0 else 2
            x = conv(x, f'resize_layers.{index}', transposed=True, stride=stride)
        elif index == 3:
            x = conv(x, 'resize_layers.3', stride=2, padding=1)
        # resize_layers[2] is the original Identity.
        projected.append(conv(x, f'scratch.layer{index+1}_rn', padding=1))

    out = projected[3]
    fusion_sizes = {4: (6, 8), 3: (12, 16), 2: (24, 32), 1: (48, 64)}
    for index in (4, 3, 2, 1):
        prefix = f'scratch.refinenet{index}'
        if index != 4:
            res = residual(projected[index-1], prefix+'.resConfUnit1')
            out = node('Add', [out, res], prefix+'.fusion_skip')
        out = residual(out, prefix+'.resConfUnit2')
        height, width = fusion_sizes[index]
        out = resize(out, prefix+'.resize', height, width, 256)
        out = conv(out, prefix+'.out_conv')
    out = conv(out, 'scratch.output_conv1', padding=1)
    out = resize(out, 'final_resize', 84, 112, 128)
    out = node('Add', [out, 'position.final'], 'final_position')
    out = conv(out, 'scratch.output_conv2.0', padding=1)
    out = node('Relu', [out], 'scratch.output_conv2.1')
    out = conv(out, 'scratch.output_conv2.2')
    # Rename the final convolution's output, without adding a redundant operator.
    nodes[-1].output[0] = 'logits'
    graph = helper.make_graph(nodes, 'original_VGGT_complete_depth_logits',
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, C.INPUT_SHAPE) for name in C.INPUT_NAMES],
        [helper.make_tensor_value_info('logits', TensorProto.FLOAT, C.OUTPUT_SHAPE)], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)],
                              producer_name='Edge-VGGT-SLAM/original-depth-static-v54')
    model.ir_version = 8
    model.doc_string = ('Complete original DPT after its input views; no learned tensors or fusion stages omitted. '
                        'Original CPU activate_head(exp/expp1) remains outside; one-frame chunks preserved.')
    onnx.checker.check_model(model, full_check=True)
    tensors = {value.name: value for value in model.graph.initializer}
    for name, original in {**state, **{'position.'+key: value for key, value in positional_embeddings.items()}}.items():
        restored = numpy_helper.to_array(tensors[name])
        if restored.dtype != np.float32 or restored.shape != original.shape or C.raw_sha256(restored) != C.raw_sha256(original):
            raise ValueError('Captured original graph initializer changed: '+name)
    return model
