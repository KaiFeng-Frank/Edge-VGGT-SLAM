"""Tiny generated protobuf fixtures; no model, ORT, SDK or compiler execution."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import onnx
from onnx import TensorProto, helper

SOURCE = Path(__file__).with_name('compose_all_retained_groups.py')
SOURCE_SHA = 'e6fa508e0b03a011a1ccb484f2a46847ffd49fc1bf6ae11ecfeb8bd45f835f60'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
spec = importlib.util.spec_from_file_location('_generated_v67_composer', SOURCE)
composer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composer)


def fixtures(start, end):
    models, rows = [], []
    bindings = composer.output_bindings([0, 4])
    for index in range(start, end + 1):
        graph = helper.make_graph([
            helper.make_node('Add', ['input', 'fixture_scalar'], ['frame_output'], name='fixture_add'),
            helper.make_node('Reshape', ['frame_output', 'fixture_shape'], ['global_output'],
                             name='fixture_view', allowzero=0)], 'generated_fixture',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, composer.FRAME)],
            [helper.make_tensor_value_info('frame_output', TensorProto.FLOAT, composer.FRAME),
             helper.make_tensor_value_info('global_output', TensorProto.FLOAT, composer.GLOBAL)],
            initializer=[helper.make_tensor('fixture_scalar', TensorProto.FLOAT, [], [float(index)]),
                         helper.make_tensor('fixture_shape', TensorProto.INT64, [3], composer.GLOBAL)])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 9
        models.append(model)
        rows.append(dict(index=index, role='pair', input_name='input', input_shape=composer.FRAME,
            output_bindings=copy.deepcopy(bindings), heads=16,
            all_36_original_parameter_values_preserved_by_parent_composition=True,
            parent_nodes_plus_one_reshape=True, original_aggregator_concat_must_be_retained=True,
            parameter_raw_sha256={role: {str(i): '0' * 64 for i in range(18)}
                                  for role in ('frame', 'global')}))
    return models, rows


for start, end in composer.GROUPS:
    models, rows = fixtures(start, end)
    before = [m.SerializeToString() for m in models]
    result, proof = composer.compose_retained_group(models, rows)
    assert [m.SerializeToString() for m in models] == before
    names = ['frame_output', 'global_output'] + (['target_global_output'] if start == 18 else [])
    assert [v.name for v in result.graph.output] == names
    assert len(result.graph.node) == len(rows) * 2 + len(rows) - 1
    assert len(result.graph.initializer) == len(rows) * 2 + len(rows) - 1
    assert proof['original_parameter_value_count'] == len(rows) * 36
    assert proof['only_retained_consumer_outputs_public']
    assert proof['public_output_sources'][:2] == [
        dict(index=0, name='frame_output', pair_index=end, role='frame'),
        dict(index=1, name='global_output', pair_index=end, role='global')]
    assert all(p['inverse_namespace_recovers_every_node_and_initializer_protobuf_bytewise']
               for p in proof['parent_preservation_proofs'])
    for index in range(start, end):
        bridge = next(n for n in result.graph.node
                      if n.name == f'global{index}_to_frame{index + 1}_original_view')
        source = 'target_global_output' if index == 20 else f'pair{index}_global_output'
        assert bridge.input[0] == source and bridge.output[0] == f'pair{index + 1}_input'
        following = next(n for n in result.graph.node if n.name == f'pair{index + 1}/fixture_add')
        assert following.input[0] == bridge.output[0]
    if start == 18:
        target = next(n for n in result.graph.node if n.name == 'pair20/fixture_view')
        assert list(target.output) == ['target_global_output']
        prior = next(n for n in result.graph.node if n.name == 'pair19/fixture_view')
        assert list(prior.output) == ['pair19_global_output']
        assert proof['target_global_output_source_pair_index'] == 20
        assert proof['public_output_sources'][2]['pair_index'] == 20
        assert proof['target_global_output_is_unnormalized'] is True
        assert 'target_global_output' not in {v.name for v in result.graph.value_info}
    else:
        assert proof['target_global_output_source_pair_index'] is None

for failure in ('missing_parent', 'reordered_parent', 'wrong_binding', 'extra_output'):
    models, rows = fixtures(18, 23)
    if failure == 'missing_parent':
        models, rows = models[:-1], rows[:-1]
    elif failure == 'reordered_parent':
        models[0], models[1] = models[1], models[0]
        rows[0], rows[1] = rows[1], rows[0]
    elif failure == 'wrong_binding':
        rows[2]['output_bindings'][0]['shape'] = [1, 53, 1024]
    else:
        models[2].graph.output.add().CopyFrom(models[2].graph.output[0])
    try:
        composer.compose_retained_group(models, rows)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted invalid fixture: ' + failure)

print('4 generated groups preserve source protobufs and global20 target bridge; 4 invalid fixtures rejected')
