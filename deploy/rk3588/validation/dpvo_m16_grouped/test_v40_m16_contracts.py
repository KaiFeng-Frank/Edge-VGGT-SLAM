"""Generated metadata/fake toolkit tests; no neural model, SDK or compiler run."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('V40_CONTRACT_SOURCE_DIR',
    HERE if (HERE/'v40_m16_grouped_contracts.py').is_file() else
    HERE.parent/'deploy'/'rk3588'/'validation'/'dpvo_m16_grouped'))
PARENT_SOURCE = Path(os.environ.get('V40_PARENT_GROUPED_SOURCE',
    SOURCE.parent/'dpvo_grouped_update'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c = load(SOURCE/'v40_m16_grouped_contracts.py', '_test_v40_m16_contracts')
c.load_parent(PARENT_SOURCE/'v37_grouped_contracts.py')
compiler = load(SOURCE/'compile-v40-m16-grouped-update.py', '_test_v40_m16_compiler')


def io(batch=2, degree=4, patches=16):
    inputs = [dict(name=name, dtype='float32', shape=[batch, channels, degree, patches])
              for name, channels in zip(c.INPUT_NAMES, (384, 384, 882))]
    outputs = [dict(name=name, dtype='float32', shape=[batch, channels, degree, patches])
               for name, channels in zip(c.OUTPUT_NAMES, (384, 2, 2))]
    return inputs, outputs


def manifest():
    # Scalar metadata only, with all predeclared true-B shapes; no checkpoint.
    parameters = {'p%d' % index: dict(shape=[1], dtype='float32', bytes=4, tensor_sha256='a'*64)
                  for index in range(50)}
    weights = {'update.'+key: value for key, value in parameters.items()}
    weights.update({'encoder%d' % index: {'synthetic': True} for index in range(44)})
    result = dict(schema=c.EXPORT_SCHEMA, script_sha256=c.EXPORTER_SHA256,
        parent_exporter_sha256=c.PARENT_EXPORTER_SHA256, original_run_sha256=c.ORIGINAL_RUN_SHA256,
        policy_sha256=c.POLICY_SHA256, input_manifest_sha256=c.INPUT_MANIFEST_SHA256,
        status='completed_grouped_Update_numerical_accepted_not_adopted', strict_FP32_gate_passed=True,
        candidate_accepted=False, strict_FP32_thresholds=dict(rtol=1e-4, atol=1e-5),
        strict_FP32_thresholds_relaxed=False, output_clipping=False,
        original_parameters_unchanged=True, original_weights_before=weights,
        original_weights_after=copy.deepcopy(weights), original_Update_parameters=parameters,
        outputs_finite=True, weight_domain_valid=True, cleanup_errors=[], affinity_restored=True,
        ORT_session_references_released=True, patches_per_frame=16, source_member_padding=False,
        neighbor_or_edge_removal=False, source_pins={'/synthetic-unavailable-board/source.py': 'b'*64},
        source_hashes_after={'/synthetic-unavailable-board/source.py': 'b'*64}, sources_and_inputs_unchanged=True,
        models={}, cases=[], strict_comparison_count=135)
    for batch, degree in c.ACTUAL_SHAPES:
        inputs, outputs = io(batch, degree)
        key = c.model_key(inputs, outputs)
        result['models'][key] = dict(inputs=inputs, outputs=outputs, batch=batch, degree=degree, patches=16,
            padding=False, external_data_used=False, numerical_validation_completed=True,
            parameter_coverage=dict(original_parameter_count=50, all_original_parameters_covered=True))
    gate = {name: dict(finite=True, gate_passed=True, failed_elements=0, rtol=1e-4, atol=1e-5)
            for name in ('net', 'delta', 'weight')}
    combinations = (((2, 4), (2, 5), (2, 6), (2, 7)),
                    ((2, 4), (2, 5), (3, 6), (2, 7)),
                    ((2, 4), (1, 5), (5, 6), (1, 7)))
    for (observation, call_index, edges), shapes in zip(c.CASE_IDENTITY, combinations):
        case = dict(observation_id=observation, frame_index=observation, original_call_index=call_index, edges=edges,
            Torch_domain=dict(finite=True, weight_domain_valid=True), ORT_domain=dict(finite=True, weight_domain_valid=True),
            degree_plan=dict(edges=edges, patches=16, all_edges_exactly_once=True,
                             original_edge_order_restored=True, padding=False, neural_calls=4), groups=[])
        for label in ('Torch_vs_original_full_gold', 'ORT_vs_original_full_gold', 'ORT_vs_grouped_Torch'):
            case[label] = copy.deepcopy(gate)
        for index, (batch, degree) in enumerate(shapes):
            group = dict(index=index, model_key='B%d_D%d_M16' % (batch, degree),
                topology=dict(sources=batch, degree=degree, patches=16, edges=batch*degree*16,
                              padding=False, dependency_groups_split=False))
            for label in ('Torch_vs_original_group_gold', 'ORT_vs_original_group_gold', 'ORT_vs_grouped_Torch'):
                group[label] = copy.deepcopy(gate)
            case['groups'].append(group)
        result['cases'].append(case)
    return result


def validate_generated(value, tmp_path, monkeypatch):
    path = tmp_path/'generated-manifest.json'
    path.write_text(json.dumps(value))
    digest = c.sha256(path)
    # Explicit synthetic override in this test module only, not source bytes.
    monkeypatch.setattr(c, 'EXPORT_MANIFEST_SHA256', digest)
    return c.validate_export_manifest(path, digest, verify_source_files=False)


def test_complete_metadata_host_does_not_require_board_source_files(tmp_path, monkeypatch):
    expected = manifest()
    report, pins = validate_generated(expected, tmp_path, monkeypatch)
    assert report == expected and len(pins) == 1
    assert len(report['models']) == 8 and report['candidate_accepted'] is False


@pytest.mark.parametrize('mutation', [
    lambda r: r['models'].pop('B1_D7_M16'),
    lambda r: r['models']['B2_D4_M16']['inputs'][0]['shape'].__setitem__(3, 48),
    lambda r: r['cases'].pop(),
    lambda r: r['cases'][0].update(observation_id=13),
    lambda r: r['cases'][1]['groups'].pop(),
    lambda r: r['cases'][2]['groups'][0]['ORT_vs_original_group_gold']['delta'].update(gate_passed=False, failed_elements=1),
    lambda r: r['original_weights_after']['update.p0'].update(tensor_sha256='c'*64),
    lambda r: r.update(candidate_accepted=True),
])
def test_reject_wrong_population_history_partial_dependency_or_changed_numerics(tmp_path, monkeypatch, mutation):
    value = manifest(); mutation(value)
    with pytest.raises(ValueError):
        validate_generated(value, tmp_path, monkeypatch)


def test_reject_another_manifest_pin_and_unseen_M16_shape(tmp_path):
    with pytest.raises(ValueError, match='actual v39'):
        c.validate_export_manifest(tmp_path/'not-read.json', 'f'*64)
    with pytest.raises(ValueError, match='actual M16'):
        c.contracts(*io(batch=8, degree=8))


def test_reuses_exact_parent_layout_and_identity_quantization_rules():
    inputs, outputs = io()
    logical, outlogical = c.contracts(inputs, outputs)
    rows = []
    for index, row in enumerate(inputs):
        b, channels, d, m = row['shape']; elems=b*channels*d*m
        rows.append(dict(index=index, name=row['name'], shape=[b,d,m,channels], format=1,
            type=1, quantization_type=2, scale=1, zero_point=0, fractional_length=0,
            n_elems=elems, size=elems*2, size_with_stride=elems*2))
    bindings = c.bind_attrs(rows, logical, output=False)
    assert bindings == c.load_parent().bind_attrs(rows, logical, output=False)
    assert all(row['input_permutation'] == (0,2,3,1) and row['transport_format'] == 1 for row in bindings)
    rows[0]['size_with_stride'] += 64
    with pytest.raises(ValueError, match='padded input'):
        c.bind_attrs(rows, logical, output=False)


class FakeToolkit:
    def __init__(self, fail=None):
        self.calls=[]; self.fail=fail
    def config(self, **kwargs):
        self.calls.append(('config', kwargs)); return -7 if self.fail=='config' else 0
    def load_onnx(self, **kwargs):
        self.calls.append(('load_onnx', kwargs)); return -7 if self.fail=='load_onnx' else 0
    def build(self, **kwargs):
        self.calls.append(('build', kwargs)); return -7 if self.fail=='build' else 0
    def export_rknn(self, filename):
        self.calls.append(('export_rknn', filename)); return -7 if self.fail=='export_rknn' else 0


def test_frozen_config_nonquantized_sequence_and_no_inference():
    toolkit, operations = FakeToolkit(), []
    compiler.compile_operations(toolkit, 'generated.onnx', 'generated.rknn', operations,
                                parent_compiler=PARENT_SOURCE/'compile-v37-grouped-update.py')
    assert [row[0] for row in toolkit.calls] == ['config','load_onnx','build','export_rknn']
    assert toolkit.calls[0][1] == dict(target_platform='rk3588', float_dtype='float16', optimization_level=3)
    assert toolkit.calls[2][1] == dict(do_quantization=False)
    assert [row['name'] for row in operations] == ['config','load_onnx','build','export_rknn']


@pytest.mark.parametrize('failure', ['config', 'load_onnx', 'build', 'export_rknn'])
def test_fake_compiler_error_propagates_and_stops(failure):
    toolkit, operations = FakeToolkit(failure), []
    with pytest.raises(RuntimeError, match=failure):
        compiler.compile_operations(toolkit, 'generated.onnx', 'generated.rknn', operations,
                                    parent_compiler=PARENT_SOURCE/'compile-v37-grouped-update.py')
    assert toolkit.calls[-1][0] == failure and operations[-1]['code'] == -7


def test_changed_parent_bytes_rejected_before_fake_toolkit(tmp_path):
    path=tmp_path/'changed-compiler.py';path.write_text('raise RuntimeError("must not import")\n')
    toolkit=FakeToolkit()
    with pytest.raises(ValueError, match='ancestor SHA'):
        compiler.compile_operations(toolkit,'generated.onnx','generated.rknn',[],parent_compiler=path)
    assert toolkit.calls == []
