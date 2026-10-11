"""Tiny generated descriptor guards; no native library, SDK or model calls."""
import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_v43_toy_full_native_audit', HERE/'audit-v43-native-full-ab.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def group(method):
    rows = []
    for index, (name, c) in enumerate(zip(('net', 'inp', 'corr'), (384, 384, 882))):
        row = dict(index=index, name=name, logical_shape=[1, c, 4, 16],
            source_dtype='float32', owned=True)
        if method == 'A':
            row.update(buffer_shape=[1, 4, 16, c], native_attribute_format=1,
                request_format=1, transport_dtype='float32', pass_through=0,
                bytes=c*4*16*4, explicit_NHWC_pack=True)
        else:
            row.update(buffer_shape=[1, (c+7)//8, 4, 16, 8], logical_attribute_format=1,
                native_attribute_format=2, native_query_commands=[8, 10], request_format=2,
                request_type=1, transport_dtype='float16', buffer_storage_dtype='uint16',
                pass_through=1, bytes=((c+7)//8)*4*16*8*2, input_transport=audit.NATIVE,
                padded_channels=((c+7)//8)*8-c, tail_padding_value='FP16_positive_zero',
                SDK_input_conversion_requested=False)
        rows.append(row)
    return dict(model_key='B1_D4_M16', timing=dict(input_records=rows), candidate_accepted=False,
        strict_FP32_gate_passed=None, strict_component_gate_passed=False)


def native_context():
    queries = []; bindings = []
    for index, (name, c) in enumerate(zip(('net', 'inp', 'corr'), (384, 384, 882))):
        shape = [1, (c+7)//8, 4, 16, 8]; elements = ((c+7)//8)*4*16*8
        attr = dict(index=index, name=name, n_dims=5, dims=shape+[0]*11, shape=shape,
            fmt=2, type=1, qnt_type=2, zp=0, scale=1., scale_finite=True,
            n_elems=elements, size=elements*2, size_with_stride=elements*2, w_stride=16, h_stride=0)
        for command in (8, 10):
            queries.append(dict(command=command, input_index=index, return_code=0,
                attribute=copy.deepcopy(attr)))
        bindings.append(dict(name=name, native_input_attribute=copy.deepcopy(attr),
            pass_through=1, transport_format=2, transport_dtype='float16',
            native_query_commands=[8, 10], transport_bytes=elements*2))
    return dict(native_input_queries=queries, input_bindings=bindings,
        native_inputs_validated=True, input_transport=audit.NATIVE, diagnostic_allow_native_fp16=True,
        transport_equivalence_validated=False,
        native_query_ancestor_sha256='89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642',
        native_input_query_log=dict(RKNN_error_detected=False, truncated=False),
        packing_FPCR_at_initialization=dict(supported=True, FPCR_modified_by_packer=False))


@pytest.mark.parametrize('method', ('A', 'B'))
def test_original_and_native_descriptor_shapes_are_distinct(method):
    assert audit.validate_pack_records(group(method), method) == 3


def test_native_pass_through_and_padded_channels_are_not_inferred():
    for field, invalid in (('pass_through', 0), ('padded_channels', 0)):
        value = group('B'); value['timing']['input_records'][2][field] = invalid
        with pytest.raises(ValueError): audit.validate_pack_records(value, 'B')


def test_transport_identity_never_overwrites_FP32_rejection():
    value = group('B'); value['strict_component_gate_passed'] = True
    with pytest.raises(ValueError): audit.validate_pack_records(value, 'B')


def test_six_queries_and_both_command_attributes_are_required():
    assert audit.validate_native_queries(native_context(), 'B1_D4_M16') == 6
    for mutation in ('missing', 'attribute', 'return_code'):
        value = native_context()
        if mutation == 'missing': value['native_input_queries'].pop()
        elif mutation == 'attribute': value['native_input_queries'][1]['attribute']['scale'] = .5
        else: value['native_input_queries'][1]['return_code'] = -1
        with pytest.raises(ValueError): audit.validate_native_queries(value, 'B1_D4_M16')


def test_unknown_model_is_not_padded_into_catalogue():
    with pytest.raises(ValueError): audit.dimensions('B2_D3_M16')
