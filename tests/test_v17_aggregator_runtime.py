"""Synthetic provenance failures, using no SDK or real model."""
import copy
import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/run-v17-aggregator-rknn-board.py'
spec = importlib.util.spec_from_file_location('agg_runtime_v17', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def valid_compile():
    return dict(status='compiled_not_runtime_validated', script_sha256=m.COMPILER_SHA,
                model_inference_executed=False, input={'sha256': 'a' * 64}, output={'sha256': 'b' * 64},
                released=True, configuration=dict(target_platform='rk3588', float_dtype='float16',
                optimization_level=3, do_quantization=False), operations=[dict(name=n, code=0) for n in
                ('config', 'load_onnx', 'build', 'export_rknn')])


def test_valid_compile_chain():
    m.check_compile(valid_compile(), 'a' * 64, 'b' * 64)


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('released', False),
                                       ('model_inference_executed', True), ('script_sha256', 'x')])
def test_compile_invalid_status(field, value):
    r = valid_compile()
    r[field] = value
    with pytest.raises(ValueError):
        m.check_compile(r, 'a' * 64, 'b' * 64)


@pytest.mark.parametrize('which', ['input', 'output'])
def test_model_chain_mismatch(which):
    r = valid_compile()
    r[which]['sha256'] = 'c' * 64
    with pytest.raises(ValueError):
        m.check_compile(r, 'a' * 64, 'b' * 64)


def test_reject_quantized_configuration():
    r = valid_compile()
    r['configuration']['do_quantization'] = True
    with pytest.raises(ValueError):
        m.check_compile(r, 'a' * 64, 'b' * 64)


@pytest.mark.parametrize('alter', ['order', 'return_code', 'missing', 'cleanup'])
def test_incomplete_or_failed_compilation(alter):
    r = valid_compile()
    if alter == 'order':
        r['operations'].reverse()
    elif alter == 'return_code':
        r['operations'][2]['code'] = 1
    elif alter == 'missing':
        r['operations'].pop()
    else:
        r['release_error'] = 'SDK cleanup failed'
    with pytest.raises(ValueError):
        m.check_compile(r, 'a' * 64, 'b' * 64)


def test_binary_pins_are_valid_sha256():
    import re
    assert all(re.fullmatch('[0-9a-f]{64}', value) for value in
               (m.RUNTIME_SHA, m.COMPILER_SHA, m.LIBRARY_SHA))


def export_fixture():
    row = dict(artifacts={'tensors': {'sha256': 'c' * 64}}, weights={'weight': {'sha256': 'd' * 64}},
               arrays={k: {'sha256': v * 64} for k, v in [('input', 'a'), ('output', 'b'), ('pos', 'e')]})
    export = dict(status='exported_and_structurally_checked', script_sha256='producer',
                  profile_sha256='profile', block='frame0', input_shape=[2, 437, 1024],
                  input_name='input', output_name='output', original_weights=True,
                  parameter_count=18, position_and_tables_fixed=True, host_trigonometry_recomputed=False,
                  host_model_inference_executed=False, host_ORT_executed=False,
                  external_data=False, external_data_used=False,
                  capture_artifacts=copy.deepcopy(row['artifacts']), captured_array_metadata=copy.deepcopy(row['arrays']),
                  captured_parameter_metadata=copy.deepcopy(row['weights']),
                  artifacts={k + '.npy': dict(shape=[2, 437, 1024], dtype='float32', raw_sha256=row['arrays'][k]['sha256'])
                             for k in ('input', 'output')})
    return export, row


def test_valid_export_chain():
    export, row = export_fixture()
    m.check_export(export, 'producer', 'profile', 'frame0', row, (2, 437, 1024))


@pytest.mark.parametrize('alter', ['status', 'weight', 'pos', 'output', 'external'])
def test_reject_unbound_export(alter):
    export, row = export_fixture()
    if alter == 'status':
        export['status'] = 'started'
    elif alter == 'weight':
        export['captured_parameter_metadata']['weight']['sha256'] = 'f' * 64
    elif alter == 'pos':
        export['captured_array_metadata']['pos']['sha256'] = 'f' * 64
    elif alter == 'output':
        export['artifacts']['output.npy']['raw_sha256'] = 'f' * 64
    else:
        export['external_data'] = True
    with pytest.raises(ValueError):
        m.check_export(export, 'producer', 'profile', 'frame0', row, (2, 437, 1024))


def test_host_rejected_before_output(tmp_path):
    values = []
    for key in ('capture-dir', 'export-receipt', 'compile-receipt', 'model', 'output'):
        values.extend(['--' + key, str(tmp_path / key)])
    for key in ('profile-sha256', 'export-sha256', 'exporter-sha256', 'compile-sha256', 'model-sha256'):
        values.extend(['--' + key, '0' * 64])
    values.extend(['--block', 'frame0'])
    with pytest.raises(RuntimeError):
        m.main(values)
    assert not (tmp_path / 'output').exists()
