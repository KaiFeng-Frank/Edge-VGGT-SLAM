"""Both frozen compiler configurations through the same fake SDK fixture.

No real compiler, model, ONNX runtime, image or hardware is used.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


HELPER_ROOT = Path(os.environ.get('V24_HELPER_ROOT', Path(__file__).resolve().parents[1]/'deploy/rk3588/validation'))
R2_PATH = HELPER_ROOT/'compile-v24-da3-rknn-r2.py'
if not R2_PATH.is_file():
    R2_PATH = Path(__file__).with_name('compile-v24-da3-rknn-r2.py')
TEST_PATH = Path(__file__).with_name('test_v24_da3_compile.py')
TEST_SHA = '735b0e41f485bef332990cb3296d93bf7ff05d26a975b274a4d2c5415f8e3154'
assert hashlib.sha256(TEST_PATH.read_bytes()).hexdigest() == TEST_SHA
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
previous = load('v24_compile_r2_frozen_test_fixture', TEST_PATH)
R1 = previous.probe
R2 = load('v24_compile_r2_synthetic', R2_PATH)
# Reuse the exact fake fixture; all per-test monkeypatches restore its namespace.
sdk = previous.sdk


@pytest.fixture(autouse=True, params=['r1','r2'])
def selected(request, monkeypatch):
    module = R1 if request.param == 'r1' else R2
    rules = ['gather_to_conv']
    if request.param == 'r2':
        rules += ['fuse_channel_shuffle_split_concat_to_gather']
    monkeypatch.setattr(previous, 'probe', module)
    return module, rules


def test_exact_known_rule_lists_and_original_config(selected, sdk):
    probe, rules = selected
    assert probe.main(sdk.argv) == 0
    report = json.loads((sdk.output/'compile.json').read_text())
    config = next(value for name,value in sdk.calls if name == 'config')
    assert config == {'target_platform':'rk3588','float_dtype':'float16',
                      'optimization_level':3,'disable_rules':rules}
    assert report['configuration'] == {**config, 'do_quantization':False}
    assert next(value for name,value in sdk.calls if name == 'build') == {'do_quantization':False}
    assert [name for name,_ in sdk.calls] == ['checker','constructor','config','load_onnx','build','export_rknn','release']
    assert report['status'] == 'compiled_not_runtime_validated' and report['released']
    assert report['script_sha256'] == hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()
    assert report['model_inference_executed'] is False
    if probe is R2:
        assert report['frozen_v24_r1_reference_sha256'] == hashlib.sha256(Path(R1.__file__).read_bytes()).hexdigest()


def test_failed_build_still_keeps_fixed_config_and_releases(selected, sdk):
    probe, rules = selected
    sdk.failure['build'] = RuntimeError('synthetic rewrite failure')
    assert probe.main(sdk.argv) == 1
    report = json.loads((sdk.output/'compile.json').read_text())
    assert report['status'] == 'failed' and report['released']
    assert report['configuration']['disable_rules'] == rules
    assert sdk.calls[-1] == ('release', {})


def test_unknown_rule_cannot_be_supplied_by_cli(selected, sdk):
    probe, _rules = selected
    with pytest.raises(SystemExit) as exc:
        probe.main(sdk.argv+['--disable-rules','unknown-rule'])
    assert exc.value.code == 2
    assert not sdk.output.exists() and not sdk.calls
