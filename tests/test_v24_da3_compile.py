"""Fake-SDK compiler protocol checks; no real ONNX/weights/compile/inference."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


HELPER_ROOT = Path(os.environ.get('V24_HELPER_ROOT', Path(__file__).resolve().parents[1]/'deploy/rk3588/validation'))
PATH = HELPER_ROOT/'compile-v24-da3-rknn.py'
if not PATH.is_file():
    PATH = Path(__file__).with_name('compile-v24-da3-rknn.py')
spec = importlib.util.spec_from_file_location('v24_compile_synthetic', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    calls = []
    versions = {'rknn-toolkit2':'2.3.2', 'torch':'2.4.0', 'numpy':'1.26.4',
                'onnx':'1.16.2', 'onnxruntime':'1.20.1', 'protobuf':'4.25.8'}
    failure = {}
    def invoke(name, kwargs):
        calls.append((name, kwargs))
        if name in failure:
            value = failure[name]
            if isinstance(value, Exception):
                raise value
            return value
        return 0
    class FakeRKNN:
        def __init__(self, **kwargs):
            calls.append(('constructor', kwargs))
        def config(self, **kwargs):
            return invoke('config', kwargs)
        def load_onnx(self, **kwargs):
            return invoke('load_onnx', kwargs)
        def build(self, **kwargs):
            return invoke('build', kwargs)
        def export_rknn(self, path):
            code = invoke('export_rknn', {'path':path})
            if code == 0:
                Path(path).write_bytes(b'fake-SDK-artifact; not a usable model')
            return code
        def release(self):
            return invoke('release', {})
    torch = SimpleNamespace(__version__='2.4.0+cpu')
    numpy = SimpleNamespace(__version__='1.26.4')
    onnx = SimpleNamespace(checker=SimpleNamespace(check_model=lambda path, **kwargs: calls.append(('checker', {'path':path, **kwargs}))))
    package = ModuleType('rknn')
    api = ModuleType('rknn.api')
    api.RKNN = FakeRKNN
    package.api = api
    for name, value in [('torch',torch), ('numpy',numpy), ('onnx',onnx), ('rknn',package), ('rknn.api',api)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(probe.importlib.metadata, 'version', versions.__getitem__)
    graph = tmp_path/'synthetic-not-real.onnx'
    graph.write_bytes(b'fake graph consumed only by fake checker')
    output = tmp_path/'compile-result'
    argv = ['--onnx',str(graph),'--onnx-sha256',hashlib.sha256(graph.read_bytes()).hexdigest(),'--output-dir',str(output)]
    return SimpleNamespace(calls=calls, versions=versions, failure=failure, graph=graph,
                           output=output, argv=argv, torch=torch, numpy=numpy)


def test_only_named_rule_disabled_and_original_fp16_settings_preserved(sdk):
    assert probe.main(sdk.argv) == 0
    report = json.loads((sdk.output/'compile.json').read_text())
    config = next(value for name,value in sdk.calls if name == 'config')
    assert config == {'target_platform':'rk3588','float_dtype':'float16','optimization_level':3,
                      'disable_rules':['gather_to_conv']}
    assert report['configuration'] == {**config, 'do_quantization':False}
    assert next(value for name,value in sdk.calls if name == 'build') == {'do_quantization':False}
    assert [name for name,_ in sdk.calls] == ['checker','constructor','config','load_onnx','build','export_rknn','release']
    assert sdk.calls[0][1]['full_check'] is True
    assert report['status'] == 'compiled_not_runtime_validated' and report['released']
    assert report['model_inference_executed'] is False and report['production_changed'] is False
    assert report['input']['sha256'] == hashlib.sha256(sdk.graph.read_bytes()).hexdigest()
    assert report['output']['sha256'] == hashlib.sha256((sdk.output/'block.rknn').read_bytes()).hexdigest()
    assert report['script_sha256'] == hashlib.sha256(PATH.read_bytes()).hexdigest()
    assert report['frozen_v12_reference_sha256'] == 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'


@pytest.mark.parametrize('stage,error', [('config',23),('load_onnx',1),('build',RuntimeError('synthetic build error')),('export_rknn',2)])
def test_sdk_failure_writes_failed_receipt_and_releases(sdk, stage, error):
    sdk.failure[stage] = error
    assert probe.main(sdk.argv) == 1
    report = json.loads((sdk.output/'compile.json').read_text())
    assert report['status'] == 'failed' and report['released']
    assert sdk.calls[-1] == ('release', {})
    assert not (sdk.output/'block.rknn').exists()
    assert report['configuration']['disable_rules'] == ['gather_to_conv']


def test_wrong_input_sha_rejected_before_output_or_sdk(sdk):
    sdk.argv[3] = '0'*64
    with pytest.raises(SystemExit) as exc:
        probe.main(sdk.argv)
    assert exc.value.code == 2
    assert not sdk.output.exists() and not sdk.calls


@pytest.mark.parametrize('wrong_version', ['torch', 'numpy', 'rknn-toolkit2'])
def test_toolchain_version_guards_unchanged(sdk, wrong_version):
    if wrong_version == 'torch':
        sdk.torch.__version__ = '2.5.0'
    elif wrong_version == 'numpy':
        sdk.numpy.__version__ = '2.0.0'
    else:
        sdk.versions[wrong_version] = '2.4.0'
    assert probe.main(sdk.argv) == 1
    report = json.loads((sdk.output/'compile.json').read_text())
    assert report['status'] == 'failed' and not sdk.calls
    assert report['model_inference_executed'] is False
