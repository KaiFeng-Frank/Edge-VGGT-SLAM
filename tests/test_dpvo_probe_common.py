"""Synthetic loader/AST tests only; no released checkpoint or real images."""
import ast
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys

import pytest
import torch


HERE = Path(__file__).resolve().parent
IMPL = HERE if (HERE / 'dpvo_probe_common.py').is_file() else HERE.parent / 'deploy/rk3588/validation'
spec = importlib.util.spec_from_file_location('_dpvo_common_synthetic_test', IMPL / 'dpvo_probe_common.py')
COMMON = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = COMMON
spec.loader.exec_module(COMMON)
SOURCE = Path(os.environ.get('DPVO_SOURCE_DIR', str(IMPL / 'official'))).expanduser()
MANIFEST = Path(os.environ.get('DPVO_SOURCE_MANIFEST', str(IMPL / 'manifest.json'))).expanduser()
OPS = IMPL / 'dpvo_cpu_ops.py'


@pytest.fixture(scope='module')
def neural():
    if not SOURCE.is_dir() or not MANIFEST.is_file():
        pytest.skip('requires pinned DPVO source checkout and source manifest; set DPVO_SOURCE_DIR '
                    'and DPVO_SOURCE_MANIFEST; no released checkpoint is used')
    model, info = COMMON.construct_neural(SOURCE, MANIFEST, OPS,
                                         cpu_ops_sha256=COMMON.sha256(OPS))
    assert info['cpu_ops']['expected_sha256'] == COMMON.sha256(OPS)
    return model


def synthetic_state(model):
    result = OrderedDict(('module.' + k, v.detach().clone()) for k, v in model.state_dict().items())
    result.update(('module.' + k, torch.zeros(shape)) for k, shape in COMMON.LEGACY_SHAPES.items())
    return result


def test_preserved_class_ast_and_forward_body():
    source = 'class Gate:\n    def forward(self, x):\n        return x + x * 2\n\nclass Unused:\n    pass\n'
    tree, record = COMMON.extract_class_asts(source, ('Gate',))
    original = ast.parse(source).body[0]
    assert ast.dump(tree.body[0], include_attributes=False) == ast.dump(original, include_attributes=False)
    assert record['Gate']['original_class_AST_preserved'] is True
    assert len(tree.body) == 1


@pytest.mark.parametrize('source', ['class Other:\n    pass\n', 'class Gate:\n    pass\nclass Gate:\n    pass\n'])
def test_missing_or_duplicated_class_is_rejected(source):
    with pytest.raises(ValueError, match='one original class'):
        COMMON.extract_class_asts(source, ('Gate',))


def test_original_architecture_has_94_keys_without_alias_registration(neural):
    state = neural.state_dict()
    assert len(state) == 94
    assert sum(v.numel() for v in state.values()) == 3384324
    assert [sum(k.startswith(prefix) for k in state) for prefix in
            ('patchify.fnet.', 'patchify.inet.', 'update.')] == [22, 22, 50]
    assert neural.fnet is neural.patchify.fnet and neural.inet is neural.patchify.inet
    assert (neural.DIM, neural.P, neural.RES) == (384, 3, 4)


def test_strict_synthetic_98_to_94_mapping_preserves_all_bytes(neural):
    state = synthetic_state(neural)
    before = {k: COMMON.tensor_info(v) for k, v in state.items()}
    result = COMMON.strict_load_state(neural, state)
    assert result['strict'] is True and result['inference_parameter_coverage'] == 1.0
    assert result['all_loaded_parameter_bytes_equal'] is True
    assert result['exact_dropped_legacy_keys'] == list(COMMON.LEGACY_SHAPES)
    assert len(result['loaded_tensors']) == 94
    assert before == {k: COMMON.tensor_info(v) for k, v in state.items()}
    assert not any(p.requires_grad for p in neural.parameters())


@pytest.mark.parametrize('mutation', [
    'missing_key', 'extra_key', 'no_module_prefix', 'nested_module_prefix',
    'wrong_shape', 'float64', 'nan', 'not_tensor', 'legacy_wrong_shape', 'unauthorized_legacy',
])
def test_state_errors_fail_before_copying_any_synthetic_parameters(neural, mutation):
    state = synthetic_state(neural)
    first = next(iter(state))
    before = next(iter(neural.state_dict().values())).detach().clone()
    state[first].fill_(7)
    last = 'module.update.w.1.bias'
    if mutation == 'missing_key':
        del state[last]
    elif mutation == 'extra_key':
        state['module.update.surprise'] = torch.zeros(1)
    elif mutation == 'no_module_prefix':
        state[last.removeprefix('module.')] = state.pop(last)
    elif mutation == 'nested_module_prefix':
        state['module.' + last] = state.pop(last)
    elif mutation == 'wrong_shape':
        state[last] = torch.zeros(3)
    elif mutation == 'float64':
        state[last] = state[last].double()
    elif mutation == 'nan':
        state[last][0] = float('nan')
    elif mutation == 'not_tensor':
        state[last] = [0.0, 0.0]
    elif mutation == 'legacy_wrong_shape':
        state['module.update.lmbda.2.bias'] = torch.zeros(2)
    else:
        state['module.update.lmbda.extra'] = state.pop('module.update.lmbda.2.bias')
    with pytest.raises(ValueError):
        COMMON.strict_load_state(neural, state)
    assert torch.equal(before, next(iter(neural.state_dict().values())))


def test_host_guard_rejects_before_checkpoint_read_or_torch_load(monkeypatch):
    monkeypatch.setattr(COMMON.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('Real checkpoint load must not run'))
    with pytest.raises(ValueError, match='aarch64 Linux'):
        COMMON.load_neural('absent', 'absent', 'absent', COMMON.CHECKPOINT_SHA256, 'absent')


def test_weights_only_cpu_load_is_explicit():
    tree = ast.parse((IMPL / 'dpvo_probe_common.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_neural')
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'torch' and n.func.attr == 'load']
    assert len(calls) == 1
    keywords = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
    assert keywords == {'map_location': 'cpu', 'weights_only': True}


@pytest.mark.parametrize('changed', ['manifest', 'extractor'])
def test_modified_original_source_is_rejected(neural, tmp_path, changed):
    root = tmp_path / 'official'
    for key in COMMON.SOURCE_PINS:
        p = root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((SOURCE / key).read_bytes())
    manifest = tmp_path / 'manifest.json'
    manifest.write_bytes(MANIFEST.read_bytes())
    target = manifest if changed == 'manifest' else root / 'dpvo/extractor.py'
    target.write_bytes(target.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='manifest SHA|source bytes'):
        COMMON.validate_source(root, manifest)


def test_wrong_cpu_operator_pin_is_rejected(neural):
    with pytest.raises(ValueError, match='CPU operator source SHA'):
        COMMON.construct_neural(SOURCE, MANIFEST, OPS, cpu_ops_sha256='0' * 64)


def test_encoder_wrapper_on_tiny_synthetic_bgr_preserves_original_division(neural):
    # Random synthetic image tensor, not any saved real image or checkpoint.
    generator = torch.Generator().manual_seed(32)
    x = torch.rand((1, 2, 3, 16, 20), generator=generator) * 2 - 0.5
    before_keys = list(neural.state_dict())
    with torch.no_grad():
        f_expected = neural.patchify.fnet(x) / 4.0
        i_expected = neural.patchify.inet(x) / 4.0
        f, i = COMMON.encode(neural, x)
        wf, wi = COMMON.make_encoder(neural)(x)
    assert f.shape == (1, 2, 128, 4, 5) and i.shape == (1, 2, 384, 4, 5)
    assert torch.equal(f, f_expected) and torch.equal(i, i_expected)
    assert torch.equal(wf, f_expected) and torch.equal(wi, i_expected)
    assert before_keys == list(neural.state_dict())


@pytest.mark.parametrize('case', ['dtype', 'shape', 'range', 'nonfinite'])
def test_encoder_input_contract_rejects_bad_synthetic_input(neural, case):
    x = torch.zeros(1, 1, 3, 16, 20)
    if case == 'dtype':
        x = x.double()
    elif case == 'shape':
        x = x[..., :-1]
    elif case == 'range':
        x.fill_(2)
    else:
        x[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='normalized CPU FP32 BGR'):
        COMMON.encode(neural, x)
