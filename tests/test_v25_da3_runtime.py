"""Synthetic configuration and source guards; no SDK or model execution."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
VALIDATION = HERE if (HERE / 'run-v25-da3-rknn-board.py').is_file() else HERE.parent / 'deploy/rk3588/validation'
spec = importlib.util.spec_from_file_location('_synthetic_v25_runtime', VALIDATION / 'run-v25-da3-rknn-board.py')
RUN = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RUN
spec.loader.exec_module(RUN)
EXPORT = dict(onnx={'sha256': 'e' * 64})


def receipt(version=4):
    config = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3, do_quantization=False)
    sources = [RUN.COMPILER_SHA, RUN.GATHER_DISABLE_COMPILER_SHA,
               RUN.SHUFFLE_DISABLE_COMPILER_SHA, RUN.SWAP_DISABLE_COMPILER_SHA]
    rules = ['gather_to_conv', 'fuse_channel_shuffle_split_concat_to_gather', 'swap_slice_transpose']
    if version > 1:
        config['disable_rules'] = rules[:version - 1]
    return dict(status='compiled_not_runtime_validated', released=True,
                script_sha256=sources[version - 1], input={'sha256': 'e' * 64}, configuration=config,
                frozen_v24_r2_reference_sha256=RUN.SHUFFLE_DISABLE_COMPILER_SHA,
                frozen_v24_r1_reference_sha256=RUN.GATHER_DISABLE_COMPILER_SHA,
                frozen_v12_reference_sha256=RUN.COMPILER_SHA,
                operations=[dict(name=name, code=0) for name in ('config', 'load_onnx', 'build', 'export_rknn')],
                versions={'rknn-toolkit2': '2.3.2'})


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_only_exact_pinned_compiler_configuration_is_accepted(version):
    value = receipt(version)
    before = copy.deepcopy(value)
    actual = RUN.validate_compilation(value, EXPORT)
    assert value == before
    assert actual['script_sha256'] == value['script_sha256']
    assert actual['configuration'] == value['configuration']


@pytest.mark.parametrize('mutation', [
    'unknown_source', 'old_compiler_three_rules', 'new_compiler_two_rules',
    'extra_rule', 'wrong_rule_order', 'cpu_target', 'quantization',
    'missing_v24_r2', 'wrong_v24_r1', 'wrong_v12',
    'graph_changed', 'build_failed', 'release_failed',
])
def test_new_compiler_fails_closed_on_unapproved_change(mutation):
    value = receipt()
    if mutation == 'unknown_source':
        value['script_sha256'] = '0' * 64
    elif mutation == 'old_compiler_three_rules':
        value['script_sha256'] = RUN.SHUFFLE_DISABLE_COMPILER_SHA
    elif mutation == 'new_compiler_two_rules':
        value['configuration']['disable_rules'].pop()
    elif mutation == 'extra_rule':
        value['configuration']['disable_rules'].append('fuse_sdpa_to_exsdpa')
    elif mutation == 'wrong_rule_order':
        value['configuration']['disable_rules'].reverse()
    elif mutation == 'cpu_target':
        value['configuration']['op_target'] = {'exSDPAttention': 'cpu'}
    elif mutation == 'quantization':
        value['configuration']['do_quantization'] = True
    elif mutation == 'missing_v24_r2':
        del value['frozen_v24_r2_reference_sha256']
    elif mutation == 'wrong_v24_r1':
        value['frozen_v24_r1_reference_sha256'] = '1' * 64
    elif mutation == 'wrong_v12':
        value['frozen_v12_reference_sha256'] = '2' * 64
    elif mutation == 'graph_changed':
        value['input']['sha256'] = 'f' * 64
    elif mutation == 'build_failed':
        value['operations'][2]['code'] = -1
    else:
        value['release_error'] = 'synthetic release error'
    with pytest.raises(ValueError):
        RUN.validate_compilation(value, EXPORT)


def test_sdk_codec_and_timing_unchanged_from_frozen_v24_r3():
    ancestors = VALIDATION
    if not (ancestors / 'run-v24-da3-rknn-board-r3.py').is_file():
        ancestors = Path('/dev/shm/vggt-v15-public-snapshot/deploy/rk3588/validation')
    before_path = ancestors / 'run-v24-da3-rknn-board-r3.py'
    assert hashlib.sha256(before_path.read_bytes()).hexdigest() == RUN.RUNTIME_V24_R3_SHA

    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    before = functions(before_path)
    after = functions(VALIDATION / 'run-v25-da3-rknn-board.py')
    assert before.keys() == after.keys()
    assert [name for name in before if before[name] != after[name]] == ['validate_compilation', 'main']
    tree = ast.parse((VALIDATION / 'run-v25-da3-rknn-board.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'pin' and len(node.args) == 3
               and isinstance(node.args[1], ast.Name) and node.args[1].id == 'RUNTIME_V24_R3_SHA'
               for node in ast.walk(main))
