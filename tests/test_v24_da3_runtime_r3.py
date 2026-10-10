"""Synthetic source/configuration guards for the isolated two-rule runtime."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
LOCAL = HERE if (HERE / 'run-v24-da3-rknn-board-r3.py').is_file() else HERE.parent / 'deploy/rk3588/validation'
spec = importlib.util.spec_from_file_location('_synthetic_v24_runtime_r3', LOCAL / 'run-v24-da3-rknn-board-r3.py')
RUN = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RUN
spec.loader.exec_module(RUN)


EXPORT = dict(onnx={'sha256': 'e' * 64})


def receipt(version=3):
    config = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3, do_quantization=False)
    source = RUN.COMPILER_SHA
    if version >= 2:
        config['disable_rules'] = ['gather_to_conv']
        source = RUN.GATHER_DISABLE_COMPILER_SHA
    if version == 3:
        config['disable_rules'].append('fuse_channel_shuffle_split_concat_to_gather')
        source = RUN.SHUFFLE_DISABLE_COMPILER_SHA
    return dict(status='compiled_not_runtime_validated', released=True,
                script_sha256=source, input={'sha256': 'e' * 64}, configuration=config,
                frozen_v24_r1_reference_sha256=RUN.GATHER_DISABLE_COMPILER_SHA,
                frozen_v12_reference_sha256=RUN.COMPILER_SHA,
                operations=[dict(name=name, code=0) for name in ('config', 'load_onnx', 'build', 'export_rknn')],
                versions={'rknn-toolkit2': '2.3.2'})


@pytest.mark.parametrize('version', [1, 2, 3])
def test_exact_frozen_compiler_configs_accepted_without_rewriting(version):
    value = receipt(version)
    before = copy.deepcopy(value)
    result = RUN.validate_compilation(value, EXPORT)
    assert value == before
    assert result['script_sha256'] == value['script_sha256']
    assert result['configuration'] == value['configuration']


@pytest.mark.parametrize('mutation', [
    'unknown_source', 'missing_rule', 'extra_rule', 'wrong_rule', 'rule_order',
    'cpu_target', 'fp32', 'quantization', 'failed_build', 'release', 'graph', 'sdk',
    'missing_v24_lineage', 'wrong_v24_lineage', 'wrong_v12_lineage',
    'legacy_with_two_rules', 'single_rule_compiler_with_two_rules',
])
def test_unapproved_rules_source_or_lifecycle_fail_closed(mutation):
    value = receipt()
    if mutation == 'unknown_source':
        value['script_sha256'] = '0' * 64
    elif mutation == 'missing_rule':
        value['configuration']['disable_rules'].pop()
    elif mutation == 'extra_rule':
        value['configuration']['disable_rules'].append('fuse_sdpa_to_exsdpa')
    elif mutation == 'wrong_rule':
        value['configuration']['disable_rules'][1] = 'fuse_sdpa_to_exsdpa'
    elif mutation == 'rule_order':
        value['configuration']['disable_rules'].reverse()
    elif mutation == 'cpu_target':
        value['configuration']['op_target'] = {'exSDPAttention': 'cpu'}
    elif mutation == 'fp32':
        value['configuration']['float_dtype'] = 'float32'
    elif mutation == 'quantization':
        value['configuration']['do_quantization'] = True
    elif mutation == 'failed_build':
        value['operations'][2]['code'] = -1
    elif mutation == 'release':
        value['released'] = False
    elif mutation == 'graph':
        value['input']['sha256'] = '1' * 64
    elif mutation == 'sdk':
        value['versions']['rknn-toolkit2'] = '2.3.1'
    elif mutation == 'missing_v24_lineage':
        del value['frozen_v24_r1_reference_sha256']
    elif mutation == 'wrong_v24_lineage':
        value['frozen_v24_r1_reference_sha256'] = '1' * 64
    elif mutation == 'wrong_v12_lineage':
        value['frozen_v12_reference_sha256'] = '2' * 64
    elif mutation == 'legacy_with_two_rules':
        value['script_sha256'] = RUN.COMPILER_SHA
    else:
        value['script_sha256'] = RUN.GATHER_DISABLE_COMPILER_SHA
    with pytest.raises(ValueError):
        RUN.validate_compilation(value, EXPORT)


def test_two_runtime_ancestors_still_exact():
    for name, expected in [('run-v24-da3-rknn-board.py', RUN.RUNTIME_PARENT_SHA),
                           ('run-v24-da3-rknn-board-r2.py', RUN.RUNTIME_PREVIOUS_SHA)]:
        assert hashlib.sha256((LOCAL / name).read_bytes()).hexdigest() == expected


def test_sdk_io_codec_and_array_validation_functions_unchanged():
    def functions(path):
        tree = ast.parse(path.read_text())
        return {node.name: ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, ast.FunctionDef)}
    before = functions(LOCAL / 'run-v24-da3-rknn-board-r2.py')
    after = functions(LOCAL / 'run-v24-da3-rknn-board-r3.py')
    assert before.keys() == after.keys()
    assert [name for name in before if before[name] != after[name]] == ['validate_compilation', 'main']
