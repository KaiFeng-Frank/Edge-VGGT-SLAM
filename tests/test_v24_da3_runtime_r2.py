"""Synthetic guards for the isolated single-rule compiler/runtime revision."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
LOCAL = HERE if (HERE / 'run-v24-da3-rknn-board-r2.py').is_file() else HERE.parent / 'deploy/rk3588/validation'
spec = importlib.util.spec_from_file_location('_synthetic_v24_runtime_r2', LOCAL / 'run-v24-da3-rknn-board-r2.py')
RUN = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RUN
spec.loader.exec_module(RUN)


def receipt(*, disabled=False):
    config = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3, do_quantization=False)
    if disabled:
        config['disable_rules'] = ['gather_to_conv']
    return dict(status='compiled_not_runtime_validated', released=True,
                script_sha256=RUN.GATHER_DISABLE_COMPILER_SHA if disabled else RUN.COMPILER_SHA,
                input={'sha256': 'e' * 64}, configuration=config,
                operations=[dict(name=name, code=0) for name in ('config', 'load_onnx', 'build', 'export_rknn')],
                versions={'rknn-toolkit2': '2.3.2'})


EXPORT = dict(onnx={'sha256': 'e' * 64})


@pytest.mark.parametrize('disabled', [False, True])
def test_only_exact_approved_compiler_configuration_passes(disabled):
    value = receipt(disabled=disabled)
    original = copy.deepcopy(value)
    result = RUN.validate_compilation(value, EXPORT)
    assert value == original  # Neither the compiler JSON nor configuration is rewritten.
    assert result['script_sha256'] == value['script_sha256']
    assert result['configuration'] == value['configuration']


@pytest.mark.parametrize('mutation', ['unknown_source', 'missing_rule', 'additional_rule', 'different_rule',
    'cpu_target', 'dtype', 'failed_build', 'cleanup', 'graph', 'sdk', 'legacy_with_rule'])
def test_unapproved_source_rule_graph_or_lifecycle_rejected(mutation):
    value = receipt(disabled=True)
    if mutation == 'unknown_source':
        value['script_sha256'] = '0' * 64
    elif mutation == 'missing_rule':
        del value['configuration']['disable_rules']
    elif mutation == 'additional_rule':
        value['configuration']['disable_rules'].append('fuse_sdpa_to_exsdpa')
    elif mutation == 'different_rule':
        value['configuration']['disable_rules'] = ['fuse_sdpa_to_exsdpa']
    elif mutation == 'cpu_target':
        value['configuration']['op_target'] = {'exSDPAttention': 'cpu'}
    elif mutation == 'dtype':
        value['configuration']['float_dtype'] = 'float32'
    elif mutation == 'failed_build':
        value['operations'][2]['code'] = -1
    elif mutation == 'cleanup':
        value['released'] = False
    elif mutation == 'graph':
        value['input']['sha256'] = '1' * 64
    elif mutation == 'sdk':
        value['versions']['rknn-toolkit2'] = '2.3.1'
    else:
        value['script_sha256'] = RUN.COMPILER_SHA
    with pytest.raises(ValueError):
        RUN.validate_compilation(value, EXPORT)


def test_frozen_runtime_ancestor_is_byte_exact():
    parent = LOCAL / 'run-v24-da3-rknn-board.py'
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == RUN.RUNTIME_PARENT_SHA
    assert len(RUN.GATHER_DISABLE_COMPILER_SHA) == 64
