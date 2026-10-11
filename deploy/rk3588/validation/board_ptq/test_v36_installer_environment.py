"""Only build_environment is called; no installer, subprocess or model run."""
import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CANDIDATE = Path(os.environ.get('V36_ARM_INSTALLER', HERE / 'install-v36-arm-toolchain-board.py'))
PARENT = Path(os.environ.get('V35_ARM_INSTALLER', HERE / 'install-v35-arm-toolchain-board.py'))
if not CANDIDATE.is_file():
    CANDIDATE = HERE.parent / 'deploy/rk3588/validation/board_ptq/install-v36-arm-toolchain-board.py'
if not PARENT.is_file():
    PARENT = HERE.parent / 'deploy/rk3588/validation/board_ptq/install-v35-arm-toolchain-board.py'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize('existing', [None, '', '/existing/first:/existing/second'])
def test_only_local_runtime_path_added_and_other_compiler_settings_identical(existing, tmp_path):
    original = module(PARENT, 'v35_installer_env')
    revised = module(CANDIDATE, 'v36_installer_env')
    inherited = {'PATH': '/existing/bin', 'PYTHONHOME': '/blocked/home',
                 'PYTHONPATH': '/blocked/import', 'UNRELATED_VARIABLE': 'retained'}
    if existing is not None:
        inherited['LD_LIBRARY_PATH'] = existing
    inherited_before, os_before = dict(inherited), dict(os.environ)
    args = [tmp_path / name for name in ('new-venv', 'isolated-sysroot', 'new-work')]
    old_env = original.build_environment(*args, inherited)
    new_env = revised.build_environment(*args, inherited)
    prefix = str(args[1].resolve() / 'usr/lib/aarch64-linux-gnu')
    assert new_env['LD_LIBRARY_PATH'] == prefix + (os.pathsep + existing if existing else '')
    assert not new_env['LD_LIBRARY_PATH'].endswith(os.pathsep)
    assert dict(inherited) == inherited_before and dict(os.environ) == os_before
    assert new_env is not inherited and old_env is not inherited
    new_other = {k: v for k, v in new_env.items() if k != 'LD_LIBRARY_PATH'}
    old_other = {k: v for k, v in old_env.items() if k != 'LD_LIBRARY_PATH'}
    assert new_other == old_other
    assert 'PYTHONHOME' not in new_env and 'PYTHONPATH' not in new_env
    assert new_env['UNRELATED_VARIABLE'] == 'retained'
    assert '--sysroot=' + str(args[1].resolve()) in new_env['CFLAGS']
    assert new_env['CFLAGS'] == new_env['CXXFLAGS'] == new_env['LDFLAGS']
