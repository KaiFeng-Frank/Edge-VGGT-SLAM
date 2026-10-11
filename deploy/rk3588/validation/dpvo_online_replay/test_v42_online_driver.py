"""Static derivation and CLI only; no board, models, SDK, ORT or images."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest

TEST=Path(__file__).resolve().parent
HERE=TEST if (TEST/'run-v42-online-npu-board.py').is_file() else Path(os.environ['V42_ONLINE_SOURCE'])
LEGACY=Path(os.environ.get('V42_V41_DRIVER',HERE.parent/'public-snapshot/deploy/rk3588/validation/dpvo_online_replay/run-v41-online-npu-board.py'))
spec=importlib.util.spec_from_file_location('_v42_cli_static_test',HERE/'run-v42-online-npu-board.py')
driver=importlib.util.module_from_spec(spec);sys.modules[spec.name]=driver;spec.loader.exec_module(driver)


def trees():
    assert hashlib.sha256(LEGACY.read_bytes()).hexdigest()==driver.V41_ONLINE_DRIVER
    return ast.parse(LEGACY.read_text()),ast.parse((HERE/'run-v42-online-npu-board.py').read_text())


def test_all_trajectory_route_health_and_SDK_count_helpers_unchanged():
    old,new=trees()
    helper=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body
        if isinstance(n,ast.FunctionDef) and n.name not in ('parse_args','main')}
    assert helper(old)==helper(new)


def test_entire_fresh_frame_loop_and_finally_cleanup_unchanged():
    old,new=trees()
    main=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    old,new=main(old),main(new)
    selected=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.For)
        and isinstance(n.target,ast.Tuple) and [v.id for v in n.target.elts]==['index','item'])
    assert ast.dump(selected(old),include_attributes=False)==ast.dump(selected(new),include_attributes=False)
    outer=lambda t:next(n for n in t.body if isinstance(n,ast.Try))
    assert [ast.dump(n,include_attributes=False) for n in outer(old).finalbody]==[ast.dump(n,include_attributes=False) for n in outer(new).finalbody]
    eof=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
        and n.value.func.attr=='terminate')
    # Termination is assigned in the original flow; check every such call.
    calls=lambda t:[ast.dump(n,include_attributes=False) for n in ast.walk(t)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='terminate']
    assert calls(old)==calls(new) and len(calls(new))==1


def test_cli_retains_original_options_and_requires_parent_online_source():
    flags=('parent-driver','policy','torch-encoder','source-dir','checkpoint','geometry','stream-module',
        'encoder-module','frame-cache-helper','library','build-receipt','encoder-capture-dir','dataset-dir',
        'output-dir','cached-cpu-driver','online-factory','online-config','undistort-helper')
    argv=[value for name in flags for value in ('--'+name,'fake')]+['--online-factory-sha256','x','--online-config-sha256','y']
    with pytest.raises(SystemExit):driver.parse_args(argv)
    parsed=driver.parse_args(argv+['--v41-online-driver','legacy','--diagnostic-allow-fp32-rejected'])
    assert parsed.v41_online_driver==Path('legacy') and parsed.diagnostic_allow_fp32_rejected
    assert parsed.npu_encoder is None


def test_new_runtime_sources_sha_match_frozen_files():
    sha=lambda name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    assert driver.ONLINE_FACTORY==sha('v42_online_factory.py')
    assert driver.ONLINE_CALLBACK==sha('v42_online_update.py')
    assert driver.ONLINE_CONTRACTS==sha('v42_online_contracts.py')
