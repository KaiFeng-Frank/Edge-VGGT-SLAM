"""Tiny/static build protocol checks; no actual board or compiler invocation."""
import importlib.util
from pathlib import Path
import platform

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v36_build_synthetic', HERE / 'build-v36-frame-corr-board.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_compile_flags_and_frozen_include_resolution():
    command = builder.build_command(Path('/candidate'), Path('/parent'), Path('/new/lib.so'),
                                    Path('/isolated/g++'), Path('/isolated/root'))
    assert command[0] == '/isolated/g++'
    assert command[-4:] == ['/parent', '/candidate/dpvo_frame_corr.cpp', '-o', '/new/lib.so']
    assert '-fopenmp' in command and '-fno-fast-math' in command and '-ffp-contract=off' in command
    assert '--sysroot=/isolated/root' in command
    assert '-B/isolated/root/usr/bin' in command
    assert '-B/isolated/root/usr/lib/gcc/aarch64-linux-gnu/11' in command
    assert '-ffast-math' not in command


def test_host_guard_before_loading_or_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.platform, 'machine', lambda: 'x86_64')
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='board only'):
        builder.main(['--source-dir', str(tmp_path/'absent-candidate'),
                      '--parent-dir', str(tmp_path/'absent-parent'),
                      '--toolchain-dir', str(tmp_path/'absent-toolchain'),
                      '--output-dir', str(output), '--source-freeze-sha256', '0'*64,
                      '--bootstrap-sha256', '0'*64])
    assert not output.exists()


def test_parent_source_identity_matches_cache_manifest():
    import json
    manifest = json.loads((HERE / 'cache-source-freeze.json').read_text())
    assert builder.PARENT_PINS == manifest['parent_source_sha256']
    assert all(builder.sha(HERE/name) == manifest['candidate_source_sha256'][name]
               for name in builder.CORE_FILES)
