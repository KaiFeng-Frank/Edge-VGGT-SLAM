"""Fake-parent wrapper ownership/report guards; no neural or native execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ['V36_FRAME_CACHE_WRAPPER']) if 'V36_FRAME_CACHE_WRAPPER' in os.environ else HERE / 'run-v36-fast-frame-cache-board.py'
if not SOURCE.is_file():
    SOURCE = HERE.parent / 'deploy/rk3588/validation/dpvo_frame_cache/run-v36-fast-frame-cache-board.py'
spec = importlib.util.spec_from_file_location('v36_fast_frame_cache_wrapper', SOURCE)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def fake_run(tmp_path, monkeypatch, *, parent_code=0, parent_error=False,
             closed=True, context_bytes=0, mutate_source=False, patch_count=48,
             validation_changes=None):
    replay, helper_file, library, old_library = [tmp_path / s for s in ('parent.py', 'helper.py', 'candidate.so', 'old.so')]
    for path in (replay, helper_file, library, old_library):
        path.write_bytes(b'fake artifact only')
    real_sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    build = tmp_path / 'build.json'
    build.write_text(json.dumps(dict(status='compiled_and_synthetic_smoke_passed', cleanup_errors=[],
        sources_and_inputs_unchanged=True, candidate_source_sha256={'dpvo_frame_corr.py': real_sha(helper_file)},
        parent_source_sha256={'dpvo_corr.cpp': '3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990',
                             'dpvo_corr.h': '1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932'},
        output={'sha256': real_sha(library)})))
    validation = tmp_path / 'AB.json'
    value = dict(status='completed_saved_array_correlation_AB_probe', all_fixed_FP32_gates_passed=True,
                 old_new_bitwise_equal=True, cleanup_errors=[], sources_and_inputs_unchanged=True,
                 incremental_preparation_checks_passed=True, candidate_build_sha256=real_sha(build),
                 candidate_library_sha256=real_sha(library))
    value.update(validation_changes or {})
    validation.write_text(json.dumps(value))
    output = tmp_path / 'output'
    calls, constructed = [], []
    sentinel = object()
    parent = SimpleNamespace(CompactCorrelation=sentinel)

    class Cache:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

        def metrics(self):
            return {'closed': closed, 'current_context_bytes': context_bytes}

    def parent_main(argv):
        calls.append(list(argv))
        assert parent.CompactCorrelation is not sentinel
        output.mkdir()
        parent.CompactCorrelation(object(), old_library, patch_count)
        status = 'completed_unpaced_FAST_NPU_encoder_diagnostic_not_realtime' if parent_code == 0 else 'failed'
        run = dict(status=status, cleanup_errors=[], frames_completed=5, first_three_FP32_gate_passed=False,
                   fake_preserved={'a': [1, 2]}, error='fake parent rejection' if parent_code else None)
        raw = json.dumps(run, indent=1).encode() + b'\n'
        (output / 'run.json').write_bytes(raw)
        parent.saved_raw = raw
        if mutate_source:
            helper_file.write_bytes(b'changed fake helper')
        if parent_error:
            raise RuntimeError('intentional parent failure')
        return parent_code

    parent.main = parent_main
    monkeypatch.setattr(wrapper, 'module', lambda p, name: parent if p == replay else SimpleNamespace(PerFrameCorrelation=Cache))
    monkeypatch.setattr(wrapper, 'sha', lambda p: wrapper.PARENT_SHA if Path(p) == replay else
                        wrapper.PARENT_LIBRARY_SHA if Path(p) == old_library else real_sha(p))
    monkeypatch.setattr(wrapper.platform, 'machine', lambda: 'aarch64')
    monkeypatch.setattr(wrapper.platform, 'system', lambda: 'Linux')
    real_read = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'rockchip,rk3588s\0' if str(p) == '/proc/device-tree/compatible' else real_read(p))
    args = ['--replay-driver', str(replay), '--frame-cache-helper', str(helper_file),
            '--frame-cache-library', str(library), '--frame-cache-build', str(build),
            '--frame-cache-validation', str(validation)]
    for flag, path in [('helper', helper_file), ('library', library), ('build', build), ('validation', validation)]:
        args += [f'--frame-cache-{flag}-sha256', real_sha(path)]
    args += ['--output-dir', str(output), '--fake-forwarded-arg', 'unchanged']
    return args, parent, sentinel, output, calls, constructed


def test_success_replaces_once_restores_and_preserves_raw(monkeypatch, tmp_path):
    args, parent, original, output, calls, constructed = fake_run(tmp_path, monkeypatch)
    assert wrapper.main(args) == 0
    assert len(calls) == len(constructed) == 1
    assert calls[0][-2:] == ['--fake-forwarded-arg', 'unchanged']
    assert parent.CompactCorrelation is original
    _, kwargs = constructed[0]
    assert kwargs['patch_count'] == 48 and kwargs['threads'] == 4 and kwargs['max_frames'] == 64
    assert kwargs['max_bytes'] == 512 * 1024 * 1024
    assert (output / 'parent-run.json').read_bytes() == parent.saved_raw
    report = json.loads((output / 'wrapper.json').read_text())
    rewritten = json.loads((output / 'run.json').read_text())
    assert report['status'] == 'completed_full_replay_with_per_frame_cache'
    assert report['parent_constructor_restored'] and report['sources_unchanged']
    assert report['original_FP32_gate_passed'] is False
    assert rewritten['parent_raw_receipt_sha256'] == hashlib.sha256(parent.saved_raw).hexdigest()
    assert rewritten['fake_preserved'] == {'a': [1, 2]}
    assert rewritten['first_three_FP32_gate_passed'] is False
    assert 'not_realtime' in rewritten['status']


@pytest.mark.parametrize('parent_code,parent_error', [(1, False), (0, True)])
def test_parent_failure_not_relabelled_success(monkeypatch, tmp_path, parent_code, parent_error):
    args, parent, original, output, calls, _ = fake_run(tmp_path, monkeypatch, parent_code=parent_code, parent_error=parent_error)
    assert wrapper.main(args) == 1
    assert len(calls) == 1 and parent.CompactCorrelation is original
    report = json.loads((output / 'wrapper.json').read_text())
    assert report['status'] == 'failed' and report['parent_constructor_restored']
    assert (output / 'run.json').read_bytes() == parent.saved_raw
    assert not (output / 'parent-run.json').exists()


@pytest.mark.parametrize('closed,context_bytes,error', [(False, 0, 'closed per-frame cache'), (True, 1, 'fully released')])
def test_close_and_zero_context_guards(monkeypatch, tmp_path, closed, context_bytes, error):
    args, parent, original, output, _, _ = fake_run(tmp_path, monkeypatch, closed=closed, context_bytes=context_bytes)
    assert wrapper.main(args) == 1
    report = json.loads((output / 'wrapper.json').read_text())
    assert report['status'] == 'failed' and error in report['error']
    assert parent.CompactCorrelation is original
    assert (output / 'run.json').read_bytes() == parent.saved_raw


def test_unmatched_patch_count_rejected_and_restored(monkeypatch, tmp_path):
    args, parent, original, output, calls, constructed = fake_run(tmp_path, monkeypatch, patch_count=96)
    assert wrapper.main(args) == 1
    assert len(calls) == 1 and not constructed
    assert parent.CompactCorrelation is original
    report = json.loads((output / 'wrapper.json').read_text())
    assert report['status'] == 'failed' and 'M48' in report['error']


def test_failed_AB_gate_blocks_parent_and_output(monkeypatch, tmp_path):
    args, parent, original, output, calls, _ = fake_run(tmp_path, monkeypatch,
                                                     validation_changes={'all_fixed_FP32_gates_passed': False})
    with pytest.raises(ValueError, match='AB validation'):
        wrapper.main(args)
    assert not calls and not output.exists() and parent.CompactCorrelation is original


@pytest.mark.parametrize('validation_changes', [
    {'candidate_build_sha256': '0' * 64},
    {'candidate_library_sha256': '0' * 64},
    {'incremental_preparation_checks_passed': False},
])
def test_AB_lineage_and_preparation_mismatch_prevents_parent(monkeypatch, tmp_path, validation_changes):
    args, parent, original, output, calls, constructed = fake_run(tmp_path, monkeypatch,
                                                                 validation_changes=validation_changes)
    with pytest.raises(ValueError, match='AB validation'):
        wrapper.main(args)
    assert not calls and not constructed and not output.exists()
    assert parent.CompactCorrelation is original


def test_changed_source_rejected_without_overwriting_parent(monkeypatch, tmp_path):
    args, parent, original, output, _, _ = fake_run(tmp_path, monkeypatch, mutate_source=True)
    assert wrapper.main(args) == 1
    report = json.loads((output / 'wrapper.json').read_text())
    assert report['status'] == 'failed' and report['sources_unchanged'] is False
    assert parent.CompactCorrelation is original
    assert (output / 'run.json').read_bytes() == parent.saved_raw
