"""New diagnostic and output guards only; never import SDK or run calibration."""
import importlib.util
from pathlib import Path
import pytest

SOURCE = Path(__file__).with_name('compile-v37-spatial-int8-board.py')
spec = importlib.util.spec_from_file_location('spatial_ptq_guards', SOURCE)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

def arguments(tmp_path):
    args = []
    for name in ('dependency-plan', 'preflight-script', 'frozen-sparse-compiler', 'rewrite-dir',
                 'reference-dir', 'parent-onnx', 'editor', 'dataset', 'output-dir'):
        args += ['--' + name, str(tmp_path / name)]
    for name in ('dependency-plan', 'preflight-script', 'dataset'):
        args += ['--' + name + '-sha256', '0' * 64]
    return args

def test_rejected_parent_needs_explicit_flag_before_filesystem_or_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'board_guard', lambda: None)
    with pytest.raises(ValueError, match='Explicit rejected spatial-parent diagnostic required'):
        entry.main(arguments(tmp_path))
    assert list(tmp_path.iterdir()) == []

def test_diagnostic_output_isolated_to_board_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'board_guard', lambda: None)
    with pytest.raises(ValueError, match='New board experiment output required'):
        entry.main(arguments(tmp_path) + ['--diagnostic-allow-fp32-rejected'])
    assert list(tmp_path.iterdir()) == []
