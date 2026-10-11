"""Small source/host-refusal checks; no native function is executed."""
import ast
import importlib.util
from pathlib import Path
import platform

import pytest

path = Path(__file__).with_name("run-v45-dot4-corr-ab-board.py")


def test_host_refuses_before_saved_features_and_library_loading(tmp_path):
    if platform.machine() == "aarch64":
        pytest.skip("Host-refusal check requires non-ARM host")
    spec = importlib.util.spec_from_file_location("dot4_ab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = []
    for name in ("parent-ab-helper", "frame-helper", "capture-dir", "motion-dir",
                 "parent-library", "parent-build", "candidate-library", "candidate-build", "output-dir"):
        argv.extend(["--" + name, str(tmp_path / name)])
    argv.extend(["--parent-build-sha256", "0" * 64])
    with pytest.raises(ValueError, match="RK3588 board only"):
        module.main(argv)
    assert list(tmp_path.iterdir()) == []


def test_AB_changes_library_only_and_preserves_callback_class():
    text = path.read_text()
    tree = ast.parse(text)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    constructors = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "PerFrameCorrelation"]
    assert len(constructors) == 1
    assert "for method, library, digest in" in text
    assert "patch_count=96, threads=4" in text
    assert '(("A", "B") if repetition % 2 == 0 else ("B", "A"))' in text
    assert '"completed_numerical_rejected"' in text
    assert 'after["current_context_bytes"] == after["current_cached_frames"] == 0' in text


def test_four_round_schedule_is_ABBA_twice_with_three_warm_each():
    order = [m for r in range(4) for m in (("A", "B") if r % 2 == 0 else ("B", "A"))]
    assert order == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert [order[2*r:2*r+2].count("A") for r in range(1, 4)] == [1, 1, 1]
