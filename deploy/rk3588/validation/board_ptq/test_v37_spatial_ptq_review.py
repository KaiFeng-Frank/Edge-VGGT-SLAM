"""Small source/text checks; no calibration arrays, model or SDK."""
import ast
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "audit-v37-spatial-ptq.py"
if not SOURCE.is_file():
    SOURCE = HERE.parent / "deploy/rk3588/validation/board_ptq/audit-v37-spatial-ptq.py"
spec = importlib.util.spec_from_file_location("spatial_ptq_metadata_audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def tree(main_return=0, helper_return=1, config=True):
    source = "CONFIGURATION = {'same': " + str(config) + "}\n"
    source += "".join("def " + name + "():\n    return " + str(helper_return) + "\n"
                      for name in audit.UNCHANGED)
    source += "def main():\n    return " + str(main_return) + "\n"
    return ast.parse(source)


def test_source_derivation_only_main_changes():
    result = audit.compare_helpers(tree(), tree(main_return=1))
    assert tuple(result) == audit.UNCHANGED
    assert all(len(value) == 64 for value in result.values())


def test_changed_helper_rejected():
    with pytest.raises(ValueError, match="Frozen helper AST changed"):
        audit.compare_helpers(tree(), tree(main_return=1, helper_return=2))


def test_changed_config_rejected():
    with pytest.raises(ValueError, match="PTQ configuration AST changed"):
        audit.compare_helpers(tree(), tree(main_return=1, config=False))


def test_launcher_SHA_is_the_argument_not_other_hash_text(tmp_path):
    path = tmp_path / "launch.sh"
    path.write_text("exec python run.py --dataset raw.txt --dataset-sha256 raw_sha --output-dir resolved_sha\n")
    assert audit.launch_argument(path, "--dataset-sha256") == "raw_sha"
    path.write_text("run --dataset-sha256 first --dataset-sha256 second\n")
    with pytest.raises(ValueError, match="Exactly one launcher argument"):
        audit.launch_argument(path, "--dataset-sha256")
