import numpy as np
import pytest

from scripts.compare_predictions import compare


def write_prediction(path, frame="one.jpg", angle=0.0):
    path.mkdir()
    c, s = np.cos(angle), np.sin(angle)
    pose = np.array([[[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0]]], dtype=np.float32)
    np.savez(path / "predictions_0000.npz", frame_names=[frame],
             depth=np.ones((1, 2, 2, 1)), depth_conf=np.ones((1, 2, 2)),
             extrinsic=pose, intrinsic=np.eye(3)[None])


def test_identical_predictions_and_known_rotation(tmp_path):
    a, b, c = [tmp_path / name for name in "abc"]
    write_prediction(a, angle=0.3)
    write_prediction(b, angle=0.3)
    write_prediction(c, angle=0.4)
    same = compare(a, b)
    assert same["submaps"][0]["max_camera_rotation_difference_degrees"] < 1e-8
    assert same["submaps"][0]["depth"]["relative_l2_error"] == 0
    assert same["source_root"] is None
    assert not same["path_identity_validation_performed"]
    changed = compare(a, c)
    assert changed["submaps"][0]["max_camera_rotation_difference_degrees"] == pytest.approx(np.degrees(0.1), abs=1e-5)
    assert not changed["accuracy_validated"]


def test_different_inputs_cannot_be_compared(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_prediction(a)
    write_prediction(b, frame="other.jpg")
    with pytest.raises(ValueError, match="Input frame mismatch"):
        compare(a, b)
import hashlib
import json
from pathlib import Path
import subprocess
import sys



def test_same_image_relative_and_absolute_require_explicit_source_root(tmp_path):
    image = tmp_path / "datasets" / "office_loop" / "frame_0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"original image")
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    write_prediction(reference, frame="datasets/office_loop/frame_0001.jpg")
    write_prediction(candidate, frame=str(image))
    paths = [reference / "predictions_0000.npz", candidate / "predictions_0000.npz"]
    before = [hashlib.sha256(path.read_bytes()).digest() for path in paths]

    with pytest.raises(ValueError, match="Input frame mismatch"):
        compare(reference, candidate)
    result = compare(reference, candidate, source_root=tmp_path)
    assert result["source_root"] == str(tmp_path.resolve())
    assert result["path_identity_validation_performed"]
    assert result["submaps"][0]["depth"]["relative_l2_error"] == 0
    assert [hashlib.sha256(path.read_bytes()).digest() for path in paths] == before


@pytest.mark.parametrize("other_contents", [b"different pixels", b"same pixels"])
def test_same_basename_different_files_are_rejected_even_with_equal_contents(tmp_path, other_contents):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "frame_0001.jpg").write_bytes(b"same pixels")
    (two / "frame_0001.jpg").write_bytes(other_contents)
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    write_prediction(reference, frame="one/frame_0001.jpg")
    write_prediction(candidate, frame="two/frame_0001.jpg")
    with pytest.raises(ValueError, match="different source files"):
        compare(reference, candidate, source_root=tmp_path)


@pytest.mark.parametrize("missing_side", ["reference", "candidate"])
def test_missing_source_image_is_rejected(tmp_path, missing_side):
    image = tmp_path / "frame_0001.jpg"
    image.write_bytes(b"original image")
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    write_prediction(reference, frame="missing.jpg" if missing_side == "reference" else image.name)
    write_prediction(candidate, frame=str(tmp_path / "missing.jpg" if missing_side == "candidate" else image))
    with pytest.raises(ValueError, match="source image missing"):
        compare(reference, candidate, source_root=tmp_path)


def test_strict_identical_names_do_not_require_existing_image(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    write_prediction(reference, frame="absent.jpg")
    write_prediction(candidate, frame="absent.jpg")
    result = compare(reference, candidate, source_root=tmp_path)
    assert not result["path_identity_validation_performed"]


def test_cli_source_root_verifies_file_identity(tmp_path):
    image = tmp_path / "frame_0001.jpg"
    image.write_bytes(b"original image")
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    write_prediction(reference, frame=image.name)
    write_prediction(candidate, frame=str(image))
    output = tmp_path / "comparison.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_predictions.py"
    result = subprocess.run([sys.executable, str(script), str(reference), str(candidate),
                             "--source-root", str(tmp_path), "--output", str(output)],
                            capture_output=True, text=True, check=True)
    report = json.loads(output.read_text())
    assert report["path_identity_validation_performed"]
    assert json.loads(result.stdout) == report
