"""Tiny NumPy-only tests; no saved real inputs, Torch or checkpoint imports."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HELPER = HERE / "audit-v26-dpvo-motion.py"
if not HELPER.is_file():
    HELPER = HERE.parent / "deploy" / "rk3588" / "validation" / HELPER.name
spec = importlib.util.spec_from_file_location("motion_auditor_synthetic", HELPER)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_oracle_fractional_xy_view_and_level_innermost():
    patches = np.zeros((1,1,2,3,3), np.float32)
    patches[0,0,:,1,1] = [1,2]
    features = np.empty((1,2,2,8,8), np.float32)
    features[0,0,0], features[0,0,1] = 3,5
    features[0,1,0], features[0,1,1] = 7,11
    coords = np.empty((1,2,2,3,3), np.float32)
    coords[:,:,0], coords[:,:,1] = 3.5,2.25
    value = audit.correlation_oracle(patches,features,coords,np.array([0,0]),np.array([0,1]),[0,1])
    grid = value.reshape(2,7,7,3,3,2)
    assert np.array_equal(grid[:,3,3,1,1], np.array([[13,13],[29,29]]))
    assert grid[0,3,0,1,1,0] == 13*.25
    assert grid[0,0,3,1,1,0] == 13
    assert value[1,((3*7+3)*3*3+1*3+1)*2+1] == 29
    assert np.count_nonzero(grid[:,:,:,0]) == 0


def test_oracle_zero_padding_outside_both_levels():
    patches = np.ones((1,1,2,3,3), np.float32)
    features = np.ones((1,1,2,8,8), np.float32)
    coords = np.full((1,1,2,3,3), -100, np.float32)
    actual = audit.correlation_oracle(patches,features,coords,np.array([0]),np.array([0]),[0])
    assert actual.shape == (1,882) and not actual.any()


def test_pool_non_square_crop_and_row_major_mean():
    value = np.arange(9*10,dtype=np.float32).reshape(1,1,1,9,10)
    actual = audit.mean_pool4(value)
    expected = np.empty((1,1,1,2,2),np.float32)
    for y in range(2):
        for x in range(2):
            expected[0,0,0,y,x] = value[0,0,0,y*4:y*4+4,x*4:x*4+4].mean()
    assert np.array_equal(actual,expected)


def test_artifact_raw_hash_mismatch_is_failure(tmp_path):
    values = np.arange(8,dtype=np.float32)
    path = tmp_path / "synthetic.npz"
    np.savez(path,values=values)
    metadata = audit.raw_metadata(values)
    metadata["tensor_sha256"] = "0"*64
    record = {"file":path.name,"bytes":path.stat().st_size,"sha256":audit.sha256(path),
              "arrays":{"values":metadata}}
    checks = audit.Checks()
    result = audit.artifact(tmp_path,record,checks,{})
    assert np.array_equal(result["values"],values)
    assert checks.failures == ["raw_metadata:synthetic.npz:values"]


def test_artifact_path_cannot_escape(tmp_path):
    with pytest.raises(ValueError,match="basename"):
        audit.safe_file(tmp_path,"../synthetic.npz")


def test_numeric_gate_does_not_hide_large_error_or_nan():
    assert audit.compare([1.0,1.01],[1.0,1.0])["outside_fixed_FP32_gate"] == 1
    with pytest.raises(ValueError,match="nonfinite"):
        audit.compare([np.nan],[0])


def test_output_refuse_overwrite_before_reading_inputs(tmp_path):
    output = tmp_path / "already.json"
    output.write_text("original\n")
    with pytest.raises(SystemExit) as caught:
        audit.main(["--run-dir",str(tmp_path),"--capture-dir",str(tmp_path),
                    "--source-dir",str(tmp_path),"--checkpoint",str(tmp_path/"no-model"),
                    "--output",str(output)])
    assert caught.value.code == 2
    assert output.read_text() == "original\n"
