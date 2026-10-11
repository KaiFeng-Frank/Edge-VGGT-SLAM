"""Tiny arithmetic/early-guard tests. Never execute the board probe/model."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
SOURCE=HERE/"run-v27-dpvo-corr-board.py"
if not SOURCE.is_file():
    SOURCE=HERE.parent/"deploy"/"rk3588"/"validation"/SOURCE.name
spec=importlib.util.spec_from_file_location("v27_corr_driver_synthetic",SOURCE)
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def arguments(tmp_path):
    result=[]
    for key in ("v26-source-dir","capture-dir","motion-dir","checkpoint",
                "native-source-dir","library","build-receipt"):
        result += ["--"+key,str(tmp_path/key)]
    return result+["--output-dir",str(tmp_path/"must-not-exist"),"--build-sha256","0"*64]


def test_fractional_shift_per_edge_exact_and_original_unchanged():
    original=np.arange(1*14*2*2*3,dtype=np.float32).reshape(1,14,2,2,3)/np.float32(8)
    before=original.tobytes()
    shifted=driver.shifted_coords(original)
    expected=original.copy()
    for edge in range(14):
        expected[0,edge,0]+=np.float32(.17)*np.float32(edge%7-3)
        expected[0,edge,1]+=np.float32(.23)*np.float32(edge%5-2)
    assert shifted.dtype==np.float32 and shifted.tobytes()==expected.tobytes()
    assert original.tobytes()==before and not np.shares_memory(shifted,original)
    assert shifted[0,0,0,0,0] < original[0,0,0,0,0]


def test_comparison_fixed_gate_and_invalid_references():
    reference=np.array([1,1],np.float32)
    close=np.array([1,1.00005],np.float32)
    assert driver.comparison(close,reference)["gate_passed"]
    far=np.array([1,1.01],np.float32)
    result=driver.comparison(far,reference)
    assert not result["gate_passed"] and result["failed_elements"]==1
    assert result["rtol"]==1e-4 and result["atol"]==1e-5
    for bad in (reference.astype(np.float64),np.array([np.nan,1],np.float32),reference[:1]):
        with pytest.raises(ValueError,match="Finite matching FP32"):
            driver.comparison(bad,reference)


def test_non_arm_guard_precedes_output_and_affinity_or_helper_load(tmp_path,monkeypatch):
    monkeypatch.setattr(driver.platform,"machine",lambda:"x86_64")
    def forbidden(*args,**kwargs):
        raise AssertionError("guard must precede side effects")
    monkeypatch.setattr(driver.os,"sched_getaffinity",forbidden)
    monkeypatch.setattr(driver,"module",forbidden)
    with pytest.raises(ValueError,match="RK3588 board only"):
        driver.main(arguments(tmp_path))
    assert not (tmp_path/"must-not-exist").exists()


def test_arm_non_rockchip_guard_precedes_output(tmp_path,monkeypatch):
    monkeypatch.setattr(driver.platform,"machine",lambda:"aarch64")
    real_path=Path
    def paths(value):
        if value=="/proc/device-tree/compatible":
            return SimpleNamespace(read_bytes=lambda:b"another-board\0")
        return real_path(value)
    monkeypatch.setattr(driver,"Path",paths)
    with pytest.raises(ValueError,match="RK3588 board only"):
        driver.main(arguments(tmp_path))
    assert not (tmp_path/"must-not-exist").exists()
