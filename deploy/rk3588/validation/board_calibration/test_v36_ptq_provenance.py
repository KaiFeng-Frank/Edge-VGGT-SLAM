"""Tiny metadata binding tests; synthetic arrays, never image/model execution."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ENTRY = Path(__file__).parent / "audit-v36-ptq-provenance.py"
if not ENTRY.exists():
    ENTRY = Path(__file__).parents[1] / "deploy/rk3588/validation/board_ptq/audit-v36-ptq-provenance.py"
spec = importlib.util.spec_from_file_location("v36_ptq_provenance_tests", ENTRY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    centers = np.ones((32, 96, 2), dtype=np.int64)
    raw_sha = hashlib.sha256(np.full(96, 153, np.int64).tobytes()).hexdigest()
    generated, compiled, pins = [], [], {}
    for i in range(32):
        files = [dict(file=f"{name}-{i:02d}.npy", bytes=size,
            sha256=hashlib.sha256(f"synthetic-{name}-{i}".encode()).hexdigest())
            for name, size in (("image", 3385472), ("indices", 896))]
        sample = dict(calibration_index=i, source_index=256 + 16*i,
            timestamp_text=str(i + 1000), source_image_sha256="1"*64, input_files=copy.deepcopy(files),
            image_array=dict(shape=[1, 3, 464, 608], dtype="float32"),
            index_array=dict(shape=[96], dtype="int64", tensor_sha256=raw_sha))
        generated.append(sample)
        absolute = [item | {"file": "/synthetic/" + item["file"]} for item in files]
        compiled.append(dict(index=i, input_files=absolute, image_dtype="float32", image_shape=[1, 3, 464, 608],
            indices_dtype="int64", indices_shape=[96], index_order_preserved=True, index_cast_performed=False,
            image_min=-.5, image_max=1.5, index_min=153, index_max=153))
        pins.update({item["file"]: item["sha256"] for item in absolute})
    return compiled, generated, centers, pins


def test_all_fixed_rows_bind_without_reading_npy_bodies():
    args = fixture()
    checks = module.Checks()
    total, bindings = module.check_rows(checks, *args)
    assert checks.failures == []
    assert total == 108363776 and len(bindings) == 32
    assert all(row["source_index"] >= 256 for row in bindings)


@pytest.mark.parametrize("mutation", ["sha", "input_order", "dtype", "selection", "index_extrema"])
def test_tampered_metadata_is_rejected(mutation):
    compiled, generated, centers, pins = fixture()
    if mutation == "sha":
        compiled[0]["input_files"][1]["sha256"] = "0"*64
    elif mutation == "input_order":
        compiled[0]["input_files"].reverse()
    elif mutation == "dtype":
        compiled[0]["indices_dtype"] = "float32"
        compiled[0]["index_cast_performed"] = True
    elif mutation == "selection":
        generated[0]["source_index"] = 191
    else:
        compiled[0]["index_min"] = 152
    checks = module.Checks()
    module.check_rows(checks, compiled, generated, centers, pins)
    assert checks.failures
