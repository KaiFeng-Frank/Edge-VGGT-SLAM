"""Tiny saved-artifact auditor checks; no model, images or native execution."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

_HERE = Path(__file__).resolve().parent
_CANDIDATES = (
    _HERE / "audit-v27-dpvo-corr.py",
    _HERE.parent / "deploy/rk3588/validation/audit-v27-dpvo-corr.py",
)
_HELPER = next((path for path in _CANDIDATES if path.is_file()), None)
if _HELPER is None:
    raise ImportError("Missing audit-v27-dpvo-corr.py; provide the published validation helper")
_SPEC = importlib.util.spec_from_file_location("v27_saved_corr_auditor", _HELPER)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class SavedAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="v27-audit-toy-", dir="/dev/shm")
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.array = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        path = self.folder / "toy.npz"
        np.savez(path, corr=self.array)
        self.record = {"file": path.name, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "arrays": {"corr": audit.metadata(self.array)}}

    def test_saved_array_owned_and_metadata_exact(self):
        values = audit.artifact(self.folder, self.record)
        self.assertTrue(audit.same_bytes(values["corr"], self.array))
        values["corr"].fill(-1)
        self.assertTrue(audit.same_bytes(audit.artifact(self.folder, self.record)["corr"], self.array))

    def test_file_SHA_tampering_rejected(self):
        path = self.folder / self.record["file"]
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "file SHA"):
            audit.artifact(self.folder, self.record)

    def test_shape_metadata_tampering_rejected(self):
        self.record["arrays"]["corr"]["shape"] = [1, 4, 2]
        with self.assertRaisesRegex(ValueError, "array metadata"):
            audit.artifact(self.folder, self.record)

    def test_raw_tensor_SHA_tampering_rejected(self):
        self.record["arrays"]["corr"]["tensor_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "array metadata"):
            audit.artifact(self.folder, self.record)

    def test_fixed_numerical_gate_and_record_tampering(self):
        expected = np.array([1, 2], np.float32)
        actual = np.array([1.001, 2.000001], np.float32)
        result = audit.comparison(actual, expected)
        self.assertEqual(result["failed_elements"], 1)
        self.assertFalse(result["gate_passed"])
        audit.recorded_comparison(result, dict(result))
        forged = dict(result, failed_elements=0)
        with self.assertRaisesRegex(ValueError, "recorded failed_elements"):
            audit.recorded_comparison(result, forged)

    def test_unsafe_path_and_different_comparison_shape_rejected(self):
        self.record["file"] = "../toy.npz"
        with self.assertRaisesRegex(ValueError, "safe artifact name"):
            audit.artifact(self.folder, self.record)
        with self.assertRaisesRegex(ValueError, "comparison shape"):
            audit.comparison(np.zeros((2, 2), np.float32), np.zeros((4,), np.float32))


if __name__ == "__main__":
    unittest.main()
