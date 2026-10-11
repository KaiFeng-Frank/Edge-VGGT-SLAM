"""Tiny synthetic validation of the NumPy-only saved-array auditor."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent
HELPER = ROOT / "audit-v29-dpvo-geometry.py"
if not HELPER.is_file():
    HELPER = ROOT.parent / "deploy/rk3588/validation/dpvo_geometry/audit-v29-dpvo-geometry.py"
spec = importlib.util.spec_from_file_location("v29_saved_geometry_audit", HELPER)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def artifact(path, values):
    return dict(file=path.name, sha256=a.sha(path), bytes=path.stat().st_size,
                arrays={key: a.meta(value) for key, value in values.items()})


def calls():
    values = []
    for repetition in range(4):
        order = ("scalar", "vectorized") if repetition % 2 == 0 else ("vectorized", "scalar")
        for index, backend in enumerate(order):
            values.append(dict(backend=backend, repetition=repetition, order_in_pair=index,
                timing_role="first_per_case" if repetition == 0 else "warm_same_input",
                BA_iterations=2, BA_seconds=2. if backend == "scalar" else 1.,
                external_copy_seconds=0., input_hash_seconds=0., post_validation_seconds=0.,
                input_sha256={"x": "in"}, output_sha256={"x": backend}))
    return values


class TestSavedAudit(unittest.TestCase):
    def test_file_and_tensor_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "toy.npz"
            values = {"x": np.array([1., 2.], np.float32), "mask": np.array([True, False])}
            np.savez(path, **values)
            record = artifact(path, values)
            checks = a.Checks()
            loaded = a.load_artifact(root, record, checks, "toy")
            self.assertFalse(checks.failures)
            self.assertFalse(loaded["x"].flags.writeable)
            mutated = copy.deepcopy(record)
            mutated["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "file SHA256"):
                a.load_artifact(root, mutated, a.Checks(), "bad file")
            mutated = copy.deepcopy(record)
            mutated["arrays"]["x"]["tensor_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "raw metadata SHA256"):
                a.load_artifact(root, mutated, a.Checks(), "bad tensor")
            mutated = copy.deepcopy(record)
            mutated["file"] = "../toy.npz"
            with self.assertRaisesRegex(ValueError, "safe basename"):
                a.load_artifact(root, mutated, a.Checks(), "bad path")

    def test_shape_dtype_finite_and_gate(self):
        x = np.array([1.], np.float32)
        for y in (np.array([[1.]], np.float32), np.array([1.], np.float64)):
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                a.compare(x, y, a.Checks(), "bad")
        with self.assertRaisesRegex(ValueError, "finite"):
            a.compare(x, np.array([np.nan], np.float32), a.Checks(), "bad")
        checks = a.Checks()
        a.compare(x, np.array([1.001], np.float32), checks, "outside")
        self.assertEqual(len(checks.failures), 1)
        checks = a.Checks()
        a.compare(np.array([False]), np.array([True]), checks, "mask")
        self.assertEqual(len(checks.failures), 1)

    def test_means_and_repeated_output_hashes(self):
        data = calls()
        recorded = {backend: dict(first_seconds=seconds, warm_seconds=[seconds] * 3, warm_mean_seconds=seconds)
                    for backend, seconds in (("scalar", 2.), ("vectorized", 1.))}
        output = {backend: {"x": backend} for backend in recorded}
        checks = a.Checks()
        result = a.summarize_calls(data, recorded, {"x": "in"}, output, checks, "toy")
        self.assertFalse(checks.failures)
        self.assertEqual(result["scalar_over_vectorized_warm_mean"], 2.)
        mutated = copy.deepcopy(recorded)
        mutated["scalar"]["warm_mean_seconds"] = 1.9
        checks = a.Checks()
        a.summarize_calls(data, mutated, {"x": "in"}, output, checks, "bad mean")
        self.assertTrue(any("warm mean" in x for x in checks.failures))
        mutated_calls = copy.deepcopy(data)
        mutated_calls[-1]["output_sha256"]["x"] = "stale"
        checks = a.Checks()
        a.summarize_calls(mutated_calls, recorded, {"x": "in"}, output, checks, "bad hash")
        self.assertTrue(any("output raw SHA" in x for x in checks.failures))

    def test_saved_algebra_rejects_mutation_without_solving(self):
        f = np.float32
        system = dict(B=np.diag(np.array([3., 2.], f)), E=np.zeros((2, 1), f),
            C=np.zeros(1, f), v=np.array([1., 2.], f), u=np.zeros(1, f),
            Q=np.array([10000.], f), S=np.diag(np.array([4.0003, 3.0002], f)),
            y=np.array([1., 2.], f), dx=np.zeros((1, 2), f), dz=np.zeros(1, f))
        checks = a.Checks()
        a.saved_system_algebra(system, checks, "toy")
        self.assertFalse(checks.failures)
        system["Q"] = np.array([5000.], f)
        checks = a.Checks()
        a.saved_system_algebra(system, checks, "bad")
        self.assertTrue(any("/Q" in x for x in checks.failures))

    def test_events_and_counter_tampering(self):
        run = dict(cases=[dict(calls=calls()) for _ in range(4)], BA_events=[],
            ordinary_BA_calls_attempted=32, ordinary_BA_calls_completed=32,
            instrumented_BA_calls_attempted=8, instrumented_BA_calls_completed=8,
            BA_iterations_in_completed_calls=80)
        clock = 0.
        for case, row in enumerate(run["cases"]):
            identities = [("ordinary", x["backend"], x["repetition"]) for x in row["calls"]]
            identities += [("instrumented", backend, None) for backend in ("scalar", "vectorized")]
            for kind, backend, repetition in identities:
                for stage in ("begin", "completed"):
                    clock += 3.
                    run["BA_events"].append(dict(case=case, kind=kind, backend=backend,
                        repetition=repetition, stage=stage, monotonic=clock))
        checks = a.Checks()
        a.verify_events(run, checks)
        self.assertFalse(checks.failures)
        mutated = copy.deepcopy(run)
        mutated["ordinary_BA_calls_completed"] = 31
        checks = a.Checks()
        a.verify_events(mutated, checks)
        self.assertTrue(any("ordinary completed" in x for x in checks.failures))
        mutated = copy.deepcopy(run)
        mutated["BA_events"][1]["monotonic"] = mutated["BA_events"][0]["monotonic"]
        with self.assertRaisesRegex(ValueError, "monotonic timestamp"):
            a.verify_events(mutated, a.Checks())


if __name__ == "__main__":
    unittest.main()
