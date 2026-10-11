"""Tiny generated states only; no real feature/image/BA execution on host."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if not (HERE / "run-v29-dpvo-geometry-board.py").is_file():
    HERE = HERE.parent / "deploy/rk3588/validation/dpvo_geometry"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load("v29_benchmark", HERE / "run-v29-dpvo-geometry-board.py")
assert driver.sha(HERE / "dpvo_cpu_geometry.py") == driver.SCALAR_SHA
assert driver.sha(HERE / "dpvo_cpu_geometry_vectorized.py") == driver.VECTOR_SHA
scalar = load("v29_original", HERE / "dpvo_cpu_geometry.py")
vector = load("v29_candidate", HERE / "dpvo_cpu_geometry_vectorized.py")


def tiny_input():
    p = np.zeros((1, 3, 7), np.float32)
    p[..., 6] = 1
    p[0, 1, 0] = -.1
    p[0, 2, 0] = -.15
    patches = np.zeros((1, 4, 3, 3, 3), np.float32)
    patches[0, :, 0] = np.array([40, 60, 90, 110], np.float32)[:, None, None]
    patches[0, :, 1] = np.array([30, 40, 50, 60], np.float32)[:, None, None]
    patches[0, :, 2] = .6
    K = np.broadcast_to(np.array([90, 92, 75, 58], np.float32), (1, 3, 4)).copy()
    kk = np.repeat(np.arange(4, dtype=np.int64), 3)
    ii = np.zeros(12, np.int64)
    jj = np.tile(np.arange(3, dtype=np.int64), 4)
    xy = scalar.reproject(p, patches, K, ii, jj, kk)[..., 1, 1]
    return dict(poses=p, patches=patches, intrinsics=K, ii=ii, jj=jj, kk=kk,
                target=xy + np.array([.3, -.2], np.float32), weight=np.ones_like(xy),
                **{"lambda": np.array([1e-4], np.float32)})


def reference(inputs):
    work = driver.copy_inputs(inputs)
    diag = scalar.BA(*driver.arguments(work), iterations=2, eff_impl=False)
    return driver.post_geometry(scalar, work), diag


class BenchmarkTests(unittest.TestCase):
    def test_interleaving_identical_input_and_system_capture(self):
        inputs = tiny_input()
        before = {k: driver.meta(v) for k, v in inputs.items()}
        expected, diag = reference(inputs)
        events = []
        report, saved = driver.run_case(scalar, vector, inputs, expected, diag,
                                        lambda *args: events.append(args))
        self.assertTrue(report["strict_FP32_gate_passed"])
        self.assertEqual([x["backend"] for x in report["calls"]],
                         ["scalar", "vectorized", "vectorized", "scalar"] * 2)
        self.assertEqual(report["ordinary_BA_calls"], 8)
        self.assertEqual(report["instrumented_validation_BA_calls"], 2)
        self.assertEqual(sum(x[0] == "ordinary" and x[3] == "completed" for x in events), 8)
        self.assertEqual(sum(x[0] == "instrumented" and x[3] == "completed" for x in events), 2)
        self.assertEqual(before, {k: driver.meta(v) for k, v in inputs.items()})
        for values in saved.values():
            self.assertIn("system0_B", values)
            self.assertIn("system1_E", values)
        for name in ("scalar", "vectorized"):
            self.assertEqual(len(report["summary"][name]["warm_seconds"]), 3)
            self.assertTrue(all(x["input_sha256"] == report["calls"][0]["input_sha256"] for x in report["calls"]))

    def test_bad_saved_reference_is_not_accepted(self):
        inputs = tiny_input()
        expected, diag = reference(inputs)
        expected["poses_after"] = expected["poses_after"].copy()
        expected["poses_after"][0, 1, 0] += 1
        report, saved = driver.run_case(scalar, vector, inputs, expected, diag)
        self.assertFalse(report["strict_FP32_gate_passed"])
        self.assertEqual(set(saved), {"scalar", "vectorized"})

    def test_trace_restore_even_if_solve_fails(self):
        original = scalar.fastba_system
        inputs = tiny_input()
        before = {k: driver.meta(v) for k, v in inputs.items()}
        with mock.patch.object(np.linalg, "cholesky", side_effect=np.linalg.LinAlgError("injected")):
            with self.assertRaises(np.linalg.LinAlgError):
                driver.collect_trace(scalar, inputs)
        self.assertIs(scalar.fastba_system, original)
        self.assertEqual(before, {k: driver.meta(v) for k, v in inputs.items()})

    def test_artifact_hash_metadata_and_safe_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            values = np.array([1, 2, 3], np.float32)
            path = folder / "toy.npz"
            np.savez(path, x=values)
            artifact = dict(file=path.name, sha256=driver.sha(path), bytes=path.stat().st_size,
                            arrays={"x": driver.meta(values)})
            def pin(path, expected):
                driver.require(driver.sha(path) == expected, "File hash differs")
                return path
            loaded = driver.load_artifact(folder, artifact, pin)
            np.testing.assert_array_equal(loaded["x"], values)
            self.assertFalse(loaded["x"].flags.writeable)
            artifact["arrays"]["x"]["tensor_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "raw metadata"):
                driver.load_artifact(folder, artifact, pin)
            artifact["file"] = "../toy.npz"
            with self.assertRaisesRegex(ValueError, "basename"):
                driver.load_artifact(folder, artifact, pin)

    def test_fixed_gate_finite_shape_and_boolean_checks(self):
        self.assertFalse(driver.comparison(np.array([.01], np.float32), np.array([0], np.float32))["passed"])
        self.assertFalse(driver.comparison(np.array([True]), np.array([False]))["passed"])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            driver.comparison(np.array([np.nan], np.float32), np.array([0], np.float32))
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            driver.comparison(np.array([0], np.float32), np.array([0], np.float64))

    def test_real_cli_rejects_host_before_opening_data_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "must-not-exist"
            result = subprocess.run([sys.executable, "-B", str(HERE / "run-v29-dpvo-geometry-board.py"),
                                     "--feedback-dir", "/nonexistent", "--scalar-geometry", "/nonexistent",
                                     "--output-dir", str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RK3588 board only", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
