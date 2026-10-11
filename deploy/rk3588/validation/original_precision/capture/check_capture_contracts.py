#!/usr/bin/env python3
"""Small generated-data/hook-contract checks; never imports or runs a real model."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


board = import_file("_capture_tiny_board", HERE / "capture-v49-pair-prefix-board.py")
c = import_file("_capture_tiny_contracts", HERE / "v49_capture_contracts.py")
# Only synthetic tensors or previously saved data are read by these checks.
import numpy as np


class FakeHandle:
    def __init__(self, module, hook_id):
        self.module, self.hook_id = module, hook_id

    def remove(self):
        for name in ("_forward_pre_hooks", "_forward_hooks", "_forward_pre_hooks_with_kwargs",
                     "_forward_hooks_with_kwargs", "_forward_hooks_always_called"):
            getattr(self.module, name).pop(self.hook_id, None)


class FakeModule:
    """An observable Python dispatch fixture, with no parameters or neural inference."""
    next_hook = 0

    def __init__(self, operation=None):
        self.operation = operation or (lambda value, **kwargs: value)
        self.body_calls = 0
        for name in ("_forward_pre_hooks", "_forward_hooks", "_forward_pre_hooks_with_kwargs",
                     "_forward_hooks_with_kwargs", "_forward_hooks_always_called"):
            setattr(self, name, OrderedDict())

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        FakeModule.next_hook += 1
        index = FakeModule.next_hook
        self._forward_pre_hooks[index] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[index] = True
        return FakeHandle(self, index)

    def register_forward_hook(self, hook):
        FakeModule.next_hook += 1
        index = FakeModule.next_hook
        self._forward_hooks[index] = hook
        return FakeHandle(self, index)

    def forward(self, value, **kwargs):
        self.body_calls += 1
        return self.operation(value, **kwargs)

    def __call__(self, *args, **kwargs):
        for index, hook in tuple(self._forward_pre_hooks.items()):
            result = (hook(self, args, kwargs) if index in self._forward_pre_hooks_with_kwargs
                      else hook(self, args))
            if result is not None:
                raise AssertionError("Capture hook must never replace dispatch inputs")
        output = self.forward(*args, **kwargs)
        for hook in tuple(self._forward_hooks.values()):
            result = hook(self, args, output)
            if result is not None:
                raise AssertionError("Capture hook must never replace an output")
        return output


class FakeDispatch(FakeModule):
    def __init__(self):
        super().__init__()
        self.aggregator = SimpleNamespace(
            patch_embed=SimpleNamespace(blocks=[FakeModule() for _ in range(24)]),
            frame_blocks=[FakeModule(lambda x, **kw: x + np.float32(1)) for _ in range(24)],
            global_blocks=[FakeModule(lambda x, **kw: x * np.float32(2)) for _ in range(24)])
        self.camera_head, self.depth_head = FakeModule(), FakeModule()

    def named_modules(self):
        yield "", self
        for group, modules in (("dino", self.aggregator.patch_embed.blocks),
                               ("frame", self.aggregator.frame_blocks), ("global", self.aggregator.global_blocks)):
            for index, module in enumerate(modules):
                yield f"{group}.{index}", module
        yield "camera_head", self.camera_head
        yield "depth_head", self.depth_head

    def prefix(self, value, pos, global_mutation=None):
        for module in self.aggregator.patch_embed.blocks:
            value = module(value)
        value = self.aggregator.frame_blocks[0](value, pos=pos)
        value = value.reshape(1, 874, 1024)
        global_pos = pos.reshape(1, 874, 2)
        if global_mutation == "input":
            value = value + np.float32(1)
        elif global_mutation == "positions":
            global_pos = global_pos + np.int64(1)
        value = self.aggregator.global_blocks[0](value, pos=global_pos)
        # These calls must never be reached, even in the fake fixture.
        self.aggregator.frame_blocks[1](value, pos=global_pos)
        self.camera_head(value)
        self.depth_head(value)


def copier(value, kind):
    shapes = {"input.npy": c.INPUT_SHAPE, "positions.npy": c.POSITIONS_SHAPE,
              "frame-reference.npy": c.INPUT_SHAPE, "global-input": c.OUTPUT_SHAPE,
              "global-positions": [1, 874, 2], "output.npy": c.OUTPUT_SHAPE}
    dtype = "int64" if "positions" in kind else "float32"
    if value is None or list(value.shape) != shapes[kind] or str(value.dtype) != dtype or not np.isfinite(value).all():
        raise ValueError("Generated tensor contract mismatch")
    return np.array(value, copy=True, order="C")


class CaptureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / "image-plan.json").read_text())
        cls.value = np.zeros(c.INPUT_SHAPE, dtype=np.float32)
        cls.pos = np.zeros(c.POSITIONS_SHAPE, dtype=np.int64)

    def assert_clean(self, model, controller):
        self.assertEqual(controller.handles, [])
        self.assertTrue(controller.hook_registry_restored)
        self.assertTrue(controller.forward_functions_unchanged)
        for _, module in model.named_modules():
            self.assertTrue(all(not v for v in board.hook_identity(module).values()))

    def test_source_and_plan_freeze(self):
        self.assertEqual(c.sha(HERE / "capture-v49-pair-prefix-board.py"), c.CAPTURE_SOURCE_SHA)
        self.assertEqual(c.sha(HERE / "image-plan.json"), c.PLAN_SHA)
        self.assertEqual(c.sha(HERE / "original-vggt-source-pins.json"), c.MODEL_SOURCE_PINS_SHA)
        self.assertEqual(len(c.validate_plan(self.plan)), 10)

    def test_calibration_stops_before_frame_body_and_copies_owned_input(self):
        model = FakeDispatch()
        value = self.value.copy()
        control = board.PrefixCapture(model, "calibration", copier)
        artifacts = control.capture(lambda: model.prefix(value, self.pos))
        self.assertEqual(set(artifacts), {"input.npy", "positions.npy"})
        self.assertTrue(np.array_equal(artifacts["input.npy"], value))
        value.fill(8)
        self.assertTrue((artifacts["input.npy"] == 0).all())
        self.assertEqual([m.body_calls for m in model.aggregator.patch_embed.blocks], [1] * 24)
        self.assertEqual([m.body_calls for m in model.aggregator.frame_blocks], [0] * 24)
        self.assertEqual([m.body_calls for m in model.aggregator.global_blocks], [0] * 24)
        c.require_call_counts(dict(split="calibration", call_counts=control.counts,
                                   controlled_stop="before_original_frame0", hooks_removed=True,
                                   hook_registry_restored=True, forward_functions_unchanged=True))
        self.assert_clean(model, control)

    def test_evaluation_exact_frame_global_bridge_and_downstream_stop(self):
        model = FakeDispatch()
        control = board.PrefixCapture(model, "evaluation", copier)
        artifacts = control.capture(lambda: model.prefix(self.value, self.pos))
        self.assertEqual(set(artifacts), {"input.npy", "positions.npy", "frame-reference.npy", "output.npy"})
        self.assertTrue((artifacts["frame-reference.npy"] == 1).all())
        self.assertTrue((artifacts["output.npy"] == 2).all())
        self.assertTrue(control.frame_to_global_view_bitwise_equal)
        c.require_call_counts(dict(split="evaluation", call_counts=control.counts,
                                   controlled_stop="after_original_global0", hooks_removed=True,
                                   hook_registry_restored=True, forward_functions_unchanged=True))
        self.assertEqual([m.body_calls for m in model.aggregator.frame_blocks], [1] + [0] * 23)
        self.assertEqual([m.body_calls for m in model.aggregator.global_blocks], [1] + [0] * 23)
        self.assertEqual(model.camera_head.body_calls + model.depth_head.body_calls, 0)
        self.assert_clean(model, control)

    def test_global_input_and_positions_must_match_frame_view(self):
        for mutation in ("input", "positions"):
            model = FakeDispatch()
            control = board.PrefixCapture(model, "evaluation", copier)
            with self.assertRaisesRegex(ValueError, "Original frame"):
                control.capture(lambda: model.prefix(self.value, self.pos, mutation))
            self.assert_clean(model, control)
            self.assertEqual(model.aggregator.global_blocks[0].body_calls, 0)

    def test_unexpected_error_is_not_swallowed_and_hooks_are_removed(self):
        error = RuntimeError("fixture failure")
        model = FakeDispatch()
        control = board.PrefixCapture(model, "calibration", copier)
        def fail():
            raise error
        with self.assertRaises(RuntimeError) as seen:
            control.capture(fail)
        self.assertIs(seen.exception, error)
        self.assert_clean(model, control)

    def test_foreign_sentinel_is_not_swallowed(self):
        model = FakeDispatch()
        control = board.PrefixCapture(model, "calibration", copier)
        foreign = board._PrefixStop()
        def fail():
            raise foreign
        with self.assertRaises(board._PrefixStop) as seen:
            control.capture(fail)
        self.assertIs(seen.exception, foreign)
        self.assert_clean(model, control)

    def test_normal_return_without_stop_is_rejected(self):
        model = FakeDispatch()
        control = board.PrefixCapture(model, "calibration", copier)
        with self.assertRaisesRegex(ValueError, "controlled sentinel"):
            control.capture(lambda: None)
        self.assert_clean(model, control)

    def test_later_block_and_head_tripwires_precede_body(self):
        for name in ("frame", "global", "camera_head", "depth_head"):
            model = FakeDispatch()
            control = board.PrefixCapture(model, "calibration", copier)
            target = (model.aggregator.frame_blocks[1] if name == "frame" else
                      model.aggregator.global_blocks[0] if name == "global" else getattr(model, name))
            with self.assertRaisesRegex(ValueError, "Unexpected block|unexpectedly reached"):
                control.capture(lambda: target(self.value, pos=self.pos))
            self.assertEqual(target.body_calls, 0)
            self.assert_clean(model, control)

    def test_constructor_failure_removes_partially_installed_hooks(self):
        model = FakeDispatch()
        model.aggregator.global_blocks.pop()
        with self.assertRaisesRegex(ValueError, "24\\+24\\+24"):
            board.PrefixCapture(model, "calibration", copier)
        self.assertTrue(all(all(not rows for rows in board.hook_identity(module).values())
                            for _, module in model.named_modules()))

    def test_existing_hooks_and_malformed_positions_are_rejected(self):
        model = FakeDispatch()
        handle = model.camera_head.register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, "existing forward hooks"):
            board.PrefixCapture(model, "calibration", copier)
        self.assertEqual(len(model.camera_head._forward_hooks), 1)
        handle.remove()
        control = board.PrefixCapture(model, "calibration", copier)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            control.capture(lambda: model.prefix(self.value, self.pos.astype(np.int32)))
        self.assert_clean(model, control)

    def test_reversed_roles_split_reuse_and_legacy_images_are_rejected(self):
        reversed_plan = copy.deepcopy(self.plan)
        reversed_plan["samples"][0]["images"].reverse()
        with self.assertRaisesRegex(ValueError, "role|order"):
            c.validate_plan(reversed_plan)
        reused_plan = copy.deepcopy(self.plan)
        reused_plan["samples"][-1]["images"][0] = copy.deepcopy(reused_plan["samples"][0]["images"][0])
        with self.assertRaisesRegex(ValueError, "Repeated|overlap"):
            c.validate_plan(reused_plan)
        legacy_plan = copy.deepcopy(self.plan)
        legacy_plan["samples"][0]["images"][0]["index"] = 0
        with self.assertRaisesRegex(ValueError, "excluded"):
            c.validate_plan(legacy_plan)

    def test_raw_timestamp_precision_and_source_index_matching(self):
        by_index = {image["index"]: image for sample in self.plan["samples"] for image in sample["images"]}
        rows = [{"index": i, "name": str(i), "timestamp_string": str(i), "relative_path": str(i)}
                for i in range(798)]
        for index, image in by_index.items():
            stamp = Path(image["name"]).stem
            rows[index] = dict(index=index, name=image["name"], timestamp_string=stamp,
                               relative_path="rgb/" + image["name"])
        self.assertTrue(c.verify_plan_rows(self.plan, rows))
        self.assertEqual(rows[640]["timestamp_string"], "1305031123.511360")
        rows[32]["name"] = rows[40]["name"]
        with self.assertRaisesRegex(ValueError, "same original image"):
            c.verify_plan_rows(self.plan, rows)

    def test_saved_array_shape_raw_hash_finite_and_owned_copy(self):
        with tempfile.TemporaryDirectory(prefix="v49-capture-generated-") as folder:
            root = Path(folder)
            value = np.arange(12, dtype=np.float32).reshape(3, 4)
            path = root / "generated.npy"
            def metadata(array):
                np.save(path, array, allow_pickle=False)
                return dict(file=path.name, bytes=path.stat().st_size, sha256=c.sha(path),
                            shape=list(array.shape), dtype=str(array.dtype), tensor_bytes=array.nbytes,
                            raw_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest())
            item = metadata(value)
            saved = c.load_array(root, item, [3, 4], "float32")
            value.fill(0)
            self.assertEqual(saved[2, 3], 11)
            invalid = dict(item, raw_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "raw SHA"):
                c.load_array(root, invalid, [3, 4], "float32")
            with self.assertRaisesRegex(ValueError, "declared shape"):
                c.load_array(root, item, [12], "float32")
            value[0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finiteness"):
                c.load_array(root, metadata(value), [3, 4], "float32")
            with self.assertRaisesRegex(ValueError, "dtype"):
                c.load_array(root, metadata(np.ones((3, 4), dtype=np.float64)), [3, 4], "float32")

    def test_artifact_paths_cannot_escape_or_contain_whitespace(self):
        with tempfile.TemporaryDirectory(prefix="v49-capture-paths-") as folder:
            root = Path(folder)
            (root / "valid.npy").write_bytes(b"fixture")
            self.assertEqual(c.safe_file(root, "valid.npy"), root / "valid.npy")
            for relative in ("../escape.npy", "/tmp/escape.npy", "white space.npy", "missing.npy"):
                with self.assertRaises(ValueError):
                    c.safe_file(root, relative)

    def test_host_execution_guard_precedes_actual_inference_imports(self):
        if platform.machine().lower() in ("aarch64", "arm64"):
            self.skipTest("Host guard check is specific to this host architecture")
        with self.assertRaisesRegex(RuntimeError, "host execution is forbidden"):
            board.require_board()


class ReceiptResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        self.records.append({"test": test.id().split(".")[-1], "passed": True})
        super().addSuccess(test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CaptureChecks)
    result = unittest.TextTestRunner(verbosity=2, resultclass=ReceiptResult).run(suite)
    report = {"schema": "v49_generated_prefix_capture_selfchecks", "passed": result.wasSuccessful(),
              "tests_run": result.testsRun, "checks": result.records,
              "actual_model_or_board_inference": False, "actual_calibration_or_compilation": False,
              "real_image_decoding": False, "generated_hook_dispatch_only": True,
              "numpy": np.__version__, "capture_script_sha256": c.sha(HERE / "capture-v49-pair-prefix-board.py"),
              "contracts_sha256": c.sha(HERE / "v49_capture_contracts.py"),
              "check_script_sha256": c.sha(__file__), "image_plan_sha256": c.sha(HERE / "image-plan.json")}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "tests_run": report["tests_run"]}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
