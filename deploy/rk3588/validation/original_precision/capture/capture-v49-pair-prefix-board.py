#!/usr/bin/env python3
"""Board-only unchanged VGGT prefix captures for fixed calibration/evaluation.

Calibration stops in a frame0 pre-hook after all original DINO computation.
Evaluation stops after original frame0 -> global0. No hook replaces a tensor.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path("/mnt/vggt-data/vggt-slam2")
SOURCE = BASE / "src"
DATASET = BASE / "datasets/rgbd_dataset_freiburg1_xyz"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_board():
    if sys.platform != "linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("Actual prefix inference requires Linux aarch64 RK3588S; host execution is forbidden")
    for path in (Path("/proc/device-tree/compatible"), Path("/sys/firmware/devicetree/base/compatible")):
        if path.is_file():
            identity = path.read_bytes().replace(b"\0", b",").decode(errors="replace")
            if "rk3588" not in identity:
                raise RuntimeError("Actual prefix capture requires RK3588 device-tree identity")
            return identity
    raise RuntimeError("RK3588 device-tree identity unavailable")


def function_sha(path, name):
    text = Path(path).read_text()
    matches = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("Unique original preprocessing function required")
    return hashlib.sha256(ast.get_source_segment(text, matches[0]).encode()).hexdigest()


def forward_identity(module):
    forward = module.forward
    return (id(getattr(forward, "__func__", forward)), id(getattr(forward, "__self__", None)))


def hook_identity(module):
    return {name: tuple(getattr(module, name, {}).items()) for name in (
        "_forward_pre_hooks", "_forward_hooks", "_forward_pre_hooks_with_kwargs",
        "_forward_hooks_with_kwargs", "_forward_hooks_always_called")}


class _PrefixStop(BaseException):
    """Private exact-instance sentinel; unrelated failures are never swallowed."""


class PrefixCapture:
    """Only observes tensors; pure fake module/tensor tests may exercise this class."""
    def __init__(self, model, split, copy_tensor):
        if split not in ("calibration", "evaluation"):
            raise ValueError("Invalid prefix capture split")
        self.model, self.split, self.copy_tensor = model, split, copy_tensor
        self.handles, self.values, self.cleanup_errors = [], {}, []
        self.stop = _PrefixStop()
        self.modules = list(model.named_modules())
        self.original_forward = {id(module): forward_identity(module) for _, module in self.modules}
        self.original_hooks = {id(module): hook_identity(module) for _, module in self.modules}
        if any(any(registry for registry in rows.values()) for rows in self.original_hooks.values()):
            raise ValueError("Original prefix requires a model with no existing forward hooks")
        self.counts = {f"{group}_{phase}": [0] * 24 for group in ("dino", "frame", "global")
                       for phase in ("entered", "completed")}
        self.counts.update(camera_head_entered=0, depth_head_entered=0)
        self.agg = model.aggregator
        try:
            for group, modules in (("dino", self.agg.patch_embed.blocks),
                                   ("frame", self.agg.frame_blocks), ("global", self.agg.global_blocks)):
                if len(modules) != 24:
                    raise ValueError("Original 24+24+24 block topology required")
                for index, module in enumerate(modules):
                    self.handles.append(module.register_forward_pre_hook(self.count_before(group, index), with_kwargs=True))
                    self.handles.append(module.register_forward_hook(self.count_after(group, index)))
            for name in ("camera_head", "depth_head"):
                module = getattr(model, name)
                if module is None:
                    raise ValueError("Original Camera/depth heads must remain part of the loaded model")
                self.handles.append(module.register_forward_pre_hook(self.forbid_head(name), with_kwargs=True))
            self.handles.append(self.agg.frame_blocks[0].register_forward_pre_hook(self.before_frame, with_kwargs=True))
            if split == "evaluation":
                self.handles.append(self.agg.frame_blocks[0].register_forward_hook(self.after_frame))
                self.handles.append(self.agg.global_blocks[0].register_forward_pre_hook(self.before_global, with_kwargs=True))
                self.handles.append(self.agg.global_blocks[0].register_forward_hook(self.after_global))
        except BaseException:
            self.close()
            raise

    def count_before(self, group, index):
        def hook(module, args, kwargs):
            self.counts[f"{group}_entered"][index] += 1
            if self.counts[f"{group}_entered"][index] != 1:
                raise ValueError("A prefix block was entered more than once")
            if group != "dino" and (index != 0 or (group == "global" and self.split == "calibration")):
                raise ValueError("Unexpected block beyond requested original prefix")
        return hook

    def count_after(self, group, index):
        def hook(module, args, output):
            self.counts[f"{group}_completed"][index] += 1
        return hook

    def forbid_head(self, name):
        def hook(module, args, kwargs):
            self.counts[name + "_entered"] += 1
            raise ValueError("Prefix capture unexpectedly reached a downstream original head")
        return hook

    def before_frame(self, module, args, kwargs):
        if self.values or not args or kwargs.get("viz_attention", False):
            raise ValueError("Unexpected first original frame0 dispatch")
        self.values["input.npy"] = self.copy_tensor(args[0], "input.npy")
        pos = kwargs.get("pos", args[1] if len(args) > 1 else None)
        self.values["positions.npy"] = self.copy_tensor(pos, "positions.npy")
        if self.split == "calibration":
            raise self.stop

    def after_frame(self, module, args, output):
        self.values["frame-reference.npy"] = self.copy_tensor(output, "frame-reference.npy")

    def before_global(self, module, args, kwargs):
        import numpy as np
        value = self.copy_tensor(args[0], "global-input")
        position = self.copy_tensor(kwargs.get("pos", args[1] if len(args) > 1 else None), "global-positions")
        if not np.array_equal(value, self.values["frame-reference.npy"].reshape(1, 874, 1024)):
            raise ValueError("Original frame output is not the contiguous global input")
        if not np.array_equal(position, self.values["positions.npy"].reshape(1, 874, 2)):
            raise ValueError("Original frame/global RoPE position view changed")
        self.frame_to_global_view_bitwise_equal = True

    def after_global(self, module, args, output):
        self.values["output.npy"] = self.copy_tensor(output, "output.npy")
        raise self.stop

    def close(self):
        for handle in reversed(self.handles):
            try:
                handle.remove()
            except BaseException as exc:
                self.cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        self.handles.clear()
        self.hook_registry_restored = all(hook_identity(module) == self.original_hooks[id(module)]
                                          for _, module in self.modules)
        self.forward_functions_unchanged = all(forward_identity(module) == self.original_forward[id(module)]
                                               for _, module in self.modules)

    def capture(self, forward):
        reached = False
        try:
            try:
                forward()
            except _PrefixStop as exc:
                if exc is not self.stop:
                    raise
                reached = True
                self.stop.__traceback__ = None
            if not reached:
                raise ValueError("Original prefix did not exit through its controlled sentinel")
        finally:
            self.close()
        if self.cleanup_errors or not self.hook_registry_restored or not self.forward_functions_unchanged:
            raise ValueError("Prefix hooks or original forwards were not restored")
        return self.values


def parameter_state(model):
    rows = []
    for category, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in values:
            rows.append((category, name, id(tensor), tuple(tensor.shape), str(tensor.dtype), str(tensor.device),
                         tensor.data_ptr(), tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                         tuple(tensor.stride()), tensor._version))
    return rows


def platform_record():
    values = {"peak_process_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "temperatures_mC": {}}
    for path in Path("/sys/class/thermal").glob("thermal_zone*"):
        if (path / "type").is_file() and (path / "temp").is_file():
            values["temperatures_mC"][(path / "type").read_text().strip()] = int((path / "temp").read_text())
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--contracts-sha256", required=True)
    parser.add_argument("--source-pins", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    board = require_board()  # Must precede NumPy/OpenCV/Torch imports, loading or decoding.
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(BASE / "experiments") or args.output_dir.exists():
        parser.error("Output must be a new experiment directory on the board data volume")
    if sha(args.contracts) != args.contracts_sha256:
        parser.error("Frozen capture contracts SHA mismatch")
    spec = importlib.util.spec_from_file_location("_v49_capture_contracts", args.contracts)
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    c = contracts
    c.require(sha(__file__) == c.CAPTURE_SOURCE_SHA, "Frozen capture script SHA mismatch")
    c.pinned(args.plan, c.PLAN_SHA)
    c.require(args.plan_sha256 == c.PLAN_SHA, "Fixed original image plan required")
    c.pinned(args.source_pins, c.MODEL_SOURCE_PINS_SHA)
    c.pinned(args.pair_dir / "export.json", c.PAIR_EXPORT_SHA)
    c.pinned(args.pair_dir / "positions.npy", c.FROZEN_POSITIONS_SHA)
    pair = c.json_file(args.pair_dir / "export.json")
    c.require(pair["onnx"]["sha256"] == c.PAIR_ONNX_SHA and pair["input_shape"] == c.INPUT_SHAPE
              and pair["output_shape"] == c.OUTPUT_SHAPE and pair["original_parameter_tensors"] == 36,
              "Fixed complete original frame0/global0 pair required")
    plan = c.json_file(args.plan)
    samples = c.validate_plan(plan)
    rgb_index = DATASET / "rgb.txt"
    rows = c.parse_rgb_index(rgb_index.read_text())
    c.verify_plan_rows(plan, rows)
    closure = c.json_file(args.source_pins)
    source_hashes = {"third_party/vggt/" + name: digest for name, digest in closure.items()}
    source_hashes.update(c.RUNTIME_SOURCE_PINS)
    all_pins = {str(SOURCE / name): c.pinned(SOURCE / name, digest) for name, digest in source_hashes.items()}
    all_pins.update({str(args.plan): c.PLAN_SHA, str(args.source_pins): c.MODEL_SOURCE_PINS_SHA,
                    str(args.pair_dir / "export.json"): c.PAIR_EXPORT_SHA,
                    str(args.pair_dir / "positions.npy"): c.FROZEN_POSITIONS_SHA,
                    str(rgb_index): sha(rgb_index), str(args.contracts): args.contracts_sha256,
                    str(Path(__file__).resolve()): c.CAPTURE_SOURCE_SHA})
    calibration = SOURCE / "deploy/rk3588/tum-fr1-local-calibration.json"
    all_pins[str(calibration)] = c.pinned(calibration, c.CALIBRATION_FILE_SHA)
    c.require(function_sha(SOURCE / "vggt_slam/edge_mapping.py", "prepare_rectified_images")
              == c.PREPROCESS_FUNCTION_SHA, "Original v17 lossless preprocessing function changed")
    for sample in samples:
        for image in sample["images"]:
            path = Path(image["path"])
            c.require(path.resolve().is_relative_to(DATASET.resolve()) and path.stat().st_size == image["bytes"],
                      "Original source PNG path/size changed")
            all_pins[str(path)] = c.pinned(path, image["sha256"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "capture.json"
    report = {"schema": c.SCHEMA, "status": "started", "script_sha256": c.CAPTURE_SOURCE_SHA,
              "contracts_sha256": args.contracts_sha256, "device_tree_compatible": board,
              "actual_inference_on_RK3588S_only": True, "actual_ORT_or_compiler_run": False,
              "full_model_forward_completed": False, "original_model_only": True,
              "real_time_validated": False, "full_model_quality_validated": False,
              "source_hashes": source_hashes, "samples": [], "cleanup_errors": [],
              "provenance_artifacts": {},
              "frozen_pair": {"export_sha256": c.PAIR_EXPORT_SHA, "onnx_sha256": c.PAIR_ONNX_SHA,
                              "original_input_raw_sha256": c.FROZEN_INPUT_RAW_SHA,
                              "original_positions_raw_sha256": c.FROZEN_POSITIONS_RAW_SHA}}
    started, original_affinity, model = time.monotonic(), os.sched_getaffinity(0), None
    original_environment = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MPLBACKEND")}
    checkpoint = BASE / "weights/model.pt"
    checkpoint_stat = None
    try:
        cpus = {4, 5, 6, 7}
        c.require(cpus <= original_affinity, "All four big CPU cores must be permitted")
        os.sched_setaffinity(0, cpus)
        os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4", MPLBACKEND="Agg")
        sys.path[:0] = [str(SOURCE), str(SOURCE / "third_party/vggt")]
        import cv2
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        from vggt_slam.image_rectification import CalibrationRectifier
        from vggt_slam.edge_mapping import prepare_rectified_images
        c.require(torch.__version__ == "2.3.1" and np.__version__ == "1.26.4", "Frozen board Torch/NumPy versions required")
        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        for name in ("_global_forward_hooks", "_global_forward_pre_hooks"):
            c.require(not getattr(torch.nn.modules.module, name), "Global module hooks must be absent")
        report["runtime"] = {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                             "threads": 4, "interop_threads": 1, "cpus": sorted(os.sched_getaffinity(0))}
        checkpoint_stat = checkpoint.stat()
        c.require(checkpoint_stat.st_size == c.CHECKPOINT_BYTES, "Original checkpoint byte count mismatch")
        checkpoint_start = time.monotonic()
        c.pinned(checkpoint, c.CHECKPOINT_SHA)
        report["checkpoint"] = {"path": str(checkpoint), "bytes": checkpoint_stat.st_size, "sha256": c.CHECKPOINT_SHA,
                                "full_SHA_checked_before_loading": True,
                                "SHA_seconds": time.monotonic() - checkpoint_start,
                                "hash_warms_filesystem_cache": True}
        rectifier = CalibrationRectifier(c.json_file(calibration))
        report["preprocessing"] = {"requested_image_size": 336, "expected_RGB_tensor_shape": c.RGB_SHAPE,
                                   "function_sha256": c.PREPROCESS_FUNCTION_SHA,
                                   "rectification": rectifier.metadata(), "same_v17_lossless_bridge": True}
        for name, origin in {"image-plan.json": args.plan, "rgb-index.txt": rgb_index,
                             "original-vggt-source-pins.json": args.source_pins,
                             "rectification-calibration.json": calibration,
                             "frozen-pair-export.json": args.pair_dir / "export.json"}.items():
            target = args.output_dir / name
            target.write_bytes(Path(origin).read_bytes())
            report["provenance_artifacts"][name] = {"file": name, "bytes": target.stat().st_size, "sha256": sha(target)}
        positions = np.load(args.pair_dir / "positions.npy", allow_pickle=False)
        c.require(list(positions.shape) == c.POSITIONS_SHAPE and str(positions.dtype) == "int64"
                  and positions.flags.c_contiguous and hashlib.sha256(memoryview(positions).cast("B")).hexdigest()
                  == c.FROZEN_POSITIONS_RAW_SHA, "Frozen RoPE position identity mismatch")
        loading = time.monotonic()
        model = load_model(checkpoint, device="cpu", image_size=336, prune_unused_heads=True, offline=True)
        report["model_load_seconds"] = time.monotonic() - loading
        c.require(not model.training and all(not p.requires_grad and p.device.type == "cpu" and p.dtype == torch.float32
                                            for p in model.parameters()), "Original CPU FP32 eval-only model required")
        identity = {"full_model_parameter_count": sum(p.numel() for p in model.parameters()),
                    "named_parameter_tensors": len(list(model.named_parameters())),
                    "all_36_pair_parameter_values_match_frozen_graph": False, "pair_bindings": {}}
        c.require(identity["full_model_parameter_count"] == 1157941492 and identity["named_parameter_tensors"] == 1341,
                  "Full original active parameter structure mismatch")
        def pair_weight_identity():
            bindings = {}
            for name, module in (("frame0", model.aggregator.frame_blocks[0]),
                                 ("global0", model.aggregator.global_blocks[0])):
                tensors = module.state_dict()
                expected = pair["original_parameter_bindings"][name]
                c.require(set(tensors) == set(expected), "Original pair parameter keys changed")
                bindings[name] = {}
                for key, tensor in tensors.items():
                    value = np.ascontiguousarray(tensor.detach().numpy())
                    raw = hashlib.sha256(memoryview(value).cast("B")).hexdigest()
                    e = expected[key]["original"]
                    c.require(list(value.shape) == e["shape"] and str(value.dtype) == e["dtype"] and raw == e["sha256"],
                              "Original pair parameter differs from fixed graph")
                    bindings[name][key] = {"shape": list(value.shape), "dtype": str(value.dtype), "raw_sha256": raw}
            return bindings
        identity["pair_bindings"] = pair_weight_identity()
        identity["all_36_pair_parameter_values_match_frozen_graph"] = True
        report["parameter_identity"] = identity
        initial_state = parameter_state(model)
        def array_info(value):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "tensor_bytes": value.nbytes,
                    "raw_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest(),
                    "finite": bool(np.isfinite(value).all())}
        def copied_tensor(value, kind):
            pos = kind in ("positions.npy", "global-positions")
            shape = ([1, 874, 2] if kind == "global-positions" else c.POSITIONS_SHAPE) if pos else (
                c.OUTPUT_SHAPE if kind in ("global-input", "output.npy") else c.INPUT_SHAPE)
            c.require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                      and value.dtype == (torch.int64 if pos else torch.float32)
                      and list(value.shape) == shape, "Original prefix tensor dtype/device/shape mismatch")
            result = np.array(value.detach().numpy(), copy=True, order="C")
            c.require(np.isfinite(result).all(), "Nonfinite original prefix tensor")
            if pos:
                c.require(np.array_equal(result, positions.reshape(shape)), "Original RoPE positions differ from fixed graph")
            return result
        observed_inputs = {"calibration": set(), "evaluation": set()}
        for sample in samples:
            sample_dir = args.output_dir / "samples" / sample["sample_id"]
            sample_dir.mkdir(parents=True, exist_ok=False)
            sample_started = time.monotonic()
            prepared = prepare_rectified_images([item["path"] for item in sample["images"]], rectifier, 336)
            c.require(prepared.device.type == "cpu" and prepared.dtype == torch.float32 and list(prepared.shape) == c.RGB_SHAPE
                      and bool(torch.isfinite(prepared).all()) and prepared.min().item() >= 0
                      and prepared.max().item() <= 1, "Original RGB preprocessing dtype/range/shape mismatch")
            rgb_before = array_info(prepared.numpy())
            controller = PrefixCapture(model, sample["split"], copied_tensor)
            prefix_started = time.monotonic()
            with torch.inference_mode():
                values = controller.capture(lambda: model(prepared, query_points=None, compute_similarity=False))
            prefix_seconds = time.monotonic() - prefix_started
            c.require(parameter_state(model) == initial_state, "Original model tensor state changed during prefix")
            roles_equal = True
            for frame in range(2):
                role = frame
                roles_equal = roles_equal and np.array_equal(values["input.npy"][frame, 0],
                    model.aggregator.camera_token.detach().numpy()[0, role, 0])
                roles_equal = roles_equal and np.array_equal(values["input.npy"][frame, 1:5],
                    model.aggregator.register_token.detach().numpy()[0, role])
            c.require(roles_equal, "Original camera/register first/other token roles changed")
            sample_row = {"sample_id": sample["sample_id"], "split": sample["split"],
                          "images": [dict(image, ordinal_in_pair=index, model_role="first" if index == 0 else "other",
                                          timestamp_string=rows[image["index"]]["timestamp_string"])
                                     for index, image in enumerate(sample["images"])],
                          "prepared_RGB": rgb_before, "images_tensor_unchanged": array_info(prepared.numpy()) == rgb_before,
                          "parameter_state_unchanged": True, "original_camera_register_roles_equal": bool(roles_equal),
                          "call_counts": controller.counts, "hooks_removed": not controller.handles,
                          "hook_registry_restored": controller.hook_registry_restored,
                          "forward_functions_unchanged": controller.forward_functions_unchanged,
                          "controlled_stop": "after_original_global0" if sample["split"] == "evaluation" else "before_original_frame0",
                          "prefix_seconds_including_copy_hooks_and_validation": prefix_seconds,
                          "platform_after": platform_record(), "artifacts": {}}
            if sample["split"] == "evaluation":
                sample_row["frame_to_global_view_bitwise_equal"] = controller.frame_to_global_view_bitwise_equal
            c.require_call_counts(sample_row)
            for name, value in values.items():
                target = sample_dir / name
                np.save(target, value, allow_pickle=False)
                sample_row["artifacts"][name] = dict(array_info(value), file=str(target.relative_to(args.output_dir)),
                                                     bytes=target.stat().st_size, sha256=sha(target))
            digest = sample_row["artifacts"]["input.npy"]["raw_sha256"]
            c.require(digest != c.FROZEN_INPUT_RAW_SHA and digest not in observed_inputs[sample["split"]]
                      and digest not in observed_inputs["evaluation" if sample["split"] == "calibration" else "calibration"],
                      "Calibration/evaluation input tensor leakage or duplication")
            observed_inputs[sample["split"]].add(digest)
            sample_row["total_seconds_including_preprocessing_hooks_save"] = time.monotonic() - sample_started
            report["samples"].append(sample_row)
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"event": "prefix_captured", "sample_id": sample["sample_id"], "split": sample["split"],
                              "prefix_seconds": prefix_seconds, "total_seconds": sample_row["total_seconds_including_preprocessing_hooks_save"]}), flush=True)
            del prepared, values, controller
        c.require(pair_weight_identity() == identity["pair_bindings"] and parameter_state(model) == initial_state,
                  "Original pair values or full model tensor state changed")
        for path, digest in all_pins.items():
            c.pinned(path, digest)
        final_checkpoint = checkpoint.stat()
        c.require((final_checkpoint.st_size, final_checkpoint.st_mtime_ns, final_checkpoint.st_ino)
                  == (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, checkpoint_stat.st_ino),
                  "Original checkpoint stat changed")
        report["checkpoint"]["stat_unchanged_after_capture"] = True
        report["source_and_inputs_unchanged"] = True
        report["parameter_state_unchanged"] = True
        report["status"] = "completed_prefix_capture"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        try:
            os.sched_setaffinity(0, original_affinity)
            report["CPU_affinity_restored"] = os.sched_getaffinity(0) == original_affinity
        except BaseException as exc:
            report["cleanup_errors"].append(f"affinity: {exc}")
        try:
            for key, old in original_environment.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old
            report["environment_restored"] = all(os.environ.get(key) == old
                                                 for key, old in original_environment.items())
        except BaseException as exc:
            report["cleanup_errors"].append(f"environment: {exc}")
        report["total_seconds_including_weight_hash_load"] = time.monotonic() - started
        report["platform_after"] = platform_record()
        if (report["cleanup_errors"] or report.get("CPU_affinity_restored") is not True
                or report.get("environment_restored") is not True):
            report["status"] = "failed"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    try:
        c.load_capture(args.output_dir, sha(report_path))
    except BaseException as exc:
        report["status"] = "failed"
        report["portable_saved_data_validation"] = {"passed": False, "type": type(exc).__name__, "message": str(exc)}
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    print(json.dumps({"event": "capture_completed", "capture_sha256": sha(report_path),
                      "samples": len(report["samples"]), "seconds": report["total_seconds_including_weight_hash_load"]}), flush=True)


if __name__ == "__main__":
    main()
