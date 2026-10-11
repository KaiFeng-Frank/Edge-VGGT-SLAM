#!/usr/bin/env python3
"""Generate the predeclared DPVO encoder calibration set on RK3588 only.

This processes images but never runs an encoder, Update, GT evaluation or RKNN.
No image content is used for selection. DPVO preprocessing/RANDOM semantics:
princeton-vl/DPVO 0ac95b656d1fda91c271d2a106460d19ad966fc7 (MIT),
Copyright (c) 2022 Princeton Vision & Learning Lab. Project helper: MIT.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

ENCODER_SHA = "fc6fc662613a720cfb779a360a63906a593a96ff08addecb9d01cfd8767ca7a8"
RGB_SHA = "532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3"
SOURCE_INDICES = tuple(256 + 16 * i for i in range(32))
EVALUATION_INDICES = tuple(range(192))
BUDGET = 256 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def board_guard(machine=None, system=None, compatible=None):
    require((platform.machine() if machine is None else machine) == "aarch64", "Actual images are board-only")
    require((platform.system() if system is None else system) == "Linux", "Linux board required")
    if compatible is None:
        path = Path("/proc/device-tree/compatible")
        compatible = path.read_bytes() if path.is_file() else b""
    require(any(t in (b"rockchip,rk3588", b"rockchip,rk3588s") for t in compatible.split(b"\0")),
            "RK3588/S device-tree identity required")


def declared_policy():
    return dict(
        schema="v36_DPVO_encoder_calibration_policy", status="predeclared_before_generation",
        dataset="official TUM rgbd_dataset_freiburg1_xyz", source_rgb_rows=798,
        rgb_index_sha256=RGB_SHA, source_index_convention="zero-based non-comment rgb.txt order",
        calibration_source_indices=list(SOURCE_INDICES), evaluation_source_indices=list(EVALUATION_INDICES),
        selection="256+16*i, i=0..31; no GT, pixel, confidence or model-output selection",
        frame_sets_disjoint=True, independent_scene_claim=False,
        raw_image_hw=[480, 640], rectification="cv2.undistort with original K and D, then [8:-8,16:-16]",
        original_K=[[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]],
        original_D=[.2624, -.9531, -.0054, .0026, 1.1633],
        cropped_image_hw=[464, 608], cropped_K=[[517.3, 0, 302.6], [0, 516.5, 247.3], [0, 0, 1]],
        color="BGR, no RGB reorder", normalization="FP32 NCHW 2*(image/255)-0.5 once",
        encoder_helper_sha256=ENCODER_SHA, feature_grid_hw=[116, 152], patches_per_frame=96,
        indices="INT64[96], y*152+x, x=1..150/y=1..114, order and repeats retained",
        random_generator="persistent Torch CPU Generator", seed=0,
        random_consumption_per_selected_frame=["randint x [1,151), 96", "randint y [1,115), 96",
                                                "rand inverse-depth FP32 [0,1), 96"],
        random_scope="Only selected calibration frames; no draws for omitted source frames; not online graph RNG equivalence",
        dataset_input_order=["normalized_bgr", "patch_indices"], output_file_budget_bytes=BUDGET,
        GT_reading_allowed=False, model_forward_allowed=False, calibration_build_executed=False,
        accuracy_or_realtime_validated=False)


def parse_rgb_index(text):
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        require(len(fields) == 2, "Exactly timestamp and relative RGB filename required")
        stamp, name = fields
        timestamp = float(stamp)
        path = Path(name)
        require(math.isfinite(timestamp) and (not rows or timestamp > rows[-1]["timestamp"]),
                "Finite strictly increasing RGB timestamps required")
        require(not path.is_absolute() and ".." not in path.parts and len(path.parts) == 2 and
                path.parts[0] == "rgb" and path.suffix == ".png" and path.stem == stamp,
                "Only original relative rgb/<timestamp>.png paths allowed")
        rows.append(dict(source_index=len(rows), timestamp_text=stamp, timestamp=timestamp,
                         relative_path=path.as_posix()))
    return rows


def select_rows(rows, policy):
    require(policy == declared_policy(), "Predeclared policy changed")
    require(len(rows) == policy["source_rgb_rows"], "Exact original RGB source population required")
    selected = [rows[i] for i in policy["calibration_source_indices"]]
    require([r["source_index"] for r in selected] == list(SOURCE_INDICES), "Source index/order changed")
    require(not set(SOURCE_INDICES).intersection(policy["evaluation_source_indices"]), "Evaluation/calibration overlap")
    return selected


def preprocess_bgr(bgr, cv2_module):
    """Used on real data only after board guard; synthetic tests may call it."""
    import numpy as np
    require(isinstance(bgr, np.ndarray) and bgr.dtype == np.uint8 and bgr.shape == (480, 640, 3),
            "Original decoded BGR uint8[480,640,3] required")
    policy = declared_policy()
    K, D = np.array(policy["original_K"], np.float64), np.array(policy["original_D"], np.float64)
    undistorted = cv2_module.undistort(bgr, K, D)
    require(undistorted.dtype == np.uint8 and undistorted.shape == bgr.shape, "Undistortion changed image contract")
    result = np.ascontiguousarray(undistorted[8:-8, 16:-16])
    require(result.shape == (464, 608, 3), "Original crop shape changed")
    return result


def array_info(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), bytes=int(array.nbytes),
                tensor_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


def sample_indices(encoder, generator, torch_module):
    import numpy as np
    before = generator.get_state().numpy().copy()
    centers = encoder.random_centers(152, 116, 96, generator)
    after_centers = generator.get_state().numpy().copy()
    require(isinstance(centers, np.ndarray) and centers.dtype == np.int64 and centers.shape == (96, 2),
            "Frozen original random centers must be INT64[96,2]")
    x, y = centers[:, 0], centers[:, 1]
    require(np.all((x >= 1) & (x < 151) & (y >= 1) & (y < 115)), "Invalid original interior patch centers")
    # Consume the original per-frame depth draw even though calibration uses no depth.
    depth = torch_module.rand((96,), dtype=torch_module.float32, generator=generator, device="cpu").numpy().copy()
    after_depth = generator.get_state().numpy().copy()
    require(depth.dtype == np.float32 and depth.shape == (96,) and
            np.isfinite(depth).all() and np.all((depth >= 0) & (depth < 1)), "Original FP32 random depth contract")
    indices = np.ascontiguousarray(y * np.int64(152) + x)
    require(indices.dtype == np.int64, "Integer Gather indices must not be cast")
    return indices, dict(centers=centers, consumed_inverse_depth=depth, state_before=before,
                         state_after_centers=after_centers, state_after_depth=after_depth)


def file_info(path):
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def checked_source(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "RGB source missing or escapes dataset root")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "rgb-index", "encoder-helper", "policy", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args(argv)
    board_guard()  # Before loading/decoding images or importing Torch/OpenCV.
    require(sha(args.policy) == args.policy_sha256, "Policy source SHA mismatch")
    policy = json.loads(args.policy.read_text())
    require(policy == declared_policy(), "Fixed predeclared policy required")
    require(sha(args.rgb_index) == RGB_SHA, "Original official RGB index SHA mismatch")
    require(sha(args.encoder_helper) == ENCODER_SHA, "Frozen encoder preprocessing helper SHA mismatch")
    output = args.output_dir.resolve()
    protected = [v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"]
    require(not any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
            "Output/source overlap")
    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    pins = {str(p.resolve()): sha(p) for p in (Path(__file__), args.policy, args.rgb_index, args.encoder_helper)}
    report = dict(schema="v36_DPVO_board_calibration_data", status="failed", script_sha256=sha(__file__),
                  policy=policy, policy_sha256=args.policy_sha256, source_pins=pins, samples=[],
                  GT_read=False, model_forward_executed=False, RKNN_build_executed=False,
                  online_graph_equivalence_claimed=False, accuracy_or_realtime_validated=False,
                  cleanup_errors=[])
    try:
        rows = select_rows(parse_rgb_index(args.rgb_index.read_text()), policy)
        spec = importlib.util.spec_from_file_location("_v36_frozen_frame_encoder", args.encoder_helper)
        encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(encoder)  # No session or model is created.
        import cv2
        import numpy as np
        import torch
        torch.set_num_threads(2)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(policy["seed"])
        report["versions"] = dict(Python=platform.python_version(), OpenCV=cv2.__version__,
                                  numpy=np.__version__, torch=torch.__version__)
        (output / "policy.json").write_bytes(args.policy.read_bytes())
        sampling = {k: [] for k in ("centers", "consumed_inverse_depth", "state_before",
                                    "state_after_centers", "state_after_depth")}
        dataset_lines = []
        for index, source in enumerate(rows):
            path = checked_source(args.dataset_dir, source["relative_path"])
            pins[str(path)] = sha(path)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            require(bgr is not None, "RGB decoding failed; no skipping/substitution allowed")
            cropped = preprocess_bgr(bgr, cv2)
            image = encoder.normalize_bgr(cropped)
            require(image.dtype == np.float32 and image.shape == (1, 3, 464, 608) and
                    np.isfinite(image).all() and image.min() >= -.5 and image.max() <= 1.5,
                    "Frozen normalization contract changed")
            indices, draws = sample_indices(encoder, generator, torch)
            image_file, index_file = output / ("image-%02d.npy" % index), output / ("indices-%02d.npy" % index)
            np.save(image_file, image, allow_pickle=False)
            np.save(index_file, indices, allow_pickle=False)
            for key in sampling:
                sampling[key].append(draws[key])
            require(sha(path) == pins[str(path)], "RGB source changed during read")
            report["samples"].append(dict(
                calibration_index=index, **source, source_image_sha256=pins[str(path)], source_image_bytes=path.stat().st_size,
                decoded_BGR=array_info(bgr), cropped_BGR=array_info(cropped), image_array=array_info(image),
                index_array=array_info(indices), input_files=[file_info(image_file), file_info(index_file)],
                RNG=dict(sequence=policy["random_consumption_per_selected_frame"],
                         arrays={k: array_info(v) for k, v in draws.items()})))
            dataset_lines.append(image_file.name + " " + index_file.name)
            require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= BUDGET, "File budget exceeded")
            print(json.dumps(dict(sample=index, source_index=source["source_index"], timestamp=source["timestamp_text"])), flush=True)
        sampling_arrays = {k: np.stack(v) for k, v in sampling.items()}
        sampling_path = output / "sampling.npz"
        np.savez(sampling_path, **sampling_arrays)
        report["sampling_artifact"] = dict(**file_info(sampling_path), arrays={k: array_info(v) for k, v in sampling_arrays.items()})
        require([r["source_index"] for r in report["samples"]] == list(SOURCE_INDICES), "Incomplete or reordered population")
        dataset = output / "dataset.txt"
        dataset.write_text("\n".join(dataset_lines) + "\n")
        report["dataset"] = file_info(dataset)
        report["status"] = "generated_fixed_calibration_data_not_quantized_or_validated"
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        report["source_hashes_after"] = {p: sha(p) if Path(p).is_file() else None for p in pins}
        report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
        if not report["sources_and_inputs_unchanged"]:
            report["status"] = "failed"
        report["generation_seconds"] = time.monotonic() - begin
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report["output_bytes_before_receipt"] = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        receipt = output / "calibration.json"
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if report["output_bytes_before_receipt"] + len(text.encode()) > BUDGET:
            report["status"], report["error"] = "failed", "Output file budget exceeded including receipt"
            text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        receipt.write_text(text)
    print(json.dumps({k: report.get(k) for k in ("status", "error")}), flush=True)
    return 0 if report["status"].startswith("generated_fixed_calibration_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
