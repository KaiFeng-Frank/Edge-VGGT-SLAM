#!/usr/bin/env python3
"""Independent saved NumPy/JSON audit; no SDK, image decode or model execution.

DPVO MIT component research. Missing dense imap and profile/repeated arrays are
explicitly reported as recorded metadata, never independently re-read arrays.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import traceback

import numpy as np

RTOL, ATOL = 1e-4, 1e-5
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
SOURCE_PINS = {
    "run-v34-dpvo-sparse-context-board.py": "1c4281dc0bef9b4eaa7978719011c2342dd43c9c4005144a8b145ff603cd231f",
    "v34_sparse_rknn.py": "646f875d1b5eea561c0d6aa4d1b9078351014a39a62976ec895a814cd466b387",
    "compile-v34-dpvo-sparse-context.py": "26e639520692619ca7d69916d11d81237c535ee31f3ea9aa848101e4742744fa",
    "run-v26-dpvo-rknn-board.py": "895ac2902a66c68048c03e98f353300c7065e7568047479db4fee2f9a80a33d3",
    "run-v12-rknn-block-board.py": "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608",
}
LINEAGE = {
    "original": {
        "recipe_sha256": "368ec845f14655bf71e2fc9d80d435197c5ab6b8dd807c7c9af92ef9e33d13e0",
        "parent_onnx_sha256": "527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82",
        "sparse_onnx_sha256": "c52cecd383e60ba564dcb5100be6c7e975fa6ecc051ed89e2c09bff73bc071dc",
        "dense_model_sha256": "3840d7d779211ca19dd296237dfb8e11bf602235fccf0071eb7dd96ab1ef7517",
        "sparse_model_sha256": "2e94aeb7f0e2969465717344ab699839e73c56b28a728d48f1498ed3c32605c7",
    },
    "spatial": {
        "recipe_sha256": "005667f44cf98447d2b622882d7d40cbcce8a545985ddf2104eaa62740273ce2",
        "parent_onnx_sha256": "c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826",
        "sparse_onnx_sha256": "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab",
        "dense_model_sha256": "b4bd0b8ce7e09a3ab9d8dffaa7ccacd58c33842f9be5d0885b113d2fdc6a9869",
        "sparse_model_sha256": "b4cba66bca9fbdacd54054149127d3db847913683b72f3ece5c9075990efd799",
    },
}
SHAPES = {"fmap": (1, 128, 116, 152), "imap": (1, 384, 116, 152), "context": (1, 96, 384)}
NORMAL_SCHEDULE = [(0, "dense", "first"), (0, "sparse", "first"), (0, "sparse", "warm"),
                   (0, "dense", "warm"), (0, "dense", "warm"), (0, "sparse", "warm"),
                   (1, "sparse", "other_frame"), (1, "dense", "other_frame"),
                   (2, "dense", "other_frame"), (2, "sparse", "other_frame")]
PROFILE_SCHEDULE = [(i, "sparse", "profile") for i in range(3)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            digest.update(data)
    return digest.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


class Checks:
    def __init__(self):
        self.count = 0

    def check(self, condition, message):
        self.count += 1
        if not condition:
            raise ValueError(message)

    def equal(self, actual, expected, message):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), message + " fields")
            for key, value in expected.items():
                self.equal(actual[key], value, message + "." + key)
        elif isinstance(expected, float):
            self.check(type(actual) in (int, float) and math.isfinite(actual) and
                       math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-14), message)
        else:
            self.check(actual == expected and (type(actual) is bool if type(expected) is bool else True), message)


def score(actual, expected):
    if actual.dtype != np.float32 or expected.dtype != np.float32 or actual.shape != expected.shape or not (
            np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError("Finite matching FP32 arrays required")
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    difference = a - b
    outside = np.abs(difference) > ATOL + RTOL * np.abs(b)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(rtol=RTOL, atol=ATOL, failed_elements=int(outside.sum()), element_count=int(outside.size),
                gate_passed=not bool(outside.any()), max_abs=float(np.max(np.abs(difference))),
                rms=float(np.sqrt(np.mean(difference * difference))),
                relative_l2=float(np.linalg.norm(difference) / nb) if nb else None,
                cosine=float(np.dot(a / na, b / nb)) if na and nb else None,
                array_equal=bool(np.array_equal(actual, expected)),
                bitwise_equal=actual.tobytes(order="C") == expected.tobytes(order="C"),
                finite=True, actual_all_zero=bool(np.all(actual == 0)))


def pinned_json(path, digest, checks):
    checks.check(sha(path) == digest, "Receipt SHA " + Path(path).name)
    return json.loads(Path(path).read_text())


def read_npz(directory, item, declared, checks):
    checks.check(isinstance(item["file"], str) and Path(item["file"]).name == item["file"], "Simple local NPZ name")
    path = Path(directory) / item["file"]
    checks.check(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "NPZ file bytes/SHA " + path.name)
    with np.load(path, allow_pickle=False) as saved:
        checks.check(set(saved.files) == set(declared), "Exact saved NPZ keys " + path.name)
        arrays = {key: saved[key] for key in saved.files}
    for key, value in arrays.items():
        checks.equal(metadata(value), declared[key], "Independently rehashed array " + path.name + "." + key)
        checks.check(np.isfinite(value).all(), "Saved finite array " + key)
    return arrays


def select_payload(arrays, indices, row, checks):
    checks.check(set(arrays) in ({"fmap", "context"}, {"fmap", "imap"}), "Canonical or complete dense payload required")
    full = "imap" in arrays
    checks.check(full == row["artifact"]["includes_full_dense_imap"], "Full imap availability declared accurately")
    if full:
        y, x = np.divmod(indices, arrays["imap"].shape[-1])
        selected = {"fmap": arrays["fmap"], "context": np.ascontiguousarray(arrays["imap"][0][:, y, x].T[None])}
        checks.equal(metadata(arrays["imap"]), row["arrays"]["imap"], "Saved complete dense imap matches actual metadata")
    else:
        selected = arrays
    for key, value in selected.items():
        checks.equal(metadata(value), row["selected_arrays"][key], "Saved selected raw metadata " + key)
        checks.check(value.dtype == np.float32 and value.shape == SHAPES[key] and np.any(value != 0), "Nonzero FP32 selected shape " + key)
    checks.equal(metadata(selected["fmap"]), row["arrays"]["fmap"], "Full actual fmap and selected fmap identical")
    if row["role"] == "sparse":
        checks.equal(row["arrays"], row["selected_arrays"], "Sparse outputs are selected outputs")
    return selected, full


def check_attr(row, index, name, shape, dtype, formats, checks):
    count = math.prod(shape)
    checks.check(row["index"] == index and row["name"] == name and row["shape"] == list(shape) and
                 row["type"] == dtype and row["format"] in formats, "Actual SDK attribute " + name)
    size = count * (8 if dtype == 8 else 2)
    checks.check(row["n_elems"] == count and row["size"] == size and row["size_with_stride"] >= size,
                 "Actual native logical bytes " + name)
    checks.check(row["quantization_type"] in (0, 2) and row["zero_point"] == 0 and row["scale"] == 1,
                 "No integer quantization of neural values or topology " + name)


def check_timing(row, checks):
    timing, sparse = row["timing"], row["role"] == "sparse"
    stages = ("input_packing_seconds", "input_set_seconds", "run_api_seconds", "outputs_get_seconds",
              "output_copy_seconds", "outputs_release_seconds")
    checks.check(all(type(timing[k]) in (int, float) and math.isfinite(timing[k]) and timing[k] >= 0 for k in stages),
                 "Finite nonnegative complete SDK stages")
    checks.check(sum(timing[k] for k in stages) <= timing["sdk_io_seconds"] + 1e-8,
                 "SDK endpoint includes packing/get/owned-copy/release")
    checks.check(timing["output_buffers_released"] is True and timing["synchronous"] is True and
                 timing["output_bytes"] == [9027584, 147456 if sparse else 27082752], "Owned logical FP32 outputs/release")
    checks.equal(row["forward_seconds"], timing["sdk_io_seconds"], "Complete API time")
    gather = row["dense_context_gather_seconds"]
    checks.check(type(gather) in (int, float) and math.isfinite(gather) and gather >= 0 and (not sparse or gather == 0),
                 "Dense gather separate; sparse has no host gather")
    checks.equal(row["feature_payload_ready_seconds"], row["forward_seconds"] + gather, "Feature-ready includes gather")
    if sparse:
        checks.check(timing["input_api_types"] == [0, 8] and timing["input_api_formats"] == [1, 3] and
                     timing["input_pass_through"] == [0, 0] and timing["input_logical_dtypes"] == ["float32", "int64"] and
                     timing["input_buffer_shapes"] == [[1, 464, 608, 3], [96]] and
                     timing["input_buffer_bytes"] == [3385344, 768] and timing["output_want_float"] == 1 and
                     timing["output_is_prealloc"] == 0, "INT64 topology separate from FP32 NHWC image; no silent cast")
    else:
        checks.check(timing["input_api_format"] == 1 and timing["input_pass_through"] == 0 and
                     timing["input_api_dtype"] == "float32" and timing["input_logical_shape"] == [1, 3, 464, 608] and
                     timing["input_buffer_shape"] == [1, 464, 608, 3] and timing["output_api_want_float"] == 1,
                     "Same dense image logical/native input contract")


def parse_profile(detail, checks):
    text = detail["raw_text"]
    checks.check(detail["query_command"] == 3 and len(text.encode()) == detail["bytes"] and
                 hashlib.sha256(text.encode()).hexdigest() == detail["raw_sha256"], "Raw SDK profile bytes/SHA")
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\d+\s", line):
            continue
        fields = line.split()
        checks.check(len(fields) >= 11 and fields[3] in ("NPU", "CPU", "GPU") and
                     re.fullmatch(r"\d+/\d+/\d+", fields[6]), "Recognized actual profile operator row")
        workloads = [f for f in fields[8:] if re.fullmatch(r"\d+(?:\.\d+)?%/\d+(?:\.\d+)?%/\d+(?:\.\d+)?%", f)]
        checks.check(len(workloads) == 1, "Actual three-core workload column")
        rows.append(dict(index=int(fields[0]), type=fields[1], target=fields[3], elapsed_us=int(fields[7]),
                         core_workload_percent=[float(x[:-1]) for x in workloads[0].split("/")], name=fields[-1]))
    checks.check(rows and [r["index"] for r in rows] == list(range(1, len(rows) + 1)), "Complete ordered SDK profile")
    total = re.search(r"Total Operator Elapsed Per Frame Time\(us\):\s*(\d+)", text)
    checks.check(total is not None and sum(r["elapsed_us"] for r in rows) == int(total[1]), "Per-operator total agrees")
    io = {"InputOperator", "OutputOperator"}
    internal = [r for r in rows if r["type"] not in io]
    return dict(raw_profile_sha256=detail["raw_sha256"], rows=len(rows), total_operator_us=int(total[1]),
                target_counts=dict(Counter(r["target"] for r in rows)),
                internal_target_counts=dict(Counter(r["target"] for r in internal)),
                CPU_IO_rows=sum(r["target"] == "CPU" and r["type"] in io for r in rows),
                internal_CPU=[r for r in internal if r["target"] == "CPU"],
                gather_operators=[r for r in rows if "gather" in r["type"].lower() or "gather" in r["name"].lower()])


def check_report(report, reference, profile, checks):
    method = report["method"]
    checks.check(method in LINEAGE and report["schema"] == "v34_DPVO_sparse_context_RKNN_comparison" and
                 report["status"] == "completed_sparse_context_SDK_comparison_not_VO", "Completed measured schema")
    checks.check(report["script_sha256"] == SOURCE_PINS["run-v34-dpvo-sparse-context-board.py"] and
                 report["sdk_helper_sha256"] == SOURCE_PINS["v34_sparse_rknn.py"] and
                 report["reference_sha256"] == REFERENCE_SHA and report["collect_perf_detail"] is profile,
                 "Measured source and frozen CPU-reference identities")
    for key in ("candidate_accepted", "ORB_used", "added_LK_PnP_frontend", "model_checkpoint_loaded", "images_decoded",
                "Update_executed", "correlation_executed", "BA_executed", "complete_VO_executed", "realtime_validated",
                "quality_validated", "all_operators_on_npu_verified", "FP32_thresholds_relaxed", "output_clipping"):
        checks.check(report[key] is False, "No unsupported claim " + key)
    checks.check(report["actual_board_execution"] is True and report["NPU_used"] is True and
                 report["original_VGGT_default_unchanged"] is True and report["different_model_from_VGGT"] is True and
                 report["normalization_applications"] == 0, "Actual research scope")
    checks.check(report["source_pins"] == report["source_hashes_after"] and report["sources_and_inputs_unchanged"] is True,
                 "Board-recorded unchanged sources; unavailable board paths not host-rehashed")
    for name, digest in SOURCE_PINS.items():
        checks.check(any(Path(k).name == name and v == digest for k, v in report["source_pins"].items()), "Exact recorded source pin " + name)
    checks.check(report["cleanup_errors"] == [] and report["affinity_restored"] is True and report["cpu_affinity"] == [4, 5, 6, 7],
                 "Recorded cleanup/CPU affinity")
    ancestor, expected = report["ancestor"], LINEAGE[method]
    for key in ("recipe_sha256", "parent_onnx_sha256", "sparse_onnx_sha256"):
        checks.equal(ancestor[key], expected[key], "Graph lineage " + key)
    checks.check(report["rewrite_sha256"] == expected["recipe_sha256"] and ancestor["method"] == method and
                 ancestor["reference_receipt_sha256"] == REFERENCE_SHA and ancestor["suffix_CPU_bitwise_gate_passed"] is True and
                 ancestor["parent_CPU_strict_gate_passed"] is (method == "original") and
                 ancestor["spatial_parent_rejection_retained"] is True, "CPU ancestor rejection boundary")
    checks.equal(ancestor["preservation"], reference["graph_preservation"][method], "Recorded frozen graph preservation")
    contexts = report["contexts"]
    checks.check(set(contexts) == ({"sparse"} if profile else {"dense", "sparse"}), "Normal/profile contexts distinct")
    for role, context in contexts.items():
        checks.check(context["model_sha256"] == expected[role + "_model_sha256"] and context["core_mask"] == 7 and
                     context["sdk_version"].startswith("2.3.2 ") and context["context_destroyed"] is True, "Actual model/context lifecycle " + role)
        checks.check(len(context["input_attrs"]) == (2 if role == "sparse" else 1) and len(context["output_attrs"]) == 2,
                     "Exact queried tensor counts")
        check_attr(context["input_attrs"][0], 0, "normalized_bgr", (1, 464, 608, 3), 1, (1,), checks)
        check_attr(context["output_attrs"][0], 0, "fmap", SHAPES["fmap"], 1, (0,), checks)
        if role == "sparse":
            check_attr(context["input_attrs"][1], 1, "patch_indices", (96,), 8, (3,), checks)
            check_attr(context["output_attrs"][1], 1, "context", SHAPES["context"], 1, (3,), checks)
        else:
            check_attr(context["output_attrs"][1], 1, "imap", SHAPES["imap"], 1, (0,), checks)
    schedule = PROFILE_SCHEDULE if profile else NORMAL_SCHEDULE
    checks.check([(r["frame_index"], r["role"], r["timing_role"]) for r in report["calls"]] == schedule and
                 [r["index"] for r in report["calls"]] == list(range(len(schedule))) and report["sdk_run_api_calls"] == len(schedule),
                 "Exact prescribed schedule/call count, no selected successful samples")
    rows = [r for r in reference["calls"] if r["method"] == method]
    checks.check([r["frame_index"] for r in rows] == [0, 1, 2], "Three fixed reference frames")
    checks.equal(report["preprocessing"], reference["preprocessing"], "Once-normalized input provenance")
    checks.equal(report["source_frames"], [r["source_image"] for r in rows], "Source frame identities/order")
    checks.equal(report["inputs"], [r["input_metadata"] for r in rows], "Raw normalized input metadata")
    return rows


def audit(args):
    checks = Checks()
    output = dict(schema="v34_DPVO_sparse_context_independent_saved_review", status="failed", audit_script_sha256=sha(__file__),
                  saved_arrays_only=True, model_ORT_SDK_or_image_execution=False, raw_artifacts_modified=False,
                  candidate_accepted=False, production_changed=False, reports=[], numerical_calls=[])
    try:
        reference = pinned_json(args.reference_dir / "ort.json", REFERENCE_SHA, checks)
        checks.check(reference["suffix_bitwise_gate_passed"] is True and reference["suffix_strict_gate_passed"] is True and
                     reference["original_CPU_strict_gate_each_method"] == {"original": True, "spatial": False}, "Frozen CPU suffix/reference boundary")
        normal = pinned_json(args.normal_dir / "run.json", args.normal_sha256, checks)
        rows = check_report(normal, reference, False, checks)
        output["method"] = normal["method"]
        # Only six canonical results are retained. Repeat calls without an NPZ
        # are checked against recorded hashes, not counted as new array reads.
        selected_cache, score_cache, indices_cache = {}, {}, {}
        file_shas, first_actual, independent_indices = {}, {}, []
        full_dense_calls, canonical_calls, unsaved_calls = [], [], []
        for row in normal["calls"]:
            frame, role = row["frame_index"], row["role"]
            ref = rows[frame]
            checks.check(row["frame_id"] == ref["frame_id"] and row["source_image"] == ref["source_image"] and
                         row["input_array"] == ref["input_metadata"] and row["indices"] == ref["indices"], "Exact current source/pixel/index identity")
            check_timing(row, checks)
            checks.check(row["finite_nonzero"] is True, "Completed call recorded finite and nonzero")
            key = frame, role
            checks.equal(row["repeat_raw_hash_equal"], None if key not in first_actual else row["arrays"] == first_actual[key], "Repeat metadata assertion")
            first_actual.setdefault(key, row["arrays"])
            if "artifact" in row:
                if frame not in indices_cache:
                    inputs = read_npz(args.reference_dir, ref["input_artifact"],
                                      {"normalized_bgr": ref["input_metadata"], "patch_indices": ref["indices"], "centers_xy": ref["centers"]}, checks)
                    indices = inputs["patch_indices"]
                    checks.check(indices.dtype == np.int64 and indices.shape == (96,) and
                                 inputs["normalized_bgr"].dtype == np.float32 and inputs["normalized_bgr"].shape == (1, 3, 464, 608) and
                                 np.array_equal(indices, inputs["centers_xy"][:, 1] * 152 + inputs["centers_xy"][:, 0]), "Exact row-major integer sampling/input shape")
                    indices_cache[frame] = indices
                arrays = read_npz(args.normal_dir, row["artifact"], row["artifact"]["arrays"], checks)
                selected, full = select_payload(arrays, indices_cache[frame], row, checks)
                target = read_npz(args.reference_dir, ref["artifact"], ref["arrays"], checks)
                calculated = {k: score(selected[k], target[k]) for k in ("fmap", "context")}
                checks.equal(row["versus_v33_CPU_ORT"], calculated, "Independent canonical vs frozen CPU ORT errors")
                selected_cache[key], score_cache[key] = selected, calculated
                file_shas[row["artifact"]["file"]] = row["artifact"]["sha256"]
                independent_indices.append(row["index"])
                (full_dense_calls if full else canonical_calls).append(row["index"])
                output["numerical_calls"].append(dict(index=row["index"], frame_index=frame, role=role,
                    arrays_independently_rehashed=True, complete_dense_imap_available=full,
                    dense_gather_independently_recomputed=full, comparisons=calculated))
                del target, arrays
            else:
                unsaved_calls.append(row["index"])
                checks.equal(row["selected_arrays"], {k: metadata(v) for k, v in selected_cache[key].items()}, "Unsaved repeat recorded raw hashes match available canonical")
                checks.equal(row["versus_v33_CPU_ORT"], score_cache[key], "Unsaved repeat reported errors corroborated by matching recorded raw hashes")
            if "sparse_vs_dense_selected" in row:
                comparison = {k: score(selected_cache[(frame, "sparse")][k], selected_cache[(frame, "dense")][k]) for k in ("fmap", "context")}
                checks.equal(row["sparse_vs_dense_selected"], comparison, "Independent corresponding canonical dense/sparse comparison")
            checks.check("perf_detail" not in row, "Normal timing excluded SDK profiling")
        checks.check(set(selected_cache) == {(i, role) for i in range(3) for role in ("dense", "sparse")} and len(file_shas) == 6,
                     "All six representative results audited, not all ten calls independently saved")
        computed_gate = all(v["gate_passed"] for r in normal["calls"] for v in r["versus_v33_CPU_ORT"].values())
        checks.equal(normal["actual_v33_CPU_ORT_strict_gate_passed"], computed_gate, "FP32 gate summary independent of evidence-audit success")
        timing_summary = {}
        for role in ("dense", "sparse"):
            warm = [r for r in normal["calls"] if r["role"] == role and r["timing_role"] == "warm"]
            seconds = [r["forward_seconds"] for r in warm]
            checks.check(len(warm) == 2, "Only two prescribed warm samples")
            checks.equal(normal["timing_summary"][role], dict(warm_seconds=seconds, warm_mean_seconds=float(np.mean(seconds))), "Ordinary warm timing summary")
            timing_summary[role] = dict(warm_count=2, sdk_io_mean_seconds=float(np.mean(seconds)),
                feature_payload_ready_mean_seconds=float(np.mean([r["feature_payload_ready_seconds"] for r in warm])),
                stage_mean_seconds={k: float(np.mean([r["timing"][k] for r in warm])) for k in
                    ("input_packing_seconds", "input_set_seconds", "run_api_seconds", "outputs_get_seconds", "output_copy_seconds", "outputs_release_seconds")})
        output["reports"].append(dict(role="normal", receipt_sha256=args.normal_sha256, calls=10,
            independently_read_array_call_indices=independent_indices, unsaved_repeat_call_indices=unsaved_calls,
            complete_dense_imap_call_indices=full_dense_calls, canonical_only_call_indices=canonical_calls,
            file_sha256=file_shas, recorded_cleanup_errors=normal["cleanup_errors"],
            all_recorded_v33_CPU_ORT_strict_gate_passed=computed_gate, timing=timing_summary,
            recorded_peak_RSS_MiB=normal["memory_final"]["VmHWM_MiB"], actual_driver_seconds=normal["actual_driver_seconds"]))
        if args.profile_dir is not None:
            profile = pinned_json(args.profile_dir / "run.json", args.profile_sha256, checks)
            check_report(profile, reference, True, checks)
            checks.check(profile["method"] == normal["method"] and profile["ancestor"] == normal["ancestor"] and
                         profile["dense_compile_sha256"] == normal["dense_compile_sha256"] and
                         profile["sparse_compile_sha256"] == normal["sparse_compile_sha256"], "Separate profile uses same source/compiled models")
            profile_calls = []
            for row in profile["calls"]:
                frame = row["frame_index"]
                checks.check(row["frame_id"] == rows[frame]["frame_id"] and row["source_image"] == rows[frame]["source_image"] and
                             row["input_array"] == rows[frame]["input_metadata"] and row["indices"] == rows[frame]["indices"], "Profile same fixed source/pixel/index identity")
                check_timing(row, checks)
                checks.check(row["finite_nonzero"] is True, "Profile recorded finite/nonzero")
                checks.equal(row["arrays"], row["selected_arrays"], "Profile sparse logical payload")
                checks.equal(row["selected_arrays"], {k: metadata(v) for k, v in selected_cache[(frame, "sparse")].items()}, "Profile recorded hashes match independently rehashed normal canonical")
                independent = not args.profile_metadata_only
                if independent:
                    arrays = read_npz(args.profile_dir, row["artifact"], row["artifact"]["arrays"], checks)
                    selected, full = select_payload(arrays, indices_cache[frame], row, checks)
                    checks.check(not full, "Sparse profile has no dense imap")
                    checks.check(all(selected[k].tobytes(order="C") == selected_cache[(frame, "sparse")][k].tobytes(order="C") for k in selected), "Saved profile/normal bytes identical")
                checks.equal(row["versus_v33_CPU_ORT"], score_cache[(frame, "sparse")], "Profile comparison corroborated by declared matching canonical hashes")
                profile_calls.append(dict(index=row["index"], frame_index=frame,
                    arrays_independently_available=independent,
                    crossreceipt_raw_metadata_matches_host_rehashed_normal=True,
                    comparison_scope="independent saved bytes" if independent else "recorded hashes and scores only; profile NPZ not read",
                    profile=parse_profile(row["perf_detail"], checks), sdk_io_seconds=row["forward_seconds"]))
            checks.equal(profile["actual_v33_CPU_ORT_strict_gate_passed"], all(v["gate_passed"] for r in profile["calls"] for v in r["versus_v33_CPU_ORT"].values()), "Profile recorded FP32 summary")
            checks.check(all(v["warm_mean_seconds"] is None and v["warm_seconds"] == [] for v in profile["timing_summary"].values()), "Profile not normal benchmark")
            output["reports"].append(dict(role="profile", receipt_sha256=args.profile_sha256, calls=3,
                array_scope="metadata_only" if args.profile_metadata_only else "independent_saved_arrays", call_reviews=profile_calls,
                recorded_cleanup_errors=profile["cleanup_errors"]))
        output.update(status="passed", numerical_acceptance=False,
            dense_gather_scope="Only saved canonical context verified; complete dense imap absent, gather not independently recomputed" if not full_dense_calls else
                "Complete dense imap gathered only for listed available calls; other canonical contexts do not prove their unsaved gather",
            source_scope="Board before/after source, library and binary pins checked as recorded; unavailable original board paths not independently host-rehashed",
            repeated_array_scope="Four unsaved ordinary repeats use recorded matching hashes; six canonical artifacts independently read",
            inherited_spatial_CPU_rejection_retained=True,
            limits=["Two warm samples per ordinary role; no thermal/frequency control or sustained throughput claim",
                    "No Update, correlation, BA, VO trajectory, original VGGT acceleration or 20 Hz acceptance",
                    "Sparse output bytes do not reduce full CNN compute; finite outputs do not establish numerical acceptance",
                    "Profile operator placement/times are actual recorded evidence, not complete SDK endpoint timing"])
    except Exception as exc:
        output.update(error=repr(exc), traceback=traceback.format_exc())
    output.update(checks=checks.count, failures=0 if output["status"] == "passed" else 1)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("normal-dir", "reference-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--normal-sha256", required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--profile-metadata-only", action="store_true",
                        help="Explicitly do not read profile NPZ; check its recorded hashes against ordinary canonical arrays")
    args = parser.parse_args(argv)
    parser.error("--profile-dir and --profile-sha256 required together") if bool(args.profile_dir) != bool(args.profile_sha256) else None
    parser.error("--profile-metadata-only requires --profile-dir") if args.profile_metadata_only and args.profile_dir is None else None
    result = audit(args)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "method", "checks", "failures", "error")}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
