#!/usr/bin/env python3
"""Saved-array audit of the original sparse DPVO encoder PTQ experiment.

MIT project helper. NumPy/JSON only: no SDK import, image decoding, model,
ONNX evaluator or calibration execution. A passed audit verifies the evidence,
not the quantized candidate's FP32 equivalence or complete VO quality.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import traceback

import numpy as np

RTOL, ATOL = 1e-4, 1e-5
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
ENTRY_SHA = "c0cf2cbba2755ced221f7c169467c9ed7e575b0259e7ee9b3aa37f9b67fb2d9c"
TRANSPORT_SHA = "646f875d1b5eea561c0d6aa4d1b9078351014a39a62976ec895a814cd466b387"
QUERY_SOURCE_SHA = "871f67c5a37ff23b703d173bb09234504a58fb16b8ace65e57991afe6f7e3d90"
SHAPES = {"fmap": (1, 128, 116, 152), "context": (1, 96, 384)}
STAGES = ("input_packing_seconds", "input_set_seconds", "run_api_seconds",
          "outputs_get_seconds", "output_copy_seconds", "outputs_release_seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def metadata(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), bytes=int(array.nbytes),
                tensor_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


class Checks:
    def __init__(self):
        self.count = 0

    def check(self, condition, message):
        self.count += 1
        if not condition:
            raise ValueError(message)

    def equal(self, actual, expected, message):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), message + " keys")
            for key in expected:
                self.equal(actual[key], expected[key], message + "." + key)
        elif isinstance(expected, float):
            self.check(type(actual) in (int, float) and math.isfinite(actual) and
                       math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-13), message)
        else:
            self.check(actual == expected and (type(actual) is bool if type(expected) is bool else True), message)


def score(actual, expected):
    if actual.dtype != np.float32 or expected.dtype != np.float32 or actual.shape != expected.shape or not (
            np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError("Matching finite FP32 arrays required")
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    difference = a - b
    failures = int(np.count_nonzero(np.abs(difference) > ATOL + RTOL*np.abs(b)))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    nd = float(np.linalg.norm(difference))
    return dict(rtol=RTOL, atol=ATOL, failed_elements=failures, element_count=int(a.size),
                gate_passed=failures == 0, bitwise_equal=actual.tobytes(order="C") == expected.tobytes(order="C"),
                array_equal=bool(np.array_equal(actual, expected)), max_abs=float(np.max(np.abs(difference))),
                rmse=float(np.sqrt(np.mean(difference*difference))),
                relative_l2=nd/nb if nb else 0.0 if nd == 0 else None,
                cosine=float(np.dot(a/na, b/nb)) if na and nb else None,
                finite=True, actual_all_zero=bool(np.all(actual == 0)))


def pinned_json(path, digest, checks):
    checks.check(sha(path) == digest, "Receipt SHA: " + Path(path).name)
    return json.loads(Path(path).read_text())


def artifact_path(directory, item, checks):
    checks.check(isinstance(item["file"], str) and Path(item["file"]).name == item["file"], "Local simple artifact name")
    path = Path(directory)/item["file"]
    checks.check(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Artifact bytes/SHA: " + path.name)
    return path


def read_npz(directory, item, declarations, checks):
    path = artifact_path(directory, item, checks)
    with np.load(path, allow_pickle=False) as saved:
        checks.check(set(saved.files) == set(declarations), "Exact NPZ keys: " + path.name)
        arrays = {key: saved[key] for key in saved.files}
    for key, array in arrays.items():
        checks.equal(metadata(array), declarations[key], "Raw array metadata: " + path.name + "." + key)
        checks.check(np.isfinite(array).all(), "Finite saved array: " + key)
    return arrays


def unchanged(value, checks, label):
    checks.check(value["sources_and_inputs_unchanged"] is True and value["source_pins"] == value["source_hashes_after"],
                 label + " board-recorded sources unchanged")
    checks.check(value["cleanup_errors"] == [], label + " recorded cleanup")
    checks.check(all(isinstance(path, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                     for path, digest in value["source_pins"].items()), label + " valid recorded pins")


def source_pin(value, basename, digest, checks):
    checks.check(any(Path(path).name == basename and got == digest for path, got in value["source_pins"].items()),
                 "Recorded pin: " + basename)


def gather_context(imap, indices):
    if imap.dtype != np.float32 or imap.shape != (1, 1, 384, 116, 152) or indices.dtype != np.int64 or indices.shape != (96,):
        raise ValueError("Original dense context/index shapes")
    y, x = np.divmod(indices, 152)
    if not np.all((1 <= x) & (x <= 150) & (1 <= y) & (y <= 114)):
        raise ValueError("Original valid integer center domain")
    return np.ascontiguousarray(imap[0, 0][:, y, x].T[None])


def load_gold(capture_dir, reference_dir, fp16_dir, fp16_sha, checks):
    capture = pinned_json(Path(capture_dir)/"capture.json", CAPTURE_SHA, checks)
    reference = pinned_json(Path(reference_dir)/"ort.json", REFERENCE_SHA, checks)
    historical = pinned_json(Path(fp16_dir)/"run.json", fp16_sha, checks)
    checks.check(capture["status"] == "completed_encoder_cpu_probe_not_VO" and capture["state_unchanged"] is True,
                 "Frozen original CPU capture and weights")
    checks.check(reference["status"] == "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO" and
                 reference["original_CPU_strict_gate_each_method"]["original"] is True and
                 reference["original_CPU_strict_gate_each_method"]["spatial"] is False and
                 reference["suffix_bitwise_gate_passed"] is True, "Frozen original reference; historical spatial rejection retained")
    checks.check(historical["schema"] == "v34_DPVO_sparse_context_RKNN_comparison" and historical["method"] == "original" and
                 historical["status"] == "completed_sparse_context_SDK_comparison_not_VO" and
                 historical["collect_perf_detail"] is False and historical["reference_sha256"] == REFERENCE_SHA,
                 "Historical FP16 same-input ordinary reference")
    for label, value in (("CPU", capture), ("ORT", reference), ("Historical FP16", historical)):
        unchanged(value, checks, label)
    checks.equal(capture["preprocessing"], reference["preprocessing"], "Frozen preprocessing")
    rows = [row for row in reference["calls"] if row["method"] == "original"]
    checks.check([row["frame_index"] for row in rows] == [0, 1, 2] and len(capture["inputs"]) == 3,
                 "Exactly all three evaluation inputs")
    gold, summaries = [], []
    for i, row in enumerate(rows):
        source = capture["inputs"][i]
        checks.check(row["frame_id"] == source["frame_id"] and row["source_image"] == source["source"], "Source RGB identity")
        inputs = read_npz(reference_dir, row["input_artifact"],
                          {"normalized_bgr": row["input_metadata"], "centers_xy": row["centers"], "patch_indices": row["indices"]}, checks)
        image, centers, indices = (inputs[k] for k in ("normalized_bgr", "centers_xy", "patch_indices"))
        checks.check(image.dtype == np.float32 and image.shape == (1, 3, 464, 608) and
                     -.500001 <= float(image.min()) <= float(image.max()) <= 1.500001, "Original normalized image domain")
        x5 = np.load(artifact_path(capture_dir, source["input"], checks), allow_pickle=False)
        checks.equal(metadata(x5), source["array"], "Original normalized singleton-view raw bytes")
        checks.check(x5.shape == (1, 1, 3, 464, 608) and image.tobytes(order="C") == x5.tobytes(order="C"),
                     "Same raw normalized input; no decode or normalization repeated")
        checks.check(centers.dtype == indices.dtype == np.int64 and centers.shape == (96, 2) and indices.shape == (96,) and
                     np.array_equal(indices, centers[:, 1]*152 + centers[:, 0]), "Original ordered index/center correspondence")
        cpu_row = next(item for item in capture["calls"] if item["frame_index"] == i)
        dense = read_npz(capture_dir, cpu_row["artifact"], cpu_row["arrays"], checks)
        checks.check(set(dense) == {"fmap", "imap"} and dense["fmap"].shape == (1, 1, 128, 116, 152), "Original full CPU features saved")
        cpu = {"fmap": np.ascontiguousarray(dense["fmap"][:, 0]), "context": gather_context(dense["imap"], indices)}
        original_ort = read_npz(reference_dir, row["artifact"], row["arrays"], checks)
        ort = {name: original_ort[name] for name in SHAPES}
        for name in SHAPES:
            checks.check(score(ort[name], cpu[name])["gate_passed"], "Original ORT versus CPU remains strict-pass: " + name)
            checks.check(original_ort["parent_" + ("fmap" if name == "fmap" else "context_gather")].tobytes(order="C") == ort[name].tobytes(order="C"),
                         "Original suffix versus saved parent bitwise: " + name)
        hrow = next(item for item in historical["calls"] if item["frame_index"] == i and item["role"] == "sparse")
        checks.check(hrow["frame_id"] == row["frame_id"] and hrow["source_image"] == row["source_image"] and
                     hrow["input_array"] == row["input_metadata"] and hrow["indices"] == row["indices"], "Historical FP16 identical input/indices")
        fp16 = read_npz(fp16_dir, hrow["artifact"], hrow["selected_arrays"], checks)
        summaries.append(dict(frame_index=i, frame_id=row["frame_id"], source_image=row["source_image"],
            input_array=row["input_metadata"], indices=row["indices"], CPU_gather_recomputed_from_saved_full_imap=True,
            original_CPU_arrays={k: metadata(v) for k, v in cpu.items()}, historical_FP16_arrays={k: metadata(v) for k, v in fp16.items()},
            historical_FP16_vs_CPU={k: score(fp16[k], cpu[k]) for k in SHAPES}))
        gold.append(dict(identity=row, cpu=cpu, ort=ort, fp16=fp16))
    return gold, summaries, historical, reference


def check_timing(row, checks):
    timing = row["timing"]
    checks.check(all(type(timing[k]) in (float, int) and math.isfinite(timing[k]) and timing[k] >= 0 for k in STAGES),
                 "Finite nonnegative SDK stage timings")
    checks.check(sum(timing[k] for k in STAGES) <= timing["sdk_io_seconds"] + 1e-8, "Nested timings not double-counted; full endpoint covers all IO")
    checks.equal(row["forward_seconds"], timing["sdk_io_seconds"], "Complete SDK endpoint time")
    checks.check(timing["input_api_types"] == [0, 8] and timing["input_api_formats"] == [1, 3] and
                 timing["input_pass_through"] == [0, 0] and timing["input_logical_dtypes"] == ["float32", "int64"] and
                 timing["input_buffer_shapes"] == [[1, 464, 608, 3], [96]] and timing["input_buffer_bytes"] == [3385344, 768],
                 "FP32 NHWC SDK quantization request; untouched INT64 topology")
    checks.check(timing["output_want_float"] == 1 and timing["output_is_prealloc"] == 0 and
                 timing["output_bytes"] == [9027584, 147456] and timing["synchronous"] is True and
                 timing["output_buffers_released"] is True and
                 timing["output_size_policy"] == "FP32 logical bytes only; never size_with_stride", "Owned logical FP32 outputs released")


def parse_profile(detail, checks):
    text = detail["raw_text"]
    checks.check(detail["query_command"] == 3 and len(text.encode()) == detail["bytes"] and
                 hashlib.sha256(text.encode()).hexdigest() == detail["raw_sha256"], "Raw profile exact bytes/hash")
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\d+\s", line):
            continue
        fields = line.split()
        checks.check(len(fields) >= 11 and fields[3] in ("NPU", "CPU", "GPU") and re.fullmatch(r"\d+/\d+/\d+", fields[6]),
                     "Recognized actual operator profile row")
        workloads = [field for field in fields[8:] if re.fullmatch(r"\d+(?:\.\d+)?%/\d+(?:\.\d+)?%/\d+(?:\.\d+)?%", field)]
        checks.check(len(workloads) == 1, "Three-core workload column")
        rows.append(dict(index=int(fields[0]), type=fields[1], dtype=fields[2], target=fields[3], elapsed_us=int(fields[7]),
                         core_workload_percent=[float(v[:-1]) for v in workloads[0].split("/")], name=fields[-1]))
    checks.check(rows and [row["index"] for row in rows] == list(range(1, len(rows) + 1)), "Complete contiguous profile rows")
    total = re.search(r"Total Operator Elapsed Per Frame Time\(us\):\s*(\d+)", text)
    checks.check(total is not None and sum(row["elapsed_us"] for row in rows) == int(total[1]), "Operator elapsed total")
    rank = {}
    for line in text.split("Operator Time Consuming Ranking Table")[-1].splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0] not in ("OpType", "Total") and all(re.fullmatch(r"\d+", field) for field in fields[1:6]):
            checks.check(fields[0] not in rank, "Unique ranking operator type")
            rank[fields[0]] = [int(v) for v in fields[1:6]]
    by_type = defaultdict(lambda: [0, 0, 0, 0, 0])
    for row in rows:
        aggregate = by_type[row["type"]]
        aggregate[0] += 1
        aggregate[{"CPU": 1, "GPU": 2, "NPU": 3}[row["target"]]] += row["elapsed_us"]
        aggregate[4] += row["elapsed_us"]
    checks.equal(dict(by_type), rank, "Independent operator sums versus SDK ranking")
    npu = [row for row in rows if row["target"] == "NPU"]
    cpu = [row for row in rows if row["target"] == "CPU"]
    io = {"InputOperator", "OutputOperator"}
    transposes = [row for row in cpu if row["type"] == "Transpose"]
    target_us = {target: sum(row["elapsed_us"] for row in rows if row["target"] == target) for target in ("CPU", "NPU", "GPU")}
    return dict(raw_profile_sha256=detail["raw_sha256"], rows=len(rows), total_operator_us=int(total[1]),
        target_counts=dict(Counter(row["target"] for row in rows)),
        target_dtype_counts={target: dict(Counter(row["dtype"] for row in rows if row["target"] == target)) for target in target_us},
        target_elapsed_us=target_us, operator_type_count_and_time=rank,
        internal_CPU_rows=[row for row in cpu if row["type"] not in io], CPU_IO_rows=sum(row["type"] in io for row in cpu),
        CPU_transpose_us=sum(row["elapsed_us"] for row in transposes),
        CPU_transpose_fraction_of_operator_elapsed=sum(row["elapsed_us"] for row in transposes)/int(total[1]),
        NPU_INT8_label_rows=sum(row["dtype"] == "INT8" for row in npu),
        SDK_dtype_labels_are_not_independent_kernel_precision_measurements=True)


def run_check(value, profile, compiled, query, gold, checks):
    checks.check(value["schema"] == "v36_DPVO_sparse_encoder_PTQ_inference_diagnostic" and
                 value["status"] == "completed_encoder_PTQ_diagnostic_not_VO" and value["script_sha256"] == ENTRY_SHA,
                 "Actual frozen original PTQ driver/schema")
    unchanged(value, checks, "INT8 profile" if profile else "INT8 ordinary")
    checks.check(value["collect_perf_detail"] is profile and value["capture_sha256"] == CAPTURE_SHA and value["reference_sha256"] == REFERENCE_SHA,
                 "Normal/profile separation and frozen gold chain")
    for key in ("candidate_accepted", "complete_VO", "realtime_validated", "Update_executed", "correlation_executed",
                "BA_executed", "images_decoded", "output_clipping", "manual_input_quantization", "production_changed", "ORB_used"):
        checks.check(value[key] is False, "No unsupported execution/acceptance claim: " + key)
    checks.check(value["actual_board_execution"] is True and value["NPU_used"] is True and value["normalization_applications"] == 0,
                 "Actual SDK use without re-decoding/normalizing")
    checks.equal(value["original_FP32_gate"], dict(rtol=RTOL, atol=ATOL), "Unchanged FP32 gate")
    checks.check(value["affinity_restored"] is True and value["cpu_affinity"] == [4, 5, 6, 7], "Four-A76 affinity/restore")
    checks.equal(value["model"], compiled["output"], "Actual quantized artifact lineage")
    checks.equal(value["configuration"], compiled["configuration"], "Exact PTQ configuration")
    checks.check(value["context"]["context_destroyed"] is True and value["context"]["destroy_return_code"] == 0 and
                 value["context"]["core_mask"] == 7, "Successful one-context destroy/core-mask")
    for key in ("input_attrs", "output_attrs", "sdk_versions"):
        checks.equal(value["context"][key], query[key], "Actual native query agrees: " + key)
    source_pin(value, "run-v36-dpvo-int8-board.py", ENTRY_SHA, checks)
    source_pin(value, "v34_sparse_rknn.py", TRANSPORT_SHA, checks)
    source_pin(value, "query-v36-int8-io-board.py", QUERY_SOURCE_SHA, checks)
    source_pin(value, "block-int8.rknn", compiled["output"]["sha256"], checks)
    checks.check(value["execution_arithmetic_delegate_sha256"] == TRANSPORT_SHA and value["expected_output_FP32_logical_bytes"] == 9175040,
                 "Frozen IO arithmetic and logical FP32 byte budget")
    expected = [(i, "profile") for i in range(3)] if profile else [(0, "cold"), (0, "warm"), (0, "warm"), (0, "warm"), (1, "other_frame"), (2, "other_frame")]
    checks.check([(row["frame_index"], row["timing_role"]) for row in value["calls"]] == expected and
                 value["sdk_run_api_calls"] == len(expected), "All prescribed actual calls; no hidden repeats")
    checks.equal(value["last_sdk_attempt"], value["calls"][-1]["timing"], "Last SDK attempt is final completed timing")
    for row in value["calls"]:
        source = gold[row["frame_index"]]["identity"]
        checks.check(row["frame_id"] == source["frame_id"] and row["source_image"] == source["source_image"] and
                     row["input_metadata"] == source["input_metadata"] and row["indices"] == source["indices"], "Actual call same raw input/indices/identity")
        check_timing(row, checks)
    warm = [row["forward_seconds"] for row in value["calls"] if row["timing_role"] == "warm"]
    expected_summary = dict(cold_seconds=None if profile else value["calls"][0]["forward_seconds"], warm_seconds=warm,
                            warm_mean_seconds=sum(warm)/len(warm) if warm else None, profile_timings_not_normal_benchmark=profile)
    checks.equal(value["timing_summary"], expected_summary, "Exact cold/three-warm summary")
    checks.check(value["actual_driver_seconds"] >= sum(row["forward_seconds"] for row in value["calls"]), "Driver includes timing-excluded verification/I/O")


def audit(args, checks, result):
    normal = pinned_json(args.normal_dir/"run.json", args.normal_sha256, checks)
    profile = pinned_json(args.profile_dir/"run.json", args.profile_sha256, checks)
    compiled = pinned_json(args.compile_json, args.compile_sha256, checks)
    query = pinned_json(args.query_json, args.query_sha256, checks)
    gold, result["references"], historical, reference = load_gold(args.capture_dir, args.reference_dir, args.fp16_dir, args.fp16_sha256, checks)
    checks.check(compiled["status"] == "compiled_board_PTQ_not_runtime_or_trajectory_validated" and compiled["released"] is True,
                 "Completed isolated board PTQ build")
    checks.equal(compiled["configuration"], dict(target_platform="rk3588", float_dtype="float16", optimization_level=3,
        quantized_dtype="w8a8", quantized_algorithm="normal", quantized_method="channel", mean_values=None,
        std_values=None, quant_img_RGB2BGR=False, do_quantization=True, auto_hybrid=False), "Frozen exact PTQ configuration")
    checks.check(query["schema"] == "v36_DPVO_board_PTQ_native_IO_query" and
                 query["status"] == "queried_native_IO_not_inference_validated" and query["sdk_run_calls"] == 0 and
                 query["model_sha256"] == compiled["output"]["sha256"] and query["compile_receipt_sha256"] == args.compile_sha256,
                 "Pinned zero-run query of same quantized model")
    for value in (normal, profile):
        checks.check(value["compile_receipt_sha256"] == args.compile_sha256 and value["query_receipt_sha256"] == args.query_sha256 and
                     value["historical_FP16_receipt_sha256"] == args.fp16_sha256, "Reported reference receipts rehashed independently")
    checks.equal(normal["source_pins"], profile["source_pins"], "Both runs use identical source/input pins")
    result["actual_native_IO"] = dict(input_attrs=query["input_attrs"], output_attrs=query["output_attrs"],
        request_input_types=["FP32", "INT64"], request_input_formats=["NHWC", "UNDEFINED"], pass_through=[0, 0],
        SDK_image_quantization_requested=True, want_float=True, logical_FP32_output_bytes=9175040, no_manual_quantization=True)
    first = {}
    result["runs"] = {}
    for label, value, directory, collect in (("normal", normal, args.normal_dir, False), ("profile", profile, args.profile_dir, True)):
        run_check(value, collect, compiled, query, gold, checks)
        calls, local_first = [], {}
        for index, row in enumerate(value["calls"]):
            checks.check(row["index"] == index, "Recorded order index")
            arrays = read_npz(directory, row["artifact"], row["arrays"], checks)
            checks.check(set(arrays) == set(SHAPES), "All actual neural output ports saved")
            frame = row["frame_index"]
            record = dict(index=index, frame_index=frame, frame_id=row["frame_id"], artifact=row["artifact"], arrays=row["arrays"], timing=row["timing"])
            for key, destination in (("versus_original_CPU", "cpu"), ("versus_v33_CPU_ORT", "ort"), ("versus_historical_FP16", "fp16")):
                record[key] = {}
                for name, array in arrays.items():
                    checks.check(array.dtype == np.float32 and array.shape == SHAPES[name] and np.any(array != 0), "Actual owned finite nonzero logical FP32: " + name)
                    computed = score(array, gold[frame][destination][name])
                    checks.equal(row[key][name], computed, "Independent numeric comparison: " + key + "." + name)
                    record[key][name] = computed
            counts = {k: dict(nonfinite=int(np.count_nonzero(~np.isfinite(v))), all_zero=bool(np.all(v == 0))) for k, v in arrays.items()}
            checks.equal(row["finite_counts"], counts, "Independently counted nonfinite/zero output")
            checks.check(row["finite_nonzero"] is True, "Finite/nonzero recorded correctly")
            repeated = None if frame not in local_first else all(arrays[k].tobytes(order="C") == local_first[frame][k].tobytes(order="C") for k in SHAPES)
            checks.equal(row["repeat_raw_hash_equal"], repeated, "Same-run repeats independently read")
            local_first.setdefault(frame, arrays)
            if label == "normal":
                first.setdefault(frame, arrays)
            else:
                checks.check(all(arrays[k].tobytes(order="C") == first[frame][k].tobytes(order="C") for k in SHAPES),
                             "Profile/ordinary outputs C-order byte-exact for same input")
            if collect:
                record["profile"] = parse_profile(row["perf_detail"], checks)
                checks.check(record["profile"]["rows"] == 95 and record["profile"]["target_counts"] == {"CPU": 15, "NPU": 80} and
                             record["profile"]["NPU_INT8_label_rows"] == 80 and record["profile"]["CPU_IO_rows"] == 4 and
                             Counter(r["type"] for r in record["profile"]["internal_CPU_rows"]) == {"Transpose": 10, "Gather": 1},
                             "Actual placement: 80 SDK INT8 NPU labels; 10 CPU transposes, gather, 4 IO")
                checks.check(record["profile"]["total_operator_us"]/1e6 <= row["timing"]["run_api_seconds"] + 0.005,
                             "Internal profile and complete synchronous run timing scopes")
            else:
                checks.check("perf_detail" not in row, "Ordinary benchmark has profiling disabled")
            calls.append(record)
        independent_gate = all(v["gate_passed"] for row in calls for v in row["versus_original_CPU"].values())
        checks.equal(value["actual_original_CPU_strict_gate_passed"], independent_gate, "All original precision gates recomputed")
        result["runs"][label] = dict(receipt_sha256=args.profile_sha256 if collect else args.normal_sha256,
            calls=calls, call_count=len(calls), saved_output_arrays=2*len(calls), original_FP32_gate_passed=independent_gate,
            timing_summary=value["timing_summary"], recorded_context=value["context"], actual_driver_seconds=value["actual_driver_seconds"],
            peak_rss_mib=value["peak_rss_mib"], verified_release_records=len(calls), recorded_successful_context_destroy_count=1,
            no_runtime_errors_in_recorded_receipt="error" not in value and value["cleanup_errors"] == [])
        timed_rows = [row for row in calls if value["calls"][row["index"]]["timing_role"] == "warm"] if not collect else calls
        result["runs"][label]["mean_nested_stage_seconds"] = {
            key: sum(row["timing"][key] for row in timed_rows)/len(timed_rows) for key in STAGES}
        result["runs"][label]["mean_stage_sum_not_additional_time_seconds"] = sum(result["runs"][label]["mean_nested_stage_seconds"].values())
        log = directory/"launcher.log"
        checks.check(log.is_file(), "Actual launcher log provided")
        text = log.read_text(errors="replace")
        errors = [line for line in text.splitlines() if re.search(r"(?:^|\s)E\s+RKNN\b|Traceback|normalize error|Unsupport CPU op|Error:", line)]
        checks.check(not errors, "No actual SDK/error markers in provided launcher log")
        result["runs"][label]["launcher_log"] = dict(file=log.name, sha256=sha(log), bytes=log.stat().st_size, error_markers=errors,
                                                     interpretation="Only the pulled launcher log is inspected; not kernel/system logs.")
    profiles = [row["profile"] for row in result["runs"]["profile"]["calls"]]
    result["profile_bottleneck"] = dict(
        CPU_transpose_us=[p["CPU_transpose_us"] for p in profiles],
        total_operator_us=[p["total_operator_us"] for p in profiles],
        CPU_transpose_operator_fraction=[p["CPU_transpose_fraction_of_operator_elapsed"] for p in profiles],
        actual_profile_exNorm_us=[p["operator_type_count_and_time"]["exNorm"][4] for p in profiles],
        actual_profile_conv_family_us=[sum(v[4] for k, v in p["operator_type_count_and_time"].items() if k.startswith("Conv")) for p in profiles],
        observed_fact="All three actual profiles retain 10 internal CPU transposes at the original fnet InstanceNormalization boundaries. Their summed elapsed time dominates the operator table; 80 NPU INT8 labels do not remove these CPU operations.",
        scope="Independent profiling runs; operator elapsed times exclude SDK packing/get/copy/release. They are not added to the enclosing run_api_seconds.")
    result["historical_comparison"] = dict(FP16_receipt_sha256=args.fp16_sha256, source_and_input_identity_checked=True,
        FP16_sparse_warm_seconds=historical["timing_summary"]["sparse"]["warm_seconds"],
        FP16_sparse_warm_mean_seconds=historical["timing_summary"]["sparse"]["warm_mean_seconds"],
        INT8_three_warm_mean_seconds=normal["timing_summary"]["warm_mean_seconds"],
        controlled_AB_test=False, causal_speedup_claim=False,
        limitation="Historical FP16 and current PTQ samples were not interleaved or matched for thermal/power state; descriptive timings and same-input saved feature errors only.")
    if args.fp16_profile_dir is not None:
        older = pinned_json(args.fp16_profile_dir/"run.json", args.fp16_profile_sha256, checks)
        unchanged(older, checks, "Historical FP16 profile")
        checks.check(older["method"] == "original" and older["collect_perf_detail"] is True and len(older["calls"]) == 3,
                     "Historical FP16 separate original profiles")
        parsed = [parse_profile(row["perf_detail"], checks) for row in older["calls"]]
        result["historical_comparison"].update(FP16_profile_receipt_sha256=args.fp16_profile_sha256,
            FP16_profile_CPU_transpose_us=[row["CPU_transpose_us"] for row in parsed],
            FP16_profile_exNorm_us=[row["operator_type_count_and_time"]["exNorm"][4] for row in parsed],
            FP16_profile_conv_family_us=[sum(v[4] for k, v in row["operator_type_count_and_time"].items() if k.startswith("Conv")) for row in parsed],
            FP16_profile_arrays_rehashed=False, FP16_profile_scope="Raw profile text rehashed/reparsed; historical profile arrays not needed or claimed.")
    result["evidence_scope"] = dict(actual_INT8_calls=9, actual_saved_INT8_output_arrays_rehashed=18,
        original_CPU_full_imap_and_index_gather_independently_recomputed=True,
        source_before_after_equality="Board-recorded SHA maps checked; unavailable board source/checkpoint/library/model not independently host-rehashed.",
        all_source_pins_host_rehashed=False, native_quantized_model_host_rehashed=False,
        SDK_profiles_independently_parsed=3, SDK_internal_INT8_labels_verified=True,
        all_internal_operators_on_NPU=False, actual_kernel_precision_independently_measured=False,
        original_FP32_equivalence=result["runs"]["normal"]["original_FP32_gate_passed"],
        complete_VO_executed=False, trajectory_accuracy_evaluated=False, realtime_20Hz_validated=False,
        historical_spatial_parent_FP32_rejection_retained=True,
        unchanged_FP32_gate=dict(rtol=RTOL, atol=ATOL), no_threshold_relaxation=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("normal-dir", "profile-dir", "capture-dir", "reference-dir", "fp16-dir", "compile-json", "query-json", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("normal", "profile", "fp16", "compile", "query"):
        parser.add_argument("--" + key + "-sha256", required=True)
    parser.add_argument("--fp16-profile-dir", type=Path)
    parser.add_argument("--fp16-profile-sha256")
    args = parser.parse_args(argv)
    result = dict(schema="v37_DPVO_encoder_INT8_saved_artifact_independent_review", status="failed", candidate_accepted=False,
                  audit_script_sha256=sha(__file__), audit_executes_model_or_SDK=False,
                  bindings={k: v for k, v in vars(args).items() if k.endswith("sha256")})
    checks = Checks()
    try:
        checks.check((args.fp16_profile_dir is None) == (args.fp16_profile_sha256 is None), "Both optional FP16 profile arguments")
        audit(args, checks, result)
        result["status"] = "passed_evidence_consistent_candidate_FP32_rejected"
        result["failed_checks"] = 0
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc(), failed_checks=1)
    result["checks"] = checks.count
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "failed_checks")}), flush=True)
    return 0 if result["failed_checks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
