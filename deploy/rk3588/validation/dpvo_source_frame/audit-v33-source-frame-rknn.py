#!/usr/bin/env python3
"""NumPy-only audit of saved v33 source-frame RKNN diagnostics.

Never imports an SDK, Torch or ORT, and never executes a model. An audit pass
means recorded evidence is consistent, not that the FP32-rejected candidate is
accepted. Keep the adjacent DPVO MIT attribution and LICENSE.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

INPUTS = ("net", "inp", "corr")
OUTPUTS = ("net_out", "delta", "weight")
PINS = {
    "run-v33-source-frame-rknn-board-r2.py": "813ab6f78663fc58474f7c624590b5dc0e341a2522c4279128e99d7427bb7115",
    "v33_rknn_source_frame.py": "322021ade54fb9035b7ce80652ec90169938a417fd2192528d28a805a83ed25e",
    "v33_rknn_source_frame_padded.py": "a5af6d39b2d19471255fc86aa42c9aa9b0247e428715c23ffd4ba79b184ad865",
    "compile-v33-source-frame-update.py": "9485c57075161e932841cf05ac9c0226e349fe66bebea462b45ec69d1d0b16df",
}
EXTERNAL_PINS = {
    "run-v12-rknn-block-board.py": "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608",
    "rknn_api_v2.3.2.h": "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a",
    "librknnrt.so": "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8",
}
MANIFEST_SHA = "3f45c586bcdfdd47cd81724fb4008e8dcac975d86257a1ec58b3fca97767ba13"
COMPILE_SHA = "864d5d3e538e2efc58c3de6724c2eeb2abc4a05e13b39fced44e90ff9221a0f7"
CONFIG = dict(target_platform="rk3588", float_dtype="float16", optimization_level=3,
              do_quantization=False)


class Checks:
    def __init__(self):
        self.passed = 0

    def check(self, condition, label):
        if not bool(condition):
            raise ValueError(label)
        self.passed += 1


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def load_artifact(folder, item, checks):
    name = item["file"]
    checks.check(isinstance(name, str) and Path(name).name == name and name not in ("", ".", ".."),
                 "Artifact basename required")
    path = Path(folder) / name
    checks.check(path.is_file() and not path.is_symlink(), "Regular artifact required")
    checks.check(path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"],
                 "Artifact file bytes/SHA differ: " + name)
    with np.load(path, allow_pickle=False) as archive:
        checks.check(len(archive.files) == len(set(archive.files)), "Duplicate NPZ keys")
        values = {key: archive[key].copy() for key in archive.files}
    checks.check(set(values) == set(item["arrays"]), "Artifact array keys differ")
    for key, value in values.items():
        checks.check(metadata(value) == item["arrays"][key], "Raw array SHA/shape/dtype differ: " + key)
    return values


def compare(actual, reference):
    if actual.dtype != np.float32 or reference.dtype != np.float32 or actual.shape != reference.shape:
        raise ValueError("Matching FP32 shapes required")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Finite saved arrays required")
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    difference = a - b
    absolute = np.abs(difference)
    tolerance = 1e-5 + 1e-4 * np.abs(b)
    norm = float(np.linalg.norm(b.ravel()))
    bad = int(np.count_nonzero(absolute > tolerance))
    return dict(finite=True, gate_passed=bad == 0, rtol=1e-4, atol=1e-5,
                failed_elements=bad, nonfinite_elements=0, max_abs=float(absolute.max()),
                relative_l2=float(np.linalg.norm(difference.ravel())) / norm if norm else None,
                max_normalized_tolerance=float((absolute / tolerance).max()))


def check_comparison(actual, recorded, checks):
    checks.check(set(recorded) == set(actual) - {"max_normalized_tolerance"}, "Comparison keys differ")
    for key, value in recorded.items():
        if key == "relative_l2" and value is not None:
            checks.check(math.isclose(value, actual[key], rel_tol=1e-10, abs_tol=1e-14), "Relative L2 differs")
        else:
            checks.check(value == actual[key], "Recorded comparison differs: " + key)


def parse_profile(record, checks):
    text = record["raw_text"]
    raw = text.encode("utf-8")
    checks.check(record["query_command"] == 3 and record["bytes"] == len(raw)
                 and record["raw_sha256"] == hashlib.sha256(raw).hexdigest(), "Raw runtime profile SHA differs")
    rows = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+", line)
        if not match:
            continue
        fields = line.split()
        checks.check(len(fields) >= 9 and re.fullmatch(r"\d+/\d+/\d+", fields[6]) is not None,
                     "Malformed runtime operator row")
        elapsed = int(fields[7])
        checks.check(elapsed >= 0, "Negative operator time")
        rows.append(dict(index=int(match[1]), operator=match[2], dtype=match[3], target=match[4],
                         cycles=[int(v) for v in fields[6].split("/")], time_us=elapsed))
    checks.check(bool(rows) and [r["index"] for r in rows] == list(range(1, len(rows) + 1)),
                 "Missing/duplicate operator IDs")
    io = [r for r in rows if r["operator"] in ("InputOperator", "OutputOperator")]
    cpu = [r for r in rows if r["target"] == "CPU" and r not in io]
    npu = [r for r in rows if r["target"] == "NPU" and r not in io]
    checks.check(all(r["target"] == "CPU" for r in io) and len(rows) == len(io) + len(cpu) + len(npu),
                 "Unclassified runtime placement")
    return dict(operator_rows=len(rows), internal_NPU=len(npu), internal_CPU=len(cpu), CPU_IO=len(io),
                internal_CPU_operators=dict(Counter(r["operator"] for r in cpu)),
                internal_CPU_rows=cpu, internal_NPU_time_us=sum(r["time_us"] for r in npu),
                internal_CPU_time_us=sum(r["time_us"] for r in cpu),
                CPU_IO_time_us=sum(r["time_us"] for r in io), raw_sha256=record["raw_sha256"],
                entire_computation_on_NPU=not cpu)


def verify_transport(session, inputs, outputs, checks):
    checks.check(len(session["input_attributes"]) == len(inputs) == 3
                 and len(session["output_attributes"]) == len(session["output_bindings"]) == len(outputs) == 3,
                 "Three input/output bindings required")
    for rows, contracts, is_output in ((session["input_attributes"], inputs, False),
                                        (session["output_attributes"], outputs, True)):
        for index, (row, contract) in enumerate(zip(rows, contracts)):
            shape = contract["shape"]
            native_shape = shape if is_output else [shape[0], shape[2], shape[3], shape[1]]
            size = math.prod(shape)
            checks.check(row["index"] == index and row["name"] == contract["name"] and row["shape"] == native_shape,
                         "SDK index/name/layout differs")
            checks.check(row["n_elems"] == size and row["size"] == size * 2 and row["type"] == 1
                         and row["format"] == (0 if is_output else 1), "FP16 native transport differs")
            checks.check(row["quantization_type"] == 2 and row["scale"] == 1.0 and row["zero_point"] == 0
                         and row["fractional_length"] == row["pass_through"] == row["h_stride"] == 0,
                         "Only neutral affine FP16 metadata accepted")
            checks.check(row["size_with_stride"] >= row["size"] if is_output
                         else row["size_with_stride"] == row["size"], "Unsupported native stride")
            if is_output:
                binding = session["output_bindings"][index]
                checks.check(binding["name"] == contract["name"] and binding["shape"] == shape
                             and binding["logical_shape"] == shape and binding["logical_dtype"] == "float32"
                             and binding["sdk_shape"] == native_shape and binding["index"] == index,
                             "Output canonical binding differs")
                checks.check(binding["sdk_native_bytes"] == row["size"]
                             and binding["sdk_size_with_stride"] == row["size_with_stride"]
                             and binding["native_output_padding_bytes"] == row["size_with_stride"] - row["size"],
                             "Output padding record differs")
                checks.check(binding["returned_FP32_bytes"] == binding["transport_bytes"] == size * 4
                             and binding["want_float"] == 1 and binding["permutation"] is None
                             and binding["transport_type"] == binding["transport_format"] == binding["pass_through"] == 0,
                             "Returned owned logical FP32 output contract differs")
                checks.check(binding["sdk_format"] == 0 and binding["sdk_type"] == 1
                             and binding["sdk_quantization_type"] == 2 and binding["neutral_affine_metadata"]
                             and binding["sdk_fractional_length"] == binding["sdk_zero_point"] == 0
                             and binding["sdk_scale"] == 1.0, "Output SDK metadata differs")


def verify_call_timing(call, contracts, checks):
    timing = call["timing"]
    parts = ("pack_seconds", "input_set_seconds", "run_api_seconds", "outputs_get_seconds",
             "owned_copy_seconds", "outputs_release_seconds")
    for key in parts + ("forward_seconds",):
        checks.check(math.isfinite(timing[key]) and timing[key] >= 0, "Invalid call time: " + key)
    checks.check(sum(timing[k] for k in parts) <= timing["forward_seconds"] + 1e-8,
                 "Forward omits SDK IO/packing/release")
    checks.check(timing["forward_seconds"] <= call["total_execute_seconds"] + 1e-8
                 and call["total_execute_seconds"] <= call["adapter_execute_wall_seconds"] + 1e-8
                 and math.isfinite(call["adapter_execute_wall_seconds"]), "Inconsistent adapter time scopes")
    checks.check(timing["synchronous"] is True and len(timing["input_records"]) == 3,
                 "Synchronous owned inputs required")
    for index, (record, contract) in enumerate(zip(timing["input_records"], contracts)):
        shape = contract["shape"]
        expected = dict(index=index, name=contract["name"], logical_shape=shape,
                        buffer_shape=[shape[0], shape[2], shape[3], shape[1]], source_dtype="float32",
                        transport_dtype="float32", native_attribute_format=1, request_format=1,
                        pass_through=0, bytes=math.prod(shape) * 4, owned=True)
        checks.check(record == expected, "Actual FP32 NHWC input packing contract differs")


def check_pins(record, checks):
    checks.check(record["source_pins"] and record["source_pins"] == record["source_hashes_after"]
                 and record["sources_and_inputs_unchanged"] is True, "Before/after source pins differ")


def digest_by_basename(pins, basename, expected, checks):
    values = [sha for path, sha in pins.items() if Path(path).name == basename]
    checks.check(values == [expected], "Source basename/digest differs: " + basename)


def audit_run(folder, manifest, compiled, references, checks, profile):
    folder = Path(folder)
    record = json.loads((folder / "run.json").read_text())
    checks.check(record["status"] == "completed_source_frame_RKNN_diagnostic_not_adopted",
                 "Completed diagnostic required")
    checks.check(record["script_sha256"] == PINS["run-v33-source-frame-rknn-board-r2.py"]
                 and record["runtime_adapter_revision"] == 2
                 and record["runtime_adapter_parent_sha256"] == PINS["v33_rknn_source_frame.py"]
                 and record["io_helper_sha256"] == PINS["v33_rknn_source_frame_padded.py"], "Runtime source revision differs")
    checks.check(record["export_sha256"] == MANIFEST_SHA and record["compile_sha256"] == COMPILE_SHA
                 and record["configuration"] == CONFIG and record["inputs"] == manifest["inputs"]
                 and record["outputs"] == manifest["outputs"]
                 and record["source_plans"] == [c["source_plan"] for c in manifest["cases"]], "Source/compile/IO chain differs")
    for key in ("candidate_accepted", "host_model_inference", "complete_VO_executed", "BA_executed",
                "realtime_validated", "trajectory_accuracy_validated", "ORB_used", "added_LK_PnP_frontend"):
        checks.check(record[key] is False, "Unsupported success/scope claim: " + key)
    checks.check(record["actual_inference_on_RK3588S"] is True and record["NPU_used"] is True
                 and record["original_VGGT_default_unchanged"] is True and record["different_model_from_VGGT"] is True,
                 "Actual diagnostic provenance differs")
    checks.check(record["collect_perf_detail"] is profile and not record["diagnostic_observe_one_invalid_output"]
                 and record["cpu_affinity"] == [4, 5, 6, 7] and record["cleanup_errors"] == []
                 and record["affinity_restored"] is True, "Mode/cleanup/affinity differs")
    check_pins(record, checks)
    for name in ("run-v33-source-frame-rknn-board-r2.py", "v33_rknn_source_frame.py", "v33_rknn_source_frame_padded.py"):
        digest_by_basename(record["source_pins"], name, PINS[name], checks)
    checks.check(all(record["source_pins"].get(k) == v for k, v in manifest["source_pins"].items()),
                 "Export source pins not inherited")
    digest_by_basename(record["source_pins"], "block.rknn", compiled["output"]["sha256"], checks)
    session = record["session"]
    count = 2 if profile else 6
    checks.check(len(record["calls"]) == count and session["sdk_run_api_calls"] == session["completed_calls"]
                 == session["SDK_outputs_releases"] == count, "Actual SDK lifecycle counters differ")
    checks.check(session["synthetic"] is False and session["npu_used"] is True and session["closed"] is True
                 and session["context_destroyed"] is True and session["cleanup_errors"] == []
                 and session["core_mask"] == 7 and session["init_flags"] == (8 if profile else 0),
                 "Real context/core/profile/cleanup contract differs")
    check_pins(session, checks)
    for name, sha in EXTERNAL_PINS.items():
        digest_by_basename(session["source_pins"], name, sha, checks)
    provenance = session["implementation_provenance"]
    checks.check(provenance["returned_buffer_bytes_verified_at_each_call"] is True
                 and provenance["packed_FP32_output_not_guaranteed_by_attribute_alone"] is True
                 and provenance["runtime_helper_sha256"] == PINS["v33_rknn_source_frame_padded.py"]
                 and provenance["parent_sha256"] == PINS["v33_rknn_source_frame.py"]
                 and provenance["header_sha256"] == EXTERNAL_PINS["rknn_api_v2.3.2.h"], "Output ownership source provenance differs")
    verify_transport(session, manifest["inputs"], manifest["outputs"], checks)
    schedule = [(0, "profile"), (1, "profile")] if profile else [(c, r) for c in (0, 1) for r in ("first", "warm", "warm")]
    representatives, profiles, call_results = {}, [], []
    gates = []
    for index, (call, (case_index, role)) in enumerate(zip(record["calls"], schedule)):
        checks.check(call["index"] == index and call["case_index"] == case_index and call["timing_role"] == role,
                     "Actual call order/identity differs")
        checks.check(call["execution_status"] == "completed" and call["outputs_released"] is True
                     and call["candidate_accepted"] is False, "Call execution/release scope differs")
        verify_call_timing(call, manifest["inputs"], checks)
        log = call["native_log"]
        checks.check(log["RKNN_error_detected"] is False and log["truncated"] is False
                     and log["bytes"] == len(log["raw_text"].encode())
                     and log["raw_sha256"] == hashlib.sha256(log["raw_text"].encode()).hexdigest(), "Native error/log evidence differs")
        if "artifact" in call:
            values = load_artifact(folder, call["artifact"], checks)
            checks.check(case_index not in representatives, "Unexpected duplicate saved representative")
            representatives[case_index] = values
        else:
            checks.check(case_index in representatives and call["repeat_raw_hash_equal"] is True,
                         "Unsaved repeat lacks earlier representative hash")
            values = representatives[case_index]
        checks.check(set(values) == set(OUTPUTS) and call["arrays"] == {k: metadata(v) for k, v in values.items()},
                     "Call/representative raw arrays differ")
        output_comparisons = {"versus_reference": {}, "versus_CPU_ORT": {}}
        for name in OUTPUTS:
            value = values[name]
            contract = next(x for x in manifest["outputs"] if x["name"] == name)
            checks.check(value.dtype == np.float32 and list(value.shape) == contract["shape"]
                         and np.isfinite(value).all(), "Canonical finite FP32 output required")
            stats = dict(elements=value.size, finite=value.size, nan=0, positive_inf=0, negative_inf=0,
                         all_zero=bool(np.all(value == 0)), minimum=float(value.min()), maximum=float(value.max()))
            checks.check(call["statistics"][name] == stats, "Saved output statistics differ")
            for comparison, ref_key in (("versus_reference", "reference"), ("versus_CPU_ORT", "ort")):
                result = compare(value, references[case_index][ref_key][name])
                check_comparison(result, call[comparison][name], checks)
                output_comparisons[comparison][name] = result
        gate = all(x["gate_passed"] for x in output_comparisons["versus_reference"].values())
        checks.check(call["strict_FP32_gate_passed"] is gate and call["strict_component_gate_passed"] is gate,
                     "Strict numerical status misreported")
        domain = bool(np.all((values["weight"] >= 0) & (values["weight"] <= 1)))
        checks.check(domain and call["weight_domain_valid"] is True and call["outputs_finite"] is True
                     and call["all_outputs_zero"] is False and any(np.any(v != 0) for v in values.values()),
                     "Finite/domain/nonzero evidence differs")
        checks.check(call["numerical_status"] == ("strict_FP32_gate_passed" if gate else "numerical_rejected_or_reference_missing"),
                     "Numerical acceptance label differs")
        gates.append(gate)
        if profile:
            placement = parse_profile(call["profile"], checks)
            checks.check(placement["operator_rows"] == 70 and placement["internal_NPU"] == 59
                         and placement["internal_CPU"] == 5 and placement["CPU_IO"] == 6
                         and placement["internal_CPU_operators"] == {"ReduceMax": 2, "Exp": 2, "Min": 1},
                         "Actual profile placement differs from this frozen graph")
            profiles.append(dict(case_index=case_index, **placement))
        else:
            checks.check("profile" not in call, "Ordinary benchmark contains profile instrumentation")
        call_results.append(dict(index=index, case_index=case_index, role=role, saved_artifact="artifact" in call,
                                 forward_seconds=call["timing"]["forward_seconds"],
                                 adapter_wall_seconds=call["adapter_execute_wall_seconds"],
                                 strict_FP32_gate_passed=gate, **output_comparisons))
    checks.check(record["actual_strict_FP32_gate_passed"] is all(gates)
                 and record["all_outputs_finite"] is True and record["all_weight_domains_valid"] is True,
                 "Run numerical summary differs")
    for case in (0, 1):
        rows = [c for c in record["calls"] if c["case_index"] == case]
        warm = [c["timing"]["forward_seconds"] for c in rows if c["timing_role"] == "warm"]
        expected = dict(first_seconds=next((c["timing"]["forward_seconds"] for c in rows if c["timing_role"] == "first"), None),
                        warm_seconds=warm, warm_mean_seconds=sum(warm) / len(warm) if warm else None)
        checks.check(record["timing"][str(case)] == expected, "Ordinary first/warm mean differs")
    checks.check(record["actual_driver_seconds"] >= sum(c["adapter_execute_wall_seconds"] for c in record["calls"]),
                 "Driver wall time omits calls")
    summary = dict(run_sha256=file_sha(folder / "run.json"), SDK_calls=count, SDK_outputs_releases=count,
                   context_destroyed=True, closed=True, cleanup_errors=[], strict_FP32_gate_passed=all(gates),
                   finite=True, weight_domain_valid=True, timing=record["timing"], call_comparisons=call_results,
                   profiles=profiles, memory_query_while_context_open=session["memory"],
                   post_destroy_DMA_zero_measured=False, peak_rss_mib=record["peak_rss_mib"],
                   sdk_version=session["sdk_version"], driver_version=session["driver_version"],
                   requested_core_mask=7, saved_output_artifacts=len(representatives))
    return summary, representatives, record


def audit(normal_dir, profile_dir, export_dir, compile_dir, source_dir, sdk_source_dir,
          attrs_dir=None, startup_dir=None):
    checks = Checks()
    export_dir, compile_dir = Path(export_dir), Path(compile_dir)
    checks.check(file_sha(export_dir / "manifest.json") == MANIFEST_SHA, "Frozen export manifest differs")
    checks.check(file_sha(compile_dir / "compile.json") == COMPILE_SHA, "Frozen compile receipt differs")
    manifest = json.loads((export_dir / "manifest.json").read_text())
    compiled = json.loads((compile_dir / "compile.json").read_text())
    checks.check(manifest["strict_FP32_gate_passed"] is True and manifest["cleanup_errors"] == []
                 and compiled["configuration"] == CONFIG and compiled["manifest_sha256"] == MANIFEST_SHA
                 and compiled["released"] is True and compiled["candidate_accepted"] is False
                 and compiled["original_parameters_unchanged"] is True, "Export/compile evidence differs")
    check_pins(compiled, checks)
    checks.check(compiled["io_helper_sha256"] == PINS["v33_rknn_source_frame.py"]
                 and compiled["input"] == manifest["model"] and compiled["inputs"] == manifest["inputs"]
                 and compiled["outputs"] == manifest["outputs"], "Compile ONNX/IO linkage differs")
    for folder, item in ((export_dir, manifest["model"]), (compile_dir, compiled["output"])):
        checks.check(Path(item["file"]).name == item["file"], "Model basename required")
        path = folder / item["file"]
        checks.check(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"]
                     and file_sha(path) == item["sha256"], "Model artifact SHA/bytes differ")
    local_sources = {}
    for name, sha in PINS.items():
        folder = Path(source_dir) if name.startswith("run-") else Path(sdk_source_dir)
        checks.check(file_sha(folder / name) == sha, "Current frozen source differs: " + name)
        local_sources[name] = sha
    references = []
    checks.check(len(manifest["cases"]) == 2, "Two real states required")
    for case in manifest["cases"]:
        references.append({key: load_artifact(export_dir, case[key + "_artifact"], checks)
                           for key in ("reference", "ort")})
        load_artifact(export_dir, case["inputs_artifact"], checks)
    normal, nvalues, nrecord = audit_run(normal_dir, manifest, compiled, references, checks, False)
    profile, pvalues, precord = audit_run(profile_dir, manifest, compiled, references, checks, True)
    checks.check(nrecord["session"]["input_attributes"] == precord["session"]["input_attributes"]
                 and nrecord["session"]["output_attributes"] == precord["session"]["output_attributes"], "Normal/profile attrs differ")
    for case in (0, 1):
        for name in OUTPUTS:
            checks.check(nvalues[case][name].tobytes() == pvalues[case][name].tobytes(),
                         "Saved normal/profile output bytes differ")
    prior = {}
    if attrs_dir is not None:
        path = Path(attrs_dir) / "query.json"
        query = json.loads(path.read_text())
        checks.check(query["status"] == "queried_attributes_no_inference" and query["inference_executed"] is False
                     and query["SDK_run_calls"] == 0 and query["context_destroyed"] is True
                     and query["cleanup_errors"] == [], "Attribute query must not count as inference")
        check_pins(query, checks)
        checks.check(query["inputs"] == nrecord["session"]["input_attributes"]
                     and query["outputs"] == nrecord["session"]["output_attributes"], "Startup queried attrs differ")
        prior["attrs_query"] = dict(sha256=file_sha(path), actual_run_calls=0, context_destroyed=True)
    if startup_dir is not None:
        path = Path(startup_dir) / "run.json"
        failed = json.loads(path.read_text())
        checks.check(failed["status"] == "failed" and failed["calls"] == [] and failed["NPU_used"] is False
                     and failed["candidate_accepted"] is False and "Padded native byte layout" in failed["error"],
                     "Original startup rejection evidence differs")
        check_pins(failed, checks)
        prior["r1_startup"] = dict(sha256=file_sha(path), actual_run_calls=0,
                                  reason="Padded native byte layout is unimplemented",
                                  context_destroyed_independently_verified_from_r1_receipt=False)
    for name, sha in local_sources.items():
        folder = Path(source_dir) if name.startswith("run-") else Path(sdk_source_dir)
        checks.check(file_sha(folder / name) == sha, "Audit source changed during saved-array review")
    return dict(schema="v33_saved_RKNN_source_frame_independent_audit", status="passed_audit_of_FP32_rejected_mixed_execution",
                checks_passed=checks.passed, failed_checks=0, auditor_sha256=file_sha(__file__),
                source_hashes=local_sources, external_board_source_pins=EXTERNAL_PINS,
                export_manifest_sha256=MANIFEST_SHA, compile_receipt_sha256=COMPILE_SHA,
                ONNX=manifest["model"], RKNN=compiled["output"], normal=normal, profile=profile,
                normal_profile_saved_outputs_bitwise_equal=True, prior_startup=prior,
                candidate_accepted=False, strict_thresholds_relaxed=False, entire_computation_on_NPU=False,
                actual_SDK_run_calls=8, independently_saved_actual_output_calls=4,
                other_normal_repeats_verified_by_recorded_raw_hashes=4,
                scope=dict(source_block=[1, 10, 96], full_graph_edges=23520, full_graph_execution=False,
                           model_inference_on_audit_host=False, ORT_execution_on_audit_host=False,
                           BA=False, fresh_observation=False, trajectory=False, realtime=False),
                limitations=["The full historical 151 MB input is absent; block input selection is bound to frozen board code and metadata, not recomputed here.",
                             "Four ordinary repeats have raw metadata but no separately saved arrays; two ordinary representatives and both profile calls are read independently.",
                             "Ownership/copy-before-release is supported by pinned runner/SDK helper and counters; pointers are not available in saved evidence.",
                             "SDK DMA is queried while the context is open. Destroy is reported, but post-destroy DMA zero is not measured.",
                             "Profile operator microseconds are instrumentation, not the ordinary end-to-end benchmark.",
                             "A D10/M96 source block is not complete Update over E23520 and does not establish VO or 20 Hz."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("normal-dir", "profile-dir", "export-dir", "compile-dir", "source-dir", "sdk-source-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("attrs-dir", "startup-dir"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("New audit output required")
    try:
        result = audit(args.normal_dir, args.profile_dir, args.export_dir, args.compile_dir,
                       args.source_dir, args.sdk_source_dir, args.attrs_dir, args.startup_dir)
    except Exception as error:
        result = dict(status="failed_saved_evidence_audit", error=repr(error), candidate_accepted=False,
                      auditor_sha256=file_sha(__file__), model_inference_on_audit_host=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks_passed", "error", "candidate_accepted")}))
    return 0 if result["status"].startswith("passed_audit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
