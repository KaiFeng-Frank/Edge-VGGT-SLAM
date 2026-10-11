#!/usr/bin/env python3
"""NumPy-only v34 saved RKNN audit, with explicitly pinned v33 helper code.

The I/O schemas remain v33 contracts; experiment identity is v34 direct Softmax
and is bound to the explicit manifest SHA. No Torch, ORT, SDK or model execution.
Original DPVO attribution: Copyright (c) 2022 Princeton University; MIT LICENSE.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np

HELPER_SHA = "30238c99ac6aaa874963f065fd6cf931ecf6e80d600f52c73adf3877f455adf1"
EXPORT_AUDITOR_SHA = "ecaea05e9958164d4b5abfa409680222ee5eff7a26b185171ff758e24739e606"
SOURCES = {
    "source_frame_update_softmax.py": "b89b9b84d3afb47fd926decddde0077222dc07930bf75ee08acb410dc4f6f3f4",
    "export-v34-source-frame-softmax-board.py": "487e9c759eb11749902e39f9e96891e561d9f56fb606d79c0c555de331633e96",
    "source_frame_update.py": "cf1adb2224750cd0659139689f4b8a5a8107f2e733e8d1c6f86ea009c86a1bb2",
    "export-v33-source-frame-board.py": "1bb1df7dc4e61e6f3d60fcd1170b267c478fd9600de1f0481ec1a73a42fa1b73",
}
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
OUTPUTS = ("net_out", "delta", "weight")
CONFIG = dict(target_platform="rk3588", float_dtype="float16", optimization_level=3, do_quantization=False)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pinned(path, expected, name):
    if file_sha(path) != expected:
        raise ValueError("Frozen audit helper SHA differs: " + Path(path).name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_v34_placement(record, checks, helpers):
    value = helpers.parse_profile(record, checks)
    checks.check(value["operator_rows"] == 64 and value["internal_NPU"] == 55
                 and value["internal_CPU"] == 3 and value["CPU_IO"] == 6
                 and value["internal_CPU_operators"] == {"Exp": 2, "Min": 1},
                 "Actual v34 profile placement differs")
    rows = []
    for line in record["raw_text"].splitlines():
        if re.match(r"^\s*\d+\s+\S+\s+\S+\s+(CPU|NPU|GPU)\s+", line):
            fields = line.split()
            rows.append(dict(operator=fields[1], target=fields[3], time_us=int(fields[7])))
    softmax = [r for r in rows if r["operator"] == "exSoftmax13"]
    transpose = [r for r in rows if r["operator"] == "Transpose"]
    checks.check(len(softmax) == 2 and all(r["target"] == "NPU" for r in softmax)
                 and len(transpose) == 10 and all(r["target"] == "NPU" for r in transpose)
                 and not any(r["operator"] == "ReduceMax" for r in rows),
                 "Runtime direct Softmax/Transpose/ReduceMax placement differs")
    value.update(NPU_Softmax_count=2, NPU_Softmax_time_us=sum(r["time_us"] for r in softmax),
                 NPU_Transpose_count=10, NPU_Transpose_time_us=sum(r["time_us"] for r in transpose),
                 ReduceMax_count=0)
    return value


def audit_run(folder, manifest, compiled, references, checks, profile, helpers, manifest_sha, compile_sha):
    folder = Path(folder)
    record = json.loads((folder / "run.json").read_text())
    checks.check(record["status"] == "completed_source_frame_RKNN_diagnostic_not_adopted",
                 "Completed diagnostic required")
    checks.check(record["script_sha256"] == PINS["run-v33-source-frame-rknn-board-r2.py"]
                 and record["runtime_adapter_revision"] == 2
                 and record["runtime_adapter_parent_sha256"] == PINS["v33_rknn_source_frame.py"]
                 and record["io_helper_sha256"] == PINS["v33_rknn_source_frame_padded.py"], "Runtime source revision differs")
    checks.check(record["export_sha256"] == manifest_sha and record["compile_sha256"] == compile_sha
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
    helpers.check_pins(record, checks)
    for name in ("run-v33-source-frame-rknn-board-r2.py", "v33_rknn_source_frame.py", "v33_rknn_source_frame_padded.py"):
        helpers.digest_by_basename(record["source_pins"], name, PINS[name], checks)
    checks.check(all(record["source_pins"].get(k) == v for k, v in manifest["source_pins"].items()),
                 "Export source pins not inherited")
    helpers.digest_by_basename(record["source_pins"], "block.rknn", compiled["output"]["sha256"], checks)
    session = record["session"]
    count = 2 if profile else 6
    checks.check(len(record["calls"]) == count and session["sdk_run_api_calls"] == session["completed_calls"]
                 == session["SDK_outputs_releases"] == count, "Actual SDK lifecycle counters differ")
    checks.check(session["synthetic"] is False and session["npu_used"] is True and session["closed"] is True
                 and session["context_destroyed"] is True and session["cleanup_errors"] == []
                 and session["core_mask"] == 7 and session["init_flags"] == (8 if profile else 0),
                 "Real context/core/profile/cleanup contract differs")
    helpers.check_pins(session, checks)
    for name, sha in EXTERNAL_PINS.items():
        helpers.digest_by_basename(session["source_pins"], name, sha, checks)
    provenance = session["implementation_provenance"]
    checks.check(provenance["returned_buffer_bytes_verified_at_each_call"] is True
                 and provenance["packed_FP32_output_not_guaranteed_by_attribute_alone"] is True
                 and provenance["runtime_helper_sha256"] == PINS["v33_rknn_source_frame_padded.py"]
                 and provenance["parent_sha256"] == PINS["v33_rknn_source_frame.py"]
                 and provenance["header_sha256"] == EXTERNAL_PINS["rknn_api_v2.3.2.h"], "Output ownership source provenance differs")
    helpers.verify_transport(session, manifest["inputs"], manifest["outputs"], checks)
    schedule = [(0, "profile"), (1, "profile")] if profile else [(c, r) for c in (0, 1) for r in ("first", "warm", "warm")]
    representatives, profiles, call_results = {}, [], []
    gates = []
    for index, (call, (case_index, role)) in enumerate(zip(record["calls"], schedule)):
        checks.check(call["index"] == index and call["case_index"] == case_index and call["timing_role"] == role,
                     "Actual call order/identity differs")
        checks.check(call["execution_status"] == "completed" and call["outputs_released"] is True
                     and call["candidate_accepted"] is False, "Call execution/release scope differs")
        helpers.verify_call_timing(call, manifest["inputs"], checks)
        log = call["native_log"]
        checks.check(log["RKNN_error_detected"] is False and log["truncated"] is False
                     and log["bytes"] == len(log["raw_text"].encode())
                     and log["raw_sha256"] == hashlib.sha256(log["raw_text"].encode()).hexdigest(), "Native error/log evidence differs")
        if "artifact" in call:
            values = helpers.load_artifact(folder, call["artifact"], checks)
            checks.check(case_index not in representatives, "Unexpected duplicate saved representative")
            representatives[case_index] = values
        else:
            checks.check(case_index in representatives and call["repeat_raw_hash_equal"] is True,
                         "Unsaved repeat lacks earlier representative hash")
            values = representatives[case_index]
        checks.check(set(values) == set(OUTPUTS) and call["arrays"] == {k: helpers.metadata(v) for k, v in values.items()},
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
                result = helpers.compare(value, references[case_index][ref_key][name])
                helpers.check_comparison(result, call[comparison][name], checks)
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
            placement = verify_v34_placement(call["profile"], checks, helpers)
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
    summary = dict(run_sha256=helpers.file_sha(folder / "run.json"), SDK_calls=count, SDK_outputs_releases=count,
                   context_destroyed=True, closed=True, cleanup_errors=[], strict_FP32_gate_passed=all(gates),
                   finite=True, weight_domain_valid=True, timing=record["timing"], call_comparisons=call_results,
                   profiles=profiles, memory_query_while_context_open=session["memory"],
                   post_destroy_DMA_zero_measured=False, peak_rss_mib=record["peak_rss_mib"],
                   sdk_version=session["sdk_version"], driver_version=session["driver_version"],
                   requested_core_mask=7, saved_output_artifacts=len(representatives))
    return summary, representatives, record

def audit(normal_dir, profile_dir, manifest_path, manifest_sha, compile_dir, compile_sha,
          source_dir, v33_source_dir, sdk_source_dir):
    source_dir, v33_source_dir, sdk_source_dir = map(Path, (source_dir, v33_source_dir, sdk_source_dir))
    helper_path = v33_source_dir / "audit-v33-source-frame-rknn.py"
    helpers = load_pinned(helper_path, HELPER_SHA, "v34_frozen_RKNN_array_helpers")
    export_helper_path = source_dir / "audit-v34-source-frame-softmax.py"
    export_helper = load_pinned(export_helper_path, EXPORT_AUDITOR_SHA, "v34_frozen_export_scope")
    checks = helpers.Checks()
    manifest_path, compile_dir = Path(manifest_path), Path(compile_dir)
    export_dir = manifest_path.parent
    checks.check(file_sha(manifest_path) == manifest_sha, "Explicit v34 export manifest SHA differs")
    compile_path = compile_dir / "compile.json"
    checks.check(file_sha(compile_path) == compile_sha, "Explicit v34 compile receipt SHA differs")
    manifest, compiled = json.loads(manifest_path.read_text()), json.loads(compile_path.read_text())
    export_helper.check_scope(manifest, checks)
    checks.check(manifest["strict_FP32_gate_passed"] is True and manifest["reference_manifest_sha256"]
                 == "3f45c586bcdfdd47cd81724fb4008e8dcac975d86257a1ec58b3fca97767ba13",
                 "v34 CPU/ORT and historical gold binding differ")
    checks.check(compiled["configuration"] == CONFIG and compiled["manifest_sha256"] == manifest_sha
                 and compiled["released"] is True and compiled["candidate_accepted"] is False
                 and compiled["original_parameters_unchanged"] is True
                 and compiled["strict_thresholds_relaxed"] is False and compiled["production_changed"] is False
                 and compiled["model_inference_executed"] is False,
                 "Compile evidence or scope differs")
    helpers.check_pins(compiled, checks)
    checks.check(compiled["io_helper_sha256"] == PINS["v33_rknn_source_frame.py"]
                 and compiled["script_sha256"] == PINS["compile-v33-source-frame-update.py"]
                 and compiled["input"] == manifest["model"] and compiled["inputs"] == manifest["inputs"]
                 and compiled["outputs"] == manifest["outputs"]
                 and compiled["declared_board_source_pins"] == compiled["declared_board_source_hashes_after"]
                 == manifest["source_pins"], "Compile ONNX/IO/original board source chain differs")
    checks.check([x["name"] for x in compiled["operations"]] == ["config", "load_onnx", "build", "export_rknn"]
                 and all(x["code"] == 0 and math.isfinite(x["seconds"]) and x["seconds"] >= 0
                         for x in compiled["operations"]), "Compilation did not complete successfully")
    counts = compiled["ONNX_operator_counts"]
    checks.check(counts == manifest["ONNX_operator_counts"] and counts.get("Softmax") == 2
                 and counts.get("ReduceMax", 0) == 0 and counts.get("Gather", 0) == counts.get("Where", 0) == 0,
                 "Compiled v34 expression identity differs")
    for folder, item in ((export_dir, manifest["model"]), (compile_dir, compiled["output"])):
        checks.check(Path(item["file"]).name == item["file"], "Model basename required")
        path = folder / item["file"]
        checks.check(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"]
                     and file_sha(path) == item["sha256"], "Model artifact SHA/bytes differ")
    local_sources = {}
    locations = {}
    for name, expected in SOURCES.items():
        folder = source_dir if name in ("source_frame_update_softmax.py", "export-v34-source-frame-softmax-board.py") else v33_source_dir
        locations[name] = folder / name
        checks.check(file_sha(locations[name]) == expected, "Frozen measured expression source differs: " + name)
        helpers.digest_by_basename(manifest["source_pins"], name, expected, checks)
        local_sources[name] = expected
    for name, expected in PINS.items():
        # The frozen runner is commonly stored with the SDK helpers, but older
        # packages place it in the v33 source directory. Either location is SHA-pinned.
        path = sdk_source_dir / name
        if not path.is_file():
            path = v33_source_dir / name
        locations[name] = path
        checks.check(file_sha(path) == expected, "Frozen reused compiler/runtime source differs: " + name)
        local_sources[name] = expected
    references = []
    checks.check(len(manifest["cases"]) == 2, "Two actual saved states required")
    for case in manifest["cases"]:
        references.append({key: helpers.load_artifact(export_dir, case[key + "_artifact"], checks)
                           for key in ("reference", "ort")})
        helpers.load_artifact(export_dir, case["inputs_artifact"], checks)
    normal, nvalues, nrecord = audit_run(normal_dir, manifest, compiled, references, checks, False,
                                        helpers, manifest_sha, compile_sha)
    profile, pvalues, precord = audit_run(profile_dir, manifest, compiled, references, checks, True,
                                         helpers, manifest_sha, compile_sha)
    checks.check(nrecord["session"]["input_attributes"] == precord["session"]["input_attributes"]
                 and nrecord["session"]["output_attributes"] == precord["session"]["output_attributes"],
                 "Normal/profile SDK attrs differ")
    for case in (0, 1):
        for name in OUTPUTS:
            checks.check(nvalues[case][name].tobytes() == pvalues[case][name].tobytes(),
                         "Normal/profile representative output bytes differ")
    for name, path in locations.items():
        checks.check(file_sha(path) == local_sources[name], "Audit source changed during saved review")
    checks.check(file_sha(helper_path) == HELPER_SHA and file_sha(export_helper_path) == EXPORT_AUDITOR_SHA
                 and file_sha(manifest_path) == manifest_sha and file_sha(compile_path) == compile_sha,
                 "Audit metadata/helper source changed")
    return dict(schema="v34_saved_direct_softmax_RKNN_source_frame_independent_audit",
                experiment_version=34, expression_variant="v34_direct_torch_softmax",
                reused_IO_contract_version=33, status="passed_audit_of_FP32_rejected_mixed_execution",
                checks_passed=checks.passed, failed_checks=0, auditor_sha256=file_sha(__file__),
                reused_auditor_sha256=HELPER_SHA, export_scope_auditor_sha256=EXPORT_AUDITOR_SHA,
                source_hashes=local_sources, external_board_source_pins=EXTERNAL_PINS,
                export_manifest_sha256=manifest_sha, compile_receipt_sha256=compile_sha,
                ONNX=manifest["model"], RKNN=compiled["output"], normal=normal, profile=profile,
                normal_profile_saved_outputs_bitwise_equal=True, candidate_accepted=False,
                strict_thresholds_relaxed=False, entire_computation_on_NPU=False,
                actual_SDK_run_calls=8, independently_saved_actual_output_calls=4,
                other_normal_repeats_verified_by_recorded_raw_hashes=4,
                controlled_parent_candidate_AB_comparison=False,
                numerical_gate_passed=normal["strict_FP32_gate_passed"] and profile["strict_FP32_gate_passed"],
                scope=dict(source_block=[1, 10, 96], full_graph_edges=23520, full_graph_execution=False,
                           model_inference_on_audit_host=False, ORT_execution_on_audit_host=False,
                           BA=False, fresh_observation=False, trajectory=False, realtime=False),
                limitations=["Only two frozen D10/M96 source blocks, not full E23520 Update or a VO run.",
                             "The full historical 151 MB input is absent; source selection is bound to frozen board code and metadata, not recomputed here.",
                             "Four ordinary repeats have raw metadata, not individually saved arrays. Both first ordinary and both profile outputs are independently read.",
                             "Copy-before-release is supported by pinned code/counters; saved files cannot reveal runtime pointers.",
                             "DMA is queried while the context is open. Destroy is reported, but post-destroy DMA zero is not measured.",
                             "Profile operator microseconds are instrumented, not ordinary end-to-end timing.",
                             "No same-process controlled parent/candidate AB speedup was measured; numerical rejection remains rejection."])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("normal-dir", "profile-dir", "manifest", "compile-dir", "source-dir", "v33-source-dir", "sdk-source-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--compile-sha256", required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("New audit output required")
    try:
        result = audit(args.normal_dir, args.profile_dir, args.manifest, args.manifest_sha256,
                       args.compile_dir, args.compile_sha256, args.source_dir, args.v33_source_dir, args.sdk_source_dir)
    except Exception as e:
        result = dict(status="failed_saved_evidence_audit", error=repr(e), candidate_accepted=False,
                      auditor_sha256=file_sha(__file__), model_inference_on_audit_host=False)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks_passed", "error", "candidate_accepted")}))
    return 0 if result["status"].startswith("passed_audit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
