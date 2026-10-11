#!/usr/bin/env python3
"""NumPy/JSON-only saved B8 transport audit; no model, ORT, SDK or native calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DRIVER_SHA = "3cbf72a1a4ad7dcb683b12b50dfcc98737b0e1423fee54aede0c8138996bc431"
PACKER_SHA = "34b94c0ba8a6e735baec21fb7a8602c09f8c8b8b56f521b0322d05a8973a0637"
MANIFEST_SHA = "0e182dcb42825525fe1e7e6b84ffd5af50c3dc02a33f183087bff3963a653331"
ORDER = ("fp32", "fp16", "fp16", "fp32", "fp32", "fp16")


class Checks:
    def __init__(self):
        self.count = 0

    def check(self, condition, message):
        self.count += 1
        if not condition:
            raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def artifact(directory, declaration, checks):
    name = declaration["file"]
    checks.check(Path(name).name == name, "Artifact path must be a basename")
    path = directory / name
    checks.check(sha(path) == declaration["sha256"] and path.stat().st_size == declaration["bytes"], "Artifact file SHA/size changed")
    with np.load(path, allow_pickle=False) as archive:
        checks.check(set(archive.files) == set(declaration["arrays"]), "Artifact keys changed")
        values = {k: archive[k].copy(order="C") for k in archive.files}
    for name, value in values.items():
        checks.check(metadata(value) == declaration["arrays"][name] and np.isfinite(value).all(), "Array raw SHA/shape/dtype/finite changed")
    return values


def permutation(ii, jj, kk, sources=8, degree=8, patches=48):
    edges = sources * degree * patches
    if any(v.dtype != np.int64 or v.shape != (edges,) for v in (ii, jj, kk)):
        raise ValueError("Ordered INT64 complete graph required")
    if not np.array_equal(ii, kk // patches):
        raise ValueError("Source patch ownership differs")
    order = np.lexsort((kk % patches, jj, ii))
    expected_i = np.repeat(np.arange(sources), degree * patches)
    expected_j = np.tile(np.repeat(np.arange(degree), patches), sources)
    expected_k = np.tile(np.arange(patches), sources * degree)
    if not (np.array_equal(ii[order], expected_i) and np.array_equal(jj[order], expected_j)
            and np.array_equal(kk[order] % patches, expected_k)):
        raise ValueError("Missing/duplicate edge or incomplete Cartesian source group")
    return order


def compare(actual, reference):
    if actual.dtype != reference.dtype or actual.shape != reference.shape or actual.dtype != np.float32:
        raise ValueError("Equal FP32 shape required")
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite arrays required")
    error = a - b
    failures = int(np.count_nonzero(np.abs(error) > 1e-5 + 1e-4 * np.abs(b)))
    norm = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, gate_passed=failures == 0, failed_elements=failures,
                relative_L2=float(np.linalg.norm(error.ravel()) / norm) if norm else None,
                max_abs=float(np.abs(error).max()), nonfinite_elements=0, rtol=1e-4, atol=1e-5)


def check_comparison(actual, recorded, checks):
    for key, value in actual.items():
        other = recorded[key]
        same = math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-14) if type(value) is float else value == other
        checks.check(same, "Recorded comparison differs: " + key)


def warm_summary(calls):
    return {method: {"warm_forward_seconds": [r["timing"]["forward_seconds"] for r in calls
                        if r["input_transport"] == method and not r["first_for_transport"]],
                    "warm_mean_forward_seconds": float(np.mean([r["timing"]["forward_seconds"] for r in calls
                        if r["input_transport"] == method and not r["first_for_transport"]]))}
            for method in ("fp32", "fp16")}


def audit(args):
    c = Checks()
    run_path = args.run_dir / "run.json"
    run = json.loads(run_path.read_text())
    c.check(run["schema"] == "v38_B8_FP32_FP16_input_transport_AB_diagnostic" and
            run["status"] == "completed_initial_group_RKNN_diagnostic_not_VO", "Unexpected actual run")
    c.check(run["script_sha256"] == sha(args.source_dir / "run-v38-B8-input-AB-board.py") == DRIVER_SHA, "Driver source differs")
    c.check(sha(args.source_dir / "fp16_input_transport.py") == PACKER_SHA, "Transport source differs")
    c.check(run["source_pins"] == run["source_hashes_after"] and run["sources_and_inputs_unchanged"] is True, "Board source before/after differs")
    c.check(run["candidate_accepted"] is False and run["VO_state_committed"] is False and run["production_changed"] is False
            and run["inherited_CPU_ORT_FP32_gate_passed"] is False and run["actual_original_FP32_gate_passed"] is False, "Reject labels may not be promoted")
    c.check(run["actual_Torch_OR_ORT_calls"] == 0 and run["host_inference_executed"] is False and run["collect_perf_detail"] is False, "Scope differs")
    c.check(run["cleanup_errors"] == [] and run["affinity_restored"] is True and run["input_packer_restored"] is True, "Cleanup incomplete")
    manifest_path = args.export_dir / "manifest.json"
    c.check(sha(manifest_path) == run["export_manifest_sha256"] == MANIFEST_SHA, "Export manifest differs")
    manifest = json.loads(manifest_path.read_text())
    case = manifest["cases"][0]
    c.check(case["observation_id"] == run["case"]["observation_id"] == 13 and case["edges"] == 3072
            and case["degree_plan"] == run["case"]["degree_plan"]
            and case["snapshot_identity"] == run["case"]["snapshot_identity"], "Complete graph identity differs")
    original = artifact(args.capture_dir, case["original_inputs_artifact"], c)
    order = permutation(original["ii"], original["jj"], original["kk"])
    golds = {"original_Torch": artifact(args.capture_dir, case["original_gold_artifact"], c),
             "grouped_Torch": artifact(args.export_dir, case["grouped_Torch_artifact"], c),
             "grouped_ORT": artifact(args.export_dir, case["grouped_ORT_artifact"], c)}
    feed = artifact(args.export_dir, case["groups"][0]["inputs_artifact"], c)
    for key, channels in (("net", 384), ("inp", 384), ("corr", 882)):
        expected = original[key][0, order].reshape(8, 8, 48, channels).transpose(0, 3, 1, 2)
        c.check(expected.tobytes(order="C") == feed[key].tobytes(order="C"), "Saved CF input packing differs")
    compile_path = args.compile_dir / "compile.json"
    c.check(sha(compile_path) == run["compile_sha256"], "Compile receipt differs")
    compiled = json.loads(compile_path.read_text())
    c.check(compiled["output"]["sha256"] == run["model_sha256"], "Compiled artifact identity differs")
    model_path = args.compile_dir / compiled["output"]["file"]
    c.check(sha(model_path) == run["model_sha256"] and model_path.stat().st_size == compiled["output"]["bytes"], "RKNN artifact file changed")
    calls = run["calls"]
    c.check(len(calls) == 6 and tuple(r["input_transport"] for r in calls) == ORDER, "Predeclared AB/BA/AB order differs")
    first = {}
    first_full = {}
    errors = None
    for index, row in enumerate(calls):
        c.check(row["call_index"] == index and row["timing"]["frame_id"] == 0 and row["pair_index"] == index // 2
                and row["first_for_transport"] is (index < 2), "Call identity/warm grouping differs")
        c.check(row["execution_status"] == "completed" and row["outputs_released"] is True and row["outputs_finite"] is True
                and row["weight_domain_valid"] is True and row["all_outputs_zero"] is False and row["native_log"]["RKNN_error_detected"] is False,
                "SDK/numerical health differs")
        c.check(row["strict_FP32_gate_passed"] is False and row["candidate_accepted"] is False, "Original precision rejection erased")
        actual = artifact(args.run_dir, row["CF_artifact"], c)
        full = artifact(args.run_dir, row["full_order_artifact"], c)
        for sdk, canonical, channels in (("net_out", "net", 384), ("delta", "delta", 2), ("weight", "weight", 2)):
            value = actual[sdk]
            c.check(value.shape == (8, channels, 8, 48) and value.dtype == np.float32
                    and full[canonical].shape == (1, 3072, channels), "Actual CF/canonical contract differs")
            restored = np.empty((1, 3072, channels), dtype=np.float32)
            restored[0, order] = value.transpose(0, 2, 3, 1).reshape(3072, channels)
            c.check(restored.tobytes() == full[canonical].tobytes(), "Original edge-order scatter differs")
            if index == 0:
                first[sdk], first_full[canonical] = value.copy(), full[canonical].copy()
            c.check(value.tobytes() == first[sdk].tobytes() and full[canonical].tobytes() == first_full[canonical].tobytes(), "AB/repeat output bytes differ")
            equality = compare(value, first[sdk])
            check_comparison(equality, row["versus_first_FP32_transport"][sdk], c)
            c.check(row["versus_first_FP32_transport"][sdk]["bitwise_equal"] is True, "Equality label differs")
            comparison = compare(full[canonical], golds["original_Torch"][canonical])
            check_comparison(comparison, row["full_versus_original"][canonical], c)
            for kind, reference in golds.items():
                packed = reference[canonical][0, order].reshape(8, 8, 48, channels).transpose(0, 3, 1, 2)
                computed = compare(value, packed)
                check_comparison(computed, row["versus_saved_references"][kind][sdk], c)
                if kind == "original_Torch":
                    check_comparison(computed, row["versus_reference"][sdk], c)
        c.check(np.all((actual["weight"] >= 0) & (actual["weight"] <= 1)) and any(np.any(v != 0) for v in actual.values()), "Actual domain differs")
        if index == 0:
            errors = {key: compare(full[key], golds["original_Torch"][key]) for key in full}
        input_rows = row["timing"]["input_records"]
        c.check([v["name"] for v in input_rows] == ["net", "inp", "corr"], "Input descriptor names differ")
        for j, (rec, channels) in enumerate(zip(input_rows, (384, 384, 882))):
            expected_bytes = 8 * 8 * 48 * channels * (4 if row["input_transport"] == "fp32" else 2)
            c.check(rec["index"] == j and rec["source_dtype"] == "float32" and rec["transport_dtype"] == ("float32" if row["input_transport"] == "fp32" else "float16")
                    and rec["bytes"] == expected_bytes and rec["logical_shape"] == [8, channels, 8, 48]
                    and rec["buffer_shape"] == [8, 8, 48, channels] and rec["request_format"] == rec["native_attribute_format"] == 1
                    and rec["pass_through"] == 0 and rec["owned"] is True and rec["explicit_NHWC_pack"] is True, "Actual transport descriptor differs")
        for key in ("pack_seconds", "input_set_seconds", "run_api_seconds", "outputs_get_seconds", "owned_copy_seconds", "outputs_release_seconds", "forward_seconds"):
            c.check(math.isfinite(row["timing"][key]) and row["timing"][key] >= 0, "Invalid timing")
        c.check(row["timing"]["synchronous"] is True and row["total_execute_seconds"] >= row["timing"]["forward_seconds"], "Timing/sync scope inconsistent")
    summary = warm_summary(calls)
    c.check(summary == run["timing_summary"] and all(len(v["warm_forward_seconds"]) == 2 for v in summary.values()), "Warm timing summary differs")
    session = run["session"]
    c.check(session["synthetic"] is False and session["npu_used"] is True and session["closed"] is True and session["context_destroyed"] is True
            and session["cleanup_errors"] == [] and session["current_DMA_bytes"] == session["current_model_host_bytes"] == 0, "Session cleanup/actual-run metadata differs")
    c.check(session["sdk_run_api_calls"] == session["completed_calls"] == session["SDK_outputs_release_attempts"] == session["SDK_outputs_releases"] == 6,
            "Actual SDK run/release counters differ")
    c.check(session["source_pins"] == session["source_hashes_after"] and session["sources_and_inputs_unchanged"] is True
            and session["core_mask"] == 7 and session["init_flags"] == 0 and session["collect_perf_detail"] is False, "Native source/core flags differ")
    return dict(schema="v38_B8_saved_array_input_transport_independent_review", status="passed_evidence_checks_precision_rejected",
                checks_passed=c.count, checks_failed=0, auditor_sha256=sha(__file__),
                run_sha256=sha(run_path), driver_sha256=DRIVER_SHA, packer_sha256=PACKER_SHA,
                manifest_sha256=MANIFEST_SHA, compile_sha256=run["compile_sha256"], model_sha256=run["model_sha256"],
                actual_saved_artifacts_checked=12, actual_saved_arrays_checked=36,
                all_six_CF_and_canonical_outputs_byte_equal=True, canonical_edge_order_independently_reconstructed=True,
                original_full_inputs_gold_and_grouped_references_file_and_raw_hashes_checked=True,
                strict_original_CPU_gate_passed=False, inherited_CPU_ORT_gate_passed=False,
                original_CPU_errors=errors, input_transport_warm_timing=summary,
                FP16_warm_over_FP32_ratio=summary["fp16"]["warm_mean_forward_seconds"] / summary["fp32"]["warm_mean_forward_seconds"],
                sdk=dict(actual_run_calls=6, output_releases=6, context_destroyed=True, final_DMA_bytes=0,
                         peak_DMA_bytes=session["memory"]["total_dma_allocated_size"], core_mask=7,
                         SDK_frame_id_zero_for_each_synchronous_call=True,
                         note="Logical call_index 0..5 is separate from the frozen SDK helper's synchronous frame_id=0."),
                board_source_pin_map_entries=len(run["source_pins"]), board_source_maps_equal=True,
                source_verification_scope="Local driver/packer and supplied artifacts rehashed; other board-only source pins checked for receipt before/after consistency, not rehashed on host.",
                peak_rss_mib=run["peak_rss_mib"], cleanup_errors=[], candidate_accepted=False, VO_state_committed=False,
                scope="One frozen complete initial E3072/B8D8M48 graph; two warm calls per transport in predeclared same-context AB/BA/AB. No per-operator profile, feedback, new frames, BA, trajectory or realtime validation.",
                inference_executed_by_auditor=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "export-dir", "capture-dir", "compile-dir", "source-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite an audit receipt")
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks_passed"], "receipt_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
