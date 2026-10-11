#!/usr/bin/env python3
"""Saved-array and static ONNX audit; never runs Torch, ORT or a model.

The full historical graph input is not required or read. Its input-to-block
selection is bound to frozen board code and metadata, not independently
recomputed here. Complete saved topology/permutation and all block arrays are
independently checked. Keep the adjacent DPVO MIT attribution and LICENSE.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


OUTPUTS = ("net_out", "delta", "weight")
INPUTS = ("net", "inp", "corr")
PINS = {
    "export-v33-source-frame-board.py": "1bb1df7dc4e61e6f3d60fcd1170b267c478fd9600de1f0481ec1a73a42fa1b73",
    "source_frame_update.py": "cf1adb2224750cd0659139689f4b8a5a8107f2e733e8d1c6f86ea009c86a1bb2",
    "source_frame_plan.py": "04e240c2c35bd1359dbe5649fb038490a0a7ff99654d8d112ad7c83acc87ac2e",
    "dpvo_probe_common.py": "ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9",
    "dpvo_cpu_ops.py": "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32",
    "manifest.json": "ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375",
    "LICENSE": "c5b0105dcef65f5423bec0aafe796364a086375e23746cf75402aba0f897489a",
    "dpvo/extractor.py": "f5cbcbe14c42e44bcec6e827a3429ebe93979a439ba9b9ef70d4f4baa3454a2b",
    "dpvo/blocks.py": "9c9e2848da524829374e56bb1b14a891e9e2c19bc683af8e958e8982ba33d661",
    "dpvo/net.py": "1aee33d54f0b1cc1f3f842b60790ff1299ef9c1e20bca6467ce7114dca802acc",
    "dpvo/dpvo.py": "7efcb45689c2da8d18e2c69774e221799c9f8472e99531955f55dcb145626fd7",
}
HISTORICAL_SHA = "25ed1a743bbda78b8fa8eeefb0ed4827ca1c3d2fdf978c0c1e2a294e97ef3520"
CHECKPOINT_SHA = "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"


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
                 "Artifact must be a basename")
    path = Path(folder) / name
    checks.check(path.is_file() and not path.is_symlink(), "Regular artifact required: " + name)
    checks.check(path.stat().st_size == item["bytes"], "Artifact bytes differ: " + name)
    checks.check(file_sha(path) == item["sha256"], "Artifact SHA differs: " + name)
    with np.load(path, allow_pickle=False) as archive:
        checks.check(len(set(archive.files)) == len(archive.files), "Duplicate NPZ names")
        values = {key: archive[key].copy() for key in archive.files}
    checks.check(set(values) == set(item["arrays"]), "NPZ keys differ: " + name)
    for key, value in values.items():
        checks.check(metadata(value) == item["arrays"][key], "Raw array metadata differs: " + name + "/" + key)
    return values


def reconstruct_topology(values, checks, patches=96, capacity=64):
    """Independent complete-Cartesian proof; does not import the v32 planner."""
    ii, jj, kk = (values[k] for k in ("ii", "jj", "kk"))
    for name, value in zip(("ii", "jj", "kk"), (ii, jj, kk)):
        checks.check(value.dtype == np.int64 and value.ndim == 1, "Int64 graph vector required: " + name)
    checks.check(ii.size > 0 and ii.shape == jj.shape == kk.shape, "Graph vector lengths differ")
    checks.check(np.all((0 <= ii) & (ii < capacity)) and np.all((0 <= jj) & (jj < capacity)), "Frame index domain differs")
    checks.check(np.all((0 <= kk) & (kk < capacity * patches)) and np.array_equal(ii, kk // patches), "Source patch ownership differs")
    plans = []
    for source in np.unique(ii):
        original = np.flatnonzero(ii == source)
        targets = sorted(set(int(v) for v in jj[original]))
        checks.check(len(original) == len(targets) * patches, "Incomplete source Cartesian edge count")
        rows = []
        # Explicit member checks and sorting, independent of production lexsort.
        for target in targets:
            members = original[jj[original] == target]
            checks.check(len(members) == patches, "Incomplete source-target count")
            local_patch = kk[members] - int(source) * patches
            checks.check(np.array_equal(np.sort(local_patch), np.arange(patches)), "Duplicate or missing source-target patch")
            rows.extend(int(members[int(np.flatnonzero(local_patch == p)[0])]) for p in range(patches))
        canonical = np.asarray(rows, dtype=np.int64)
        checks.check(np.array_equal(np.sort(canonical), original), "Canonical permutation is not a bijection")
        metadata_plan = dict(source_slot=int(source), sorted_target_slots=targets,
                             degree=len(targets), edges=len(original), canonical_shape=[len(targets), patches],
                             original_edge_order_preserved_on_restore=True,
                             all_patch_target_members_present=True, duplicate_edges=False)
        plans.append((metadata_plan, canonical))
    return min(plans, key=lambda item: (item[0]["degree"], item[0]["source_slot"])), plans


def residuals(actual, reference):
    if actual.dtype != np.float32 or reference.dtype != np.float32 or actual.shape != reference.shape:
        raise ValueError("Matching FP32 shapes required")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Finite arrays required")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    tolerance = 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))
    norm = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    bad = int(np.count_nonzero(absolute > tolerance))
    return dict(rtol=1e-4, atol=1e-5, failed_elements=bad, gate_passed=bad == 0,
                max_abs=float(absolute.max()),
                relative_l2=float(np.linalg.norm(difference.ravel())) / norm if norm else None,
                bitwise_equal=actual.tobytes(order="C") == reference.tobytes(order="C"),
                max_normalized_tolerance=float((absolute / tolerance).max()))


def check_recorded_residual(actual, recorded, checks):
    checks.check(set(recorded) == set(actual) - {"max_normalized_tolerance"}, "Recorded comparison keys differ")
    for key, value in recorded.items():
        if key == "relative_l2" and value is not None:
            checks.check(math.isclose(value, actual[key], rel_tol=1e-10, abs_tol=1e-14), "Recorded relative L2 differs")
        else:
            checks.check(value == actual[key], "Recorded comparison differs: " + key)
    checks.check(actual["gate_passed"], "Strict FP32 comparison failed")


def verify_initializers(model, originals, coverage, checks, onnx):
    checks.check(len(model.graph.initializer) == len({v.name for v in model.graph.initializer}), "Duplicate initializer name")
    checks.check(coverage["original_tensor_count"] == len(originals), "Parameter count differs")
    recorded = coverage["mapping"]
    checks.check(coverage["initializer_count"] == len(model.graph.initializer) == len(recorded), "Initializer count differs")
    covered, rows = set(), []
    for initializer, item in zip(model.graph.initializer, recorded):
        checks.check(initializer.data_location != onnx.TensorProto.EXTERNAL and not initializer.external_data, "External initializer data")
        array = onnx.numpy_helper.to_array(initializer)
        checks.check(array.dtype == np.float32 and np.isfinite(array).all(), "Finite FP32 initializer required")
        checks.check(item["initializer"] == initializer.name and item["array"] == metadata(array), "Initializer raw metadata differs")
        # Reverse any recorded transpose, then hash the original orientation.
        direct = metadata(array)
        transposed = metadata(array.T) if array.ndim == 2 else None
        matches = []
        for name, original in originals.items():
            if original == direct:
                matches.append(dict(parameter=name, transform="identity"))
            if transposed is not None and original == transposed:
                matches.append(dict(parameter=name, transform="transpose"))
        checks.check(bool(matches), "Initializer is not an original parameter or exact transpose")
        checks.check(matches == item["original_matches"], "Original initializer mapping differs")
        covered.update(m["parameter"] for m in matches)
        rows.append(dict(initializer=initializer.name, original_matches=matches))
    checks.check(covered == set(originals) and coverage["all_original_parameters_covered"] is True, "Original parameter coverage incomplete")
    return dict(parameters=len(originals), initializers=len(rows), all_covered=True,
                mapping_basis="Original-oriented C-order raw SHA, shape/dtype/bytes; 2D transpose independently reversed",
                original_parameter_payload_not_separately_loaded=True)


def graph_contract(value):
    tensor = value.type.tensor_type
    if tensor.elem_type != 1 or not all(v.HasField("dim_value") and v.dim_value > 0 for v in tensor.shape.dim):
        raise ValueError("Fixed FP32 ONNX I/O required")
    return dict(name=value.name, dtype="float32", shape=[v.dim_value for v in tensor.shape.dim])


def audit_export(run_dir, *, manifest_sha256, source_dir, plan_source, v26_source_dir, historical_run):
    import onnx  # Static protobuf/checker only. No runtime/provider is imported.
    run_dir, source_dir, v26_source_dir = map(Path, (run_dir, source_dir, v26_source_dir))
    checks = Checks()
    manifest_path = run_dir / "manifest.json"
    checks.check(file_sha(manifest_path) == manifest_sha256, "Manifest SHA differs")
    report = json.loads(manifest_path.read_text())
    checks.check(report["status"] == "completed_source_frame_export_diagnostic_not_adopted", "Unexpected export status")
    checks.check(report["cleanup_errors"] == [], "Cleanup errors")
    for field in ("strict_FP32_gate_passed", "outputs_finite", "weight_domain_valid", "original_parameters_unchanged", "sources_and_inputs_unchanged", "affinity_restored", "ORT_session_reference_released"):
        checks.check(report[field] is True, "Board success condition differs: " + field)
    for field in ("NPU_used", "BA_executed", "valid_pose_output", "candidate_accepted", "realtime_validated", "trajectory_accuracy_validated", "host_model_inference", "FP32_thresholds_relaxed", "output_clipping"):
        checks.check(report[field] is False, "Scope flag differs: " + field)
    checks.check(report["actual_board_execution"] is True and report["actual_inference_on_RK3588S"] is True, "Board execution record required")
    checks.check(report["terminal_sigmoid"] == "stable", "Unexpected sigmoid method")
    checks.check(report["source_pins"] == report["source_hashes_after"], "Board source before/after hashes differ")
    sources = {name: source_dir / name for name in ("source_frame_update.py", "export-v33-source-frame-board.py")}
    sources["source_frame_plan.py"] = Path(plan_source)
    sources.update({name: v26_source_dir / name for name in ("dpvo_probe_common.py", "dpvo_cpu_ops.py", "manifest.json")})
    sources.update({name: v26_source_dir / "official" / name for name in PINS if name.startswith("dpvo/") or name == "LICENSE"})
    rehashed = {}
    for name, path in sources.items():
        digest = file_sha(path)
        checks.check(digest == PINS[name], "Frozen host source differs: " + name)
        matches = [v for p, v in report["source_pins"].items() if p.endswith("/" + name)]
        checks.check(matches == [digest], "Board source binding differs: " + name)
        rehashed[name] = digest
    checks.check(report["script_sha256"] == PINS["export-v33-source-frame-board.py"], "Runtime driver binding differs")
    checkpoints = [v for p, v in report["source_pins"].items() if p.endswith("/dpvo.pth")]
    checks.check(checkpoints == [CHECKPOINT_SHA], "Original checkpoint pin differs")
    checks.check(file_sha(historical_run) == HISTORICAL_SHA, "Frozen historical metadata differs")
    historical = json.loads(Path(historical_run).read_text())
    checks.check(historical["status"] == "completed_numerical_rejected" and not historical["cleanup_errors"], "Unexpected historical status")
    checks.check(report["original_weights_before"] == report["original_weights_after"] == historical["original_weights_before"], "Original model raw-parameter metadata differs")
    checks.check(len(report["original_weights_before"]) == 94, "Original model state count differs")
    originals = report["original_Update_parameters"]
    checks.check(len(originals) == 50, "Original Update must contain fifty tensors")
    for key, value in originals.items():
        checks.check(value == report["original_weights_before"]["update." + key], "Update parameter not bound to original full model")
    binding = report["original_binding"]
    checks.check(binding["state_tensors"] == 50 and binding["original_parameter_objects_shared"] is True and binding["original_module_objects_shared"] is True and binding["state_dict_names_preserved"] is True, "Original module binding record differs")
    checks.check(binding["terminal_sigmoid"] == "stable" and binding["stable_formula"] == "exp(min(x,0))/(1+exp(-abs(x)))", "Stable final sigmoid binding differs")
    checks.check(len(report["cases"]) == len(historical["states"]) == 2, "Both recurrent inputs required")
    cases, comparisons, artifacts = [], [], []
    for index, case in enumerate(report["cases"]):
        previous = historical["states"][index]
        checks.check(case["index"] == index and case["role"] == previous["role"], "Case identity differs")
        checks.check(case["original_input_artifact_sha256"] == previous["inputs_artifact"]["sha256"], "Historical full input binding differs")
        loaded = {}
        for role in ("inputs", "graph", "reference", "wrapper", "ort"):
            item = case[role + "_artifact"]
            loaded[role] = load_artifact(run_dir, item, checks)
            artifacts.append(dict(role=role, case=index, file=item["file"], sha256=item["sha256"], bytes=item["bytes"]))
        graph = loaded["graph"]
        checks.check(set(graph) == {"ii", "jj", "kk", "canonical_edges"}, "Graph artifact keys differ")
        checks.check(case["total_graph_edges"] == previous["edges"] == len(graph["ii"]) == 23520, "Expected full E23520 topology")
        for name in ("ii", "jj", "kk"):
            checks.check(metadata(graph[name]) == previous["inputs_artifact"]["arrays"][name], "Saved graph differs from frozen full input metadata")
        (selected, canonical), plans = reconstruct_topology(graph, checks)
        checks.check(selected == case["source_plan"], "Selected complete source metadata differs")
        checks.check(len(plans) == case["all_source_blocks"] == 17, "Full source partition count differs")
        checks.check(selected["degree"] == 10 and selected["source_slot"] == 0 and selected["canonical_shape"] == [10, 96], "Predeclared minimum source differs")
        checks.check(graph["canonical_edges"].dtype == np.int64 and np.array_equal(canonical, graph["canonical_edges"]), "Saved canonical permutation differs")
        feed = loaded["inputs"]
        checks.check(tuple(feed) == INPUTS, "Ordered three input arrays required")
        for name, channels in zip(INPUTS, (384, 384, 882)):
            value = feed[name]
            checks.check(value.dtype == np.float32 and value.shape == (1, 10, 96, channels) and np.isfinite(value).all(), "Finite canonical FP32 input required: " + name)
        for role in ("reference", "wrapper", "ort"):
            values = loaded[role]
            checks.check(tuple(values) == OUTPUTS, "Ordered three output arrays required")
            for name, channels in zip(OUTPUTS, (384, 2, 2)):
                value = values[name]
                checks.check(value.dtype == np.float32 and value.shape == (1, 10, 96, channels) and np.isfinite(value).all(), "Finite canonical output required")
            weight = values["weight"]
            checks.check(np.all((weight >= 0) & (weight <= 1)), "Output confidence outside domain")
            if role in ("wrapper", "ort"):
                domain = dict(outputs_finite=True, weight_domain_valid=True, weight_min=float(weight.min()), weight_max=float(weight.max()))
                checks.check(domain == case[role + "_domain"], "Recorded output domain differs")
        local_comparisons = {}
        for group, candidate, reference in (("wrapper_vs_original_full", "wrapper", "reference"),
                                             ("ORT_vs_original_full", "ort", "reference"),
                                             ("ORT_vs_wrapper", "ort", "wrapper")):
            values = {name: residuals(loaded[candidate][name], loaded[reference][name]) for name in OUTPUTS}
            for name in OUTPUTS:
                check_recorded_residual(values[name], case[group][name], checks)
                comparisons.append(values[name])
            local_comparisons[group] = values
        for name, recorded in case["fresh_full_reference_vs_historical"].items():
            checks.check(name in OUTPUTS and recorded == dict(rtol=1e-4, atol=1e-5, failed_elements=0, gate_passed=True, max_abs=0.0, relative_l2=0.0, bitwise_equal=True), "Board historical-reference check differs")
        cases.append(dict(index=index, role=case["role"], source_plan=selected,
                          complete_source_degrees=[p[0]["degree"] for p in plans], comparisons=local_comparisons,
                          timings_from_board_record={name: case[name] for name in ("original_full_Update_seconds", "Torch_block_seconds", "ORT_block_seconds")},
                          full_input_selection_independently_recomputed=False,
                          full_historical_output_comparison_independently_recomputed=False))
    model_item = report["model"]
    checks.check(Path(model_item["file"]).name == model_item["file"], "Model basename required")
    model_path = run_dir / model_item["file"]
    checks.check(model_path.is_file() and not model_path.is_symlink(), "Regular ONNX model required")
    checks.check(model_path.stat().st_size == model_item["bytes"] and file_sha(model_path) == model_item["sha256"], "ONNX file bytes/SHA differ")
    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    checks.check(not model.functions and not any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for n in model.graph.node for a in n.attribute), "Unexpected nested model graph")
    inputs = [graph_contract(v) for v in model.graph.input]
    outputs = [graph_contract(v) for v in model.graph.output]
    expected_inputs = [dict(name=k, dtype="float32", shape=[1, 10, 96, c]) for k, c in zip(INPUTS, (384, 384, 882))]
    expected_outputs = [dict(name=k, dtype="float32", shape=[1, 10, 96, c]) for k, c in zip(OUTPUTS, (384, 2, 2))]
    checks.check(inputs == report["inputs"] == expected_inputs and outputs == report["outputs"] == expected_outputs, "Static ONNX three-I/O contract differs")
    operations = sorted({node.op_type for node in model.graph.node})
    checks.check(operations == report["onnx_operations"] and len(model.graph.node) == report["onnx_nodes"], "Recorded node operations/count differ")
    checks.check(not {"Gather", "GatherElements", "GatherND", "Where"} & set(operations) and report["group_Gather_Where_absent"] is True, "Generic Gather/Where grouping present")
    initializers = verify_initializers(model, originals, report["parameter_coverage"], checks, onnx)
    checks.check(report["ORT_providers"] == ["CPUExecutionProvider"] and report["ORT_graph_optimization"] == "ORT_DISABLE_ALL", "Board ORT provider/options differ")
    for case in cases:
        checks.check(all(math.isfinite(v) and v > 0 for v in case["timings_from_board_record"].values()), "Invalid recorded component timing")
    checks.check(math.isfinite(report["actual_driver_seconds"]) and report["actual_driver_seconds"] > 0 and math.isfinite(report["peak_rss_mib"]) and report["peak_rss_mib"] > 0, "Invalid recorded driver timing/RSS")
    return dict(schema="v33_source_frame_saved_array_static_ONNX_independent_audit", status="passed_saved_array_static_audit",
                checks_passed=checks.passed, manifest_sha256=manifest_sha256, auditor_sha256=file_sha(__file__),
                source_hashes_recomputed=rehashed, historical_metadata_sha256=HISTORICAL_SHA,
                artifacts_verified=artifacts, saved_array_comparisons=len(comparisons), strict_failed_elements=sum(x["failed_elements"] for x in comparisons),
                strict_rtol=1e-4, strict_atol=1e-5, strict_block_FP32_gate_passed=True,
                cases=cases, ONNX=dict(sha256=model_item["sha256"], bytes=model_item["bytes"], nodes=len(model.graph.node),
                                       operations=operations, inputs=inputs, outputs=outputs, initializers=initializers),
                board_cleanup_record=dict(cleanup_errors=[], source_hashes_unchanged=True, weights_metadata_unchanged=True,
                                          ORT_reference_released=True, affinity_restored=True),
                board_measures=dict(driver_seconds=report["actual_driver_seconds"], peak_rss_mib=report["peak_rss_mib"]),
                scope=dict(host_Torch_ORT_model_or_BA_execution=False, board_execution_by_auditor=False,
                           full_graph_input_selection_recomputed=False, full_graph_historical_output_comparison_recomputed=False,
                           full_original_checkpoint_reloaded=False, NPU_runtime_validated=False, production_adopted=False,
                           trajectory_or_realtime_validated=False),
                limitations=["Host does not have the full historical 151 MB graph input; input selection and full original-vs-historical checks are board-record/code/metadata bindings only.",
                             "Canonical saved block outputs, all full topology and its selected permutation, file/raw hashes and static ONNX parameter mappings were independently recomputed.",
                             "Original parameter payload is not loaded separately; static initializers are bound to recorded original raw hashes, with exact transpose reversal.",
                             "First calls and different full-graph/single-block workloads do not establish a controlled speedup. Component clock includes output copies as frozen board driver states.",
                             "This is a fixed-input component FP32 check, not candidate recurrence, NPU execution, VO or 20 Hz proof."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "source-dir", "plan-source", "v26-source-dir", "historical-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("New output receipt required")
    try:
        result = audit_export(args.run_dir, manifest_sha256=args.manifest_sha256,
                              source_dir=args.source_dir, plan_source=args.plan_source,
                              v26_source_dir=args.v26_source_dir, historical_run=args.historical_run)
        code = 0
    except Exception as error:
        result = dict(status="failed_audit", error=str(error), auditor_sha256=file_sha(__file__))
        code = 1
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "error", "checks_passed", "strict_failed_elements")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
