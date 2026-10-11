#!/usr/bin/env python3
"""Board-only v34 direct-softmax Update comparison/export over saved v33 blocks.

The v33 manifest schema is the stable IO contract, not the experiment version.
No E23520 full graph is re-executed. Saved original CPU, v33 wrapper and v33 ORT
references are explicitly historical. Strict rejection is saved, never adopted.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
PARENT_SHA = "cf1adb2224750cd0659139689f4b8a5a8107f2e733e8d1c6f86ea009c86a1bb2"
UTILITY_SHA = "1bb1df7dc4e61e6f3d60fcd1170b267c478fd9600de1f0481ec1a73a42fa1b73"
REFERENCE_SHA = "3f45c586bcdfdd47cd81724fb4008e8dcac975d86257a1ec58b3fca97767ba13"
CHECKPOINT_SHA = "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"
PINS = {
    "dpvo_probe_common.py": "ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9",
    "dpvo_cpu_ops.py": "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32",
    "manifest.json": "ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375",
}
INPUTS = ("net", "inp", "corr")
OUTPUTS = ("net_out", "delta", "weight")


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def require_board():
    require(platform.system() == "Linux" and platform.machine() == "aarch64"
            and b"rockchip,rk3588" in Path("/proc/device-tree/compatible").read_bytes(),
            "Actual model execution/export is RK3588-only")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v26-source-dir", "checkpoint", "reference-dir", "parent-source", "utility-source",
                 "wrapper-source", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    args = parser.parse_args(argv)
    require_board()  # Before reading real inputs, checkpoint, or importing Torch/ORT.
    out = args.output_dir.resolve()
    require(out.is_relative_to(BASE) and out != BASE, "Isolated experiment output required")
    protected = [v.resolve() for k, v in vars(args).items() if k != "output_dir" and isinstance(v, Path)]
    require(not any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected),
            "Output overlaps input/source")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    affinity = os.sched_getaffinity(0)
    pins, neural, before, session, wrapped, manual = {}, None, None, None, None, None
    report = dict(schema="v33_DPVO_source_frame_Update_board_export", experiment_version=34,
                  expression_variant="v34_direct_torch_softmax", status="failed", script_sha256=sha(__file__),
                  wrapper_sha256=args.wrapper_sha256, parent_wrapper_sha256=PARENT_SHA,
                  parent_export_utility_sha256=UTILITY_SHA, reference_manifest_sha256=REFERENCE_SHA,
                  actual_board_execution=True, actual_inference_on_RK3588S=True, host_model_inference=False,
                  original_VGGT_default_unchanged=True, different_model_from_VGGT=True, ORB_used=False,
                  added_LK_PnP_frontend=False, NPU_used=False, BA_executed=False, valid_pose_output=False,
                  realtime_validated=False, trajectory_accuracy_validated=False, candidate_accepted=False,
                  strict_FP32_gate_passed=False, outputs_finite=False, weight_domain_valid=False,
                  original_parameters_unchanged=False, terminal_sigmoid="stable", cases=[], cleanup_errors=[],
                  full_E23520_reference_reexecuted=False,
                  reference_boundary="Frozen v33 full-graph CPU reference, wrapper and ORT artifacts are loaded; only two D10/M96 source blocks are executed in this run.",
                  source_plan_recomputed=False,
                  scope="Direct group softmax expression only; no new observation, full graph, BA, pose or feedback.")

    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, "Pinned source/input differs: " + str(path))
        pins[str(path)] = expected
        return path

    def load_saved(item):
        require(isinstance(item["file"], str) and Path(item["file"]).name == item["file"]
                and item["file"] not in ("", ".", ".."), "Artifact basename required")
        path = pin(args.reference_dir / item["file"], item["sha256"])
        require(path.stat().st_size == item["bytes"] and not path.is_symlink(), "Artifact regular bytes differ")
        with np.load(path, allow_pickle=False) as saved:
            require(len(saved.files) == len(set(saved.files)), "Duplicate NPZ keys")
            values = {k: saved[k].copy() for k in saved.files}
        require({k: utility.meta(v) for k, v in values.items()} == item["arrays"], "Saved raw metadata differs")
        return values

    def save(name, values):
        path = out / name
        with path.open("xb") as stream:
            np.savez(stream, **values)
        return dict(file=name, bytes=path.stat().st_size, sha256=sha(path),
                    arrays={k: utility.meta(v) for k, v in values.items()})

    try:
        pin(__file__, report["script_sha256"])
        pin(args.wrapper_source, args.wrapper_sha256)
        pin(args.parent_source, PARENT_SHA)
        pin(args.utility_source, UTILITY_SHA)
        pin(args.checkpoint, CHECKPOINT_SHA)
        for name, digest in PINS.items():
            pin(args.v26_source_dir / name, digest)
        old = json.loads(pin(args.reference_dir / "manifest.json", REFERENCE_SHA).read_text())
        require(old["schema"] == report["schema"] and old["status"] == "completed_source_frame_export_diagnostic_not_adopted"
                and old["strict_FP32_gate_passed"] and old["original_parameters_unchanged"]
                and old["sources_and_inputs_unchanged"] and old["source_pins"] == old["source_hashes_after"]
                and old["cleanup_errors"] == [] and len(old["cases"]) == 2,
                "Frozen successful v33 CPU reference/export required")
        require({4, 5, 6, 7} <= affinity, "Four A76 CPUs required")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import numpy as np
        import torch
        import onnx
        import onnxruntime as ort
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        utility = module(args.utility_source, "_v34_frozen_export_utils")
        common = module(args.v26_source_dir / "dpvo_probe_common.py", "_v34_original_loader")
        parent = module(args.parent_source, "_v34_original_expression")
        variant = module(args.wrapper_source, "_v34_direct_expression")
        require(variant.PARENT_SOURCE_SHA256 == PARENT_SHA, "Variant ancestry differs")
        neural, loader = common.load_neural(args.v26_source_dir / "official", args.v26_source_dir / "manifest.json",
                                             args.checkpoint, CHECKPOINT_SHA, args.v26_source_dir / "dpvo_cpu_ops.py",
                                             cpu_ops_sha256=PINS["dpvo_cpu_ops.py"])
        for name, item in loader["source_hashes"].items():
            pin(args.v26_source_dir / "official" / name, item["sha256"])
        before = {k: utility.meta(v.detach().numpy()) for k, v in neural.state_dict().items()}
        require(before == old["original_weights_before"] and len(before) == 94, "Original 94 weights differ")
        params = {k: utility.meta(v.detach().numpy()) for k, v in neural.update.state_dict().items()}
        require(params == old["original_Update_parameters"] and len(params) == 50, "Original Update 50 parameters differ")
        manual = parent.SourceFrameUpdate(neural.update, terminal_sigmoid="stable").eval()
        wrapped = variant.SourceFrameUpdate(neural.update, terminal_sigmoid="stable").eval()
        report.update(loader=loader, original_weights_before=before, original_Update_parameters=params,
                      original_binding=wrapped.verify_original_binding(neural.update),
                      parent_original_binding=manual.verify_original_binding(neural.update),
                      aggregation_expression=variant.AGGREGATION_EXPRESSION,
                      cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=4, torch_interop_threads=1,
                      torch_version=torch.__version__, numpy_version=np.__version__, onnx_version=onnx.__version__,
                      ORT_version=ort.__version__)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
            options.add_session_config_entry(key, "0")
        gates = []
        for index, previous in enumerate(old["cases"]):
            require(previous["index"] == index and previous["total_graph_edges"] == 23520
                    and previous["source_plan"]["degree"] == 10 and previous["source_plan"]["edges"] == 960
                    and previous["source_plan"]["canonical_shape"] == [10, 96], "Frozen complete source identity differs")
            feeds = load_saved(previous["inputs_artifact"])
            require(tuple(feeds) == INPUTS and [utility.meta(feeds[k])["shape"] for k in INPUTS]
                    == [[1, 10, 96, 384], [1, 10, 96, 384], [1, 10, 96, 882]], "Frozen input contracts differ")
            refs = {key: load_saved(previous[key + "_artifact"]) for key in ("reference", "wrapper", "ort")}
            for values in refs.values():
                utility.require_outputs(values, 10, 96)
            arguments = tuple(torch.from_numpy(feeds[k]) for k in INPUTS)
            wrapped.validate_inputs(*arguments)
            manual.validate_inputs(*arguments)
            case = dict(index=index, role=previous["role"], source_plan=previous["source_plan"], total_graph_edges=23520,
                        all_source_blocks=previous["all_source_blocks"], prior_inputs_sha256=previous["inputs_artifact"]["sha256"],
                        historical_reference_bindings={k: previous[k + "_artifact"] for k in refs},
                        inputs_artifact=save("case-%d-inputs.npz" % index, feeds),
                        reference_artifact=save("case-%d-reference.npz" % index, refs["reference"]))
            report["cases"].append(case)
            with torch.inference_mode():
                t = time.perf_counter()
                result = manual(*arguments)
                manual_values = {k: v.detach().numpy().copy() for k, v in zip(OUTPUTS, result)}
                case["parent_Torch_block_seconds"] = time.perf_counter() - t
                case["parent_artifact"] = save("case-%d-parent.npz" % index, manual_values)
                case["parent_domain"] = utility.require_outputs(manual_values, 10, 96)
                t = time.perf_counter()
                result = wrapped(*arguments)
                values = {k: v.detach().numpy().copy() for k, v in zip(OUTPUTS, result)}
                case["Torch_block_seconds"] = time.perf_counter() - t
                case["wrapper_artifact"] = save("case-%d-wrapper.npz" % index, values)
                case["wrapper_domain"] = utility.require_outputs(values, 10, 96)
                for label, candidate in (("parent", manual_values), ("wrapper", values)):
                    comparisons = {name: {k: utility.compare(candidate[k], reference[k]) for k in OUTPUTS}
                                   for name, reference in refs.items()}
                    case[label + "_vs_saved"] = comparisons
                    gates.extend(item["gate_passed"] for group in comparisons.values() for item in group.values())
                case["wrapper_vs_parent"] = {k: utility.compare(values[k], manual_values[k]) for k in OUTPUTS}
                gates.extend(item["gate_passed"] for item in case["wrapper_vs_parent"].values())
                if index == 0:
                    path = out / "update-d10-m96-softmax.onnx"
                    export = dict(input_names=list(INPUTS), output_names=list(OUTPUTS), opset_version=17, do_constant_folding=True)
                    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                        export["dynamo"] = False
                    t = time.perf_counter()
                    torch.onnx.export(wrapped, arguments, str(path), **export)
                    report["export_seconds"] = time.perf_counter() - t
                    graph = onnx.load(path, load_external_data=False)
                    onnx.checker.check_model(graph, full_check=True)
                    counts = Counter(n.op_type for n in graph.graph.node)
                    require(counts["Softmax"] == 2 and counts["ReduceMax"] == 0, "Expected two direct group Softmax nodes")
                    require(not {"Gather", "GatherElements", "GatherND", "Where"} & set(counts), "Unexpected group selection")
                    inputs = [utility.model_contract(v) for v in graph.graph.input]
                    outputs = [utility.model_contract(v) for v in graph.graph.output]
                    require(inputs == old["inputs"] and outputs == old["outputs"], "Stable three-input/output contracts differ")
                    report.update(model=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)), inputs=inputs, outputs=outputs,
                                  parameter_coverage=utility.initializer_coverage(graph, neural.update.state_dict(), onnx),
                                  onnx_nodes=len(graph.graph.node), onnx_operations=sorted(counts), ONNX_operator_counts=dict(counts),
                                  group_Gather_Where_absent=True)
                    t = time.perf_counter()
                    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                    report["ORT_session_seconds"] = time.perf_counter() - t
                    require(session.get_providers() == ["CPUExecutionProvider"], "Only CPU ORT required")
                    report.update(ORT_providers=session.get_providers(), ORT_graph_optimization="ORT_DISABLE_ALL")
            t = time.perf_counter()
            ort_values = {k: np.array(v, copy=True, order="C") for k, v in zip(OUTPUTS, session.run(list(OUTPUTS), feeds))}
            case["ORT_block_seconds"] = time.perf_counter() - t
            case["ort_artifact"] = save("case-%d-ort.npz" % index, ort_values)
            case["ort_domain"] = utility.require_outputs(ort_values, 10, 96)
            case["ORT_vs_saved"] = {name: {k: utility.compare(ort_values[k], reference[k]) for k in OUTPUTS}
                                    for name, reference in refs.items()}
            case["ORT_vs_wrapper"] = {k: utility.compare(ort_values[k], values[k]) for k in OUTPUTS}
            gates.extend(item["gate_passed"] for group in case["ORT_vs_saved"].values() for item in group.values())
            gates.extend(item["gate_passed"] for item in case["ORT_vs_wrapper"].values())
            require({k: utility.meta(feeds[k]) for k in INPUTS} == case["inputs_artifact"]["arrays"], "Inputs mutated")
            manual.verify_original_binding(neural.update)
            wrapped.verify_original_binding(neural.update)
            print(json.dumps(dict(phase="case_completed", case_index=index, Torch_seconds=case["Torch_block_seconds"],
                                  ORT_seconds=case["ORT_block_seconds"], strict_comparisons_passed=all(gates))), flush=True)
        report.update(status="completed_source_frame_export_diagnostic_not_adopted", strict_FP32_gate_passed=all(gates),
                      outputs_finite=True, weight_domain_valid=True, FP32_thresholds_relaxed=False, output_clipping=False,
                      timing_scope="One first call per method/state including owned output copy. No warm or controlled performance comparison; full original graph is not rerun.",
                      limitations=["Direct softmax is the same real-arithmetic group normalization, not guaranteed bitwise equivalence.",
                                   "All original fullgraph gold outputs are saved v33 artifacts, not fresh execution.",
                                   "Only D10/M96 blocks from two fixed states; no complete graph, trajectory, new frame, NPU or 20 Hz evidence."])
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        session = None
        report["ORT_session_reference_released"] = True
        if neural is not None and before is not None:
            try:
                after = {k: utility.meta(v.detach().numpy()) for k, v in neural.state_dict().items()}
                report["original_weights_after"] = after
                report["original_parameters_unchanged"] = after == before
                require(after == before, "Original weights changed")
            except Exception as error:
                report["cleanup_errors"].append(repr(error))
                report["status"] = "failed"
        try:
            report["source_hashes_after"] = {p: sha(p) for p in pins}
            report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
            require(report["sources_and_inputs_unchanged"], "Sources/inputs changed")
        except Exception as error:
            report["cleanup_errors"].append(repr(error))
            report["status"] = "failed"
        try:
            os.sched_setaffinity(0, affinity)
            report["affinity_restored"] = os.sched_getaffinity(0) == affinity
            require(report["affinity_restored"], "Affinity restore failed")
        except Exception as error:
            report["cleanup_errors"].append(repr(error))
            report["status"] = "failed"
        report.update(source_pins=pins, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      actual_driver_seconds=time.monotonic() - started)
        (out / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "strict_FP32_gate_passed", "outputs_finite",
                                               "weight_domain_valid", "cleanup_errors")}), flush=True)
    return 0 if report["status"] == "completed_source_frame_export_diagnostic_not_adopted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
