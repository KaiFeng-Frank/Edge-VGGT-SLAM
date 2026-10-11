#!/usr/bin/env python3
"""Board-only v35 channel-first Update comparison/export over saved v34 blocks.

The v35 schema and CF shapes are a new, explicit IO/layout contract.
No E23520 full graph is re-executed. Original v33 CPU gold inherited through
v34 and saved v34 comparison arrays are historical. Strict rejection is saved,
never adopted.
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
PARENT_SHA = "b89b9b84d3afb47fd926decddde0077222dc07930bf75ee08acb410dc4f6f3f4"
UTILITY_SHA = "1bb1df7dc4e61e6f3d60fcd1170b267c478fd9600de1f0481ec1a73a42fa1b73"
REFERENCE_SHA = "09c19fd17c28060f2011f13ad3e500ed05748ed8b5e40482ac5903e51a50934b"
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



def pack_channel_first(values):
    import numpy as np
    return {key: np.ascontiguousarray(value.transpose(0, 3, 1, 2)) for key, value in values.items()}


def restore_canonical(values):
    import numpy as np
    return {key: np.ascontiguousarray(value.transpose(0, 2, 3, 1)) for key, value in values.items()}


def require_cf_outputs(values, utility, degree=10, patches=96):
    import numpy as np
    require(tuple(values) == OUTPUTS, "Ordered CF outputs required")
    for name, channels in zip(OUTPUTS, (384, 2, 2)):
        value = values[name]
        require(value.dtype == np.float32 and value.shape == (1, channels, degree, patches),
                "Explicit B,C,D,M output shape differs: " + name)
    return utility.require_outputs(restore_canonical(values), degree, patches)


def initializer_coverage(model, parameters, onnx, utility):
    """Original bytes with exact Conv kernel/affine broadcast reshape only."""
    import numpy as np
    originals = {}
    for name, tensor in parameters.items():
        value = tensor.detach().numpy()
        choices = [("identity", value)]
        if value.ndim == 2:
            choices.append(("conv1x1_kernel_reshape", value.reshape(*value.shape, 1, 1)))
        if value.ndim == 1:
            choices.append(("channel_affine_broadcast_reshape", value.reshape(1, -1, 1, 1)))
        for transform, array in choices:
            key = (tuple(array.shape), str(array.dtype), hashlib.sha256(array.tobytes(order="C")).hexdigest())
            originals.setdefault(key, []).append(dict(parameter=name, transform=transform))
    mapping, covered = [], set()
    for tensor in model.graph.initializer:
        require(tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data,
                "External initializer data prohibited")
        value = onnx.numpy_helper.to_array(tensor)
        key = (tuple(value.shape), str(value.dtype), hashlib.sha256(value.tobytes(order="C")).hexdigest())
        matches = originals.get(key, [])
        require(bool(matches), "Unbound learned initializer " + tensor.name)
        covered.update(v["parameter"] for v in matches)
        mapping.append(dict(initializer=tensor.name, array=utility.meta(value), original_matches=matches))
    require(covered == set(parameters), "Original 50 parameters not completely covered")
    return dict(all_original_parameters_covered=True, original_tensor_count=len(parameters),
                initializer_count=len(mapping), mapping=mapping,
                permitted_transforms=["identity", "conv1x1_kernel_reshape", "channel_affine_broadcast_reshape"],
                parameter_values_unchanged=True)

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
    report = dict(schema="v35_DPVO_source_frame_channel_first_board_export", experiment_version=35,
                  expression_variant="v35_channel_first_original_parameter_conv", status="failed", script_sha256=sha(__file__),
                  wrapper_sha256=args.wrapper_sha256, parent_wrapper_sha256=PARENT_SHA,
                  parent_export_utility_sha256=UTILITY_SHA, reference_manifest_sha256=REFERENCE_SHA,
                  actual_board_execution=True, actual_inference_on_RK3588S=True, host_model_inference=False,
                  original_VGGT_default_unchanged=True, different_model_from_VGGT=True, ORB_used=False,
                  added_LK_PnP_frontend=False, NPU_used=False, BA_executed=False, valid_pose_output=False,
                  realtime_validated=False, trajectory_accuracy_validated=False, candidate_accepted=False,
                  strict_FP32_gate_passed=False, outputs_finite=False, weight_domain_valid=False,
                  original_parameters_unchanged=False, terminal_sigmoid="stable", cases=[], cleanup_errors=[],
                  full_E23520_reference_reexecuted=False,
                  reference_boundary="Frozen v34 blocks and historical original v33 CPU gold are loaded. Only two D10/M96 source blocks execute here; no new fullgraph gold.",
                  source_plan_recomputed=False,
                  scope="Channel-first Conv/channel-LayerNorm expression; all real source members retained. No new observation, full graph, BA, pose or feedback.",
                  input_layout="B,C,D,M", output_layout="B,C,D,M", canonical_layout="B,D,M,C",
                  layout_bridge="Exact axis permutation on CPU outside neural forward; both layouts saved",
                  FP32_thresholds_relaxed=False, output_clipping=False,
                  SDK_or_NPU_placement_validated=False, controlled_speedup_validated=False)

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
        require(old["schema"] == "v33_DPVO_source_frame_Update_board_export"
                and old["experiment_version"] == 34 and old["expression_variant"] == "v34_direct_torch_softmax" and old["status"] == "completed_source_frame_export_diagnostic_not_adopted"
                and old["strict_FP32_gate_passed"] and old["original_parameters_unchanged"]
                and old["sources_and_inputs_unchanged"] and old["source_pins"] == old["source_hashes_after"]
                and old["cleanup_errors"] == [] and len(old["cases"]) == 2,
                "Frozen successful v34 CPU reference/export required")
        require({4, 5, 6, 7} <= affinity, "Four A76 CPUs required")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import numpy as np
        import torch
        import onnx
        import onnxruntime as ort
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        utility = module(args.utility_source, "_v35_frozen_export_utils")
        common = module(args.v26_source_dir / "dpvo_probe_common.py", "_v35_original_loader")
        parent = module(args.parent_source, "_v35_original_expression")
        variant = module(args.wrapper_source, "_v35_direct_expression")
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
        wrapped = variant.SourceFrameUpdateChannelFirst(neural.update, terminal_sigmoid="stable").eval()
        report.update(loader=loader, original_weights_before=before, original_Update_parameters=params,
                      original_binding=wrapped.verify_original_binding(neural.update),
                      parent_original_binding=manual.verify_original_binding(neural.update),
                      aggregation_expression=variant.AGGREGATION_EXPRESSION, channel_norm_expression=variant.CHANNEL_NORM_FORMULA,
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
            canonical_feeds = load_saved(previous["inputs_artifact"])
            require(tuple(canonical_feeds) == INPUTS
                    and [utility.meta(canonical_feeds[k])["shape"] for k in INPUTS]
                    == [[1, 10, 96, 384], [1, 10, 96, 384], [1, 10, 96, 882]], "Historical canonical inputs differ")
            refs = {key: load_saved(previous[key + "_artifact"])
                    for key in ("reference", "parent", "wrapper", "ort")}
            for reference in refs.values():
                utility.require_outputs(reference, 10, 96)
            packed_start = time.perf_counter()
            feeds = pack_channel_first(canonical_feeds)
            require(all(np.array_equal(restore_canonical(feeds)[k], canonical_feeds[k]) for k in INPUTS),
                    "Input axis permutation changed values")
            packing_seconds = time.perf_counter() - packed_start
            arguments = tuple(torch.from_numpy(feeds[k]) for k in INPUTS)
            canonical_arguments = tuple(torch.from_numpy(canonical_feeds[k]) for k in INPUTS)
            wrapped.validate_inputs(*arguments)
            manual.validate_inputs(*canonical_arguments)
            reference_cf = pack_channel_first(refs["reference"])
            case = dict(index=index, role=previous["role"], source_plan=previous["source_plan"], total_graph_edges=23520,
                        all_source_blocks=previous["all_source_blocks"], prior_inputs_sha256=previous["inputs_artifact"]["sha256"],
                        historical_reference_bindings={k: previous[k + "_artifact"] for k in refs},
                        input_layout="B,C,D,M", output_layout="B,C,D,M", comparison_layout="B,D,M,C",
                        CPU_input_pack_and_check_seconds=packing_seconds,
                        inputs_artifact=save("case-%d-inputs.npz" % index, feeds),
                        canonical_inputs_artifact=save("case-%d-canonical-inputs.npz" % index, canonical_feeds),
                        reference_artifact=save("case-%d-reference.npz" % index, reference_cf),
                        canonical_reference_artifact=save("case-%d-canonical-reference.npz" % index, refs["reference"]))
            report["cases"].append(case)
            with torch.inference_mode():
                t = time.perf_counter()
                result = manual(*canonical_arguments)
                manual_values = {k: v.detach().numpy().copy() for k, v in zip(OUTPUTS, result)}
                case["parent_Torch_block_seconds"] = time.perf_counter() - t
                case["parent_artifact"] = save("case-%d-parent.npz" % index, pack_channel_first(manual_values))
                case["canonical_parent_artifact"] = save("case-%d-canonical-parent.npz" % index, manual_values)
                case["parent_domain"] = utility.require_outputs(manual_values, 10, 96)
                t = time.perf_counter()
                result = wrapped(*arguments)
                values_cf = {k: v.detach().numpy().copy() for k, v in zip(OUTPUTS, result)}
                case["Torch_block_seconds"] = time.perf_counter() - t
                case["wrapper_artifact"] = save("case-%d-wrapper.npz" % index, values_cf)
                case["wrapper_domain"] = require_cf_outputs(values_cf, utility)
                t = time.perf_counter()
                values = restore_canonical(values_cf)
                case["CPU_wrapper_restore_seconds"] = time.perf_counter() - t
                case["canonical_wrapper_artifact"] = save("case-%d-canonical-wrapper.npz" % index, values)
                for label, candidate in (("parent", manual_values), ("wrapper", values)):
                    comparisons = {name: {k: utility.compare(candidate[k], reference[k]) for k in OUTPUTS}
                                   for name, reference in refs.items()}
                    case[label + "_vs_saved"] = comparisons
                    gates.extend(item["gate_passed"] for group in comparisons.values() for item in group.values())
                case["wrapper_vs_parent"] = {k: utility.compare(values[k], manual_values[k]) for k in OUTPUTS}
                gates.extend(item["gate_passed"] for item in case["wrapper_vs_parent"].values())
                if index == 0:
                    path = out / "update-cf-d10-m96.onnx"
                    export = dict(input_names=list(INPUTS), output_names=list(OUTPUTS), opset_version=17, do_constant_folding=True)
                    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                        export["dynamo"] = False
                    t = time.perf_counter()
                    torch.onnx.export(wrapped, arguments, str(path), **export)
                    report["export_seconds"] = time.perf_counter() - t
                    graph = onnx.load(path, load_external_data=False)
                    onnx.checker.check_model(graph, full_check=True)
                    counts = Counter(n.op_type for n in graph.graph.node)
                    softmax_axes = [next(onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "axis")
                                    for n in graph.graph.node if n.op_type == "Softmax"]
                    reduce_axes = [next(onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "axes")
                                   for n in graph.graph.node if n.op_type == "ReduceMean"]
                    require(counts["Softmax"] == 2 and sorted(softmax_axes) == [2, 3], "CF group Softmax axes differ")
                    require(counts["Conv"] == 21 and counts["ReduceMean"] == 8 and all(a == [1] for a in reduce_axes),
                            "Original Linear/channel LayerNorm graph differs")
                    require(not {"Transpose", "MatMul", "Gemm", "Gather", "GatherElements", "GatherND", "Where", "Pad"} & set(counts),
                            "Unexpected layout transform/group selection")
                    inputs = [utility.model_contract(v) for v in graph.graph.input]
                    outputs = [utility.model_contract(v) for v in graph.graph.output]
                    expected_inputs = [dict(name=k, dtype="float32", shape=[1, c, 10, 96]) for k, c in zip(INPUTS, (384,384,882))]
                    expected_outputs = [dict(name=k, dtype="float32", shape=[1, c, 10, 96]) for k, c in zip(OUTPUTS,(384,2,2))]
                    require(inputs == expected_inputs and outputs == expected_outputs, "New explicit CF IO contracts differ")
                    report.update(model=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)), inputs=inputs, outputs=outputs,
                                  canonical_inputs=old["inputs"], canonical_outputs=old["outputs"],
                                  parameter_coverage=initializer_coverage(graph, neural.update.state_dict(), onnx, utility),
                                  onnx_nodes=len(graph.graph.node), onnx_operations=sorted(counts), ONNX_operator_counts=dict(counts),
                                  Softmax_axes=softmax_axes, channel_ReduceMean_axes=reduce_axes,
                                  group_Gather_Where_absent=True, ONNX_Transpose_absent=True)
                    t = time.perf_counter()
                    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                    report["ORT_session_seconds"] = time.perf_counter() - t
                    require(session.get_providers() == ["CPUExecutionProvider"], "Only CPU ORT required")
                    report.update(ORT_providers=session.get_providers(), ORT_graph_optimization="ORT_DISABLE_ALL")
            t = time.perf_counter()
            ort_cf = {k: np.array(v, copy=True, order="C") for k,v in zip(OUTPUTS,session.run(list(OUTPUTS),feeds))}
            case["ORT_block_seconds"] = time.perf_counter() - t
            case["ort_artifact"] = save("case-%d-ort.npz" % index,ort_cf)
            case["ort_domain"] = require_cf_outputs(ort_cf,utility)
            t = time.perf_counter()
            ort_values = restore_canonical(ort_cf)
            case["CPU_ORT_restore_seconds"] = time.perf_counter() - t
            case["canonical_ort_artifact"] = save("case-%d-canonical-ort.npz" % index,ort_values)
            case["ORT_vs_saved"] = {name:{k:utility.compare(ort_values[k],reference[k]) for k in OUTPUTS}
                                    for name,reference in refs.items()}
            case["ORT_vs_wrapper"] = {k:utility.compare(ort_values[k],values[k]) for k in OUTPUTS}
            gates.extend(item["gate_passed"] for group in case["ORT_vs_saved"].values() for item in group.values())
            gates.extend(item["gate_passed"] for item in case["ORT_vs_wrapper"].values())
            require({k:utility.meta(feeds[k]) for k in INPUTS}==case["inputs_artifact"]["arrays"],"CF inputs mutated")
            require({k:utility.meta(canonical_feeds[k]) for k in INPUTS}==case["canonical_inputs_artifact"]["arrays"],"Canonical inputs mutated")
            manual.verify_original_binding(neural.update)
            wrapped.verify_original_binding(neural.update)
            print(json.dumps(dict(phase="case_completed",case_index=index,Torch_seconds=case["Torch_block_seconds"],
                                  ORT_seconds=case["ORT_block_seconds"],strict_comparisons_passed=all(gates))),flush=True)
        report.update(status="completed_channel_first_export_diagnostic_not_adopted", strict_FP32_gate_passed=all(gates),
                      strict_comparison_count=len(gates), outputs_finite=True, weight_domain_valid=True,
                      timing_scope="One first neural call per method/state including owned output copy. CF packing and canonical restore/validation/save are separate; no warm/controlled speedup comparison.",
                      limitations=["Conv and centered-population channel LayerNorm change FP32 rounding; no bitwise equivalence is claimed.",
                                   "All fullgraph original gold is saved v33 data inherited through v34, not fresh fullgraph execution.",
                                   "CF inputs/outputs are a new IO contract; old SDK shape/layout checks must not be reused as if unchanged.",
                                   "ONNX lacks Transpose by construction; actual RKNN scheduling and numerical gates remain untested here.",
                                   "Only two D10/M96 source blocks; no new frame, full graph, correlation, BA, feedback, trajectory, NPU or 20 Hz proof."])
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
    return 0 if report["status"] == "completed_channel_first_export_diagnostic_not_adopted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
