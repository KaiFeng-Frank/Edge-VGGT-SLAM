#!/usr/bin/env python3
"""Board-only PTQ diagnostic of the frozen FP32-rejected spatial sparse encoder.

build(do_quantization=True) includes CPU calibration inference. Therefore this
entrypoint requires Linux/aarch64 and the RK3588 device-tree identity before
loading NumPy, ONNX, or RKNN. Other ARM hosts are also rejected.
No production source or historical FP16 acceptance threshold is changed.
DPVO reference: MIT, Copyright (c) 2022 Princeton Vision & Learning Lab.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

SPARSE_COMPILER_SHA = "26e639520692619ca7d69916d11d81237c535ee31f3ea9aa848101e4742744fa"
GRAPH_SHA = "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
CONFIGURATION = dict(target_platform="rk3588", float_dtype="float16", optimization_level=3,
                     quantized_dtype="w8a8", quantized_algorithm="normal", quantized_method="channel",
                     mean_values=None, std_values=None, quant_img_RGB2BGR=False)


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
    require((platform.machine() if machine is None else machine) == "aarch64",
            "Host PTQ prohibited: RKNN calibration is neural execution; run only on aarch64 board")
    require((platform.system() if system is None else system) == "Linux", "Linux board required")
    if compatible is None:
        path = Path("/proc/device-tree/compatible")
        compatible = path.read_bytes() if path.is_file() else b""
    require(any(t in (b"rockchip,rk3588", b"rockchip,rk3588s") for t in compatible.split(b"\0")),
            "RK3588 device-tree identity required; other ARM hosts prohibited")


def load_pinned(path, expected, name, pins):
    path = Path(path).resolve()
    require(sha(path) == expected, "Source SHA mismatch: " + path.name)
    pins[str(path)] = expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_calibration_arrays(image, indices):
    # NumPy import is reached only after main's board guard. Tiny tests call
    # this helper with synthetic arrays; no neural graph is executed.
    import numpy as np
    require(image.dtype == np.float32 and image.shape == (1, 3, 464, 608),
            "Calibrator requires original NCHW FP32 normalized BGR shape")
    require(np.isfinite(image).all() and image.min() >= -.5 and image.max() <= 1.5,
            "Original once-normalized BGR range [-.5,1.5] required")
    require(indices.dtype == np.int64 and indices.shape == (96,),
            "Dynamic patch_indices must remain INT64[96], never floating point")
    y, x = indices // 152, indices % 152
    require(np.all((x >= 1) & (x < 151) & (y >= 1) & (y < 115)),
            "Original 3x3 patch centers must be interior of 116x152 feature grid")
    return dict(image_dtype=str(image.dtype), image_shape=list(image.shape),
                image_min=float(image.min()), image_max=float(image.max()),
                indices_dtype=str(indices.dtype), indices_shape=list(indices.shape),
                index_min=int(indices.min()), index_max=int(indices.max()),
                index_order_preserved=True, index_cast_performed=False)


def prepare_dataset(path, expected_sha, output, pins):
    import numpy as np
    path = Path(path).resolve()
    require(sha(path) == expected_sha, "Calibration dataset list SHA mismatch")
    pins[str(path)] = expected_sha
    rows, lines = [], []
    for index, line in enumerate(path.read_text().splitlines()):
        require(bool(line.strip()), "Blank calibration row is not allowed")
        fields = line.split()
        require(len(fields) == 2, "Each row must contain image.npy then indices.npy")
        paths = [(path.parent / value).resolve() for value in fields]
        require(all(p.suffix == ".npy" and p.is_file() and not any(c.isspace() for c in str(p)) for p in paths),
                "Two existing .npy paths without whitespace required")
        before = [sha(p) for p in paths]
        arrays = [np.load(p, allow_pickle=False) for p in paths]
        meta = validate_calibration_arrays(*arrays)
        require([sha(p) for p in paths] == before, "Calibration inputs changed while reading")
        for p, digest in zip(paths, before):
            pins[str(p)] = digest
        rows.append(dict(index=index, input_files=[dict(file=str(p), sha256=digest, bytes=p.stat().st_size)
                                                  for p, digest in zip(paths, before)], **meta))
        lines.append(" ".join(map(str, paths)))
    require(rows, "At least one calibration input set required")
    require(sum(p["bytes"] for r in rows for p in r["input_files"]) <= 256 * 1024**2,
            "Calibration source byte budget exceeded")
    resolved = output / "dataset-resolved.txt"
    resolved.write_text("\n".join(lines) + "\n")
    pins[str(resolved)] = sha(resolved)
    return resolved, rows


def perform_ptq(factory, graph, dataset, output, report):
    board_guard()
    toolkit = None
    try:
        toolkit = factory(verbose=True)
        operations = [
            ("config", lambda: toolkit.config(**CONFIGURATION)),
            ("load_onnx", lambda: toolkit.load_onnx(model=str(graph))),
            ("board_CPU_calibration_and_build", lambda: toolkit.build(
                do_quantization=True, dataset=str(dataset), auto_hybrid=False)),
            ("export_rknn", lambda: toolkit.export_rknn(str(output / "block-int8.rknn")))]
        for name, operation in operations:
            print("START", name, flush=True)
            begin = time.monotonic()
            code = operation()
            report["operations"].append(dict(name=name, code=code, seconds=time.monotonic() - begin))
            require(code is None or type(code) is int and code == 0, "Failed operation: " + name)
        model = output / "block-int8.rknn"
        require(0 < model.stat().st_size <= 128 * 1024**2, "Compiled artifact size budget exceeded")
        report["output"] = dict(file=model.name, bytes=model.stat().st_size, sha256=sha(model))
    finally:
        if toolkit is not None:
            primary = sys.exc_info()[0] is not None
            try:
                toolkit.release()
                report["released"] = True
            except Exception as exc:
                report["cleanup_errors"].append("RKNN release: " + repr(exc))
                if not primary:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dependency-plan", "preflight-script", "frozen-sparse-compiler", "rewrite-dir",
                 "reference-dir", "parent-onnx", "editor", "dataset", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("dependency-plan", "preflight-script", "dataset"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--diagnostic-allow-fp32-rejected", action="store_true")
    args = parser.parse_args(argv)
    board_guard()  # before creating output, reading data or importing SDK
    require(args.diagnostic_allow_fp32_rejected, "Explicit rejected spatial-parent diagnostic required")
    output = args.output_dir.resolve()
    require(output.is_relative_to(Path("/mnt/vggt-data/vggt-slam2/experiments")) and output != Path("/mnt/vggt-data/vggt-slam2/experiments"), "New board experiment output required")
    protected = [v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"]
    require(not any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
            "Output/source overlap")
    output.mkdir(parents=True, exist_ok=False)
    pins = {str(Path(__file__).resolve()): sha(__file__)}
    report = dict(schema="v37_board_only_DPVO_spatial_encoder_PTQ", status="failed", diagnostic_only=True,
                  script_sha256=sha(__file__),
                  candidate_accepted=False, production_changed=False, host_calibration_executed=False,
                  diagnostic_allow_fp32_rejected=True, historical_parent_FP32_gate_passed=False,
                  historical_parent_rejection_retained=True,
                  parent_PTQ_source_sha256="0f66e0ead40ab6eff546a47367ee93ec044b2b84e064eb8e9876352e0ca05fb0",
                  calibration_location="aarch64 board CPU", PTQ_requested=True,
                  INT8_operator_coverage_validated=False, deployment_inference_executed=False,
                  configuration={**CONFIGURATION, "do_quantization": True, "auto_hybrid": False},
                  original_FP32_gate={"rtol": 1e-4, "atol": 1e-5, "measured": False},
                  source_pins=pins, operations=[], released=False, cleanup_errors=[])
    try:
        require(sha(args.dependency_plan) == args.dependency_plan_sha256, "Dependency plan SHA mismatch")
        pins[str(args.dependency_plan.resolve())] = args.dependency_plan_sha256
        plan = json.loads(args.dependency_plan.read_text())
        preflight = load_pinned(args.preflight_script, args.preflight_script_sha256, "_v35_preflight", pins)
        report["environment"] = preflight.inspect_environment(plan)
        require(report["environment"]["status"] == "ready_for_board_PTQ_diagnostic", "Isolated ARM environment not ready")
        base = load_pinned(args.frozen_sparse_compiler, SPARSE_COMPILER_SHA, "_v35_frozen_v34", pins)
        spec = base.MODELS["spatial"]
        recipe_path = base.pin(args.rewrite_dir / "rewrite.json", spec["recipe_sha"], pins)
        ref_path = base.pin(args.reference_dir / "ort.json", REFERENCE_SHA, pins)
        recipe, reference = [json.loads(p.read_text()) for p in (recipe_path, ref_path)]
        report["ancestor"] = base.validate_lineage(recipe, reference, "spatial", args.diagnostic_allow_fp32_rejected)
        require(report["ancestor"]["parent_CPU_strict_gate_passed"] is False and report["ancestor"]["spatial_parent_rejection_retained"] is True, "Historical rejection must remain false")
        graph = base.pin(args.rewrite_dir / recipe["output"]["file"], GRAPH_SHA, pins)
        parent = base.pin(args.parent_onnx, spec["parent_sha"], pins)
        editor = load_pinned(args.editor, base.EDITOR_SHA, "_v35_frozen_editor", pins)
        import onnx
        original_graph = onnx.load(parent, load_external_data=False)
        sparse_graph = onnx.load(graph, load_external_data=False)
        editor._reject_complex_storage(sparse_graph)
        onnx.checker.check_model(sparse_graph, full_check=True)
        require(editor.preservation(original_graph, sparse_graph) == recipe["preservation"],
                "Spatial-parent parameters/non-suffix nodes changed")
        require(editor._contract(sparse_graph.graph.input) == base.INPUTS and
                editor._contract(sparse_graph.graph.output) == base.OUTPUTS, "Frozen graph IO changed")
        report["input"] = dict(file=str(graph), sha256=GRAPH_SHA, original_graph_weights_preserved=True)
        dataset, rows = prepare_dataset(args.dataset, args.dataset_sha256, output, pins)
        report["calibration"] = dict(dataset_sha256=sha(dataset), samples=len(rows), rows=rows,
            mean_std_identity=True, input_normalization_contract="2*(BGR/255)-0.5, once before this entrypoint",
            preprocessing_provenance_independently_verified=False,
            note="Input range/shape guards do not independently prove preprocessing provenance or representativeness")
        from rknn.api import RKNN
        perform_ptq(RKNN, graph, dataset, output, report)
        report["status"] = "compiled_board_spatial_PTQ_parent_rejection_retained_not_runtime_validated"
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        report["source_hashes_after"] = {p: sha(p) if Path(p).is_file() else None for p in pins}
        report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if not report["sources_and_inputs_unchanged"] or report["cleanup_errors"]:
            report["status"] = "failed"
        (output / "compile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "released")}), flush=True)
    return 0 if report["status"] == "compiled_board_spatial_PTQ_parent_rejection_retained_not_runtime_validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
