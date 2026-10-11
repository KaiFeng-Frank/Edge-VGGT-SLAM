#!/usr/bin/env python3
"""RK3588/S-only half-resolution M16 encoder SDK diagnostic, not VO.

MIT project helper; DPVO copyright (c) 2022 Princeton Vision & Learning Lab.
Uses the frozen v34 typed transport and v37 arbitrary-shape attr validator.
One context queries actual descriptors before FP32 NHWC/INT64 input submission.
Normal: image0 cold+3warm, images1/2 once; profile: a separate three-call run.
All three saved new-size references retain rtol=1e-4, atol=1e-5. No adoption.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
POLICY_SHA = "cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a"
PROBE_SHA = "bdf0feea251e478d761a7bf5c9635e46fee39cb0e9a9720b0949bb54b18e7c8a"
PROBE_DRIVER_SHA = "01e4e81c71216a491b2d03200201a0e55dc3fcfc06bc94c3075d638ff8949e4f"
COMPILER_SHA = "fe8411d7bc75efc6a2ab565faa74f70d233f3feb6d243af9369a8c02ce7cc1c1"
GRAPH_SHA = "a9ccc241d364b8de2ed0413d8a27f7a8d30fafe768919b365c46e8c1ff9eb7b5"
QUERY_SHA = "c61a377ede3d33b5a2412b1c7f526ea75deff2f18ed185f0f937792e41b1aca2"
TRANSPORT_SHA = "646f875d1b5eea561c0d6aa4d1b9078351014a39a62976ec895a814cd466b387"
ABI_SHA = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"
HEADER_SHA = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
LIBRARY_SHA = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
INPUT_SHAPE = (1, 3, 232, 304)
OUTPUTS = (("fmap", (1, 128, 58, 76)), ("context", (1, 16, 384)))
CONFIGURATION = dict(target_platform="rk3588", float_dtype="float16", optimization_level=3,
    quantized_dtype="w8a8", quantized_algorithm="normal", quantized_method="channel",
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, do_quantization=True, auto_hybrid=False)


def require(ok, text):
    if not ok:
        raise ValueError(text)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def board_guard():
    require(platform.machine() == "aarch64" and platform.system() == "Linux", "RK3588/S Linux aarch64 board only")
    compatible = Path("/proc/device-tree/compatible").read_bytes().split(b"\0")
    require(b"rockchip,rk3588" in compatible or b"rockchip,rk3588s" in compatible, "RK3588/S device required")


def unchanged(value):
    require(value["source_pins"] == value["source_hashes_after"] and value["sources_and_inputs_unchanged"] is True and
            value["cleanup_errors"] == [], "Unchanged, clean source evidence required")


def validate_compile(value, compiler_sha, probe, model_sha, model_bytes):
    require(compiler_sha == COMPILER_SHA, "Frozen half-size compiler SHA required")
    require(value["schema"] == "v39_board_only_DPVO_half_M16_encoder_PTQ" and
            value["status"] == "compiled_board_half_M16_PTQ_not_runtime_validated_historical_rejection_retained" and
            value["script_sha256"] == compiler_sha and compiler_sha in value["source_pins"].values(), "Completed pinned half-size board PTQ")
    unchanged(value)
    require(value["configuration"] == CONFIGURATION and value["released"] is True and
            [r["name"] for r in value["operations"]] == ["config", "load_onnx", "board_CPU_calibration_and_build", "export_rknn"] and
            all(type(r["code"]) is int and r["code"] == 0 for r in value["operations"]), "Frozen PTQ configuration/operations")
    require(value["policy_sha256"] == POLICY_SHA and value["probe_sha256"] == PROBE_SHA and
            value["probe_driver_sha256"] == PROBE_DRIVER_SHA and value["rewrite_sha256"] == probe["rewrite_sha256"] and
            value["new_size_CPU_strict_gate_each_method"] == {"original": True, "reciprocal": True}, "Same policy/new-size CPU evidence")
    require(value["input"]["sha256"] == GRAPH_SHA and GRAPH_SHA in probe["source_pins"].values() and value["input"]["original_graph_weights_preserved"] is True and
            value["input"]["inputs"] == [dict(name="normalized_bgr", dtype="FLOAT", shape=list(INPUT_SHAPE)),
                dict(name="patch_indices", dtype="INT64", shape=[16])] and
            value["input"]["outputs"] == [dict(name=k, dtype="FLOAT", shape=list(s)) for k, s in OUTPUTS], "Exact half M16 graph IO/weight lineage")
    require(value["candidate_accepted"] is False and value["production_changed"] is False and value["diagnostic_only"] is True and
            value["diagnostic_allow_fp32_rejected"] is True and value["historical_parent_FP32_gate_passed"] is False and
            value["historical_parent_rejection_retained"] is True and value["host_calibration_executed"] is False and
            value["calibration_location"] == "RK3588/S aarch64 board CPU" and value["PTQ_requested"] is True and
            value["deployment_inference_executed"] is False and value["INT8_operator_coverage_validated"] is False,
            "Diagnostic-only board calibration; historical rejection cannot be promoted")
    calibration = value["calibration"]
    require(calibration["samples"] == len(calibration["rows"]) == 32 and calibration["image_shape"] == list(INPUT_SHAPE) and
            calibration["index_shape"] == [16] and calibration["indices_dtype"] == "int64" and
            calibration["source_indices"] == [256+16*i for i in range(32)] and calibration["evaluation_disjoint_first192"] is True and
            calibration["normalization_applications_per_image"] == 1 and calibration["GT_or_content_selection"] is False,
            "Predeclared new half-size calibration, distinct from first192 evaluation")
    require(value["output"]["sha256"] == model_sha and value["output"]["bytes"] == model_bytes and
            0 < model_bytes <= 256*1024**2, "Exact compiled model bytes/SHA and bounded size")


def validate_image_indices(image, indices):
    import numpy as np
    require(image.dtype == np.float32 and image.shape == INPUT_SHAPE and np.isfinite(image).all() and
            image.min() >= -.5 and image.max() <= 1.5, "Once-normalized half-size FP32 BGR")
    require(indices.dtype == np.int64 and indices.shape == (16,), "Original ordered INT64 M16 indices; no coercion")
    y, x = np.divmod(indices, 76)
    require(np.all((x >= 1) & (x < 75) & (y >= 1) & (y < 57)), "Interior indices on58x76 grid")


def pack_input(image, fmt):
    import numpy as np
    require(image.dtype == np.float32 and image.shape == INPUT_SHAPE and fmt == 1,
            "This helper submits explicit FP32 NHWC/pass_through0 only")
    return np.array(image.transpose(0, 2, 3, 1), order="C", copy=True)


def comparison(actual, expected):
    import numpy as np
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32 and
            np.isfinite(actual).all() and np.isfinite(expected).all(), "Matching finite canonical FP32 features")
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    d = a-b
    count = int(np.count_nonzero(np.abs(d) > 1e-5+1e-4*np.abs(b)))
    norm = float(np.linalg.norm(b))
    err = float(np.linalg.norm(d))
    return dict(rtol=1e-4, atol=1e-5, gate_passed=count == 0, failed_elements=count, element_count=int(a.size),
        bitwise_equal=actual.tobytes(order="C") == expected.tobytes(order="C"), max_abs=float(np.max(np.abs(d))),
        relative_l2=err/norm if norm else 0.0 if err == 0 else None,
        rmse=float(np.sqrt(np.mean(d*d))), finite=True, actual_all_zero=bool(np.all(actual == 0)))


def schedule(profile):
    return [(i, "profile") for i in range(3)] if profile else [(0, "cold")]+[(0, "warm")]*3+[(1, "other_frame"), (2, "other_frame")]


class HalfContext:
    """A single fixed-model context; saved attrs precede any guard failure.

    No CPU fallback. Output data are copied by frozen execute_sparse before
    release. Caller serializes calls; this diagnostic has one owner/thread.
    """
    def __init__(self, api, abi, validator, transport, model, profile, record, cleanup):
        self.api, self.abi, self.transport = api, abi, transport
        self.record, self.cleanup = record, cleanup
        self.context, self.closed = abi.CONTEXT(), False
        self.completed_calls = 0
        try:
            filename = ct.create_string_buffer(os.fsencode(model))
            started = time.perf_counter()
            code = int(api.rknn_init(ct.byref(self.context), ct.cast(filename, ct.c_void_p), 0, 8 if profile else 0, None))
            record["init_return_code"] = code
            abi.check_call(code, "rknn_init")
            require(self.context.value != 0, "Successful init returned null context")
            record["init_seconds"] = time.perf_counter()-started
            abi.check_call(api.rknn_set_core_mask(self.context, 7), "rknn_set_core_mask")
            version = abi.query(api, self.context, 5, abi.SDKVersion())
            record["sdk_versions"] = dict(api=bytes(version.api_version).decode(errors="replace"),
                                         driver=bytes(version.drv_version).decode(errors="replace"))
            require(record["sdk_versions"]["api"].startswith("2.3.2"), "Pinned RKNN runtime2.3.2 required")
            counts = abi.query(api, self.context, 0, abi.InputOutputNum())
            record.update(core_mask=7, init_flags=8 if profile else 0,
                          input_count=int(counts.n_input), output_count=int(counts.n_output))
            require(counts.n_input == counts.n_output == 2, "Exactly two sparse inputs/outputs")
            record["input_attrs"] = [abi.attr_record(abi.query(api, self.context, 1, abi.TensorAttr(index=i))) for i in range(2)]
            record["output_attrs"] = [abi.attr_record(abi.query(api, self.context, 2, abi.TensorAttr(index=i))) for i in range(2)]
            contracts = [(record["input_attrs"][0], 0, "normalized_bgr", INPUT_SHAPE, dict(spatial=True)),
                (record["input_attrs"][1], 1, "patch_indices", (16,), dict(integer=True)),
                (record["output_attrs"][0], 0, "fmap", OUTPUTS[0][1], dict(spatial=True)),
                (record["output_attrs"][1], 1, "context", OUTPUTS[1][1], {})]
            record["validated_native_contracts"] = [validator.validate_attr(r, i, n, s, **kw) for r, i, n, s, kw in contracts]
            # want_float converts dtype, not arbitrary spatial layout. Never
            # reshape an NHWC output as NCHW or infer layout from byte count.
            require(record["output_attrs"][0]["format"] == 0 and record["output_attrs"][1]["format"] in (0, 3),
                    "Canonical fmap NCHW and context unambiguous logical outputs required")
            self.index_attr = record["input_attrs"][1]
            memory = abi.query(api, self.context, 6, abi.MemSize())
            record["model_memory_bytes"] = {k: int(getattr(memory, k)) for k, _ in memory._fields_ if k != "reserved"}
            require(memory.total_dma_allocated_size <= 1024**3, "Context DMA exceeds isolated1GiB budget")
            record["native_IO_contract_checked"] = True
        except BaseException:
            self.close()
            raise

    def execute(self, image, indices, activity):
        require(not self.closed, "Closed half-size context")
        validate_image_indices(image, indices)
        result = self.transport.execute_sparse(self.api, self.abi, SimpleNamespace(pack_input=pack_input),
            self.context, image, indices, 1, self.index_attr, self.cleanup, activity, outputs=OUTPUTS, timeout_ms=60000)
        self.completed_calls += 1
        self.record["completed_output_copy_release_calls"] = self.completed_calls
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.record["closed"] = True
        if self.context.value:
            try:
                code = int(self.api.rknn_destroy(self.context))
                self.record.update(destroy_return_code=code, context_destroyed=code == 0)
                self.abi.check_call(code, "rknn_destroy")
            except BaseException as exc:
                self.record["context_destroyed"] = False
                self.cleanup.append("rknn_destroy: "+repr(exc))
        else:
            self.record["context_destroyed"] = None


def load_rows(directory, probe, artifact):
    import numpy as np
    unchanged(probe)
    require(probe["schema"] == "v39_DPVO_half_encoder_board_probe" and
            probe["status"] == "completed_half_M16_CPU_teacher_and_ORT_diagnostic_not_VO" and
            probe["script_sha256"] == PROBE_DRIVER_SHA and probe["policy_sha256"] == POLICY_SHA and
            probe["strict_gate_each_method"] == {"original": True, "reciprocal": True} and
            probe["state_unchanged"] is True and probe["state_before"] == probe["state_after"] and
            len(probe["state_before"]) == 94 and probe["historical_parent_FP32_gate_passed"] is False and
            probe["historical_parent_rejection_retained"] is True and probe["candidate_accepted"] is False and
            probe["normalization_applications_per_image"] == 1 and probe["original_FP32_gate"] == dict(rtol=1e-4, atol=1e-5) and
            len(probe["calls"]) == 3, "Clean original-weight new-size reference probe, without promoting historical rejection")
    rows = []
    def arrays(row):
        with np.load(artifact(directory, row), allow_pickle=False) as archive:
            require(set(archive.files) == set(row["arrays"]), "Saved reference artifact key mismatch")
            values = {k: archive[k] for k in archive.files}
        require(all(metadata(v) == row["arrays"][k] for k, v in values.items()), "Reference raw tensor metadata mismatch")
        return values
    for i, row in enumerate(probe["calls"]):
        require(row["index"] == row["source_index"] == i, "Three original source identities/order")
        inputs = arrays(row["input_artifact"])
        image, indices = inputs["normalized_bgr"], inputs["patch_indices"]
        validate_image_indices(image, indices)
        require(metadata(image) == row["image_array"] and metadata(indices) == row["index_array"] and
                inputs["centers_xy"].shape == (16, 2) and inputs["centers_xy"].dtype == np.int64 and
                np.array_equal(indices, inputs["centers_xy"][:, 1]*76+inputs["centers_xy"][:, 0]), "Input/ordered centers binding")
        teacher = arrays(row["teacher_artifact"])
        original = arrays(row["original"]["artifact"])
        reciprocal = arrays(row["reciprocal"]["artifact"])
        require(teacher["imap"].shape == original["imap"].shape == (1, 384, 58, 76), "New-size dense context teacher/reference")
        def gather(imap):
            return np.ascontiguousarray(imap[0, :, indices//76, indices%76][None])
        require(np.array_equal(gather(teacher["imap"]), teacher["context"]), "CPU patchify must bind original ordered context indices")
        refs = dict(original_CPU={k: teacher[k] for k, _ in OUTPUTS},
            original_ORT=dict(fmap=original["fmap"], context=gather(original["imap"])), reciprocal_ORT=reciprocal)
        for values in refs.values():
            require(set(values) == {k for k, _ in OUTPUTS} and all(v.shape == s and v.dtype == np.float32 and
                np.isfinite(v).all() and np.any(v != 0) for k, s in OUTPUTS for v in (values[k],)), "Finite new-size canonical references")
        require(all(comparison(refs[m][k], refs["original_CPU"][k])["gate_passed"] for m in ("original_ORT", "reciprocal_ORT") for k, _ in OUTPUTS),
                "Independently recomputed new-size CPU reference gates")
        rows.append(dict(identity=row, image=image, indices=indices, references=refs))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("compile-dir", "probe-dir", "policy", "query-helper", "transport-helper", "abi-helper", "header", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--compile-sha256", required=True)
    p.add_argument("--compiler-sha256", required=True)
    p.add_argument("--library", type=Path, default=Path("/usr/lib/librknnrt.so"))
    p.add_argument("--collect-perf-detail", action="store_true")
    p.add_argument("--diagnostic-allow-fp32-rejected", action="store_true")
    args = p.parse_args(argv)
    board_guard()  # Before NumPy/SDK/library/data import or output creation.
    require(args.diagnostic_allow_fp32_rejected, "Explicit diagnostic flag; historical/trajectory rejection retained")
    out = args.output_dir.resolve()
    require(out.is_relative_to(BASE) and out != BASE and not out.exists(), "New isolated board experiment")
    protected = [v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"]
    require(not any(out == v or out.is_relative_to(v) or v.is_relative_to(out) for v in protected), "Source/output overlap")
    out.mkdir(parents=True, exist_ok=False)
    pins, activity, context = {}, {}, None
    prior = os.sched_getaffinity(0)
    start = time.monotonic()
    report = dict(schema="v39_DPVO_half_M16_encoder_SDK_diagnostic", status="failed", script_sha256=sha(__file__),
        collect_perf_detail=args.collect_perf_detail, normal_benchmark=not args.collect_perf_detail,
        compile_receipt_sha256=args.compile_sha256, compiler_sha256=args.compiler_sha256,
        probe_sha256=PROBE_SHA, policy_sha256=POLICY_SHA, original_FP32_gate=dict(rtol=1e-4, atol=1e-5),
        diagnostic_allow_fp32_rejected=True, candidate_accepted=False, historical_parent_FP32_gate_passed=False,
        historical_parent_rejection_retained=True, complete_VO=False, realtime_validated=False, GT_read=False,
        production_changed=False, CPU_fallback_used=False, all_internal_ops_NPU_verified=False,
        source_pins=pins, cleanup_errors=[], context={}, calls=[], activity=activity,
        timing_scope="sdk_io includes input copy/NHWC packing, set/run/get, owned output copy and release; excludes saved reference load/comparison, file save and perf query")
    def pin(path, expected):
        path = Path(path).resolve()
        require(isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected) and
                sha(path) == expected, "SHA mismatch: "+path.name)
        pins[str(path)] = expected
        return path
    def artifact(directory, row):
        require(Path(row["file"]).name == row["file"], "Relative simple artifact filename")
        path = pin(Path(directory)/row["file"], row["sha256"])
        require(path.stat().st_size == row["bytes"], "Artifact bytes mismatch")
        return path
    try:
        import numpy as np
        pin(__file__, report["script_sha256"])
        policy = json.loads(pin(args.policy, POLICY_SHA).read_text())
        require(policy["image_size_hw"] == list(INPUT_SHAPE[2:]) and policy["stream_config"]["patches_per_frame"] == 16, "Single declared half-M16 policy")
        probe = json.loads(pin(args.probe_dir/"probe.json", PROBE_SHA).read_text())
        rows = load_rows(args.probe_dir, probe, artifact)
        compiled = json.loads(pin(args.compile_dir/"compile.json", args.compile_sha256).read_text())
        model = artifact(args.compile_dir, compiled["output"])
        validate_compile(compiled, args.compiler_sha256, probe, sha(model), model.stat().st_size)
        report.update(model_sha256=sha(model), model_bytes=model.stat().st_size, source_graph_sha256=compiled["input"]["sha256"],
            preprocessing=policy["preprocessing"], new_size_CPU_strict_gate_each_method=probe["strict_gate_each_method"],
            input_contract=dict(normalized_bgr=list(INPUT_SHAPE), patch_indices=[16]),
            output_contract={k: list(s) for k, s in OUTPUTS}, references=[r["identity"] for r in rows])
        abi = load(pin(args.abi_helper, ABI_SHA), "_v39_frozen_ABI")
        validator = load(pin(args.query_helper, QUERY_SHA), "_v39_frozen_attribute_guard")
        transport = load(pin(args.transport_helper, TRANSPORT_SHA), "_v39_frozen_typed_transport")
        pin(args.header, HEADER_SHA);pin(args.library, LIBRARY_SHA)
        report["ABI_layout"] = abi.abi_record()
        os.sched_setaffinity(0, {4, 5, 6, 7})
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        api = abi.bind_api(ct.CDLL(str(args.library)))
        context = HalfContext(api, abi, validator, transport, model, args.collect_perf_detail, report["context"], report["cleanup_errors"])
        for index, (frame, phase) in enumerate(schedule(args.collect_perf_detail)):
            row = rows[frame]
            call = dict(call_index=index, frame_index=frame, phase=phase, frame_id=row["identity"]["frame_id"],
                source_index=row["identity"]["source_index"], source_image_sha256=row["identity"]["source_image_sha256"],
                input_array=metadata(row["image"]), indices=metadata(row["indices"]))
            report["calls"].append(call)
            values, timing = context.execute(row["image"], row["indices"], activity)
            call["timing"] = timing
            # Persist raw outputs before finite/allzero validation. Numeric
            # disagreement is recorded, not silently promoted or relaxed.
            target = out/f"call-{index:02d}-frame-{frame}.npz"
            with target.open("xb") as stream:
                np.savez(stream, **values)
            call["artifact"] = dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target),
                arrays={k: metadata(v) for k, v in values.items()})
            require(set(values) == {k for k, _ in OUTPUTS} and all(v.dtype == np.float32 and v.shape == s and
                    np.isfinite(v).all() and np.any(v != 0) for k, s in OUTPUTS for v in (values[k],)),
                    "Saved SDK output has wrong shape/dtype, nonfinite or allzero features")
            call["comparison"] = {m: {k: comparison(values[k], ref[k]) for k, _ in OUTPUTS} for m, ref in row["references"].items()}
            if args.collect_perf_detail:
                call["perf_detail"] = abi.perf_detail(api, context.context)
            print(json.dumps(dict(call_index=index, frame_index=frame, phase=phase, sdk_io_seconds=timing["sdk_io_seconds"])), flush=True)
        require(activity.get("sdk_run_api_calls") == context.completed_calls == len(schedule(args.collect_perf_detail)), "Actual SDK call/copy/release count mismatch")
        report["strict_gate_each_reference"] = {m: all(c["comparison"][m][k]["gate_passed"] for c in report["calls"] for k, _ in OUTPUTS) for m in rows[0]["references"]}
        report["strict_FP32_gate_passed"] = report["strict_gate_each_reference"]["original_CPU"]
        report["SDK_inference_executed"] = True
        warm = [c["timing"]["sdk_io_seconds"] for c in report["calls"] if c["phase"] == "warm"]
        report["warm_sdk_io_seconds"] = dict(samples=len(warm), mean=sum(warm)/len(warm) if warm else None)
        report["status"] = "completed_half_M16_SDK_diagnostic_not_VO_not_adopted"
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        if context is not None:
            context.close()
        try:
            os.sched_setaffinity(0, prior)
            report["affinity_restored"] = os.sched_getaffinity(0) == prior
            require(report["affinity_restored"], "Affinity restoration mismatch")
        except BaseException as exc:
            report["cleanup_errors"].append("affinity: "+repr(exc))
        report["source_hashes_after"] = {p: sha(p) if Path(p).is_file() else None for p in pins}
        report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
        report.update(actual_driver_seconds=time.monotonic()-start,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            sdk_run_api_calls=activity.get("sdk_run_api_calls", 0), SDK_run_api_called=activity.get("sdk_run_api_calls", 0) > 0)
        if not report["sources_and_inputs_unchanged"] or report["cleanup_errors"]:
            report["status"] = "failed"
        (out/"run.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "strict_gate_each_reference", "sdk_run_api_calls")}), flush=True)
    return 0 if report["status"] == "completed_half_M16_SDK_diagnostic_not_VO_not_adopted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
