#!/usr/bin/env python3
"""Board-only sparse encoder PTQ inference diagnostic; never complete VO.

MIT project helper; DPVO is MIT licensed. SDK arithmetic delegates to the
frozen v34 typed transport. This entrypoint accepts the actual INT8 descriptors
instead of changing the historical v34 floating-output context guard.
Normal: frame0 cold+3 warm, frame1/2 once. Profiling: 3 calls in a separate run.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
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
QUERY_SOURCE_SHA = "871f67c5a37ff23b703d173bb09234504a58fb16b8ace65e57991afe6f7e3d90"
TRANSPORT_SHA = "646f875d1b5eea561c0d6aa4d1b9078351014a39a62976ec895a814cd466b387"
ABI_SHA = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"
HEADER_SHA = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
LIBRARY_SHA = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
OUTPUTS = (("fmap", (1, 128, 116, 152)), ("context", (1, 96, 384)))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def comparison(actual, expected):
    import numpy as np
    require(actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape,
            "Exact canonical FP32 output/reference contract")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite comparison inputs")
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    difference = a - b
    failures = int(np.count_nonzero(np.abs(difference) > 1e-5 + 1e-4*np.abs(b)))
    norm, actual_norm = float(np.linalg.norm(b)), float(np.linalg.norm(a))
    error_norm = float(np.linalg.norm(difference))
    return dict(rtol=1e-4, atol=1e-5, failed_elements=failures, element_count=int(a.size), gate_passed=failures == 0,
        bitwise_equal=actual.tobytes(order="C") == expected.tobytes(order="C"), array_equal=bool(np.array_equal(actual, expected)),
        max_abs=float(np.max(np.abs(difference))), rmse=float(np.sqrt(np.mean(difference*difference))),
        relative_l2=error_norm/norm if norm else 0.0 if error_norm == 0 else None,
        cosine=float(np.dot(a/actual_norm, b/norm)) if norm and actual_norm else None,
        finite=True, actual_all_zero=bool(np.all(actual == 0)))


def schedule(profile):
    return [(i, "profile") for i in range(3)] if profile else [(0, "cold")] + [(0, "warm")]*3 + [(1, "other_frame"), (2, "other_frame")]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def board_guard():
    require(platform.machine() == "aarch64" and platform.system() == "Linux", "Linux aarch64 RK3588/S board only")
    compatible = Path("/proc/device-tree/compatible").read_bytes().split(b"\0")
    require(b"rockchip,rk3588" in compatible or b"rockchip,rk3588s" in compatible, "RK3588/S device identity required")


def pack_input(images, fmt):
    import numpy as np
    require(images.dtype == np.float32 and images.shape == (1, 3, 464, 608), "Original once-normalized FP32 input required")
    require(fmt in (0, 1), "No packed/ambiguous image layout")
    return np.array(images if fmt == 0 else images.transpose(0, 2, 3, 1), copy=True, order="C")


def validate_query(value, compile_sha, model_sha):
    require(value["schema"] == "v36_DPVO_board_PTQ_native_IO_query" and
            value["status"] == "queried_native_IO_not_inference_validated" and value["script_sha256"] == QUERY_SOURCE_SHA,
            "Successful frozen zero-run native IO query required")
    require(value["compile_receipt_sha256"] == compile_sha and value["model_sha256"] == model_sha,
            "Query must bind this PTQ compile/model")
    require(value["sdk_run_calls"] == 0 and value["model_inference_executed"] is False and value["native_IO_contract_checked"] is True,
            "Zero-run query contract")
    require(value["context_destroyed"] is True and value["cleanup_errors"] == [] and
            value["sources_and_inputs_unchanged"] is True and value["source_pins"] == value["source_hashes_after"], "Clean zero-run query provenance")


class PTQContext:
    def __init__(self, api, abi, query_helper, proof, model, profile, report):
        self.api, self.abi, self.record, self.cleanup = api, abi, report["context"], report["cleanup_errors"]
        self.context, self.closed = abi.CONTEXT(), False
        try:
            filename = ct.create_string_buffer(os.fsencode(model))
            started = time.perf_counter()
            abi.check_call(api.rknn_init(ct.byref(self.context), ct.cast(filename, ct.c_void_p), 0, 8 if profile else 0, None), "rknn_init")
            require(self.context.value != 0, "Null successful context")
            self.record["init_seconds"] = time.perf_counter() - started
            abi.check_call(api.rknn_set_core_mask(self.context, 7), "rknn_set_core_mask")
            self.record["core_mask"] = 7
            version = abi.query(api, self.context, 5, abi.SDKVersion())
            self.record["sdk_versions"] = dict(api=bytes(version.api_version).decode(errors="replace"),
                                              driver=bytes(version.drv_version).decode(errors="replace"))
            require(self.record["sdk_versions"] == proof["sdk_versions"], "Runtime/driver changed since zero-run query")
            counts = abi.query(api, self.context, 0, abi.InputOutputNum())
            require(counts.n_input == counts.n_output == 2, "Exact sparse input/output count")
            self.record["input_attrs"] = [abi.attr_record(abi.query(api, self.context, 1, abi.TensorAttr(index=i))) for i in range(2)]
            self.record["output_attrs"] = [abi.attr_record(abi.query(api, self.context, 2, abi.TensorAttr(index=i))) for i in range(2)]
            require(self.record["input_attrs"] == proof["input_attrs"] and self.record["output_attrs"] == proof["output_attrs"],
                    "Actual native ports differ from pinned query; no dtype/layout fallback")
            for row, idx, name, shape, options in [
                (self.record["input_attrs"][0], 0, "normalized_bgr", (1, 3, 464, 608), dict(spatial=True)),
                (self.record["input_attrs"][1], 1, "patch_indices", (96,), dict(integer=True)),
                (self.record["output_attrs"][0], 0, "fmap", OUTPUTS[0][1], dict(spatial=True)),
                (self.record["output_attrs"][1], 1, "context", OUTPUTS[1][1], {})]:
                query_helper.validate_attr(row, idx, name, shape, **options)
            require(all(r["type"] == 2 for r in [self.record["input_attrs"][0], *self.record["output_attrs"]]) and
                    self.record["input_attrs"][0]["format"] == 1 and self.record["output_attrs"][0]["format"] == 0 and
                    self.record["output_attrs"][1]["format"] == 3, "This version implements actual INT8 NHWC input / NCHW+undefined outputs only")
            self.index_attr = self.record["input_attrs"][1]
            memory = abi.query(api, self.context, 6, abi.MemSize())
            self.record["model_memory_bytes"] = {key: int(getattr(memory, key)) for key, _ in memory._fields_ if key != "reserved"}
            require(memory.total_dma_allocated_size <= 1024**3, "Context exceeds isolated 1GiB DMA budget")
        except Exception:
            self.close()
            raise

    def execute(self, transport, image, indices, activity):
        require(not self.closed, "Closed PTQ context")
        return transport.execute_sparse(self.api, self.abi, SimpleNamespace(pack_input=pack_input),
            self.context, image, indices, 1, self.index_attr, self.cleanup, activity,
            outputs=OUTPUTS, timeout_ms=60000)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.context.value:
            try:
                code = int(self.api.rknn_destroy(self.context))
                self.record["destroy_return_code"], self.record["context_destroyed"] = code, code == 0
                self.abi.check_call(code, "rknn_destroy")
            except Exception as exc:
                self.record["context_destroyed"] = False
                self.cleanup.append("rknn_destroy: " + repr(exc))
        else:
            self.record["context_destroyed"] = None


def load_rows(args, pin, artifact):
    import numpy as np
    require(args.capture_sha256 == CAPTURE_SHA and args.reference_sha256 == REFERENCE_SHA, "Frozen original CPU/ORT reference receipts")
    capture = json.loads(pin(args.capture_dir / "capture.json", CAPTURE_SHA).read_text())
    reference = json.loads(pin(args.reference_dir / "ort.json", REFERENCE_SHA).read_text())
    require(capture["status"] == "completed_encoder_cpu_probe_not_VO" and capture["state_unchanged"] is True and
            capture["cleanup_errors"] == [] and capture["sources_and_inputs_unchanged"] is True and
            capture["source_pins"] == capture["source_hashes_after"], "Original CPU capture/source contract")
    require(reference["status"] == "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO" and
            reference["original_CPU_strict_gate_each_method"]["original"] is True and reference["suffix_bitwise_gate_passed"] is True and
            reference["cleanup_errors"] == [] and reference["sources_and_inputs_unchanged"] is True and
            reference["source_pins"] == reference["source_hashes_after"], "Original sparse CPU ORT reference/source contract")
    require(capture["preprocessing"] == reference["preprocessing"], "Exact original preprocessing identity")
    for record in (capture, reference):
        for path, digest in record["source_pins"].items():
            pin(Path(path), digest)  # Rehash existing source bytes; never decode them.
    rows = [r for r in reference["calls"] if r["method"] == "original"]
    require([r["frame_index"] for r in rows] == [0, 1, 2] and len(capture["inputs"]) == 3, "All three fixed evaluation inputs")
    result = []
    for i, row in enumerate(rows):
        source = capture["inputs"][i]
        require(row["frame_id"] == source["frame_id"] and row["source_image"] == source["source"], "Same original RGB identity")
        with np.load(artifact(args.reference_dir, row["input_artifact"]), allow_pickle=False) as archive:
            require(set(archive.files) == {"normalized_bgr", "centers_xy", "patch_indices"}, "Input archive keys")
            image, centers, indices = (archive[name] for name in ("normalized_bgr", "centers_xy", "patch_indices"))
        require(metadata(image) == row["input_metadata"] and metadata(centers) == row["centers"] and metadata(indices) == row["indices"], "Exact saved input/index raw bytes")
        x5 = np.load(artifact(args.capture_dir, source["input"]), allow_pickle=False)
        require(metadata(x5) == source["array"] and image.dtype == np.float32 and image.shape == (1, 3, 464, 608) and
                x5.shape == (1, 1, 3, 464, 608) and image.tobytes(order="C") == x5.tobytes(order="C"), "Byte-exact original normalized input, no reprocessing")
        require(np.isfinite(image).all() and -.500001 <= float(image.min()) <= float(image.max()) <= 1.500001,
                "Finite once-normalized source pixels")
        require(centers.dtype == indices.dtype == np.int64 and centers.shape == (96, 2) and indices.shape == (96,) and
                np.array_equal(indices, centers[:, 1]*np.int64(152) + centers[:, 0]), "Untouched ordered original integer indices")
        y, x = np.divmod(indices, 152)
        require(np.all((x >= 1) & (x <= 150) & (y >= 1) & (y <= 114)), "Original patch center domain")
        cpu_row = next(r for r in capture["calls"] if r["frame_index"] == i)
        with np.load(artifact(args.capture_dir, cpu_row["artifact"]), allow_pickle=False) as archive:
            dense = {name: archive[name] for name in ("fmap", "imap")}
        require(all(metadata(value) == cpu_row["arrays"][name] for name, value in dense.items()), "Original CPU feature raw bytes")
        require(dense["fmap"].shape == (1, 1, 128, 116, 152) and dense["imap"].shape == (1, 1, 384, 116, 152), "Original CPU singleton view axes")
        cpu = dict(fmap=np.ascontiguousarray(dense["fmap"][:, 0]), context=np.ascontiguousarray(dense["imap"][0, 0][:, y, x].T[None]))
        with np.load(artifact(args.reference_dir, row["artifact"]), allow_pickle=False) as archive:
            ort = {name: archive[name] for name, _ in OUTPUTS}
        require(all(metadata(value) == row["arrays"][name] for name, value in ort.items()), "Original sparse ORT raw feature bytes")
        require(all(comparison(ort[k], cpu[k])["gate_passed"] for k in cpu), "Reference still passes original CPU precision gate")
        result.append(dict(identity=row, image=image, indices=indices, cpu=cpu, ort=ort))
    return result, capture, reference


def load_historical_fp16(directory, digest, rows, pin, artifact):
    import numpy as np
    value = json.loads(pin(directory / "run.json", digest).read_text())
    require(value["schema"] == "v34_DPVO_sparse_context_RKNN_comparison" and
            value["status"] == "completed_sparse_context_SDK_comparison_not_VO" and value["method"] == "original" and
            value["collect_perf_detail"] is False and value["reference_sha256"] == REFERENCE_SHA and
            value["cleanup_errors"] == [] and value["sources_and_inputs_unchanged"] is True and
            value["source_pins"] == value["source_hashes_after"], "Completed matching original FP16 ordinary run required")
    saved = []
    for i, source in enumerate(rows):
        row = next(r for r in value["calls"] if r["frame_index"] == i and r["role"] == "sparse")
        require(row["frame_id"] == source["identity"]["frame_id"] and row["source_image"] == source["identity"]["source_image"] and
                row["input_array"] == source["identity"]["input_metadata"] and row["indices"] == source["identity"]["indices"],
                "Historical FP16 must use identical image/index bytes and identities")
        with np.load(artifact(directory, row["artifact"]), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name, _ in OUTPUTS}
        require(all(metadata(v) == row["selected_arrays"][name] for name, v in arrays.items()), "Historical FP16 saved output raw bytes")
        saved.append(arrays)
    return saved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("compile-dir", "query-dir", "query-helper", "capture-dir", "reference-dir", "transport-helper", "abi-helper", "header", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("compile", "query", "capture", "reference"):
        p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--library", type=Path, default=Path("/usr/lib/librknnrt.so"))
    p.add_argument("--fp16-run-dir", type=Path, help="Optional frozen original v34 ordinary saved-array comparison")
    p.add_argument("--fp16-run-sha256")
    p.add_argument("--collect-perf-detail", action="store_true")
    a = p.parse_args(argv)
    board_guard()  # Before NumPy, source import, model/library bytes or directory creation.
    require((a.fp16_run_dir is None) == (a.fp16_run_sha256 is None), "Both historical FP16 arguments required together")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import numpy as np
    output = a.output_dir.resolve()
    require(output.is_relative_to(BASE) and output != BASE, "Isolated project experiment output")
    require(not any(output.is_relative_to(path.resolve()) for path in
        (a.compile_dir, a.query_dir, a.capture_dir, a.reference_dir)), "Output must not overlap input directories")
    output.mkdir(parents=True, exist_ok=False)
    prior, pins, context, abi = os.sched_getaffinity(0), {}, None, None
    started = time.monotonic()
    report = dict(schema="v36_DPVO_sparse_encoder_PTQ_inference_diagnostic", status="failed", script_sha256=sha(__file__),
        compile_receipt_sha256=a.compile_sha256, query_receipt_sha256=a.query_sha256,
        capture_sha256=a.capture_sha256, reference_sha256=a.reference_sha256,
        collect_perf_detail=a.collect_perf_detail, context={}, calls=[], cleanup_errors=[], sdk_run_api_calls=0,
        source_pins=pins, actual_board_execution=True, NPU_used=False, candidate_accepted=False,
        production_changed=False, original_VGGT_default_unchanged=True, different_model_from_VGGT=True, ORB_used=False,
        original_FP32_source_graph=True, PTQ_changes_numeric_execution=True, complete_VO=False, realtime_validated=False,
        Update_executed=False, correlation_executed=False, BA_executed=False, images_decoded=False, normalization_applications=0,
        original_FP32_gate=dict(rtol=1e-4, atol=1e-5), output_clipping=False, manual_input_quantization=False,
        INT8_internal_operator_coverage_verified=False, execution_arithmetic_delegate_sha256=TRANSPORT_SHA,
        expected_output_FP32_logical_bytes=9175040,
        timing_scope="FP32 NHWC/INT64 packing + SDK set/run/get + owned logical FP32 copies + release; comparisons/hashes/saving/profile query excluded",
        limitations=["Sparse encoder component diagnostic on three fixed saved inputs; not a complete fresh-frame trajectory or speed/quality acceptance.",
            "Native INT8 ports do not prove internal INT8 coverage; separate actual SDK profile is required.",
            "Strict FP32 gate remains unchanged and is reported even when PTQ fails it; finite/all-zero transport failures are fatal."])

    def pin(path, digest):
        path = Path(path).resolve()
        require(sha(path) == digest and (str(path) not in pins or pins[str(path)] == digest), "Pinned source/input SHA differs: " + path.name)
        pins[str(path)] = digest
        return path

    def artifact(directory, item):
        require(Path(item["file"]).name == item["file"], "Simple artifact name required")
        path = pin(directory / item["file"], item["sha256"])
        require(path.stat().st_size == item["bytes"], "Artifact bytes differ")
        return path

    try:
        pin(__file__, report["script_sha256"])
        require({4, 5, 6, 7} <= prior, "Four A76 CPUs required")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        query_helper = load(pin(a.query_helper, QUERY_SOURCE_SHA), "_v36_pinned_query")
        transport = load(pin(a.transport_helper, TRANSPORT_SHA), "_v36_frozen_typed_transport")
        abi = load(pin(a.abi_helper, ABI_SHA), "_v36_pinned_abi")
        pin(a.header, HEADER_SHA)
        pin(a.library, LIBRARY_SHA)
        compiled = json.loads(pin(a.compile_dir / "compile.json", a.compile_sha256).read_text())
        model = artifact(a.compile_dir, compiled["output"])
        query_helper.validate_compile(compiled, compiled["output"]["sha256"], model.stat().st_size)
        proof = json.loads(pin(a.query_dir / "query.json", a.query_sha256).read_text())
        validate_query(proof, a.compile_sha256, compiled["output"]["sha256"])
        rows, capture, reference = load_rows(a, pin, artifact)
        historical = None if a.fp16_run_dir is None else load_historical_fp16(a.fp16_run_dir, a.fp16_run_sha256, rows, pin, artifact)
        report.update(preprocessing=capture["preprocessing"], inputs=[r["identity"]["input_metadata"] for r in rows],
            source_frames=[r["identity"]["source_image"] for r in rows], sampling_rule=reference["sampling_rule"],
            model=compiled["output"], configuration=compiled["configuration"], abi=abi.abi_record())
        report["historical_FP16_receipt_sha256"] = a.fp16_run_sha256
        require(report["abi"]["context_size"] == report["abi"]["pointer_size"] == 8, "Pinned 64-bit ABI required")
        api = abi.bind_api(ct.CDLL(str(a.library.resolve())))
        context = PTQContext(api, abi, query_helper, proof, model, a.collect_perf_detail, report)
        fingerprints = {}
        for index, (frame, role) in enumerate(schedule(a.collect_perf_detail)):
            row = rows[frame]
            actual, timing = context.execute(transport, row["image"], row["indices"], report)
            report["NPU_used"] = True
            finite = bool(all(np.isfinite(v).all() and np.any(v != 0) for v in actual.values()))
            record = dict(index=index, frame_index=frame, frame_id=row["identity"]["frame_id"], timing_role=role,
                source_image=row["identity"]["source_image"], input_metadata=row["identity"]["input_metadata"],
                indices=row["identity"]["indices"], timing=timing, forward_seconds=timing["sdk_io_seconds"],
                arrays={name: metadata(value) for name, value in actual.items()}, finite_nonzero=finite,
                finite_counts={name: dict(nonfinite=int(np.count_nonzero(~np.isfinite(value))), all_zero=bool(np.all(value == 0)))
                               for name, value in actual.items()})
            path = output / f"call-{index:02d}-frame-{frame}.npz"
            with path.open("xb") as stream:
                np.savez(stream, **actual)
            record["artifact"] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))
            report["calls"].append(record)
            require(finite, "Nonfinite/all-zero SDK output; raw arrays saved")
            record["versus_original_CPU"] = {name: comparison(value, row["cpu"][name]) for name, value in actual.items()}
            record["versus_v33_CPU_ORT"] = {name: comparison(value, row["ort"][name]) for name, value in actual.items()}
            if historical is not None:
                record["versus_historical_FP16"] = {name: comparison(value, historical[frame][name]) for name, value in actual.items()}
            record["repeat_raw_hash_equal"] = None if frame not in fingerprints else fingerprints[frame] == record["arrays"]
            fingerprints.setdefault(frame, record["arrays"])
            if a.collect_perf_detail:
                record["perf_detail"] = abi.perf_detail(api, context.context)
            print(json.dumps(dict(event="PTQ_encoder_call", index=index, frame=frame, seconds=record["forward_seconds"],
                strict=all(v["gate_passed"] for v in record["versus_original_CPU"].values()))), flush=True)
        require(len(report["calls"]) == len(schedule(a.collect_perf_detail)), "Complete prescribed schedule")
        report["actual_original_CPU_strict_gate_passed"] = all(v["gate_passed"] for r in report["calls"] for v in r["versus_original_CPU"].values())
        warm = [r["forward_seconds"] for r in report["calls"] if r["timing_role"] == "warm"]
        report["timing_summary"] = dict(cold_seconds=None if a.collect_perf_detail else report["calls"][0]["forward_seconds"],
            warm_seconds=warm, warm_mean_seconds=sum(warm)/len(warm) if warm else None,
            profile_timings_not_normal_benchmark=a.collect_perf_detail)
        report["status"] = "completed_encoder_PTQ_diagnostic_not_VO"
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        if context is not None:
            context.close()
        try:
            report["source_hashes_after"] = {path: sha(path) for path in pins}
            report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
            require(report["sources_and_inputs_unchanged"], "Pinned source/input changed")
        except Exception as exc:
            report["cleanup_errors"].append("source rehash: " + repr(exc))
        try:
            os.sched_setaffinity(0, prior)
            report["affinity_restored"] = os.sched_getaffinity(0) == prior
            require(report["affinity_restored"], "Affinity not restored")
        except Exception as exc:
            report["cleanup_errors"].append("affinity: " + repr(exc))
        if report["cleanup_errors"]:
            report["status"] = "failed"
        report["actual_driver_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "actual_original_CPU_strict_gate_passed", "timing_summary")}), flush=True)
    return 0 if report["status"] == "completed_encoder_PTQ_diagnostic_not_VO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
