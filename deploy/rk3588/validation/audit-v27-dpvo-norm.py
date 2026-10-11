#!/usr/bin/env python3
"""Independent saved-array audit of the v27 DPVO normalization rewrite.

NumPy and standard library only: no Torch, image decoding, ONNX/ORT execution,
compiler, or board access.  An audit can pass while the candidate FP32 gate
fails; that candidate must not proceed as a validated acceleration artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
EXPORT_SHA = "45fa589fa3a25a9722dc0832f5d14ffcec662e70d1dff135b86be641eeeba4b2"
OLD_ORT_SHA = "153da765b1145d7bb74bc635c1982ae74c652dac28cf99be56cb40534c44b667"
REWRITE_SHA = "66eb8d135165f7538c97733d0c83198fa5a674e0cd2bdf64eb190c24819c5109"
PARENT_GRAPH_SHA = "527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82"
NEW_GRAPH_SHA = "c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826"
EDITOR_SHA = "671ceca472d104cedc318b380cd6dfaeb8852e4207383bf533fd4c98d0937f11"
CHECKER_SHA = "f2ce1b906a58343925007ad4f3429f690ede07f5f230b4e2c2d7b743aa9e91cf"
RTOL, ATOL = 1e-4, 1e-5
SHAPES = {"fmap": (1, 1, 128, 116, 152), "imap": (1, 1, 384, 116, 152)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (
        _ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))


def metadata(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes,
            "tensor_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def compare(actual, reference):
    require(actual.dtype == reference.dtype == np.float32 and actual.shape == reference.shape
            and actual.size > 0 and np.isfinite(actual).all() and np.isfinite(reference).all(),
            "Matching nonempty finite FP32 arrays required")
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    difference = np.abs(a - b)
    tolerance = ATOL + RTOL * np.abs(b)
    failed = difference > tolerance
    delta, norm = (a - b).ravel(), float(np.linalg.norm(b.ravel()))
    return {"rtol": RTOL, "atol": ATOL, "failed_elements": int(failed.sum()),
            "element_count": actual.size, "gate_passed": not bool(failed.any()),
            "max_abs": float(difference.max()), "rms": float(np.sqrt(np.mean(delta * delta))),
            "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
            "maximum_tolerance_ratio": float((difference / tolerance).max()),
            "failed_reference_abs_range": [float(np.abs(b[failed]).min()), float(np.abs(b[failed]).max())] if failed.any() else None,
            "bitwise_equal": actual.tobytes(order="C") == reference.tobytes(order="C")}


def verify_reported_gate(saved, calculated):
    require(saved.get("rtol") == RTOL and saved.get("atol") == ATOL,
            "The unchanged original FP32 gate is required")
    for key in ("failed_elements", "gate_passed", "bitwise_equal"):
        require(saved.get(key) == calculated[key], "Saved gate field differs: " + key)
    for key in ("max_abs", "relative_l2"):
        actual, expected = saved.get(key), calculated[key]
        require(actual is None and expected is None or actual is not None and expected is not None
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-14),
                "Saved numeric field differs: " + key)


def require_gate_before_acceleration(receipt):
    require(receipt.get("status") == "completed_board_ORT_rewrite_gate_not_VO"
            and receipt.get("exact_FP32_gate_passed") is True,
            "Candidate failed the original FP32 gate; do not treat it as validated for compilation")
    calls = receipt.get("calls", [])
    require(len(calls) == 3, "Three numerical gate calls required")
    for call in calls:
        for field in ("versus_original_CPU", "versus_original_ORT"):
            require(set(call.get(field, {})) == set(SHAPES), "Both output gates are required")
            for gate in call[field].values():
                require(gate.get("rtol") == RTOL and gate.get("atol") == ATOL
                        and gate.get("gate_passed") is True and gate.get("failed_elements") == 0,
                        "A per-output original FP32 gate failed")


class Audit:
    def __init__(self):
        self.pins, self.checks = {}, 0

    def check(self, value, message):
        self.checks += 1
        require(value, message)

    def pin(self, path, expected):
        path = Path(path)
        self.check(isinstance(expected, str) and len(expected) == 64
                   and all(c in "0123456789abcdef" for c in expected), "Invalid SHA256 pin")
        self.check(sha(path) == expected, "File SHA mismatch: " + path.name)
        self.pins[path.resolve()] = expected
        return path

    def artifact(self, directory, item):
        name = item["file"]
        self.check(isinstance(name, str) and Path(name).name == name, "Simple artifact filename required")
        path = self.pin(directory / name, item["sha256"])
        self.check(path.stat().st_size == item["bytes"], "Artifact size differs")
        return path

    def features(self, directory, row):
        with np.load(self.artifact(directory, row["artifact"]), allow_pickle=False) as saved:
            self.check(set(saved.files) == set(SHAPES), "Exact fmap/imap NPZ keys required")
            result = {key: saved[key].copy() for key in SHAPES}
        for name, value in result.items():
            self.check(value.dtype == np.float32 and value.shape == SHAPES[name]
                       and np.isfinite(value).all() and np.any(value != 0), "Feature dtype/shape/finite/nonzero check failed")
            self.check(metadata(value) == row["arrays"][name], "Raw feature SHA/metadata mismatch")
        return result


def audit(args):
    check = Audit()
    expected = (CAPTURE_SHA, EXPORT_SHA, OLD_ORT_SHA, REWRITE_SHA, args.rewrite_ort_sha256)
    supplied = (args.capture_sha256, args.export_sha256, args.old_ort_sha256, args.rewrite_sha256, args.rewrite_ort_sha256)
    check.check(supplied == expected, "Frozen original and rewrite receipt identities differ")
    paths = [(args.capture_dir, "capture.json"), (args.export_dir, "export.json"),
             (args.old_ort_dir, "ort.json"), (args.rewrite_dir, "rewrite.json"), (args.rewrite_ort_dir, "ort.json")]
    captured, exported, old_ort, rewrite, new_ort = [read_json(check.pin(directory / filename, digest))
                                                   for (directory, filename), digest in zip(paths, expected)]
    check.check(captured["status"] == "completed_encoder_cpu_probe_not_VO"
                and exported["status"] == "exported_not_runtime_validated"
                and old_ort["status"] == "completed_CPU_ORT_gate"
                and old_ort["exact_reference_gate_passed"] is True, "Successful original CPU/export/ORT chain required")
    check.check(exported["capture_sha256"] == old_ort["capture_sha256"] == CAPTURE_SHA
                and old_ort["export_sha256"] == EXPORT_SHA
                and old_ort["onnx"] == exported["onnx"]
                and exported["onnx"]["sha256"] == rewrite["input"]["sha256"] == PARENT_GRAPH_SHA,
                "Original graph/source lineage differs")
    check.check(rewrite["script_sha256"] == EDITOR_SHA and rewrite["output"]["sha256"] == NEW_GRAPH_SHA
                and rewrite["status"] == "rewritten_and_statically_checked_not_runtime_validated"
                and rewrite["external_data_used"] is False, "Frozen static rewrite differs")
    for key in ("original_initializers_byte_exact", "non_instance_norm_nodes_byte_exact",
                "input_output_protobuf_byte_exact", "model_header_byte_exact"):
        check.check(rewrite["preservation"].get(key) is True, "Static preservation proof failed: " + key)
    check.artifact(args.export_dir, exported["onnx"])
    check.artifact(args.rewrite_dir, rewrite["output"])
    check.check(new_ort["schema"] == "v27_DPVO_InstanceNorm_rewrite_board_ORT"
                and new_ort["script_sha256"] == CHECKER_SHA
                and new_ort["actual_board_execution"] is True
                and new_ort["capture_sha256"] == CAPTURE_SHA and new_ort["export_sha256"] == EXPORT_SHA
                and new_ort["old_ort_sha256"] == OLD_ORT_SHA and new_ort["rewrite_sha256"] == REWRITE_SHA
                and new_ort["onnx"] == rewrite["output"], "Board rewrite ORT lineage differs")
    check.check(new_ort["providers"] == ["CPUExecutionProvider"]
                and new_ort["graph_optimization"] == "ORT_DISABLE_ALL"
                and new_ort["affinity_restored"] is True and not new_ort["cleanup_errors"],
                "Board ORT provider/affinity/cleanup differs")
    for receipt in (captured, exported, old_ort, new_ort):
        check.check(receipt["sources_and_inputs_unchanged"] is True
                    and receipt["source_hashes_after"] == receipt["source_pins"], "Recorded board source recheck differs")
    check.check(len(captured["inputs"]) == len(old_ort["pairs"]) == len(new_ort["calls"]) == 3,
                "Exactly three source frames required")
    frames, prior_timestamp = [], -math.inf
    for index, (seed, old_call, new_call) in enumerate(zip(captured["inputs"], old_ort["pairs"], new_ort["calls"])):
        frame_id = seed["frame_id"]
        check.check(seed["frame_index"] == old_call["frame_index"] == new_call["frame_index"] == index
                    and seed["source"]["frame_id"] == new_call["frame_id"] == frame_id
                    and isinstance(frame_id, str) and math.isfinite(float(frame_id))
                    and float(frame_id) > prior_timestamp, "Original source frame/order mismatch")
        prior_timestamp = float(frame_id)
        x = np.load(check.artifact(args.capture_dir, seed["input"]), allow_pickle=False)
        check.check(x.shape == (1, 1, 3, 464, 608) and x.dtype == np.float32
                    and np.isfinite(x).all() and metadata(x) == seed["array"], "Original normalized input differs")
        cpu_call = next(call for call in captured["calls"] if call["frame_index"] == index)
        check.check(cpu_call["frame_id"] == frame_id, "CPU reference frame identity differs")
        cpu = check.features(args.capture_dir, cpu_call)
        old = check.features(args.old_ort_dir, old_call)
        new = check.features(args.rewrite_ort_dir, new_call)
        checked_cpu, checked_old = {}, {}
        for name in SHAPES:
            original_gate = compare(old[name], cpu[name])
            verify_reported_gate(old_call["checks"][name], original_gate)
            check.check(original_gate["gate_passed"], "Original CPU ORT gate was not independently reproduced")
            checked_cpu[name], checked_old[name] = compare(new[name], cpu[name]), compare(new[name], old[name])
            verify_reported_gate(new_call["versus_original_CPU"][name], checked_cpu[name])
            verify_reported_gate(new_call["versus_original_ORT"][name], checked_old[name])
            check.check(True, "Saved numerical checks were independently reproduced")
        frames.append({"frame_index": index, "frame_id": frame_id, "input_array": seed["array"],
                       "source_image_sha256": seed["source"]["sha256"],
                       "artifact": new_call["artifact"], "actual_arrays": new_call["arrays"],
                       "versus_original_CPU": checked_cpu, "versus_original_ORT": checked_old,
                       "reported_board_ORT_forward_seconds": new_call["forward_seconds"]})
    passed = all(row[field][name]["gate_passed"] for row in frames
                 for field in ("versus_original_CPU", "versus_original_ORT") for name in SHAPES)
    check.check(type(new_ort["exact_FP32_gate_passed"]) is bool and new_ort["exact_FP32_gate_passed"] == passed,
                "Top-level original FP32 gate claim differs from actual arrays")
    if passed:
        require_gate_before_acceleration(new_ort)
        rejection = None
    else:
        check.check(new_ort["status"] == "failed" and "Original FP32 gate failed" in new_ort.get("error", ""),
                    "Numerical failure was not recorded as a failed board gate")
        try:
            require_gate_before_acceleration(new_ort)
        except ValueError as error:
            rejection = str(error)
        else:
            raise ValueError("Failed real receipt incorrectly accepted as validated")
    for path, digest in check.pins.items():
        check.check(sha(path) == digest, "Local raw receipt/artifact changed during audit")
    totals = {field: {name: {"failed_elements": sum(row[field][name]["failed_elements"] for row in frames),
                            "element_count": sum(row[field][name]["element_count"] for row in frames)}
                      for name in SHAPES} for field in ("versus_original_CPU", "versus_original_ORT")}
    return {"kind": "v27_DPVO_norm_rewrite_independent_saved_array_audit", "status": "passed",
            "audit_script_sha256": sha(__file__), "checks": check.checks, "audit_failures": 0,
            "receipt_sha256": dict(zip(("capture", "export", "original_ORT", "rewrite", "rewritten_board_ORT"), expected)),
            "parent_graph_sha256": PARENT_GRAPH_SHA, "candidate_graph_sha256": NEW_GRAPH_SHA,
            "candidate_original_FP32_gate_passed": passed, "eligible_for_validated_compilation": passed,
            "validation_rejection": rejection, "rtol": RTOL, "atol": ATOL, "thresholds_relaxed": False,
            "all_arrays_finite_and_nonzero": True, "all_local_artifact_SHAs_and_raw_metadata_verified": True,
            "original_CPU_ORT_gate_independently_reproduced": True,
            "unmodified_imap_vs_original_ORT_bitwise_equal_all_three_frames": all(row["versus_original_ORT"]["imap"]["bitwise_equal"] for row in frames),
            "total_errors": totals, "frames": frames,
            "board_cleanup": {"cleanup_errors": new_ort["cleanup_errors"], "affinity_restored": new_ort["affinity_restored"]},
            "scope": {"saved_arrays_only": True, "host_model_or_ORT_execution": False, "image_decoding": False,
                      "board_jobs_started": False, "candidate_compilation_performed": False},
            "limitations": ["Audit passed means saved evidence is consistent, not that the candidate passed its numerical gate.",
                            "Original board source rechecks are recorded evidence; board-only paths and runtime libraries were not rehashed on the host.",
                            "Small output errors do not waive the unchanged elementwise gate.",
                            "No staged normalization activations were captured; output differences alone do not identify an individual kernel's rounding contribution.",
                            "Encoder-only fixed inputs do not establish VO accuracy, speed, or operator placement."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "export", "old-ort", "rewrite", "rewrite-ort"):
        parser.add_argument("--" + name + "-dir", type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Audit output must be new")
    report = audit(args)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "checks", "candidate_original_FP32_gate_passed", "total_errors")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
