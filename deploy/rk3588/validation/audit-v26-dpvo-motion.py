#!/usr/bin/env python3
"""Saved-array-only DPVO motion-probe audit. NumPy; no Torch/CUDA/model calls.

The correlation oracle uses independent scalar float64 channel sums, FP32
coordinates/pooling, zero padding and the original x/y/patch/level ordering.
Its fixed FP32 arithmetic gate is rtol=1e-4, atol=1e-5. This is not a CUDA
bitwise equivalence test or a VO accuracy/speed acceptance test.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
RTOL, ATOL = 1e-4, 1e-5
COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
PINS = {
    "dpvo_cpu_ops.py": "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32",
    "dpvo_probe_common.py": "ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9",
    "run-v26-dpvo-motion-board.py": "d266bdd513faf7c180688ec6724a3219afd81b8782b6c5b48c46c46164e833ae",
    "run-v26-dpvo-encoders-board.py": "e2f56fac7a6432761bf67ea2703a7b72b9d695577302d8f02d5cc8af0c0db9a5",
    "manifest.json": "ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375",
    "input-manifest.json": "bffcd221ee1e7c4b17045bf9b642df7272747ee000e67d5e682610b635804539",
}
CHECKPOINT_SHA = "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_metadata(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.nbytes, "tensor_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def compare(actual, expected):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise ValueError("Comparison shape/nonfinite error")
    error = np.abs(a - e)
    denominator = float(np.linalg.norm(e.ravel()))
    return {"elements": int(a.size), "max_abs": float(error.max(initial=0)),
            "RMSE": float(np.sqrt(np.mean((a-e)**2))) if a.size else 0.0,
            "relative_L2": float(np.linalg.norm((a-e).ravel()) / denominator) if denominator else None,
            "outside_fixed_FP32_gate": int(np.count_nonzero(error > ATOL + RTOL*np.abs(e))),
            "rtol": RTOL, "atol": ATOL}


class Checks:
    def __init__(self):
        self.count, self.failures = 0, []

    def require(self, condition, name):
        self.count += 1
        if not bool(condition):
            self.failures.append(name)


def safe_file(directory, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Artifact name must be a local basename")
    path = Path(directory) / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing/linked artifact: {name}")
    return path


def artifact(directory, record, checks, evidence, arrays=None):
    path = safe_file(directory, record["file"])
    actual_sha = sha256(path)
    checks.require(path.stat().st_size == record["bytes"] and actual_sha == record["sha256"],
                   f"artifact_file:{path.name}")
    evidence[path] = actual_sha
    metadata = record.get("arrays", arrays)
    if metadata is None:
        return None
    with np.load(path, allow_pickle=False) as source:
        checks.require(set(source.files) == set(metadata), f"artifact_keys:{path.name}")
        result = {key: source[key] for key in source.files}
    for key, value in result.items():
        checks.require(raw_metadata(value) == metadata.get(key), f"raw_metadata:{path.name}:{key}")
        checks.require(np.isfinite(value).all(), f"finite:{path.name}:{key}")
    return result


def mean_pool4(fmap):
    """Independent explicit row-major FP32 sum, matching average-pool scope."""
    height, width = fmap.shape[-2:]
    h, w = height // 4, width // 4
    total = np.zeros(fmap.shape[:-2] + (h, w), np.float32)
    for y in range(4):
        for x in range(4):
            total += fmap[..., y:4*h:4, x:4*w:4]
    return total / np.float32(16)


def correlation_oracle(gmap, fmap, coords, patch_ids, target_ids, edge_ids):
    """Compute all 882 values of each predeclared selected edge, with loops."""
    pyramid = (fmap, mean_pool4(fmap))
    result = np.zeros((len(edge_ids), 7, 7, 3, 3, 2), np.float64)
    for row, edge in enumerate(edge_ids):
        patch = gmap[0, int(patch_ids[edge])].astype(np.float64)
        view = int(target_ids[edge])
        for level_index, (level, features) in enumerate(zip((1, 4), pyramid)):
            target = features[0, view]
            height, width = target.shape[-2:]
            xy = coords[0, edge] / np.float32(level)
            for py in range(3):
                for px in range(3):
                    x, y = xy[:, py, px]
                    ix, iy = int(np.floor(x)), int(np.floor(y))
                    dx = float(np.float32(x - np.floor(x)))
                    dy = float(np.float32(y - np.floor(y)))
                    vector = patch[:, py, px]
                    for ox in range(7):
                        for oy in range(7):
                            value = 0.0
                            for sy, sx, weight in ((0,0,(1-dy)*(1-dx)), (0,1,(1-dy)*dx),
                                                   (1,0,dy*(1-dx)), (1,1,dy*dx)):
                                tx, ty = ix+ox-3+sx, iy+oy-3+sy
                                if weight and 0 <= tx < width and 0 <= ty < height:
                                    value += weight * float(np.sum(vector * target[:, ty, tx], dtype=np.float64))
                            result[row, ox, oy, py, px, level_index] = value
    return result.reshape(len(edge_ids), 882)


def _close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def _pin_by_suffix(pins, suffix):
    matches = [value for name, value in pins.items() if name.endswith("/"+suffix)]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous/missing recorded pin: {suffix}")
    return matches[0]


def audit_run(run_dir, capture_dir, source_dir, checkpoint):
    run_dir, capture_dir, source_dir = map(Path, (run_dir, capture_dir, source_dir))
    checks, evidence = Checks(), {}
    def read_json(path):
        path = Path(path)
        evidence[path] = sha256(path)
        return json.loads(path.read_text())
    run = read_json(run_dir / "motion.json")
    capture = read_json(capture_dir / "capture.json")
    manifest = read_json(source_dir / "manifest.json")
    inputs = read_json(source_dir / "input-manifest.json")
    additions = read_json(source_dir / "source_manifest_additions.json")
    evidence[Path(__file__)] = sha256(__file__)
    checks.require(run["status"] == "completed_dynamic_neural_probe_not_VO", "motion_status")
    checks.require(capture["status"] == "completed_encoder_cpu_probe_not_VO", "capture_status")
    checks.require(run["schema"] == "v26_DPVO_fixed_identity_dynamic_neural_probe", "motion_schema")
    for key in ("original_DPVO_weights", "actual_board_execution", "original_weights_unchanged",
                "sources_and_inputs_unchanged", "affinity_restored", "repeat_outputs_bitwise_equal"):
        checks.require(run.get(key) is True, "true_claim:"+key)
    for key in ("NPU_used", "CUDA_used", "BA_executed", "tracked_trajectory", "valid_pose_output",
                "VO_accuracy_validated", "realtime_validated", "original_VGGT_weights_used", "ORB_used",
                "production_changed", "CNN_capture_included_in_dynamic_timing"):
        checks.require(run.get(key) is False, "false_claim:"+key)
    checks.require(run["cleanup_errors"] == [], "cleanup_errors_empty")
    checks.require(run["machine"] == "aarch64" and run["cpu_affinity"] == [4,5,6,7]
                   and run["torch_threads"] == 4 and run["torch_interop_threads"] == 1, "execution_platform")
    checks.require(run["capture_sha256"] == evidence[capture_dir/"capture.json"], "capture_receipt_binding")
    checks.require(run["source_hashes_after"] == run["source_and_input_hashes"], "recorded_before_after_pins")
    checks.require(capture["source_hashes_after"] == capture["source_pins"], "capture_recorded_before_after_pins")
    for name, pinned in PINS.items():
        path = source_dir / name
        actual = sha256(path)
        evidence[path] = actual
        checks.require(actual == pinned == _pin_by_suffix(run["source_and_input_hashes"], name), "source_pin:"+name)
    checks.require(run["script_sha256"] == PINS["run-v26-dpvo-motion-board.py"], "motion_entrypoint_pin")
    checks.require(capture["script_sha256"] == PINS["run-v26-dpvo-encoders-board.py"], "encoder_entrypoint_pin")
    checks.require(manifest["repo"] == "princeton-vl/DPVO" and manifest["commit"] == COMMIT
                   and additions["commit"] == COMMIT and additions["parent_manifest_sha256"] == PINS["manifest.json"], "official_commit")
    for row in manifest["retrieved_files"] + additions["retrieved_files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe official-source manifest path")
        path = source_dir / "official" / relative
        if row.get("available_in_pinned_tree") is False:
            checks.require(not path.exists(), "official_declared_absent:"+row["path"])
            continue
        evidence[path] = sha256(path)
        checks.require(path.stat().st_size == row["bytes"] and evidence[path] == row["sha256"], "official_source:"+row["path"])
        checks.require(hashlib.sha1(b"blob "+str(path.stat().st_size).encode()+b"\0"+path.read_bytes()).hexdigest()
                       == row["git_blob_sha1"], "official_git_blob:"+row["path"])
    checkpoint = Path(checkpoint)
    evidence[checkpoint] = sha256(checkpoint)
    checks.require(evidence[checkpoint] == CHECKPOINT_SHA == capture["checkpoint_sha256"]
                   == run["loader"]["checkpoint"]["sha256"], "checkpoint_sha256")
    checks.require(run["loader"]["commit"] == COMMIT and run["loader"]["original_Update_class_AST_preserved"] is True
                   and run["loader"]["load_state"]["strict"] is True
                   and run["loader"]["load_state"]["inference_parameter_coverage"] == 1.0, "strict_original_Update_loader")
    weights = run["original_weights_before"]
    loaded = run["loader"]["load_state"]["loaded_tensors"]
    checks.require(len(weights) == 94 and set(weights) == set(loaded) == set(capture["state_before"]), "94_inference_state_keys")
    for key, value in weights.items():
        checks.require(value == capture["state_before"][key]
                       and all(value[k] == loaded[key][k] for k in ("shape", "bytes", "tensor_sha256")), "weight_metadata:"+key)
    checks.require(sum(math.prod(row["shape"]) for row in weights.values()) == 3384324, "inference_parameters")
    checks.require(run["frame_ids"] == [row["frame_id"] for row in inputs["frames"]]
                   and np.all(np.diff([float(x) for x in run["frame_ids"]]) > 0), "exact_three_source_timestamps")
    checks.require(_pin_by_suffix(run["source_and_input_hashes"], "rgb.txt") == inputs["rgb_index_sha256"], "rgb_index_recorded_binding")
    for row in inputs["frames"]:
        checks.require(_pin_by_suffix(run["source_and_input_hashes"], row["relative_path"]) == row["sha256"], "source_PNG_recorded_binding:"+row["frame_id"])

    features, contexts = [], []
    for frame, call_index in enumerate((0, 4, 5)):
        row = capture["calls"][call_index]
        value = artifact(capture_dir, row["artifact"], checks, evidence, row["arrays"])
        checks.require(row["frame_index"] == frame and row["frame_id"] == run["frame_ids"][frame], "feature_frame_identity")
        checks.require(value["fmap"].shape == (1,1,128,116,152) and value["imap"].shape == (1,1,384,116,152), "captured_feature_shape")
        features.append(value["fmap"])
        contexts.append(value["imap"])
    fmap = np.concatenate(features, axis=1)
    imap = np.concatenate(contexts, axis=1)
    for row in capture["calls"]:
        artifact(capture_dir, row["artifact"], checks, evidence)
    for row in capture["inputs"]:
        path = safe_file(capture_dir, row["input"]["file"])
        artifact(capture_dir, row["input"], checks, evidence)
        value = np.load(path, allow_pickle=False)
        checks.require(raw_metadata(value) == row["array"], "captured_input_raw:"+path.name)
        checks.require(value.shape == (1,1,3,464,608) and value.dtype == np.float32 and np.isfinite(value).all(), "captured_input_contract")

    patches = artifact(run_dir, run["patches"], checks, evidence)
    centers = patches["centers"]
    checks.require(centers.shape == (3,96,2) and centers.dtype == np.float32
                   and np.array_equal(centers, np.floor(centers))
                   and np.all((centers[...,0] >= 1) & (centers[...,0] < 151))
                   and np.all((centers[...,1] >= 1) & (centers[...,1] < 115)), "integer_interior_patch_centers")
    checks.require(run["seed"] == 26, "recorded_patch_seed")
    checks.require(np.array_equal(patches["source_frame"], np.repeat(np.arange(3, dtype=np.int64),96)), "patch_source_view_order")
    expected_gmap = np.empty((1,288,128,3,3), np.float32)
    expected_ctx = np.empty((1,288,384), np.float32)
    expected_xy = np.empty((1,288,3,3,3), np.float32)
    dy, dx = np.meshgrid(np.arange(-1,2), np.arange(-1,2), indexing="ij")
    for view in range(3):
        for number, (cx,cy) in enumerate(centers[view].astype(np.int64)):
            index = view*96 + number
            expected_gmap[0,index] = fmap[0,view,:,cy-1:cy+2,cx-1:cx+2]
            expected_ctx[0,index] = imap[0,view,:,cy,cx]
            expected_xy[0,index] = np.stack((cx+dx,cy+dy,np.ones((3,3))))
    for key, expected in (("gmap",expected_gmap),("contexts",expected_ctx),("patches",expected_xy)):
        checks.require(np.array_equal(patches[key], expected), "independent_patch_sampling:"+key)

    summaries = []
    checks.require([row["name"] for row in run["scenarios"]] == ["first_motion_0_to_1", "three_frame_all_pairs"], "exact_scenarios")
    previous_end = None
    for scenario, size in zip(run["scenarios"], (96,864)):
        graph = artifact(run_dir, scenario["graph"], checks, evidence)
        if size == 96:
            kk, ii, jj = np.arange(96,dtype=np.int64), np.zeros(96,np.int64), np.ones(96,np.int64)
        else:
            kk = np.repeat(np.arange(288,dtype=np.int64),3)
            ii, jj = kk // 96, np.tile(np.arange(3,dtype=np.int64),288)
        for key, expected in (("kk",kk),("ii",ii),("jj",jj),("coords",expected_xy[:,kk,:2]),
                              ("ctx",expected_ctx[:,kk]),("net0",np.zeros((1,size,384),np.float32))):
            checks.require(np.array_equal(graph[key], expected), "exact_graph:"+scenario["name"]+":"+key)
        checks.require(scenario["edge_count"] == size and scenario["patches_per_view"] == 96
                       and scenario["feedback_iterations"] == 0 and scenario["fresh_zero_net_each_call"] is True, "fixed_identity_graph_scope")
        output0, call_timings, output_statistics = None, [], []
        checks.require(len(scenario["calls"]) == 4, "four_calls:"+scenario["name"])
        for index, call in enumerate(scenario["calls"]):
            result = artifact(run_dir, call["artifact"], checks, evidence)
            for key, shape in (("corr",(1,size,882)),("net",(1,size,384)),("delta",(1,size,2)),("weight",(1,size,2))):
                checks.require(result[key].shape == shape and result[key].dtype == np.float32
                               and np.count_nonzero(result[key]) > 0, "nonzero_output_contract:"+scenario["name"]+":"+str(index)+":"+key)
            checks.require(np.all((result["weight"] >= 0) & (result["weight"] <= 1)), "sigmoid_weight_range")
            if output0 is None:
                output0 = result
                output_statistics = {key:{"min":float(value.min()), "max":float(value.max()),
                                         "nonzero_elements":int(np.count_nonzero(value)), "elements":int(value.size)}
                                     for key,value in result.items()}
            else:
                for key in output0:
                    checks.require(result[key].tobytes() == output0[key].tobytes(), "actual_repeat_bytes:"+scenario["name"]+":"+str(index)+":"+key)
            median = float(np.median(np.linalg.norm(result["delta"].astype(np.float64), axis=-1)))
            checks.require(math.isclose(median,call["delta_norm_median_feature_pixels"],rel_tol=2e-6,abs_tol=2e-6), "reported_delta_norm")
            begin, middle, end = (call[key] for key in ("call_begin_monotonic","correlation_end_monotonic","update_end_monotonic"))
            checks.require(begin < middle < end and (previous_end is None or begin >= previous_end), "sequential_call_brackets")
            previous_end = end
            checks.require(call["call_index"] == index and call["timing_role"] == ("cold" if index == 0 else "warm")
                           and _close(middle-begin,call["corr_seconds"]) and _close(end-middle,call["update_seconds"])
                           and _close(end-begin,call["total_dynamic_seconds"]), "timing_decomposition")
            call_timings.append({key:call[key] for key in ("corr_seconds","update_seconds","total_dynamic_seconds")})
        warm = {key:sum(row[key] for row in call_timings[1:])/3 for key in call_timings[0]}
        checks.require(all(_close(warm[key],scenario["warm_means"][key]) for key in warm), "reported_warm_means")
        selected = sorted({0,1,2,size//2,size//2+1,size-2,size-1})
        expected = correlation_oracle(patches["gmap"],fmap,graph["coords"],kk,jj,selected)
        sample = compare(output0["corr"][0,selected],expected)
        checks.require(sample["outside_fixed_FP32_gate"] == 0, "independent_correlation_samples:"+scenario["name"])
        self_summary = None
        if size == 864:
            self_edges = np.flatnonzero(ii == jj)
            actual = output0["corr"].reshape(1,size,7,7,3,3,2)[0,self_edges,3,3,:,:,0]
            expected = np.sum(patches["gmap"][0,kk[self_edges]].astype(np.float64)**2,axis=1)
            self_summary = compare(actual, expected)
            self_summary["edges"] = int(self_edges.size)
            checks.require(self_edges.size == 288 and self_summary["outside_fixed_FP32_gate"] == 0
                           and np.all(actual >= 0), "all_identity_self_center_dot_products")
        summaries.append({"name":scenario["name"], "edges":size, "calls":4,
                          "cold_and_warm_seconds":call_timings, "warm_mean_seconds":warm,
                          "correlation_share_of_dynamic_warm":warm["corr_seconds"]/warm["total_dynamic_seconds"],
                          "graph_setup_seconds":scenario["graph_setup_seconds"], "output_statistics":output_statistics,
                          "repeat_all_saved_outputs_byte_equal":True,
                          "correlation_sample_edges":selected, "independent_correlation":sample,
                          "all_self_identity_centers":self_summary})
    for directory in (run_dir,capture_dir):
        path = directory/"platform_samples.json"
        if path.is_file():
            evidence[path] = sha256(path)
    checks.require(all(sha256(path) == digest for path,digest in evidence.items()), "audited_input_files_unchanged_during_audit")
    return {"schema":"v26_saved_DPVO_motion_numpy_audit", "status":"passed" if not checks.failures else "failed",
            "checks":{"count":checks.count,"failed":len(checks.failures),"failures":checks.failures},
            "auditor_sha256":evidence[Path(__file__)], "motion_receipt_sha256":evidence[run_dir/"motion.json"],
            "capture_receipt_sha256":evidence[capture_dir/"capture.json"],
            "provenance":{"repo":manifest["repo"],"commit":COMMIT,"checkpoint_sha256":evidence[checkpoint],
                          "original_inference_state_keys":94,"original_inference_parameters":3384324,
                          "source_pins":PINS,"frame_ids":run["frame_ids"],
                          "source_images_current_host_rehashed":False,
                          "source_image_provenance_scope":"Recorded before/after board SHA plus input-manifest binding; saved preprocessed NPY files rehashed, no image decode.",
                          "all_read_files_SHA256_unchanged":True,"read_file_count":len(evidence)},
            "patch_sampling":{"views":3,"patches_per_view":96,"integer_interior_xy":True,
                              "original_gmap_context_and_unit_depth_patch_values_checked":True,
                              "seed_recorded":26,"RNG_regeneration_performed":False},
            "correlation_convention":{"radius":3,"levels":[1,4],"pyramid_shapes":[[116,152],[29,38]],
                                       "flatten_order":["offset_x","offset_y","patch_y","patch_x","level"],
                                       "reference":"Independent NumPy channel-dot + bilinear loops; explicit FP32 row-major average pooling.",
                                       "sample_policy":"Edges 0,1,2,E//2,E//2+1,E-2,E-1; all offsets, patch positions, levels; plus all self level1 center correlations.",
                                       "strict_CUDA_bitwise_equivalence_verified":False},
            "scenarios":summaries,"shared_patch_setup_seconds":run["shared_patch_setup_seconds"],
            "separate_encoder_first_frame_warm_seconds":capture["first_frame_warm_mean_seconds"],
            "motion_peak_rss_mib":run["peak_rss_mib"],"cleanup_errors":run["cleanup_errors"],
            "bottleneck":"Current CPU correlation reference dominates: two levels, Python offset/chunk loops and Torch gather/channel reductions. Its timing includes pyramid pooling; it is not a hardware or algorithmic lower bound.",
            "scope":{"audit_uses_only_numpy_saved_arrays":True,"host_Torch_or_model_forward":False,
                     "NPU_used":False,"CUDA_used":False,"BA_executed":False,"valid_pose_output":False,
                     "VO_accuracy_validated":False,"realtime_validated":False,"production_changed":False},
            "limits":["Fixed identity projections on three frames, zero initial net, no feedback iterations; not a tracked trajectory.",
                      "Correlation samples and all self centers checked; whole learned Update not independently re-executed. Saved outputs are finite/nonzero and repeats byte-identical.",
                      "Dynamic timing excludes CNN, static patch/graph setup and BA. Encoder and motion timings from separate processes cannot be reported as measured end-to-end latency.",
                      "No SE3 estimate, BA, full VO lifecycle, trajectory/GT accuracy or 20 Hz validation. Optimizing only Update/CNN would leave this measured correlation reference dominant.",
                      "Four calls per scenario on one observed board configuration, no controlled thermal or statistically established benchmark."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    try:
        report = audit_run(args.run_dir,args.capture_dir,args.source_dir,args.checkpoint)
    except Exception as error:
        report = {"schema":"v26_saved_DPVO_motion_numpy_audit","status":"failed",
                  "auditor_sha256":sha256(__file__),"error":f"{type(error).__name__}: {error}"}
    with args.output.open("x") as output:
        json.dump(report,output,indent=2,allow_nan=False)
        output.write("\n")
    print(json.dumps({"status":report["status"],"checks":report.get("checks"),
                      "error":report.get("error"),"output":args.output.name,"sha256":sha256(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
