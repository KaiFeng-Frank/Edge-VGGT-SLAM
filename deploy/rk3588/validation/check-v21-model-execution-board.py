#!/usr/bin/env python3
"""Board-only original VGGT execution-contract regression on one saved TUM pair.

Four complete original FP32 forwards: direct/adapter normal and direct/adapter
compute_similarity. No NPU, ORB, frontend, GT or realtime benchmark. Already
computed normal results are reused for two independent first-submap SL4 graphs.
The checkpoint hash is checked before load and warms the filesystem cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path("/mnt/vggt-data/vggt-slam2")
SOURCE = BASE / "src"
CHECKPOINT_SIZE = 5026874952
CHECKPOINT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
KEYS = ("extrinsic", "intrinsic", "depth", "depth_conf")
REFERENCE_KINDS = (
    "fixed_rectified_original_VGGT_resolution_pairs",
    "fixed_r11_original_VGGT_quality_only_pairs",
)
ORIGINAL_SOURCE_KEYS = (
    "vggt_slam/runtime.py", "vggt_slam/image_rectification.py",
    "third_party/vggt/vggt/models/vggt.py", "third_party/vggt/vggt/models/aggregator.py",
    "third_party/vggt/vggt/heads/camera_head.py", "third_party/vggt/vggt/heads/dpt_head.py",
    "third_party/vggt/vggt/layers/vision_transformer.py",
    "third_party/vggt/vggt/layers/block.py", "third_party/vggt/vggt/layers/attention.py",
    "third_party/vggt/vggt/utils/load_fn.py", "third_party/vggt/vggt/utils/pose_enc.py",
)
REGRESSION_SOURCE_KEYS = ORIGINAL_SOURCE_KEYS + (
    "vggt_slam/model_execution.py", "vggt_slam/solver.py", "vggt_slam/edge_mapping.py",
    "vggt_slam/submap.py", "vggt_slam/map.py", "vggt_slam/graph.py",
    "vggt_slam/scale_solver.py", "vggt_slam/timestamp_replay.py",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def board_guard(machine, source, output):
    if machine.lower() not in ("aarch64", "arm64"):
        raise ValueError("Actual regression is allowed on the aarch64 board only")
    if not Path(source).resolve().is_relative_to(BASE):
        raise ValueError("Source checkout must be under the deployed board BASE")
    if not Path(output).resolve().is_relative_to(BASE / "experiments"):
        raise ValueError("Output must be a new directory under board BASE/experiments")
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError("Output already exists; raw experiments are never overwritten")


def validate_reference(receipt):
    if (receipt.get("kind") not in REFERENCE_KINDS
            or receipt.get("status") != "completed_fixed_input_diagnostic"
            or receipt.get("image_size") != 112 or receipt.get("precision") != "float32"
            or receipt.get("device") != "cpu" or receipt.get("original_weights") is not True
            or receipt.get("ORB_used") is not False or receipt.get("npu_used") is not False):
        raise ValueError("Reference must be a completed original CPU FP32 rectified 112 capture")
    if (receipt.get("checkpoint", {}).get("bytes") != CHECKPOINT_SIZE
            or receipt["checkpoint"].get("sha256") != CHECKPOINT_SHA):
        raise ValueError("Reference checkpoint must have the original official size/SHA")
    pair, selection = receipt["pairs"][0], receipt["selected_inputs"][0]
    if (pair["revision"] != 1 or selection["revision"] != 1
            or pair["frame_ids"] != selection["frame_ids"]
            or len(pair["frame_ids"]) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in pair["frame_ids"])
            or pair["frame_ids"][0] >= pair["frame_ids"][1]
            or len(selection["images"]) != 2):
        raise ValueError("Reference must identify the first exact two-frame batch")
    if pair["input_tensor"]["shape"] != [2, 3, 84, 112]:
        raise ValueError("Expected the actual TUM 84x112 two-image tensor")
    for frame_id, image in zip(pair["frame_ids"], selection["images"]):
        if (image["frame_id"] != frame_id or image["timestamp"] != frame_id
                or image["schedule_token"]["frame_id"] != frame_id
                or image["schedule_token"]["timestamp"] != frame_id
                or image.get("samefile_validated") is not True):
            raise ValueError("Source IDs and original timestamps must match exactly")
    expected_shapes = {"extrinsic":[1,2,3,4], "intrinsic":[1,2,3,3],
                       "depth":[1,2,84,112,1], "depth_conf":[1,2,84,112]}
    for key, shape in expected_shapes.items():
        if (key not in pair["arrays"] or pair["arrays"][key].get("shape") != shape
                or pair["arrays"][key].get("dtype") != "float32"
                or pair["arrays"][key].get("finite") is not True):
            raise ValueError("Reference has no complete canonical prediction metadata")
    return pair, selection


def array_info(array):
    import numpy as np
    array = np.asarray(array)
    if array.dtype != np.float32 or not array.size or not np.isfinite(array).all():
        raise ValueError("Prediction must be nonempty finite FP32")
    contiguous = np.ascontiguousarray(array)
    return dict(shape=list(array.shape), dtype=str(array.dtype), finite=True,
                bytes=int(array.nbytes), tensor_sha256=hashlib.sha256(contiguous.tobytes()).hexdigest())


def exact_comparison(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError("Canonical prediction shape or FP32 dtype differs")
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Canonical predictions must be finite and nonempty")
    delta = a.astype(np.float64) - b.astype(np.float64)
    reference_norm = float(np.linalg.norm(b.astype(np.float64)))
    return dict(array_equal=bool(np.array_equal(a, b)),
                bitwise_equal=a.tobytes(order="C")==b.tobytes(order="C"),
                max_abs=float(np.max(np.abs(delta))),
                rmse=float(np.sqrt(np.mean(delta * delta))),
                relative_l2=float(np.linalg.norm(delta) / reference_norm) if reference_norm else None,
                predefined_gate="bitwise_original_FP32_exact")


def compare_predictions(actual, expected):
    comparisons = {key: exact_comparison(actual[key], expected[key]) for key in KEYS}
    if not all(row["bitwise_equal"] for row in comparisons.values()):
        raise ValueError(f"Original canonical FP32 exact gate failed: {comparisons}")
    return comparisons


def load_original_predictions(path, pair):
    """Verify original array identities before using them as an oracle."""
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["frame_ids"]
        if ids.dtype != np.float64 or ids.shape != (2,) or ids.tolist() != pair["frame_ids"]:
            raise ValueError("Saved prediction source IDs must match the original doubles exactly")
        result = {}
        for key in KEYS:
            value = saved[key]
            if array_info(value) != pair["arrays"][key] or value.shape[:2] != (1, 2):
                raise ValueError(f"Saved original prediction array identity differs: {key}")
            result[key] = np.ascontiguousarray(value[0])
    return result


def compare_opaque_tokens(actual, original):
    """Preserve the original optional object; never use tokens for retrieval."""
    if actual is None or original is None:
        if actual is not None or original is not None:
            raise ValueError("Original target_tokens availability changed")
        return dict(availability_unchanged=True, both_none=True)
    def array(value):
        return value.detach().float().cpu().numpy() if hasattr(value, "detach") else value
    comparison = exact_comparison(array(actual), array(original))
    if not comparison["bitwise_equal"]:
        raise ValueError("Original target_tokens changed")
    return comparison


def canonical_direct(raw, image_hw, decoder):
    """Independent old call/codec boundary; does not call the new executor."""
    import numpy as np
    extrinsic, intrinsic = decoder(raw["pose_enc"], image_hw)
    result = {}
    for key, tensor in {"extrinsic":extrinsic, "intrinsic":intrinsic,
                        "depth":raw["depth"], "depth_conf":raw["depth_conf"],
                        "pose_enc":raw["pose_enc"]}.items():
        if tensor.shape[0] != 1:
            raise ValueError("Original forward must return explicit B=1")
        result[key] = np.ascontiguousarray(tensor.detach().float().cpu().numpy()[0])
    ratio = raw.get("image_match_ratio")
    result["image_match_ratio"] = None if ratio is None else float(ratio)
    return result


def check_prepared_submap(submap, images, paths, frame_ids):
    """Check the actual Solver's pixel storage and double-precision metadata."""
    import numpy as np
    stored = submap.get_all_frames()
    stored_array = stored.detach().numpy()
    if (stored.device.type != "cpu" or submap.get_id() != 0
            or submap.get_frame_ids() != frame_ids or submap.img_names != paths
            or submap.get_last_non_loop_frame_index() != 1
            or not np.array_equal(stored_array, images)
            or not np.shares_memory(stored_array, images)):
        raise ValueError("Actual Solver submap must preserve source doubles/order and share prepared CPU pixels")
    return dict(frame_ids=submap.get_frame_ids(),image_names=submap.img_names,
                submap_id=submap.get_id(),last_non_loop_frame_index=1,
                stored_images=array_info(stored_array),shares_prepared_storage=True)


def check_first_graph(prediction, images, paths, *, solver=None):
    """Use public existing Solver/Submap methods, not guessed GTSAM internals."""
    import numpy as np
    import torch
    from vggt_slam.solver import Solver
    from vggt_slam.submap import Submap
    if solver is None:
        solver = Solver(init_conf_threshold=25., device="cpu", image_size=112,
                        max_loops=0, offline=True, enable_viewer=False)
        submap = Submap(0)
        submap.add_all_frames(torch.from_numpy(images.copy()))
        submap.set_frame_ids(paths)
        submap.set_img_names(paths)
        submap.set_last_non_loop_frame_index(1)
        solver.current_working_submap = submap
    values = {key:np.array(prediction[key],copy=True) for key in KEYS}
    values.update(images=images.copy(), detected_loops=[])
    solver.add_points(values)
    solver.graph.optimize()
    result = dict(nodes=int(solver.graph.values.size()), factors=int(solver.graph.graph.size()),
                  submaps=solver.map.get_num_submaps(), loops=int(solver.graph.get_num_loops()),
                  graph_error=float(solver.graph.graph.error(solver.graph.values)),
                  frame_ids=solver.map.get_submap(0).get_frame_ids(),
                  homographies=[np.asarray(solver.graph.get_homography(i)).tolist() for i in range(2)])
    if (result["nodes"],result["factors"],result["submaps"],result["loops"]) != (2,2,1,0):
        raise ValueError("First original SL4 graph must have 2 nodes, 2 factors, 1 submap and no loops")
    if not math.isfinite(result["graph_error"]) or not np.isfinite(result["homographies"]).all():
        raise ValueError("First graph returned nonfinite values")
    return result


def run(a):
    import numpy as np
    receipt = dict(kind="original_VGGT_execution_contract_regression",status="started",
                   script_sha256=sha(__file__), original_weights=True, architecture_changed=False,
                   inference_on_board_only=True, host_inference_executed=False, ORB_used=False,
                   npu_used=False, GT_used=False, real_time_validated=False, accuracy_validated=False,
                   predefined_gate="bitwise_original_FP32_exact", image_size=112,
                   forward_plan=["direct_normal","adapter_normal","direct_loop","adapter_loop"],
                   forwards=[], completed_call_endpoints=[],
                   checkpoint_hash_before_load_warms_filesystem_cache=True)
    started = time.monotonic()
    previous_affinity = os.sched_getaffinity(0)
    source_pins = None
    try:
        cpus = {int(v) for v in a.cpus.split(",")}
        if not cpus or not cpus.issubset(previous_affinity) or a.threads < 1:
            raise ValueError("Requested CPU set/threads invalid or outside allowed affinity")
        os.sched_setaffinity(0, cpus)
        os.environ.update(OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS="1",
                          MKL_NUM_THREADS=str(a.threads),MPLBACKEND="Agg",
                          OMP_WAIT_POLICY="PASSIVE",GOMP_SPINCOUNT="0")
        sys.path[:0] = [str(SOURCE),str(SOURCE/"third_party/vggt")]
        source_pins = {key:sha(SOURCE/key) for key in REGRESSION_SOURCE_KEYS}
        receipt["source_hashes"] = source_pins
        if source_pins["vggt_slam/model_execution.py"] != a.execution_sha256 or source_pins["vggt_slam/solver.py"] != a.solver_sha256:
            raise ValueError("New executor or Solver source does not match the explicit frozen SHA")
        if sha(a.reference_receipt) != a.reference_sha256:
            raise ValueError("Original reference capture receipt SHA mismatch")
        reference = json.loads(a.reference_receipt.read_text())
        pair, selection = validate_reference(reference)
        for key in ORIGINAL_SOURCE_KEYS:
            if source_pins[key] != reference["source_hashes"][key]:
                raise ValueError(f"Original model/preprocessing source changed: {key}")
        if sha(a.reference_predictions) != pair["sha256"]:
            raise ValueError("Original saved prediction file SHA mismatch")
        if a.checkpoint.stat().st_size != CHECKPOINT_SIZE or sha(a.checkpoint) != CHECKPOINT_SHA:
            raise ValueError("Official original checkpoint size/SHA mismatch")
        checkpoint_stat = a.checkpoint.stat()
        receipt["checkpoint"] = dict(bytes=CHECKPOINT_SIZE,sha256=CHECKPOINT_SHA)
        receipt["reference"] = dict(receipt_sha256=a.reference_sha256,
                                     prediction_sha256=pair["sha256"], frame_ids=pair["frame_ids"],
                                     kind=reference["kind"])
        images_meta = selection["images"]
        paths = [item["path"] for item in images_meta]
        for image in images_meta:
            if not Path(image["path"]).samefile(Path(image["schedule_token"]["path"])) or sha(image["path"]) != image["sha256"]:
                raise ValueError("Original source image identity or SHA mismatch")
        calibration = Path(reference["calibration"]["path"])
        if sha(calibration) != reference["calibration"]["sha256"]:
            raise ValueError("Original calibration SHA changed")
        rgb_index = Path(reference["rgb_index"]["path"])
        if sha(rgb_index) != reference["rgb_index"]["sha256"]:
            raise ValueError("Original RGB timestamp index SHA changed")
        receipt["source_frames"] = images_meta
        receipt["calibration_sha256"] = sha(calibration)
        imported = time.monotonic()
        import cv2
        import torch
        from vggt_slam.runtime import load_model
        from vggt_slam.image_rectification import CalibrationRectifier
        from vggt_slam.edge_mapping import prepare_rectified_images
        from vggt_slam.model_execution import TorchVGGTExecutor, execute
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        cv2.setNumThreads(1)
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(1)
        receipt["import_seconds"] = time.monotonic()-imported
        receipt["runtime"] = dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                                  opencv=cv2.__version__,cpus=sorted(os.sched_getaffinity(0)),
                                  threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads(),
                                  environment={key:os.environ.get(key) for key in (
                                      "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                                      "OMP_WAIT_POLICY","GOMP_SPINCOUNT","MPLBACKEND")})
        rectifier = CalibrationRectifier(json.loads(calibration.read_text()))
        if rectifier.metadata() != reference["rectification"]:
            raise ValueError("Rectification metadata changed")
        receipt["rectification"] = rectifier.metadata()
        prepared_begin = time.monotonic()
        images_tensor = prepare_rectified_images(paths,rectifier,112)
        images = np.ascontiguousarray(images_tensor.numpy())
        input_info = array_info(images)
        if input_info != pair["input_tensor"]:
            raise ValueError("Prepared tensor is not the original bitwise TUM pair")
        receipt["preprocessing_seconds"] = time.monotonic()-prepared_begin
        receipt["input_tensor"] = input_info
        np.save(a.output/"input.npy",images,allow_pickle=False)
        old = load_original_predictions(a.reference_predictions, pair)
        load_begin = time.monotonic()
        model = load_model(a.checkpoint,device="cpu",image_size=112,prune_unused_heads=True,offline=True)
        receipt["model_load_seconds"] = time.monotonic()-load_begin
        if (model.training or sum(p.numel() for p in model.parameters()) != 1157941492
                or not all(p.device.type=="cpu" and p.dtype==torch.float32 for p in model.parameters())):
            raise ValueError("Original pruned model must be eval CPU FP32 with exact parameter count")
        executor = TorchVGGTExecutor(model)
        from vggt_slam.solver import Solver
        normal_solver = Solver(init_conf_threshold=25.,device="cpu",image_size=112,
                               max_loops=0,offline=True,enable_viewer=False)
        outputs = {}
        for request, compute_similarity in [("normal",False),("loop",True)]:
            direct_begin = time.monotonic()
            with torch.no_grad():
                raw = model(torch.from_numpy(images.copy()),compute_similarity=compute_similarity)
            direct_model_seconds = time.monotonic()-direct_begin
            receipt["completed_call_endpoints"].append(f"direct_{request}")
            direct = canonical_direct(raw,images.shape[-2:],pose_encoding_to_extri_intri)
            direct_filename=f"direct-{request}.npz"
            np.savez(a.output/direct_filename,**{key:direct[key] for key in KEYS},
                     frame_ids=np.asarray(pair["frame_ids"],np.float64))
            receipt.setdefault("direct_artifacts",{})[request] = dict(
                file=direct_filename,sha256=sha(a.output/direct_filename))
            if request=="normal":
                receipt["direct_normal_vs_saved_original"] = compare_predictions(direct,old)
            if compute_similarity and (direct["image_match_ratio"] is None or not math.isfinite(direct["image_match_ratio"])):
                raise ValueError("Direct loop ratio must be finite")
            if not compute_similarity and direct["image_match_ratio"] is not None:
                raise ValueError("Normal model must not return a loop ratio")
            forward_input = images.copy()
            adapter_begin = time.monotonic()
            if compute_similarity:
                actual = execute(executor,forward_input,compute_similarity=True)
            else:
                actual = normal_solver.run_predictions(
                    paths,executor,0,None,None,prepared_images=forward_input)
            adapter_seconds = time.monotonic()-adapter_begin
            receipt["completed_call_endpoints"].append(f"adapter_{request}")
            if not compute_similarity:
                receipt["actual_solver_submap"] = check_prepared_submap(
                    normal_solver.current_working_submap,forward_input,paths,pair["frame_ids"])
            comparisons = compare_predictions(actual,direct)
            if (array_info(forward_input)!=input_info or array_info(actual["images"])!=input_info):
                raise ValueError("Executor altered input pixels/order")
            if actual["image_match_ratio"] != direct["image_match_ratio"]:
                raise ValueError("Loop similarity ratio must match original exactly")
            extras = {}
            if "pose_enc" in raw:
                extras["pose_enc"] = exact_comparison(actual["pose_enc"],direct["pose_enc"])
                if not extras["pose_enc"]["bitwise_equal"]:
                    raise ValueError("Optional original pose encoding changed")
            if "target_tokens" in raw:
                extras["target_tokens"] = compare_opaque_tokens(actual["target_tokens"],raw["target_tokens"])
            filename=f"{request}-predictions.npz"
            np.savez(a.output/filename,**{f"direct_{k}":direct[k] for k in KEYS},
                     **{f"adapter_{k}":np.asarray(actual[k]) for k in KEYS},
                     frame_ids=np.asarray(pair["frame_ids"],np.float64))
            receipt["forwards"].append(dict(request=request,compute_similarity=compute_similarity,
                frame_ids=pair["frame_ids"],direct_original_model_seconds=direct_model_seconds,
                adapter_execution_including_validation_seconds=adapter_seconds,
                adapter_endpoint="execute(compute_similarity=True)" if compute_similarity else "Solver.run_predictions(prepared_images)",
                comparisons=comparisons,extras=extras,image_match_ratio=direct["image_match_ratio"],
                artifact=dict(file=filename,sha256=sha(a.output/filename))))
            outputs[request]=(direct,actual)
            del raw
        graph_begin=time.monotonic()
        direct_graph=check_first_graph(outputs["normal"][0],images,paths)
        adapter_graph=check_first_graph(outputs["normal"][1],images,paths,solver=normal_solver)
        if (direct_graph["frame_ids"] != pair["frame_ids"] or adapter_graph["frame_ids"] != pair["frame_ids"]
                or not np.allclose(direct_graph["homographies"],adapter_graph["homographies"],rtol=0,atol=1e-12)):
            raise ValueError("First SL4 graph/source identity differs")
        receipt["first_graph"] = dict(direct=direct_graph,adapter=adapter_graph,
            H_allclose_atol=1e-12,H_allclose_rtol=0,seconds=time.monotonic()-graph_begin,
            additional_model_forwards=0,overlap_scale_or_true_loop_graph_validated=False)
        for image in images_meta:
            if sha(image["path"])!=image["sha256"]:
                raise ValueError("Source image changed during regression")
        if {key:sha(SOURCE/key) for key in REGRESSION_SOURCE_KEYS}!=source_pins:
            raise ValueError("Source changed during regression")
        if sha(calibration)!=reference["calibration"]["sha256"]:
            raise ValueError("Calibration changed during regression")
        if sha(rgb_index)!=reference["rgb_index"]["sha256"]:
            raise ValueError("RGB timestamp index changed during regression")
        if (a.checkpoint.stat().st_size,a.checkpoint.stat().st_mtime_ns)!=(checkpoint_stat.st_size,checkpoint_stat.st_mtime_ns):
            raise ValueError("Checkpoint changed during regression")
        receipt.update(status="passed_original_FP32_execution_regression",source_and_inputs_unchanged=True,
                       full_model_forward_count=4,loop_graph_accepted=False,
                       similarity_request_not_natural_revisit_claim=True)
    except BaseException as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    finally:
        try:
            os.sched_setaffinity(0,previous_affinity)
            receipt["cpu_affinity_restored"] = True
        except OSError as exc:
            receipt["cpu_affinity_restored"] = False
            receipt["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if receipt["status"] != "failed":
                receipt.update(status="failed",error=receipt["cleanup_error"])
        receipt.update(total_seconds=time.monotonic()-started,
                       peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (a.output/"regression.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    return 0 if receipt["status"]=="passed_original_FP32_execution_regression" else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-receipt",type=Path,required=True)
    p.add_argument("--reference-sha256",required=True)
    p.add_argument("--reference-predictions",type=Path,required=True)
    p.add_argument("--execution-sha256",required=True)
    p.add_argument("--solver-sha256",required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--cpus",default="4,5,6,7")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        board_guard(platform.machine(),SOURCE,a.output)
        a.output.mkdir(parents=True,exist_ok=False)
    except (ValueError,FileExistsError) as exc:
        p.error(str(exc))
    return run(a)


if __name__=="__main__":
    raise SystemExit(main())
