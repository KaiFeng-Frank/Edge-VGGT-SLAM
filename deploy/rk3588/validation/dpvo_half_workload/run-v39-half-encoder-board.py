#!/usr/bin/env python3
"""Board-only new-size CPU teacher/ORT diagnostic or calibration generation.

MIT project helper; DPVO copyright (c) 2022 Princeton Vision & Learning Lab.
The half-resolution preprocessing and M16 are an explicit research preset.
Original weights are loaded only in probe mode, by the frozen strict loader.
Calibration generation decodes images but executes no model. Actual work must
run on RK3588/S; host imports/tiny synthetic helper tests do not execute models.
"""
import argparse
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

POLICY_SHA = "cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a"
FRAME_SHA = "fc6fc662613a720cfb779a360a63906a593a96ff08addecb9d01cfd8767ca7a8"
COMMON_SHA = "ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9"
OPS_SHA = "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
CHECKPOINT_SHA = "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"
SHAPES = {"fmap": (1, 128, 58, 76), "imap": (1, 384, 58, 76), "context": (1, 16, 384)}
CALIBRATION_INDICES = [256+16*i for i in range(32)]


def require(ok, text):
    if not ok:
        raise ValueError(text)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def board_guard(machine=None, system=None, compatible=None):
    machine = platform.machine() if machine is None else machine
    system = platform.system() if system is None else system
    if compatible is None:
        p = Path("/proc/device-tree/compatible")
        compatible = p.read_bytes() if p.is_file() else b""
    require(machine == "aarch64" and system == "Linux" and any(v in (b"rockchip,rk3588", b"rockchip,rk3588s") for v in compatible.split(b"\0")), "RK3588/S Linux aarch64 board only")


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def file_meta(path):
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def compare(a, b):
    import numpy as np
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "Matching finite FP32 features")
    x, y = a.astype(np.float64), b.astype(np.float64)
    d = x-y
    count = int(np.count_nonzero(np.abs(d) > 1e-5+1e-4*np.abs(y)))
    return dict(rtol=1e-4, atol=1e-5, gate_passed=count == 0, failed_elements=count, element_count=a.size,
        bitwise_equal=a.tobytes(order="C") == b.tobytes(order="C"), max_abs=float(np.max(np.abs(d))),
        relative_l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(y.ravel()), 1e-30)))


def half_bgr(cropped, cv2):
    import numpy as np
    require(cropped.dtype == np.uint8 and cropped.shape == (464, 608, 3), "Original cropped uint8 BGR required")
    half = cv2.resize(cropped, (304, 232), interpolation=cv2.INTER_AREA)
    require(half.dtype == np.uint8 and half.shape == (232, 304, 3), "Exact-half uint8 INTER_AREA resize")
    return np.ascontiguousarray(half)


def sample(frame_helper, generator, torch):
    import numpy as np
    before = generator.get_state().numpy().copy()
    centers = frame_helper.random_centers(76, 58, 16, generator)
    after_centers = generator.get_state().numpy().copy()
    depth = torch.rand((16,), generator=generator, dtype=torch.float32).numpy().copy()
    require(centers.dtype == np.int64 and centers.shape == (16, 2) and
            np.all((centers[:, 0] >= 1) & (centers[:, 0] < 75) & (centers[:, 1] >= 1) & (centers[:, 1] < 57)), "Original-order new-grid integer centers")
    return dict(centers_xy=centers, patch_indices=np.ascontiguousarray(centers[:, 1]*76+centers[:, 0]),
        consumed_inverse_depth=depth, state_before=before, state_after_centers=after_centers,
        state_after_depth=generator.get_state().numpy().copy())


def gather_context(imap, indices):
    import numpy as np
    require(imap.dtype == np.float32 and imap.shape == SHAPES["imap"] and np.isfinite(imap).all(), "New-size dense FP32 context")
    require(indices.dtype == np.int64 and indices.shape == (16,) and np.all((indices >= 0) & (indices < 58*76)), "M16 row-major integer indices")
    value = np.ascontiguousarray(imap[0, :, indices//76, indices%76][None])
    require(value.shape == SHAPES["context"], "Preserved sample order including duplicate indices")
    return value


def rgb_rows(text):
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        words = line.split()
        require(len(words) == 2 and words[1].startswith("rgb/") and ".." not in Path(words[1]).parts, "Official RGB index row")
        rows.append(dict(source_index=len(rows), frame_id=words[0], relative_path=words[1]))
    require(len(rows) == 798 and all(float(a["frame_id"]) < float(b["frame_id"]) for a, b in zip(rows, rows[1:])), "Pinned original 798 ordered RGB rows")
    return rows


def calibration_selection(rows):
    require(len(rows) == 798 and not set(CALIBRATION_INDICES).intersection(range(192)), "Calibration separated from first192 evaluation")
    return [rows[i] for i in CALIBRATION_INDICES]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("probe", "calibration"))
    for name in ("policy", "frame-helper", "dataset-dir", "rgb-index", "old-capture-dir", "rewrite-dir", "editor",
                 "original-onnx", "reciprocal-onnx", "reciprocal-recipe", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--rewrite-sha256", required=True)
    p.add_argument("--editor-sha256", required=True)
    for name in ("common", "cpu-ops", "source-dir", "source-manifest", "checkpoint"):
        p.add_argument("--"+name, type=Path)
    args = p.parse_args(argv)
    board_guard()  # Before model/ORT/Torch/OpenCV import, decoding or output creation.
    out = args.output_dir.resolve()
    require(out.is_relative_to(Path("/mnt/vggt-data/vggt-slam2/experiments")) and not out.exists(), "New isolated board experiment")
    protected = [v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"]
    require(not any(out == v or out.is_relative_to(v) or v.is_relative_to(out) for v in protected), "Source/output overlap")
    out.mkdir(parents=True, exist_ok=False)
    pins, sessions, prior = {}, [], os.sched_getaffinity(0)
    start = time.monotonic()
    report = dict(schema="v39_DPVO_half_encoder_board_"+args.mode, status="failed", mode=args.mode, script_sha256=sha(__file__),
        source_pins=pins, calls=[], samples=[], cleanup_errors=[], candidate_accepted=False, historical_parent_FP32_gate_passed=False,
        historical_parent_rejection_retained=True, complete_VO=False, realtime_validated=False, GT_read=False,
        original_FP32_gate=dict(rtol=1e-4, atol=1e-5), normalization_applications_per_image=1,
        old_resolution_outputs_used_as_new_gold=False, original_pixels_preserved=False,
        original_DPVO_checkpoint_sha256=CHECKPOINT_SHA, model_forward_executed=False)
    def pin(path, digest):
        path = Path(path).resolve()
        require(sha(path) == digest, "Source/input SHA: "+path.name)
        pins[str(path)] = digest
        return path
    def load(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        value = importlib.util.module_from_spec(spec);sys.modules[name] = value;spec.loader.exec_module(value)
        return value
    def save(name, values):
        path = out/name
        with path.open("xb") as f:
            np.savez(f, **values)
        return dict(**file_meta(path), arrays={k: meta(v) for k, v in values.items()})
    try:
        import cv2
        import numpy as np
        import torch
        pin(__file__, report["script_sha256"])
        policy = json.loads(pin(args.policy, POLICY_SHA).read_text())
        capture = json.loads(pin(args.old_capture_dir/"capture.json", CAPTURE_SHA).read_text())
        require(capture["sources_and_inputs_unchanged"] and capture["state_unchanged"] and capture["cleanup_errors"] == [], "Original CPU source lineage")
        require(policy["image_size_hw"] == [232, 304] and policy["stream_config"]["patches_per_frame"] == 16, "Single declared half M16 preset")
        report.update(policy_sha256=POLICY_SHA, preprocessing=policy["preprocessing"], rng_policy=policy["rng"],
            old_capture_sha256=CAPTURE_SHA, rewrite_sha256=args.rewrite_sha256, versions=dict(torch=torch.__version__, numpy=np.__version__, OpenCV=cv2.__version__))
        helper = load(args.frame_helper, FRAME_SHA, "_v39_frozen_frame")
        editor = load(args.editor, args.editor_sha256, "_v39_static_shape_editor")
        recipe = json.loads(pin(args.rewrite_dir/"rewrite.json", args.rewrite_sha256).read_text())
        require(recipe["schema"] == "v39_DPVO_half_resolution_encoder_shape_rewrite" and recipe["script_sha256"] == args.editor_sha256 and
                recipe["patch_count"] == 16 and recipe["historical_parent_rejection_retained"] is True and
                recipe["historical_parent_FP32_gate_passed"] is False and recipe["original_weights_preserved"] is True and
                recipe["candidate_accepted"] is False and recipe["sources_and_parents_unchanged"] is True and
                recipe["source_pins"] == recipe["source_hashes_after"], "New half graph source contract")
        parent_paths = {"original": pin(args.original_onnx, editor.ORIGINAL_SHA),
                        "reciprocal": pin(args.reciprocal_onnx, editor.RECIPROCAL_SHA)}
        pin(args.reciprocal_recipe, editor.RECIPROCAL_RECIPE_SHA)
        require(sorted(recipe["source_pins"].values()) == sorted([editor.ORIGINAL_SHA, editor.RECIPROCAL_SHA,
                editor.RECIPROCAL_RECIPE_SHA, args.editor_sha256]), "Exact parent/editor source hashes; paths may relocate")
        import onnx
        graph_paths = {k: pin(args.rewrite_dir/r["file"], r["sha256"]) for k, r in recipe["models"].items()}
        require(set(graph_paths) == {"original", "reciprocal"}, "Original/new expression half graphs")
        for method, graph_path in graph_paths.items():
            row = recipe["models"][method]
            require(row["parent_onnx_sha256"] == sha(parent_paths[method]), "Exactly pinned method parent graph")
            parent = onnx.load(parent_paths[method], load_external_data=False)
            candidate = onnx.load(graph_path, load_external_data=False)
            require(editor.validate_preservation(parent, candidate, method) == row["preservation"], "Independent exact shape-only preservation")
            require(editor.contract(candidate.graph.input) == row["inputs"] and editor.contract(candidate.graph.output) == row["outputs"], "New graph actual static IO")
        report["shape_preservation_rechecked"] = True
        # Source images are fixed by the original index; no GT/content selection.
        rgb_digest = next(d for path, d in capture["source_pins"].items() if Path(path).name == "rgb.txt")
        rows = rgb_rows(pin(args.rgb_index, rgb_digest).read_text())
        chosen = calibration_selection(rows) if args.mode == "calibration" else rows[:3]
        os.sched_setaffinity(0, {4, 5, 6, 7});torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        generator = torch.Generator(device="cpu").manual_seed(policy["rng"]["seed"])
        if args.mode == "probe":
            require(all(getattr(args, k) is not None for k in ("common", "cpu_ops", "source_dir", "source_manifest", "checkpoint")), "Original neural sources required")
            common = load(args.common, COMMON_SHA, "_v39_frozen_common")
            ops = load(args.cpu_ops, OPS_SHA, "_v39_frozen_ops")
            pin(args.checkpoint, CHECKPOINT_SHA)
            begin = time.monotonic()
            neural, provenance = common.load_neural(args.source_dir, args.source_manifest, args.checkpoint, CHECKPOINT_SHA, args.cpu_ops, cpu_ops_sha256=OPS_SHA)
            neural.eval().requires_grad_(False);encoder = common.make_encoder(neural)
            report.update(load_seconds=time.monotonic()-begin, loader=provenance)
            weights = {k: meta(v.detach().cpu().numpy()) for k, v in neural.state_dict().items()}
            require(weights == capture["state_before"], "All94 original parameter bytes match original CPU capture")
            report["state_before"] = weights
            for path, digest in capture["source_pins"].items():
                if Path(path).name in ("net.py", "blocks.py", "extractor.py", "LICENSE"):
                    actual = args.source_dir/("dpvo/"+Path(path).name if Path(path).name != "LICENSE" else "LICENSE")
                    pin(actual, digest)
            pin(args.source_manifest, common.MANIFEST_SHA256)
            import onnxruntime as ort
            options = ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            for method in ("original", "reciprocal"):
                s=ort.InferenceSession(str(graph_paths[method]),sess_options=options,providers=["CPUExecutionProvider"]);sessions.append(s)
                require(s.get_providers()==["CPUExecutionProvider"], "CPU ORT only")
                wanted=recipe["models"][method]
                for actual, declared in ((s.get_inputs(),wanted["inputs"]),(s.get_outputs(),wanted["outputs"])):
                    require([(v.name,v.shape,v.type) for v in actual] == [(v["name"],v["shape"],"tensor(int64)" if v["dtype"]=="INT64" else "tensor(float)") for v in declared], "Exact actual half ORT IO contract")
            report.update(ORT_version=ort.__version__, providers=["CPUExecutionProvider"], graph_optimization="ORT_DISABLE_ALL")
        K=np.array([[517.3,0,318.6],[0,516.5,255.3],[0,0,1]],np.float64)
        D=np.array([.2624,-.9531,-.0054,.0026,1.1633],np.float64)
        dataset_lines=[];sampling={k:[] for k in ("centers_xy","consumed_inverse_depth","state_before","state_after_centers","state_after_depth")}
        for index, source in enumerate(chosen):
            path=(args.dataset_dir/source["relative_path"]).resolve();require(path.is_relative_to(args.dataset_dir.resolve()), "Source path traversal")
            source_digest=sha(path);pin(path,source_digest)
            begin=time.monotonic();bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
            require(bgr is not None and bgr.dtype==np.uint8 and bgr.shape==(480,640,3), "Original TUM BGR image")
            cropped=np.ascontiguousarray(cv2.undistort(bgr,K,D)[8:-8,16:-16])
            if args.mode=="probe":
                old=capture["inputs"][index]
                require(old["frame_id"]==source["frame_id"] and old["source"]["sha256"]==source_digest and
                        meta(cropped.transpose(2,0,1))==old["cropped_BGR_uint8"], "Original decoded/cropped source byte lineage")
            half=half_bgr(cropped,cv2);image=helper.normalize_bgr(half);preprocessing_seconds=time.monotonic()-begin
            require(image.dtype==np.float32 and image.shape==(1,3,232,304) and np.isfinite(image).all() and image.min()>=-.5 and image.max()<=1.5, "Once-normalized half BGR")
            draws=sample(helper,generator,torch);indices=draws["patch_indices"]
            record=dict(index=index,**source,source_image_sha256=source_digest,source_image_bytes=path.stat().st_size,
                cropped_BGR=meta(cropped),resized_BGR=meta(half),image_array=meta(image),index_array=meta(indices),preprocessing_seconds=preprocessing_seconds,
                RNG={k:meta(v) for k,v in draws.items()},calibration_not_online_RNG_equivalence=True)
            if args.mode=="calibration":
                files=[]
                for name,value in ((f"image-{index:02d}.npy",image),(f"indices-{index:02d}.npy",indices)):
                    target=out/name;np.save(target,value,allow_pickle=False);files.append(file_meta(target))
                record["input_files"]=files;dataset_lines.append(" ".join(f["file"] for f in files))
                for key in sampling:sampling[key].append(draws[key])
            else:
                record["input_artifact"]=save(f"input-{index}.npz",dict(normalized_bgr=image,**draws))
                with torch.inference_mode():
                    begin=time.monotonic();f,im=encoder(torch.from_numpy(image[:,None]));forward=time.monotonic()-begin
                    f,im=f[:,0].contiguous(),im[:,0].contiguous()
                    coords=torch.from_numpy(draws["centers_xy"].astype(np.float32))[None]
                    begin=time.monotonic();gmap=ops.patchify(f,coords,1);context=ops.patchify(im,coords,0)[...,0,0];sampling_seconds=time.monotonic()-begin
                teacher={k:v.numpy().copy() for k,v in dict(fmap=f,imap=im,context=context,gmap=gmap).items()}
                require(all(teacher[k].shape == shape and teacher[k].dtype == np.float32 for k,shape in SHAPES.items()) and
                        teacher["gmap"].shape == (1,16,128,3,3) and teacher["gmap"].dtype == np.float32 and
                        all(np.isfinite(v).all() and np.any(v!=0) for v in teacher.values()), "Finite/nonzero new-size original CPU teacher")
                record.update(teacher_forward_seconds=forward,teacher_CPU_patchify_seconds=sampling_seconds,teacher_artifact=save(f"teacher-{index}.npz",teacher))
                report["model_forward_executed"]=True
                outputs={}
                for method,session in zip(("original","reciprocal"),sessions):
                    feeds={"normalized_bgr":image}
                    names=["fmap","imap"] if method=="original" else ["fmap","context"]
                    if method=="reciprocal":feeds["patch_indices"]=indices
                    begin=time.monotonic();actual=dict(zip(names,session.run(names,feeds)));seconds=time.monotonic()-begin
                    comparisons={k:compare(v,teacher[k]) for k,v in actual.items()}
                    outputs[method]=actual
                    record[method]=dict(forward_seconds=seconds,artifact=save(f"{method}-{index}.npz",actual),versus_new_size_original_CPU=comparisons)
                dense=outputs["original"]["imap"]
                gather=gather_context(dense,indices)
                record["reciprocal_vs_current_original_half_ORT"]={"fmap":compare(outputs["reciprocal"]["fmap"],outputs["original"]["fmap"]),"context":compare(outputs["reciprocal"]["context"],gather)}
                report["calls"].append(record)
            report["samples"].append(record)
            require(sha(path)==source_digest, "Image source changed during read")
            print(json.dumps(dict(index=index,mode=args.mode,frame_id=source["frame_id"])),flush=True)
        if args.mode=="calibration":
            path=out/"dataset.txt";path.write_text("\n".join(dataset_lines)+"\n");report["dataset"]=file_meta(path)
            report["sampling_artifact"]=save("sampling.npz",{k:np.stack(v) for k,v in sampling.items()})
            report.update(status="generated_half_M16_calibration_not_quantized_or_validated",calibration_source_indices=CALIBRATION_INDICES,evaluation_source_indices=list(range(192)),GT_or_content_selection=False)
        else:
            report["strict_gate_each_method"]={k:all(s["gate_passed"] for r in report["calls"] for s in r[k]["versus_new_size_original_CPU"].values()) for k in ("original","reciprocal")}
            after={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()};require(after==weights, "Original94 weights changed")
            report.update(state_after=after,state_unchanged=True,status="completed_half_M16_CPU_teacher_and_ORT_diagnostic_not_VO")
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        sessions.clear()
        if "s" in locals():s=None
        if "session" in locals():session=None
        import gc;gc.collect()
        report["session_references_cleared"]=True
        try:os.sched_setaffinity(0,prior);report["affinity_restored"]=os.sched_getaffinity(0)==prior
        except Exception as exc:report["cleanup_errors"].append(repr(exc))
        report["source_hashes_after"]={p:sha(p) if Path(p).is_file() else None for p in pins};report["sources_and_inputs_unchanged"]=report["source_hashes_after"]==pins
        if not report["sources_and_inputs_unchanged"] or report["cleanup_errors"]:report["status"]="failed"
        report.update(actual_driver_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        report["artifact_bytes_before_receipt"]=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if report["artifact_bytes_before_receipt"] >= 256*1024**2:
            report.update(status="failed",error="Output budget256MiB exceeded")
        report["file_budget_bytes"]=256*1024**2
        (out/("probe.json" if args.mode=="probe" else "calibration.json")).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k) for k in ("status","error","strict_gate_each_method")}),flush=True)
    return 0 if report["status"].startswith(("completed_half_","generated_half_")) else 1


if __name__=="__main__":
    raise SystemExit(main())
