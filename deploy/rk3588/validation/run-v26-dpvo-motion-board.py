#!/usr/bin/env python3
"""Board-only original DPVO neural Update on captured features and identity graphs.

This is not a tracked trajectory, a pose estimator or a complete DPVO port.
CNN capture is separate; no GT, sensor depth, CUDA extension or NPU is used.
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

sys.dont_write_bytecode = True
COMMON_SHA = "ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9"
CPU_OPS_SHA = "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32"
CAPTURE_SCRIPT_SHA = "e2f56fac7a6432761bf67ea2703a7b72b9d695577302d8f02d5cc8af0c0db9a5"
SCENARIOS = ("first_motion_0_to_1", "three_frame_all_pairs")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def board_guard():
    require(platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"),
            "Real captured features and checkpoint execution are RK3588 board-only")
    require(b"rk3588" in Path("/proc/device-tree/compatible").read_bytes(),
            "An RK3588/RK3588S device tree is required")


def build_patch_inputs(fmap, imap, ops, *, patches_per_view=96, seed=26):
    """CPU synthetic-testable setup; real inputs are only loaded after board guard."""
    import torch
    require(fmap.device.type == imap.device.type == "cpu" and fmap.dtype == imap.dtype == torch.float32,
            "CPU FP32 captured feature tensors required")
    require(fmap.ndim == imap.ndim == 5 and fmap.shape[:3] == (1,3,128)
            and imap.shape[:3] == (1,3,384) and fmap.shape[-2:] == imap.shape[-2:],
            "Exactly three captured views and original 128/384 feature channels required")
    height, width = fmap.shape[-2:]
    require(min(height,width) >= 4 and type(patches_per_view) is int and patches_per_view > 0,
            "Positive patch count and an interior sampling grid required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Same x-then-y integer interior sampling formula as original Patchifier.
    # CPU RNG is deliberately not claimed byte-identical to original CUDA RNG.
    x = torch.randint(1,width-1,(3,patches_per_view),generator=generator)
    y = torch.randint(1,height-1,(3,patches_per_view),generator=generator)
    centers = torch.stack((x,y),-1).float()
    gmap = ops.patchify(fmap[0],centers,1).reshape(1,-1,128,3,3)
    contexts = ops.patchify(imap[0],centers,0).reshape(1,-1,384)
    gy,gx = torch.meshgrid(torch.arange(height),torch.arange(width),indexing="ij")
    grid = torch.stack((gx,gy,torch.ones_like(gx)),0).float()[None].expand(3,-1,-1,-1)
    patches = ops.patchify(grid,centers,1).reshape(1,-1,3,3,3)
    return dict(centers=centers,gmap=gmap,contexts=contexts,patches=patches,
                source_frame=torch.arange(3,dtype=torch.int64).repeat_interleave(patches_per_view),
                patches_per_view=patches_per_view)


def make_scenario(patch_inputs, name):
    import torch
    count = patch_inputs["patches_per_view"]
    if name == "first_motion_0_to_1":
        kk = torch.arange(count,dtype=torch.int64)
        jj = torch.ones_like(kk)
    elif name == "three_frame_all_pairs":
        kk = torch.arange(3*count,dtype=torch.int64).repeat_interleave(3)
        jj = torch.arange(3,dtype=torch.int64).repeat(3*count)
    else:
        raise ValueError("Unknown fixed graph scenario")
    ii = patch_inputs["source_frame"][kk]
    # Uniform captured K across views and identity poses imply these original
    # feature-grid xy values. No pose/depth estimation or Lie-library call.
    coords = patch_inputs["patches"][:,kk,:2].contiguous()
    return dict(ii=ii,jj=jj,kk=kk,coords=coords,
                ctx=patch_inputs["contexts"][:,kk].contiguous(),
                net0=torch.zeros(1,len(kk),384,dtype=torch.float32))


def load_features(capture_dir, captured, manifest, pin):
    import numpy as np
    import torch
    require(captured.get("status") == "completed_encoder_cpu_probe_not_VO"
            and captured.get("script_sha256") == CAPTURE_SCRIPT_SHA
            and captured.get("state_unchanged") is True
            and captured.get("sources_and_inputs_unchanged") is True
            and captured.get("cleanup_errors") == [], "A completed frozen CPU encoder capture is required")
    rows = manifest["frames"]
    require(len(rows) == len(captured["inputs"]) == 3 and [c["frame_index"] for c in captured["calls"]] == [0,0,0,0,1,2],
            "Exactly the three consecutive captured frames and six original calls are required")
    def artifact(record):
        path = (capture_dir / record["file"]).resolve(strict=True)
        require(path.is_relative_to(capture_dir.resolve()), "Capture artifact path escapes its directory")
        require(path.stat().st_size == record["bytes"], "Capture artifact size mismatch")
        pin(path,record["sha256"])
        return path
    features = {"fmap":[], "imap":[]}
    for i,row in enumerate(captured["inputs"]):
        require(row["frame_index"] == i and row["frame_id"] == rows[i]["frame_id"]
                and row["source"] == rows[i], "Captured frame identity/order differs from the pinned manifest")
        input_path = artifact(row["input"])
        prepared = np.load(input_path,allow_pickle=False)
        require(meta(prepared) == row["array"] and prepared.shape == (1,1,3,464,608)
                and prepared.dtype == np.float32 and np.isfinite(prepared).all(), "Captured normalized input differs")
        selected = [call for call in captured["calls"] if call["frame_index"] == i]
        require(all(c["frame_id"] == rows[i]["frame_id"] and c["arrays"] == selected[0]["arrays"] for c in selected),
                "Captured repeated features or frame identity differ")
        for call in selected:
            artifact(call["artifact"])
        with np.load(capture_dir/selected[0]["artifact"]["file"],allow_pickle=False) as values:
            require(set(values.files) == {"fmap","imap"}, "Unexpected encoder artifact keys")
            for key,channels in (("fmap",128),("imap",384)):
                value = values[key]
                require(value.shape == (1,1,channels,116,152) and value.dtype == np.float32
                        and np.isfinite(value).all() and meta(value) == selected[0]["arrays"][key],
                        "Captured feature shape/dtype/finite/raw SHA mismatch: "+key)
                features[key].append(value.copy())
    return {key:torch.from_numpy(np.concatenate(values,axis=1)) for key,values in features.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("common","cpu-ops","source-dir","source-manifest","checkpoint","input-manifest","capture-dir","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("common","cpu-ops","source-manifest","checkpoint","input-manifest","capture"):
        parser.add_argument("--"+name+"-sha256",required=True)
    parser.add_argument("--seed",type=int,default=26)
    args = parser.parse_args(argv)
    board_guard()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report = dict(schema="v26_DPVO_fixed_identity_dynamic_neural_probe", status="failed",
                  script_sha256=sha(__file__), machine=platform.machine(), seed=args.seed,
                  actual_board_execution=True, model_name="DPVO", original_DPVO_weights=True,
                  original_VGGT_weights_used=False, production_changed=False, ORB_used=False,
                  CNN_capture_included_in_dynamic_timing=False, NPU_used=False, CUDA_used=False,
                  BA_executed=False, tracked_trajectory=False, valid_pose_output=False,
                  VO_accuracy_validated=False, realtime_validated=False, scenarios=[],cleanup_errors=[])
    pins = {}
    report["source_and_input_hashes"] = pins
    def pin(path,expected):
        path = Path(path).resolve(strict=True)
        actual = sha(path)
        require(actual == expected,"Pinned file SHA mismatch: "+str(path))
        pins[str(path)] = actual
    error, neural, before = None, None, None
    affinity = os.sched_getaffinity(0)
    try:
        require(args.common_sha256 == COMMON_SHA and args.cpu_ops_sha256 == CPU_OPS_SHA,
                "This probe uses only the frozen original common and verified CPU operators")
        pin(Path(__file__),report["script_sha256"])
        for name in ("common","cpu_ops","source_manifest","checkpoint","input_manifest"):
            pin(getattr(args,name),getattr(args,name+"_sha256"))
        pin(args.capture_dir/"capture.json",args.capture_sha256)
        captured = json.loads((args.capture_dir/"capture.json").read_text())
        require(captured["checkpoint_sha256"] == args.checkpoint_sha256
                and captured["input_manifest_sha256"] == args.input_manifest_sha256,
                "Capture weights/input manifest do not match this run")
        for path,digest in captured["source_pins"].items():
            pin(Path(path),digest)
        require(captured["source_pins"].get(str(args.common.resolve())) == args.common_sha256
                and captured["source_pins"].get(str(args.cpu_ops.resolve())) == args.cpu_ops_sha256,
                "Capture did not use the exact common and CPU operator implementation")
        source_manifest = json.loads(args.source_manifest.read_text())
        for row in source_manifest["retrieved_files"]:
            if "sha256" in row:
                pin(args.source_dir/row["path"],row["sha256"])
        manifest = json.loads(args.input_manifest.read_text())
        require(manifest.get("schema") == "v26_DPVO_original_TUM_three_frames", "Wrong fixed input manifest schema")
        os.sched_setaffinity(0,{4,5,6,7})
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
                      torch_interop_threads=torch.get_num_interop_threads(),versions=dict(torch=torch.__version__,numpy=np.__version__),
                      OMP_WAIT_POLICY=os.environ.get("OMP_WAIT_POLICY"),GOMP_SPINCOUNT=os.environ.get("GOMP_SPINCOUNT"),
                      capture_sha256=args.capture_sha256,frame_ids=[r["frame_id"] for r in manifest["frames"]])
        ops = module(args.cpu_ops,"_v26_motion_cpu_ops")
        common = module(args.common,"_v26_motion_common")
        begin = time.monotonic()
        neural,provenance = common.load_neural(args.source_dir,args.source_manifest,args.checkpoint,
                                             args.checkpoint_sha256,args.cpu_ops,cpu_ops_sha256=args.cpu_ops_sha256)
        neural.eval().requires_grad_(False)
        report.update(loader=provenance,load_seconds=time.monotonic()-begin)
        before = {k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
        require(before == captured["state_before"],"The captured features must use these exact original 94 weight tensors")
        report["original_weights_before"] = before
        features = load_features(args.capture_dir.resolve(),captured,manifest,pin)
        def save_npz(filename, values):
            arrays={key:value.detach().cpu().numpy().copy() for key,value in values.items()}
            require(all(np.isfinite(value).all() for value in arrays.values()),"Nonfinite evidence array")
            path=args.output_dir/filename
            np.savez(path,**arrays)
            return dict(file=filename,bytes=path.stat().st_size,sha256=sha(path),arrays={key:meta(value) for key,value in arrays.items()})
        with torch.inference_mode():
            begin=time.monotonic()
            patch_inputs=build_patch_inputs(features["fmap"],features["imap"],ops,seed=args.seed)
            report["shared_patch_setup_seconds"]=time.monotonic()-begin
            report["patches"]=save_npz("patch-inputs.npz",{k:patch_inputs[k] for k in ("centers","gmap","contexts","patches","source_frame")})
            for name in SCENARIOS:
                begin=time.monotonic()
                graph=make_scenario(patch_inputs,name)
                setup=time.monotonic()-begin
                graph_artifact=save_npz(name+"-graph.npz",graph)
                scene=dict(name=name,edge_count=graph["kk"].numel(),patches_per_view=96,
                           graph_setup_seconds=setup,graph=graph_artifact,calls=[],
                           geometry="Analytical identity projection: original patch xy coordinates, unit inverse depth; no estimated poses",
                           edge_order="source patch then target view; first scenario source view0 to target view1",
                           feedback_iterations=0, fresh_zero_net_each_call=True, timing_scope="Independent CPU corr_two_levels including avg-pool pyramid, followed by the entire original Update. Graph/context/net0 setup, CNNs, evidence validation/copies/hash/save and BA are excluded.")
                report["scenarios"].append(scene)
                reference=None
                for call in range(4):
                    begin=time.monotonic()
                    correlation=ops.corr_two_levels(patch_inputs["gmap"],features["fmap"],graph["coords"],graph["kk"],graph["jj"],radius=3)
                    after_corr=time.monotonic()
                    net,(delta,weight,unused)=neural.update(graph["net0"],graph["ctx"],correlation,None,graph["ii"],graph["jj"],graph["kk"])
                    after_update=time.monotonic()
                    edge_count=scene["edge_count"]
                    require(correlation.shape==(1,edge_count,882) and net.shape==(1,edge_count,384)
                            and delta.shape==weight.shape==(1,edge_count,2) and unused is None,
                            "Original Update output contract mismatch")
                    require(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in (correlation,net,delta,weight)),
                            "Original correlation/update produced invalid FP32 outputs")
                    require(bool(((weight>=0)&(weight<=1)).all()),"Original sigmoid weight outside [0,1]")
                    artifact=save_npz(f"{name}-call-{call:02d}.npz",dict(corr=correlation,net=net,delta=delta,weight=weight))
                    if reference is None: reference=artifact["arrays"]
                    require(artifact["arrays"]==reference,"Repeated fixed graph output changed")
                    record=dict(call_index=call,timing_role="cold" if call==0 else "warm",corr_seconds=after_corr-begin,
                                update_seconds=after_update-after_corr,total_dynamic_seconds=after_update-begin,
                                call_begin_monotonic=begin,correlation_end_monotonic=after_corr,update_end_monotonic=after_update,
                                artifact=artifact,repeat_bitwise_equal=True,
                                delta_norm_median_feature_pixels=float(torch.quantile(delta.norm(dim=-1),.5)))
                    scene["calls"].append(record)
                    print(json.dumps(dict(scenario=name,call=call,corr_seconds=record["corr_seconds"],update_seconds=record["update_seconds"],total_dynamic_seconds=record["total_dynamic_seconds"])),flush=True)
                scene["warm_means"]={key:sum(r[key] for r in scene["calls"][1:])/3 for key in ("corr_seconds","update_seconds","total_dynamic_seconds")}
                require({key:meta(value.detach().numpy()) for key,value in graph.items()}==graph_artifact["arrays"],
                        "Update mutated the fixed initial graph/context/net0")
        report.update(status="completed_dynamic_neural_probe_not_VO",repeat_outputs_bitwise_equal=True,
                      source_feature_scaling="Already /4 exactly once in original encoder capture; not re-scaled here",
                      limitations=["Fixed identity graph with independently sampled CPU patches; no actual tracked trajectory or BA/pose output.",
                                   "Captured CNN time, decoding and graph setup are outside corr+Update timing; per-frame end-to-end VO rate is unmeasured.",
                                   "CPU RNG/reduction results are not claimed CUDA byte-exact or an NPU execution result.",
                                   "Cold+3 warm repeats of the same features are workload observations, not a sustained real-time benchmark."])
    except BaseException as exc:
        error=exc
        report.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
    finally:
        try:
            if neural is not None and before is not None:
                report["original_weights_unchanged"] = before=={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
                require(report["original_weights_unchanged"],"Original weights changed")
            report["source_hashes_after"]={path:sha(path) for path in pins}
            report["sources_and_inputs_unchanged"]=report["source_hashes_after"]==pins
            require(report["sources_and_inputs_unchanged"],"Pinned source or input changed")
        except BaseException as exc:
            report["cleanup_errors"].append(repr(exc));error=error or exc
        try:
            os.sched_setaffinity(0,affinity)
            report["affinity_restored"]=os.sched_getaffinity(0)==affinity
            require(report["affinity_restored"],"Affinity restoration failed")
        except BaseException as exc:
            report["cleanup_errors"].append(repr(exc));error=error or exc
        if error is not None: report["status"]="failed"
        report["peak_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        receipt=args.output_dir/"motion.json"
        receipt.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["status"],error=report.get("error"),receipt=str(receipt),receipt_sha256=sha(receipt))),flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
