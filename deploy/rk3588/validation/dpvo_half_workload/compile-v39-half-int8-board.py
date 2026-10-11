#!/usr/bin/env python3
"""Board-only PTQ of the CPU-checked half-resolution M16 DPVO encoder.

MIT project helper; DPVO copyright (c) 2022 Princeton Vision & Learning Lab.
SDK build(do_quantization=True) executes real calibration on the board CPU.
Never run this entrypoint on a host or another ARM device. Historical full-size
rejection is retained; new-size CPU gates and future INT8 gates are separate.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

POLICY_SHA = "cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a"
EDITOR_SHA = "63a45d4b20510261cc18f9392d114b4a99aeb8cedea962321a1e398499f8a091"
PROBE_SHA = "01e4e81c71216a491b2d03200201a0e55dc3fcfc06bc94c3075d638ff8949e4f"
PTQ_SHA = "0fc013d5b6ed45cb97d592e8cd115304d971a000e2ab5353314413f56d41a3fa"
BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
SUCCESS = "compiled_board_half_M16_PTQ_not_runtime_validated_historical_rejection_retained"
CAL_INDICES = [256+16*i for i in range(32)]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve()
    require(isinstance(expected,str) and len(expected)==64 and all(c in "0123456789abcdef" for c in expected) and sha(path)==expected,
            "Source/receipt SHA mismatch: "+path.name)
    pins[str(path)] = expected
    return path


def load(path, expected, name, pins):
    spec = importlib.util.spec_from_file_location(name, pin(path,expected,pins))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def board_guard(machine=None, system=None, compatible=None):
    machine = platform.machine() if machine is None else machine
    system = platform.system() if system is None else system
    if compatible is None:
        path = Path("/proc/device-tree/compatible")
        compatible = path.read_bytes() if path.is_file() else b""
    require(machine=="aarch64" and system=="Linux" and any(v in (b"rockchip,rk3588",b"rockchip,rk3588s") for v in compatible.split(b"\0")),
            "Host PTQ prohibited: calibration is neural execution; RK3588/S Linux board required")


def unchanged(value):
    require(value["source_pins"]==value["source_hashes_after"] and value["sources_and_inputs_unchanged"] is True and value["cleanup_errors"]==[],
            "Source/cleanup evidence must be unchanged and clean")


def probe_guard(value, policy, rewrite_sha, graph_sha):
    unchanged(value)
    require(value["schema"]=="v39_DPVO_half_encoder_board_probe" and value["mode"]=="probe" and
            value["status"]=="completed_half_M16_CPU_teacher_and_ORT_diagnostic_not_VO" and value["script_sha256"]==PROBE_SHA and
            value["policy_sha256"]==POLICY_SHA and value["rewrite_sha256"]==rewrite_sha and
            graph_sha in value["source_pins"].values(), "Same new-size actual CPU teacher/probe chain required")
    require(value["preprocessing"]==policy["preprocessing"] and value["rng_policy"]==policy["rng"] and
            value["normalization_applications_per_image"]==1 and value["old_resolution_outputs_used_as_new_gold"] is False and
            value["original_FP32_gate"]==dict(rtol=1e-4,atol=1e-5), "New-size teacher and original strict gate required")
    require(value["state_before"]==value["state_after"] and len(value["state_before"])==94 and value["state_unchanged"] is True and
            value["shape_preservation_rechecked"] is True and value["model_forward_executed"] is True,
            "Original94 parameter bytes/new-size source checks required")
    require(value["providers"]==["CPUExecutionProvider"] and value["graph_optimization"]=="ORT_DISABLE_ALL" and
            value["cpu_affinity"]==[4,5,6,7] and value["affinity_restored"] is True and value["session_references_cleared"] is True,
            "CPU-only original gate/session cleanup required")
    require(value["historical_parent_FP32_gate_passed"] is False and value["historical_parent_rejection_retained"] is True and
            value["candidate_accepted"] is False and len(value["calls"])==3 and value["samples"]==value["calls"],
            "Exactly three new-size source images; historical rejection retained")
    for i,row in enumerate(value["calls"]):
        require(row["index"]==row["source_index"]==i and row["image_array"]["shape"]==[1,3,232,304] and row["index_array"]["shape"]==[16],
                "Same first-three half M16 inputs required")
        for method,keys in (("original",{"fmap","imap"}),("reciprocal",{"fmap","context"})):
            gates=row[method]["versus_new_size_original_CPU"]
            require(set(gates)==keys and all(g["gate_passed"] is True and g["failed_elements"]==0 and
                    g["rtol"]==1e-4 and g["atol"]==1e-5 for g in gates.values()), "New-size original/candidate CPU gate failed")
    require(value["strict_gate_each_method"]=={"original":True,"reciprocal":True}, "Both new-size teacher gates must pass")


def calibration_guard(value, policy, rewrite_sha):
    unchanged(value)
    require(value["schema"]=="v39_DPVO_half_encoder_board_calibration" and value["mode"]=="calibration" and
            value["status"]=="generated_half_M16_calibration_not_quantized_or_validated" and value["script_sha256"]==PROBE_SHA and
            value["policy_sha256"]==POLICY_SHA and value["rewrite_sha256"]==rewrite_sha, "New half M16 calibration receipt required")
    require(value["preprocessing"]==policy["preprocessing"] and value["rng_policy"]==policy["rng"] and
            value["model_forward_executed"] is False and value["GT_read"] is False and value["GT_or_content_selection"] is False and
            value["normalization_applications_per_image"]==1 and value["calls"]==[], "Calibration preprocessing/selection must be declared")
    require(value["calibration_source_indices"]==CAL_INDICES and value["evaluation_source_indices"]==list(range(192)) and
            len(value["samples"])==32 and [r["source_index"] for r in value["samples"]]==CAL_INDICES,
            "Fixed32 calibration indices must remain disjoint from first192")
    require(value["candidate_accepted"] is False and value["historical_parent_rejection_retained"] is True and
            value["historical_parent_FP32_gate_passed"] is False, "Calibration does not promote historical parent")


def validate_arrays(image, indices):
    import numpy as np
    require(image.dtype==np.float32 and image.shape==(1,3,232,304) and np.isfinite(image).all() and
            image.min()>=-.5 and image.max()<=1.5, "New once-normalized FP32 half image required")
    require(indices.dtype==np.int64 and indices.shape==(16,), "M16 INT64 indices required without coercion")
    y,x=indices//76,indices%76
    require(np.all((x>=1)&(x<75)&(y>=1)&(y<57)), "Interior centers on new58x76 grid")


def artifact(directory, row, pins):
    require(Path(row["file"]).name==row["file"], "Simple relative artifact name required")
    path=pin(Path(directory)/row["file"],row["sha256"],pins)
    require(path.stat().st_size==row["bytes"], "Artifact byte count mismatch")
    return path


def prepare_dataset(directory, calibration, expected_sha, output, pins, metadata):
    import numpy as np
    dataset=artifact(directory,calibration["dataset"],pins)
    require(sha(dataset)==expected_sha, "New relative dataset list SHA mismatch")
    lines=dataset.read_text().splitlines()
    require(len(lines)==32, "Exactly32 two-input calibration rows required")
    sampling_path=artifact(directory,calibration["sampling_artifact"],pins)
    sampling=np.load(sampling_path,allow_pickle=False)
    names={"centers_xy","consumed_inverse_depth","state_before","state_after_centers","state_after_depth"}
    require(set(sampling.files)==names, "Actual new-size RNG archive keys")
    try:
        require(all(metadata(sampling[k])==calibration["sampling_artifact"]["arrays"][k] for k in names), "Sampling raw arrays metadata mismatch")
        require(sampling["centers_xy"].shape==(32,16,2) and sampling["centers_xy"].dtype==np.int64 and
                sampling["consumed_inverse_depth"].shape==(32,16) and sampling["consumed_inverse_depth"].dtype==np.float32 and
                np.isfinite(sampling["consumed_inverse_depth"]).all() and np.all((sampling["consumed_inverse_depth"]>=0)&(sampling["consumed_inverse_depth"]<1)),
                "New M16 actual RNG archive shapes/range")
        states=[sampling[k] for k in ("state_before","state_after_centers","state_after_depth")]
        require(all(s.dtype==np.uint8 and s.ndim==2 and s.shape[0]==32 and s.shape[1]>0 for s in states) and
                all(np.array_equal(states[2][i],states[0][i+1]) for i in range(31)), "Persistent calibration RNG state continuity")
        rows,resolved=[],[]
        for i,(line,record) in enumerate(zip(lines,calibration["samples"])):
            require(line.split()==[f"image-{i:02d}.npy",f"indices-{i:02d}.npy"] and record["index"]==i,
                    "Graph order is new image then untouched INT64 indices")
            files=[artifact(directory,f,pins) for f in record["input_files"]]
            require([p.name for p in files]==line.split(), "Dataset/receipt file order mismatch")
            image,indices=[np.load(p,allow_pickle=False) for p in files]
            validate_arrays(image,indices)
            require(metadata(image)==record["image_array"] and metadata(indices)==record["index_array"], "Raw calibration input metadata mismatch")
            centers=sampling["centers_xy"][i]
            require(np.all((centers[:,0]>=1)&(centers[:,0]<75)&(centers[:,1]>=1)&(centers[:,1]<57)) and
                    np.array_equal(indices,centers[:,1]*76+centers[:,0]) and all(metadata(sampling[k][i])==record["RNG"][k] for k in names),
                    "Indices/centers/RNG must match actual per-row archive")
            require(metadata(indices)==record["RNG"]["patch_indices"], "Untouched INT64 raw bytes")
            rows.append(dict(index=i,source_index=record["source_index"],frame_id=record["frame_id"],source_image_sha256=record["source_image_sha256"],
                input_files=[dict(file=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],image_array=record["image_array"],index_array=record["index_array"]))
            require(not any(any(c.isspace() for c in str(p)) for p in files), "SDK dataset paths must not contain whitespace")
            resolved.append(" ".join(map(str,files)))
        require(sum(f["bytes"] for r in rows for f in r["input_files"])<256*1024**2, "Calibration input budget256MiB")
    finally:
        sampling.close()
    path=output/"dataset-resolved.txt";path.write_text("\n".join(resolved)+"\n");pins[str(path)]=sha(path)
    return path,rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("policy","dependency-plan","preflight-script","frozen-ptq-helper","half-editor","half-probe-driver",
                 "rewrite-dir","probe-dir","calibration-dir","original-onnx","reciprocal-onnx","reciprocal-recipe","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("dependency-plan","preflight-script","rewrite","probe","calibration","dataset"):
        p.add_argument("--"+name+"-sha256",required=True)
    p.add_argument("--diagnostic-allow-fp32-rejected",action="store_true")
    args=p.parse_args(argv)
    board_guard() # Before output, model/SDK/NumPy/ONNX imports or data loading.
    require(args.diagnostic_allow_fp32_rejected, "Explicit retained historical FP32 rejection diagnostic flag required")
    output=args.output_dir.resolve()
    require(output.is_relative_to(BASE) and output!=BASE and not output.exists(), "New separate board PTQ directory")
    protected=[v.resolve() for k,v in vars(args).items() if isinstance(v,Path) and k!="output_dir"]
    require(not any(output==v or output.is_relative_to(v) or v.is_relative_to(output) for v in protected), "Output/source overlap")
    output.mkdir(parents=True,exist_ok=False)
    pins={str(Path(__file__).resolve()):sha(__file__)}
    report=dict(schema="v39_board_only_DPVO_half_M16_encoder_PTQ",status="failed",script_sha256=sha(__file__),
        frozen_PTQ_helper_sha256=PTQ_SHA,policy_sha256=POLICY_SHA,diagnostic_only=True,diagnostic_allow_fp32_rejected=True,
        historical_parent_FP32_gate_passed=False,historical_parent_rejection_retained=True,candidate_accepted=False,
        production_changed=False,host_calibration_executed=False,calibration_location="RK3588/S aarch64 board CPU",
        PTQ_requested=True,deployment_inference_executed=False,INT8_operator_coverage_validated=False,
        source_pins=pins,operations=[],released=False,cleanup_errors=[],cpu_affinity=sorted(os.sched_getaffinity(0)))
    start=time.monotonic()
    def receipt(path,digest):return json.loads(pin(path,digest,pins).read_text())
    try:
        base=load(args.frozen_ptq_helper,PTQ_SHA,"_v39_frozen_PTQ",pins)
        probe_driver=load(args.half_probe_driver,PROBE_SHA,"_v39_frozen_probe",pins)
        editor=load(args.half_editor,EDITOR_SHA,"_v39_frozen_editor",pins)
        policy=receipt(args.policy,POLICY_SHA)
        require(policy["image_size_hw"]==[232,304] and policy["stream_config"]["patches_per_frame"]==16, "Single fixed half M16 policy")
        recipe=receipt(args.rewrite_dir/"rewrite.json",args.rewrite_sha256)
        require(recipe["schema"]=="v39_DPVO_half_resolution_encoder_shape_rewrite" and recipe["script_sha256"]==EDITOR_SHA and
                recipe["source_pins"]==recipe["source_hashes_after"] and recipe["sources_and_parents_unchanged"] is True and
                recipe["original_weights_preserved"] is True and recipe["historical_parent_rejection_retained"] is True and
                recipe["historical_parent_FP32_gate_passed"] is False and recipe["candidate_accepted"] is False,
                "Frozen new-size shape-only rewrite chain required")
        candidate=recipe["models"]["reciprocal"];graph=artifact(args.rewrite_dir,candidate,pins)
        parent=pin(args.reciprocal_onnx,editor.RECIPROCAL_SHA,pins)
        pin(args.original_onnx,editor.ORIGINAL_SHA,pins);pin(args.reciprocal_recipe,editor.RECIPROCAL_RECIPE_SHA,pins)
        import onnx
        model=onnx.load(graph,load_external_data=False)
        proof=editor.validate_preservation(onnx.load(parent,load_external_data=False),model,"reciprocal")
        require(proof==candidate["preservation"] and editor.contract(model.graph.input)==candidate["inputs"] and
                editor.contract(model.graph.output)==candidate["outputs"], "Exact half-size original weights/shape-only IO")
        onnx.checker.check_model(model,full_check=True)
        probe=receipt(args.probe_dir/"probe.json",args.probe_sha256)
        probe_guard(probe,policy,args.rewrite_sha256,candidate["sha256"])
        for row in probe["calls"]:
            for key in ("input_artifact","teacher_artifact"):artifact(args.probe_dir,row[key],pins)
            for method in ("original","reciprocal"):artifact(args.probe_dir,row[method]["artifact"],pins)
        calibration=receipt(args.calibration_dir/"calibration.json",args.calibration_sha256)
        calibration_guard(calibration,policy,args.rewrite_sha256)
        dataset,rows=prepare_dataset(args.calibration_dir,calibration,args.dataset_sha256,output,pins,probe_driver.meta)
        plan=receipt(args.dependency_plan,args.dependency_plan_sha256)
        preflight=load(args.preflight_script,args.preflight_script_sha256,"_v39_frozen_preflight",pins)
        report["environment"]=preflight.inspect_environment(plan)
        require(report["environment"]["status"]=="ready_for_board_PTQ_diagnostic", "Isolated ARM toolkit environment not ready")
        report.update(rewrite_sha256=args.rewrite_sha256,editor_sha256=EDITOR_SHA,probe_sha256=args.probe_sha256,
            probe_driver_sha256=PROBE_SHA,calibration_receipt_sha256=args.calibration_sha256,
            original_DPVO_checkpoint_sha256=probe["original_DPVO_checkpoint_sha256"],new_size_CPU_strict_gate_each_method=probe["strict_gate_each_method"],
            graph_preservation=proof,input=dict(file=str(graph),sha256=candidate["sha256"],bytes=graph.stat().st_size,
                inputs=candidate["inputs"],outputs=candidate["outputs"],original_graph_weights_preserved=True),
            configuration={**base.CONFIGURATION,"do_quantization":True,"auto_hybrid":False},
            calibration=dict(samples=32,source_dataset_sha256=args.dataset_sha256,dataset_sha256=sha(dataset),rows=rows,
                image_shape=[1,3,232,304],index_shape=[16],indices_dtype="int64",normalization_applications_per_image=1,
                source_indices=CAL_INDICES,evaluation_disjoint_first192=True,GT_or_content_selection=False,
                new_calibration_not_old_full_resolution_dataset=True,RNG_archive_and_input_raw_metadata_verified=True,
                preprocessing_provenance="Frozen board-generator source and saved hashes; not redecoded by PTQ entry"))
        from rknn.api import RKNN
        base.perform_ptq(RKNN,graph,dataset,output,report)
        report["status"]=SUCCESS
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        report["source_hashes_after"]={p:sha(p) if Path(p).is_file() else None for p in pins}
        report["sources_and_inputs_unchanged"]=report["source_hashes_after"]==pins
        report.update(actual_driver_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if not report["sources_and_inputs_unchanged"] or report["cleanup_errors"]:report["status"]="failed"
        (output/"compile.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k) for k in ("status","error","released")}),flush=True)
    return 0 if report["status"]==SUCCESS else 1


if __name__=="__main__":
    raise SystemExit(main())
