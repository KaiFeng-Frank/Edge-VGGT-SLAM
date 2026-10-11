#!/usr/bin/env python3
"""Saved NumPy arrays and static ONNX audit; no neural/ORT/native execution.

DPVO topology derives from Princeton-VL/DPVO 0ac95b656d1fda91c271d2a106460d19ad966fc7,
MIT Copyright (c) 2022 Princeton University. Keep the adjacent DPVO LICENSE.
Group outputs were saved only after full-edge scatter, not as separate raw
callback arrays: the independent check reconstructs their groups from those
full outputs, and does not pretend to read additional callback payloads.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

DRIVER_SHA = "473bd9441c983c08aa3ab3a97a7f14b82da8f3d508690403b08468ba25d2dc72"
CAPTURE_SHA = "3b53b69bda5c91bf94b359aa318007000daf263469e1ae85e3fd092428675460"
CAPTURE_AUDIT_SHA = "aaf9fd58729519674b3995a4036e2791d160bd5d10dbc8bf05f42528203d3b98"
INPUTS = ("net", "inp", "corr")
OUTPUTS = ("net", "delta", "weight")


def require(value, name):
    if not value:
        raise ValueError(name)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def comparison(actual, reference):
    """Original asymmetric FP32 gate, with bounded FP64 statistic scratch."""
    require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32,
            "Exact FP32 comparison shape/dtype")
    a, b = actual.ravel(), reference.ravel()
    count, maximum, norm_error, norm_reference, worst = 0, 0., 0., 0., 0.
    failures = []
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Finite comparison arrays")
    for start in range(0, a.size, 262144):
        x, y = a[start:start+262144].astype(np.float64), b[start:start+262144].astype(np.float64)
        error, allowed = x-y, 1e-5+1e-4*np.abs(y)
        absolute = np.abs(error)
        bad = np.flatnonzero(absolute > allowed)
        count += int(bad.size)
        maximum = max(maximum, float(absolute.max(initial=0)))
        worst = max(worst, float((absolute/allowed).max(initial=0)))
        norm_error += float(np.sum(error*error))
        norm_reference += float(np.sum(y*y))
        for i in bad[:max(0, 8-len(failures))]:
            failures.append(dict(coordinate=[int(v) for v in np.unravel_index(start+int(i), actual.shape)],
                actual=float(x[i]), reference=float(y[i]), abs_error=float(absolute[i]),
                allowed_error=float(allowed[i]), normalized_error=float(absolute[i]/allowed[i])))
    return dict(finite=True, gate_passed=count == 0, failed_elements=count, max_abs=maximum,
        relative_L2=float(np.sqrt(norm_error/norm_reference)) if norm_reference else None,
        bitwise_equal=actual.tobytes(order="C") == reference.tobytes(order="C"),
        rtol=1e-4, atol=1e-5, max_normalized_error=worst, failing_elements=failures)


def stats_match(actual, recorded):
    exact = ("finite", "gate_passed", "failed_elements", "bitwise_equal", "rtol", "atol", "max_abs")
    if not all(actual[k] == recorded[k] for k in exact):
        return False
    x, y = actual["relative_L2"], recorded["relative_L2"]
    return x is y if x is None or y is None else bool(np.isclose(x, y, rtol=1e-11, atol=1e-15))


def independent_batches(ii, jj, kk, patches=48):
    """Dictionary Cartesian reconstruction, independent of producer lexsort."""
    require(all(isinstance(v, np.ndarray) and v.dtype == np.int64 and v.ndim == 1 for v in (ii,jj,kk))
            and ii.shape == jj.shape == kk.shape and ii.size > 0, "Nonempty ordered INT64 topology")
    require(np.all((ii >= 0) & (ii < 64) & (jj >= 0) & (jj < 64))
            and np.array_equal(ii, kk//patches) and np.all((kk >= 0) & (kk < patches*64)), "Patch/source ownership")
    lookup = {}
    for edge, (source,target,patch) in enumerate(zip(ii.tolist(), jj.tolist(), (kk % patches).tolist())):
        key = (source,target,patch)
        require(key not in lookup, "Duplicate source/target/patch edge")
        lookup[key] = edge
    by_degree = {}
    for source in sorted(set(ii.tolist())):
        targets = sorted(set(jj[ii == source].tolist()))
        require(0 < len(targets) <= 25, "Declared complete target degree")
        try:
            indices = np.asarray([[lookup[source,target,patch] for patch in range(patches)]
                                  for target in targets], dtype=np.int64)
        except KeyError as error:
            raise ValueError("Missing member of complete Cartesian source group") from error
        by_degree.setdefault(len(targets), []).append((source,targets,indices))
    result = []
    for degree, members in sorted(by_degree.items()):
        result.append(dict(degree=degree, patches=patches, source_slots=[v[0] for v in members],
            target_slots=[v[1] for v in members], indices=np.stack([v[2] for v in members])))
    restored = np.concatenate([v["indices"].ravel() for v in result])
    require(np.array_equal(np.sort(restored), np.arange(ii.size)), "Every original edge exactly once")
    return result


def batch_metadata(batch):
    return dict(source_slots=batch["source_slots"],target_slots=batch["target_slots"],
        sources=len(batch["source_slots"]),degree=batch["degree"],patches=batch["patches"],
        edges=int(batch["indices"].size),padding=False,dependency_groups_split=False,
        physical_layout="source_batch,channels,target,patch")


def pack(value, batch):
    return np.ascontiguousarray(value[0,batch["indices"],:].transpose(0,3,1,2))


def scatter(groups, batches, edges, channels):
    result = np.full((1,edges,channels),np.nan,np.float32)
    for value,batch in zip(groups,batches):
        require(value.dtype == np.float32 and value.shape == (len(batch["source_slots"]),channels,batch["degree"],batch["patches"]),
                "Matching physical CF group output")
        result[0,batch["indices"].ravel(),:] = value.transpose(0,2,3,1).reshape(-1,channels)
    require(np.isfinite(result).all(), "All output edges restored")
    return result


def parameter_coverage(graph, parameters, onnx):
    """Every initializer's actual raw bytes bound to original parameter hash."""
    covered, mapping = set(), []
    for tensor in graph.graph.initializer:
        require(not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL, "No external initializer data")
        value = onnx.numpy_helper.to_array(tensor)
        info, matches = meta(value), []
        for name, original in parameters.items():
            shapes = [("identity",original["shape"])]
            if len(original["shape"]) == 2: shapes.append(("conv1x1_reshape",original["shape"]+[1,1]))
            if len(original["shape"]) == 1: shapes.append(("channel_affine_reshape",[1,original["shape"][0],1,1]))
            for transform,shape in shapes:
                if info["dtype"] == original["dtype"] and info["bytes"] == original["bytes"] and info["shape"] == shape and info["tensor_sha256"] == original["tensor_sha256"]:
                    matches.append(dict(parameter=name,transform=transform));covered.add(name)
        require(matches, "Initializer not original parameter bytes: "+tensor.name)
        mapping.append(dict(initializer=tensor.name,array=info,original_matches=matches))
    require(covered == set(parameters), "All original parameter identities represented")
    return dict(original_parameter_count=len(parameters),all_original_parameters_covered=True,mapping=mapping)


def io_contract(value):
    t = value.type.tensor_type
    require(t.elem_type == 1 and all(d.HasField("dim_value") and d.dim_value > 0 for d in t.shape.dim), "Fixed FP32 ONNX port")
    return dict(name=value.name,dtype="float32",shape=[d.dim_value for d in t.shape.dim])


def audit(export_dir,capture_dir,capture_audit,parent_dir,driver,*,manifest_sha256):
    folder,cap = Path(export_dir),Path(capture_dir)
    report=dict(schema="v37_saved_full_M48_grouped_Update_export_audit",status="failed",auditor_sha256=sha(__file__),
        neural_ORT_native_BA_or_board_executed=False,candidate_accepted=False,checks=[],cases=[],models={},
        separate_group_output_payloads_available=False,independent_gold_forward_executed=False)
    def check(value,name):
        report["checks"].append(dict(name=name,passed=bool(value)))
        require(value,name)
    try:
        manifest_path=folder/"manifest.json"
        check(sha(manifest_path)==manifest_sha256,"Pinned actual export manifest")
        check(sha(driver)==DRIVER_SHA,"Frozen producer source SHA")
        check(sha(cap/"capture.json")==CAPTURE_SHA and sha(capture_audit)==CAPTURE_AUDIT_SHA,"Frozen original capture and independent audit")
        r=json.loads(manifest_path.read_text());capture=json.loads((cap/"capture.json").read_text())
        prior=json.loads(Path(capture_audit).read_text());parent=json.loads((Path(parent_dir)/"run.json").read_text())
        check(sha(Path(parent_dir)/"run.json")==capture["original_parent_receipt"]["run_sha256"]==prior["input_bindings"]["parent_run_sha256"],"Actual historical fullUpdate parent binding")
        check(r["schema"]=="v36_full_M48_grouped_Update_CPU_ONNX_diagnostic" and r["script_sha256"]==DRIVER_SHA and r["capture_sha256"]==CAPTURE_SHA and r["capture_audit_sha256"]==CAPTURE_AUDIT_SHA,"Actual producer experiment identity")
        check(r["cleanup_errors"]==[] and r["affinity_restored"] and r["ORT_session_references_released"] and r["sources_and_inputs_unchanged"] and r["source_pins"]==r["source_hashes_after"],"Recorded cleanup and source/input stability")
        check(r["original_parameters_unchanged"] and r["original_weights_before"]==r["original_weights_after"]==parent["original_weights_before"] and len(r["original_weights_before"])==94,"Same original94 parameter bytes")
        parameters=r["original_Update_parameters"]
        check(len(parameters)==50 and {"update."+k:v for k,v in parameters.items()}=={k:v for k,v in r["original_weights_before"].items() if k.startswith("update.")},"All original50 Update parameters bound to94 state")
        check(r["strict_FP32_thresholds"]==dict(rtol=1e-4,atol=1e-5) and not r["strict_FP32_thresholds_relaxed"] and not r["output_clipping"] and not r["candidate_accepted"],"Original fixed gate and no adoption/clipping")
        check(not r["NPU_used"] and not r["BA_executed"] and not r["VO_state_committed"] and not r["realtime_validated"] and not r["original_full_Update_reexecuted"] and not r["source_member_padding"] and not r["neighbor_or_edge_removal"],"No unexecuted algorithm/hardware claims")
        report["input_sha256"]=dict(export_manifest=manifest_sha256,capture=CAPTURE_SHA,capture_audit=CAPTURE_AUDIT_SHA,parent=sha(Path(parent_dir)/"run.json"),driver=DRIVER_SHA)
        seen_files, used_models, full_gates, group_gates = {},set(),[],[]
        def read_artifact(directory,item,npz=True):
            require(Path(item["file"]).name==item["file"],"Safe artifact basename")
            path=Path(directory)/item["file"]
            check(sha(path)==item["sha256"] and path.stat().st_size==item["bytes"],"Actual file bytes/SHA "+item["file"])
            seen_files[str(path)]=dict(file=item["file"],sha256=item["sha256"],bytes=item["bytes"])
            if not npz:return path
            with np.load(path,allow_pickle=False) as archive: values={k:archive[k] for k in archive.files}
            check(set(values)==set(item["arrays"]),"Exact archive keys "+item["file"])
            for key,value in values.items():check(meta(value)==item["arrays"][key] and np.isfinite(value).all(),"Actual raw tensor "+item["file"]+"/"+key)
            return values
        check([c["observation_id"] for c in r["cases"]]==[13,127,191] and len(capture["records"])==3,"All predeclared actual full graph cases")
        for case,original in zip(r["cases"],capture["records"]):
            obs,edges=case["observation_id"],case["edges"]
            check(case["original_inputs_artifact"]==original["inputs_artifact"] and case["original_gold_artifact"]==original["original_gold_artifact"] and case["snapshot_identity"]==original["snapshot_identity"] and case["original_call_index"]==original["actual_original_call_index"],"Same-call full input/gold/identity obs"+str(obs))
            arrays=read_artifact(cap,case["original_inputs_artifact"]);gold=read_artifact(cap,case["original_gold_artifact"])
            torch=read_artifact(folder,case["grouped_Torch_artifact"]);ort=read_artifact(folder,case["grouped_ORT_artifact"])
            check(arrays["net"].shape==(1,edges,384) and arrays["inp"].shape==(1,edges,384) and arrays["corr"].shape==(1,edges,882),"Original full graph input shape obs"+str(obs))
            check(arrays["inp"].tobytes()==arrays["frame_contexts"].reshape(-1,384)[arrays["kk"]][None].tobytes(),"Original context from ordered kk obs"+str(obs))
            batches=independent_batches(*(arrays[k] for k in ("ii","jj","kk")))
            fingerprint=hashlib.sha256(b"".join(arrays[k].tobytes() for k in ("ii","jj","kk"))+str((48,64)).encode()).hexdigest()
            plan=dict(edges=edges,patches=48,full_source_blocks=sum(len(b["source_slots"]) for b in batches),neural_calls=len(batches),topology_sha256=fingerprint,all_edges_exactly_once=True,original_edge_order_restored=True,padding=False,batch_items_independent_sources=True,batches=[batch_metadata(b) for b in batches])
            check(plan==case["degree_plan"] and len(batches)==len(case["groups"]),"Independent trueB full-source Cartesian plan obs"+str(obs))
            summary=dict(observation_id=obs,edges=edges,true_batches=[batch_metadata(b) for b in batches],full_comparisons={},groups=[])
            for method,values in (("gold",gold),("Torch",torch),("ORT",ort)):
                check(set(values)==set(OUTPUTS),"Canonical full outputs "+method+str(obs))
                for key,channels in zip(OUTPUTS,(384,2,2)):
                    check(values[key].dtype==np.float32 and values[key].shape==(1,edges,channels),"Complete FP32 output "+method+str(obs)+key)
                    restored=scatter([pack(values[key],b) for b in batches],batches,edges,channels)
                    check(restored.tobytes()==values[key].tobytes(),"Independent inverseCF full-edge roundtrip "+method+str(obs)+key)
                check(np.all((values["weight"]>=0)&(values["weight"]<=1)),"Original output weight domain "+method+str(obs))
            for label,a,b in (("Torch_vs_original_full_gold",torch,gold),("ORT_vs_original_full_gold",ort,gold),("ORT_vs_grouped_Torch",ort,torch)):
                summary["full_comparisons"][label]={}
                for key in OUTPUTS:
                    stats=comparison(a[key],b[key]);full_gates.append(stats["gate_passed"])
                    check(stats_match(stats,case[label][key]),"Independently recomputed strict full gate "+str(obs)+label+key)
                    for failure in stats["failing_elements"]:
                        edge=failure["coordinate"][1]
                        failure.update(source_slot=int(arrays["ii"][edge]),target_slot=int(arrays["jj"][edge]),patch_id=int(arrays["kk"][edge]))
                    summary["full_comparisons"][label][key]=stats
            for number,(batch,group) in enumerate(zip(batches,case["groups"])):
                check(group["index"]==number and group["topology"]==batch_metadata(batch) and group["canonical_edge_indices"]==meta(batch["indices"]),"Original group order/permutation "+str(obs)+"/"+str(number))
                key="B%d_D%d_M48"%(len(batch["source_slots"]),batch["degree"])
                check(group["model_key"]==key,"True model B/D identity "+key);used_models.add(key)
                feeds=read_artifact(folder,group["inputs_artifact"])
                check(set(feeds)==set(INPUTS),"Exactly three physicalCF feeds")
                for name in INPUTS:check(feeds[name].tobytes()==pack(arrays[name],batch).tobytes() and feeds[name].shape==pack(arrays[name],batch).shape,"Actual independently packed group input "+str(obs)+"/"+str(number)+name)
                local_gates=[]
                for label,a,b in (("Torch_vs_original_group_gold",torch,gold),("ORT_vs_original_group_gold",ort,gold),("ORT_vs_grouped_Torch",ort,torch)):
                    for name in OUTPUTS:
                        stats=comparison(pack(a[name],batch),pack(b[name],batch))
                        check(stats_match(stats,group[label][name]),"Independently reconstructed group gate "+str(obs)+"/"+str(number)+label+name)
                        local_gates.append(stats["gate_passed"])
                group_gates.extend(local_gates)
                summary["groups"].append(dict(model_key=key,strict_gate_passed=all(local_gates)))
                del feeds
            summary["timing"]={k:case[k] for k in ("grouped_Torch_total_seconds","grouped_ORT_total_seconds","original_callback_seconds")}
            for method in ("Torch","ORT"):
                nested=sum(g[method+"_callback_seconds"] for g in case["groups"])
                check(0<=nested<=case["grouped_"+method+"_total_seconds"],"Nested single-call "+method+" timing obs"+str(obs))
            report["cases"].append(summary)
            del arrays,gold,torch,ort
        check(used_models==set(r["models"]) and len(used_models)==14,"All14 required trueB/D graphs and no unused candidate")
        import onnx
        report["onnx_reader_version"]=onnx.__version__
        for key,model in r["models"].items():
            path=read_artifact(folder,model["model"],False);graph=onnx.load(path,load_external_data=False)
            onnx.checker.check_model(graph,full_check=True)
            size,degree,patches=[int(v[1:]) for v in key.split("_")]
            wanted_in=[dict(name=n,dtype="float32",shape=[size,c,degree,patches]) for n,c in zip(INPUTS,(384,384,882))]
            wanted_out=[dict(name=n,dtype="float32",shape=[size,c,degree,patches]) for n,c in zip(("net_out","delta","weight"),(384,2,2))]
            check(patches==48 and [io_contract(v) for v in graph.graph.input]==model["inputs"]==wanted_in and [io_contract(v) for v in graph.graph.output]==model["outputs"]==wanted_out,"Actual trueB/D fixedFP32 ONNX ports "+key)
            counts=dict(Counter(n.op_type for n in graph.graph.node));axes=[a.i for n in graph.graph.node if n.op_type=="Softmax" for a in n.attribute if a.name=="axis"]
            check(counts==model["ONNX_operator_counts"] and counts.get("Conv")==21 and counts.get("Softmax")==2 and sorted(axes)==[2,3] and not {"Transpose","Gather","GatherElements","GatherND","Where","Pad","Gemm","MatMul"}&set(counts),"Original staticCF operations and full group axes "+key)
            coverage=parameter_coverage(graph,parameters,onnx)
            check(coverage==model["parameter_coverage"],"All original50 actualinitializer bytes/reshape bindings "+key)
            report["models"][key]=dict(inputs=model["inputs"],outputs=model["outputs"],operator_counts=counts,original_parameters_bound=50,external_data=False)
            del graph
        accepted=all(full_gates)
        check(accepted==all(group_gates)==r["strict_FP32_gate_passed"] and len(full_gates)==r["strict_fullgraph_comparison_count"]==27,"Original whole cohort strict rejection preserved")
        check(r["status"]==("completed_grouped_Update_numerical_accepted_not_adopted" if accepted else "completed_grouped_Update_numerical_rejected"),"Status reports measured numeric gate without adoption")
        check(r["outputs_finite"] is True and r["weight_domain_valid"] is True and r["actual_board_execution"] is True and r["host_model_inference"] is False,"Actual recorded execution and output domain scope")
        check(r["ordinary_grouped_Torch_forward_calls"]==r["grouped_ORT_calls"]==sum(len(c["groups"]) for c in r["cases"])==16 and r["actual_export_trace_forward_calls"]==sum(v["actual_export_trace_forward_calls"] for v in r["models"].values())==14 and r["actual_total_Torch_wrapper_forward_calls"]==30,"Recorded actual16Torch16ORT14trace call accounting")
        report.update(status="passed_saved_arrays_static_ONNX_numerical_rejection_confirmed" if not accepted else "passed_saved_arrays_static_ONNX_numeric_gate_not_adopted",strict_FP32_gate_passed=accepted,outputs_finite=True,weight_domain_valid=True,all50_original_initializer_bytes_verified_per_graph=True,export_artifacts_verified=sum(Path(p).parent==folder for p in seen_files),original_capture_artifacts_verified=sum(Path(p).parent==cap for p in seen_files),actual_driver_seconds=r["actual_driver_seconds"],peak_RSS_MiB=r["peak_RSS_MiB"],timing_scope=r["timing_scope"],ordinary_Torch_calls=16,ORT_calls=16,export_trace_calls=14,
            limitations=["Historical fullUpdate gold is saved from the once-called original callback; no gold or grouped neuralforward reexecuted here","Saved full outputs and reconstructed group statistics checked; separately saved raw callback outputs do not exist","Single method/group call; not controlled interleaved performance or full recurrent VO acceptance","Metadata records cleanup,affinity and original source/weight before/after; no independent liveboard observation","No SDK B>1,INT8,placement,trajectory accuracy or20Hz claim"])
    except Exception as error:report["error"]=repr(error)
    report["check_count"]=len(report["checks"])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("export-dir","capture-dir","capture-audit","parent-dir","driver","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--manifest-sha256",required=True)
    a=p.parse_args();r=audit(a.export_dir,a.capture_dir,a.capture_audit,a.parent_dir,a.driver,manifest_sha256=a.manifest_sha256)
    with a.output.open("x") as f:f.write(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:r.get(k) for k in ("status","check_count","strict_FP32_gate_passed","error")}))
    return 0 if r["status"].startswith("passed_") else 1


if __name__=="__main__":raise SystemExit(main())
