#!/usr/bin/env python3
"""Saved-array representation mathematics only; no model, SDK or calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
CAPTURE_SHA = "bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc"
NORMAL_SHA = "05ac1187d9bda3893d43f2d9c834d3b4de51f97d8c455496daa0c5a6f803e591"
PAIR_EXPORT_SHA = "149450ffc675ce59950c6f89cf84adb41e388cd050eeb731b6d09a6fe16e9fe1"
RTOL, ATOL = 1e-4, 1e-5
INPUT_SHAPE, OUTPUT_SHAPE = [2,437,1024], [1,874,1024]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, f"Saved evidence SHA mismatch: {path}")
    pins[str(path)] = expected
    return path


def read_array(directory, item, shape, pins):
    import numpy as np
    relative = Path(item["file"])
    require(not relative.is_absolute() and ".." not in relative.parts, "NPY path escaped private directory")
    path = (directory / relative).resolve(strict=True)
    require(path.is_relative_to(directory.resolve()) and path.stat().st_size == item["bytes"]
            and item["bytes"] < 16 * 1024 * 1024, "NPY path/byte count changed")
    pin(path,item["sha256"],pins)
    a = np.load(path,allow_pickle=False)
    require(list(a.shape) == item["shape"] == shape and str(a.dtype) == item["dtype"] == "float32"
            and a.flags.c_contiguous and np.isfinite(a).all() and a.nbytes == item["tensor_bytes"]
            and hashlib.sha256(memoryview(a).cast("B")).hexdigest() == item["raw_sha256"],
            "Saved original/output NPY identity changed")
    return a


def nearest_fp16(a):
    """Finite nearest FP16 via rounded center and both adjacent candidates.

    Local neighbor minimality is checked for every chosen value. On an ordered
    finite real lattice, a value no farther than each adjacent member is a
    global nearest member. This also covers subnormals, ties and endpoints.
    """
    import numpy as np
    r = np.asarray(a,dtype=np.float64)
    require(np.isfinite(r).all(), "Finite reference required")
    with np.errstate(over="ignore",under="ignore"):
        center = np.clip(r,-65504.0,65504.0).astype(np.float16)
        lower = np.nextafter(center,np.float16(-np.inf),dtype=np.float16)
        upper = np.nextafter(center,np.float16(np.inf),dtype=np.float16)
        lower = np.where(np.isfinite(lower),lower,center)
        upper = np.where(np.isfinite(upper),upper,center)
        candidates = np.stack((center,lower,upper)).astype(np.float64)
        chosen = np.take_along_axis(candidates,np.argmin(np.abs(candidates-r),axis=0)[None],axis=0)[0]
        chosen16 = chosen.astype(np.float16)
        prev = np.nextafter(chosen16,np.float16(-np.inf),dtype=np.float16).astype(np.float64)
        after = np.nextafter(chosen16,np.float16(np.inf),dtype=np.float16).astype(np.float64)
    distance = np.abs(chosen-r)
    require(np.all(distance <= np.abs(prev-r)) and np.all(distance <= np.abs(after-r)),
            "Nearest-FP16 adjacent-neighbor certificate failed")
    return chosen.astype(np.float32)


def nearest_grid(a,grid):
    import numpy as np
    r = np.asarray(a,dtype=np.float64)
    g = np.asarray(grid,dtype=np.float64)
    require(g.ndim == 1 and g.size and np.isfinite(g).all() and np.all(g[1:] >= g[:-1])
            and np.isfinite(r).all(), "Finite sorted nonempty lattice/reference required")
    index = np.searchsorted(g,r)
    left,right = g[np.clip(index-1,0,len(g)-1)],g[np.clip(index,0,len(g)-1)]
    return np.where(np.abs(r-left) <= np.abs(r-right),left,right).astype(np.float32)


def metrics(a,reference):
    import numpy as np
    x,r = np.asarray(a,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    require(x.shape == r.shape and np.isfinite(x).all() and np.isfinite(r).all(), "Same finite metric shape required")
    error = x-r
    absolute = np.abs(error)
    tolerance = ATOL+RTOL*np.abs(r)  # Always the original reference magnitude.
    norm = float(np.linalg.norm(r))
    return {"elements":r.size,"max_abs_error":float(absolute.max()),"RMSE":float(np.sqrt(np.mean(error*error))),
            "relative_L2":float(np.linalg.norm(error)/norm) if norm else None,
            "outside_reference_tolerance_elements":int(np.count_nonzero(absolute>tolerance)),
            "max_reference_tolerance_ratio":float(np.max(absolute/tolerance)),
            "strict_original_gate_passed":bool(np.all(absolute<=tolerance))}


def stats(a):
    import numpy as np
    x = np.asarray(a,dtype=np.float64)
    absolute = np.abs(x)
    return {"elements":x.size,"min":float(x.min()),"max":float(x.max()),"mean":float(x.mean()),
            "std":float(x.std()),"RMS":float(np.sqrt(np.mean(x*x))),"zeros":int(np.count_nonzero(x==0)),
            "absolute_quantiles":{str(q):float(np.quantile(absolute,q)) for q in (.5,.9,.99,.999,.9999,1)},
            "abs_gt_counts":{str(v):int(np.count_nonzero(absolute>v)) for v in (1,2,4,8)}}


def magnitude_floor_distribution(reference,nearest):
    """Descriptive counts only; no quantizer fitting or downstream propagation."""
    import numpy as np
    r = np.asarray(reference,dtype=np.float64)
    absolute = np.abs(r)
    outside = np.abs(np.asarray(nearest,dtype=np.float64)-r) > ATOL+RTOL*absolute
    edges = (0,0.001,0.01,0.1,0.5,1,2,4)
    rows = []
    for i,lower in enumerate(edges):
        upper = edges[i+1] if i+1<len(edges) else None
        mask = (absolute>=lower) if upper is None else (absolute>=lower)&(absolute<upper)
        rows.append({"abs_reference_lower_inclusive":lower,"abs_reference_upper_exclusive":upper,
                     "elements":int(np.count_nonzero(mask)),
                     "nearest_representation_outside_reference_tolerance_elements":int(np.count_nonzero(mask&outside))})
    require(sum(x["elements"] for x in rows)==r.size
            and sum(x["nearest_representation_outside_reference_tolerance_elements"] for x in rows)==np.count_nonzero(outside),
            "Magnitude distribution partitions must cover the original reference")
    return rows


def large_patch_channel_distribution(a):
    import numpy as np
    patch = np.asarray(a[:,5:],dtype=np.float64)
    large = np.abs(patch)>4
    counts = np.count_nonzero(large,axis=(0,1))
    channels = np.flatnonzero(counts)
    return {"scope":"Original DINO patch tokens only; threshold |x| > 4 is descriptive and fixed before reading values.",
            "large_elements":int(counts.sum()),"channels_with_large_elements":int(channels.size),
            "tokens_with_any_large_element":int(np.count_nonzero(np.any(large,axis=-1))),
            "maximum_large_elements_in_one_patch_token":int(np.count_nonzero(large,axis=-1).max()),
            "channel_counts":[{"channel_index":int(c),"large_elements":int(counts[c])} for c in channels]}


def int8_grids(attr):
    import numpy as np
    require(attr["type"] == 2 and attr["quantization_type"] == 2 and -128 <= attr["zero_point"] <= 127
            and np.isfinite(attr["scale"]) and attr["scale"] > 0, "Actual signed INT8 affine descriptor required")
    centered = np.arange(-128,128,dtype=np.float64)-attr["zero_point"]
    scale = float(np.float32(attr["scale"]))
    half_scale = float(np.float16(scale))
    require(scale == attr["scale"], "Reported SDK scale must retain its FP32 value")
    return {"FP32_scale_FP32_product":(centered*scale).astype(np.float32),
            "FP32_scale_FP16_rounded_product":(centered*scale).astype(np.float16).astype(np.float32),
            "FP16_rounded_scale_FP32_product":(centered*half_scale).astype(np.float32),
            "FP16_rounded_scale_FP16_rounded_product":(centered*half_scale).astype(np.float16).astype(np.float32)}


def lattice_membership(a,grid):
    import numpy as np
    closest = nearest_grid(a,grid)
    distance = np.abs(np.asarray(a,dtype=np.float64)-closest.astype(np.float64))
    return {"elements":a.size,"exact_value_members":int(np.count_nonzero(distance==0)),
            "max_distance_to_lattice":float(distance.max()),
            "all_values_on_lattice":bool(np.all(distance==0))}


def static_mlp_scope(pair):
    bindings = pair["original_parameter_bindings"]
    total_bytes = sum(item["original"]["bytes"] for group in bindings.values() for item in group.values())
    weights = {group+"/"+name:item for group,rows in bindings.items() for name,item in rows.items()
               if name in ("mlp.fc1.weight","mlp.fc2.weight")}
    require(len(weights)==4 and total_bytes==100788224, "Frozen original pair parameter scope changed")
    selected_bytes = sum(item["original"]["bytes"] for item in weights.values())
    mlp_linear = 2*874*sum(item["original"]["shape"][0]*item["original"]["shape"][1] for item in weights.values())
    all_linear = 2*874*2*12*1024**2
    attention = 4*1024*(2*437**2+874**2)
    return {"four_original_MLP_weight_initializers":[item["initializer"] for item in weights.values()],
            "selected_original_FP32_weight_bytes":selected_bytes,"all_original_pair_FP32_parameter_bytes":total_bytes,
            "selected_weight_fraction_of_original_parameter_bytes":selected_bytes/total_bytes,
            "pair_MLP_MatMul_FLOP":mlp_linear,"pair_all_linear_MatMul_FLOP":all_linear,
            "pair_attention_QK_and_AV_MatMul_FLOP":attention,
            "MLP_fraction_of_linear_MatMul_FLOP":mlp_linear/all_linear,
            "MLP_fraction_of_counted_all_MatMul_FLOP":mlp_linear/(all_linear+attention),
            "FLOP_definition":"one multiply plus one add = 2; no norm, softmax, RoPE, GELU or memory/SDK costs counted",
            "source":"SHA-pinned original pair export original parameter shapes and B1/S2/P437/C1024",
            "measured_runtime_fraction_or_supported_hybrid_configuration":False}


def analyze(capture_dir,normal_dir,pair_export):
    import numpy as np
    pins = {}
    capture = json.loads(pin(capture_dir/"capture.json",CAPTURE_SHA,pins).read_text())
    normal = json.loads(pin(normal_dir/"run.json",NORMAL_SHA,pins).read_text())
    pair = json.loads(pin(pair_export,PAIR_EXPORT_SHA,pins).read_text())
    require(capture["status"]=="completed_prefix_capture" and normal["capture_manifest_sha256"]==CAPTURE_SHA
            and normal["normal_benchmark"] is True and normal["candidate_accepted"] is False,
            "Fixed original saved capture/normal experiment required")
    attrs = {d:normal["context_records"]["ptq_w8a8"][d+"_attr"] for d in ("input","output")}
    grids = {d:int8_grids(attr) for d,attr in attrs.items()}
    input_rows,output_rows = [],[]
    for sample in capture["samples"]:
        sid = sample["sample_id"]
        original_input = read_array(capture_dir,sample["artifacts"]["input.npy"],INPUT_SHAPE,pins)
        grid = grids["input"]["FP32_scale_FP32_product"]
        input_rows.append({"sample_id":sid,"split":sample["split"],"original_input_distribution":stats(original_input),
                           "special_5_tokens_distribution":stats(original_input[:,:5]),
                           "DINO_patch_432_tokens_distribution":stats(original_input[:,5:]),
                           "conditional_FP32_scale_INT8_input_grid_nearest_floor":metrics(nearest_grid(original_input,grid),original_input),
                           "elements_outside_INT8_input_grid_endpoints":int(np.count_nonzero((original_input<grid[0])|(original_input>grid[-1]))),
                           "FP16_input_grid_nearest_floor":metrics(nearest_fp16(original_input),original_input),
                           "large_patch_channel_distribution":large_patch_channel_distribution(original_input)})
        if sample["split"] != "evaluation":
            continue
        reference = read_array(capture_dir,sample["artifacts"]["output.npy"],OUTPUT_SHAPE,pins)
        actuals = {}
        for role in ("nonquant_fp16","ptq_w8a8"):
            call = next(row for row in normal["calls"] if row["sample_id"]==sid and row["role"]==role and row["round_index"]==0)
            actuals[role] = read_array(normal_dir,call["output"],OUTPUT_SHAPE,pins)
        nearest_half = nearest_fp16(reference)
        finite_half = np.arange(65536,dtype=np.uint16).view(np.float16)
        half_values = np.unique(finite_half[np.isfinite(finite_half)].astype(np.float32))
        oracle_half = nearest_grid(reference,half_values)
        require(np.array_equal(np.abs(nearest_half.astype(np.float64)-reference),
                               np.abs(oracle_half.astype(np.float64)-reference)),
                "Independent all-finite-FP16 enumeration disagrees for a saved original reference")
        row = {"sample_id":sid,"original_FP32_distribution":stats(reference),
               "actual_saved_call_scope":{"round_index":0,"roles":["nonquant_fp16","ptq_w8a8"]},
               "actual_nonquant_FP16_error":metrics(actuals["nonquant_fp16"],reference),
               "actual_PTQ_error":metrics(actuals["ptq_w8a8"],reference),
               "nearest_finite_FP16_output_floor":metrics(nearest_half,reference),
               "FP16_adjacent_neighbor_minimality_certified_for_each_element":True,
               "FP16_full_finite_value_enumeration_independently_agrees_for_each_element":True,
               "FP16_output_floor_by_original_magnitude":magnitude_floor_distribution(reference,nearest_half),
               "actual_nonquant_values_exactly_FP16_representable":bool(np.array_equal(nearest_fp16(actuals["nonquant_fp16"]),actuals["nonquant_fp16"])),
               "INT8_output_grid_hypotheses":{}}
        for name,grid in grids["output"].items():
            row["INT8_output_grid_hypotheses"][name] = {
                "finite_grid_endpoints":[float(grid[0]),float(grid[-1])],
                "original_reference_elements_outside_grid_endpoints":int(np.count_nonzero((reference<grid[0])|(reference>grid[-1]))),
                "nearest_representation_floor_against_original":metrics(nearest_grid(reference,grid),reference),
                "observed_PTQ_output_membership":lattice_membership(actuals["ptq_w8a8"],grid)}
        output_rows.append(row)
    for path,digest in pins.items():
        require(sha(path)==digest,"Saved source/array bytes changed during analysis")
    return {"schema":"v50_original_pair_saved_representation_resolution","analysis_source_sha256":sha(__file__),
            "capture_manifest_sha256":CAPTURE_SHA,"normal_run_sha256":NORMAL_SHA,"pair_export_sha256":PAIR_EXPORT_SHA,
            "numpy":np.__version__,"SDK_logical_INT8_descriptors":attrs,"input_samples":input_rows,"heldout_outputs":output_rows,
            "static_original_MLP_scope":static_mlp_scope(pair),"rtol":RTOL,"atol":ATOL,
            "raw_arrays_or_original_parameter_values_saved_by_analysis":False,"all_read_file_bytes_unchanged":True,
            "actual_model_ORT_SDK_calibration_compilation_or_runtime_input_generation":False,
            "read_file_SHA_pins":pins,
            "interpretation":{"floor_scope":"Each SHA-pinned saved tensor and the explicitly defined finite representation grid only.",
                              "nearest_FP16_proof":"Rounded center plus predecessor/successor; every result is no farther than either adjacent finite value, hence globally nearest on the ordered finite FP16 lattice.",
                              "reference_tolerance":"atol + rtol*abs(original FP32 reference), calculated in float64 from exact saved FP32 values.",
                              "FP16_boundary_MLP_only":"Useful only as a controlled diagnostic of lower approximation error and measured operator behavior. FP16 output representation alone cannot pass the unchanged component gate on these two references.",
                              "internal_error_causal_attribution":False,
                              "quality_gate_distinction":"Strict component equivalence is separate from full original-model SLAM ATE/RPE +10–20% acceptance; no trajectory inference or quality measurement occurs here.",
                              "hardware_impossibility_or_20Hz_or_trajectory_claim":False,
                              "MLP_intermediate_distributions_observed":False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir",type=Path,required=True)
    parser.add_argument("--normal-dir",type=Path,required=True)
    parser.add_argument("--pair-export",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.capture_dir.resolve())
            and not args.output.resolve().is_relative_to(args.normal_dir.resolve()),"New analysis receipt outside saved data required")
    result = analyze(args.capture_dir.resolve(),args.normal_dir.resolve(),args.pair_export.resolve())
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"heldout_outputs":len(result["heldout_outputs"]),"input_samples":len(result["input_samples"]),
                      "receipt_sha256":sha(args.output),"actual_inference_or_calibration":False}))


if __name__=="__main__":
    main()
