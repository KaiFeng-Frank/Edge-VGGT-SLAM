#!/usr/bin/env python3
"""Generated finite-number/lattice checks only; no real model, data or SDK."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_generated_resolution",HERE/"analyze_saved_resolution.py")
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    checks = []
    def record(name,count=1):
        checks.append({"check":name,"passed":True,"generated_cases":count})
    # Independent oracle enumerates every finite FP16 encoding, sorted by
    # numeric value, rather than trusting a float conversion rounding mode.
    all_half = np.arange(65536,dtype=np.uint16).view(np.float16)
    levels = np.unique(all_half[np.isfinite(all_half)].astype(np.float32))
    assert len(levels)==63487
    assert np.array_equal(a.nearest_fp16(levels),levels)
    record("all_finite_FP16_numeric_values_return_themselves",len(levels))
    middle = ((levels[:-1].astype(np.float64)+levels[1:].astype(np.float64))/2).astype(np.float32)
    before = np.nextafter(middle,np.float32(-np.inf),dtype=np.float32)
    after = np.nextafter(middle,np.float32(np.inf),dtype=np.float32)
    generated = np.concatenate((before,middle,after))
    independent = a.nearest_grid(generated,levels)
    chosen = a.nearest_fp16(generated)
    assert np.array_equal(np.abs(chosen.astype(np.float64)-generated),
                          np.abs(independent.astype(np.float64)-generated))
    record("every_adjacent_FP16_interval_midpoint_and_FP32_neighbors_match_full_enumerated_oracle",len(generated))
    tiny = np.float32(2**-24)
    cases = np.array([0,-0.0,2**-149,-2**-149,tiny/2,-tiny/2,tiny,-tiny,
                      2**-14-tiny/2,2**-14,1.0,1+2**-11,65504,65520,-65520,
                      np.finfo(np.float32).max,-np.finfo(np.float32).max],dtype=np.float32)
    chosen = a.nearest_fp16(cases)
    oracle = a.nearest_grid(cases,levels)
    assert np.isfinite(chosen).all() and np.array_equal(np.abs(chosen.astype(np.float64)-cases),
                                                     np.abs(oracle.astype(np.float64)-cases))
    record("signed_zero_subnormals_normal_boundary_ties_extremes_and_finite_endpoints",len(cases))
    reference = np.array([1.0004],dtype=np.float32)
    m = a.metrics(a.nearest_fp16(reference),reference)
    assert m["outside_reference_tolerance_elements"]==1 and m["max_reference_tolerance_ratio"]>1
    assert m["strict_original_gate_passed"] is False
    record("finite_reference_can_be_impossible_under_fixed_gate_even_with_nearest_FP16")
    near_zero = np.array([tiny/2,-tiny/2,0],dtype=np.float32)
    assert a.metrics(a.nearest_fp16(near_zero),near_zero)["strict_original_gate_passed"] is True
    record("absolute_tolerance_uses_original_magnitude_and_handles_near_zero",len(near_zero))
    # A finite FP32 reference and an exactly FP16-representable candidate can
    # pass a candidate-magnitude tolerance while failing the required reference
    # magnitude tolerance. This checks the direction of the strict comparison.
    reference = np.array([0.031175846233963966],dtype=np.float32)
    candidate = np.array([0.03118896484375],dtype=np.float16).astype(np.float32)
    error = np.abs(candidate.astype(np.float64)-reference.astype(np.float64))
    assert np.all(error <= a.ATOL+a.RTOL*np.abs(candidate.astype(np.float64)))
    assert a.metrics(candidate,reference)["outside_reference_tolerance_elements"]==1
    record("finite_FP32_reference_magnitude_is_used_even_when_candidate_magnitude_would_relax_gate")
    for scale in (np.float32(1/32),np.float32(0.022345)):
        attr = {"type":2,"quantization_type":2,"zero_point":5,"scale":float(scale)}
        grids = a.int8_grids(attr)
        assert len(grids)==4
        for name,grid in grids.items():
            assert a.lattice_membership(grid,grid)["exact_value_members"]==256
            generated_ref = np.array([grid[0]-1,grid[-1]+1,0,grid[3],(grid[3]+grid[4])/2],dtype=np.float32)
            nearest = a.nearest_grid(generated_ref,grid)
            brute = np.min(np.abs(generated_ref.astype(np.float64)[:,None]-grid.astype(np.float64)[None]),axis=1)
            assert np.array_equal(np.abs(generated_ref.astype(np.float64)-nearest),brute)
            record("finite_INT8_lattice_full_search_oracle_"+str(float(scale))+"_"+name,len(generated_ref))
    for invalid in (np.array([np.nan],dtype=np.float32),np.array([np.inf],dtype=np.float32)):
        try:
            a.nearest_fp16(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite reference must reject")
    record("nonfinite_references_rejected",2)
    assert not any(name in sys.modules for name in ("torch","onnxruntime","rknn.api","rknn_runtime"))
    record("no_actual_model_ORT_or_SDK_modules_imported")
    result = {"schema":"v50_generated_representation_resolution_math_checks","passed":True,"checks":checks,
              "check_count":len(checks),"analysis_source_sha256":a.sha(HERE/"analyze_saved_resolution.py"),
              "check_source_sha256":a.sha(__file__),"numpy":np.__version__,
              "actual_model_SDK_ORT_calibration_or_real_saved_arrays_used":False,
              "finite_FP16_numeric_levels_in_independent_oracle":len(levels)}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":True,"checks":len(checks),"receipt_sha256":a.sha(args.output)}))


if __name__=="__main__":
    main()
