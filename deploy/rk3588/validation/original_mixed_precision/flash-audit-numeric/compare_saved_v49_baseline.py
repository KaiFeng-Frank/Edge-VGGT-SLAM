#!/usr/bin/env python3
"""Cheap saved first-output comparison across two original FP16 compilations."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
AUDITOR_SHA = "d2ccb21f2a77a4e4671d410cffd754052618dae65fb790e32e50f64d9ef642af"
V49_NORMAL_SHA = "05ac1187d9bda3893d43f2d9c834d3b4de51f97d8c455496daa0c5a6f803e591"


def main():
    import hashlib
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v49-normal-dir", "v50-normal-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    helper = HERE / "audit_saved_flash_pair.py"
    if hashlib.sha256(helper.read_bytes()).hexdigest() != AUDITOR_SHA:
        raise ValueError("Frozen saved-data helper source changed")
    spec = importlib.util.spec_from_file_location("_saved_baseline_helper", helper)
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    a.require(not args.output.exists() and not any(args.output.resolve().is_relative_to(d.resolve())
              for d in (args.v49_normal_dir, args.v50_normal_dir)), "New comparison receipt outside saved data required")
    pins = {}
    old = json.loads(a.pin(args.v49_normal_dir / "run.json", V49_NORMAL_SHA, pins).read_text())
    new = json.loads(a.pin(args.v50_normal_dir / "run.json", a.NORMAL_SHA, pins).read_text())
    a.require(old["capture_manifest_sha256"] == new["capture_sha256"] == a.CAPTURE_SHA
              and old["pair_graph_sha256"] == new["original_graph_sha256"] == a.GRAPH_SHA
              and old["original_checkpoint"]["sha256"] == new["original_checkpoint"]["sha256"] == a.CHECKPOINT_SHA
              and old["normal_benchmark"] is new["normal_benchmark"] is True, "Same original model/graph/capture required")
    rows = []
    for sid in a.SAMPLES:
        choices = [[row for row in run["calls"] if row["sample_id"] == sid and row["round_index"] == 0 and row["role"] == role]
                   for run, role in ((old, "nonquant_fp16"), (new, "ordinary"))]
        a.require(all(len(x) == 1 for x in choices), "Unique first output for each original heldout input required")
        prior, current = (x[0] for x in choices)
        sdk = old["sdk_calls"][prior["sdk_call_index"]]
        a.require(sdk["sample_id"] == sid and sdk["role"] == "nonquant_fp16" and sdk["round_index"] == 0
                  and sdk["input_raw_sha256_before"] == sdk["input_raw_sha256_after"] == current["input_raw_sha256"],
                  "Different original input or old ledger identity")
        x = a.load_array(args.v49_normal_dir, prior["output"], a.OUTPUT_SHAPE, pins)
        y = a.load_array(args.v50_normal_dir, current["output"], a.OUTPUT_SHAPE, pins)
        equal = a.raw_sha(x) == a.raw_sha(y) and np.array_equal(x, y)
        rows.append({"sample_id": sid, "v49_role": "nonquant_fp16", "v50_role": "ordinary", "round_index": 0,
                     "same_original_input_raw_sha256": sdk["input_raw_sha256_before"], "bitwise_equal": bool(equal),
                     "v49_output_raw_sha256": a.raw_sha(x), "v50_output_raw_sha256": a.raw_sha(y),
                     "max_abs_difference": float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))})
    for path, digest in pins.items():
        a.require(a.sha(path) == digest, "Saved data changed during comparison")
    result = {"schema": "v50_saved_ordinary_vs_v49_nonquant_FP16_first_output_comparison",
              "source_sha256": a.sha(__file__), "saved_data_helper_sha256": AUDITOR_SHA,
              "v49_normal_run_sha256": V49_NORMAL_SHA, "v50_normal_run_sha256": a.NORMAL_SHA,
              "saved_outputs_read_and_checked": 4, "comparisons": rows,
              "all_first_outputs_bitwise_equal": all(x["bitwise_equal"] for x in rows),
              "all_read_files_unchanged": True, "actual_model_ORT_SDK_calibration_or_board_execution": False,
              "weights_arrays_exports_or_binaries_published": False,
              "limits": ["First saved output per same heldout input only; full repeat audit remains the separate Flash and v49 numeric receipts.",
                         "Output equality after recompilation does not prove internal operator equivalence or trajectory quality.",
                         "No cross-process timing speedup inference or strict original FP32 gate relaxation."]}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_first_outputs_bitwise_equal": result["all_first_outputs_bitwise_equal"], "receipt_sha256": a.sha(args.output)}))


if __name__ == "__main__":
    main()
