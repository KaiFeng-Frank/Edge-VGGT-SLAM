# Dynamic DPVO Update research (v31)

This optional experiment tests one CPU ONNX graph across changing edge and group dimensions. It leaves the default VGGT model and SL4 mapper unchanged. All actual encoder and Update execution was on RK3588S. The host inspected saved arrays and graphs and ran tiny synthetic tests. No ORB, LK or PnP pose frontend was added.

**The candidate has not passed the existing FP32 equivalence gate.** The 14-frame diagnostic stopped on its second 6,144-edge initialization update: one hidden-state element and one delta element exceeded `atol=1e-5 + rtol=1e-4 * abs(reference)`. Thirteen observations completed, but initialization did not commit and no fresh pose was published. The separate 23,520-edge experiment also rejected one of two fixed states. Finite outputs and valid confidence weights do not overturn that result; trajectory accuracy remains untested for this candidate.

## Files and contracts

- `v31_update_topology.py` constructs the ten external topology arrays with stable sorts and bounded owned caching. It preserves original neighbor ties, integer grouping, per-group member order, padding and edge order. The two-entry cache retains at most 32 MiB of NumPy payload; this is not total process memory.
- `run-v31-dpvo-update-export-board.py` exports the original Update parameters through the pinned v28 adapter, with dynamic `E`, `Gkk`, `Lkk`, `Gij`, and `Lij`. Batch and channel dimensions stay fixed. Only the terminal confidence Sigmoid uses the already documented v29 stable identity. The graph is standard ONNX; RKNN dynamic-shape support has not been tested.
- `v31_dynamic_update.py` checks the export/model hashes and I/O contract, runs CPU ORT with four threads, returns owned arrays and rejects nonfinite outputs or confidence outside `[0,1]`. It does not certify FP32 or trajectory equivalence.
- `run-v31-dpvo-stream-board.py` retains the frozen v30 lifecycle and accepts `--check-original-update` for same-current-input Torch references. A mismatch stops processing before publishing a new pose. The adjacent v30 core and encoder are byte-identical copies for reproducibility.
- `run-v31-large-update-board.py` reconstructs the 17 retained images and saved patch centers of two v30 states. It executes Torch/ORT, ORT/Torch, Torch/ORT on the same reset input state, saves representative outputs and reports numerical rejection independently of successful execution. It does not run BA or publish poses.
- The audit scripts inspect saved artifacts; they do not run real model inference on the host.

## Reproduction

Each board entrypoint provides `--help`. Use an isolated new output directory under `/mnt/vggt-data/vggt-slam2/experiments`; completed experiment directories are not overwritten. Supply the external source/checkpoint from v26, CPU encoder ONNX/export/ORT receipts, v27 native correlation library and build receipt, v29 vectorized geometry, and original TUM fr1_xyz data. All measured helpers and relevant inputs are hash-bound in the [evidence](../../../../results/dpvo_dynamic_update_v31.json). The original checkpoint and compiled models are not included.

Export requires the frozen `dpvo_update_export.py` and `rewrite-v29-weight-sigmoid.py` one directory above this directory, plus the v28 saved capture. The exporter is guarded to the RK3588 board. Its generated `manifest.json` remains labelled `exported_dynamic_ONNX_numerical_validation_pending`; read the separate runtime evidence for the actual failed gate.

For the measured streaming diagnostic, supply `--source-manifest source-freeze.json`, the generated `--update-manifest` and its exact `--update-manifest-sha256`, plus `--frames 14 --seed 20261011 --run-time-limit 1200 --check-original-update`. It should reproduce the failed numerical gate, not an accepted trajectory. The recorded aggregate `strict_original_gate_passed` field is null on this failed driver path because it was assigned only after successful completion; the failing per-call counts, exception and independent array audit establish rejection. No subsequent ordinary 192-frame candidate replay was run.

The large-state comparator also requires the v30 r2 directory, frozen v30 runner and adjacent v30 dependencies, and `state-reconstruction-contract.json`. `--save-inputs` retains next-step inputs on the board. Recomputed features share original pixels and centers, but v30 did not retain the old feature bytes, so historical feature equality cannot be certified. Snapshot target/weight arrays are prior-step results and are not references for the new next-step Update.

## Measurement boundary

At 23,520 edges, the two per-state warm means were 2.538/2.931 s for Torch and 1.842/2.237 s for ORT, including each adapter's input/output copies and ORT topology handling. These are two warm calls per method per state, not sustained throughput. Hashing, statistics, reconstruction, projection and correlation are outside those Update intervals. Neither temperature nor frequency was controlled, and the later calls grew slower. The 42.073 s diagnostic process peaked at 2,157.316 MiB RSS on a 16 GB board; it holds both executors and saved comparison arrays, so it is not production memory or 8 GB acceptance.

There is no NPU, physical-camera latency, fresh-pose rate, dense-map, loop-closure, endurance or trajectory-accuracy acceptance in v31. See [engineering notes 36–39](../../../../docs/ENGINEERING_NOTES.md) and the separate [v30 original Torch-Update replay](../../../../results/dpvo_stream_v30.json).

The [129 passing targeted synthetic tests](../../../../results/synthetic_tests_v31.json) cover topology/ownership, fake runtime contracts, tiny generated dynamic graphs, comparison scheduling and saved-array audit rejection. Their success validates these test cases; the actual model's failed FP32 gate remains separate.

DPVO-derived code retains the adjacent [MIT license](LICENSE), Copyright (c) 2022 Princeton Vision & Learning Lab. That code license does not license the external checkpoint or its ONNX/RKNN derivatives. [Weight and artifact licensing](../../../../MODEL_LICENSES.md) remains separate.

## Source-frame layout plan (v32; no new neural execution)

`source_frame_plan.py` checks the complete source-frame Cartesian product and copies between original `[B,E,C]` edge order and canonical `[B,D,M,C]` order. Each source must retain every patch and every target edge, with `ii == kk // M`; integer domains exclude original `ii*12345+jj` collisions. Missing/duplicate rows, mixed ownership and capacity overflow are rejected. Inputs must remain stable during construction, for example through an owner-held graph snapshot. Returned indices and copied values have independent ownership.

For the saved 23,520-edge graph, all 17 source blocks satisfy these conditions. The plan closes both group dependencies and neighbor links, but it is not a neural wrapper or a finite/dtype/state-commit validator. A future executor must preserve c1/c2 ordering and boundary bias, validate every block against the same graph revision, restore original edge order and commit only after all blocks succeed. Total original Linear work remains 63.7695 GMAC; no new export, RKNN invocation or block speedup was measured.

The [saved-array Where diagnosis and implementation contract](../../../../results/dpvo_update_plan_v32.json) separate measured failures in an instrumented graph from hypotheses about the original graph. The tiny layout tests use only generated integers/arrays and are included in [v32 validation](../../../../results/synthetic_tests_v32.json). The adjacent MIT license applies to this source adaptation; checkpoints and binaries remain external.
