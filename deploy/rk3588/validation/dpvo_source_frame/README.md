# Complete-source-frame DPVO Update research

These isolated helpers preserve the original DPVO Update parameters and its
sequential dependencies while arranging one complete source frame as
`[batch, sorted_target, source_patch, channels]`. They do not replace the
production model or execute complete VO, BA, a trajectory, or a 20 Hz pipeline.
DPVO is a different model from VGGT. The original DPVO-derived code remains
under the adjacent [MIT license](LICENSE-DPVO).

The recorded experiment uses two fixed recurrent inputs from an E23520 graph.
Its predetermined minimum-degree source has D10 and M96: `net` and `inp` are
FP32 `[1,10,96,384]`, and `corr` is `[1,10,96,882]`. Outputs are `net_out`
`[1,10,96,384]`, plus `delta` and `weight` `[1,10,96,2]`. All source patches
and target members remain present. The wrapper retains c1 before c2, both
SoftAgg reductions, the original GRU/LayerNorm/Linear modules, and all original
parameter objects. Only the final weight Sigmoid uses its documented stable
formula; the internal gates remain original.

The real Torch/CPU ORT component comparisons passed the fixed elementwise
`rtol=1e-4, atol=1e-5` gate. The FP16 RKNN component did **not** pass that strict
FP32 gate and remains a rejected research candidate. Successful SDK calls,
finite outputs or valid weight ranges do not override numerical rejection.
No tolerances are relaxed and no output is clipped. Component execution is
not evidence of full-graph speed, new poses or trajectory accuracy.

[Board evidence and preserved rejection](../../../../results/dpvo_source_frame_v33.json).

## Files and boundaries

- `source_frame_plan.py` checks the complete Cartesian topology and restores
  each source's original edge order; `source_frame_update.py` applies the
  original neural modules to that layout.
- `export-v33-source-frame-board.py` runs the original full Update, the source
  wrapper and CPU ORT on the board, saving references and original-parameter
  provenance. `audit-v33-source-frame-export.py` independently checks saved
  arrays and static ONNX; it never runs a model or ORT.
- `compile-v33-source-frame-update.py` uses the pinned Toolkit2 2.3.2,
  nonquantized FP16, optimization level 3 configuration. Host compilation
  checks manifest/ONNX bytes; board source paths are declarations until the
  board runner rehashes them.
- `run-v33-source-frame-rknn-board.py` and `v33_rknn_source_frame.py` preserve
  the initial startup failure on padded output attributes. The R2 runner uses
  `v33_rknn_source_frame_padded.py`, while the compilation ancestor remains
  the byte-identical parent helper. The padding adapter permits only the
  internal output stride footprint; inputs remain strict. Returned buffers
  must still have the exact logical FP32 byte count and are copied before
  SDK release. It does not read native zero-copy padded buffers.
- `audit-v33-source-frame-rknn.py` checks saved NumPy outputs, numerical
  rejection, SDK profiles and cleanup without running a model or SDK.
- `query-v33-source-frame-board.py` preserves the experiment-specific
  attribute query, including its fixed board paths. It performs no inference.

The SDK ABI helper, SDK header/library, checkpoints and saved experiment
arrays are external, SHA-pinned dependencies; they are not bundled here.
Hardware entry points reject non-RK3588 execution. Use separate new output
directories for ordinary and profiled runs; profile timings are diagnostic.

## Reproduction entry points

Each entry point describes its exact required paths and SHA arguments:

```bash
python deploy/rk3588/validation/dpvo_source_frame/export-v33-source-frame-board.py --help
python deploy/rk3588/validation/dpvo_source_frame/compile-v33-source-frame-update.py --help
python deploy/rk3588/validation/dpvo_source_frame/run-v33-source-frame-rknn-board-r2.py --help
python deploy/rk3588/validation/dpvo_source_frame/audit-v33-source-frame-export.py --help
python deploy/rk3588/validation/dpvo_source_frame/audit-v33-source-frame-rknn.py --help
```

Compile with the parent `v33_rknn_source_frame.py` as `--io-helper`; run R2
with `v33_rknn_source_frame_padded.py` as `--io-helper`. Keep both beside one
another. The unchanged parent SHA is
`322021ade54fb9035b7ce80652ec90169938a417fd2192528d28a805a83ed25e`.
The R2 runner separately pins the new runtime and old compilation ancestor.
Ordinary mode runs first/warm/warm for each of two fixed states; profile mode
runs one call per state in a separate process. Invalid output may be observed
only with both `--collect-perf-detail` and
`--diagnostic-observe-one-invalid-output`, and remains rejected.

The `tests/test_v33_*.py` tests use tiny generated topology, random toy modules
or fake SDK buffers. They do not load real weights, run a real model or SDK,
compile RKNN graphs, or access the board.
