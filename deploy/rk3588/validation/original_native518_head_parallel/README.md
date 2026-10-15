# Experimental native518 attention head parallelism

This standalone operator check preserves original global0 attention: 16 heads,
2082 queries and all 2082 keys per head, dimension 64, scale 0.125 and the
original QK/Softmax/PV equations. Inputs are genuine original CPU post-RoPE
Q/K and V captures. The parent block and complete model remain outside this
component test; their NPU intermediates can differ from this CPU reference.

Route A uses one 16-head RKNN context with core mask 7. Route B partitions
heads `[0:6]`, `[6:11]`, `[11:16]` across three concurrent contexts with masks
1/2/4. Each dedicated worker owns its context's construction, execution and
close. Export verifies the original attention nodes/scale and all 18 parent
parameter values; board FP32 gates precede unquantized FP16 compilation.

The ordinary flags0 ABBA trial starts fresh contexts in each phase and records
one first call plus three subsequent calls per phase: six steady calls per
route. Full boundaries include input copies/slices, logical layout conversion,
SDK input/set/run/get, owned copies and release, all joins, head concatenation,
original projection-input reformat and finite checks. Initialization/destruction,
hashes, comparisons and saved-array writes are recorded outside call timing.

R1 stopped before its first SDK run because input descriptors were logical
NHWC `[1,2082,64,H]`. Its context closed cleanly. R2 validates exact descriptor
names/shapes/types, converts original BHSD to BSDH and restores output using
its queried logical format. Actual outputs were NCHW BHSD. `want_float=1`
requests float output; it supplies no output-format override. The frozen owned
runtime remains unchanged. Distinct-axis fixtures verify both layout directions.

Private prerequisites must already exist: original source/checkpoint provenance,
the complete pinned R103 attention capture, verified export/check/compile
receipts and compiled graphs, frozen S100/S99 runtime helpers, RKNN 2.3.2,
and board Python/sysroot. This folder supplies validation source only; captured
arrays, compiled graphs, weights and SDK binaries are not shipped, and no
automatic download/build workflow is supplied. Repository code and model-weight
licenses remain separate; see [model provenance](../../../../MODEL_LICENSES.md).

| Source | SHA256 |
| --- | --- |
| `export_sdpa_head_parts.py` | `8ebf9183d8b04cf3769f5de0e873ef912551b88ce26dac56ab66db883329573c` |
| `check_compile_sdpa_head_parts_board.py` | `6426825d57a9751a22ece3b74e3e50fa1ad1efa6898a36905837862ead73290c` |
| `run_sdpa_head_parallel_abba.py` (R2) | `e74ad2969551f90ef7d6f6eabebd3a1aa67bbb4dd819738f161a37189ee429f6` |

After staging verified files in S112, use the bounded controller. The launcher
sets process-local libraries/threads and exact receipt pins. For a new run,
change its worker output directory and controller output to fresh paths.

```bash
BASE=/mnt/vggt-data/vggt-slam2
S112="$BASE/experiments/vggt-v112-source-r1"
PRODPY="$BASE/venv/bin/python"
CONTROL="$BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py"
"$PRODPY" -B "$CONTROL" --execute-frequency-test --frequency-hz 1000000000 \
  --temperature-limit-c 75 --timeout-seconds 180 \
  --launcher "$S112/run_board_abba_r2.sh" \
  --output "$BASE/experiments/vggt-v112-abba-control-new"
```

Observed steady full-boundary means were A **392.704 ms** and B **208.646 ms**,
46.869% lower for B. All 16 A/B/repeated head and consumer outputs were
byte-identical. Eight contexts, 32 runs and 32 output buffers closed/released;
the controller completed in 10.66 s, reached 61°C and restored 300 MHz.
Both routes failed the unchanged original CPU FP32 gate: relative L2
0.00131118, maximum absolute error 0.041678, with `rtol=1e-4`, `atol=1e-5`,
relative L2 at most `1e-5` and maximum absolute error at most `1e-3` retained.
The standalone A boundary is not the parent's approximately 245 ms fused SDPA
profiling row. Whole-block/model speed, precision propagation and integration
benefit are unmeasured; this result establishes no realtime or SLAM acceptance.
