# Original VGGT FC2 input-channel partition

This board diagnostic retains the original frame0/global0 pair, all 106 tokens,
all parameters and its two SDK outputs. Each FC2 uses K=4096 and N=1024.
Slice the input and the transposed weight rows into two or four K parts, sum the
full-N partial outputs in increasing part order, then apply the original bias
once. FC1, exact GELU, attention, LayerScale and residual nodes are unchanged.

K partition changes floating-point reduction order. Both candidates pass the
existing `rtol=1e-4, atol=1e-5` check against the original FP32 ONNX on two saved
real inputs before compilation; outputs are not byte-identical. Available
original Torch references pass for those FP32 graphs. The second input has no
saved Torch reference. FP16 SDK outputs change and fail the original FP32 strict
gate, including the baseline. This is not trajectory-quality acceptance.

At 300 MHz, core mask 7 and CPUs 4–7, ten ABCCBA cycles per input give 38 warm
calls per graph after excluding the first call per input and graph:

| Graph | Complete pair mean | Reduction |
| --- | ---: | ---: |
| Original | 39.067 ms | — |
| Two FC2 K parts | 38.347 ms | 1.84% |
| Four FC2 K parts | 35.766 ms | 8.45% |

The measurements include NEON input conversion, SDK run, both output copies,
release, widening and finite checks. Profiling is a separate six-call process.
Saved actual profiles show one internal NPU Split and two/four three-core
partial Conv operations per FC2; sum work is fused into ConvAdd. The original
FC1 remains a single-core ConvExGelu. Including Split, partial Conv, sums, bias,
LayerScale and residual, the two FC2 paths total 6.929→4.561 ms for one input
and 7.310→5.072 ms for the other (original→four parts). These profile samples
are not the normal warm-call means. Only input/output shims appear on CPU.
[Saved profile analysis](profile_saved_analysis.json).

These are two inputs to one original pair with deterministic ordering. There is
no new full-model, timestamp replay or trajectory result; neither candidate is
installed in the production runtime. Do not multiply this gain by layer count.

All ORT, compilation and actual SDK work ran on the RK3588S. The host inspected
source and saved arrays only. Weight-bearing graphs, arrays and SDK files stay
private. The original checkpoint license remains separate from the code license.

```sh
# Prepared RK3588S compiler environment; fresh output directories required.
python build_pair_fc2_k_splits_board.py --chunks 2 --output-dir SPLIT2_DIR
python build_pair_fc2_k_splits_board.py --chunks 4 --output-dir SPLIT4_DIR

# Prepared RK3588S runtime environment; pin actual build result.json hashes.
python probe_pair_fc2_k_splits_board.py \
  --compiled SPLIT2_DIR SPLIT2_RESULT_SHA256 \
  --compiled SPLIT4_DIR SPLIT4_RESULT_SHA256 \
  --rounds 10 --output-dir NORMAL_DIR
python probe_pair_fc2_k_splits_board.py \
  --compiled SPLIT2_DIR SPLIT2_RESULT_SHA256 \
  --compiled SPLIT4_DIR SPLIT4_RESULT_SHA256 \
  --profile --output-dir PROFILE_DIR
```

Exact launch arguments are in the command JSON files. `normal-command.json`
records an unsuccessful launch: the interpreter filename mistakenly included its
patch version. No child or SDK call started. `normalretry-command.json` uses the
existing `venv/bin/python` and is the measured run; the failed receipt is retained.
The read-only monitor records child exit, 300 MHz and temperature; it stops its
own process group at 75 C or the deadline and changes no system settings.

[Measurements and evidence hashes](../../../../results/original_pair_fc2_partition_v72.json).

The independent [saved-array audit](audit_saved.json) verifies 278 arrays,
126 SDK calls, 252 released outputs and six closed contexts. All 84 original
baseline outputs checked against the prior diagnostic remain byte-identical.
The audit can be rerun against the private evidence snapshots:

```sh
python audit_saved.py --root V72_PRIVATE_ROOT --original-root V70_PRIVATE_ROOT \
  --output NEW_AUDIT_JSON
```

The audit source and receipt contain identities and numeric summaries only;
actual input/output arrays are not published.
