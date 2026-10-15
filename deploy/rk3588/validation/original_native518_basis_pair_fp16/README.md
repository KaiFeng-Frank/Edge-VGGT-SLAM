# Original native518 residual-basis FP16 screen

The current basis pair is **not promoted**: B averages **957.398 ms**, versus
**852.748 ms** for original A, with no material final-global precision gain.
Identity-fold control F averages **849.710 ms**; its 0.356% difference is a
short-sample observation, not a demonstrated stable speedup.
See [results](../../../../results/original_native518_basis_pair_fp16_v127.json)
and [executed scripts, launchers and evidence pins](sources.json).

A keeps the original frame0 → global0 pair. F folds original norm affine
terms and LayerScale with identity basis. B uses the fixed mean-preserving
residual basis across both blocks, with one initial and two inverse boundary
MatMuls. These are additional measured work; original learned dense products
remain. All 36 source tensor identities, 1041 tokens per frame, 2082 global
tokens, q/k normalization, RoPE, Erf and FC2K4 partitions/sum order remain.
The [preceding FP32 gate](../original_native518_basis_pair/README.md) passed.

All three graphs compile on RK3588S with identical RKNN2.3.2 settings:
nonquantizing FP16, no calibration or hybrid selection. Runtime uses flags0,
core mask1 and six fresh contexts in **A,F,B,B,F,A** order. Each context runs
`cpu_prefix` then `npu_prefix`. Both tensors come from the same real TUM pair
[32,40], through different CPU/NPU prefixes; they are not independent scenes.

| Input prefix | Route | Mean call, ms | Frame relative L2 | Global relative L2 |
| --- | --- | ---: | ---: | ---: |
| CPU | A | 848.795 | 0.00170337 | 0.00615141 |
| CPU | F | 845.692 | 0.00166369 | 0.00615394 |
| CPU | B | 943.749 | 0.00166046 | 0.00615070 |
| NPU | A | 856.701 | 0.00170196 | 0.00635995 |
| NPU | F | 853.727 | 0.00166262 | 0.00635333 |
| NPU | B | 971.046 | 0.00167442 | 0.00636057 |

Errors use the same-input original A FP32 graph. Each table row has only two
calls in separate contexts. Folding explains most of the small frame-error
change; the basis adds no clear final-global benefit. All endpoint elementwise
`rtol=1e-4`, `atol=1e-5` diagnostics fail. No new FP16 acceptance threshold
is introduced. Comparisons with each route's FP32 graph and actual FP16 A,
maximum errors and residual diagnostics are in the JSON, without causal
attribution to an unobserved internal operator.

Call timers include preparation, SDK FP32 conversion/set/run/get, owned
copies, release and finite checks. Context/model rehash, initialization and
destruction are reported separately in phase lifecycle totals; comparisons,
hashing of outputs and saving are outside. Compilation times are not inference
latency. This screen does not measure steady throughput or P95.

All repeated outputs within each variant/input are byte-identical. The saved
audit verifies 24 outputs, two inputs, 12 FP32 references and 72 comparisons.
Six contexts, 12 runs and 24 output buffers close cleanly. The bounded
1 GHz / 75°C / 180-second controller completes in **18.919 s**, peaks at
**66.538°C**, and restores the prior 1 GHz baseline governor/limits.

Reproduction requires private R126 gated graphs/inputs/references, frozen
SDK helpers, original source/checkpoint lineage and the board toolchain.
They are not shipped or downloaded automatically. Stage the pinned source
bundle; use the three recorded compile launchers with the observed-frequency
CPU controller, then `run_runtime_r1.sh` through the restoring NPU controller.
Preserve process-local library/thread settings and use fresh worker/controller
outputs. Do not substitute different reference inputs.

Stop repeating this FP16 basis micro-tuning. A separate planned screen keeps
FP16 residual/exits while testing internal W8A8 with new disjoint calibration
and evaluation inputs; it has no result here. Whole48-block/head/SL4 behavior,
trajectory quality, 20 Hz/P95, map cadence, stability and physical8GB operation
remain unvalidated by this pair. Defaults are unchanged. Weights, arrays,
weight-bearing graphs and SDK binaries remain private; code and original
weights retain separate licenses.
