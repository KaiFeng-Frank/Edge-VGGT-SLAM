# Original native518 DINO0 with three native NPU owners

One complete original DINO0 block executes all weighted and attention products
across three fixed owners on individual NPU cores, preserving the candidate's
output bytes. **Reject deployment and further tuning of this schedule:** it
is slower than the matched fused core7 baseline and still fails original
precision gates. See [results](../../../../results/original_native518_three_core_block_v124.json).

All 14 original parameters, 1,041 queries/keys and 16 heads remain. Four
weighted operations become twelve full-K products, partitioning output columns:
QKV `1024/1024/1024`, projection/FC2 `352/352/320`, FC1 `1376/1376/1344`.
Attention heads use `6/5/5` owner groups. RKNN type1 takes FP16 operands and
returns FP32; internal accumulation precision is unproven. Nothing splits K,
omits tokens or substitutes saved intermediates. Original bias, LayerNorm,
LayerScale, residuals, Erf GELU and stable softmax remain on onboard Mali.

Fixed workers own every native init/sync/bind/run/close on masks `1/2/4`.
Weighted shards share their leader's A and own separate B/C allocations.
GPU consumers select twelve C4 buffers directly by original logical channel;
no C gather is added. All native producers join before GPU consumption.
WG256, standard `erf`/`exp`, scale `0.125` and `FP_CONTRACT OFF` remain.

An eight-call **ABBAABBA** trial uses the original fused **core7** model for A.
A separate four-call preflight validates that mask and identical fused bytes
(152.128 ms subsequent-three mean); it is not pooled into the matched trial.
Each route retains its first call separately and averages the following three.

| Measurement | A: fused core7 | B: three native owners |
| --- | ---: | ---: |
| First complete block | 136.940 ms | 292.977 ms |
| Following three mean | **151.464 ms** | **293.912 ms** |
| Relative L2 vs original CPU FP32 | 2.54823% | 0.025077% |
| Maximum absolute error | 0.229286 | 0.00524735 |
| Strict and fixed reference gates | Fail | Fail |

B takes **1.940 times** A's duration. Its complete timer includes upload, all 44
native products, GPU work, sync, owned output and finite checks. Setup,
teardown, comparisons, saving and diagnostics are separate. Candidate
complete-stage means include softmax **90.925 ms**, FC2 **33.489 ms**, QK
**14.427 ms** and PV **17.234 ms**. Softmax GPU events average **56.577 ms**
inside its stage wall. Overlapping blocking SDK call intervals are host timing,
not device utilization; these observations do not isolate an overhead cause.
Three subsequent samples establish neither P95 nor sustained throughput.

All four B outputs match the prior native candidate byte for byte; precision
is unchanged and still fails `rtol=1e-4, atol=1e-5` and L2 ≤ `1e-5` /
maximum-error ≤ `1e-3`. This single-block result establishes no trajectory gain.
The [saved-only auditor](audit_native_threecore_saved.py) verifies 16 arrays, numerical records and cleanup.
After 176 products, all 44 contexts, 132 buffers, 28 GPU events and 136 OpenCL
objects close; borrowers release before their shared-A leader and workers join.
The 1 GHz NPU controller completes in 7.602 s, observes 55.461°C maximum and
restores 300 MHz. GPU policy is unchanged and not continuously sampled.
Peak process RSS is 862.695 MiB, without physical 8 GB acceptance.

The executed [source bundle](r1/sources.json) and launcher require private
pinned original weights, references and board libraries under restoring
control. No arrays, weights or SDK binaries are distributed; see
[licenses](../../../../MODEL_LICENSES.md). Defaults, the **26.7345 s** cached-pair
result and [R123 preload](../original_native518_rolling_preload/README.md) remain
unchanged. No full-model, real-time, map or trajectory acceptance is inferred.
