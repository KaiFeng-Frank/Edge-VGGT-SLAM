# Original native518 continuous native DINO0 block

Six persistent native type-1 contexts execute all four original weighted
products and all 32 attention products. Mali imports 14 native buffers for
original FP32 LayerNorm, bias, LayerScale, residual, Erf GELU and softmax,
writing Half inputs directly. All 14 original parameters, 1,041 tokens and
16 heads remain; LayerNorm epsilon is `1e-6`. No reference intermediate feeds
the computation. Static weights upload at setup; dynamic head packing stays
inside every complete call.

The historical matched core1 **ABBAABBA** screen averages **177.689 ms** for
the fused block and **437.224 ms** for the candidate, after the first call of
each route. Setup/destruction and saved comparisons are separate. Relative L2
against original CPU FP32 improves from **2.54823% to 0.025077%**; maximum error
is **0.00524735**. Both strict and fixed reference gates still fail.

All route outputs repeat exactly. Six contexts, 18 buffers, 144 products,
208 events and 38 OpenCL objects close, with GPU release before NPU storage.
The restoring 1 GHz controller observes 55.461°C maximum and returns to 300 MHz;
GPU policy is unchanged. Reject this serial schedule for deployment. Neither
kernel/SDK time sums nor their residual wall time prove physical limits or
pure CPU overhead. No whole-model or trajectory improvement is established.

The executed [bundle](r1/sources.json), [launcher](r1/run_native_dino_r1.sh)
and [saved-only auditor](audit_native_dino_saved.py) require private pinned data.
See [results](../../../../results/original_native518_native_block_v121.json).

Recovery note: this concise README was rebuilt from the frozen `v121-native518-native-block-r1` receipts; its bytes/hash are not claimed to match the lost document.
Source identities are recorded in the [recovery manifest](recovered-source-manifest.json).
Original arrays, weights and SDK binaries stay private; [licenses](../../../../MODEL_LICENSES.md) remain separate.
