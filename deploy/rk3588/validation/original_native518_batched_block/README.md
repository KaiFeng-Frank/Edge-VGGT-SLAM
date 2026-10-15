# Original native518 DINO0 with batched GPU stages

Batch packing, softmax and merge reduce GPU dispatches from 52 to 7 per block.
All four original weighted products, 32 attention products, 14 parameters,
1,041 tokens and 16 heads remain. Thirty-six native type-1 contexts and 104 DMA
imports run all NPU products serially on core1; no saved intermediate replaces
computation. R2 groups four queries in a 1,024-lane workgroup, retaining each
query's original 256-lane reduction tree; inactive tail queries never store.

| Separate ABBAABBA trial | Fused core1 warm mean | Candidate warm mean |
| --- | ---: | ---: |
| R1: one-query softmax | 180.985 ms | 277.819 ms |
| R2: four-query softmax | 179.143 ms | 284.153 ms |

Each mean excludes the route's first call and uses its following three.
Calls include uploads/products, GPU work/sync, finite owned output; setup,
destruction, saving and comparisons are separate. Complete softmax stage walls
are **86.192 / 94.327 ms**, including pre/post RKNN sync; nested GPU events are
**56.580 / 65.968 ms**. Separate runs under automatic GPU policy do not isolate
a causal layout gain. Both candidates lose to their matched fused comparator.

Every B output matches the preceding native candidate byte for byte. Original
CPU FP32 relative L2 remains **0.025077%**, max error **0.00524735**; strict and
fixed gates still fail. Each run closes 36 contexts, 108 buffers, 28 events and
128 OpenCL objects. NPU controllers restore 300 MHz; GPU policy is unchanged.
Reject both schedules; no full-model, trajectory or real-time gain follows.

Executed bundles/launchers: [R1](r1/sources.json), [R1 launcher](r1/run_native_dino_r1.sh),
[R2](r2/sources.json), [R2 launcher](r2/run_native_dino_r2.sh).
Use the saved-only [R1 auditor](audit_native_dino_saved.py) or [R2 auditor](audit_native_dino_r2_saved.py).
See [results](../../../../results/original_native518_batched_block_v122.json).

Recovery note: this concise README was rebuilt from the frozen `v122-native518-batched-block-r1-r2` receipts; its bytes/hash are not claimed to match the lost document.
Source identities are recorded in the [recovery manifest](recovered-source-manifest.json).
Original arrays, weights and SDK binaries stay private; [licenses](../../../../MODEL_LICENSES.md) remain separate.
