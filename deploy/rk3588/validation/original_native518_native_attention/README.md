# Original native518 same-QKV native attention

Two persistent type-1 native RKNN contexts execute every QK and PV product for
all **16 heads and 1,041 queries/keys**. FP16 operands produce FP32 output;
internal accumulation precision is unproven. Fifteen storage-padding keys are
cropped before original scale `0.125` and stable CPU FP32 softmax; probability
and V padding stay zero, with one Half probability boundary before PV.

Original CPU FP32 SDPA on the exact NPU QKV reproduces the frozen reference.
CPU Half SDPA gives **0.04444%** relative L2, versus fused NPU **4.54930%**.
The native route gives **0.011224%**, but still fails the original strict
comparison. Its complete warm boundary is **517.48 ms**; approximately 24 ms
of raw products does not count as complete attention latency.

A separate copy-order ABBA screen measures **520.926 / 516.783 ms**, with
identical outputs; this small difference was not adopted. Restoring 1 GHz
controllers close both contexts and six buffers. These are component screens,
without full-block, trajectory or real-time acceptance.
Use the [native worker](run_native_attention_board.py),
[CPU control](cpu_sdpa_same_qkv_board.py) and
[saved-only auditor](audit_same_qkv_native_saved.py) with private pinned data.
See [results](../../../../results/original_native518_native_attention_v118.json).

Recovery note: this concise README was rebuilt from the frozen `v118-native518-native-attention-r1` receipts; its bytes/hash are not claimed to match the lost document.
Source identities are recorded in the [recovery manifest](recovered-source-manifest.json).
Original arrays, weights and SDK binaries stay private; [licenses](../../../../MODEL_LICENSES.md) remain separate.
