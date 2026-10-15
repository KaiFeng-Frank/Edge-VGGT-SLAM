# Original native518 DINO0 attention dump

One unchanged original DINO0 graph executes all 1,041 tokens and 16 heads.
The SDK debug dump preserves the previously observed norm1/QKV and untapped
final bytes. On the exact dumped Q/K/V, fused NPU attention differs from
original CPU FP32 SDPA by **4.54930% relative L2**. This crosses backend and
precision boundaries; it does not identify a faulty QK, softmax or PV kernel.
The debug run is not an ordinary performance benchmark.

Changing only compilation `optimization_level` from 3 to 0 produces the same
**26,517,686-byte** RKNN artifact, SHA `fe8acac4…8b32c0`; board comparison and
host hashes agree. No runtime comparison of identical binaries was performed.
The option therefore creates no distinct deployed candidate in this test.

The [executed worker](run_dino0_debug_dump_executed_r1.py),
[launcher](run_debug_dump_r1.sh) and
[saved-only auditor](audit_dino0_debug_dump_saved.py) require private pinned
inputs, references and board libraries. The debug controller restores the
prior NPU policy. No full-model, trajectory or real-time acceptance follows.
See [results](../../../../results/original_native518_attention_dump_v117.json).

Recovery note: this concise README was rebuilt from the frozen `v117-native518-attention-dump-r1` receipts; its bytes/hash are not claimed to match the lost document.
Source identities are recorded in the [recovery manifest](recovered-source-manifest.json).
Original arrays, weights and SDK binaries stay private; [licenses](../../../../MODEL_LICENSES.md) remain separate.
