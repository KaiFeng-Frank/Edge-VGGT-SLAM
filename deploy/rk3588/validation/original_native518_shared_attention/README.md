# Original native518 shared-memory attention

Native NPU QK writes its FP32 C4 buffer; onboard Mali imports that DMA buffer
and writes Half probabilities directly into native PV A8 storage. All 16 heads,
1,041 original keys/queries, scale `0.125` and zero padding remain. No large
score matrix passes through a CPU staging array on this route.

An eight-call **ABBAABBA** screen averages **527.049 ms** for native CPU
softmax and **161.919 ms** for shared softmax, excluding each route's first
call. Complete timing includes all 32 products, dynamic packing/transfers,
synchronization, GPU event/status handling and finite owned output.
Initialization/destruction are separate; three warm samples do not prove P95.

Last-head GPU probabilities pass the unchanged strict tolerance against CPU
softmax on those exact NPU scores, rounded once to Half. Whole attention still
fails original FP32 strict tolerance. Resources close GPU-first; the 1 GHz
NPU controller restores 300 MHz. GPU policy is untouched, with 300 MHz endpoints
and no continuous trace. Initial nested GPU counters alias later cleanup and
are not an independent initial snapshot; final ledgers are separately checked.

Use the [driver](run_shared_attention_board.py), [launcher](run_shared_attention_r1.sh)
and [auditor](audit_shared_attention_saved.py); see [results](../../../../results/original_native518_shared_attention_v119.json).
The [complete-block integration](../original_native518_shared_block/README.md)
was slower; this component gain is not a full-model improvement.

Recovery note: this concise README was rebuilt from the frozen `v119-native518-shared-attention-r1` receipts; its bytes/hash are not claimed to match the lost document.
Source identities are recorded in the [recovery manifest](recovered-source-manifest.json).
Original arrays, weights and SDK binaries stay private; [licenses](../../../../MODEL_LICENSES.md) remain separate.
