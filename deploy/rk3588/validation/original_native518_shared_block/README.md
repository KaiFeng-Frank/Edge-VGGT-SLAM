# Original native518 DINO0 shared-attention cuts

The actual original norm1/QKV prefix feeds shared NPU/Mali attention, whose
output and the original input residual feed the original projection/MLP
suffix. All **31 nodes, 23 initializers and 14 original parameters** remain.
Board FP32 cut composition is byte-identical to its parent and passes all
five original reference gates. Only the weighted cuts are compiled.

The full eight-call **ABBAABBA** screen averages **175.830 ms** for fused
execution and **396.803 ms** for the real connected candidate, excluding first
calls. Complete prefix/attention/suffix means are **60.999 / 180.693 / 151.714 ms**;
ordinary IO, conversion, owned copies and finite checks remain within calls.
Setup/destruction are separate. Candidate output relative L2 against original
CPU FP32 improves from **2.54823% to 1.27336%**, but strict precision still fails.

All repeated route outputs are byte-stable. Weighted/native contexts and GPU
imports/events close; the restoring 1 GHz controller returns to 300 MHz.
Reject deployment or expansion of this cut schedule. The preceding component
improvement does not establish block, full-model or trajectory acceptance.

Use the [exporter](export_dino0_cuts.py), [gate/compiler](check_compile_dino0_cuts_board.py),
[worker](run_dino0_shared_block_board.py) and [auditor](audit_dino0_shared_block_saved.py)
with private pinned artifacts. See [results](../../../../results/original_native518_shared_block_v120.json).

Recovery note: this concise README was rebuilt from the frozen `v120-native518-shared-block-r1` receipts; its bytes/hash are not claimed to match the lost document.
Source identities are recorded in the [recovery manifest](recovered-source-manifest.json).
Original arrays, weights and SDK binaries stay private; [licenses](../../../../MODEL_LICENSES.md) remain separate.
