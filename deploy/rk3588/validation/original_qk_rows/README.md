# Original frame0 Q/K row-layout diagnostic

This RK3588S experiment retained the original VGGT checkpoint, all 36 pair00 parameters and all 63 original initializers. It changed only frame0's Q/K/V layout to `[32,53,64]`, preserving each head's 64-channel normalization, original RoPE, 53-key attention and original output layout. Global0 remained unchanged. Four new INT64 layout constants were added.

The candidate was **not adopted**. At the authorized, bounded 1 GHz setting, normal complete-pair warm time was **17.419717 ms for A** and **17.883380 ms for B**, a **2.661715% slowdown**. Both real inputs were slower. This was a component test; it added no full-model, trajectory or timestamp-replay result.

| Validation | Observed result |
| --- | --- |
| FP32 ONNX, two real inputs and both complete outputs | All four outputs were byte-identical to the original graph and passed the fixed `rtol=1e-4, atol=1e-5` gate. |
| W8A8 normal execution | 28 calls, 56 owned output arrays; 12 warm calls per route after excluding the first call for each input/route. All repeats were byte-identical within their route. |
| W8A8 accuracy | All 56 outputs failed the fixed original FP32 ONNX tensor gate. On the v53 input, B differed from A in 4 frame-output elements and 46 global-output elements; the calibration input outputs matched A bytewise. |
| SDK resources and clocks | Both normal and profile jobs closed their two contexts and released every acquired output. Recorded samples were 1 GHz; the controller restored the prior 300 MHz request, governor and limits. |

A reused the existing original pair00 W8A8 artifact. B used exactly the same eight original FP32 frame0 activation files and dataset bytes, with `w8a8`, `normal`, `channel` and no automatic hybrid selection. The existing A leaf lacks a release-status field; this is not treated as historical cleanup proof. This round's runtime lifecycle was checked separately. The second numerical test input is `v56-calib00`, so these two inputs do not constitute a held-out quality evaluation.

The candidate build completed with all four API return codes equal to zero. Its SDK layer table lists 47 NPU rows, three CPU input/output rows and two NPU SDPA operators. INT8 activation annotations do not establish that every internal operation or accumulator uses INT8; FLOAT16/FLOAT normalization constants remain. Both routes used the same FP32 SDK conversion interface, which does not imply internal FP32 execution.

The four profile calls used `flags=8`, one call per input/route. They are placement diagnostics and are excluded from the normal `flags=0` means. Timing covers finite input checks, SDK conversion/run/output retrieval, owned copies, release and finite output checks; hashing, saved files and comparisons are outside this component timer.

The compiler moved the source Split after four shared QKV layout conversions. This removed several later operand transposes but retained roughly 2 ms of instrumented layout work before attention. Fewer operators therefore did not establish a speedup. The separate profiles do not isolate the cause of the normal-run slowdown; see the [profile audit](profile-audit.json).

See the [result summary](../../../../results/original_qk_rows_v86.json), [FP32 audit](FP32-audit.json), [compile-lineage audit](compile-audit.json), [saved component audit](component-audit.json) and [source identities](source-copy-list.json).

For reproduction, use the recorded launchers in order: [rewrite](run_qk_rows_rewrite.sh), [FP32 gate](run_qk_rows_FP32.sh), [compile B](run_qk_rows_compile.sh), then separate [normal](run_qk_rows_probe_1ghz.sh) and [profile](run_qk_rows_profile_1ghz.sh) jobs. The first three use the isolated compiler environment; the SDK jobs use the production environment. Keep the gate and source SHA arguments, a new output directory for each job, and the existing bounded controller around compilation and authorized 1 GHz execution. Private checkpoints, graphs, calibration arrays and SDK files must be prepared from their pinned provenance; they are not included in this package.
