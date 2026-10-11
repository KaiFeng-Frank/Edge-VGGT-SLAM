# DPVO encoder layout diagnostic (v32)

This optional learned-VO component experiment retains the original DPVO parameters and the default VGGT mapper. It adds no ORB/LK/PnP frontend. All actual neural execution was on the 16 GB RK3588S; the host compiled RKNN and independently inspected saved arrays. **The candidate still fails the original FP32 gate and has not been adopted.**

The v27 rewrite replaces ten InstanceNorm nodes with spatial mean/variance arithmetic. Its 44 original initializers, 91 other nodes and I/O are preserved. The prior CPU ORT comparison failed `rtol=1e-4, atol=1e-5`; this new compiler explicitly requires `--diagnostic-allow-fp32-rejected`, verifies that rejected ancestor and retains the failure in every receipt. Successful compilation does not change that status.

## Actual results

| Measurement | Original RKNN | Spatial-norm RKNN |
| --- | ---: | ---: |
| First complete SDK call, frame 0 | 423.071 ms | 228.697 ms |
| Mean of two later calls, frame 0 | 419.830 ms | 207.028 ms |
| Strict FP32 gate against original CPU | Failed | Failed |

Frame 0 ran in same-process AB/BA/AB order; frames 1 and 2 ran BA and AB once each. The observed warm component duration fell 50.7%. Timed calls include input packing, set/run/get, owned FP32 output copies and release. They exclude preprocessing, model/context loading, diagnostics, saving, correlation, Update, BA and pose publication. Two warm samples and uncontrolled temperature/frequency do not establish sustained throughput or end-to-end latency.

All ten ordinary and three separate instrumented calls returned finite, nonzero outputs. Three representative spatial `fmap` relative L2 errors against original CPU are 0.002279/0.002268/0.002337, versus 0.003802/0.003502/0.004135 for the original RKNN graph. The first frame's maximum absolute error increases, so this is not an improvement in every error metric. The `imap` arrays are bitwise identical between the two graphs. No candidate trajectory was run.

Each separate profile has **159 internal NPU operators and three CPU I/O operators**, with no internal CPU Transpose. Total operator time is 139.295–140.566 ms. The CPU I/O rows sum to only 0.757–0.810 ms, while actual `outputs_get` takes 64.897–65.274 ms and owned output copying takes 4.196–20.815 ms in these instrumented calls. The full logical FP32 outputs occupy 36,110,336 bytes. Operator rows therefore cannot substitute for full SDK timing. Profile timings are not the ordinary timings above.

The [complete evidence](../../../../results/dpvo_encoder_layout_v32.json) includes the original receipts, raw SHA256 bindings, environment samples and a 997-check saved-array review. Nine representative NPZ were independently rehashed and recomputed; repeated unsaved calls are covered by driver-recorded raw-array metadata. Normal/profile spatial outputs are byte-identical on all three saved frames. The [engineering records](../../../../docs/ENGINEERING_NOTES.md) describe reusable lessons and unresolved limits.

## Helpers and reproduction contract

- `compile-v32-dpvo-encoder-diagnostic.py`: host RKNN Toolkit2 2.3.2 graph compilation only; verifies the exact rejected v27 ancestor and unchanged graph parameters. No host model forward or real-model ORT session.
- `run-v32-dpvo-encoder-board.py`: RK3588S-only ordinary comparison or separate `--collect-perf-detail` process. It reuses the frozen v26 complete-I/O implementation, checks SDK/header/source identity, owns copies before release and destroys contexts on failure.
- `audit-v32-dpvo-encoder.py`: NumPy saved-array comparisons and pinned v26 profile parser; no model execution. It independently checks schedule, means, array identity/error, placement, provenance and cleanup.

Use `python -B SCRIPT --help` for the complete required arguments. Public driver bytes equal the measured source bytes. The embedded `host_compile_launch`, `board_normal.launch` and `board_profile.launch` records contain the exact measured argument vectors; `<host-home>` is a redacted local prefix. Environment samples and logs are identified separately. Output directories must be new; do not overwrite failed or completed experiments.

The required external artifact chain is:

1. Pinned DPVO source and independently obtained original checkpoint, identified in [v26 provenance](../../../../results/dpvo_sources_v26.json); v26 capture, export, CPU ORT and original RKNN compile directories described in [encoder evidence](../../../../results/dpvo_encoders_v26.json).
2. The exact v27 spatial rewrite ONNX and `rewrite.json`, plus the preserved failed board CPU ORT directory in [v27 evidence](../../../../results/dpvo_norm_rewrite_v27.json). Recreating an artifact with different receipt bytes requires a separately reviewed lineage, not relabeling its hash.
3. External RKNN Toolkit2/runtime/API header matching the pinned 2.3.2 hashes; the published v12 compiler/ABI helper, v26 runner/profile auditor, v27 rewriter and verified external DPVO `extractor.py`.
4. Compile using the diagnostic flag, copy only the resulting model/receipt to the board, and run ordinary and profile modes into distinct directories. Compare both against the same already-normalized original inputs and original CPU/ORT references.
5. For the saved-array audit, supply `--normal-dir`, `--profile-dir`, `--capture-dir`, `--ort-dir`, `--compile-dir`, `--candidate-compile-dir`, `--rewrite-dir`, `--failed-ort-dir`, the published source/parser paths, the four required receipt/source hashes, and a new `--output`. Optional old profile evidence requires both its directory and SHA; it is explicitly historical, not a same-time baseline.

This is an artifact-bound diagnostic harness, not a standalone installer. Checkpoints, ONNX/RKNN, SDK dependencies, images and NPZ are intentionally external. The adjacent [MIT notice](LICENSE) preserves DPVO attribution; the code license does not grant a license for the separately unverified DPVO checkpoint or change the original VGGT weight restrictions.

Tiny regression checks use generated metadata/arrays and fake SDK objects only:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  tests/test_v32_encoder_diagnostic_compile.py \
  tests/test_v32_encoder_runtime.py \
  tests/test_v32_encoder_audit.py \
  tests/test_v32_source_frame_plan.py
```

The separately published [source-frame plan](../dpvo_dynamic_update/source_frame_plan.py) and [Update diagnosis](../../../../results/dpvo_update_plan_v32.json) are static/saved-array research. They have not yet produced a new neural wrapper, export, NPU execution or speedup. Resolution reduction and sparse context-output experiments are also pending. The encoder alone remains above the 50 ms frame budget; no complete VO, 20 Hz tracking, camera latency, dense mapping, endurance or 8 GB acceptance follows from this result.
