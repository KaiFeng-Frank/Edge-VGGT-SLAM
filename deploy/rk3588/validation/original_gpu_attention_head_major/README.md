# Original attention: head-major GPU boundary failure

This rejected experiment exposes the original attention values as `[B,16,S,64]`, retaining the original downstream transpose/flatten, all tokens, 36 parameters and 63 initializers. The current complete pipeline remains [v82](../original_calibrated_camera/README.md).

Static export and board-only FP16 RKNN compilation succeeded (51,163,047 bytes). Compilation used original CPU ORT callbacks, not GPU runtime validation. During GPU registration, the SDK-generated `pack_nchw_to_nchwc8_buf.cl` reported eight undeclared `To8` errors, custom initialization returned −1, and the worker terminated with SIGSEGV (−11).

No complete-pair result or output was produced: accuracy, determinism and latency are unmeasured. The observed layout-program failure does not establish an exclusive crash cause or a fix. Three saved NPU samples were 1 GHz, maximum 51.769°C; the controller restored 300 MHz and the original governor, limits and request. No runtime resource-cleanup receipt survived the crash.

| Executed entry | Purpose |
|---|---|
| `run_export_head_major.sh` | Static export with `build_pair_attention_head_major_board.py` |
| `run_compile_head_major.sh` | Nonquantized compile with `compile_pair_attention_head_major_board.py` |
| `run_head_major_pair_1ghz.sh` | Complete-pair A/B; stopped during candidate initialization |
| `rknn_original_attention_head_major_custom.py` | Typed callback registration and ownership |
| `original_attention_tail_borrowed_headmajor.py` | Strict GPU kernels with the original head-major output |

The [source list](source-copy-list.json) pins five executed Python files and three exact board-staged launchers. Unexecuted drafts and the inaccurate private source-freeze list are excluded. See the [saved failure audit](runtime-failure-audit.json) and [result](../../../../results/original_gpu_attention_head_major_v84.json).

Reproduction requires private original weights, v53/v56 captures and graphs, v83 helpers and the pinned board libraries. Launchers retain their actual staging/output paths; prepare those artifacts and fresh outputs first. Compile and runtime Python environments are separate. Use only the existing [bounded 1 GHz / 180 s / 75°C controller](../original_clock_test/npu_frequency_test.py); no GPU governor or limit changes were made.

This GPU route is stopped. There is no new full-model, trajectory, replay or 20 Hz/100 ms acceptance claim. No arrays, graphs, weights, headers, binaries or firmware are distributed; [code and original weights have separate licenses](../../../../MODEL_LICENSES.md).
