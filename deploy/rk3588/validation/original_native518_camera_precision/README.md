# Original native518 CameraHead precision diagnosis

This record compares three saved endpoints for one TUM pair, source indices **[32, 40]**, with original native preparation **518 → 392 × 518**:

- **p0:** original CPU FP32 CameraHead on original CPU upstream tokens.
- **p1:** the same original CPU FP32 CameraHead on actual saved NPU upstream tokens.
- **p2:** the previously saved actual NPU CameraHead output on those same NPU tokens.

The board ran two unchanged original CPU head calls, each with all four refinements and all **69 original parameters**, including the learned empty pose seed. The first call reproduced the original token-normalization output and all four activated poses byte for byte. Only then did the second diagnostic call run. Its input was reconstructed from actual NPU frame/global outputs; no reference tokens were substituted.

See the [public result summary](../../../../results/original_native518_camera_precision_v113.json) for pinned identities, independently checked statistics, and execution limits. Raw receipts, arrays, parameter archives, and full private audits remain private.

## Results

For the final pose encoding, `u = p1 − p0`, `h = p2 − p1`, and `t = p2 − p0`:

| Representation | Upstream propagated difference ‖u‖₂ | Same-token head difference ‖h‖₂ | Total difference ‖t‖₂ |
| --- | ---: | ---: | ---: |
| Full pose encoding | 0.015205910 | 0.003033632 | 0.015396261 |
| Translation, original model gauge | 0.007159964 | 0.001152977 | 0.007058553 |
| Quaternion coefficients, XYZW | 0.002041563 | 0.002498235 | 0.003205144 |
| FoV, radians | 0.013258456 | 0.001277651 | 0.013302207 |

The vectors add, but their norms do not. Their negative dot product shows slight cancellation. No additive contribution percentages are assigned. The head term is larger for quaternion coefficients, so the aggregate pose norm does not establish rotation-error dominance.

Using saved camera-from-world extrinsics, relative two-frame geometry is `E1 @ inv(E0)`. Rotation differences are **0.23964°** for p1 versus p0, **0.07594°** for p2 versus p1, and **0.29758°** for p2 versus p0. The corresponding translation-vector difference norms are **0.00715892**, **0.00106353**, and **0.00625594** in the original model gauge; they are not metric errors in metres.

An independent saved-data check also compares original CPU and actual NPU retained tokens:

| Retained layer | Camera-token relative L2 | Patch-token relative L2 |
| --- | ---: | ---: |
| 4 | 0.0243493 | 0.0729218 |
| 11 | 0.0327871 | 0.0905978 |
| 17 | 0.1377242 | 0.1525976 |
| 23 | 0.0722699 | 0.1296910 |

These are cumulative path discrepancies. Each layer has a different reference norm and representation; this trend does not isolate a faulty block or identify an additive error budget. The final camera inputs match the R113 diagnosis exactly.

## Reproduction and limits

[capture_camera_precision_board.py](capture_camera_precision_board.py) runs the original CameraHead and pose codec on the board. [audit_camera_precision_saved.py](audit_camera_precision_saved.py) performs only saved-array hashing, statistics, and small NumPy geometry calculations. Reproduction requires the pinned original 69-parameter archive, v99 CPU reference, v100 actual NPU outputs, source images, original source, and complete private dependency layout. Published source alone does not supply those inputs.

The CPU-only inline launch used a 180-second TERM timeout, one Torch/BLAS thread, CUDA disabled, and the existing process-local sysroot library path. It finished in **13.0023 s**, with peak RSS **1141.23 MiB**. No thermal or NPU controller was used, and no temperature history was collected. The later read-only NPU 300 MHz observation is not a historical run sample. These timings describe component diagnosis, not full-model performance.

All saved-array identity and recorded cleanup checks passed. Strict elementwise differences remain reported separately from practical quality acceptance. This single pair does **not** definitively explain the 16-pair v111 trajectory gap, establish a deployable mixed executor, or validate ATE/RPE, map accuracy, realtime, stability, or 8 GB operation.

Repository code, original-model source, checkpoint, and dependencies have separate licenses; see [MODEL_LICENSES.md](../../../../MODEL_LICENSES.md).
