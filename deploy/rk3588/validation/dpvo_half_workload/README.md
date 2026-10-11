# Half-resolution DPVO workload research (v39)

This optional experiment changes the DPVO workload while preserving its original checkpoint. It does not replace the default VGGT model. The first 192 consecutive TUM fr1_xyz RGB observations run entirely on RK3588S. No ORB, added LK/PnP, GT input, sensor depth or interpolated fresh poses are used.

The frozen [policy](policy.json) declares 232×304 input, M16/L4/R8/O4/KF2, original motion/keyframe thresholds, shared CPU RNG seed 20261011, four A76 threads, unchanged stream lifecycle, original Torch Update, native per-frame correlation and CPU BA. These are changed algorithm settings, not equivalence to original FAST. The original full-resolution crop is resized with INTER_AREA; intrinsics include the half-pixel correction. Level-4 pooling of the 58×76 feature map floors to 14×19. R8 is the minimum removal window compatible with the unchanged startup lifecycle; R6 and R7 fail before pruning.

## Actual full replay and quality

The instrumented CPU replay completed 192 observations, with 158 fresh poses starting at observation 34. Input-active replay took 92.874 s (1.701 fresh poses/s); software admission-to-publication P95 was 668.453 ms, including JSON event serialization/flush. It also includes diagnostic capture at observations 34/127/191. Those three full Update states contain 704/800/800 edges. EOF optimization is separate (3.434 s). Peak RSS was 361.332 MiB on the 16 GB board, with 94 original parameters unchanged and cache cleanup complete.

Evaluation uses the same 158 common fresh observations and 128 common one-second endpoint pairs planned from all 192 source timestamps before success filtering. Each population has one positive Sim3 alignment for monocular scale. Original full-resolution FAST CPU versus this half configuration:

| Metric | Original FAST CPU | Half CPU | Increase |
| --- | ---: | ---: | ---: |
| ATE RMSE | 1.113 cm | 1.971 cm | 77.15% |
| Translation RPE RMSE | 1.709 cm | 2.969 cm | 73.78% |
| Rotation RPE RMSE | 0.986° | 1.668° | 69.26% |
| Fresh coverage | 179/192 | 158/192 | — |

This configuration fails the preliminary 10–20% quality-loss target and remains unadopted. This compares two DPVO workloads, not original VGGT accuracy. Separate runs were not thermally interleaved; no controlled speedup claim is made. Physical-camera latency, 20 Hz, dense mapping, 30-minute endurance and 8 GB operation remain unvalidated.

## New-size encoder checks

The static editor changes only three exact Reshape constants and declared I/O dimensions in the original encoder. The reciprocal sparse candidate additionally changes its flatten shape and patch count to 16; all other numeric parameters and nodes remain unchanged. The graph editor runs without inference on the host.

On the board, three newly preprocessed inputs establish original Torch references at the new size. Both the original half-resolution ONNX and the reciprocal half-resolution ONNX pass the unchanged `rtol=1e-4, atol=1e-5` gate against those references. This small new-size result does not erase the full-resolution parent's historical rejection. Calibration uses the predeclared 32 source rows 256+16i, disjoint from the replay prefix but from the same scene; it preserves BGR normalization and INT64[16] indices.

## Actual quantized SDK and complete M16 Update components

The new 32-sample board PTQ build took 16.169 s and produced a 1,521,328-byte RKNN artifact. Six ordinary calls and three separate profiling calls execute on RK3588S. Three warm complete SDK calls average **22.024 ms**, including typed FP32 NHWC/INT64 input packing, set/run/get, owned FP32 output copies and release. Original CPU, original ORT and reciprocal ORT strict gates all fail. Feature relative L2 versus original CPU is **10.243–11.177%** for fmap and **2.647–3.258%** for context. These errors are not trajectory ATE; no full replay uses this quantized half encoder.

The three complete M16 Update captures also run grouped Torch and CPU ORT on the board, using eight actual `(B,D,16)` shapes and four groups per full graph. Every original Update parameter and edge is retained; original full-order net/delta/weight comparisons pass the fixed FP32 gate on all three captures. This is saved-state component validation, with no recurrent feedback, BA or NPU Update inference. The original configuration's trajectory rejection remains. New RKNN compilation and execution of these eight shapes are still required.

## Reproduction boundaries

- `run-v39-half-torch-board.py --help` lists the full CPU replay's pinned inputs. Supply the frozen v34 helper, v30 stream/sampling, v36 native frame-cache library/build and original checkpoint. It refuses non-RK3588 execution and new output paths must be under the isolated board experiment root.
- `rewrite-v39-dpvo-half-encoder.py --help` lists the three pinned parent artifacts. It edits ONNX structure only, with no model/ORT/SDK execution.
- `run-v39-half-encoder-board.py probe --help` and `calibration --help` produce the new references and calibration tensors on the board. Calibration data preparation does not itself run a model; PTQ calibration does and must also stay on the board.
- `evaluate-v39-workload.py --help` and `audit-saved-v39.py --artifact-root ROOT` read saved results only. GT is used only after inference. The latter expects the isolated experiment directory structure recorded in the receipt.
- `compile-v39-half-int8-board.py --help`, `run-v39-half-int8-board.py --help` and `export-v39-m16-grouped-update-board.py --help` expose separate board-only PTQ, SDK and complete Update checks. The latter reuses the frozen helpers in `../dpvo_grouped_update`; its eight exported graphs are not included in Git.
- Tiny pytest cases cover shape preservation, preprocessing contracts, board guards and evaluation timing/coverage; they do not execute the real network on the host.

[Full metadata evidence](../../../../results/dpvo_half_workload_v39.json) preserves raw receipt hashes, source identities, all observations, failures and separate EOF results. Model weights, ONNX/RKNN, input/feature arrays, native/SDK binaries and dataset images are not published. DPVO-derived code attribution is in the adjacent MIT [LICENSE](LICENSE); model-weight permissions remain separate in [MODEL_LICENSES.md](../../../../MODEL_LICENSES.md).

For generated-pose tests set `EDGE_DPVO_EVALUATOR_HELPER` to `../dpvo_fast/evaluate-v34-fast-trajectory.py` as an absolute path; for generated M16 topology tests set `V39_GROUPED_SOURCE_DIR` to the absolute `../dpvo_grouped_update` directory.
