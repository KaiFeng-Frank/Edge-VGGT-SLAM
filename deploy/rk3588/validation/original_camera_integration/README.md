# Original CameraHead split4 in complete VGGT and timestamp replay

The optional camera path uses the same original VGGT checkpoint and all four
CameraHead refinements. Only the four trunk FC1s inside the existing shared
step graph are partitioned along their output channels. The original learned
seed, unactivated FP32 recurrence, pose activations, aggregator, DPT, pose codec,
submap equations and SL4 remain present. It adds one optional execution choice,
not another model. Defaults are unchanged.

On the RK3588S at 300 MHz, one CPU model thread and CPUs 4–7, two sequential
complete runs used the same 17 real TUM images and 16 overlapping pairs:

| Measurement | Existing camera graph | Original split4 graph |
| --- | ---: | ---: |
| Complete model + SL4, all 16 pairs | 1.813212 s | 1.738321 s |
| Same service, excluding first pair | 1.754025 s | 1.695149 s |
| Camera phase, all 16 pairs | 211.622 ms | 170.791 ms |
| Process peak RSS | 7,198.77 MiB | 7,078.84 MiB |
| Sim3-aligned ATE RMSE | 0.174760 m | 0.174725 m |
| Adjacent matched-pose translation RPE RMSE | 0.315017 m | 0.314895 m |
| Rotation RPE RMSE | 10.886887 deg | 10.859345 deg |

Complete service falls 4.13%, or 3.36% excluding each first pair. These are
sequential measurements; changes in the other phases and startup caching
prevent attributing every saved millisecond solely to the camera graph.
All 80 retained/target arrays and all 32 depth/confidence arrays remain
byte-identical. Pose encoding, extrinsics and intrinsics change. Both routes
still fail all 80 strict original CPU FP32 tensor checks. The candidate's
three trajectory RMSE changes relative to the same original CPU baseline are
−1.392%, −1.581%, and +0.876%. This short, sparsely sampled trajectory does not
establish general map quality, metric scale or failure rate.

A separate timestamp replay consumed a source of 798 images and published
18 fresh pose/map updates in 29.554 s including EOF drain: **0.609 Hz** and
**complete input-to-pose P95 2.396 s**. Startup was 37.246 s, separately
recorded; peak process RSS was 7,198.125 MiB on the 16 GB board. The one-slot
queue remains bounded by dropping older frames. This is below the real-time
target. The selected frames differ from both the fixed-input comparison and
previous replays; those trajectory results cannot be transferred to this run.
This experiment did not change the 300 MHz clock or run a new 1 GHz test.

## Reproduce on the prepared board

The fixed original compiled artifacts and v69 build/ORT receipts must already
be present. They contain no substitute weights and are not distributed here.
The adapter verifies their hashes, exact FC1 reconstruction proof, original
69 camera parameters and typed SDK bindings before installation. It uses one
persistent camera context for all four refinements. The original forward
source is unchanged. A failed context destroy now preserves its handle and
reports the outstanding context instead of clearing it.

Source the existing `deploy/rk3588/env.sh` first. The three command JSON files
contain exact argument arrays; each needs a new output directory:

- `baseline-command.json`: complete original retained-group path, old camera.
- `candidate-command.json`: identical inputs with `--camera-split4`.
- `replay-command.json`: timestamp replay with `--rknn-camera-split4` and
  `--rknn-camera-head`; the decoder remains on CPU0.

`run_board_stage.py` records the child return code, frequency and temperatures
and stops only its own process group at its timeout or 75 C observation limit.
It does not change clock, governor or cooling. Normal completion must be
verified from its terminal receipt, because an outer ADB return code alone is
insufficient.

Host checks use saved NumPy arrays and generated service/SDK doubles only.
All actual original-model execution occurred on the board. There are 74
passing targeted generated checks; the complete and replay runs separately
verify actual installation, finite outputs and recorded SDK cleanup.
Saved-array auditing reuses the original trajectory evaluator and the retained
groups' replay auditor; it performs no model inference.

Physical camera, loop closure, 30-minute stability and 8 GB operation remain
unvalidated. Only `max_loops=0` is supported.

[Measured results and evidence hashes](../../../../results/original_camera_integration_v71.json).
