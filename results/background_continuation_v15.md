The continuous frontend now advances delayed depth candidates in a background worker before applying them to the live tracker. Two RK3588S runs used the original VGGT weights at image size 112, CPU XNNPACK FP16 linear kernels, and a Shi–Tomasi/LK/PnP frontend with triangulated landmark replenishment. Neither run used ORB or the NPU. The input was the 798 original RGB frames from TUM `fr1_xyz`, replayed with their original irregular timestamps over 26.572 seconds; this is software replay, not a physical-camera test.

| Recorded result | r4: VGGT-estimated frontend intrinsics | r5: explicit TUM frontend calibration |
| --- | ---: | ---: |
| Source frames / delivered observations / overwritten frames | 798 / 698 / 100 | 798 / 739 / 59 |
| Fresh valid local poses | 532 (66.67% of source) | 576 (72.18% of source) |
| Waiting observations / lost observations | 166 / 0 | 163 / 0 |
| First valid publication after replay start | 5.592 s | 5.494 s |
| Valid-pose latency, median / P95 / maximum | 64.78 / 101.35 / 115.63 ms | 56.49 / 89.83 / 106.30 ms |
| Largest gap between valid publications | 77.90 ms | 62.94 ms |
| Valid publications in full one-second bins, seconds 6–25 | 15–31 | 20–31 |
| Mapper updates / processing-time range | 7 / 3.515–5.254 s | 6 / 3.593–5.105 s |
| Main-thread candidate-application time | 6.14–11.31 ms | 8.12–11.01 ms |
| Mapper-process peak RSS | 7,137.56 MiB | 7,155.09 MiB |

Latency starts at the scheduled RGB arrival, before image decoding, and ends when the tracking JSONL write and flush return. Startup and waiting periods remain in the per-second counts. The r5 post-initialization window meets the 20 Hz / P95 ≤100 ms timing targets for this short replay, but the complete run starts with about 5.49 seconds without valid geometry. No complete stable-rate acceptance is claimed. Mapper processing includes preparing and publishing the map; the applied geometry was already 5.09–7.58 seconds old across these runs once historical replay and catch-up were included. A configured three-second submission period therefore does not establish a three-second usable-geometry age.

An independent saved-log check matched every tracking row to exactly one publication with strictly advancing source frame IDs and capture times, verified valid finite SE3/PnP results and null poses for waiting rows, and reconciled the 297 / 307 historical replay poses separately from the 532 / 576 fresh poses. Each run had one post-EOF depth adoption, containing 16 / 44 historical poses; these produced no fresh-pose rows. One r4 publication and two r5 publications completed after reader EOF for genuine frames captured before EOF. They remain distinct new observations, not synthesized tail poses or repeated poses.

Observed candidates used at most two background continuation rounds and at most one remaining observation during main-thread application. Recorded capacities are one capture slot, one mapping job/result slot, one pending geometry packet and one candidate worker. The tracker retained one gray image, at most six local keyframes, and bounded history (300 observations / ten seconds); observed history peaks were 282 / 286 and landmark peaks 379 / 397, below the configured 1,200-landmark cap. Both mapper processes exited with code zero and empty cleanup-error lists. These short-run checks do not establish 30-minute endurance or an 8 GB combined-process memory bound.

The r5 calibration applies only to the frontend. Its nonzero distortion makes a single linear transform from calibrated local coordinates to the raw-image VGGT map invalid: every valid r5 row explicitly sets model association unavailable, leaves `H_model_from_local` null, and emits no global association or graph revision. Its results are local poses in VGGT-depth units, with a new gauge after depth adoption. The r4 graph association is projective rather than a validated metric SE3 trajectory. Neither run establishes GT accuracy, metric scale, rigid global continuity, nonblocking loop closure, or physical-camera latency.
