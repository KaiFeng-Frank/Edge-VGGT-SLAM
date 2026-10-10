Saved-data TUM trajectory diagnostic: the calibrated frontend has lower error on 256 common one-second RPE endpoint pairs. Both runs use the original VGGT weights at 112 pixels with CPU/XNNPACK-f16 and no ORB.

| Diagnostic | Model K, r4 | Official frontend K + distortion, r5 |
|---|---:|---:|
| Common 1 s rotation RPE RMS, 256 pairs | 11.367° | 3.635° |
| Common translation-direction median, 253 pairs | 55.352° | 6.191° |
| Fresh valid poses / 798 source frames | 532 (66.67%) | 576 (72.18%) |
| All evaluated 1 s rotation RPE RMS | 10.483° (337 pairs) | 3.605° (405 pairs) |
| First fresh pose after replay start | 5.592 s | 5.494 s |
| Replay valid-pose latency P95 | 101.352 ms | 89.826 ms |

Common endpoints control evaluation timestamps, not the different landmarks, anchor schedules, coverage or model batches. Three common pairs with GT translation below 1 cm retain unknown direction. These are diagnostic observations, not a controlled accuracy regression or stable 20 Hz acceptance.

TUM GT uses camera-to-world poses and scalar-last quaternions. Relative GT is `inv(Twc_current) @ Twc_anchor`; one-second RPE stays within each local gauge. Nearest GT must be within 20 ms; no time offset, interpolation or fitted GT scale is used. Independent saved-value checks confirmed timestamp/anchor bindings and pose direction. The [official TUM format and calibration](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) define these conventions.

r5 calibration applies only to the local frontend. Its mapper still receives raw RGB and estimates model K; every valid r5 global association remains unavailable because nonlinear distortion lacks a common undistorted model input. This does not validate global metric ATE, 518 accuracy retention, NPU trajectory performance, physical-camera latency or 30-minute endurance. Exact provenance, denominators and hashes are in [trajectory_v15.json](trajectory_v15.json).
