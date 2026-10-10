# VGGT-SLAM 2 on RK3588S

An experimental RK3588S port of [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM), retaining the original VGGT-1B weights and SL4 mapping backend. The current frontend uses LK, PnP and triangulation; it does not use ORB. All measured model inference runs on the board. Host tools export and compile graphs or inspect saved arrays.

**Status: sustained accurate 20 Hz tracking, 2–5-second map updates, 30-minute endurance, physical-camera latency and 8 GB operation have not passed validation.** The continuous RGB replay is an experimental diagnostic, not a completed real-time SLAM system. The current hybrid mapper uses CPU/XNNPACK for most of VGGT and RKNN for the original four-iteration CameraHead. It is not an all-NPU implementation.

## Measured baseline

On the tested 16 GB RK3588S, original CPU FP32 VGGT with four A76 threads and two office images took 3.831 seconds at a requested image size of 112 (actual input 56×112). Its camera head took about 1.250 seconds in the separate hooked run. Size 84 took 3.161 seconds for the full forward; its camera head remained about 1.268 seconds. These are short measurements, not sustained throughput or latency percentiles; they exclude image loading, graph optimization and publication.

An original-weight DINO block with input `[1,37,1024]` executed using RKNN 2.3.2: its last three complete I/O calls averaged 5.489 ms. Four separate SDK profile calls reported all 14 internal operators on NPU and only input/output operators on CPU. Its relative L2 error against PyTorch FP32 was 1.2253%, so full-network pose/depth accuracy remains unvalidated. A CPU-Softmax/Erf variant is a mixed implementation and is not the all-NPU result.

[Measurement evidence](MEASUREMENTS.json) records receipt hashes and the measured operator targets.

The **complete original CameraHead** also ran separately through RKNN, retaining all four refinement iterations and 16 trunk-block calls. Its first complete I/O call took 185.979 ms and the next three averaged 203.646 ms; three FP32 CPU reference calls averaged 1.254 s. Actual SDK profiles show 207 internal NPU operators and one final CPU Reshape, plus CPU input/output. Relative L2 error against FP32 is 0.7669%; normalized quaternion differences are 0.778° and 0.869° for the two frames. The strict elementwise comparison does not pass. SDK DMA allocation is 3,424,231,424 bytes. That unrolled variant was not integrated into the replay mapper; the shared-step version described below now runs inside the mapper. [Timing, numerical error and provenance](results/camera_head_v15.json) are published separately from the whole-pipeline results.

The full model and the continuous tracker still require further work. Dataset timestamps, startup, lost frames, latest-slot replacement and EOF work must remain visible in the results; high input rate alone is not a valid-pose rate.

## Continuous replay results

The 26.572-second TUM RGB sequence has now run end to end on the board. These are complete-run results, including startup and losses:

| Run | New valid poses / 798 RGB frames | First valid pose | Successful-pose latency P95 | Map update time |
| --- | ---: | ---: | ---: | ---: |
| CPU FP32, 480-wide/800 features | 0 / 798 | unavailable | unavailable | 7.71–8.03 s |
| CPU FP32, 320-wide/400 features | 251 / 798 | 17.34 s | 90.19 ms | 7.23–7.88 s |
| XNNPACK FP16 CPU, 320-wide/400 features | 397 / 798 | 5.49 s | 90.55 ms | 3.85–5.93 s |
| XNNPACK + background depth continuation (r4) | 532 / 798 | 5.59 s | 101.35 ms | 3.51–5.25 s |
| Same, official frontend calibration diagnostic (r5) | 576 / 798 | 5.49 s | 89.83 ms | 3.59–5.11 s |

Runs r2–r5 replay every third retained historical observation when depth arrives; online observations still run PnP individually. This is not an equivalent-numerics optimization. The original XNNPACK run (r3) has a 6.27-second gap in valid pose publication. Background continuation in r4 moves delayed observation catch-up off the tracking owner thread: online adoption takes 6.14–11.31 ms, and there are no lost delivered observations after initialization in this short sequence. Its successful-pose P95 still exceeds 100 ms, and some subsequent full-second windows contain fewer than 20 valid poses.

The r5 diagnostic supplies the official TUM camera intrinsics and distortion to the **local frontend only**. It has 20–31 valid publications in each complete one-second window starting at second 6; the first five full windows have zero poses, and second 5 has 16. The mapper still receives original distorted RGB and estimates its own intrinsics. Therefore no single projective matrix connects this calibrated frontend to the unchanged mapper, and the runner explicitly reports global association as unavailable. These local timing results do not establish a connected, accurate global SLAM system.

Mapper peak RSS is approximately 7.0 GiB in these XNNPACK runs; this excludes the OS and is not a whole-board simultaneous memory measurement. No loop closure was enabled. [Machine-readable results](results/tum_fr1_xyz_v15.json) retain complete-run coverage, per-second counts, losses, calibration scope and source hashes. Physical camera latency, accuracy against a full-quality baseline and 30-minute endurance remain unvalidated.

Depth geometry was already 5.09–7.58 seconds old when the frontend applied it, including historical replay and continuation. Map processing time therefore differs from usable-geometry age. [The saved-log review](results/background_continuation_v15.md) details queue bounds, adoption times and EOF accounting.

Ground-truth diagnostics show substantial remaining geometric error. On the 256 identical one-second endpoint pairs available in both runs, rotation RPE RMS is 11.367° for r4 and 3.635° for r5; translation-direction median error is 55.35° and 6.19° on the 253 pairs with sufficient ground-truth translation. This is a local relative-pose diagnostic: initialization, selected map anchors and depth geometry differ between runs. It does not establish metric ATE, global continuity or accuracy within 10–20% of the full-resolution baseline. [Evaluation results](results/trajectory_v15.json) retain coverage, association rules and input hashes.

## Shared rectification and integrated NPU CameraHead (v16)

Capture and VGGT now receive the same calibrated, rectified RGB pixels. Rectification preserves the original K and image extent, with identity rotation and no crop. Capture decode/remap runs after the original scheduled arrival and is included in pose latency. The mapper retains original VGGT preprocessing through a lossless in-memory PNG bridge. Both processes verify identical calibration metadata. Projective association is restored; a general SL4 transform still does not define a metric global SE3 pose.

The capture source keeps one decoded frame and selects the latest already-due source image before decoding when it falls behind. It records skipped-before-decode and overwritten-after-decode separately. Original timestamps remain unchanged. This removed the multi-second backlog observed in r6.

| Run | Fresh valid poses / 798 | First valid pose | Successful-pose latency P95 | Map processing |
| --- | ---: | ---: | ---: | ---: |
| r6, shared rectification, old decoder | 595 | 5.82 s | 3667.42 ms | 3.65–5.30 s |
| r7, latest-due decoder, CPU/XNNPACK | 515 | 5.64 s | 139.65 ms | 3.62–5.18 s |
| r8, capture shares A76 core with mapper | 258 | 16.11 s | 111.85 ms | 6.15–9.62 s |
| r9, NPU head, 2 mapper cores, 400 features | 432 | 7.19 s | 121.47 ms | 4.74–6.13 s |
| r10, same allocation, 300 features | 511 | 6.92 s | 103.65 ms | 4.63–6.05 s |
| r11, same allocation, 240 features | 567 | 6.55 s | 93.94 ms | 4.59–5.90 s |

In r11, complete publication windows from second 7 through second 25 contain **22–31 new-observation poses each**. The first six complete seconds contain zero poses; second 6 contains 14. All 798 original images were decoded, 36 were overwritten in the latest slot, and 762 reached tracking. The 567 valid poses have model/global **projective** associations; rigid global poses remain unavailable. Six map jobs executed 24 actual RKNN calls. EOF depth adoption does not count as a fresh pose. These are results for one 26.572-second sequence, not sustained 20 Hz acceptance, physical-camera testing, or a completed SLAM accuracy result. [Complete coverage, timing and memory samples](results/rectification_v16.json), [independent replay audit](results/replay_review_v16.json) and [trajectory diagnostics](results/trajectory_v16.json) include the slower and failed-target configurations. Temperature and scheduling differed across runs, so these are not controlled causal speedup measurements.

The shared CameraHead graph retains all original weights and all four refinement iterations, executing the same context four times with CPU FP32 recurrence. RKNN file size is 434,452,453 bytes; SDK DMA is 869,425,152 bytes, down from 3,424,231,424 for the unrolled graph. The isolated complete head averages 212.707 ms over its last three calls; relative L2 versus the original CPU reference is 0.7873%. Actual profiles show 57 internal NPU operators and two CPU Reshapes per step, plus CPU I/O. [Operator, recurrence and provenance audit](results/camera_step_v16.json).

The production module verifies all 69 original CPU parameter tensors against the capture/export metadata, installs the shared NPU head before packing remaining CPU Linear layers, and aborts on unsupported inputs or SDK errors without a CPU fallback. On three identical TUM input pairs, the integrated head takes 0.191–0.200 s, while the **whole forward still takes 4.15–4.90 s** with four CPU threads. Depth relative L2 differences from original FP32 at the same resolution are 1.58%, 1.81% and 6.46%; pose-encoding differences are 1.04%, 1.38% and 6.20%. These tensor errors are not trajectory errors. [Fixed-input whole-model results](results/hybrid_pairs_v16.json).

### Resolution accuracy remains a blocker

The first three exact r7 mapping pairs were also run through the original CPU FP32 model at 112 and 518, using the same source images, rectification and weights. Actual tensors are 84×112 and 392×518. Rotation error against TUM ground truth is **5.33–5.85° at 112 versus 0.95–1.09° at 518**; translation-direction error is 9.59–94.21° versus 0.61–2.39°. Even after one positive depth scale per two-frame pair, depth differences remain 18.73–20.38% relative L2. This is a three-pair diagnostic, not a full trajectory or depth-ground-truth evaluation, but it shows that 112 cannot currently support the planned accuracy claim. Original 518 forward times were 142–193 s per pair. [Exact paired-resolution evidence](results/resolution_comparison_v16.json).

No loop closure, 30-minute endurance or 8 GB acceptance is established. Mapper peak RSS in these hybrid runs is about 7.0 GiB, with SDK DMA reported separately; RSS and DMA must not simply be added as if disjoint. Sparse Linux memory/temperature samples are observations, not a simultaneous peak-memory or thermal qualification.

## Lightweight tests

With NumPy, OpenCV and pytest installed, the geometry, queue and calibration tests need no weights or hardware:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_continuous_tracking.py tests/test_fast_tracking.py \
  tests/test_tracking_geometry.py tests/test_edge_runtime.py \
  tests/test_edge_trajectory_evaluation.py tests/test_image_rectification.py \
  tests/test_rectified_mapping.py
```

Torch enables the additional synthetic RKNN lifecycle/contract tests. The public snapshot's complete focused test command and result are recorded in [the test receipt](results/synthetic_tests_v16.json). Synthetic SDK calls never establish hardware execution. No host real-image tracking or model inference is used in these checks.

## Setup and external dependencies

The validated deployment uses Linux aarch64, glibc 2.35, Python 3.11.17, PyTorch 2.3.1 CPU, torchvision 0.18.1 and a GTSAM develop build containing SL4. A pinned download manifest and an offline installer are under `deploy/rk3588/dependencies/`. Their downloaded wheels, Python runtime and Debian packages are not committed.

Clone the pinned VGGT implementation and apply the portable patch from the repository root:

```sh
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt
git -C third_party/vggt checkout 6e6e16107b88e8e76c751826af10d4295d87ecd2
git -C third_party/vggt apply ../../deploy/rk3588/vggt-spark.patch
```

Use the official [facebook/VGGT-1B checkpoint](https://huggingface.co/facebook/VGGT-1B). The reference `model.pt` is 5,026,874,952 bytes with SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. `deploy/rk3588/download_weights.py` verifies that checkpoint. Weights are not redistributed here. A similarly shaped StreamVGGT checkpoint is not an interchangeable baseline.

Loop closure is optional and is disabled in the minimal replay command. Enabling it also requires local pinned [SALAD](https://github.com/serizba/salad) and [DINOv2](https://github.com/facebookresearch/dinov2) sources and the local SALAD checkpoint; see `UPSTREAMS.json`. The runtime does not silently download these when loop closure is disabled.

The SDK, `librknnrt.so`, RKNN compiler wheel and Rockchip headers are not included. Obtain the licensed [official RKNN SDK](https://github.com/airockchip/rknn-toolkit2) yourself. The tested runtime/ABI is 2.3.2; users must provide matching `rknn_api.h` and `rknn_matmul_api.h` under the filenames expected by each diagnostic. Header hashes are checked before hardware calls. XNNPACK ABI tests also need the matching external XNNPACK/pthreadpool headers. A fresh public clone therefore does not contain all hardware-test prerequisites.

## Timestamped RGB replay

Obtain the official [TUM RGB-D freiburg1/xyz sequence](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and use its original `rgb.txt`. It contains 798 RGB images over 26.572059 seconds, about 29.994 Hz. Replay uses original intervals; it is software-simulated capture, not a physical camera test. Sensor depth and ground truth are not inputs to the tracker.

Run on the RK3588S from the repository root, in the prepared environment. Use an output directory that does not exist:

```sh
PYTHONPATH=.:third_party/vggt python -u scripts/run_edge.py \
  --rgb-index /path/to/rgbd_dataset_freiburg1_xyz/rgb.txt \
  --checkpoint /path/to/model.pt --output /path/to/new-run \
  --image-size 112 --backend cpu --threads 3 \
  --mapper-cpus 4,5,6 --frontend-cpus 7 --capture-cpus 0,1 \
  --candidate-cpus 2,3 --map-period 3 --bootstrap-gap 0.25 \
  --history-frames 300 --history-seconds 10 \
  --tracking-width 320 --tracking-features 400 --replay-stride 3 --max-loops 0
```

`--backend xnnpack-f16` explicitly selects an experimental CPU backend, not NPU. Queue capacities, original/delivered frame counts, replacements, fresh valid poses, map publications and deferred depth adoption are recorded. Historical catch-up and EOF-only adoption do not count as online new poses. Local tracking poses use model units; the SL4 transform is projective and does not automatically define a metric global SE3 pose. Map capacity and a memory guard are explicit, not proof of endurance.

The measured XNNPACK diagnostics explicitly used `--mapper-memory-limit-mib 8500` on the 16 GB board; the default 6500 MiB guard is too small for their observed mapper RSS. This override does not demonstrate 8 GB operation. The r5 frontend-only diagnostic additionally uses `--local-calibration deploy/rk3588/tum-fr1-local-calibration.json`; it must not be described as a globally calibrated map.

For the shared-calibration runner, use `--rectify-calibration deploy/rk3588/tum-fr1-local-calibration.json`. The r11 allocation is `--threads 2 --mapper-cpus 4,5 --capture-cpus 6 --frontend-cpus 7 --candidate-cpus 2,3 --tracking-features 240`. Its CPU Linear backend is `--backend xnnpack-f16`, with the separate experimental NPU head selected by `--camera-backend rknn-step`. The NPU head requires all four artifact paths and their explicit SHA256 values, as shown in [the v16 reproduction notes](results/camera_step_v16.md); it currently accepts two-frame batches with `--max-loops 0`. A hash, shape or SDK failure aborts the selected backend.

The trajectory evaluator reads saved poses after the experiment; ground truth never enters tracking or model inference:

```sh
python scripts/evaluate_edge_trajectory.py \
  --run-dir /path/to/new-run \
  --groundtruth /path/to/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  --output /path/to/new-diagnostic.json
```

It associates actual timestamps within 20 ms without interpolation and computes anchor-relative errors and one-second RPE within the same local gauge. Translation direction excludes ground-truth movements below 1 cm. It does not stitch projective gauges into a metric ATE trajectory or fit ground-truth scale by default.

## Reproducing NPU subnetwork experiments

`deploy/rk3588/validation/` contains board-only capture/runtime programs and host-only ONNX export/compiler programs. The DINO pipeline is `capture-v12-dino-block-board.py` → `export-v12-dino-block.py` → `export-v12-dino-block-gelu.py` → `compile-v12-rknn-block.py` → `run-v12-rknn-block-board.py`. The small-input versions are `capture-v14-small-dino-board.py`, `export-v14-small-dino.py` and `run-v14-small-rknn-board.py`. Run `--help` for the exact required input, capture and SHA256 arguments.

For the complete CameraHead experiment, use `capture-v15-camera-board.py`, `export-v15-camera-board.py`, `check-v15-camera-onnx-board.py`, `compile-v15-camera.py` and `run-v15-camera-board.py`. Here capture, ONNX export with tracing, numerical checks and actual inference all run on the board; only RKNN compilation runs on the host. The exported ONNX shares 69 original weight tensors, but the compiled model is 1.711 GB. Do not infer its runtime memory use from ONNX weight sharing.

Captures use the official checkpoint and real image. Export preserves original block weights and exact GELU; CPU PyTorch references are produced on the board. Execute normal timing separately from `--collect-perf-detail`, then inspect actual SDK operator targets. Compilation success or a core-mask request does not prove NPU execution. Numerical error and complete input/run/output-copy/release time must accompany every hardware result. No model, ONNX, RKNN, captured tensor or SDK binary is committed.

The frozen capture drivers also pin their original image and source hashes. For their exact experiment, obtain `office_loop.zip` from the upstream repository separately. Its numbered JPEG files have no original acquisition timestamps; they cannot establish camera input rate. The capture's fixed board layout can be recreated outside this repository at `/mnt/vggt-data/vggt-slam2`.

## Attribution and licenses

The original project is MIT-SPARK's VGGT-SLAM 2.0, based here on commit `35327ac28b7d193df9ccc39ba6346052bb6f1207`. Its BSD-2-Clause license and copyright remain in `LICENSE`; its original README is in `README_UPSTREAM.md`.

Third-party code and model weights keep their own licenses. VGGT_SPARK is CC-BY-NC-4.0, SALAD is GPL-3.0 and DINOv2 is Apache-2.0. The root BSD license does not relicense them or Rockchip's SDK. Dependency sources, commits and license locations are recorded in `UPSTREAMS.json` and `THIRD_PARTY_NOTICES.md`.
