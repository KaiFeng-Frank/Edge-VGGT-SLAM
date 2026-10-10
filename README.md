# VGGT-SLAM 2 on RK3588S

An experimental RK3588S port of [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM), retaining the original VGGT-1B weights and SL4 mapping backend. The current frontend uses LK, PnP and triangulation; it does not use ORB. All measured model inference runs on the board. Host tools export and compile graphs or inspect saved arrays.

**Status: 20 Hz valid tracking, a 3-second map-update budget, 30-minute endurance, physical-camera latency and 8 GB operation have not passed validation.** The continuous RGB replay is an experimental diagnostic, not a completed real-time SLAM system. CPU/XNNPACK execution is identified as CPU; the NPU results below describe one subnetwork, not the complete VGGT network.

## Measured baseline

On the tested 16 GB RK3588S, original CPU FP32 VGGT with four A76 threads and two office images took 3.831 seconds at a requested image size of 112 (actual input 56×112). Its camera head took about 1.250 seconds in the separate hooked run. Size 84 took 3.161 seconds for the full forward; its camera head remained about 1.268 seconds. These are short measurements, not sustained throughput or latency percentiles; they exclude image loading, graph optimization and publication.

An original-weight DINO block with input `[1,37,1024]` executed using RKNN 2.3.2: its last three complete I/O calls averaged 5.489 ms. Four separate SDK profile calls reported all 14 internal operators on NPU and only input/output operators on CPU. Its relative L2 error against PyTorch FP32 was 1.2253%, so full-network pose/depth accuracy remains unvalidated. A CPU-Softmax/Erf variant is a mixed implementation and is not the all-NPU result.

[Measurement evidence](MEASUREMENTS.json) records receipt hashes and the measured operator targets.

The full model and the continuous tracker still require further work. Dataset timestamps, startup, lost frames, latest-slot replacement and EOF work must remain visible in the results; high input rate alone is not a valid-pose rate.

## Continuous replay results

The 26.572-second TUM RGB sequence has now run end to end on the board. These are complete-run results, including startup and losses:

| Run | New valid poses / 798 RGB frames | First valid pose | Successful-pose latency P95 | Map update time |
| --- | ---: | ---: | ---: | ---: |
| CPU FP32, 480-wide/800 features | 0 / 798 | unavailable | unavailable | 7.71–8.03 s |
| CPU FP32, 320-wide/400 features | 251 / 798 | 17.34 s | 90.19 ms | 7.23–7.88 s |
| XNNPACK FP16 CPU, 320-wide/400 features | 397 / 798 | 5.49 s | 90.55 ms | 3.85–5.93 s |

The latter two runs replay every third retained historical observation when depth arrives; online observations still run PnP individually. This is not an equivalent-numerics optimization. The XNNPACK run has a 6.27-second gap in valid pose publication and a mapper peak RSS of 7155 MiB. Good latency among successful poses does not establish continuous tracking or 8 GB operation. No loop closure was enabled in these runs. [Machine-readable results](results/tum_fr1_xyz_v15.json) include coverage, per-second valid-pose counts and failures. Physical camera latency, ground-truth accuracy and 30-minute endurance remain unvalidated.

## Lightweight tests

With NumPy, OpenCV and pytest installed, run the synthetic geometry, queue and timing checks without loading images, weights or hardware:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_continuous_tracking.py tests/test_fast_tracking.py \
  tests/test_tracking_geometry.py tests/test_edge_runtime.py
```

Other tests include optional SDK/header or native Torch prerequisites, described below.

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

## Reproducing NPU subnetwork experiments

`deploy/rk3588/validation/` contains board-only capture/runtime programs and host-only ONNX export/compiler programs. The DINO pipeline is `capture-v12-dino-block-board.py` → `export-v12-dino-block.py` → `export-v12-dino-block-gelu.py` → `compile-v12-rknn-block.py` → `run-v12-rknn-block-board.py`. The small-input versions are `capture-v14-small-dino-board.py`, `export-v14-small-dino.py` and `run-v14-small-rknn-board.py`. Run `--help` for the exact required input, capture and SHA256 arguments.

Captures use the official checkpoint and real image. Export preserves original block weights and exact GELU; CPU PyTorch references are produced on the board. Execute normal timing separately from `--collect-perf-detail`, then inspect actual SDK operator targets. Compilation success or a core-mask request does not prove NPU execution. Numerical error and complete input/run/output-copy/release time must accompany every hardware result. No model, ONNX, RKNN, captured tensor or SDK binary is committed.

The frozen capture drivers also pin their original image and source hashes. For their exact experiment, obtain `office_loop.zip` from the upstream repository separately. Its numbered JPEG files have no original acquisition timestamps; they cannot establish camera input rate. The capture's fixed board layout can be recreated outside this repository at `/mnt/vggt-data/vggt-slam2`.

## Attribution and licenses

The original project is MIT-SPARK's VGGT-SLAM 2.0, based here on commit `35327ac28b7d193df9ccc39ba6346052bb6f1207`. Its BSD-2-Clause license and copyright remain in `LICENSE`; its original README is in `README_UPSTREAM.md`.

Third-party code and model weights keep their own licenses. VGGT_SPARK is CC-BY-NC-4.0, SALAD is GPL-3.0 and DINOv2 is Apache-2.0. The root BSD license does not relicense them or Rockchip's SDK. Dependency sources, commits and license locations are recorded in `UPSTREAMS.json` and `THIRD_PARTY_NOTICES.md`.
