# Edge-VGGT-SLAM

Ports **VGGT-SLAM2's original pretrained VGGT** to RK3588S for camera poses and dense maps. Model execution is separated from the original submap construction and SL4 graph optimization so embedded backends can be implemented and validated.

Only the original VGGT checkpoint is supported. Its internal DINO, frame/global attention, CameraHead and DPT are retained, with no additional model or traditional pose frontend. All inference, calibration and model validation run on the RK3588S board.

## Current status

The original CPU reference path and an experimental complete RKNN path have run on real TUM data with timestamp-paced replay. The current experimental configuration requests W8A8 for the aggregator and CameraHead, with FP16 DINO and DPT. Original preprocessing, pose decoding, submaps and SL4 remain in place.

The latest complete replay used a **16 GB RK3588S**, a **1 GHz NPU** and two-frame `84×112` inputs. It was a short recorded-dataset experiment:

| Metric | Measured |
| --- | ---: |
| Dataset frames / fresh pose updates | 798 / 31 |
| Average fresh pose rate | **1.1167 Hz** |
| Complete input-to-pose P95 | **1.0688 s** |
| Replay duration, including EOF drain | 27.7595 s |
| Startup, measured separately | 57.3481 s |
| Peak process RSS | 3380.19 MiB |

**The real-time targets remain unmet: at least 20 Hz fresh poses and complete input-to-pose P95 at most 100 ms.** One of 30 publication gaps exceeded one second; the maximum was 1.0414 s. An average rate does not guarantee a fresh update every second.

Quality remains experimental. On a fixed 17-frame sequence with Sim3 alignment, ATE and RPE RMSE changes relative to the original CPU reference were within 10%, but rotation RPE RMSE increased 7.30% and the worst rotation interval reached 27.24°. Strict FP32 tensor comparisons failed. These results establish neither broad trajectory/map quality nor quality for a separate replay that selects different frames.

Physical-camera tests are deferred. Thirty-minute stability and operation on physical 8 GB hardware remain unvalidated. No loop-closure implementation is validated; the Solver supports only `max_loops=0`.

[Measurements, quality limits and reproduction](deploy/rk3588/validation/original_calibrated_camera/README.md) · [Machine-readable results](results/original_calibrated_camera_v82.json)

## Prepare the board

The tested environment is Linux aarch64, Python 3.11.17, CPU PyTorch 2.3.1, torchvision 0.18.1 and a GTSAM build with SL4 support. The NPU path uses RKNN SDK/runtime 2.3.2.

Follow the [deployment guide](deploy/rk3588/README.md) to prepare the original source and [external dependencies](deploy/rk3588/dependencies/README.md). Commands below use the validated board directory `/mnt/vggt-data/vggt-slam2`, with this repository under `src/`. Prepare the required `python/`, `wheelhouse/` and other files, then run the offline installer:

```sh
sh /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/install-offline.sh
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
```

Download the original weights separately; they are not distributed in this repository:

```sh
python /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/download_weights.py \
  --output /mnt/vggt-data/vggt-slam2/weights/model.pt
```

The only supported checkpoint is the original `facebook/VGGT-1B/model.pt`, **5,026,874,952 bytes**. The downloader and loader require its pinned full SHA256 before loading; see [source, checkpoint versions and hashes](UPSTREAMS.json). Generate ONNX/RKNN artifacts in an authorized environment using the relevant reproduction directory.

## Run timestamp replay

Prepare the TUM dataset's `rgb.txt` and images, then run the original CPU FP32 reference path on the board:

```sh
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
python -B -u /mnt/vggt-data/vggt-slam2/src/scripts/replay_original_vggt.py \
  --rgb-index /mnt/vggt-data/vggt-slam2/datasets/rgbd_dataset_freiburg1_xyz/rgb.txt \
  --checkpoint /mnt/vggt-data/vggt-slam2/weights/model.pt \
  --output /mnt/vggt-data/vggt-slam2/experiments/original-replay-new-run
```

Use a new output directory. Image width defaults to 112; this dataset produces height 84. The decoder uses CPU0 and the model defaults to CPU4–7, so keep CPU0 available to the process. See `python scripts/replay_original_vggt.py --help` for all options.

Outputs include `poses.tum`, `map.json`, `map_chunks/` and `results.json`. Each fresh pose and map is published after the new observation's model computation and original SL4 processing. The input queue has capacity 1.

The NPU experiment above requires pinned compiled artifacts and experimental wrappers; follow the [current complete NPU reproduction guide](deploy/rk3588/validation/original_calibrated_camera/README.md). The regular replay command defaults to the CPU reference. Authorized frequency tests use a bounded thermal controller and restore the prior clock state; the replay entrypoint does not change clocks or governors.

## Development and evidence

- [Architecture](ARCHITECTURE.md): original prediction contract, caching and SL4 boundaries.
- [Real-time feasibility review](docs/REALTIME_FEASIBILITY_REVIEW.md): measured bottlenecks, validation gaps and decision gates for the 20 Hz / 100 ms target.
- [Project constraints](AGENTS.md): original-model-only scope, board execution and acceptance targets.
- [Measurement index](MEASUREMENTS.json) and [results](results/): complete runs, component experiments and failure records.
- [Engineering notes](docs/ENGINEERING_NOTES.md): useful findings, supporting evidence and remaining validation conditions.

With development dependencies installed, run the existing lightweight regression checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_runtime_loading.py tests/test_model_execution.py \
  tests/test_solver_execution.py tests/test_patch_cache.py
```

These checks use generated data or fake services. Board inference, trajectory quality and real-time acceptance require separate validation. Component timings and end-to-end replay results are reported separately.

## Licensing and attribution

Based on MIT-SPARK's [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) and the pinned [VGGT_SPARK](https://github.com/MIT-SPARK/VGGT_SPARK) revision. MIT-SPARK copyright notices and the repository's [BSD-2-Clause license](LICENSE) are retained.

**The repository code license does not grant rights to the model weights.** The original VGGT weights and pinned implementation have separate noncommercial terms, which also apply to weights embedded in compiled artifacts. See [model provenance and licenses](MODEL_LICENSES.md), [third-party notices](THIRD_PARTY_NOTICES.md) and [upstream revisions](UPSTREAMS.json). This repository distributes no weights, captured arrays, weight-bearing exports or SDK binaries.
