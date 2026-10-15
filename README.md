# Edge-VGGT-SLAM

Ports **VGGT-SLAM2's original pretrained VGGT** to RK3588S for camera poses and dense maps. Model execution is separated from the original submap construction and SL4 graph optimization so embedded backends can be implemented and validated.

Only the original VGGT checkpoint is supported. Its internal DINO, frame/global attention, CameraHead and DPT are retained, with no additional model or traditional pose frontend. Deployment inference, calibration and performance measurements run on the RK3588S board. Matched original FP32 references may run on the host CPU; the host GPU is unused.

## Current status

The experimental complete native518 route now runs from fresh PNGs to predictions on a **16 GB RK3588S**, preserving two `392×518` frames, all 24 internal DINO layers per image, all 48 aggregator blocks, four CameraHead refinements, both DPT frames and the original pose codec. In the latest two-pair check, complete PNG-to-prediction calls take **32.84–36.80 s**, with startup and terminal verification measured separately. Keeping two original groups resident preserves all current control output bytes, but **both routes fail all ten same-input original-FP32 tensor gates**. These two pairs establish neither a trajectory result nor real-time operation. See [R205](results/original_native518_fresh_pixel_persistent_v205.json) and the [engineering evidence](docs/ENGINEERING_NOTES.md#270-fresh-pngs-complete-the-original-pipeline-with-two-persistent-groups).

**Native-size trajectory quality has been measured and fails the current quality goal.** The matched 17-observation comparison used the complete original CPU FP32 model and unchanged original SL4 on identical frames. With Sim3 alignment, translation ATE RMSE increased **24.4%** and translation RPE RMSE increased **41.6%**; rotation RPE RMSE decreased 9.7%. These increases exceed the 20% limit on this short clip. Map comparisons measure agreement with the CPU reference, without establishing ground-truth map accuracy. See [quality and reproduction](deploy/rk3588/validation/original_native518_trajectory/README.md) and the [v111 results](results/original_native518_trajectory_quality_v111.json).

Operator, precision, file-IO and execution experiments, including rejected routes, are recorded in the [engineering notes](docs/ENGINEERING_NOTES.md). The [native execution and cache guide](deploy/rk3588/validation/original_native518_tail_reuse/README.md) retains the earlier reuse-based measurements; they are not matched speed comparisons with the fresh-pair result above.

**The real-time targets remain unmet: at least 20 Hz fresh poses and complete input-to-pose P95 at most 100 ms.** Acceptance requires original 518-width preprocessing and all original layers and tokens. Physical-camera tests are deferred; 2–5 s map cadence, thirty-minute stability and operation on physical 8 GB hardware remain unvalidated. The Solver supports only `max_loops=0`.

Historical v82 timestamp-paced replay achieved 1.1167 Hz with input-to-pose P95 1.0688 s using two-frame `84×112` inputs. Its reduced-resolution performance, memory and quality results do not establish native518 acceptance. The [historical reproduction guide](deploy/rk3588/validation/original_calibrated_camera/README.md), [v82 results](results/original_calibrated_camera_v82.json) and [first native integration record](results/original_native518_integration_v100.json) remain available.

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
  --image-size 518 \
  --output /mnt/vggt-data/vggt-slam2/experiments/original-replay-new-run
```

Use a new output directory. The CPU replay defaults to the original preprocessing width of 518. These `640×480` TUM images become `392×518` (height × width) in original crop mode. Explicit `--image-size 112` is reserved for reproducing historical reduced-resolution diagnostics; it does not satisfy original-size acceptance. The decoder uses CPU0 and the model defaults to CPU4–7, so keep CPU0 available to the process. See `python scripts/replay_original_vggt.py --help` for all options.

Outputs include `poses.tum`, `map.json`, `map_chunks/` and `results.json`. Each fresh pose and map is published after the new observation's model computation and original SL4 processing. The input queue has capacity 1.

The historical `112` NPU timestamp-replay wrappers require that size explicitly and pinned compiled artifacts; follow their [reproduction guide](deploy/rk3588/validation/original_calibrated_camera/README.md). The separate [native518 harness](deploy/rk3588/validation/original_native518_trajectory/README.md) supplies complete offline board predictions and matched saved-prediction SL4 quality evaluation. It does not establish a real-time timestamp-paced native518 service. The regular command above defaults to the original-size CPU reference. Authorized frequency tests use a bounded thermal controller and restore the prior clock state; the replay entrypoint does not change clocks or governors.

## Development and evidence

- [Architecture](ARCHITECTURE.md): original prediction contract, caching and SL4 boundaries.
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
