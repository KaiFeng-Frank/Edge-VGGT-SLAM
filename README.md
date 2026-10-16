# Edge-VGGT-SLAM

An RK3588S port of **VGGT-SLAM2 with its original pretrained VGGT**, producing camera poses and dense maps from image sequences.
Model execution is decoupled from the original SL4 backend to support embedded CPU/NPU runtimes.

## Features

- **Original model only:** native 518-width preprocessing, all original layers and tokens, including VGGT's internal DINO, CameraHead and DPT. No auxiliary learned model or traditional pose frontend is added.
- **Original SLAM backend:** retains the pose codec, submap construction and SL4 graph optimization. Loop closure is disabled (`max_loops=0`).
- **Dataset replay:** exports camera trajectories, dense map chunks and run metadata from timestamped image sequences.

## Status

The experimental native518 route completes fresh-image predictions on a **16 GB RK3588S**, with complete prediction calls taking **tens of seconds**.
Real-time performance, original-FP32 numerical agreement and trajectory-quality targets remain unmet.
The targets are at least **20 Hz fresh poses** and complete input-to-pose **P95 ≤ 100 ms**, retaining original resolution and all layers.
Physical-camera input, 30-minute stability and physical 8 GB operation remain unvalidated.

## Prepare the board

Start with the [RK3588 deployment guide](deploy/rk3588/README.md), including the pinned upstream source and [external dependencies](deploy/rk3588/dependencies/README.md). The tested layout places this repository at `/mnt/vggt-data/vggt-slam2/src`. Prepare the required offline packages before running:

```sh
sh /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/install-offline.sh
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
```

Obtain the original checkpoint separately:

```sh
python "$VGGT_DEPLOY_ROOT/src/deploy/rk3588/download_weights.py" \
  --output "$VGGT_DEPLOY_ROOT/weights/model.pt"
```

Only the pinned original `facebook/VGGT-1B/model.pt` is supported. The downloader and loader verify its identity before loading; [UPSTREAMS.json](UPSTREAMS.json) records the source revisions and checkpoint SHA256.
NPU experiments require separately prepared, pinned compiled artifacts and the matching external RKNN runtime, as described in their reproduction guides.

## CPU reference replay

Run the **original FP32 CPU reference on the board** with a TUM dataset (`rgb.txt` and images).
Source the environment above and choose a new output directory:

```sh
python -B -u "$VGGT_DEPLOY_ROOT/src/scripts/replay_original_vggt.py" \
  --rgb-index "$VGGT_DEPLOY_ROOT/datasets/rgbd_dataset_freiburg1_xyz/rgb.txt" \
  --checkpoint "$VGGT_DEPLOY_ROOT/weights/model.pt" \
  --image-size 518 \
  --output "$VGGT_DEPLOY_ROOT/experiments/original-replay-new-run"
```

Original crop preprocessing converts these `640×480` images to `392×518`. Outputs include `poses.tum`, `map.json`, `map_chunks/` and `results.json`; fresh predictions pass through the original SL4 processing. Keep CPU0 available for decoding. See the replay script's `--help` for supported options.

## Documentation

- [Architecture](ARCHITECTURE.md): prediction contract, execution boundaries and SL4 integration.
- [Native518 validation](deploy/rk3588/validation/original_native518_trajectory/README.md): offline board predictions and matched trajectory evaluation.
- [Measurements](MEASUREMENTS.json) and [results](results/): scoped numerical and performance evidence.
- [Engineering notes](docs/ENGINEERING_NOTES.md): implementation findings, reproduction conditions and known limits.
- [Project constraints](AGENTS.md): original-model scope and validation requirements.

## Licensing and attribution

Based on MIT-SPARK's [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) and pinned [VGGT_SPARK](https://github.com/MIT-SPARK/VGGT_SPARK). Existing attribution and the repository's [BSD-2-Clause license](LICENSE) are retained.

**The code license does not license the model weights.**
Original VGGT weights and the pinned model implementation have separate noncommercial terms; compiled artifacts containing weights retain those terms. See [model provenance and licenses](MODEL_LICENSES.md) and [third-party notices](THIRD_PARTY_NOTICES.md). This repository distributes no weights, captured tensors, weight-bearing exports or SDK binaries.
