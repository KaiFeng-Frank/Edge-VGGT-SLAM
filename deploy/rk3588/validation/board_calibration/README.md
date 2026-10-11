# Fixed board calibration data (v36)

Actual fixed calibration data was generated on RK3588S; the saved sampling audit passed 1,787 checks. [Generation, audit and subsequent PTQ receipts](../../../../results/board_ptq_v36.json). The root operator executes actual image processing on the RK3588 board; this helper does not run a neural encoder, Update, RKNN build, GT reader or accuracy evaluation. Helpers are MIT, with the original DPVO MIT attribution retained through the frozen v30 encoder helper.

The predeclared policy uses zero-based `rgb.txt` source indices `256+16*i` for `i=0..31`, ending at 752. The exact official TUM fr1_xyz index contains 798 rows and is pinned by SHA256 `532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3`. Calibration frame IDs are disjoint from evaluation indices 0–191. They remain from the **same sequence/environment**; this is not independent-scene validation. Nothing is selected using GT, pixels, confidence or model outputs. Missing/invalid data fails instead of substituting another frame.

Preprocessing follows the original DPVO TUM reader: OpenCV BGR uint8 640×480 → `cv2.undistort` using original K/D → crop `[8:-8,16:-16]` to 464×608 → the frozen `normalize_bgr` once, yielding FP32 NCHW `[1,3,464,608]`. There is no resize or RGB reorder. Cropped K has principal point `(302.6,247.3)`.

A persistent Torch **CPU** Generator with predeclared seed 0 samples only these 32 selected frames. Each selected frame consumes original x randint draws, y randint draws, then 96 FP32 random inverse-depth values. Depth values are saved for consumption auditing and are not model/calibration inputs. This does not recreate the RNG state of an online run through all preceding/omitted frames, nor claim CUDA byte equivalence. Indices remain INT64 `[96]`, `y*152+x`, with original order and repetitions.

Outputs are `image-XX.npy`, `indices-XX.npy`, `dataset.txt` in graph order, a `sampling.npz` containing centers/depth and before/after RNG states, `policy.json`, and `calibration.json` with all file/raw-array hashes and original source identities. Expected raw image+index payload is 108,355,584 bytes; metadata/RNG/file headers add a small amount. The total file budget including the receipt is 256 MiB. The output directory must be new.

```bash
CAL_PROJECT=/mnt/vggt-data/vggt-slam2
CAL_SOURCE="$CAL_PROJECT/experiments/vggt-v36-calibration-source"
CAL_ENV="$CAL_PROJECT/venv"
CAL_POLICY_SHA="${CAL_POLICY_SHA:?Set the frozen calibration-policy.json SHA256}"
"$CAL_ENV/bin/python" -B "$CAL_SOURCE/generate-v36-dpvo-calibration-board.py" \
  --dataset-dir "$CAL_PROJECT/datasets/rgbd_dataset_freiburg1_xyz" \
  --rgb-index "$CAL_PROJECT/datasets/rgbd_dataset_freiburg1_xyz/rgb.txt" \
  --encoder-helper "$CAL_PROJECT/experiments/vggt-v30-dpvo-stream-r2-source/v30_frame_encoder.py" \
  --policy "$CAL_SOURCE/calibration-policy.json" --policy-sha256 "$CAL_POLICY_SHA" \
  --output-dir "$CAL_PROJECT/experiments/vggt-v36-calibration-data-repro"
```

Point `--encoder-helper` to the board's existing frozen file; its required SHA256 is `fc6fc662613a720cfb779a360a63906a593a96ff08addecb9d01cfd8767ca7a8`. No ORT session is created by importing this helper. Check `calibration.json` reports `generated_fixed_calibration_data_not_quantized_or_validated`, sources unchanged and all 32 rows before supplying `dataset.txt` to the separately guarded board PTQ step. This output is calibration input, not neural/trajectory acceptance.

Host tests use synthetic pixel buffers, fake undistortion, small CPU random draws and source-path guards. They do not process any real image or perform model inference.
