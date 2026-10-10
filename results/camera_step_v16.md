# Shared original CameraHead on RKNN

The v16 graph executes one original refinement iteration. The caller reuses one context four times, using the learned empty pose only as the first condition. The first prediction is the first delta; later deltas accumulate in CPU FP32 and the unactivated prediction conditions the next step. All four original trunk blocks execute in each step. Translation/quaternion remain linear and FOV uses the original ReLU.

Reuse the verified original `capture-v15-camera-board.py` capture. The v16 pipeline is:

1. On the RK3588S, run `export-v16-camera-step-board.py` to export the single-step graph and verify all four CPU iterations against the original head.
2. On the board, run `check-v16-camera-step-onnx-board.py` for original and perturbed-input ONNX Runtime closed-loop comparison.
3. On the host, run `compile-v16-camera-step.py` with RKNN 2.3.2. Host compilation executes no model inference.
4. On the board, run `run-v16-camera-step-board.py` once normally and separately with `--collect-perf-detail`. Obtain timing, output comparisons, operator targets and DMA from actual SDK calls.

Each tool exposes explicit paths and SHA256 arguments via `--help`. The frozen diagnostic layout is `/mnt/vggt-data/vggt-slam2`, with this source checkout at `src`. The matching official SDK header, runtime and compiler must be supplied separately. Production additionally verifies the tested `librknnrt.so` SHA256 `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`; SDK version text alone is insufficient.

For the measured artifacts, add these arguments to the documented `run_edge.py` command, using paths relative to that board deployment directory:

```sh
--camera-backend rknn-step \
--camera-model experiments/vggt-v16-camera-step/camera-step-fp16.rknn \
--camera-model-sha256 0821683fe1f9526c52777b0ef1ba45ba4c744375f9fd731f0f3d2068e4355af9 \
--camera-export-receipt experiments/vggt-v16-camera-step/export112-r1/export.json \
--camera-export-receipt-sha256 8ccf7ff593346962766b4eae6da7eb95ce43eeaa3183a8fc0e82bf134c172503 \
--camera-compile-receipt experiments/vggt-v16-camera-step/compile.json \
--camera-compile-receipt-sha256 c8771d743d9840bcd32b22c6894f791ed239b4ad3795db703e7f7122329fbc7a \
--camera-capture-receipt experiments/vggt-v15-camera/capture112-r1/capture.json \
--camera-capture-receipt-sha256 1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1
```

The explicit receipt hashes identify these measurements. Reproducing captures/compilation creates new receipts (which contain paths, timing and versions), so pass the SHA256 of the receipts actually generated and preserve their valid provenance chain. The model currently requires two frames and four iterations, with loop candidates disabled. Runtime failure does not silently switch to the CPU head.

`capture-v16-hybrid-pairs-board.py` pins the three original CPU112 pairs and checks each actual preprocessed input tensor byte-for-byte before integrated inference. `capture-v16-resolution-pairs-board.py` and `compare-v16-resolution-pairs.py` establish the separate 112-versus-518 diagnostic. These fixed-input timings exclude online queues and delayed initialization.

The graph's lower DMA allocation is verified; accurate real-time SLAM is not. The full model still uses CPU computation, and the low-resolution quality result fails to support the intended accuracy claim. No weights, datasets, captured tensors, compiled models or SDK files are redistributed here.
