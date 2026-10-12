# Original VGGT DPT W8A8 experiment

The W8A8 candidate was rejected: on two real captured frames, the complete depth-head component was **19.37% slower** than the current FP16 graph. The current complete pipeline retains FP16 DPT. This experiment did not run a new complete model, trajectory or timestamp replay.

At the same bounded 1 GHz NPU setting, warm means were **26.458 ms FP16** and **31.584 ms W8A8**, with 12 measured calls per route after excluding its first call for each frame. Timing includes input preparation, synchronous FP32 SDK I/O, owned output copies, release, the original CPU activation and final finite checks. Both routes repeated byte for byte; all 28 calls and output buffers completed, and both contexts closed.

Against the original CPU FP32 reference, W8A8 depth relative L2 errors were 1.6554% / 1.7392%, and confidence errors were 12.8565% / 9.0568% for the two frames. Both routes fail the fixed FP32 tolerance (`rtol=1e-4`, `atol=1e-5`); these errors do not establish trajectory or map quality. The slower candidate was not integrated into the complete model.

The original ONNX graph is unchanged: four retained inputs from layers 4/11/17/23, all 62 learned parameters, five positional constants, spatial operators, one-frame chunking and CPU `activate_head` remain. Eight disjoint real TUM calibration pairs produce 16 rows with four FP32 inputs each. Captured depth/confidence hashes match the original full-model capture; evaluation images are excluded.

The first capture was stopped at a sampled 75.769°C. The same frozen capture source completed on retry using process pause/resume thresholds of 65°C / 55°C, without clock or governor changes. The subsequent RKNN Toolkit2 2.3.2 normal/channel W8A8 build completed in 21.326 s (controller wall time 22.142 s), with clean source/parameter lineage and API release. Its artifact is 35,366,319 bytes, versus 68,958,190 bytes for FP16; file size is not a speed measurement.

The compiler table reports 53 NPU rows and 10 CPU rows, including five Resize operations assigned to CPU and five I/O boundaries. Operator data-type entries say INT8, but norm constants include FLOAT16/FLOAT; this does not prove exclusively INT8 internal arithmetic. Five error-tagged `Unkown op target: 0` messages coexist with successful build/export returns. No exclusive causal explanation is asserted.

See the [result summary](../../../../results/original_calibrated_depth_v85.json), [source identities](source-copy-list.json) and the [compile](compile-audit.json), [capture](capture-audit.json) and [component](depth-probe-audit.json) audits. The audit independently checks saved provenance and all 96 calibration NPY identities without executing a model or SDK.

Reproduction requires the pinned original checkpoint and private calibration arrays/ONNX graph, which are not distributed. Actual capture, calibration and inference run only on RK3588S. Use the production environment for capture and the isolated RKNN compiler environment for build.

```sh
# Run under the existing bounded thermal controller after capture completes.
"$RKNN_PYTHON" compile_depth_int8_board.py \
  --capture-dir "$CAPTURE_DIR" --capture-sha256 "$CAPTURE_SHA256" \
  --capture-source "$CAPTURE_SOURCE" --capture-source-sha256 "$CAPTURE_SOURCE_SHA256" \
  --output-dir "$NEW_BUILD_DIR" \
  --controller-source "$THERMAL_CONTROLLER" --controller-source-sha256 "$CONTROLLER_SHA256"
```

The frozen launchers record the actual board paths and hashes. Stage the listed sources in their specified experiment directory and use new output directories for a repeat:

| Stage | Launcher | Controller |
| --- | --- | --- |
| Successful original capture | `run_depth_capture_r2.sh` | `run_capture_thermal_stage.py`, 180 s / 75°C hard limit |
| Original graph W8A8 compilation | `run_depth_compile.sh` | Existing [thermal controller](../original_gpu_attention/run_thermal_stage.py), no frequency writes |
| Component A/B | `run_depth_probe_1ghz.sh` | Existing [NPU controller](../original_clock_test/npu_frequency_test.py), authorized 1 GHz / 180 s / 75°C, restores prior state |

`run_depth_capture.sh` records the interrupted first attempt. The depth adapter's lineage and I/O helpers were exercised; installation into a complete model was not tested.

The script defaults to the standard board paths for the original depth manifest, passed FP32 check and calibration lineage. Explicit path options retain their fixed identity checks. Repository code and original model weights have separate licenses. No real-time, 20 Hz, 100 ms, long-duration or 8 GB acceptance is claimed.
