# Complete original VGGT aggregator calibration diagnostic

This experiment calibrates and compiles all 24 original frame/global pairs
with RKNN W8A8 PTQ. It uses unchanged original ONNX graphs and retains the
original parameter sources, tokens and layers. The remaining original DINO, CameraHead and DPT backends are
unchanged. This diagnostic is not enabled by the production replay command.

All original FP32 capture, SDK compilation and full inference ran on RK3588S.
Eight real TUM fr1_xyz calibration pairs use indices `(0,48)` through
`(672,720)` in steps of 96. Each pair has its own activation input at every
one of the 24 aggregator layers. The 17 evaluation image indices are disjoint
from calibration, but all belong to the same short sequence; this is not an
independent-sequence quality test. Calibration also retains original DINO
inputs privately for subsequent experiments.

The frozen scripts are the sources used for the measured run. They require
the recorded original v55 pair manifest and private original artifacts:

1. Run `capture_original_calibration_board.py --help` on the board and use a
   new output directory. Capture verifies the original checkpoint before load.
2. Run `compile_calibrated_pairs.py` in the isolated RKNN compiler environment
   with `--manifest PARENT/compiled.json --manifest-sha256 SHA
   --capture CAPTURE/capture.json --capture-sha256 SHA --output NEW_DIRECTORY`.
   Workers execute sequentially. All 24 graphs are required by the runtime.
3. Keep `rknn_calibrated_pairs.py` beside the complete comparison and replay
   scripts. Run `compare_calibrated_complete.py` with the candidate manifest,
   `--paired-aggregator --model-threads 1`, the original DINO/depth manifests
   and the recorded CPU reference. `replay_calibrated_original.py` accepts
   the same replay options as the existing paired path. Use fresh output
   directories and preserve failed results.

The runtime pins the recorded capture and parent manifest. A new capture has a
new report hash, even for identical images; review its original-source, input
and completion checks and explicitly bind that new identity before using the
adapter. The frozen measured pins must not be treated as generic manifests.

The measured compile took 531.68 seconds. Complete 16-pair service improved
from 2.1773 to 1.8801 seconds. Compared with original CPU FP32, ATE increased
2.38% and translation RPE increased 3.66%; mean depth-confidence relative L2
error was 39.46%. All 80 strict original FP32 array comparisons failed.
Full timestamp replay delivered 17 fresh poses at 0.5699 Hz, with complete
P95 latency 2.4943 seconds. Quality and real-time acceptance remain false.

No weights, calibration arrays, ONNX/RKNN files or SDK binaries are public.
Code licensing does not grant rights to original model weights. See
[measured results](../../../../results/original_calibration_half_io_v56.json)
and [model provenance](../../../../MODEL_LICENSES.md).

The additional `compile_calibrated_dino.py` compiles all 24 unchanged original
DINO graphs using the same original-FP32 capture's 16 individual-frame inputs
per layer. `rknn_calibrated_dino.py` and
`compare_calibrated_dino_complete.py --calibrated-dino` retain the original
parameter checks and execute this diagnostic alongside the calibrated
aggregator. Copy them beside `rknn_calibrated_pairs.py` in the recorded board
experiment directory. `analyze_calibrated_dino_saved.py` reads the resulting
private outputs and original reference. The fixed evaluation images are
disjoint from calibration.

This candidate is rejected: complete mean service was 1.7281 seconds, but
rotation RPE increased from 10.7651 to 14.4809 degrees (+34.5%) versus the
original CPU reference. No timestamp replay was performed for it. The normal
replay entry continues to select the original FP16 graphs.
