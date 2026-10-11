# Complete original VGGT with RKNN aggregator blocks

This experimental backend runs all 24 original frame blocks and all 24 original
global blocks through RKNN. It retains the original aggregator's alternating
execution, concatenated frame/global outputs, selected intermediate layers and
target tokens. Original DINO, CameraHead, DPT and pose decoding remain CPU FP32;
the compiled aggregator uses RKNN's nonquantized FP16 configuration. No original
layer or token computation is omitted. This is not a fully NPU-resident model.

The fixed input is two RGB frames, `[2,3,84,112]`. Frame blocks consume
`[2,53,1024]`; global blocks consume `[1,106,1024]`. Runtime installation verifies
the original checkpoint identity recorded by capture, all 864 block parameters,
original source hashes, each exported/compiled artifact and fixed positions.
All 48 contexts must initialize before any forward is replaced. SDK failures
raise without CPU fallback; closing restores the original forwards. Only normal
`max_loops=0` inference is supported. Similarity and attention visualization need
extra block outputs and are rejected.

All original-model execution and SDK compilation below run on RK3588S. Use the
existing isolated compiler environment described in the deployment records;
do not replace the production Torch environment. Each output directory must be
new. Checkpoints, captured arrays and ONNX/RKNN files contain licensed model
material and remain private.

1. Source `deploy/rk3588/env.sh` and run `capture_original_blocks_board.py`
   with `--repo`, `--checkpoint`, two actual `--images`, and `--output-dir`.
   It executes one complete original FP32 forward, including both active heads,
   and captures all 48 blocks. Its hooked forward time includes file writes and
   is not a performance measurement.
2. In the isolated ONNX environment run `export_original_blocks.py` with
   `--capture-dir`, the exact `--capture-sha256`, `--exporter` pointing to the
   unchanged sibling `../export-v17-aggregator-block-r3.py`, and `--output-dir`.
   The original R3 builder and its four frozen helpers must remain together.
3. Run `compile_original_blocks.py --manifest EXPORT/export.json
   --manifest-sha256 SHA256` in the same board compiler environment. It compiles
   every layer in a separate sequential SDK process and produces
   `EXPORT/compiled.json`. Completed artifacts can be resumed after identity
   verification. A failed layer directory is retained for diagnosis; preserve
   it before retrying, since incomplete directories are not overwritten.
4. Return to the production environment. The existing
   `scripts/replay_original_vggt.py` accepts `--rknn-manifest
   EXPORT/compiled.json --rknn-manifest-sha256 SHA256`, in addition to its RGB
   index, original checkpoint, new output directory and `--image-size 112`.
   Without these two options it keeps the original CPU FP32 path. Adding
   `--rknn-camera-head` also uses the existing original shared CameraHead step
   graph. This optional adapter retains all four refinements, original learned
   initial pose, FP32 pose accumulation and original activation. Its pinned
   private artifact is `experiments/vggt-v16-camera-step/camera-step-fp16.rknn`;
   the original v16 export and compile tools remain in the parent directory.
   It validates all 69 original head parameters against the retained capture
   metadata before replacing the head forward.

`compare_complete_model.py` uses the same fixed 17 real TUM frames for CPU,
RKNN aggregator, and RKNN aggregator plus CameraHead branches, and runs the
unchanged SL4 graph. It saves
canonical prediction arrays and trajectories privately. CPU and NPU branches
run sequentially, so timing differences alone are not a controlled thermal
comparison. Timestamp replay separately measures actual fresh-pose throughput,
decoding, queue time, full inference, graph work and publication, including EOF
drain. Neither successful compilation nor finite predictions establish
trajectory accuracy, realtime operation, loop closure or 8 GB stability.

The repository's code license and original VGGT checkpoint terms remain
separate; see [model licensing](../../../../MODEL_LICENSES.md).
