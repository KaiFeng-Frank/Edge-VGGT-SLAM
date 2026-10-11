# Original DPVO FAST CPU replay research (v34)

This experiment uses the separate DPVO model. The default VGGT executor and SL4 mapper remain unchanged. It runs the original DPVO CPU ORT encoders, original Torch Update, native CPU correlation and vectorized CPU geometry. It adds no ORB, LK or PnP frontend. The measured run is a short, serial, unpaced CPU FP32 replay; it does not establish 20Hz operation, physical-camera latency, endurance or accuracy versus VGGT.

The configuration comes from the pinned official `dpvo/config.py` followed by `config/fast.yaml`: 48 patches, lifetime 11, removal window 16 and optimization window 7. `config/default.yaml` is not applied. The original shared Torch CPU generator consumes x, y and depth draws in order. Changing 96 patches to 48 changes RNG consumption and sampled centers; the same seed does not make FAST a subset of the older patches, and does not reproduce CUDA random bytes. The existing research limits remain 64 retained frames and 4,096 observations, with loop closure disabled.

`run-v34-dpvo-fast-board.py --help` lists all required external artifacts. Run only on Linux RK3588, with the frozen successful v30 r2 parent driver, stream module, encoder module and source manifest from [dpvo_stream](../dpvo_stream/README.md). Supply the original v26 source/weights, CPU ORT encoder model/export/ORT receipts and three-frame capture, v27 native correlation library/build receipt, v29 vectorized geometry and the TUM fr1_xyz RGB sequence. Models, data and compiled vendor/native binaries are not redistributed. The board environment needs Torch CPU, NumPy, OpenCV, CPU ONNX Runtime and four available A76 CPUs.

Deploy this driver into a new experiment directory and invoke it with these required arguments, plus the paths shown by `--help`:

```bash
python -B run-v34-dpvo-fast-board.py \
  --source-dir "$DPVO_SOURCE" --checkpoint "$DPVO_CHECKPOINT" \
  --geometry "$GEOMETRY" --native-helper "$CORR_HELPER" \
  --library "$CORR_LIBRARY" --build-receipt "$CORR_BUILD_RECEIPT" \
  --encoder-model "$ENCODER_MODEL" --encoder-export "$ENCODER_EXPORT" \
  --encoder-ort "$ENCODER_ORT_RECEIPT" --encoder-capture-dir "$ENCODER_CAPTURE" \
  --source-manifest "$V30_SOURCE_MANIFEST" --stream-module "$V30_STREAM" \
  --encoder-module "$V30_ENCODER" --parent-driver "$V30_R2_DRIVER" \
  --dataset-dir "$TUM_RGB_SEQUENCE" --output-dir "$NEW_BOARD_EXPERIMENT" \
  --frames 192 --seed 20261011 --run-time-limit 1200
```

Set the variables to verified artifact paths; the output must be a new directory under the board's experiments directory. Use the same CPU thread environment as the v30 run. The first three checks retain the original dense encoder-output gates and rebuild patch references at the actual 48 centers. They do not compare 48-patch outputs with archived 96-patch tensors. `run.json` and `input-manifest.json` record all requested frames, source/weight hashes, waiting/skipped/fresh observations, timings and cleanup. Keep large before/after-termination NPZ snapshots on the board when host space is limited. EOF's twelve refinement updates and reconstructed poses remain offline, never fresh.

`evaluate-v34-fast-trajectory.py` reads only saved JSON and GT text. It verifies identical complete input manifests, preserves separate fresh/EOF populations and reuses the frozen v30 evaluator's one-second all-source endpoint plan and nearest GT association within 20ms. The additional common-fresh comparison uses the intersection of actual fresh source IDs, before GT/error filtering. Each run gets one positive Sim3 fit on that same observation set; the same endpoint subset is used for ATE/RPE. This alignment is evaluation-only, not a GT prior or metric-scale estimate from inference.

```bash
python -B evaluate-v34-fast-trajectory.py \
  --default-run-dir "$DEFAULT_RUN" --default-run-sha256 "$DEFAULT_RUN_SHA256" \
  --fast-run-dir "$FAST_RUN" --fast-run-sha256 "$FAST_RUN_SHA256" \
  --groundtruth "$TUM_GT" --groundtruth-sha256 "$TUM_GT_SHA256" \
  --parent-evaluator ../dpvo_stream/evaluate-v30-dpvo-trajectory.py \
  --output "$NEW_COMPARISON_JSON"
```

Supply verified hashes and a new output file. Endpoint pairs overlap, so their count is not a count of independent trials. The tests use generated arrays, toy trajectories and fake geometry only; they do not run real weights, images, ORT or BA.

DPVO-derived code retains the adjacent [MIT license](LICENSE), Copyright (c) 2022 Princeton Vision & Learning Lab. The external checkpoint's redistribution/commercial license remains separately unverified; this notice does not license its weights.
