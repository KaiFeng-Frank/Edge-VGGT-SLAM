# Experimental native518 trajectory validation

This harness uses 17 actual TUM `freiburg1_xyz` RGB frames at indices
`32 + 24*i` (`i=0..16`), forming 16 overlapping two-frame pairs across about
12.8 seconds of original dataset time. Each request preserves the original
`[2,3,392,518]` pixels, VGGT checkpoint, 24 DINO layers, all 48 aggregator
blocks, four camera refinements, both DPT frames and original pose codec.
Loop closure remains disabled (`max_loops=0`). Production defaults are unchanged.

Four board workers process four pairs each with NPU core mask 7. Each worker
starts with an empty previous-tail cache; exact overlap reuse applies only
within its batch. Every new tail, aggregator and head executes normally.
The separate numerical baseline runs the complete original model in FP32 on
the host CPU. Host timings are reference-capture timings, not board performance.

Private prerequisites must already exist: the original checkpoint/source,
frozen S100 helpers/config, linked compiled DINO/aggregator/head artifacts,
the frozen v110 executor, RKNN runtime 2.3.2, board Python/sysroot, original
TUM files and ground truth. Their pinned identities and absolute paths are
required. This folder supplies validation source only; weights, model exports,
captured prediction arrays, SDK binaries and an automatic download/build
workflow are not supplied. Code and model-weight licenses remain separate;
see [model provenance](../../../../MODEL_LICENSES.md).

| Source | SHA256 |
| --- | --- |
| `run_native518_trajectory_batch.py` | `78cb60f56df0b98879a987b34db1172c62cd0d51b5364cb1c3c81fd6f3ac1fe9` |
| `run_original_fp32_native518.py` | `23e29fa0679e80755153b055702129309f39e495cf3349a9d080fa3980a5190c` |
| `replay_saved_native518_sl4.py` | `a0d37e54b1b1655bf4443f5fa80c07043a1ada9c12dbf1d0cb851b41dc7af271` |

Stage verified sources and launchers in S111. The launchers set process-local
library/thread settings and a 170-second worker deadline. Run batches
sequentially under the existing controller: 1 GHz NPU, 75°C limit, at most
180 seconds, restoring the prior clock state. Before a new run, change each
launcher's worker `--output-dir` and controller output to fresh paths.

```bash
BASE=/mnt/vggt-data/vggt-slam2
S111="$BASE/experiments/vggt-v111-source-r1"
PRODPY="$BASE/venv/bin/python"
CONTROL="$BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py"
for START in 0 4 8 12; do
  "$PRODPY" -B "$CONTROL" --execute-frequency-test --frequency-hz 1000000000 \
    --temperature-limit-c 75 --timeout-seconds 180 \
    --launcher "$S111/run_board_batch${START}.sh" \
    --output "$BASE/experiments/vggt-v111-batch${START}-control-r1"
done
```

The board saved-prediction consumer accepts four native manifests or one
original FP32 manifest with explicit SHA256 pins. It verifies source PNGs,
prepared pixel bytes and all five canonical NPY outputs. One unchanged
Solver consumes all 16 pairs across batch boundaries using original
`add_points` and SL4 optimization. It exports 17 original TUM poses, 32
homographies and fixed pixel samples from all 32 frame occurrences, retaining
each overlap occurrence and the original full/sample confidence masks.
This CPU geometry replay requires a bounded process timeout, not an NPU test.

Saved-only evaluation fits each route's trajectory to TUM ground truth with
Sim3 and reports translation ATE and adjacent selected-frame RPE (about
0.8 seconds apart). The same alignment transforms fixed map samples for
CPU/NPU point disagreement; confidence-mask IoU uses identical pixel roles.
These map comparisons measure FP32-reference agreement, not ground-truth map
accuracy. The internal world remains the original optimized SL4 gauge.
Dataset timestamps describe observation spacing; offline elapsed runtime is
reported separately. This short check establishes no realtime/P95, physical
camera, 30-minute stability or 8 GB acceptance.
