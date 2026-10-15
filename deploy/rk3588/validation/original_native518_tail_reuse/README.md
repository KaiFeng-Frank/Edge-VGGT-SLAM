# Experimental native518 tail reuse and NPU core mapping

This board-only `ModelExecutor` retains the original two-frame `[2,3,392,518]`
pixel grid, 24-layer DINO, all 48 aggregator blocks, four camera updates, both
DPT frames, original pose codec and original Solver/SL4 map path (`max_loops=0`).
It remains an experimental validation adapter; production defaults are unchanged.

The cache holds only the immediately preceding successful pair's tail-frame
FP32 DINO output before final normalization and pair-role camera/register tokens.
A hit requires matching model/config/core identity and exact prepared pixel
bytes, beyond their SHA256. The new tail always traverses all four DINO groups;
both CPU token preparations still run. Every aggregator and head runs again.
The single entry occupies 6,700,608 bytes; failure and close clear it.

Private prerequisites must already exist: the original checkpoint, frozen S100
helpers/config and all linked compiled DINO/aggregator/head artifacts, runtime
2.3.2, board Python/sysroot, TUM RGB files, v100 references and the same-input
v109 pair-chain receipt plus arrays/poses/SL4/PLY. Their pinned absolute paths
are required; no weights, captured arrays, RKNN/SDK binaries or automatic
download/build workflow are supplied here. Checkpoint/source/artifact identities
remain independently checked, including the original checkpoint before loading.
Repository code and model-weight licenses remain separate; see
[model provenance](../../../../MODEL_LICENSES.md).

| Staged source | SHA256 |
| --- | --- |
| `v110_native518_executor.py` | `e153a5f71ef0b10119207b2e43589b628a2e85ab1abeaeb72d2df2a7dd6fb638` |
| `run_native518_tail_reuse_chain.py` | `d645622ad58e4a7f291347b902ee5d55e16e12c0252e20f99c772cb32ba85f72` |

These recorded commands use the tested controller: 1 GHz NPU, 75°C limit,
at most 180 seconds, and restoration of the prior clock/governor state.
The staged launchers set process-local library/thread settings, exact executor
pin, core mask and worker timeout. For another run, change each launcher's
worker `--output-dir` and the controller `--output` to fresh paths first.
Run the two stages sequentially after staging verified sources in S110.

```bash
BASE=/mnt/vggt-data/vggt-slam2
S110="$BASE/experiments/vggt-v110-source-r1"
PRODPY="$BASE/venv/bin/python"
CONTROL="$BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py"
"$PRODPY" -B "$CONTROL" --execute-frequency-test --frequency-hz 1000000000 \
  --temperature-limit-c 75 --timeout-seconds 180 \
  --launcher "$S110/run_tail_reuse_core1.sh" \
  --output "$BASE/experiments/vggt-v110-tail-reuse-core1-control-r1"
"$PRODPY" -B "$CONTROL" --execute-frequency-test --frequency-hz 1000000000 \
  --temperature-limit-c 75 --timeout-seconds 180 \
  --launcher "$S110/run_tail_reuse_core7.sh" \
  --output "$BASE/experiments/vggt-v110-tail-reuse-core7-control-r1"
```

Both routes processed `[32,40]` then `[40,48]`: one miss followed by one exact
tail hit. All ten canonical outputs and final poses, SL4 matrices and PLY were
byte-identical to uncached v109. Each route balanced 20 contexts, 32 runs and
42 output buffers; core mask 7 was forwarded to every model context.
Prediction times were 34.8016/30.9910 s with mask 1 and 29.8890/27.0809 s with
mask 7. Mask 7 pose publication took 29.9510/27.1631 s including diagnostic saves.
These are separately run two-pair checks, without interleaved ABBA or latency
distribution. They establish neither CPU FP32 equivalence nor realtime/P95,
trajectory accuracy, map cadence, 30-minute stability or 8 GB operation.
