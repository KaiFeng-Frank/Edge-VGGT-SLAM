# Original VGGT immutable execution validation

This experiment reduces repeated Python model-state reflection under an explicit `exclusive_immutable_model` contract. It preserves the original VGGT weights, all layers and tokens, four CameraHead refinements, depth computation, existing compiled graphs and FP16/FP32 boundaries. It does not establish equivalence to the previous mutable-model guard semantics.

The production replay option is **`--immutable-execution`**, default **off**. It requires retained/paired aggregation, grouped DINO, camera split4, depth, half I/O and the patch cache. The full-model comparator uses the wrapper's explicit `--mode immutable`; its baseline entrypoint remains unchanged.

After backend, observer and cache installation, one worker thread exclusively owns the model. Module/class/forward/config/registry/hook mutations, Tensor replacement, `.data` writes and external model mutation are forbidden until close. The session performs the original full validation at installation and again before closure. Both original cache guards keep their mode, owner and method checks; fixed Tensor references retain per-call version/dtype/device/data-pointer/requires-grad checks. Retained/DINO keep their method and global-hook checks, and retained backend identities are checked. Arbitrary live module or scalar monkeypatches are no longer checked every call; final validation is not a guarantee against a mutation that is later undone.

Only three instance methods are overridden: `cache._fingerprint`, `retained._check_lifecycle` and `dino._check_lifecycle`. No global class methods or original model calculations are replaced. Close validates and restores session-owned overrides before cache/backend cleanup; validation failures are reported and must not prevent remaining SDK cleanup. An active or foreign-thread close is rejected without dismantling the session.

## Saved full-model result

The matched 1 GHz runs used 16 pairs of original TUM `freiburg1_xyz` images, `[2,3,84,112]`, one model thread on CPUs 4–7, and the same original graph manifests. The 17 image ordinals were `32,80,128,176,224,272,320,368,416,464,512,560,608,656,704,752,797`.

| Full original-model service | Baseline | Immutable |
| --- | ---: | ---: |
| All 16 pairs mean | 1.3901335175 s | 1.2482811046 s |
| Last 15 pairs mean | 1.3568222977 s | 1.2270444233 s |

The independent saved-data audit verified 32 complete NPZ files and the saved trajectory are byte-identical, with **160 prediction/aggregator arrays byte-equal** between these configurations. This compares the same existing compiled FP16 execution; the original CPU FP32 strict checks still fail and are not relaxed. The immutable receipt records 1,343 fixed tensors, 32 version checks, 64 method checks, successful seal/final validation, restored overrides and no session cleanup errors. These fixed-input results are not fresh-observation Hz or a trajectory/map quality acceptance.

Six generated metadata tests cover the unchanged two cache guards, fixed Tensor changes and invalidation, final scalar-mutation rejection, backend/global-hook checks, active/foreign-thread close safety and restoration ownership. They import no Torch and execute no model or SDK. Separate entrypoint tests cover opt-in dependencies and cleanup: 87 tests and 7 subtests passed, including 22 new immutable-session tests. See [generated-check receipt](public-tests-result.json).

## Board reproduction

First prepare the original checkpoint, upstream TUM data, captured references and the prior v54/v58/v67/v69 graph artifacts used by the launchers. They are private artifacts, not bundled with this source directory; all launcher SHA pins must match. No new export or calibration is part of this comparison.

On the board, copy the wrapper and launchers from this validation directory into their fixed experiment source directory. Copy the policy from its production source, `src/vggt_slam/immutable_execution_session.py`:

```sh
EDGE_BASE=/mnt/vggt-data/vggt-slam2
EDGE_VALIDATION="$EDGE_BASE/src/deploy/rk3588/validation/original_immutable_execution"
EDGE_STAGE="$EDGE_BASE/experiments/vggt-v77-source-r1"
mkdir -p "$EDGE_STAGE"
cp "$EDGE_VALIDATION/run_immutable_comparison_board.py" "$EDGE_STAGE/"
cp "$EDGE_VALIDATION/run_baseline_1ghz.sh" "$EDGE_STAGE/"
cp "$EDGE_VALIDATION/run_immutable_1ghz_r2.sh" "$EDGE_STAGE/"
cp "$EDGE_VALIDATION/run_replay_immutable_1ghz.sh" "$EDGE_STAGE/"
cp "$EDGE_BASE/src/vggt_slam/immutable_execution_session.py" "$EDGE_STAGE/"
```

The frozen wrapper's historical replay allowlist targets the prior CLI revision; use it only for the unchanged comparator here. The new replay uses its explicit production flag directly.

The copied policy's SHA256 must be `a731fb1f26ea82f31fa7123792625e1731e19d3e35723930aef18e331dbe6897`. The wrapper is used with the pinned original comparator. Replay uses the production replay CLI directly, including `--immutable-execution`.

Run the three tests **sequentially through the existing bounded controller**, within an authorized temporary 1 GHz test. Do not run the launcher alone and assume it changes or restores frequency:

```sh
/usr/bin/python3 "$EDGE_BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py" \
  --execute-frequency-test --frequency-hz 1000000000 \
  --launcher "$EDGE_STAGE/run_baseline_1ghz.sh" \
  --output "$EDGE_BASE/experiments/vggt-v77-baseline-1ghz-control-r1"

/usr/bin/python3 "$EDGE_BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py" \
  --execute-frequency-test --frequency-hz 1000000000 \
  --launcher "$EDGE_STAGE/run_immutable_1ghz_r2.sh" \
  --output "$EDGE_BASE/experiments/vggt-v77-immutable-1ghz-control-r2"

/usr/bin/python3 "$EDGE_BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py" \
  --execute-frequency-test --frequency-hz 1000000000 \
  --launcher "$EDGE_STAGE/run_replay_immutable_1ghz.sh" \
  --output "$EDGE_BASE/experiments/vggt-v77-replay-immutable-1ghz-control-r1"
```

The controller has a maximum 180 s duration and 75 °C stop, rejects observed frequency drift, terminates the child process group and restores/verifies the previous effective NPU frequency, governor and limits. Read each control receipt; a child exit alone does not prove restoration. Deployment does not enable 1 GHz or change clocks.

All controller, model-output and policy-receipt paths must be new. Existing result paths are deliberately refused. For another run, use fresh names in copied launchers and controller arguments and record the changed launcher hashes; preserve the previous evidence.

The first immutable attempt failed **before model loading/inference**: the wrapper's policy receipt parent created the same directory the comparator then required to be absent. The child started and exited nonzero; this is not an SDK inference failure. `run_immutable_1ghz_r2.sh` places the policy receipt at the separate `experiments/vggt-v77-immutable-policy-r2.json` path. That retry is the successful comparison reported above.

## Replay and limits

The new immutable replay receipt reports **26 fresh poses**, **0.899908369852497 Hz** over elapsed time including EOF drain, and complete input-to-pose **P95 1.3665765015830402 s**. Startup was **38.75220335298218 s** and the process peak RSS was **3,920.41796875 MiB**.

The latest-frame queue has capacity one. Its saved counts are 798 original frames = 797 generated + one skipped before decoding; 27 were delivered including the seed and 770 were overwritten. The session reports successful installation/final validation, 1,343 fixed tensors and all three overrides restored. The saved audit independently binds all original timestamps, recomputes the 26 complete latency records, verifies 26 map chunks / 366,881 finite points and 52 finite SL4 nodes, and checks the four checkpoint-residency stages. All 12 contexts and 422 SDK calls completed; all 234 aggregator output buffers were released. The 138 control and 27 platform samples were at 1 GHz, with a maximum sampled temperature of 65.615 °C; frequency, governor and limits were restored to their original settings, including 300 MHz. These replay values are separate from the fixed-input table and count fresh observations, not repeated predictions.

The 20 Hz / 100 ms target and original FP32 quality acceptance remain unmet. The short dataset replay does not establish a 30-minute run, physical-camera timing, 8 GB hardware operation or trajectory/map quality. Public files contain source and compact evidence, not model weights, captured arrays, weight-bearing graphs or SDK binaries.

[Complete saved audit](complete-ab-audit.json), [replay saved audit](replay-saved-audit.json), [numerical summary](../../../../results/original_immutable_execution_v77.json). The [original-target context analysis](target-token-context-saved.json) is a separate saved-feature diagnostic; it does not enable loop closure.
