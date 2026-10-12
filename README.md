# Edge-VGGT-SLAM

This project ports **VGGT-SLAM2 with its original pretrained VGGT model** to RK3588S. The pose and dense-depth path remains original VGGT, CameraHead, DPT, the original pose codec, submap construction and SL4 graph optimization. VGGT's internal DINO modules remain part of that model.

The current tree contains only this model's execution path and relevant component diagnostics. Improvements may change operator implementations, storage, layout and validated reuse of unchanged computations. They must retain the original checkpoint and pass comparisons against the original model on identical inputs. No additional learned model or added traditional pose frontend is permitted. [Project constraints](AGENTS.md), [execution architecture](ARCHITECTURE.md).

**The real-time target remains unachieved.** The target is at least 20 fresh-observation poses per second with complete input-to-pose P95 latency at most 100 ms. Map cadence, bounded queues, trajectory quality, 30-minute stability and operation within 8 GB each need separate validation. A component's timing, repeated prediction or offline reconstruction does not satisfy those targets. Current measurements use dataset files on a 16 GB RK3588S; physical-camera validation is deferred.

**Latest 1 GHz timestamp replay: two parallel instances of the original VGGT
feed one ordered original SL4 solver, producing 32 fresh pose/map updates from
798 frames at 1.120 Hz. Complete input-to-pose P95 is 1.976 s.** Dispatching
new requests at least 0.8 s apart reduces publication-gap P95 to 1.031 s and
maximum gap to 1.219 s. Two of 31 gaps still exceed one second, so this does
**not** establish one fresh update every second or meet the project real-time
target. Results are published as soon as original inference, SL4 and map
publication finish; no poses are interpolated or repeated. Startup is 93.698 s;
peak process RSS is 6217.49 MiB on the 16 GB board. All 24 SDK contexts close.

The unstaggered two-worker replay measured 1.101 Hz / P95 2.364 s, with
14 of 31 publication gaps exceeding one second. The earlier single-worker run
measured 0.900 Hz / P95 1.367 s. These are separate short replays with different
selected frames; throughput, latency, cadence and trajectory quality must be
assessed separately.

A separate matched eight-pair complete-model comparison using three workers
reduces offline batch time from **10.773 to 5.143 s (2.095× throughput)**. All
40 prediction comparisons are byte-identical; ordered original SL4 produces
identical saved poses and maps. This batch excludes initialization and saved
output writes. The three-worker timestamp replay subsequently fails at its
first inference with an SDK allocation error and SIGSEGV; only the two-worker
replay completes. Original Torch FP32 quality, physical 8 GB operation and
30-minute stability remain unvalidated. All frequency tests restore 300 MHz.
[Execution sources, measurements and reproduction](deploy/rk3588/validation/original_parallel_execution/README.md).

The preceding optional `--immutable-execution` optimization keeps the original
model arithmetic and reduces repeated model-tree inspection under an explicit
exclusive-owner contract. It retains per-call tensor versions, ownership and
I/O checks, with full validation at session start and close; it does not detect
arbitrary live model mutation. The flag defaults off.
[Contract and matched comparison](deploy/rk3588/validation/original_immutable_execution/README.md).

Further original-model execution diagnostics found no production improvement: FC2 output-channel partition and native tensor metadata checks were slower; a five-pair FC2 K4 comparison at 1 GHz gave only a small, variable component difference. Profiles of all four current groups confirm that their main residual stream already retains its channel layout. Those component diagnostics were not adopted; the later session improvement above changes only execution checks. [Evidence and reproduction](deploy/rk3588/validation/original_execution_diagnostics/README.md).

A subsequent original-pair quantization experiment is also **not adopted**.
Power-of-two Norm/linear scaling preserves all four tested complete FP32
outputs byte-for-byte, but its INT8 output errors remain about **9–12%**.
At 1 GHz the two-block component takes 32.682 ms in FP16, 20.957 ms with plain
INT8 and 20.666 ms with the scaling. The small difference from plain INT8
does not establish a new speed or full-model quality improvement. The latest
complete replay result above is unchanged.
[Measurements and reproduction](deploy/rk3588/validation/original_power2_smoothing/README.md).

The subsequent **300 MHz retained-group path** reduces mean complete-model
service on the same 16 pairs from **1.958 s to 1.797 s (8.23%)**. All 80
prediction arrays, 80 retained/target arrays and the saved trajectory remain
byte-identical to the preceding FP16 path; the original FP32 strict gate
still fails. It executes all 48 original aggregator blocks using four graph
calls and keeps the original consumer outputs. Complete-model contexts fall
from 32 to 12. Its separate timestamp replay produced **17 fresh poses at
0.594 Hz, complete P95 2.071 s**, with 7,187.75 MiB peak RSS on the 16 GB board.
Startup was 30.71 s, separate from the 28.60-second replay. These results
remain below the real-time targets. Defaults are unchanged; select the
experimental path with `--rknn-retained-aggregator` and its compiled manifest.
[Measured results and limits](results/original_retained_groups_v67.json),
[reproduction](deploy/rk3588/validation/original_retained_groups/README.md).

The subsequent original-MLP native MatMul diagnostic is **not adopted**:
at 300 MHz on one NPU core, the matched fused rank3 compiled MLP takes
15.139 ms and the native route takes 33.878 ms. Both fail the strict original
FP32 tensor gate; this is one component on one real input, with no new full-model
or replay improvement. An initially unfused rank2 comparison is preserved as
a compiler-layout pitfall, not a speedup claim.
[Measurement and reproduction](deploy/rk3588/validation/original_native_mlp/README.md).

A new original-CameraHead diagnostic splits each trunk FC1's output channels
into four parts while retaining every original parameter and all four
refinements. At 300 MHz, matched complete-head warm calls fall from
**213.300 to 177.352 ms (16.85%)**. Actual profiles show three-core convolutions
and a separate GELU in place of the single-core fused FC1/GELU. Outputs change
and both routes still fail the original FP32 strict gate. That component
comparison used one input and three warm calls per route.
[Measurements and reproduction](deploy/rk3588/validation/original_camera_partition/README.md).

The optional complete-model integration now measures **1.813 to 1.738 s
(4.13%)** on the same 16 pairs at 300 MHz, or **1.754 to 1.695 s (3.36%)**
excluding first calls. All retained features and depth/confidence arrays remain
byte-identical; pose and intrinsic outputs change. The short matched trajectory's
ATE/RPE changes relative to original CPU are under 2%, while the strict tensor
gate still fails. A separate 798-frame timestamp replay produces **18 fresh
updates at 0.609 Hz, complete P95 2.396 s**, below the real-time target.
Select it explicitly with `--rknn-camera-split4` alongside the existing camera
flag; defaults remain unchanged.
[Complete comparison, replay and limits](deploy/rk3588/validation/original_camera_integration/README.md).

Applying the same output-channel idea to both FC1 layers of the original
frame0/global0 pair gives a smaller component gain: **39.019 ms to 37.850 ms
(two parts, 3.00%) or 37.914 ms (four parts, 2.83%)**, at 300 MHz with 38 warm
calls per graph across two real inputs. FP32 ONNX outputs remain byte-identical;
NPU outputs change and the original FP32 strict checks fail. This pair-only
diagnostic is not installed in the full-model runtime.
[Evidence and reproduction](deploy/rk3588/validation/original_pair_partition/README.md).

A separate **FC2 input-channel partition** diagnostic reduces the complete
original pair from **39.067 to 35.766 ms (8.45%)** using four K parts, with
38 warm calls per graph across two real inputs at 300 MHz. Its FP32 graph passes
the existing tolerance but changes reduction order and is not byte-identical.
FP16 outputs change and the original FP32 strict checks still fail. The result
is a component measurement; full-model integration and quality remain untested.
[Evidence and reproduction](deploy/rk3588/validation/original_pair_fc2_partition/README.md).

Extending that FC2 rewrite to all four retained groups fails the fixed FP32
gate in group 2: one element exceeds tolerance on the second real input.
A balanced-sum follow-up fails at the same element. Neither version was
compiled or integrated; the pair-level gain is not a full-model gain.
[Full-group checks](deploy/rk3588/validation/original_retained_fc2/README.md).

A separate original-checkpoint memory experiment reclaims clean file-backed
pages after backend initialization. Process RSS changes from **7094.8 to
2703.1 MiB**, then stays at **2744.5–2762.1 MiB** in 16 complete inference
snapshots. All 160 saved arrays and the trajectory remain byte-identical.
The earlier initialization peak remains **7095.7 MiB**; speed does not improve.
This is an explicit diagnostic, with no production-default change or physical
8 GB acceptance. [Measurements and reproduction](deploy/rk3588/validation/original_checkpoint_pageout/README.md).

Reclaiming those pages **after each backend installation** also lowers the
observed initialization peak: **7169.9 to 3920.1 MiB** in a complete 16-pair
comparison, with all 160 saved arrays and the trajectory byte-identical.
The new explicit `--checkpoint-pageout` replay option produces 18 fresh updates
from 798 timestamped frames at **0.621 Hz, complete P95 2.030 s**, with peak
process RSS **3920.4 MiB**. This improves measured memory usage; there is no
speed improvement or real-time acceptance. The option is off by default, and
physical 8 GB operation remains untested.
[Implementation, measurements and reproduction](deploy/rk3588/validation/original_staged_pageout/README.md).

The additional-model retrieval and semantic paths have been removed. **Loop closure has no replacement implementation:** `max_loops=0` is the supported Solver configuration. Requests to enable loops or supply legacy auxiliary-model arguments must fail before loading or inference. The original VGGT model's numerical similarity capability does not itself implement loop retrieval.

The original checkpoint is `facebook/VGGT-1B`, `model.pt`, **5,026,874,952 bytes**, SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. Source-code and checkpoint licensing are separate; see [model provenance](MODEL_LICENSES.md) and [upstream pins](UPSTREAMS.json). No weights, captured arrays, weight-bearing exports or SDK binaries are distributed. The loader verifies this exact byte size and full SHA256 before deserialization, including explicit local paths and cached downloads. Automatic downloads use the recorded immutable revision; there is no runtime identity-check bypass.

## Run original-model timestamp replay

The board replay command runs original VGGT FP32, the original pose codec, dense map insertion and SL4, with a one-slot source queue and optional reuse of the previous frame's original DINO output. It writes `poses.tum`, `map.json`, `map_chunks/` and `results.json` after actual new-image updates:

```sh
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
python -B -u /mnt/vggt-data/vggt-slam2/src/scripts/replay_original_vggt.py \
  --rgb-index /mnt/vggt-data/vggt-slam2/datasets/rgbd_dataset_freiburg1_xyz/rgb.txt \
  --checkpoint /mnt/vggt-data/vggt-slam2/weights/model.pt \
  --output /mnt/vggt-data/vggt-slam2/experiments/original-replay-new-run
```

Use a new output directory. Default image size is 112; `--no-patch-cache` disables the original DINO tail cache. Do not restrict the outer process to CPUs 4–7: the decoder uses CPU0 and the model uses CPUs 4–7.

`--model-cpus 6 7 --model-threads 1` can set model-process affinity independently of the thread count. The matched complete-model comparator accepts the same options. This leaves the decoder on CPU0 and changes no system clocks or governors. The default affinity remains unchanged: a matched 16-pair check measured 1.952 s with CPUs 4–7 and 1.956 s with CPUs 6–7, with identical outputs and no observed improvement. [Measured CPU-affinity comparison](results/original_cpu_affinity_v60.json).

The earlier original-aggregator FC1 tap diagnostic was rejected: matched component calls became 40–45% slower after extra output handling. [Measured diagnostic and failure](results/original_fc1_tap_v61.json). The CameraHead output-channel experiment above adds no SDK output or diagnostic tap.

The first full timestamp replay served **7 new pose/map updates from 798 input frames**, at **0.195 Hz** over 35.900 s including the EOF drain. Complete software-arrival-to-published-pose **P95 was 8.160 s**; 790 frames were skipped or overwritten. Peak process RSS was 4,977.2 MiB on the 16 GB board. These measurements include decoding, queuing, original model execution, SL4 and pose/map publication. They do **not** meet the real-time target. Resolution quality, trajectory accuracy, 30-minute operation and 8 GB deployment remain unvalidated. [Measured record](results/original_timestamp_replay_v52.json).

The experimental complete-model path now executes **all 48 original aggregator blocks, all 24 original DINO blocks, all four original CameraHead refinements and original DPT logits through RKNN**. Original preprocessing, boundary operations, activations and SL4 remain on CPU. On 17 identical TUM frames, 16 complete overlapping pairs per configuration measured:

| Complete original-model configuration | Mean service time | Sim3-aligned ATE RMSE |
| --- | ---: | ---: |
| CPU FP32 | 7.458 s | 0.17719 m |
| RKNN aggregator, CPU heads | 5.366 s | 0.17857 m |
| RKNN aggregator and CameraHead | 3.609 s | 0.17808 m |
| Add all 24 original DINO blocks | 3.087 s | 0.17476 m |
| Add original DPT logits | 2.665 s | 0.17476 m |
| Pair all 48 aggregator blocks, four CPU threads | 2.590 s | 0.17476 m |
| Same paired path, one CPU thread | 2.177 s | 0.17476 m |
| Explicit FP16 SDK I/O, same original FP16 graphs | 2.077 s | 0.17476 m |
| Add DINO FP16 I/O and preserve both cache guards with faster lookup | 1.975 s | 0.17476 m |
| Optional six-group DINO, no observed service improvement | 1.982 s | 0.17476 m |
| Calibrated W8A8 aggregator diagnostic, quality unaccepted | 1.880 s | 0.18141 m |
| Calibrated W8A8 aggregator + DINO, rejected rotation error | 1.728 s | 0.19823 m |

These are sequential observations across runs, not an interleaved thermal comparison. Service timing excludes paced input queues and file publication. This short, sparse, low-resolution segment does not validate overall trajectory quality. All strict FP32 tensor comparisons still fail, and the FP16 candidate is not accepted as numerically equivalent.

The separate 798-frame timestamp replay with RKNN aggregator and CameraHead delivered **10 fresh updates at 0.312 Hz, P95 4.878 s** with four A76 threads. Two threads delivered 9 updates at 0.267 Hz, P95 6.239 s, so four remains the default. These results still miss realtime. Peak process RSS was about 6.64 GiB on the 16 GB board; SDK allocation metrics are reported separately and do not establish 8 GB operation. [Complete measurements and limits](results/original_full_npu_v53.json).

The replay enabling original DINO and DPT execution through RKNN measured:
**12 fresh updates at 0.398 Hz, complete input-to-pose P95 3.212 s** from the
same 798-frame source. It publishes 12 dense chunks; 785 source frames are
skipped or overwritten. Peak process RSS is 7,412.98 MiB on the 16 GB board.
All 74 RKNN contexts close cleanly. The result still misses the pose-rate and
latency targets; 30-minute and 8 GB operation remain unvalidated. The new DPT
FP32 ONNX graph separately passes both original-reference checks before SDK
compilation, preserving all 62 learned tensors and original inplace residual
semantics. [DINO/DPT measurements](results/original_dino_depth_v54.json).

The preceding complete replay uses **24 frame/global graph pairs** with both
original outputs, all original DINO/CameraHead/DPT computations, and one CPU
thread. It delivers **14 fresh updates at 0.478 Hz, complete P95 2.669 s** from
798 source frames. All 80 prediction arrays and the saved trajectory on the
matched comparison inputs remain bitwise identical to the preceding FP16
candidate. Peak RSS is 7,296.82 MiB; all 50 contexts close cleanly. This is a
29.26-second test, still below the real-time target. Add
`--rknn-paired-aggregator --model-threads 1` and use the paired manifest.
[Results and limitations](results/original_fused_aggregator_v55.json),
[pair export and replay instructions](deploy/rk3588/validation/original_fused_aggregator/README.md).

The current optional FP16 I/O path covers the original aggregator and DINO,
with faster cache-state lookup that retains both complete guards. It preserves
all 80 prediction arrays and the trajectory bitwise against the preceding FP16
path. Complete timestamp replay measured **16 fresh poses at 0.538 Hz, P95
2.581 s**; RSS was 7,259.60 MiB. This improves throughput from 0.513 Hz, but
does not improve the preceding 2.551 s P95. The test covers 29.72 seconds,
including the EOF drain, not sustained real-time operation.
[Current measured results](results/original_dino_io_guard_v57.json).
Add the two `--half-io-library` options to the paired command as described in
[FP16 I/O setup](deploy/rk3588/validation/original_half_io/README.md).

An optional six-group DINO backend reduces complete-model RKNN contexts from
50 to 32 and preserves all 80 saved prediction arrays and poses bitwise against
the preceding FP16 path. **It did not improve matched complete-model service:**
1.982 s versus 1.975 s. Its separate 798-frame replay produced 16 fresh poses
at **0.543 Hz, complete P95 2.472 s**, with 7,259.28 MiB peak RSS. Startup took
38.74 s, reported separately from the 29.49-second replay. These sequential
short runs establish no real-time breakthrough. The default backend selection
is unchanged; add `--rknn-grouped-dino` only with the six-group manifest.
[Results, including the stopped INT4 experiment](results/original_grouped_dino_v58.json),
[grouped graph build and runtime](deploy/rk3588/validation/original_grouped_dino/README.md).

A separate original-aggregator W8A8 calibration experiment measured **0.570 Hz,
P95 2.494 s**. Its short matched trajectory ATE increased 2.38% versus CPU FP32,
but mean depth-confidence relative L2 error was 39.46%. It remains an unaccepted
quality experiment. Both paths retain all original layers and tokens, fail the
strict original FP32 comparison and miss the real-time targets.
[Measured results](results/original_calibration_half_io_v56.json),
[calibration reproduction](deploy/rk3588/validation/original_calibrated_pairs/README.md).

After privately producing the pinned original artifacts, add `--rknn-manifest PATH/compiled.json --rknn-manifest-sha256 SHA256 --rknn-camera-head` to the replay command. `--model-threads 2` selects CPUs 6–7; one and the default four use CPUs 4–7. The one-thread setting reduces Torch CPU parallelism while retaining the three-core NPU setting. These settings affect only the process. [Build and reproduction instructions](deploy/rk3588/validation/original_full_aggregator/README.md).

The latest configuration additionally uses `--rknn-dino-manifest DINO/compiled.json
--rknn-dino-manifest-sha256 SHA256 --rknn-depth-manifest DEPTH/compiled.json
--rknn-depth-manifest-sha256 SHA256`. All selected backends install before the
original tail cache. Unsupported inputs and SDK failures raise without fallback.

The optional `--checkpoint-pageout` requires the complete retained/paired
aggregator, native half I/O, grouped DINO, camera split4, depth and patch-cache
configuration. It advises only verified clean mappings of the original
checkpoint in its own process, after each of the four backend installations.
Its startup cost is included in `startup_seconds`; replay does not add per-frame
memory snapshots. Exact measured arguments and artifact identities are in the
[staged pageout reproduction](deploy/rk3588/validation/original_staged_pageout/README.md).

## Original-model evidence

The current original-only runtime is deployed on RK3588S. Its 168 relevant generated checks and four actual original-model forwards pass; the fixed 112-scale TUM pair and first SL4 graph preserve the original outputs. A separate 518-scale diagnostic covers three overlapping submaps: all five prediction arrays and saved graph arrays match bitwise with the DINO tail cache on/off, reducing DINO computation from six image occurrences to four. Complete forwards still took **134–195 seconds**. These are numerical regression results, not real-time acceptance. [Current evidence and limits](results/original_only_v47.json).

| Experiment | Measured result | Boundary |
| --- | --- | --- |
| Original CPU FP32 VGGT, two office images, four A76 threads, requested 112 | 3.831 s complete forward; actual 56×112 input | Excludes preprocessing, graph optimization and publication; no real-time claim |
| Original DINO block 0, `[1,37,1024]`, RKNN | Three warm complete I/O calls average 5.489 ms | One original-weight block; relative L2 error 1.2253% against FP32; no complete-model accuracy acceptance |
| Original complete frame0 → global0 pair, RKNN, actual image shape `[2,3,252,336]` | Three warm complete I/O calls average 338.683 ms; all 52 internal operators on NPU | Two blocks only; relative L2 error 0.483569% against original FP32, strict elementwise gate fails |
| Original complete CameraHead, RKNN | Three warm complete I/O calls average 203.646 ms | All four refinement iterations retained; strict FP32 elementwise gate fails |
| Original CPU DINO tail reuse, three overlapping pairs | Six input frames require four actual DINO computations; saved pose/depth/confidence match original references bitwise | Fixed-input evidence; frame/global attention and heads still run, arbitrary-input and trajectory equivalence unproven |

A same-process AB/BA check found bitwise-identical separate/fused NPU outputs and observed warm SDK I/O means of **396.442 / 334.831 ms** (15.54% lower in this short test). Both routes still fail the original FP32 gate. This is a component optimization, with full-model speed and trajectory quality unproven.

A new original-pair PTQ diagnostic used eight real calibration pairs and two disjoint heldout pairs, entirely on RK3588S. Normal SDK IO means were **337.086 / 255.640 ms** for FP16 / PTQ (24.16% lower), but PTQ tensor relative L2 errors rose to **10.31–10.55%**. Both NPU implementations fail the fixed original FP32 gate; the candidate remains unadopted. The new board CPU ORT check also preserves a one-element strict-gate failure on one heldout input. [Precision result and limits](results/original_precision_v49.json), [reproduction and independent audits](deploy/rk3588/validation/original_precision/README.md).

The next original-graph diagnostic found **no observed Flash Attention flag benefit on RK3588S**: ordinary/off versus on took 336.829 / 336.963 ms mean normal SDK IO, with the same visible compiled operator placement. A saved-reference analysis also proves that even the closest finite FP16 representation cannot meet the existing strict component tolerance for every element of these two outputs. Those strict failures remain recorded; full-model ATE/RPE acceptance is a separate, unmeasured question. [Engineering findings](docs/ENGINEERING_NOTES.md#102-fp16-可表示精度与原-fp32-组件门槛是不同问题), [representation proof](deploy/rk3588/validation/original_mixed_precision/accuracy/RESOLUTION_NOTE.md).

A conservative mixed-precision original-pair candidate retained floating logical I/O and requested INT8 at four FC1/GELU outputs. It measured **336.673 / 319.928 ms** normal SDK IO for ordinary / hybrid (4.97% lower), but its tensor relative L2 errors were **6.67–7.91%** against original FP32. It remains unadopted. The first SDK selector rejection, corrected configuration, raw-evidence hashes and scope limits are preserved in the [v50 record](results/original_mixed_precision_v50.json) and [reproduction tools](deploy/rk3588/validation/original_mixed_precision/README.md).

The [continuous pair record](results/original_pair_v48.json) binds the new board CPU check, nonquantized compile, actual SDK calls and independent saved-output audit. Its [reproduction tools](deploy/rk3588/validation/original_pair/README.md) retain the complete original blocks and the numerical rejection.

The [measurement summary](MEASUREMENTS.json), [complete-head record](results/camera_head_v15.json), [tail-cache record](results/patch_cache_v19.json) and [complete-block attention record](results/hybrid_block_v20.json) preserve each experiment's inputs, numerical failures and timing scope. The [current arithmetic budget](deploy/rk3588/validation/original_budget/README.md) audits original source, saved tensor dimensions and the fresh-observation denominator; it uses no hardware peak assumption. The [earlier capacity calculation](results/capacity_v20.json) is a source-based workload/storage projection, not measured hardware throughput or endurance. [Engineering notes](docs/ENGINEERING_NOTES.md) explain the remaining execution and validation issues.

Historical records keep their original identity and failed gates. Component RKNN execution does not establish a complete NPU model, and its FP16 numerical rejection is not removed by successful compilation. Archived files remain in repository history; they are not current implementation or reproduction options.

## Setup and checks

The tested environment uses Linux aarch64, Python 3.11.17, PyTorch 2.3.1 CPU, torchvision 0.18.1 and an external GTSAM build providing SL4. The [deployment guide](deploy/rk3588/README.md) lists original source/checkpoint pins and external SDK requirements. Real inference, tracing/export validation and calibration run only on RK3588S. Host work is limited to source/static inspection, saved-data audits, non-calibrating compilation and small generated checks.

Use the original-model entrypoint's `--help` for current dataset arguments. Lightweight checks for the retained execution contracts are:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_runtime_loading.py tests/test_model_execution.py \
  tests/test_solver_execution.py tests/test_patch_cache.py
```

These checks use generated data or fake services; they do not establish board inference or SL4 trajectory quality. A fresh clone needs the licensed external dependencies listed in the deployment guide before hardware diagnostics can run.

## Attribution

The root [BSD-2-Clause license](LICENSE) and MIT-SPARK copyright are retained. The pinned VGGT implementation, original model weights, runtime dependencies and dataset retain their own terms. [Third-party notices](THIRD_PARTY_NOTICES.md) identify those boundaries. The repository license grants no additional commercial rights to the original checkpoint or compiled artifacts containing it.
