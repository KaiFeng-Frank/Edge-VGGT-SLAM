# Edge-VGGT-SLAM

This project ports **VGGT-SLAM2 with its original pretrained VGGT model** to RK3588S. The pose and dense-depth path remains original VGGT, CameraHead, DPT, the original pose codec, submap construction and SL4 graph optimization. VGGT's internal DINO modules remain part of that model.

The current tree contains only this model's execution path and relevant component diagnostics. Improvements may change operator implementations, storage, layout and validated reuse of unchanged computations. They must retain the original checkpoint and pass comparisons against the original model on identical inputs. No additional learned model or added traditional pose frontend is permitted. [Project constraints](AGENTS.md), [execution architecture](ARCHITECTURE.md).

**The real-time target remains unachieved.** The target is at least 20 fresh-observation poses per second with complete input-to-pose P95 latency at most 100 ms. Map cadence, bounded queues, trajectory quality, 30-minute stability and operation within 8 GB each need separate validation. A component's timing, repeated prediction or offline reconstruction does not satisfy those targets. Current measurements use dataset files on a 16 GB RK3588S; physical-camera validation is deferred.

The additional-model retrieval and semantic paths have been removed. **Loop closure has no replacement implementation:** `max_loops=0` is the supported Solver configuration. Requests to enable loops or supply legacy auxiliary-model arguments must fail before loading or inference. The original VGGT model's numerical similarity capability does not itself implement loop retrieval.

The original checkpoint is `facebook/VGGT-1B`, `model.pt`, **5,026,874,952 bytes**, SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. Source-code and checkpoint licensing are separate; see [model provenance](MODEL_LICENSES.md) and [upstream pins](UPSTREAMS.json). No weights, captured arrays, weight-bearing exports or SDK binaries are distributed. The loader verifies this exact byte size and full SHA256 before deserialization, including explicit local paths and cached downloads. Automatic downloads use the recorded immutable revision; there is no runtime identity-check bypass.

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
