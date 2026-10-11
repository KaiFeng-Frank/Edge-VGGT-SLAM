# Edge-VGGT-SLAM

**NPU-Accelerated Visual SLAM on Embedded Devices — experimental, first target: RK3588S.**

An embedded execution project based on [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM), focused on accelerating the original network while retaining the original VGGT-1B weights and SL4 mapping backend. RK3588S is the only board tested so far. Current NPU execution covers selected subnetworks; a complete all-NPU model and support for other vendors are not validated. All measured model inference runs on the board. Host tools export and compile graphs or inspect saved arrays.

**License scope:** the repository's BSD-2-Clause code license does **not** license the external model weights. The pinned original VGGT-1B checkpoint is **CC-BY-NC-4.0**, with a non-commercial restriction. Compiling it to RKNN does not grant commercial-use rights. See [model licenses](MODEL_LICENSES.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

The optional LK/PnP/triangulation frontend is a historical custom architecture experiment, separate from the original VGGT-SLAM 2 frontend; it does not use ORB. It is not the current route for claiming original-network real-time performance.

In the pinned upstream implementation, LK selects keyframes for VGGT inference ([main.py](https://github.com/MIT-SPARK/VGGT-SLAM/blob/35327ac28b7d193df9ccc39ba6346052bb6f1207/main.py), [frame_overlap.py](https://github.com/MIT-SPARK/VGGT-SLAM/blob/35327ac28b7d193df9ccc39ba6346052bb6f1207/vggt_slam/frame_overlap.py)). The continuous LK/PnP poses added here are not original-model outputs. Their publication rate is neither the original network's inference rate nor evidence that the original architecture meets the real-time target.

**Status: sustained accurate 20 Hz tracking, 2–5-second map updates, 30-minute endurance, physical-camera latency and 8 GB operation have not passed validation.** The continuous RGB replay below is a custom-frontend diagnostic, not original-architecture acceptance or a completed real-time SLAM system. The current hybrid mapper uses CPU/XNNPACK for most of VGGT and RKNN for the original four-iteration CameraHead. It is not an all-NPU implementation.

可复用的问题、根因证据和修复边界整理在[中文工程记录](docs/ENGINEERING_NOTES.md)，包括严格权重加载、ONNX 导出、RKNN 优化器与许可边界。

## Fresh-observation learned VO research (v30)

An optional DPVO CPU research replay now processes the first **192 consecutive TUM fr1_xyz RGB frames entirely on RK3588S**, preserving the original learned frontend's motion admission, eight-view initialization, recurrent graph and keyframe deletion. It uses original CPU ORT encoders, the original Torch Update at changing graph sizes, native CPU correlation and the online BA equations. It adds no ORB/LK/PnP pose frontend and leaves the default VGGT mapper unchanged.

The 16 GB board produced **179 fresh-observation poses**, removed 169 historical keyframes and finished with 17 retained frames and 23,520 active edges. Replay took **699.765 s** for 6.368 s of source video; fresh output averaged **0.256 Hz** and software processing P95 was **6.655 s**. Separate EOF optimization took **71.088 s**, and peak process RSS was **1,369.723 MiB**. This is an unpaced CPU baseline, not accepted real-time, camera, NPU, dense-mapping, endurance or 8 GB performance. [Board provenance, failed startup, complete lifecycle and cleanup audit](results/dpvo_stream_v30.json).

Post-run GT analysis reports **1.085 cm Sim3-aligned ATE RMSE** for the 179 online poses and **0.969° rotation RPE RMSE** on 149 of 162 predeclared one-second endpoint pairs. The separately aligned EOF reconstruction covers all 192 observations and has 1.066 cm ATE RMSE; reconstructed observations do not count as fresh. This short prefix uses GT scale alignment and has no independent original CUDA or full-VGGT trajectory baseline, so it does not establish metric online scale or the accuracy-loss target. [Separate trajectory populations and coverage](results/dpvo_trajectory_v30.json).

The [implementation and reproduction contract](deploy/rk3588/validation/dpvo_stream/README.md), [88 passing targeted synthetic checks](results/synthetic_tests_v30.json) and engineering records 31–35 document the new lifecycle, cache index domains, timing definitions, startup manifest fix and thermal confounding. No model weights or dataset files are published.

## Model execution and SLAM backend (v21)

`ModelExecutor` separates hardware-specific inference from submap construction and SL4 optimization. It receives the already-preprocessed CPU RGB batch and returns canonical camera matrices, depth and confidence. The original `TorchVGGTExecutor` owns device transfer, the model call and pose decoding; `Solver` uses this interface for both normal prediction and loop verification. Another NPU can implement the same contract without exposing Torch parameters. Unsupported requests and device errors must fail explicitly.

The [execution contract](ARCHITECTURE.md) defines frame order, image geometry, output ownership and loop capabilities. It preserves the original model and 0.95 loop decision threshold. SALAD and Submap image storage still use Torch, and a full RKNN executor or a second-vendor backend has not been implemented. Existing measured NPU subnetworks remain separate experimental paths.

On RK3588S, four original FP32 forwards compared direct and adapter calls for normal and similarity requests on the same rectified TUM pair. All four canonical output arrays match **bitwise** in both comparisons; normal predictions also match the saved original reference. The normal adapter call goes through the actual `Solver.run_predictions` entrypoint, preserving source IDs and stored pixels. Its first SL4 graph matches the direct reference: two nodes, two factors and identical homographies. The original similarity ratio is **1.008865714**, demonstrating why it must not be clamped to a probability. This was a similarity request, not a natural revisitation or accepted loop-graph test. Calls still took approximately **4.6–5.1 seconds** at actual 84×112 input; this refactor demonstrates compatibility, not acceleration or real-time acceptance. [Board regression and saved-array checks](results/model_execution_v21.json).

The [v21 synthetic receipt](results/synthetic_tests_v21.json) records **219 passed, zero skipped** across execution, loop-contract, calibration, geometry and legacy-entrypoint checks. The board regression helper is `deploy/rk3588/validation/check-v21-model-execution-board.py`; its `--help` lists the required frozen-source and reference-capture hashes. Historical v20 source pins remain tied to their original revision; use that revision for its optional source-hash check, or the portable evidence sidecar for scalar recomputation.

## Measured baseline

On the tested 16 GB RK3588S, original CPU FP32 VGGT with four A76 threads and two office images took 3.831 seconds at a requested image size of 112 (actual input 56×112). Its camera head took about 1.250 seconds in the separate hooked run. Size 84 took 3.161 seconds for the full forward; its camera head remained about 1.268 seconds. These are short measurements, not sustained throughput or latency percentiles; they exclude image loading, graph optimization and publication.

An original-weight DINO block with input `[1,37,1024]` executed using RKNN 2.3.2: its last three complete I/O calls averaged 5.489 ms. Four separate SDK profile calls reported all 14 internal operators on NPU and only input/output operators on CPU. Its relative L2 error against PyTorch FP32 was 1.2253%, so full-network pose/depth accuracy remains unvalidated. A CPU-Softmax/Erf variant is a mixed implementation and is not the all-NPU result.

[Measurement evidence](MEASUREMENTS.json) records receipt hashes and the measured operator targets.

The **complete original CameraHead** also ran separately through RKNN, retaining all four refinement iterations and 16 trunk-block calls. Its first complete I/O call took 185.979 ms and the next three averaged 203.646 ms; three FP32 CPU reference calls averaged 1.254 s. Actual SDK profiles show 207 internal NPU operators and one final CPU Reshape, plus CPU input/output. Relative L2 error against FP32 is 0.7669%; normalized quaternion differences are 0.778° and 0.869° for the two frames. The strict elementwise comparison does not pass. SDK DMA allocation is 3,424,231,424 bytes. That unrolled variant was not integrated into the replay mapper; the shared-step version described below now runs inside the mapper. [Timing, numerical error and provenance](results/camera_head_v15.json) are published separately from the whole-pipeline results.

The full model and the separate custom continuous tracker still require further work. Dataset timestamps, startup, lost frames, latest-slot replacement and EOF work must remain visible in the results; high input rate alone is not a valid-pose rate.

## Historical custom-frontend replay experiments

The custom LK/PnP/triangulation frontend has run with the VGGT mapper on the 26.572-second TUM RGB sequence. This adds continuous pose estimation to the upstream architecture. The following complete-run results retain startup and losses, but their pose counts and latency do not measure original VGGT-SLAM 2 throughput or establish its real-time acceptance:

| Run | New valid poses / 798 RGB frames | First valid pose | Successful-pose latency P95 | Map update time |
| --- | ---: | ---: | ---: | ---: |
| CPU FP32, 480-wide/800 features | 0 / 798 | unavailable | unavailable | 7.71–8.03 s |
| CPU FP32, 320-wide/400 features | 251 / 798 | 17.34 s | 90.19 ms | 7.23–7.88 s |
| XNNPACK FP16 CPU, 320-wide/400 features | 397 / 798 | 5.49 s | 90.55 ms | 3.85–5.93 s |
| XNNPACK + background depth continuation (r4) | 532 / 798 | 5.59 s | 101.35 ms | 3.51–5.25 s |
| Same, official frontend calibration diagnostic (r5) | 576 / 798 | 5.49 s | 89.83 ms | 3.59–5.11 s |

Runs r2–r5 replay every third retained historical observation when depth arrives; online observations still run PnP individually. This is not an equivalent-numerics optimization. The original XNNPACK run (r3) has a 6.27-second gap in valid pose publication. Background continuation in r4 moves delayed observation catch-up off the tracking owner thread: online adoption takes 6.14–11.31 ms, and there are no lost delivered observations after initialization in this short sequence. Its successful-pose P95 still exceeds 100 ms, and some subsequent full-second windows contain fewer than 20 valid poses.

The r5 diagnostic supplies the official TUM camera intrinsics and distortion to the **local frontend only**. It has 20–31 valid publications in each complete one-second window starting at second 6; the first five full windows have zero poses, and second 5 has 16. The mapper still receives original distorted RGB and estimates its own intrinsics. Therefore no single projective matrix connects this calibrated frontend to the unchanged mapper, and the runner explicitly reports global association as unavailable. These local timing results do not establish a connected, accurate global SLAM system.

Mapper peak RSS is approximately 7.0 GiB in these XNNPACK runs; this excludes the OS and is not a whole-board simultaneous memory measurement. No loop closure was enabled. [Machine-readable results](results/tum_fr1_xyz_v15.json) retain complete-run coverage, per-second counts, losses, calibration scope and source hashes. Physical camera latency, accuracy against a full-quality baseline and 30-minute endurance remain unvalidated.

Depth geometry was already 5.09–7.58 seconds old when the frontend applied it, including historical replay and continuation. Map processing time therefore differs from usable-geometry age. [The saved-log review](results/background_continuation_v15.md) details queue bounds, adoption times and EOF accounting.

Ground-truth diagnostics show substantial remaining geometric error. On the 256 identical one-second endpoint pairs available in both runs, rotation RPE RMS is 11.367° for r4 and 3.635° for r5; translation-direction median error is 55.35° and 6.19° on the 253 pairs with sufficient ground-truth translation. This is a local relative-pose diagnostic: initialization, selected map anchors and depth geometry differ between runs. It does not establish metric ATE, global continuity or accuracy within 10–20% of the full-resolution baseline. [Evaluation results](results/trajectory_v15.json) retain coverage, association rules and input hashes.

## NPU CameraHead integration and custom-frontend replay (v16)

In this custom-frontend experiment, capture and VGGT receive the same calibrated, rectified RGB pixels. These replay changes are not part of the original frontend; the original-weight NPU CameraHead measurements are reported separately below. Rectification preserves the original K and image extent, with identity rotation and no crop. Capture decode/remap runs after the original scheduled arrival and is included in pose latency. The mapper retains original VGGT preprocessing through a lossless in-memory PNG bridge. Both processes verify identical calibration metadata. Projective association is restored; a general SL4 transform still does not define a metric global SE3 pose.

The capture source keeps one decoded frame and selects the latest already-due source image before decoding when it falls behind. It records skipped-before-decode and overwritten-after-decode separately. Original timestamps remain unchanged. This removed the multi-second backlog observed in r6.

| Run | Fresh valid poses / 798 | First valid pose | Successful-pose latency P95 | Map processing |
| --- | ---: | ---: | ---: | ---: |
| r6, shared rectification, old decoder | 595 | 5.82 s | 3667.42 ms | 3.65–5.30 s |
| r7, latest-due decoder, CPU/XNNPACK | 515 | 5.64 s | 139.65 ms | 3.62–5.18 s |
| r8, capture shares A76 core with mapper | 258 | 16.11 s | 111.85 ms | 6.15–9.62 s |
| r9, NPU head, 2 mapper cores, 400 features | 432 | 7.19 s | 121.47 ms | 4.74–6.13 s |
| r10, same allocation, 300 features | 511 | 6.92 s | 103.65 ms | 4.63–6.05 s |
| r11, same allocation, 240 features | 567 | 6.55 s | 93.94 ms | 4.59–5.90 s |

In r11, complete publication windows from second 7 through second 25 contain **22–31 new-observation poses each** from the custom LK/PnP frontend, not from continuous VGGT inference or the original VGGT-SLAM 2 architecture. The first six complete seconds contain zero poses; second 6 contains 14. All 798 original images were decoded, 36 were overwritten in the latest slot, and 762 reached tracking. The 567 valid poses have model/global **projective** associations; rigid global poses remain unavailable. Six map jobs executed 24 actual RKNN calls. EOF depth adoption does not count as a fresh pose. These are results for one 26.572-second sequence, not sustained 20 Hz acceptance, physical-camera testing, or a completed SLAM accuracy result. [Complete coverage, timing and memory samples](results/rectification_v16.json), [independent replay audit](results/replay_review_v16.json) and [trajectory diagnostics](results/trajectory_v16.json) include the slower and failed-target configurations. Temperature and scheduling differed across runs, so these are not controlled causal speedup measurements.

The shared CameraHead graph retains all original weights and all four refinement iterations, executing the same context four times with CPU FP32 recurrence. RKNN file size is 434,452,453 bytes; SDK DMA is 869,425,152 bytes, down from 3,424,231,424 for the unrolled graph. The isolated complete head averages 212.707 ms over its last three calls; relative L2 versus the original CPU reference is 0.7873%. Actual profiles show 57 internal NPU operators and two CPU Reshapes per step, plus CPU I/O. [Operator, recurrence and provenance audit](results/camera_step_v16.json).

The production module verifies all 69 original CPU parameter tensors against the capture/export metadata, installs the shared NPU head before packing remaining CPU Linear layers, and aborts on unsupported inputs or SDK errors without a CPU fallback. On three identical TUM input pairs, the integrated head takes 0.191–0.200 s, while the **whole forward still takes 4.15–4.90 s** with four CPU threads. Depth relative L2 differences from original FP32 at the same resolution are 1.58%, 1.81% and 6.46%; pose-encoding differences are 1.04%, 1.38% and 6.20%. These tensor errors are not trajectory errors. [Fixed-input whole-model results](results/hybrid_pairs_v16.json).

### Resolution accuracy remains a blocker

The first three exact r7 mapping pairs were also run through the original CPU FP32 model at 112 and 518, using the same source images, rectification and weights. Actual tensors are 84×112 and 392×518. Rotation error against TUM ground truth is **5.33–5.85° at 112 versus 0.95–1.09° at 518**; translation-direction error is 9.59–94.21° versus 0.61–2.39°. Even after one positive depth scale per two-frame pair, depth differences remain 18.73–20.38% relative L2. This is a three-pair diagnostic, not a full trajectory or depth-ground-truth evaluation, but it shows that 112 cannot currently support the planned accuracy claim. Original 518 forward times were 142–193 s per pair. [Exact paired-resolution evidence](results/resolution_comparison_v16.json).

No loop closure, 30-minute endurance or 8 GB acceptance is established. Mapper peak RSS in these hybrid runs is about 7.0 GiB, with SDK DMA reported separately; RSS and DMA must not simply be added as if disjoint. Sparse Linux memory/temperature samples are observations, not a simultaneous peak-memory or thermal qualification.

## Higher-resolution diagnostics (v17)

The same three original TUM pairs now have CPU FP32 results at 224 and 336 as well as 112 and 518. At 336 (actual 252×336), forward time is **35.06–57.02 s** and rotation errors are **2.204°, 0.463°, 2.611°**, versus **1.038°, 1.093°, 0.946°** at 518. A single positive scale per two-frame pair reduces depth differences from 518 to **8.03–8.79% relative L2**. These are paired model-output diagnostics, not full trajectory or ground-truth map accuracy. [224 results](results/resolution_224_v17.json), [336 results](results/resolution_336_v17.json).

An independent source/array audit found no H/W or focal resize error in the reviewed path. Across the 24 saved camera outputs, intrinsic reconstruction from the original FOV encoding agrees to a maximum relative difference of 9.13e-8. At 336 the predicted original-pixel focal length remains 1.37–1.58 times the known calibration; the narrow FOV is already present in the original network output. Changing the focal conversion by an arbitrary resolution factor is therefore unsupported. The specific cause of this resolution sensitivity is not established. [Source and FOV evidence](results/fov_source_audit_v17.json).

A board-only 336 stage capture recorded all 24 DINO, 24 frame-attention and 24 global-attention calls. Its uninstrumented forward took 34.07 s; the subsequent instrumented forward took 50.83 s, including hooks/copies and a different thermal state. The latter contains 12.25 s of DINO blocks, 14.30 s of frame blocks, 15.92 s of global blocks and 6.10 s of depth-head work. These nested timings cannot be summed indiscriminately or treated as a controlled speedup comparison. Capturing the first blocks did not change the complete pose/depth/confidence outputs. [Actual shapes, source hashes and thermal samples](results/profile_336_v17.json).

New isolated aggregator exporters retain all 18 original parameters, four normalization operations and the captured 2D RoPE tables. The first ONNX representation encountered an RKNN Split/Concat fusion error; R2 uses an equivalent fixed Gather/sign representation. For the global block, R3 only normalizes two rank-four Gather axes from -1 to 3 to avoid a compiler transpose-optimization failure. It preserves all parameters and other nodes.

Both first-block exports passed the board CPU ONNX Runtime check against the saved original FP32 outputs, with predefined `rtol=1e-4, atol=1e-5`. Each compiled RKNN graph then completed four normal SDK calls and four separately instrumented calls on the board. The last three normal complete I/O calls averaged **183.626 ms for frame0** (`[2,437,1024]`) and **187.139 ms for global0** (`[1,874,1024]`). Relative L2 errors against original FP32 are **0.183794% and 0.435856%**. These FP16 outputs do not pass the strict FP32 elementwise gate. These are isolated first blocks, not a chained aggregator or full network; accumulated pose/depth error, sustained performance and whole-system real-time operation remain unvalidated. [Frame-block audit](results/frame_block_v17.json), [global-block audit](results/global_block_v17.json).

Original CPU FP32 VGGT reference predictions have been captured at 112 and 518 for all six fixed input pairs recorded in r11, using the same source images, rectification and original weights. These are offline model-output quality references, not a completed trajectory or map-quality evaluation. Two-frame model batches do not reproduce the paper's 32-frame benchmark. [Capture provenance](results/quality_pairs_capture_v18.json).

## Original-model cache and layout experiments (v19)

The existing exact-image DINO cache was tested on the same three overlapping, rectified TUM pairs at 336 (actual 252×336). Six image inputs required four image feature computations and produced two cache hits. All three complete pose/depth/confidence outputs were **bitwise equal** to their saved CPU FP32 references. The one-entry feature cache peaked at 1,769,472 bytes and released its contents on close. Full forwards still took **35.11, 45.24 and 49.98 seconds**. Different thermal/runtime states prevent treating comparison with the earlier runs as a controlled speedup. This board-only diagnostic does not establish online integration or real-time performance. [Cache evidence](results/patch_cache_v19.json).

Spelling the two frame-block Gather axes as positive indices produced the same eight NPU outputs as the prior export. Its three warm complete-I/O calls averaged **183.415 ms**, compared with the earlier **183.626 ms**; actual profiles retained 32 internal NPU operators and two CPU I/O operators. This experiment found no confirmed improvement. [Frame-axis comparison](results/frame_axis_comparison_v19.json).

Reshaping the global-block MLP to two-dimensional Gemm also passed the original FP32 board ORT gate, but compiled back to the same NPU kernels. Its eight saved NPU outputs were bitwise equal to the previous export. Three warm complete-I/O calls averaged **187.907 ms**, versus **187.139 ms** previously; internal memory increased by 1,752 KiB. This candidate has no measured benefit and is not integrated into the mapper. [MLP layout comparison](results/mlp_layout_v19.json).

Original post-normalization/RoPE attention operands were captured without changing the CPU block outputs. The global attention has 16 independent heads and 874 complete key tokens. Splitting only the heads into 6/5/5 preserves the attention equation; the full graph and concatenated shards produced bitwise-equal board CPU ORT outputs and passed the original SDPA FP32 gate. [Operand-capture evidence](results/attention_capture_v19.json).

The R3 standalone attention probe explicitly packs FP32 inputs into the SDK's NHWC layout. Four normal and four separate profile rounds per mode produced **16 bitwise-equal NPU outputs** across the full-head and three-context variants. Relative L2 error against original CPU FP32 is **0.109052%**, with maximum absolute error **0.0211375**; **332,056 of 894,976 elements fail** the original `rtol=1e-4, atol=1e-5` gate. The last three normal rounds averaged **123.470 ms** for the full-head graph and **73.856 ms** for the 6/5/5-head variant. These times include both the initial NCHW preparation and explicit NHWC copies, SDK input/run/output-copy/release, worker submission/joining and head concatenation. They are short fixed-input measurements of one attention operation; they establish neither whole-block/full-network speed nor sustained throughput or pose/map accuracy. [Numerical results, actual SDK profiles and timing scope](results/attention_v19.json).

The earlier R2 transport attempt returned API success despite SDK log errors rejecting NCHW input normalization, and every output was zero. Its 96.8/42.0-ms durations are **invalid as acceleration evidence**. The failed receipts and error-log hashes are retained in the attention audit; R3 fixes the actual buffer layout rather than changing the numerical tolerance.

## Complete-block attention integration (v20)

The three-context attention path was connected to the **original complete global0 block**, with all 18 original parameter tensors, QKV, normalization, RoPE, projection, MLP and residual operations retained. Four CPU/hybrid pairs ran in alternating AB/BA order on the board using the same captured TUM input. After the first pair, mean whole-block time was **631.042 ms on CPU and 549.810 ms with NPU attention**, a 12.9% reduction in this short experiment. The attention portion was 142.734 and 59.115 ms respectively. Hybrid timing includes live QKV conversion, both input packing steps, worker scheduling, synchronous SDK calls, output copies and concatenation. Hashing, error calculation, file writes and context initialization are outside these forward times.

All four CPU block outputs match the original reference bitwise. All four hybrid block outputs are stable, and their attention outputs match the earlier standalone NPU result bitwise. Complete-block relative L2 error is **0.014927%**, maximum absolute error is **0.00403687**, and **6,743 of 894,976 elements still fail** the unchanged FP32 `rtol=1e-4, atol=1e-5` gate. All 12 synchronous NPU calls released their output buffers; all three contexts were destroyed and the original SDPA callable was restored. Native SDK log errors are treated as failures even if C API return codes indicate success. This is one isolated block, with no full-network, trajectory, sustained-throughput or P95-latency acceptance. [Independently checked arrays, timings and lifecycle evidence](results/hybrid_block_v20.json).

Static arithmetic explains why attention-only acceleration is insufficient: for two 252×336 inputs, the original 24 DINO, 24 frame and 24 global blocks require **866.91 GMAC** (1.73382 trillion arithmetic operations when one MAC counts as two). QKᵀ/AV attention accounts for about **8.66%** of this count; the MLPs account for about **60.9%**. Twenty overlap-plus-fresh pairs per second would require 34.68 trillion arithmetic operations per second for these blocks alone, excluding the camera and depth heads. This is an architecture budget, not measured hardware FLOPs or an absolute impossibility proof. One repeated overlap image must not be counted as a second fresh online pose.

The [capacity audit](results/capacity_v20.json) also counts the camera and depth heads, giving 963.206 GMAC per pair. The pruned original FP32 parameter payload is 4.314 GiB; the additional FP16 copies reported for an earlier optional per-Linear backend are separate from the default CPU path. Under the current dense Submap retention policy, 30 minutes of two-image 252×336 map updates every 2–5 seconds would retain approximately 4.40–1.76 GiB of RGB/XYZ/color/confidence payload, before graph, allocator and export workspaces. This is a storage projection, not a measured memory peak or endurance result. The portable calculator is `deploy/rk3588/validation/capacity-v20/calculate_capacity.py`; its bundled evidence sidecar permits scalar recomputation without weights or private receipts.

The reproduction entrypoint is `run-v20-aggregator-attention-board.py`: use the v19 parallel model/capture/export arguments plus the original `--profile-dir`, `--profile-sha256` and `--block global0`. It uses the frozen R3 transport and three persistent contexts. The offline `audit-v20-hybrid.py` checks the saved arrays using NumPy without importing a model runtime. Their `--help` output lists the required paths. The measured attention-only hybrid is a diagnostic and is not installed as the production mapper.

A source review identified [Co-Me](https://github.com/co-me-tokens/CoMe) as the next approximation research candidate. Its official loader targets VGGT-1B, and its token merging reduces MLP work in the last nine DINO blocks and all 48 aggregator blocks. It adds a trained confidence predictor and changes token computation; its equivalence to original VGGT is not claimed. The published CUDA implementation and GPU speed results do not establish Rockchip support or 20 Hz. The next proposed experiment isolates mean-merge/MLP/split on one original-weight block before considering whole-model changes. At v20, no Co-Me checkpoint had been downloaded or production backend changed. The v22 research below evaluates the separately released predictor and MLP-only approximation. [Pinned primary sources, checkpoint metadata, limitations and experiment proposal](results/acceleration_research_v20.json).

## Original-weight MLP token-merging research (v22)

A CPU port of Co-Me's separately released BF16 confidence predictor selects 3×3 patch groups from the original DINO block[14] output. This experiment averages only the original MLP inputs and repeats their outputs; original attention, LayerScale, per-token residuals, camera and depth heads are retained. It changes the model's computation and is **an approximation, not an equivalent cache or a complete Co-Me implementation**. The helper is confined to validation scripts. The Co-Me adaptation retains its own CC BY-NC 4.0 scope; neither its predictor checkpoint nor Meta's weights are redistributed. [Pinned source and predictor provenance](results/come_sources_v22.json).

On one fixed rectified TUM pair at actual 252×336, the strict scene-median rule merged 48 of 96 groups, reducing MLP rows from 874 to 490. Four original/four merged measurements in ABBA order, after one warmup each, averaged **187.482 ms and 116.949 ms** for norm2 output through MLP, split, LayerScale and the full residual. This excludes attention and norm2 itself. The separately measured BF16 predictor took **523.151 ms**, plus 11.611 ms for mask/plan construction; its conversion and loading times are also recorded. This is a CPU experiment, with no new NPU execution claim.

All original replays, retained patch rows and special tokens were bitwise equal. Merged computation changed the MLP output by **14.903% relative L2** over all tokens; the reconstructed block output changed by **5.399%**. All merged and unmerged regions are scored, with no confidence or border filtering. The original full-model capture and isolated probe passed independent saved-array audits of 92 and 161 checks. These checks establish provenance and arithmetic, **not pose/map accuracy acceptance**. [Full regional errors, timing scope and artifact hashes](results/come_mlp_v22.json). The portable saved-array auditor is `deploy/rk3588/validation/audit-v22-come-mlp.py`. The [v22 synthetic receipt](results/synthetic_tests_v22.json) records **200 passed, zero skipped** across the new research contracts and the existing model-execution/solver tests.

Applying this same 48-group plan to the last nine DINO MLPs and all 48 frame/global MLPs would remove 183.610 GMAC, approximately **19.1% of the previously counted full-model work**, before adding predictor and merge/split overhead. This arithmetic projection is not a measured whole-model speedup or a real-time claim.

The complete MLP-only research wrapper then ran **eight full-model calls** on three predeclared overlapping TUM pairs, in ABBA/AB/BA order (A=original, B=merged). Each B call generates a fresh scene plan and executes exactly 9 DINO, 24 frame and 24 global MLP branches. All four A calls reproduce all five saved original prediction arrays bitwise; the two repeated B calls for pair 1 also match bitwise. Original weights and forward functions are restored after each B call. This remains a separate validation driver, with no default-executor change.

| Fixed pair | Original forward seconds | Merged forward seconds | Raw depth relative L2 vs original | Relative rotation difference | Translation-direction difference |
| --- | --- | --- | --- | --- | --- |
| 1 | 34.716 / 57.224 | 39.654 / 45.353 | 9.144% | 0.497° | 40.555° |
| 2 | 60.473 | 51.257 | 9.174% | 2.376° | 8.146° |
| 3 | 62.409 | 51.197 | 17.231% | 3.716° | 7.854° |

Every depth comparison includes all 169,344 pixels from both views. Fitting and scoring one shared positive depth scale on those same pixels gives descriptive relative L2 differences of 9.048%, 9.097% and 11.384%; this is not held-out accuracy. The pose columns compare pair-relative model outputs, not ground truth or trajectory ATE/RPE. Pair 2 merges 47 groups (498 rows), while the other pairs merge 48 (490 rows); BF16 median ties are retained rather than forcing a quota.

Forward timing includes BF16 prediction, fresh plan construction, merge and split. Separate context installation/restoration verifies all original parameter bytes and costs about 6.6–6.7 seconds per B call; these research checks, preprocessing, pose decoding and file I/O are outside forward timing. A single read-only snapshot during the run recorded big-core temperatures of 87.666°C; frequency and thermal conditions were not controlled, so the table establishes neither a stable nor a causal speedup. Peak process RSS was 5,562.22 MiB on the tested 16 GB board, excluding OS and map growth; this is not an 8 GB or endurance result. The additional independent scene audit passed 353 checks, including BF16 bit patterns, scene masks, all token mappings and original special-token boundaries. [All calls, numerical comparisons, lifecycle and environment evidence](results/come_fullmodel_v22.json).

**The MLP-only approximation is not adopted as the default.** It still takes tens of seconds, changes pose/depth predictions and has not passed the complete accuracy, online latency or endurance targets. Reproduction uses `run-v22-come-fullmodel-board.py`; saved-output and scene auditors are `compare-v22-come-fullmodel.py` and `audit-v22-come-fullmodel-scenes.py` under `deploy/rk3588/validation/`. `--help` lists the required arguments; frozen pins are recorded in the source and result receipts.

## Original MLP execution and compression research (v23)

The original global block's MLP now has a standalone compiled execution probe. It retains all 4,096 hidden channels and all 874 normalized input rows; no token merging is used. A full graph is compared with three independent hidden-channel slices on NPU cores 0/1/2. The partial graphs omit FC2 bias; the CPU sums their owned FP32 outputs, adds that bias once, and applies the original LayerScale and per-token residual. Contexts and their constant weights persist across calls. Both graph representations pass the board CPU ONNX check against the saved original MLP over all 894,976 elements (`rtol=1e-4, atol=1e-5`).

| Representation | Full graph, mean ms | Three slices, mean ms | Full / sliced MLP relative L2 vs original FP32 |
| --- | ---: | ---: | ---: |
| R1 Gemm, internal CPU Transpose and Erf | 312.686 | 140.663 | 0.04324% / 0.03973% |
| R2 rank-three MatMul/Add, NPU ConvExGelu | 71.538 | 81.065 | 1.29281% / 1.29268% |

Each mean uses four calls in an interleaved ABBAABBA schedule, after one warmup per variant. Times include input preparation, SDK transfers, worker synchronization, FP32 reduction/cropping and original LayerScale/residual; model initialization, error analysis and file writes are excluded. Separate actual SDK profiles confirm that R1 has internal CPU work, while R2 has six internal NPU operators and CPU I/O. CPU reduction and residual work remain outside the compiled graphs. This is a short, fixed-input experiment with read-only thermal/frequency samples, not sustained throughput or latency percentiles.

**Hidden-channel splitting is not adopted:** it gives no net timing benefit for R2 in this run, and none of the FP16 variants passes the strict FP32 elementwise gate. R2's reconstructed block error is 0.43515% relative L2 for the full graph and 0.43513% for the slices; whole-network pose/depth accuracy remains untested. SDK DMA is 61,460,480 bytes for the full context and 72,450,048 bytes across the three slice contexts. All four coexist during the comparison (133,910,528 reported bytes); this does not establish the memory needed for all model layers. [Measurements, operator profiles, failed first ORT setup and independent checks](results/mlp_execution_v23.json). The export/run/audit helpers are under `deploy/rk3588/validation/`; [81 synthetic tests passed, zero skipped](results/synthetic_tests_v23.json).

The [CameraHead bottleneck audit](results/camera_bottleneck_v23.json) finds that its FC1 ConvExGelu operations account for 42.02% of profiled operator time and execute on one core. The tested SDK does not accept a direct `conv_eltwise_activation_fuse` configuration switch. Disabling GELU recognition instead retains CPU Erf; it is not an established all-NPU fix.

[New primary-source research](results/acceleration_research_v23.json) identifies DA3-SMALL as a complete pretrained depth/pose alternative worth a separate experiment. Its inspected safetensors header contains 34,299,463 FP32 parameters and describes a 137,248,940-byte file; the model card's “0.08B” does not match that file. At v23, only metadata was downloaded, and no replacement model had been run or installed. Its code and this specific model card separately declare Apache-2.0; it is a different model, not a license or weight update to VGGT. QuantVGGT's released reference path quantizes the aggregator, leaves the DINO backbone and output heads unchanged, and requires additional work to implement its dynamic scales and low-bit operations on RK3588. Neither candidate establishes 20 Hz or the required accuracy. The production model and execution interface remain unchanged.

## Complete DA3-SMALL model research (v24)

The complete, separately pretrained DA3-SMALL checkpoint now runs on the RK3588S CPU using its unmodified official network. This is a model-substitution experiment, not acceleration of the original VGGT weights, and the default executor remains VGGT. The 137,248,940-byte checkpoint matches its pinned official SHA256. Its 437 stored tensors cover all 34,299,463 parameters; six additional state names are declared shared LayerNorm aliases. The loader verifies those exact aliases against both checkpoint metadata and the original shared parameter objects before `strict=True` loading. No distinct parameter is omitted or randomly initialized.

The three fixed TUM pairs retain the exact rectified RGB pixels used by original VGGT336, with one official DA3 ImageNet normalization and no camera/GT conditioning. The official decoder supplies world-to-camera matrices and pixel intrinsics; its pose encoding is not passed through VGGT's decoder. Six complete CPU FP32 calls succeeded. Pair 1 took **2.370 s initially**, then **2.120 / 2.058 / 2.074 s** (mean **2.084 s**); pairs 2 and 3 took **2.194 / 2.128 s**. Repeated outputs are bitwise identical. Peak process RSS was **903.617 MiB** on the tested 16 GB board. These intervals include the complete depth/ray heads, camera decoder, pose-capture hook and official camera codec, but exclude loading, normalization and file I/O. They are not online frame rates, latency percentiles, an 8 GB deployment test or an endurance result.

| Fixed pair | DA3 / VGGT518 rotation error to GT | DA3 / VGGT518 translation-direction error to GT | DA3 depth difference from VGGT518 after one positive scale fit |
| --- | --- | --- | --- |
| 1 | 3.331° / 1.038° | 21.103° / 2.395° | 18.665% relative L2 |
| 2 | 4.426° / 1.093° | 9.755° / 1.222° | 14.175% relative L2 |
| 3 | 3.618° / 0.946° | 7.215° / 0.606° | 15.986% relative L2 |

Depth fitting and scoring use the same 169,344 pixels across both views, without confidence or border filtering. These depth differences are neither held-out accuracy nor sensor-depth error. The pose results cover only three selected pairs, not trajectory ATE/RPE. **DA3-SMALL is not adopted as a qualified replacement:** this cohort has larger geometric errors than the high-resolution VGGT reference, and the CPU run does not meet the timing goal. An independent saved-array audit passed 538 provenance, normalization, output and lifecycle checks; that confirms the experiment, not task accuracy. [CPU measurements, failed initial alias load, weight/dependency provenance and all pair comparisons](results/da3_cpu_v24.json).

The fixed two-view ONNX export now passes board CPU ONNX Runtime checks on all three pairs: depth, confidence, raw pose, decoded extrinsics and intrinsics each satisfy `rtol=1e-4, atol=1e-5`, with zero failing elements. Export adapters cache only official fixed-grid integer positions and FP32 DPT positional constants; learned weights and the original reference network are unchanged and restored. An independent NumPy-only audit passed **101 checks**.

The learned depth/pose graph took **0.955 / 0.920 / 0.914 s** on the board CPU, plus **0.063 / 0.0026 / 0.0010 s** for the official CPU camera codec. Unused auxiliary ray operations may be pruned; loading, normalization, comparison, saving and the SLAM backend are excluded. These single calls do not establish sustained throughput or a controlled speedup over the complete PyTorch network. The initial RKNN build failed in `gather_to_conv` with an index error; disabling that single rule then exposed an out-of-range Gather during compiler constant folding. Disabling the producing `fuse_channel_shuffle_split_concat_to_gather` rule as well then compiled the unchanged ONNX successfully. The resulting mixed execution is measured below. [Export, board ORT, independent checks and failed-attempt evidence](results/da3_export_v24.json).

The complete learned depth/pose graph subsequently ran through the board RKNN SDK: **6 normal calls**, plus **3 separate profile calls**. The first call took **1.138 s**, three repeats averaged **1.075 s**, and the other two pairs took **1.098 / 1.089 s**, including FP32 input packing, SDK transfers, owned output copies/release and the official CPU camera codec. These short fixed-input calls do not show a benefit over the measured board CPU ORT calls; timing conditions and scopes are retained rather than treating that comparison as a controlled benchmark.

Actual profiles report **520 internal NPU operators, 23 internal CPU operators and 4 CPU I/O operators** per call. Internal CPU work includes five Transposes, ten Reshapes, five Resizes, one ScatterND and two Exps. This is **mixed CPU/NPU execution**. The camera decoder's CPU Transpose alone takes about 54 ms in the profile, while two large depth-head CPU Resizes each take about 35–40 ms. Compiling for `rk3588` does not imply that the whole graph runs on NPU.

The strict FP32 elementwise gate fails. Depth relative L2 differences from the same DA3 CPU model are **3.153% / 1.050% / 0.963%**, and raw-pose differences are **0.794% / 0.718% / 1.556%** across the three pairs. SDK DMA allocation is **300,908,544 bytes**; process peak RSS is **404.730 MiB**, a separate measure that does not establish whole-board memory or map capacity. An independent saved-array and SDK-profile audit passed **1,170 checks**, including all 45 canonical output arrays and bitwise normal/profile repeatability. All nine SDK calls completed and both contexts were destroyed; the candidate remains a separate experiment without real-time or accuracy acceptance. [Normal calls, actual operator profiles, numerical differences and lifecycle evidence](results/da3_rknn_v24.json).

Reproduction helpers are `run-da3-small-board.py`, `export-v24-da3-board-r3.py`, `compare-v24-da3-pairs.py` and `audit-v24-da3-export.py` under `deploy/rk3588/validation/`. The exporter retains hash-bound R1/R2 helpers alongside it. `compile-v24-da3-rknn-r2.py` fixes the two compiler-rule exclusions; `run-v24-da3-rknn-board-r3.py` requires its frozen R1/R2 siblings and records normal/profile runs separately. `audit-v24-da3-rknn.py` reproduces the saved-array and profile audit using NumPy only. Use `--help` for the required source, weight and input pins. [115 synthetic tests passed, zero skipped](results/synthetic_tests_v24.json); these tests use synthetic tensors and fake SDK responses, not host inference of the real model.

## Camera-token layout and performance budget (v25)

An RKNN rewrite moved the camera-token slice after a CPU transpose of all 433 tokens per view. Keeping the identical ONNX and additionally disabling `swap_slice_transpose` restores the early slice. The camera transpose now handles only two tokens and takes **0.395 / 0.394 / 0.394 ms** in separate actual profiles, compared with approximately 54 ms previously. The camera fully connected layers already processed only two tokens; the problem was the preceding layout work.

Four groups ran on the board in **A1/B1/B2/A2 order**, each with six calls; A uses the previous two-rule configuration, B adds the third exclusion. Combining the three first-pair warm repeats in each group gives **1.087435 s for A and 1.057424 s for B**, a **2.76% observed time reduction**. Read-only thermal/frequency samples accompany every group, but frequencies and cooling were not controlled. Normal timing includes packing, SDK transfers/copy/release and the official CPU codec; the three separate B profiles are excluded. This is a short fixed-input comparison, not sustained performance or capture-to-pose latency. The independent saved-array audit passed 3,259 checks across all 135 canonical arrays; A/B and normal/profile repeats are bitwise identical on these inputs. The strict FP32 gate still fails. [Full compile, call, profile and numerical evidence](results/da3_layout_v25.json).

The [dependency and operator budget](results/da3_budget_v25.json) explains why this local improvement cannot establish 20 Hz. Pose still depends on all 12 backbone blocks. Removing depth-exclusive rows from the prior compiled profiles leaves about **853–897 ms** of pose/shared row time; setting their CPU time to zero still leaves about **673–675 ms** of NPU row time. These are counterfactual sums holding other rows fixed, not measurements of a pose-only graph or a hardware lower bound. Reducing map frequency alone does not remove the pose backbone cost.

Reproduction uses `compile-v25-da3-rknn.py` and `run-v25-da3-rknn-board.py` in `deploy/rk3588/validation/`; the runner retains its three frozen v24 predecessors and validates the exact compiler source/configuration. [40 synthetic checks passed, zero skipped](results/synthetic_tests_v25.json), covering configuration guards and the portable NumPy-only `audit-v25-da3-rknn.py`. [Engineering notes 9–11](docs/ENGINEERING_NOTES.md) record the layout issue, separate pose/map budgets and the need to preserve `align_corners` interpolation semantics. The original VGGT executor remains the default.

The [next learned-pose research review](results/acceleration_research_v25.json) compares DPVO, TartanVO and Monodepth2. DPVO's released checkpoint metadata covers **3,384,324 inference parameters**, making its CNN and recurrent update a concrete smaller-network experiment. At v25 only bounded archive metadata was read. The subsequent v26 component experiment below downloads the full checkpoint and measures the original neural modules on the board. The official dynamic correlation, scatter, SE3 and bundle-adjustment path requires substantial CUDA porting, and its sparse SE3 output does not directly replace VGGT's dense SL4 submaps. Any probe must disable classical loop closure to exclude ORB and separately measure all dynamic work. No candidate has established the board's timing or accuracy targets; a repository code license is not treated as independently verified checkpoint licensing.

## Lightweight tests

With NumPy, OpenCV and pytest installed, the geometry, queue and calibration tests need no weights or hardware:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_continuous_tracking.py tests/test_fast_tracking.py \
  tests/test_tracking_geometry.py tests/test_edge_runtime.py \
  tests/test_edge_trajectory_evaluation.py tests/test_image_rectification.py \
  tests/test_rectified_mapping.py
```

Torch enables the additional synthetic RKNN lifecycle/contract tests. The [v20 test receipt](results/synthetic_tests_v20.json) records 123 focused synthetic cases, including 15 new complete-block bridge checks; the [v19 receipt](results/synthetic_tests_v19.json) and [v17 receipt](results/synthetic_tests_v17.json) retain the earlier suites at their recorded source hashes. Synthetic SDK calls never establish hardware execution. No host real-image tracking or model inference is used in these checks.

## Setup and external dependencies

The validated deployment uses Linux aarch64, glibc 2.35, Python 3.11.17, PyTorch 2.3.1 CPU, torchvision 0.18.1 and a GTSAM develop build containing SL4. A pinned download manifest and an offline installer are under `deploy/rk3588/dependencies/`. Their downloaded wheels, Python runtime and Debian packages are not committed.

Clone the pinned VGGT implementation and apply the portable patch from the repository root:

```sh
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt
git -C third_party/vggt checkout 6e6e16107b88e8e76c751826af10d4295d87ecd2
git -C third_party/vggt apply ../../deploy/rk3588/vggt-spark.patch
```

Use the official [facebook/VGGT-1B checkpoint](https://huggingface.co/facebook/VGGT-1B). The reference `model.pt` is 5,026,874,952 bytes with SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. `deploy/rk3588/download_weights.py` verifies that checkpoint. Weights are not redistributed here. A similarly shaped StreamVGGT checkpoint is not an interchangeable baseline.

Loop closure is optional and is disabled in the minimal replay command. Enabling it also requires local pinned [SALAD](https://github.com/serizba/salad) and [DINOv2](https://github.com/facebookresearch/dinov2) sources and the local SALAD checkpoint; see `UPSTREAMS.json`. The runtime does not silently download these when loop closure is disabled.

The SDK, `librknnrt.so`, RKNN compiler wheel and Rockchip headers are not included. Obtain the licensed [official RKNN SDK](https://github.com/airockchip/rknn-toolkit2) yourself. The tested runtime/ABI is 2.3.2; users must provide matching `rknn_api.h` and `rknn_matmul_api.h` under the filenames expected by each diagnostic. Header hashes are checked before hardware calls. XNNPACK ABI tests also need the matching external XNNPACK/pthreadpool headers. A fresh public clone therefore does not contain all hardware-test prerequisites.

## Reproducing the custom-frontend timestamped RGB experiment

This section reproduces the optional custom LK/PnP/triangulation frontend through `scripts/run_edge.py`, not the original VGGT-SLAM 2 frontend. Its per-observation pose rate does not establish original-network throughput or real-time acceptance. Obtain the official [TUM RGB-D freiburg1/xyz sequence](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and use its original `rgb.txt`. It contains 798 RGB images over 26.572059 seconds, about 29.994 Hz. Replay uses original intervals; it is software-simulated capture, not a physical camera test. Sensor depth and ground truth are not inputs to the tracker.

Run on the RK3588S from the repository root, in the prepared environment. Use an output directory that does not exist:

```sh
PYTHONPATH=.:third_party/vggt python -u scripts/run_edge.py \
  --rgb-index /path/to/rgbd_dataset_freiburg1_xyz/rgb.txt \
  --checkpoint /path/to/model.pt --output /path/to/new-run \
  --image-size 112 --backend cpu --threads 3 \
  --mapper-cpus 4,5,6 --frontend-cpus 7 --capture-cpus 0,1 \
  --candidate-cpus 2,3 --map-period 3 --bootstrap-gap 0.25 \
  --history-frames 300 --history-seconds 10 \
  --tracking-width 320 --tracking-features 400 --replay-stride 3 --max-loops 0
```

`--backend xnnpack-f16` explicitly selects an experimental CPU backend, not NPU. Queue capacities, original/delivered frame counts, replacements, fresh valid poses, map publications and deferred depth adoption are recorded. Historical catch-up and EOF-only adoption do not count as online new poses. Local tracking poses use model units; the SL4 transform is projective and does not automatically define a metric global SE3 pose. Map capacity and a memory guard are explicit, not proof of endurance.

The measured XNNPACK diagnostics explicitly used `--mapper-memory-limit-mib 8500` on the 16 GB board; the default 6500 MiB guard is too small for their observed mapper RSS. This override does not demonstrate 8 GB operation. The r5 frontend-only diagnostic additionally uses `--local-calibration deploy/rk3588/tum-fr1-local-calibration.json`; it must not be described as a globally calibrated map.

For the shared-calibration runner, use `--rectify-calibration deploy/rk3588/tum-fr1-local-calibration.json`. The r11 allocation is `--threads 2 --mapper-cpus 4,5 --capture-cpus 6 --frontend-cpus 7 --candidate-cpus 2,3 --tracking-features 240`. Its CPU Linear backend is `--backend xnnpack-f16`, with the separate experimental NPU head selected by `--camera-backend rknn-step`. The NPU head requires all four artifact paths and their explicit SHA256 values, as shown in [the v16 reproduction notes](results/camera_step_v16.md); it currently accepts two-frame batches with `--max-loops 0`. A hash, shape or SDK failure aborts the selected backend.

The trajectory evaluator reads saved poses after the experiment; ground truth never enters tracking or model inference:

```sh
python scripts/evaluate_edge_trajectory.py \
  --run-dir /path/to/new-run \
  --groundtruth /path/to/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  --output /path/to/new-diagnostic.json
```

It associates actual timestamps within 20 ms without interpolation and computes anchor-relative errors and one-second RPE within the same local gauge. Translation direction excludes ground-truth movements below 1 cm. It does not stitch projective gauges into a metric ATE trajectory or fit ground-truth scale by default.

## Reproducing NPU subnetwork experiments

`deploy/rk3588/validation/` contains board-only capture/runtime programs and host-only ONNX export/compiler programs. The DINO pipeline is `capture-v12-dino-block-board.py` → `export-v12-dino-block.py` → `export-v12-dino-block-gelu.py` → `compile-v12-rknn-block.py` → `run-v12-rknn-block-board.py`. The small-input versions are `capture-v14-small-dino-board.py`, `export-v14-small-dino.py` and `run-v14-small-rknn-board.py`. Run `--help` for the exact required input, capture and SHA256 arguments.

For the complete CameraHead experiment, use `capture-v15-camera-board.py`, `export-v15-camera-board.py`, `check-v15-camera-onnx-board.py`, `compile-v15-camera.py` and `run-v15-camera-board.py`. Here capture, ONNX export with tracing, numerical checks and actual inference all run on the board; only RKNN compilation runs on the host. The exported ONNX shares 69 original weight tensors, but the compiled model is 1.711 GB. Do not infer its runtime memory use from ONNX weight sharing.

Captures use the official checkpoint and real image. Export preserves original block weights and exact GELU; CPU PyTorch references are produced on the board. Execute normal timing separately from `--collect-perf-detail`, then inspect actual SDK operator targets. Compilation success or a core-mask request does not prove NPU execution. Numerical error and complete input/run/output-copy/release time must accompany every hardware result. No model, ONNX, RKNN, captured tensor or SDK binary is committed.

The v19 attention diagnostic is `capture-v19-attention-board.py` → `export-v19-attention.py` → `check-v19-attention-onnx-board.py` → `compile-v12-rknn-block.py` → **`run-v19-attention-board-r3.py`**. Export and compilation run on the host; actual CPU ORT and NPU calls run on the board. Compile the full graph, the six-head graph and one five-head graph; the last compiled model is shared by two separate runtime contexts. Use `--mode baseline` with one model/compile-receipt/SHA entry, or `--mode parallel` with three ordered entries (six-head, five-head, five-head). The R3 entrypoint needs the frozen R1/R2 helpers alongside it. R1/R2 alone retain failed layout experiments. Run each normal/profile case in a new directory, check the saved numerical errors and SDK stderr as well as API return codes, and never treat an ADB transport exit code as proof that the board diagnostic passed.

The frozen capture drivers also pin their original image and source hashes. For their exact experiment, obtain `office_loop.zip` from the upstream repository separately. Its numbered JPEG files have no original acquisition timestamps; they cannot establish camera input rate. The capture's fixed board layout can be recreated outside this repository at `/mnt/vggt-data/vggt-slam2`.

## Attribution and licenses

The original project is MIT-SPARK's VGGT-SLAM 2.0, based here on commit `35327ac28b7d193df9ccc39ba6346052bb6f1207`. Its BSD-2-Clause license and copyright remain in `LICENSE`; its original README is in `README_UPSTREAM.md`.

Third-party code and model weights keep their own licenses. VGGT_SPARK is CC-BY-NC-4.0, SALAD is GPL-3.0 and DINOv2 is Apache-2.0. The root BSD license does not relicense them or Rockchip's SDK. Dependency sources, commits and license locations are recorded in `UPSTREAMS.json` and `THIRD_PARTY_NOTICES.md`.


## DPVO neural-component porting research (v26)

This is an isolated study of a **different pretrained model**, with the original VGGT executor unchanged. It uses three consecutive TUM RGB frames and the official DPVO checkpoint (94 current inference tensors, 3,384,324 parameters; four explicitly identified legacy tensors excluded as in the official loader). The original CNN and Update class bodies and all loaded parameter bytes are preserved. No ORB or added LK/PnP frontend, GT pose, or sensor depth is used. [Source, checkpoint and input provenance](results/dpvo_sources_v26.json).

The original TUM preprocessing is **BGR**, undistortion, a crop from 640×480 to **608×464**, then `2*(image/255)-0.5`. Both encoders divide their outputs by four. The two CNNs took **0.782 s** on average over three same-frame warm CPU PyTorch calls. Board CPU ORT took **0.293 / 0.289 / 0.277 s** for the three distinct frames; all six feature arrays passed the original FP32 elementwise gate (`rtol=1e-4, atol=1e-5`). The NumPy-only saved-array review passed 120 checks.

The same encoder graph compiles to a **6,065,704-byte RKNN**, but normal complete SDK calls average **426.785 ms** over three first-frame warm repeats. Three independent profiles each show **82 internal NPU operators, five internal CPU Transposes and three CPU I/O operators**. The five Transposes precede high-resolution fnet InstanceNorm operations and total **216.530 / 219.718 / 216.708 ms**. Each call also retrieves and copies **36,110,336 bytes** of FP32 features; packing, SDK transfer, owned copies and release are included in normal timing. Thus this version shows no net gain over board CPU ORT; clocks/cooling were not controlled and this is not a sustained comparison. The strict FP32 gate fails: feature relative L2 differences are **0.350–0.414% for fmap** and **0.090–0.100% for imap**, with no trajectory-quality conclusion. [CPU, export, ORT, compilation, normal/profile and saved-array evidence](results/dpvo_encoders_v26.json).

The CPU port of patch sampling, two-level correlation, neighbor grouping and scatter runs the **entire original learned Update** on actual board-captured features. Its measured warm component times are:

| Fixed diagnostic graph | Correlation including pyramid creation | Original Update | Combined |
| --- | ---: | ---: | ---: |
| 96 edges, first view to second | 710.015 ms | 103.582 ms | 813.597 ms |
| 864 edges, three-view all-pairs | 6,018.016 ms | 276.762 ms | 6,294.778 ms |

These are timings of the current Python/Torch CPU reference, not a lower bound on an optimized implementation. Every call starts from zero recurrent state and identity-projected patch coordinates. CNNs, graph setup and bundle adjustment are excluded; the original online implementation also caches feature pyramids, whereas this probe rebuilds them inside correlation. The outputs are per-edge pixel corrections and weights, **not camera poses**. BA/SE3 updates, tracked graph lifecycle, initialization, sparse-to-dense integration, ATE/RPE, 20 Hz tracking and endurance remain unvalidated. [Dynamic component measurements and independent saved-array audit](results/dpvo_motion_v26.json).

The board helpers and NumPy auditors are under `deploy/rk3588/validation/`, with `--help` exposing source/receipt hash arguments. [Engineering notes 12–15](docs/ENGINEERING_NOTES.md) capture input/weight contracts, custom CUDA indexing, dynamic-stage budgets, and normalization/layout/I/O costs. [124 synthetic tests passed, zero skipped](results/synthetic_tests_v26.json), using tiny synthetic tensors and fake SDK calls; no host checkpoint/image forward is used. Separate saved-array audits passed 720 RKNN checks and 463 dynamic-component checks; these verify evidence consistency, not trajectory accuracy. Tests that construct the original architecture need an external checkout at commit `0ac95b656d1fda91c271d2a106460d19ad966fc7`: set `DPVO_SOURCE_DIR` to that checkout and `DPVO_SOURCE_MANIFEST` to this repository's `results/dpvo_source_manifest_v26.json`. The standalone CPU/ORT auditor also accepts `results/dpvo_input_manifest_v26.json`. Original checkout, weights, image/feature arrays and compiled binaries remain external. DPVO's code is MIT; an independent weight license has not been verified and is not inferred from the code license.

## Native correlation and normalization experiments (v27)

An isolated C++/NEON CPU implementation now evaluates the original two-level DPVO forward correlation on the RK3588S. Four fixed graphs cover 96 and 864 edges, each with identity and fractional shifted coordinates, using the same three captured real frames. Each graph runs four interleaved rounds of the Torch reference, native one-shot preparation/query/destruction, and prepared native queries. All **48 calls** finish; native correlation and the following unchanged original-weight Update pass the existing **rtol=1e-4, atol=1e-5** comparison. Source/parameter checks and context cleanup also pass. This is a numerical comparison to the CPU reference, not CUDA byte identity. [Measurements and provenance](results/dpvo_native_corr_v27.json).

Three warm repeats per graph give these means, in milliseconds:

| Graph | Reference correlation | Native prepare + query + close | Native cached query | Original Update after cached query | Cached query + Update |
|---|---:|---:|---:|---:|---:|
| 96 edges, identity | 756.247 | 51.053 | 11.952 | 80.527 | 92.479 |
| 96 edges, shifted | 691.117 | 51.163 | 9.293 | 89.474 | 98.766 |
| 864 edges, identity | 5,675.967 | 80.952 | 48.681 | 200.200 | 248.881 |
| 864 edges, shifted | 5,557.756 | 82.360 | 48.285 | 192.203 | 240.487 |

Cached preparation costs a separate **18.679–46.435 ms** in these four cases. The context owns its feature snapshots and requires an explicit generation key; changing source features requires rebuilding it. Query timing includes wrapper validation, packing and output allocation. These short interleaved tests exclude CNN inference, graph setup and BA, start Update from zero recurrent state on every call, and do not control clocks/cooling. They establish a useful component improvement but **no pose rate, complete VO or 20 Hz result**. See the [native source, MIT notice and API](deploy/rk3588/validation/dpvo_native/README.md).

A separate candidate expresses all ten fnet InstanceNorm nodes through spatial mean, centered population variance and the original epsilon/affine values. Original parameters, other nodes and I/O remain byte-identical. On-board CPU ORT rejects the candidate at the **unchanged FP32 gate**: fmap has **4 / 5 / 53** failing elements against the original CPU arrays and **10 / 2 / 69** against original ORT; imap passes and is byte-identical to original ORT. Maximum fmap absolute differences against CPU are **1.2621e-5 / 1.2606e-5 / 1.7166e-5**. These are small arithmetic differences, but do not establish trajectory accuracy or satisfy the chosen gate. The candidate was **not compiled or profiled on NPU**. The failed receipt and rewrite helper are retained so this path is reproducible. [Normalization evidence](results/dpvo_norm_rewrite_v27.json) and [engineering notes](docs/ENGINEERING_NOTES.md).

[82 targeted synthetic tests pass, zero skipped](results/synthetic_tests_v27.json), covering native ownership/semantics, an independent NumPy oracle, normalization arithmetic and receipt tampering/rejection. The separate saved-array reviews pass 3,020 correlation checks and 168 normalization evidence checks. The former also tests 24,696 predeclared scalar correlation values against an independent oracle; the latter confirms the candidate's numerical **failure**, not acceptance. These v27 studies use no ORB or added classical pose frontend and leave the default VGGT executor unchanged.

## Complete Update and fixed-graph feedback research (v28)

The original DPVO Update now has an export adapter with 13 explicit neural/topology inputs. Its original parameters and sequential neighbor/group operations are preserved. Four real graph cases, each with four captured recurrent states, pass the unchanged original FP32 comparison through the wrapper and board CPU ORT. Three repeated zero-state ORT calls average **6.4–7.3 ms for 96 edges** and **48.1–52.5 ms for 864 edges**, depending on graph and optimization mode. Timing includes session execution and owned output returns; it excludes topology preparation, correlation, CNNs and BA. Each nonzero ORT input is an original CPU capture, so this is not ORT's own recurrent rollout. Independent saved-array checks pass 5,242 checks. However, some ORT sigmoid weights are slightly negative (minimum **−1.1920928955078125e-7**) despite passing the FP32 error gate. Their use in BA is **not accepted**; outputs have not been clipped. [CPU/export evidence](results/dpvo_update_cpu_v28.json).

Both fixed-capacity graphs compile to RKNN. The first ordinary E864 call fails output validation. A separate one-call diagnostic preserves the raw fault: hidden state is entirely NaN/+Inf, delta is entirely +Inf, and weights are a wrong constant. Actual profiling shows **86 internal NPU and 24 internal CPU operators**, plus 16 CPU I/O operators. Cleanup and input/source checks pass, but the candidate is rejected and the root cause remains unconfirmed. Its instrumented failure timing is not a valid inference or speedup result. [SDK contract, failure, raw-output metadata and independent review](results/dpvo_update_rknn_v28.json).

A separate **CPU** experiment closes four real feedback rounds on the three captured frames and 864 fixed edges: current geometry → reprojection → native correlation → original Torch Update with carried state → target → two CPU BA steps. All four rounds and eight BA steps complete; 214 independent saved-array checks verify state continuity, geometry and residual accounting. No active edge is dropped in these rounds. Total measured round time is **1.169 s**, with Update accounting for 56.6%, BA 31.6% and correlation 7.2%. CNN capture, model loading, one-time cache preparation and file writes are excluded. This uses identity/unit-depth initialization, not official eight-frame initialization, and adds no new frames. It is **not complete online VO, trajectory accuracy or real-time acceptance**. The default VGGT executor is unchanged and no ORB or added LK/PnP frontend is used. [Feedback evidence](results/dpvo_feedback_v28.json).

[Engineering notes 19–24](docs/ENGINEERING_NOTES.md) document topology, SDK types, numerical domains, failure isolation and geometry semantics. The [136 passing synthetic tests, zero skipped](results/synthetic_tests_v28.json) cover tiny random tensors, numerical geometry and fake SDK behavior; all real checkpoint/image inference remains on RK3588S. The [geometry research directory](deploy/rk3588/validation/dpvo_geometry/README.md) explains prerequisites, fixed raw-receipt bindings and MIT attribution. Public redacted evidence is not a substitute for the raw SHA-pinned captures required by these research entrypoints.

## Stable weights and vectorized geometry (v29)

The final Update weight Sigmoid is now expressed as `exp(min(x,0))/(1+exp(-abs(x)))`, preserving the original parameters, other nodes and I/O without clipping. All **16 board CPU ORT saved states** pass both the unchanged FP32 comparison and the independent `[0,1]` weight-domain check. Finite-padding and arithmetic-selection candidates still retain the original Sigmoid's domain violation on three states and remain rejected for CPU use. [CPU evidence and independent saved-array checks](results/dpvo_update_cpu_v29.json).

The CPU BA assembler now uses ordered, batched accumulation, including self-edge Jacobian merging and duplicate-index accumulation. On the same four saved states, old/new first calls and three warm calls alternate; **32 ordinary calls** are followed by eight separate instrumented calls. Two-step BA warm means fall from **72.309–72.569 ms to 13.981–14.058 ms**. The 224 saved scalar/vector tensors are byte-identical, including both steps' normal systems and updates; 3,381 independent checks pass without rerunning BA on the host. This roughly 5.16× ratio concerns only this BA component. [Geometry measurements](results/dpvo_geometry_v29.json).

These CPU changes also complete four real recurrent feedback rounds on the fixed three captured views. Each round recomputes native correlation from current geometry, carries ORT's own hidden state, and performs two BA steps. An additional original Torch Update consumes that round's exact current input and passes the same FP32 gate; it is a one-step reference, not a separate golden recurrent trajectory.

| Four-round measurement | Total |
| --- | ---: |
| Feedback, including checks/copies and geometry diagnostics | 438.817 ms |
| Update component within feedback | 228.592 ms |
| Two-step BA component within feedback | 59.238 ms |
| Correlation component within feedback | 95.842 ms |
| Four separate original Torch references, outside feedback | 530.121 ms |
| Entire research driver, including loading, references and file work | 4.076 s |

The feedback mean of **109.704 ms** excludes CNN inference and new-frame admission. Historical v28 took 1.169 s for four rounds, but these are separate short runs with uncontrolled thermal/frequency conditions and different recurrent arithmetic; no controlled whole-system speedup is claimed. Peak process RSS is **407.148 MiB**, excluding the earlier CNN capture, the operating system and map growth. Identity/unit-depth initialization, fixed topology and same-input neural checks do not establish online VO, trajectory accuracy, 20 Hz, 8 GB operation or endurance. [Feedback and independent review](results/dpvo_feedback_v29.json).

RKNN fault isolation progresses separately. Removing the two negative-infinity padding constants alone leaves all eleven ordinary calls nonfinite. Adding finite arithmetic selection in place of six Where nodes yields finite outputs and bounded weights in **eleven ordinary calls plus one separate profile**, but every call still fails the strict FP32 gate. Both candidates remain unaccepted. The arithmetic candidate profile contains 100 internal NPU and 24 internal CPU operators, plus 16 CPU I/O operators. Output taps expose large selection errors before a nonfinite Exp in the instrumented graph; taps can alter fusion, so they do not identify a unique first faulty kernel in the original graph. These are diagnostic observations, not accepted NPU inference latency. [Raw-receipt provenance, comparisons and profiles](results/dpvo_update_rknn_v29.json).

The default VGGT executor remains unchanged; this DPVO research uses no ORB or added LK/PnP frontend. [Engineering notes 25–30](docs/ENGINEERING_NOTES.md) retain the failed hypotheses and numerical/measurement boundaries. New tests exercise only synthetic tensors, saved-array tampering and fake runtimes; all real inference remains on RK3588S. [Targeted synthetic validation](results/synthetic_tests_v29.json). Geometry adapters and online-equation adaptations retain their MIT notice; no weight, captured array or compiled SDK model is redistributed.
