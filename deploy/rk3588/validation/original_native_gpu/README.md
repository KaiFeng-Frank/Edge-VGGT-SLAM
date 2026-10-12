These v81 diagnostics execute the original VGGT frame-0 MLP on RK3588S: all 106 rows, the original 1024→4096→1024 weights, both biases and the original Erf GELU equation remain. They investigate native RKNN layouts, an onboard Mali operator bridge and FC2 output-channel partitioning. Actual model and operator execution takes place on the board; the host computer GPU is prohibited and was not used. The host performs source inspection and saved-array audits.

The [public result](../../../../results/original_native_gpu_v81.json) contains timings, numerical comparisons, environment provenance and resource receipts. Each successful route retains its first call and measures 12 subsequent calls, including input conversion, copying, synchronization, SDK calls and owned output assembly. Initialization and post-call numerical diagnostics are separate. NPU tests use the authorized 1 GHz, 180-second/75°C controller and restore the prior 300 MHz state. GPU clocks and governor were not changed.

| Independent run and scope | Baseline warm mean | Candidate warm mean | Finding within that run |
| --- | ---: | ---: | --- |
| Native-layout ABC, complete MLP | Normal A/C, core 1: 26.544 ms | Native A/C, core 1: 33.383 ms | Native layout is slower; B/C outputs are byte-identical. Compiled/core 7 is 6.409 ms. |
| GPU ABCD, complete MLP | Compiled/core 7: 6.634 ms | Native/core 1 + GPU GELU: 12.182 ms | The GPU route remains slower; native/core 1 + CPU GELU is 36.962 ms. |
| FC2, three independent input copies | Unsplit/core 1: 11.248 ms | Three single-core parts: 10.230 ms | Complete component time decreases 9.05%. |
| FC2, shared input A | Unsplit/core 1: 10.2965 ms | Three parts sharing A: 8.1557 ms | Complete component time decreases 20.79%. |
| GPU + shared-A FC2, complete MLP | Compiled/core 7: 7.4937 ms | Combined route: 11.8407 ms | The combined route is 58.01% slower and is not adopted. |

These are separate runs with their own baselines. The 12.182→11.841 ms difference is not a measured improvement from composing the optimizations. FC2 component gains also do not establish a complete MLP or full-model gain. In the combined run, warm FC1, GPU bridge and FC2 dispatch/run/owned-output/concatenation stages take 2.9423, 3.1457 and 5.5745 ms respectively.

The initial native MatMul core-mask-7 request returned −1 before any completed inference call. Its three created contexts and six allocated memories were released. The fixed successful-run accounting gate was false because the run failed; this is not evidence of a leak. Subsequent native calls explicitly use individual cores. Compiled/core-7 execution is a separate SDK path and does not establish combined-core support for native FP16 MatMul. A separate SDK-call profile places about 79–82% of the two native MatMul run intervals in FC2, without claiming device utilization.

OpenCL initially had no discoverable library/platform. Loading a pinned **g6p0 userspace library** privately exposed a second failure: the kernel could not find `mali_csffw.bin`. After explicit user authorization, the previously absent firmware was added from the exact **g11p0-01eac0** vendor version matching the board kernel. The userspace library is still g6p0. No kernel, driver or system library was replaced. The original-Erf bridge then ran successfully with DMA-fd imports, checked allocation sizes/offsets, explicit RKNN cache synchronization and blocking OpenCL completion. Imported GPU objects close before their NPU allocations; failures do not silently fall back to CPU.

FC2 retains K=4096 and partitions only the 1024 output channels into 352/352/320, using masks 1/2/4. The first variant pays three input copies and synchronizations. The shared-A variant binds the same owner buffer after one copy and one device synchronization; all 39 part bindings succeed. Both variants' 26 saved outputs are byte-identical. Worker SDK-call overlap is host timing evidence, not proof of simultaneous device execution. Unused part-owned A allocations remain allocated, so physical memory savings are not claimed.

Numerical limits remain: GPU ABCD outputs are stable across all 13 calls per route, but original FP32 strict comparisons fail. On the same NPU FC1 values, GPU GELU differs from CPU GELU rounded once to FP16 at 1685 elements; all 637 strict failures differ by one FP16 ULP. The largest distance is four ULP. GPU-versus-native-CPU complete MLP has one strict failure. These observations do not validate trajectory or map accuracy. Saved receipts report balanced NPU contexts, buffers, OpenCL events and owned DMA fds; the raw arrays remain private.

The launchers reproduce the board paths using pinned v53 captures and v68 references: [native layout](run_native_abc2_1ghz.sh), [SDK-call profile](run_native_profile_1ghz.sh), [GPU MLP](run_native_gpu_1ghz.sh), [three-copy FC2](run_fc2_three_cores_1ghz.sh), [shared-A FC2](run_fc2_shared_a_1ghz.sh) and [combined MLP](run_native_gpu_shared_1ghz.sh). [Source manifests](source-copy-list.json) bind the actual scripts and hashes; [saved audits](native-gpu-saved-audit.json) preserve the numerical limits. Raw receipts reside under board `experiments/vggt-v81-{native-abc-r1,native-abc-r2,native-profile-r1,native-gpu-r1,fc2-three-cores-r1,fc2-shared-a-r1,native-gpu-shared-r1}` and their control directories.

For another run, copy a launcher and select an unused output directory; the
frozen launchers intentionally preserve the executed paths. Invoke the copy
through the existing board frequency controller, whose guard restores the
prior state even when a worker fails:

```sh
/mnt/vggt-data/vggt-slam2/venv/bin/python \
  /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py \
  --execute-frequency-test --frequency-hz 1000000000 \
  --temperature-limit-c 75 --timeout-seconds 180 \
  --launcher /path/to/new-launcher.sh --output /path/to/new-control-directory
```

The [combined-MLP saved audit](native-gpu-shared-saved-audit.json) verifies all
26 timed outputs and five intermediate arrays. Its 13 candidate outputs are
byte-identical to the prior GPU route's saved output; this is a numerical
comparison across runs, not a timing comparison. All five native contexts,
15 allocations, 52 native runs, 13 compiled outputs and 13 GPU events are
accounted for and released. [Clock restoration audit](final-clock-restoration-audit.json).

Production execution remains unchanged. The latest full original-model timestamp replay is v80, about **1.0748 Hz**, with input-to-pose P95 **1.1572 s**. No new full-model, 20 Hz/100 ms, physical-camera, 8 GB or 30-minute acceptance is established here. Original weights and Mali binaries have separate [licenses](../../../../MODEL_LICENSES.md); this directory publishes source and statistics, not weights or vendor binaries.
