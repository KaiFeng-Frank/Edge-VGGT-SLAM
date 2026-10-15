# Original native518 DINO0 CPU precision controls

This experimental board test diagnoses one original DINO0 block on its pinned
`[1,1041,1024]` input. It preserves all 14 original parameter tensors before
conversion, original classes, 16 heads, LayerNorm epsilon `1e-6` and Erf GELU.
It compares three distinct controls:

- Original FP32: all five observed points must reproduce saved FP32 bytes.
- Storage rounding only: input/weights round FP32→Half→FP32; execution is FP32.
- CPU Half: fresh original input/weights and observed tensors use Half; an
  unsupported operator would be recorded without a fallback.

All three forwards completed. The original FP32 norm1, QKV, attention,
projection input and final output were byte identical to their references.
The readonly SDPA observer forwarded original arguments and returned the
original result; hooks, source/input/state and process cleanup checks passed.

| Error against original FP32 | Storage rounding, FP32 | CPU Half | Prior NPU |
| --- | ---: | ---: | ---: |
| Whole-output relative L2 | 0.000506622 | 0.000710586 | 0.025482276 |
| Same-original-input residual-update relative L2 | 0.000900344 | 0.001262820 | 0.045285890 |

The prior NPU comparison uses its saved same-input tap payload, whose final
output matched the untapped baseline bytes. No new NPU inference ran here.
The NPU discrepancy is much larger than either CPU control on this input;
these measurements do not identify its internal cause. Half tensor dtype
does not prove the accumulation precision inside Torch or NPU kernels.
Neither non-FP32 control is treated as passing the unchanged strict gate.

Independent saved-data auditing verified ten control NPYs, five original
reference NPYs and all three NPU payload tensors, including file/raw hashes,
shape/dtype/finiteness, attention layout and 26 recomputed comparison records.
The original FP32 forward was not resaved: its observed raw hashes matched
the pinned references. Control arrays total 44,772,608 bytes and remain private.

The worker completed in 20.060 monotonic seconds, peaked at 573.87 MiB RSS and
returned zero. Timers include observers, copies, hashes and file IO; this is
not a deployment benchmark. NPU frequency was 300 MHz before and after;
temperature was not sampled. Board wall time was stale and is not reliable UTC.

Prerequisites are pinned original source, original DINO0 parameter NPZ and
R100 FP32/NPU diagnostic artifacts; none is downloaded or shipped here.
`run_cpu_control_r1.sh` uses board CPU Python, process-local LD/thread settings
and a 180-second timeout. Use fresh source, control and output paths for another run.
No complete encoder/model, speedup, trajectory, realtime, thermal or 8 GB
acceptance follows. Repository code and original model weights retain separate
licenses; the code license does not grant rights to the original weights.
