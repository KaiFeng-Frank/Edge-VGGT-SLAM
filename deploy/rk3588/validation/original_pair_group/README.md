# Two consecutive original VGGT pairs

This diagnostic composes the original `frame0/global0/frame1/global1` into
one RKNN graph. It retains every original node and initializer and exposes all
four original outputs. The only new operation is the original global0-to-frame1
C-order reshape, `[1,106,1024]` to `[2,53,1024]`. No FC1 tap, token/layer removal,
quantization or production-runtime replacement is involved.

On two saved real TUM inputs, all eight FP32 group outputs match the separate
ONNX chain byte-for-byte. On RK3588S at sampled 300 MHz, the normal FP16-I/O
component means were **81.196 ms separate / 77.027 ms grouped**, excluding the
first call for each input and route (six subsequent calls per route). All
grouped outputs match the separate NPU path byte-for-byte. The original Torch
FP32 strict checks still fail on the input with saved Torch references; the
second input has no saved original four-output Torch reference.

This is a small component improvement, not measured full-model improvement.
The production backend remains unchanged; there is no new replay rate or P95.
All four output payloads still cross the SDK boundary. Only the second input
submission/conversion and one graph call are removed. Larger groups, whole-model
outputs, trajectory/map quality and runtime integration need their own checks.
[Public measured record](../../../../results/original_pair_group_v64.json).

## Reproduce on RK3588S

Use the pinned original v55 graphs, v53 original capture, v56 original capture,
SDK 2.3.2 and existing NEON library. Artifacts are private and must be produced
by the preceding original-model validation steps; none are bundled here.
Each output directory must be new. Keep actual ORT, compilation and SDK calls
on the board, sequentially. These scripts do not change system clocks.

1. Run `compose_original_pair_group.py --manifest PARENT/compiled.json
   --manifest-sha256 1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42
   --output-dir GROUP` using the board toolchain Python. This statically creates
   `GROUP/export.json` and the weight-bearing ONNX, retaining reversible
   protobuf checks for every original node and initializer.
2. Run `check_compile_pair_group.py --mode check --manifest GROUP/export.json
   --sha256 EXPORT_SHA --parent-manifest PARENT/compiled.json
   --capture-dir V53_CAPTURE --calibration-dir V56_CAPTURE --output-dir CHECK`.
   The board CPU ORT check uses one thread and `ORT_DISABLE_ALL`. All four
   outputs must match the sequential original graphs before compilation.
3. Run the same script with `--mode compile --manifest GROUP/export.json
   --sha256 EXPORT_SHA --parent-manifest PARENT/compiled.json
   --check-report CHECK/check.json --check-report-sha256 CHECK_SHA
   --output-dir COMPILED`. Inspect `COMPILED/result.json` and cleanup before
   running its resulting `group.rknn`.
4. Using the production board environment, run `probe_original_pair_group.py
   --manifest GROUP/export.json --manifest-sha256 EXPORT_SHA
   --compile-result COMPILED/result.json --compile-result-sha256 COMPILE_SHA
   --check-result CHECK/check.json --check-result-sha256 CHECK_SHA
   --output-dir PROBE`. SHA arguments are SHA256 of the referenced files.

The probe uses ABBAABBA on each real input, no hidden warmup or profile flags.
The separate path's second pair consumes the **actual NPU first-pair global
output**, reshaped; it does not substitute a captured Torch activation. Both
routes include explicit NEON FP16 I/O, all owned output copies, finite checks
and output release. Hashing and saved-array writes are outside measured calls.
Normal outputs, call timing, clock/temperature samples and cleanup are saved.

`test_probe_original_pair_group.py` covers owned outputs and partial-failure
cleanup with fake SDK/NEON buffers. `test_check_compile_pair_group_generated.py`
uses generated arrays and fake ORT/SDK objects to check four-output composition,
failed gates and cleanup. These nine generated checks do not execute a model.

Do not publish the generated ONNX/RKNN, captured tensors, original checkpoint
or SDK library. Code licensing does not replace the original weight license.

## SDK 2.3.2 weight-compression rejection (v65)

The recorded board attempt changed only `quantize_weight=False` to `True` for
this same four-output graph. Planned build options remained
`do_quantization=False, dataset=None, auto_hybrid=False`. SDK 2.3.2 rejected
`RKNN.config` with `The quantized_dtype = 'w8a16' not support in 'rk3588'!`.
It did not reach ONNX loading, build or export. No compressed RKNN or new
file-size, SDK-memory, inference-time or accuracy result exists. SDK release,
environment/affinity restoration and unchanged source/input hashes passed.
[Failure record and evidence hashes](../../../../results/original_weight_compression_v65.json).

Using the isolated SDK 2.3.2 board interpreter, from the repository root:

```sh
python deploy/rk3588/validation/original_pair_group/build_weight_compressed_pair_group.py \
  --manifest GROUP/export.json \
  --check-report CHECK/check.json \
  --baseline-compile COMPILED/result.json \
  --output-dir COMPRESSION_ATTEMPT
```

`GROUP`, `CHECK` and `COMPILED` must contain the exact recorded v64 artifacts;
the script pins their SHA256 values, including the ordinary FP16 baseline.
`COMPRESSION_ATTEMPT` must be a new directory. These private weight-bearing
artifacts are not bundled. The expected recorded result is a nonzero exit and
a failed `result.json` at config, followed by cleanup. This branch stopped
after that rejection; no dtype override was attempted. The result establishes
this SDK/target configuration limit, not all hardware or SDK-version limits.
