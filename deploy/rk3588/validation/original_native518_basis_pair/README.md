# Original native518 frame/global residual-basis gate

This experiment checks the [mean-preserving residual-basis proposal](../original_native518_graph_basis/README.md)
on one complete original frame0 → global0 pair. It is an FP32 numerical gate,
not an NPU speed result or a full-model quality decision. See the
[compact results](../../../../results/original_native518_basis_pair_v126.json).

Three graphs isolate the transformation: A keeps the original pair; F folds
norm affine terms and LayerScale with identity basis; B uses the fixed
`R=((J4-2I4)/2) tensor-power5` basis across both residual blocks. The original
36 learned tensors are recovered and hash-checked before deriving constants
in float64 and storing FP32 weights. Two B1 frame branches share original
weights. All 1041 tokens per frame, 2082 global tokens, q/k normalization,
RoPE, original Erf and global FC2K4 partitions and sum order remain.

Norm1/norm2 retain bare original LayerNorm with unchanged epsilon. Their
affine terms fold into QKV/FC1; projection/FC2 fold the original LayerScale
and output basis. QKV/FC1 retain original coordinates in real arithmetic.
B rotates the initial input once, feeds its actual rotated frame result
directly into global0, and restores only the two public outputs. The three
boundary MatMuls are part of this gate, not a measured deployment schedule.
No saved reference supplies an internal inference value.

Both cases use the **same TUM image pair [32,40]**. `cpu_prefix` is the
original CPU DINO24/prefix boundary and has matched original Torch outputs.
`npu_prefix` is a separately captured actual NPU DINO24/prefix boundary;
each transformed graph is compared with A on that exact input. These are
not two independent scenes, calibration data or held-out quality evidence.
Internal QKV/FC1 arrays are not separately observed in this first screen.

The hard gate remains the original R98 B1 relative L2 `<=1e-5` and maximum
absolute error `<=1e-3`. Elementwise `rtol=1e-4`, `atol=1e-5` and byte equality
are independent diagnostics. All six recovered R2 calls pass the numeric
gate and all elementwise diagnostics. B versus A across both cases has worst
relative L2 `8.6705e-7` and maximum error `7.6294e-6`. For the matched
CPU-prefix Torch reference, B's worst values are `1.3610e-6` and `1.6332e-5`.
F/B are not byte-identical to A. An independent saved-only audit verifies
all 12 outputs, both inputs, both matched Torch references and reported gates.

Reproduction requires the private pinned R53/R98 graph/reference receipts,
original source and checkpoint lineage, original B1/K4 ONNX parents, real
input arrays and board SDK Python with ONNX and ORT. They are not shipped or
downloaded automatically. The source/config/proof and actual R2 launchers
are pinned in [sources.json](sources.json). Config paths are the recorded
board layout, not a portable dataset installer. No checkpoint is deserialized.

Stage the bundle at `/dev/shm/edge-vggt-v126-source-r1` and provide the pinned
artifacts at the config paths. The launchers use process-local library and
thread settings, CPU4–7, one ORT session at a time and fresh rootfs `/tmp`
outputs. The bounded R2 commands use the existing observed 1 GHz value:

```sh
BASE=/mnt/vggt-data/vggt-slam2
S126=/dev/shm/edge-vggt-v126-source-r1
PRODPY="$BASE/venv/bin/python"
mkdir -p "$S126/tmp"
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH="$BASE/sysroot/usr/lib/aarch64-linux-gnu" \
  "$PRODPY" -B "$S126/run_sampled_cpu_stage_r126.py" \
  --stage native518basis126export2 --expected-npu-hz 1000000000 --timeout 180 \
  -- /bin/sh "$S126/run_export_r2.sh"
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH="$BASE/sysroot/usr/lib/aarch64-linux-gnu" \
  "$PRODPY" -B "$S126/run_sampled_cpu_stage_r126.py" \
  --stage native518basis126check2 --expected-npu-hz 1000000000 --timeout 180 \
  -- /bin/sh "$S126/run_check_r2.sh"
```

The expected Hz is an observation, not a requested frequency change; use
the existing observed value for another run. This CPU controller
observes frequency, pauses at 70°C, resumes at 63°C and stops at 75°C;
it does not change clocks. Export/check complete in `6.5208 / 37.3958 s`,
with maxima `59.153 / 70.230°C`; check includes four cooling pauses.
These are controlled CPU stage times, not NPU speed measurements.
For another run, change both worker outputs, update check's export receipt
SHA and use fresh controller stages. Keep evidence on persistent storage.

Host and board restarts of unestablished cause lost the initial tmpfs graphs
and arrays before durable archival.
The exact original exporter/config were recovered by SHA and the experiment
was repeated with `/tmp` outputs. R1 logs/controls remain historical evidence;
they cannot replace the recovered arrays. The basis changes finite rounding
and preserves dense arithmetic count. FP16/INT8 behavior, compiler placement,
all48-block/head integration, trajectory quality, 20 Hz, P95, stability and
physical 8 GB operation remain unvalidated. Defaults are unchanged.
Repository code and original weights retain separate licenses.
