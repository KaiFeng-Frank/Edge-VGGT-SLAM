# Original VGGT GPU attention diagnostic

This directory records GPU integration tests for the original attention computation. Standalone board GPU operators passed fixed-input checks. GPU-target SDK callbacks inside the complete frame0/global0 pair produced unstable outputs; a CPU-target callback invoking the board GPU was stable but slower. Neither route was adopted. Current complete-model results are documented in [v82](../original_calibrated_camera/README.md).

Only the original VGGT checkpoint, parameters and all tokens are used. Original Q/K LayerNorm, two-axis RoPE, complete QK/softmax/V and projection-input layout run on the RK3588S Mali GPU. QKV, projection, residuals and MLP remain in the original RKNN graph. The host computer GPU was not used.

## Measured scope

| Check | Result |
|---|---|
| Two original CPU inputs, standalone frame/global GPU tails | 28 outputs passed `rtol=1e-4, atol=1e-5`; 48 saved arrays independently checked |
| Six warm complete tails, including upload/download, finite checks and release | frame 7.640–7.969 ms; global 11.642–14.281 ms |
| Original FP16 pair A / GPU custom-op pair B, six warm calls each in the same run | 26.986 / 92.455 ms |
| Four repeats of each input | A stable; both B outputs changed on every repeat |
| B relative L2 against original CPU block outputs | frame 15.94–32.98%; global 38.10–52.60% |
| CPU-target callback invoking the board GPU, separate matched warm A/B run | 27.153 / 72.639 ms; B stable across repeats, original CPU relative L2 0.425–0.476%, fixed strict checks still failed |

Standalone tail checks do not establish correct SDK tensor handoff. Separate per-role sources/names, an owned strict OpenCL program and entry queue-finish/FROM_DEVICE controls still failed. Actual SDK program options included `-cl-fast-relaxed-math -cl-mad-enable -DCL_DTYPE_half`; identical kernel text does not prove identical compiled arithmetic. These observations do not establish the root cause.

Four callback snapshots were replayed with their captured GPU-side inputs; norm, RoPE and outputs passed the fixed strict checks. GPU outputs were byte-equal to CPU readback after FROM. Across two frame callbacks, all 11 saved GPU arrays and readback outputs matched, while 13,326 elements changed in the downstream complete frame output. Extra output TO and SDK-example import properties did not resolve the failure. The stable CPU-target control was still 2.675 times slower, with all original tail arithmetic executed on the board GPU. These fixed-input diagnostics establish neither the root cause nor complete-model quality.

See the [result JSON](../../../../results/original_gpu_attention_v83.json). The [source list](source-copy-list.json) pins each source's bytes and SHA; four saved-data audits read existing NumPy files without rerunning the model. Diagnostic capture costs are excluded from normal performance claims.

## Reproduction

Use the original RK3588S environment and weights, plus privately generated v53/v56 activations, ONNX/RKNN graphs and receipts. This repository supplies neither captured arrays nor weight-bearing exports. Frozen scripts retain actual board paths and SHA pins; prepare and verify those artifacts before starting. These scripts are experimental diagnostics, not a production backend.

| Stage | Source entrypoint |
|---|---|
| Original FP32 local capture, four original block forwards | `capture_original_attention_tail_board.py` |
| Standalone GPU tail, initial diagnostic separate from six warm calls | `probe_original_attention_tail_gpu_board_r2.py`, `original_attention_tail_opencl.py` |
| Two original tail cuts and conversion-time references | `build_pair_attention_custom_board.py`, `build_pair_attention_custom_board_r2.py` |
| RKNN 2.3.2 nonquantized FP16 compilation | `compile_pair_attention_custom_board_r2.py` |
| Complete pair A/B, normal flags0/core7 | `probe_attention_custom_pair_board.py`; `run_attention_custom_pair_1ghz.sh` |
| Source, strict-program and synchronization controls | `run_attention_custom_pair_{distinct,strict,handoff}_1ghz.sh` |
| Bounded callback snapshots and GPU-input replay | `rknn_original_attention_custom_diagnostic.py`, `probe_callback_shadow_board_r2.py` |
| CPU-target callback invoking the board GPU | `rknn_original_attention_custom_cpu_staging.py`, `run_attention_custom_pair_cpu_staging_1ghz.sh` |
| Board custom-op ABI, 36 size/offset fields | `build_custom_abi_board.py`, `probe_custom_op_abi.c` |

Use separate production and isolated RKNN compiler Python environments as shown in the executed `run_*.sh` files. The GPU library, NEON library, typed RKNN ABI and [existing OpenCL helper](../original_native_gpu/native_erf_gelu_opencl.py) have independent pins and are not redistributed here. `custom-op-abi.json` contains 36 board-verified size/offset values. The original C probe requires the existing board SDK header, which is not redistributed.

Run 1 GHz tests through the existing [bounded controller](../original_clock_test/npu_frequency_test.py), with a 180-second/75°C limit and restoration in `finally`. Normal probes do not write frequencies. The GPU retained `simple_ondemand`; observed frequency changes are recorded, with no governor/min/max changes. Prepare the required artifacts and a new output directory before using a launcher.

```sh
BASE=/mnt/vggt-data/vggt-slam2
DOC="$BASE/src/deploy/rk3588/validation/original_gpu_attention"
"$BASE/venv/bin/python" "$BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py" \
  --execute-frequency-test --frequency-hz 1000000000 \
  --timeout-seconds 180 --temperature-limit-c 75 \
  --launcher "$DOC/run_attention_custom_pair_1ghz.sh" \
  --output "$BASE/experiments/attention-pair-control-new"
```

This command matches the controller interface used for the executed launcher. It does not create the required private graphs, inputs or source snapshots. Complete saved capture, component and pair files are required for independent audit:

```sh
python audit_attention_custom_pair_saved.py \
  --directory PRIVATE_PAIR_RESULT_DIR \
  --capture-directory PRIVATE_CAPTURE_DIR --output audit.json
```

No new complete-model replay, trajectory or map quality validation was completed for these routes; the 20 Hz/100 ms targets remain unmet. Original weights and graphs retain their separate model-license obligations; see [licensing](../../../../MODEL_LICENSES.md). This directory publishes original source, hashes and statistics, with no captured arrays, model weights, SDK binaries or firmware.
