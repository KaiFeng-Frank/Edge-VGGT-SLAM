# Original QKV layout and RoPE compiler controls

[R212](../../../../../results/original_native518_cf_rope_controls_v212.json) separates static/CPU equivalence, actual compiler placement, ordinary returned-output ABI and original-FP32 accuracy. The CF native producer removes internal CPU layouts, but its board Q/K accuracy fails badly. Replacing Gather with four Slices plus Concat passes CPU equivalence and is rebuilt as Gather → ConvAdd by the compiler. The quarter-arithmetic control passes CPU word-equivalence and its final compile table has no internal CPU operator, Gather or ConvAdd, but its [subsequent actual NPU gate](../../../../../results/original_native518_quarter_rope_NPU_diagnostic_v213.json) rejects original Q/K precision. Handoff cost remains unmeasured. No producer-to-consumer speedup or deployment default is accepted.

## Published source

- `extract_original_qkv_prefix.py`: exact original 20-node/16-initializer prefix extraction and five half-output layouts from the pinned original-A parent.
- `export_cf_rope_direct4.py`: direct rank4 control and channel-first (CF) RoPE variants. All eight learned TensorProtos remain exact; the original full 64-channel norms stay before RoPE.
- `export_cf_rope_slice_concat.py`: replace only the two CF channel Gathers with four Slices and a Concat; retain the Mul/Add sequence and all parent constants.
- `export_cf_rope_quarter_arithmetic.py`: perform each quarter's original sign/cosine/sine Mul operations and Add before final channel Concat. The norms are still complete 64-channel operations.
- `producer_output_owner.py`: the executed diagnostic's exact single-attempt typed-output owner and role inverse, extracted as a library. It loads no SDK at import and has no inference CLI.

Pure `build()` functions, static proof helpers and the output-owner class are AST-identical to the sources pinned in R212. Only artifact paths, CLI/report orchestration and publication metadata are adapted. The [adaptation manifest](../../../../../docs/PUBLICATION_SOURCE_ADAPTATIONS.json) distinguishes historical source identities from these published programs. New executions require new manifests and receipts; this adapted code is not claimed to have produced the historical device receipts.

## Static reproduction

The commands below construct ONNX graphs only. ONNX is required only for `--execute`; default invocation prints metadata without opening model files or importing a tensor framework. No ORT, SDK, GPU, tracing or calibration is used by these builders.

Original weights, captured inputs, weight-bearing ONNX/RKNN files and the vendor SDK are **not distributed**. Obtain the original checkpoint and implementation separately under [their licenses](../../../../../MODEL_LICENSES.md), verify the [pinned original provenance](../../../../../UPSTREAMS.json), and prepare the original uncentered global0 parent. The extraction requires exactly 102,417,440 bytes / SHA256 `57b60feceea0edb57ca4cea38200d6ea47d4e4ee51e4a546357267cbc9f676bb`; another parent fails closed. This record does not supply the earlier full-model export pipeline or private captured case.

From this directory, with `ORIGINAL_A` pointing to that separately prepared parent and fresh output directories:

```sh
python -B extract_original_qkv_prefix.py
python -B extract_original_qkv_prefix.py --execute --parent "$ORIGINAL_A" --output-dir ./new-prefix
python -B export_cf_rope_direct4.py --execute \
  --canonical-parent ./new-prefix/canonical.onnx \
  --native-half-parent ./new-prefix/native_half.onnx --output-dir ./new-direct4
python -B export_cf_rope_slice_concat.py --execute \
  --cf-canonical-parent ./new-direct4/cf_canonical.onnx \
  --cf-native-parent ./new-direct4/cf_native.onnx --output-dir ./new-slice
python -B export_cf_rope_quarter_arithmetic.py --execute \
  --cf-canonical-parent ./new-direct4/cf_canonical.onnx \
  --cf-native-parent ./new-direct4/cf_native.onnx --output-dir ./new-quarter
```

Every builder checks exact parent serialization, closure, unchanged learned data, explicit inverse and inferred output contracts. Output files contain original weights: keep them outside public commits. R212 pins the expected graph hashes. Compiler results use RKNN 2.3.2, the published 13-field configuration, `do_quantization=False`, `dataset=None` and `auto_hybrid=False`; successful compilation does not replace CPU or NPU numerical gates.

## Runtime boundary

The diagnostic owner takes the existing [typed ABI](../../run-v12-rknn-block-board.py) (SHA256 `37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608`), an already loaded API, and NumPy. Its caller must enforce actual RK3588S identity, full artifact hashes, original-X-only feed, CPU4/core7/flags0 policy, fresh capacity/storage guards, bounded thermal control and restoration. Those transport/controller responsibilities are not a standalone entrypoint in this extracted library.

Ordinary `want_float=0` requests the declared output dtype; it does not select native9 storage. Each of the five role names is joined to the actual logical cmd2 index. The owner requires FLOAT16 logical descriptors and exact returned spans before typed dereference, owns each copy before release, and never retries release/destroy after unknown native state. Bounded unexpected spans are copied only as uninterpreted bytes. Native9 descriptors are recorded separately. All twenty diagnostic outputs are saved before numerical verdicts.

The direct rank4 word order is `Y[h//8,t,d,h%8] = HALF(X[0,h,start+t,d])`, with three query ranges `[0,694)`, `[694,1388)`, `[1388,2082)` and full K/V. An exact word map is a CPU representation test, not an NPU-precision guarantee. Across separate identical-input NPU calls, F32 readback matching widened HALF proves cross-run readback/determinism consistency for that case; it does not prove a same-run conversion path, correct absolute head order or original-FP32 equivalence. No zero-copy handoff or complete-chain timing is demonstrated.

Repository code is BSD-2-Clause; original implementation, weights and SDK have separate terms. The code license grants no model-weight rights. The real-time, trajectory, stability and physical 8 GB goals remain unaccepted.
