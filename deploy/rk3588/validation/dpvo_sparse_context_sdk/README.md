# Sparse-context RKNN component comparison (v34)

This is isolated DPVO encoder research on RK3588S. Original DPVO parameters,
608×464 once-normalized BGR inputs and full fmap computation remain unchanged.
The default VGGT executor is unchanged; this directory contains no ORB, added
LK/PnP pose frontend, Update, correlation, BA or complete VO implementation.
DPVO-derived work retains the [MIT notice](LICENSE-DPVO).

The v33 suffix accepts exact INT64 `[96]` row-major indices, preserves their
order and repeats, and returns fmap `[1,128,116,152]` plus context `[1,96,384]`.
Both original and spatial graphs still compute the complete imap internally.
Logical FP32 return data changes from 36,110,336 to 9,175,040 bytes.

Two warm complete SDK calls per role average **427.916 → 411.549 ms** for the
original graph and **217.184 → 178.058 ms** for the spatial graph. Normal runs
use frame 0 AB/BA/AB, followed by frames 1/2 in crossed order. Each role has five
calls over three fixed real inputs; profiling is a separate three-call process.
The sparse Gather executes on CPU. Both methods remain rejected by the unchanged
`rtol=1e-4, atol=1e-5` original FP32 gate; neither is adopted. Finite results,
matching RKNN-parent canonical bytes and shorter component timing do not prove
trajectory accuracy, 20 Hz or all-NPU execution. These short samples do not
control temperature/frequency. [Results and limitations](../../../../results/dpvo_sparse_context_v34.json).

## Prerequisites and compilation

Raw SHA-pinned v33 CPU artifacts and rewrite receipts, corresponding parent
ONNX/RKNN artifacts, RKNN Toolkit2 2.3.2 and the frozen generic compiler are
required. They are not bundled here. Compiler configuration is exactly RK3588,
FP16, optimization level 3 and nonquantized. The spatial parent's historical CPU
rejection is retained; spatial compilation and execution require the explicit
`--diagnostic-allow-fp32-rejected` flag. Every output directory must be new.

Set the variables below to your preserved artifacts. Run compilation on the
compiler host, without a real model forward:

```bash
TOOLS=deploy/rk3588/validation/dpvo_sparse_context_sdk
python "$TOOLS/compile-v34-dpvo-sparse-context.py" \
  --method original --rewrite-dir "$REWRITE_DIR" --rewrite-sha256 "$REWRITE_SHA" \
  --reference-dir "$V33_REFERENCE_DIR" --reference-sha256 24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168 \
  --parent-onnx "$PARENT_ONNX" --editor "$V33_EDITOR" \
  --frozen-compiler "$FROZEN_COMPILER" --output-dir "$NEW_COMPILE_DIR"
```

## Board execution and layout

The driver guards Linux/aarch64/RK3588 before SDK initialization. It uses four
A76 CPUs and core mask 7 without sysfs changes. Actual SDK attributes must match:
FP16 NHWC image input, INT64 index input with format 3 (UNDEFINED), NCHW fmap and
format-3 context. Indices are passed as INT64; no float or INT32 fallback exists.
The SDK receives a FP32 NHWC image copy and converts to native FP16. Outputs use
`want_float=1, is_prealloc=0`; their returned bytes must equal FP32 logical sizes,
and arrays are copied before release. Context's native 73,728-byte footprint is
not its returned 147,456-byte FP32 buffer.

```bash
python "$TOOLS/run-v34-dpvo-sparse-context-board.py" \
  --method original --reference-dir "$V33_REFERENCE_DIR" \
  --reference-sha256 24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168 \
  --rewrite-dir "$REWRITE_DIR" --rewrite-sha256 "$REWRITE_SHA" \
  --dense-compile-dir "$DENSE_COMPILE_DIR" --dense-compile-sha256 "$DENSE_COMPILE_SHA" \
  --sparse-compile-dir "$NEW_COMPILE_DIR" --sparse-compile-sha256 "$SPARSE_COMPILE_SHA" \
  --parent-source "$FROZEN_V26_RUNNER" --compiler-source "$TOOLS/compile-v34-dpvo-sparse-context.py" \
  --sdk-helper "$TOOLS/v34_sparse_rknn.py" \
  --sdk-helper-sha256 646f875d1b5eea561c0d6aa4d1b9078351014a39a62976ec895a814cd466b387 \
  --abi-helper "$FROZEN_ABI" --header "$SDK_HEADER" --output-dir "$NEW_BOARD_OUTPUT"
```

Use a separate new output/process with `--collect-perf-detail` for profiling.
Complete SDK timing includes input packing, set/run/get, owned FP32 copies and
release. Dense host context gather is timed separately and added to feature-ready
time. Context initialization, reference comparisons, hashes, saves and profile
queries are excluded. Profile rows do not represent complete API timing.

## Independent saved-array audit

The NumPy/stdlib auditor executes no model, SDK, image decoder or ORT session.
It verifies six saved canonical results per method, exact identity/indices,
ordinary schedule, actual attrs, errors, timing and cleanup. Four unsaved ordinary
repeats are checked through recorded raw hashes, not independently reread arrays.
Without full dense imap it verifies saved context but cannot independently
recompute that call's gather. Profile NPZ files may remain on board; the explicit
metadata-only flag verifies their recorded hashes against rehashed ordinary
canonical arrays and parses saved profiles, without claiming profile array reads.

```bash
OPENBLAS_NUM_THREADS=1 python "$TOOLS/audit-v34-dpvo-sparse-context.py" \
  --normal-dir "$NORMAL_DIR" --normal-sha256 "$NORMAL_SHA" \
  --reference-dir "$V33_REFERENCE_DIR" --profile-dir "$PROFILE_DIR" \
  --profile-sha256 "$PROFILE_SHA" --profile-metadata-only --output "$NEW_AUDIT_JSON"
```

Omit `--profile-metadata-only` only when profile NPZ files are actually available.
Audit status `passed` establishes consistency of available evidence; it does not
replace the false numerical-acceptance result. Only source and small synthetic
tests are published, with no weights, model binaries or dataset arrays.
