# Saved hybrid R2 audit

The original frame0→global0 pair diagnostic has a verified floating boundary, but it fails the unchanged original Torch FP32 component gate. It is not adopted.

The auditor only reads saved data. It checks all 32 output NPYs and four original input/reference NPYs for file/raw SHA, FP32 dtype, finite values, contiguous layout and exact shape. It independently reconstructs the fixed AB/BA schedule, synchronous input-set/run/get/copy/release intervals, six-call means and same-role repeat equality. All four role/input groups have eight bitwise equal outputs across normal and separate profile runs. Ordinary and hybrid outputs are allowed to differ.

| Observation | Ordinary | Hybrid R2 |
| --- | ---: | ---: |
| Normal complete SDK I/O mean, exactly six subsequent calls | 336.673159 ms | 319.927622 ms |
| Separate profile mean, exactly six subsequent calls | 348.996875 ms | 332.516061 ms |
| Original FP32 relative L2, evaluation_00 | 0.004452537 | 0.066717348 |
| Original FP32 relative L2, evaluation_01 | 0.004977551 | 0.079081068 |
| Elements outside strict gate, evaluation_00 / 894976 | 523092 | 576966 |
| Elements outside strict gate, evaluation_01 / 894976 | 527279 | 580558 |

The normal mean decreased by 16.7455365 ms (4.973825819%). Each role excludes round zero for both inputs, leaving three calls per input and a denominator of six; this remains a small repeated-input component comparison. The queried logical input and output are FP16 for both roles, with identity affine metadata. The API receives FP32 (`pass_through=0`) and returns an owned FP32 copy (`want_float=1`); every saved output value is exactly representable in FP16. This proves boundary observations and saved-value membership, not internal arithmetic precision, operator placement or an error cause.

The compiler request is limited to 65 actual internal output selectors: four fc1/GELU INT8 requests and 61 FP16 requests. Exact original MLP-only arithmetic precision remains unproven. The retained compile-time contract describes what was known at compilation; the separate runtime descriptors establish the observed floating boundary.

The strict component gate remains `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)` for the same-input original Torch FP32 reference. No component gate is relaxed. Component error does not measure ATE/RPE; full-model trajectory quality can be tested separately without relabeling these failures. The two calibration-excluded inputs have been reused for development. Fresh 20 Hz poses, end-to-end P95 ≤100 ms, trajectory quality, stability and 8 GB operation remain unvalidated.

`audit_saved_hybrid_pair_r2.py` preserves the prepared auditor's numerical and timing logic. Its only operational changes bind the actual R2 runtime basename explicitly and pin the adjacent reused R1 contracts source before and after the audit. The frozen prepared source is unchanged. The auditor imports only SHA-pinned saved-data helpers and pure compile-receipt contracts, without model, ORT, SDK, compiler, calibration or board execution.

The generated checks use small synthetic tensors only. `source-freeze-r2.json` binds the source, checks, evidence pins and public-safe receipt. No raw arrays, checkpoint, weight-bearing exports or SDK binaries are published.

## Reproduce from saved private evidence

Requires Python and NumPy. From this directory, pass the private evidence directories through the task-specific variables below. The Flash helper and both hybrid contracts must retain their frozen bytes. R2 contracts import the adjacent R1 contracts; retain both files together in `../hybrid/`. The auditor does not import or execute the board runtime or compiler sources; it hashes them.

```sh
python3 check_saved_hybrid_audit_r2.py \
  --flash-auditor ../flash-audit-numeric/audit_saved_flash_pair.py \
  --output /tmp/hybrid-r2-generated-checks.json

python3 audit_saved_hybrid_pair_r2.py \
  --normal-dir "$HYBRID_NORMAL_DIR" \
  --profile-dir "$HYBRID_PROFILE_DIR" \
  --capture-dir "$ORIGINAL_CAPTURE_DIR" \
  --runtime-source-dir ../hybrid-runtime \
  --hybrid-source-dir ../hybrid \
  --ordinary-compile-dir "$ORDINARY_COMPILE_DIR" \
  --hybrid-compile-dir "$HYBRID_COMPILE_DIR" \
  --flash-auditor ../flash-audit-numeric/audit_saved_flash_pair.py \
  --pins evidence-pins-r2.json \
  --pins-sha256 20eb0345d966d390a1131677d25c9d691e2a3b7f3864dc6ced9c191bb3c451d9 \
  --output /tmp/hybrid-r2-saved-audit.json
```

Both output paths must be new; actual audit output must be outside the evidence directories. Public receipts contain aggregate statistics and hashes. The private evidence arrays and receipts are required for reproduction; the published receipt alone cannot reproduce the numerical audit.
