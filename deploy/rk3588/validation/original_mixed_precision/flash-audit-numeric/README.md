The saved Flash off/on experiment preserves exactly the same output for each of the two original VGGT inputs and shows no measured normal I/O improvement. All 32 saved output files pass independent file SHA, raw SHA, shape `[1,874,1024]`, FP32 dtype, contiguous-layout and finiteness checks. For each input, all 16 outputs across both compiler settings, four rounds and the normal/profile processes are bitwise equal. These outputs still fail the unchanged same-input original Torch FP32 gate (`rtol=1e-4`, `atol=1e-5`). Audit integrity passing does not adopt either candidate.

| Measurement | Ordinary | Flash flag |
| --- | ---: | ---: |
| Normal subsequent complete I/O mean, six calls each | 0.33682933266666665 s | 0.33696277700000005 s |
| Profile subsequent complete I/O mean, six separate calls each | 0.3506255285 s | 0.3494538403333333 s |

The normal mean difference is +0.133444 ms for the Flash flag, or +0.039618%. Six repeated-input calls per setting do not establish a statistically reliable speed difference. Profile contexts are initialized separately, with flag 8 instead of normal flag 0; their times remain separate. Operator-table interpretation belongs to the independent operator audit.

Each process submits exactly 16 serial SDK calls: two ordered inputs, four rounds, and two settings. The first input alternates ordinary/Flash then Flash/ordinary, and the second input reverses that order. The subsequent means exclude round zero for both inputs, including the second input's first observation in an already used context. Ordinary uses SDK indices `[3,4,7,10,13,14]`; Flash uses `[2,5,6,11,12,15]`. Ordered floating-point addition reproduces the actual board's CPython 3.11 mean exactly, regardless of the host Python `sum()` implementation.

Complete I/O spans input submission, synchronous run, output retrieval, owned output copy and release. Each recorded stage is checked in integer nanoseconds against the start/end interval. Initialization, source hashing, numerical comparisons, snapshots, file saves and the later profile query remain outside that interval. The saved API reports FP32 input with `pass_through=0`, SDK-owned output with `want_float=1`, and an owned copy before release; logical input/output descriptors are FP16. Source and saved-data SHA checks, declared environment/affinity restoration and cleanup are preserved.

The unchanged strict component failures are 523,092 / 894,976 elements for evaluation_00 and 527,279 / 894,976 for evaluation_01. Their relative L2 errors are 0.0044525369243 and 0.0049775506393. Both inputs were excluded from calibration and reused for development; this is not a new blind quality test. This audit covers only the complete original frame0/global0 pair. Full original-model ATE/RPE, fresh-pose 20 Hz, complete input-to-pose P95 at most 100 ms, long stability and memory limits remain separately unvalidated.

The optional cheap comparison reads four first-output arrays across v49 nonquant FP16 and v50 ordinary. Both same-input outputs are bitwise equal after recompilation. The comparison makes no claim about internal operator equivalence or a cross-process speed change.

Run the portable saved-data auditor with Python and NumPy. All source/evidence directories are explicit arguments, and the result must be a new file outside those directories:

```sh
python3 audit_saved_flash_pair.py --normal-dir /path/to/v50-normal --profile-dir /path/to/v50-profile --capture-dir /path/to/v49-capture --flash-source-dir /path/to/flash-source --runtime-source-dir /path/to/v49-runtime-source --output /path/to/new-audit.json
python3 check_saved_flash_audit.py --output /path/to/new-generated-checks.json
python3 compare_saved_v49_baseline.py --v49-normal-dir /path/to/v49-normal --v50-normal-dir /path/to/v50-normal --output /path/to/new-baseline-comparison.json
```

Keep these three audit scripts in the same directory. The comparator pins the main auditor's source SHA before importing its saved-file helper. The 12 generated checks use no real saved arrays and cover missing/duplicated mean entries, call ordering, host-independent ordered addition, complete I/O stage coverage, output ownership, original-reference tolerance, nonfinite rejection and signed-zero byte equality.

All source and receipts in this directory are public-safe. Reproduction requires private original saved arrays and pinned experiment records; no arrays, weights, weight-bearing exports or SDK binaries are included. No actual model, ORT, SDK, compiler, calibration or board execution occurs in these auditors.
