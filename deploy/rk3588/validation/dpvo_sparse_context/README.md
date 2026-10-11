# DPVO sparse context-output diagnostic (v33)

This optional learned-VO component study retains the default VGGT path. It changes only which DPVO encoder outputs are returned: the full correlation feature map remains, while runtime integer patch indices select 96 context vectors from the full context map. It changes no pixels, weights, normalization, feature computation or original `/4` scaling. It adds no ORB/LK/PnP frontend.

The two static candidates append only a reshape, Gather and transpose to the original v26 graph or the previously rejected v27 spatial-norm graph. Inputs are `normalized_bgr` FP32 `[1,3,464,608]` and `patch_indices` INT64 `[96]`; outputs are `fmap` FP32 `[1,128,116,152]` and `context` FP32 `[1,96,384]`. Indices are `y*152+x`, preserving order and duplicates. The original dense context map is still computed internally.

The logical FP32 output size falls from 36,110,336 to 9,175,040 bytes. This is an output contract calculation, **not measured RKNN transfer savings**. Sparse Gather placement, SDK output formats, complete I/O timing and trajectory quality have not been measured here.

## Actual CPU ORT suffix check

All neural execution was on RK3588S. The board ran both original and spatial parent/sparse pairs on the same three frozen, once-normalized inputs, with CPUExecutionProvider, optimization disabled and four A76 threads. There were six paired records and twelve graph calls. The centers were predeclared diagnostic indices with corners, repeats and reordered selection; they were not original RNG tracking observations or chosen from successful pixels.

All six sparse `fmap`/`context` outputs were byte-identical to their current parent `fmap`/dense-context gather and to the corresponding saved historical parent values. The original graph passes the original CPU elementwise gate, `rtol=1e-4, atol=1e-5`. The spatial graph retains its historical rejection: its three `fmap` outputs fail at 4, 5 and 53 elements, while the sampled contexts pass. Exact sampling does not validate a rejected parent. No sparse RKNN model or complete VO trajectory was executed, and no default was changed.

The NumPy-only saved-array auditor rehashes all nine NPZ files, checks input bytes and source identities, reconstructs ordered integer sampling, recomputes strict errors and byte equality, and retains provenance/cleanup limits. Complete current parent `fmap` arrays are saved. Complete current parent `imap` arrays are not saved: their board-recorded raw hashes are compared with independently rehashed historical arrays, while the saved current sampled contexts can be compared directly. The audit does not claim an independent full-array snapshot of current `imap`, or a host rehash of unavailable board source paths.

## Helpers and external artifacts

- `rewrite-v33-dpvo-sparse-context.py`: static ONNX editing/checking only. The original nodes and initializers are byte-preserved; no real-model forward or ORT session runs on the host.
- `run-v33-dpvo-sparse-context-ort-board.py`: RK3588-only CPU ORT comparison. It requires a new output directory, exact frozen capture/export/old ORT receipts, the spatial rewrite and failed spatial ORT receipt, both sparse recipes, the frozen editor and external official source checkout. Session references are dropped on success or failure; no nonexistent public ORT native-destroy API is claimed.
- `audit-v33-dpvo-sparse-context.py`: portable NumPy saved-array audit. Supply `--run-dir --run-sha256 --capture-dir --old-ort-dir --spatial-ort-dir --original-sparse-dir --spatial-sparse-dir --output`. The driver/editor paths default to adjacent published sources. Output must be new, preserving earlier receipts.
- `calculate-v33-fast-config-budget.py`: standard-library-only source/config arithmetic. FAST is loaded as `config.py` plus `fast.yaml`, not layered on `default.yaml`. Its all-admitted/no-keyframe-deletion envelope is explicitly hypothetical; it executes no flow, poses, BA or neural network. MAC counts exclude non-Linear arithmetic, topology, correlation and I/O; one MAC conventionally counts as two arithmetic FLOPs, not total model FLOPs. It does not establish hardware throughput.

Run `python -B SCRIPT --help` for required arguments. The [v26 source provenance](../../../../results/dpvo_sources_v26.json), [original encoder artifacts](../../../../results/dpvo_encoders_v26.json) and [failed spatial-norm evidence](../../../../results/dpvo_norm_rewrite_v27.json) define the external artifact lineage. Images, checkpoints, ONNX models, SDK/ORT installations and NPZ files are not bundled. Exact receipt hashes are deliberate; new artifact bytes require a separately reviewed lineage.

The adjacent [MIT notice](LICENSE-DPVO) preserves DPVO source attribution. The software license does not verify the separately obtained checkpoint license or change the original VGGT weight restrictions.

Tiny tests use synthetic arithmetic, tiny ONNX graphs and fake sessions only:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  tests/test_v33_sparse_context.py \
  tests/test_v33_sparse_context_ort.py \
  tests/test_v33_sparse_context_audit.py \
  tests/test_v33_fast_budget.py
```

This study proves neither 20 Hz pose publication nor complete camera latency, map cadence, endurance, accuracy or an 8 GB deployment target.
