# Original VGGT retained-group FC2 partition checks

This extends the measured pair-level four-way FC2 K partition to all 48
original frame/global blocks, in the existing four retained groups. Every
original parameter value, layer and token remains. Outputs remain 2/2/2/3,
including the unnormalized global20 target consumed by the original code.
Only FC2 MatMul is replaced: four contiguous activation/weight K slices,
four partial products and three sums, followed by the original bias once.
Floating-point reduction order changes; parameter reconstruction alone does
not prove numerical equivalence.

All four groups were checked on the RK3588S using two saved real inputs and
CPU ORT with optimization disabled. Against the exact original FP32 ONNX
groups, 17 of 18 output checks pass the unchanged `rtol=1e-4, atol=1e-5` gate.
Group 2 fails: one of 108544 frame-output elements exceeds the tolerance on
the second input. Left-associated partial sums give error `1.62572e-5` at
`[1,39,41]`, versus allowed `1.01253e-5`. A single follow-up with balanced
adjacent sums fails at the same element, with error `2.17333e-5`.

Neither version was compiled or installed. There is no new NPU timing,
trajectory, or realtime result for these candidates. Comparisons with saved
original Torch outputs are separate: some original ONNX groups already fail
that gate, so those failures are not attributed to this rewrite.

```sh
# Run only in the prepared RK3588S compiler environment; each output is new.
python build_retained_fc2_k4_board.py --group-index 0 --check-only --output-dir G0
python build_retained_fc2_k4_board.py --group-index 1 --check-only --output-dir G1
python build_retained_fc2_k4_board.py --group-index 2 --check-only --output-dir G2
python build_retained_fc2_k4_board.py --group-index 3 --check-only --output-dir G3
python build_retained_fc2_k4_balanced_board.py --group-index 2 --check-only --output-dir G2_BALANCED

# Saved-data audit on host or board; no model execution.
python audit_retained_fc2_saved.py --check G2/check.json CHECK_SHA256 \
  --build G2/result.json RESULT_SHA256 --builder-sha256 BUILDER_SHA256 \
  --output NEW_AUDIT.json
```

The balanced builder requires check-only mode. The left-associated builder
also retains a gated compiler interface, which was not used in this experiment.
Failure receipts, saved arrays and weight-bearing graphs remain private;
published summaries retain evidence hashes. The monitor reads frequency,
temperature and worker memory, and stops only its own process group at its
deadline, 75 C, or a departure from 300 MHz. It does not set system clocks.
`smaps` reads add monitoring work and are not atomic memory snapshots.

[Results](../../../../results/original_retained_fc2_v73.json).
