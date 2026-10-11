# DPVO CPU geometry and fixed-graph feedback research

These helpers adapt the online equations from `princeton-vl/DPVO` commit
`0ac95b656d1fda91c271d2a106460d19ad966fc7`. The adjacent [LICENSE](LICENSE)
preserves Princeton Vision & Learning Lab's MIT notice. This is separate from
the repository's BSD license and the independently unverified checkpoint license.

- `dpvo_cpu_geometry.py`: bounded NumPy dense reference for world-to-camera SE3,
  patch reprojection and two-step online-style BA. It distinguishes normalized
  LieTorch quaternion operations from the raw CUDA center projection equations.
- `run-v28-dpvo-feedback-board.py`: RK3588-only four-round experiment using three
  already captured real views, original Torch Update, native correlation and the
  CPU geometry reference. No new frames, complete VO or NPU execution.
- `audit-v28-dpvo-feedback.py`: NumPy-only verification of saved artifacts,
  state continuity, selected independent projections and fixed-mask residuals.
  It does not execute the model, correlation library or BA.

The driver defaults to the adjacent geometry helper; `--geometry` permits an
explicit path whose expected SHA remains checked. Run each script with `--help`
for its inputs. The experiment requires the external pinned v26 source checkout,
checkpoint, original CNN capture/motion artifacts and the v27 native library and
build receipt. Driver constants deliberately bind the measured original raw
receipts and helper versions. The preceding v26/v27 validation scripts document
their capture/build steps. Regenerating captures may change receipt hashes;
that requires a separately identified new experiment and audit, not editing the
old evidence to match.

Public [v28 evidence](../../../../results/dpvo_feedback_v28.json) contains redacted
receipt data, raw file hashes and per-array metadata. It excludes the real image,
feature, pose-array and checkpoint files. The public aggregate JSON cannot be
passed as a raw capture receipt with the same SHA. This directory is a research
reproduction reference, not a ready-to-run general dataset tracker.

The pure synthetic tests are `tests/test_dpvo_cpu_geometry.py`,
`tests/test_dpvo_feedback_probe.py` and `tests/test_dpvo_feedback_saved_audit.py`.
They require NumPy and pytest and use tiny generated arrays. Jacobian finite
differences and Schur/full-normal comparisons validate the reference equations;
they do not prove CUDA numerical equivalence, trajectory accuracy or real-time
performance. See the [engineering notes](../../../../docs/ENGINEERING_NOTES.md).

The three published test files only adapt helper lookup paths to this directory.
Historical synthetic receipts retain their original test hashes; the
[public v28 test receipt](../../../../results/synthetic_tests_v28.json) records
the rerun and hashes of the actual published copies. Measured implementation
files remain byte-identical to their board-tested versions.

## v29 CPU component and recurrent-feedback experiments

`dpvo_cpu_geometry_vectorized.py` retains the v28 geometry equations while replacing the per-edge assembler with ordered batched accumulation. `run-v29-dpvo-geometry-board.py` accepts the frozen v28 feedback directory and scalar geometry source, runs 32 ordinary and eight separately instrumented BA calls, and saves both steps' systems. `audit-v29-dpvo-geometry.py` checks saved arrays using NumPy; it neither executes BA nor solves a system.

`run-v29-dpvo-ort-feedback-board.py --help` lists the original v26 capture/motion/checkpoint, v27 native library/build receipt, frozen v28 driver/adapter, and v29 accepted candidate/geometry evidence paths. It requires the sigmoid-only candidate manifest and all sixteen accepted board CPU ORT states before creating a session. The ORT package must be available to the board Python environment. Use a new output directory on every execution; inspect the saved status and cleanup fields rather than the ADB exit code. Fixed E864 topology is built once, and every round recomputes correlation, carries ORT state and executes two BA steps. Four original Torch calls verify the same current inputs separately from feedback timing.

The preflight reads correlation from the original SHA-pinned E864 identity-state capture, because the v26 motion graph does not contain a correlation array. That preflight array is only used for input validation; actual feedback computes correlation afresh. Original source, weights, input, topology and feature ownership checks remain enabled. The public JSON files wrap redacted receipts and retain their raw hashes; they are not interchangeable with original board receipts required by these entrypoints.

The experiments use three already-captured views and identity/unit inverse-depth initialization. They do not execute CNNs, official initialization, new-frame admission or a full online trajectory. The component timings and same-input neural comparisons do not establish real-time SLAM. Both new geometry files retain this directory's MIT attribution; the checkpoint's independent license is not inferred from that code license.
