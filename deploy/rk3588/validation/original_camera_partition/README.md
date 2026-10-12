# Original CameraHead output-channel partition

The complete original CameraHead measured **213.300 → 177.352 ms** at
300 MHz, a 16.85% reduction in this short matched diagnostic. All original
parameters, four trunk blocks and four refinement iterations remain.
This candidate is not enabled in the full-model runtime. Its outputs differ
from the previous FP16 graph and both fail the original FP32 strict gate;
there is no new trajectory, replay or real-time acceptance.

## Measured scope

The original two-image capture supplies raw camera tokens `[1,2,2048]`.
Reconstructing the original CPU head reproduces all four archived poses
byte for byte. Hooks capture the real trunk-0 FC1 inputs and post-GELU
outputs at each refinement. No aggregator is rerun for this capture.

| Diagnostic | Normal complete warm call | Calls in warm mean |
| --- | ---: | ---: |
| Original trunk-0 FC1 + GELU, unsplit | 6.270834 ms | 11 |
| Same FC1, 2 output-channel parts | 6.442387 ms | 11 |
| Same FC1, 4 output-channel parts | 3.365477 ms | 11 |
| Complete original head, existing graph | 213.299646 ms | 3 |
| Complete original head, four trunk FC1s partitioned | 177.352407 ms | 3 |

The component test uses three balanced rounds over the four genuine
refinement inputs; only each graph's first call is excluded. The complete-head
test uses `ABBAABBA` on one fixed raw-token pair. Each head independently
starts with the original empty-pose token, performs all four SDK steps,
accumulates **unactivated FP32** predictions and computes all four original
activations. Normal timing includes submission, synchronous run, output
conversion/copy/release and the recurrence. Hashes, file writes and error
metrics are outside it. Profiling uses separate contexts with flag 8;
profile timings are not substituted for normal calls.

Actual component profiles show that two parts retain single-core
`ConvExGelu`. Four parts become four three-core `Conv` operations, an internal
`Concat`, and one `exGelu`. The complete-head profile confirms the same change
at all four trunk FC1s. It does not show independent branches running in
parallel; the individual convolutions use all three cores. No diagnostic
SDK output is added.

All component FP32 ONNX checks (12) and complete-step FP32 ONNX checks (8)
pass against the original captured Torch references. The FP16 component
strict comparisons fail. Four-part component outputs differ from the unsplit
FP16 results by up to 0.0009765625. Final-pose relative L2 error against the
original FP32 reference is 0.00787266 for the existing head and 0.00787447 for
the partitioned head; both have all 18 scalars outside the strict tolerance
on this input. These are encoded-pose tensor errors, not ATE/RPE.

SDK-reported weight allocation stays 432,555,136 bytes for both complete
head graphs; total DMA is 869,425,152 / 869,842,944 bytes. Of 68 original
initializers, 60 remain intact and eight FC1 weight/bias tensors become 32
slices, for a total of 92. No learned parameter values are added or duplicated.
All other original node/initializer protobufs
and the two-input/one-delta interface are checked unchanged. The original
empty-pose parameter remains outside the graph.

## Reproduction on RK3588S

Use the original private v15/v16 capture, export and compiled baseline
described by the existing CameraHead validation scripts. The new scripts pin
those receipts, original weight values and graph hashes. Model arrays, ONNX,
RKNN and SDK binaries must remain private under their separate licenses.
All actual Torch, ORT, compilation and SDK execution for this diagnostic ran
on RK3588S. The host only handled source and saved-array audits.

Run each command with a new output directory. The compiler uses the isolated
RKNN Toolkit 2.3.2 environment; capture uses the original board Torch 2.3.1
environment. Invoke the compiler environment without the production
`PYTHONPATH`, `PYTHONHOME` or `VIRTUAL_ENV`, with the board sysroot library
path. Use process affinity CPUs 4–7 and retain the measured 300 MHz setting.
`run_board_stage.py` wraps a child command, records temperatures/frequency and
stops its own process group on a 75°C sample or timeout; it never writes
frequencies. Stage names must be new because receipts cannot be overwritten.

1. Run `capture_camera_fc1_board.py --output-dir NEW_CAPTURE` with production
   Python. The four saved original poses must remain byte-identical.
2. Run `compile_camera_fc1_splits_board.py --chunks N --output-dir NEW_GRAPH`
   in separate compiler processes for `N=1,2,4`.
3. Run `probe_camera_fc1_board.py --capture NEW_CAPTURE --capture-sha256 SHA
   --compiled GRAPH1 SHA1 --compiled GRAPH2 SHA2 --compiled GRAPH4 SHA4
   --output NEW_NORMAL` with compiler-environment Python (ORT and typed SDK).
   SHA values bind each directory's JSON receipt, not its weight file.
   Repeat in a new directory with `--profile` for operator placement.
4. Run `compile_camera_step_split_board.py --output-dir NEW_HEAD_GRAPH` in
   the compiler environment. Keep both compiler scripts in the same directory.
   The parent ONNX defaults to the pinned original v16 step graph.
5. Run `probe_camera_step_split_board.py --candidate-dir NEW_HEAD_GRAPH
   --candidate-sha256 SHA --check-onnx --output-dir NEW_CHECK`.
6. With the passed check's receipt hash, run the same probe with
   `--candidate-dir NEW_HEAD_GRAPH --candidate-sha256 SHA
   --check-receipt NEW_CHECK/result.json --check-sha256 CHECK_SHA
   --output-dir NEW_HEAD_NORMAL`. Use another directory and `--profile`
   for the separate placement check.

The [result record](../../../../results/original_camera_partition_v69.json)
binds the source, compiled artifacts, actual outputs and monitor receipts.
No full-model speedup may be inferred by multiplying component savings by
layer counts. Generalization across images, trajectory/map quality, physical
camera capture, long-run stability and the real-time target remain unverified.
