# Original VGGT FC1 fusion diagnostic

This experiment changes only diagnostic output branches of the original
frame0/global0 pair. It does not replace VGGT, remove layers, or provide a
production backend. See [measured results](../../../../results/original_fc1_tap_v61.json).

Run construction, ORT validation, compilation and SDK execution on RK3588S.
The scripts require the pinned original v55 pair manifest, v53 capture and
RKNN Toolkit2/runtime 2.3.2. Weight-bearing graphs and captured arrays are
private prerequisites, not repository downloads. Their original-model license
and the repository code license are separate.

1. Run `build_fc1_tap.py` with `--baseline-manifest`, `--capture-dir` and a
   new `--output-dir`. It appends the original FC1+bias tensors before GELU
   as outputs. Removing those declarations must recover the original ONNX
   bytes. Original output equality is checked with board CPU ORT before one
   non-quantized FP16 build.
2. Run `probe_fc1_tap.py` with `--candidate-manifest`, its actual
   `--candidate-manifest-sha256` and a new `--output-dir`. The normal ABBAABBA
   sequence retrieves all outputs, including the extra taps. Separate fresh
   contexts collect profiles. No extra-output time is subtracted.
3. Run `build_fc1_tap_transposed.py` with the same original inputs, a new
   output directory and `--tap-reference-dir` pointing to step 1. It appends
   two transpose branches with permutation `[0,2,1]`. Removing those branches
   and outputs must recover the original graph bytes. Inverse-transposed ORT
   taps must match the saved step-1 taps byte for byte.
4. Run `probe_fc1_tap_transposed.py` against step 3's actual manifest and
   SHA256. This additionally tests whether this exact SDK supports retrieving
   only original output indices 0 and 1 during normal runs. It validates all
   four output declarations first and fails without fallback if the subset
   request fails. The recorded runtime rejected it with `-5`; that run stopped
   and released both contexts. This is an empirical SDK check, not a guarantee
   for other runtime versions.
5. `probe_fc1_tap_transposed_all.py` is the separate, explicit all-output
   experiment after that failure. It retrieves all four candidate outputs in
   normal and profile calls. Transposed taps removed the extra NPU transposes,
   but normal calls still took 54.15 ms versus the matched baseline's 38.68 ms.
   Both tap candidates are rejected for latency and are not integrated.

Use the isolated compiler environment for builders and the deployment
environment (`deploy/rk3588/env.sh`) for probes. Each probe uses NPU mask 7
and CPU affinity 4–7, then restores affinity. No script changes clocks or
governors. Normal timings include SDK input, execution, requested output
retrieval, owned copies and release; array audits, saving, initialization and
profile contexts are excluded and must not be presented as complete SLAM
latency. This is one original pair and one real captured input, not a trajectory
or real-time acceptance test.

`python3 -m unittest discover -s deploy/rk3588/validation/original_fc1_tap`
runs generated output-ownership checks without a model or SDK.
