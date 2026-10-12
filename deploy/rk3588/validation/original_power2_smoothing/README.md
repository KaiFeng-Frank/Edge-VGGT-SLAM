# Original pair power-of-two smoothing at 1 GHz

This component experiment is **not adopted**. Reversible channel scaling before
W8A8 calibration gives only a small error improvement on the original
frame0/global0 pair. It has no new complete-model or SLAM speed result.

| Route | Warm complete SDK route, mean | Image-disjoint input: frame relative L2 | Image-disjoint input: global relative L2 |
|---|---:|---:|---:|
| A: original FP16 | 32.682 ms | 0.422% | 0.473% |
| B: ordinary W8A8 | 20.957 ms | 9.545% | 12.352% |
| C: power-two-smoothed W8A8 | 20.666 ms | 9.393% | 12.220% |

Each route has 22 warm measurements: `ABCCBA` repeated six times for each of
two real inputs, excluding the first call per input and route. The second input
is part of calibration; its frame/global errors are 9.457%/12.095% for B and
9.317%/11.949% for C. Errors use the original FP32 ONNX outputs, with a separate
original Torch comparison on the image-disjoint input. All three NPU routes
fail the fixed `rtol=1e-4, atol=1e-5` tensor gate.

C is 36.77% faster than A in this component, but B already provides most of
that difference. The 1.39% C/B timing difference does not establish a new
speed improvement. Small local error improvements do not establish acceptable
full-model depth confidence, pose, trajectory or map quality. We stopped at
one candidate and did not compile all 24 pairs or integrate it into replay.

The latest complete timestamp replay remains the preceding v77 result:
**0.899908 fresh poses/s, complete input-to-pose P95 1.366577 s** on a 16 GB
RK3588S, below the real-time targets. No camera, 30-minute, physical 8 GB or
closure acceptance is added here.

## Transformation and numerical checks

Only the original VGGT checkpoint and its original pair graph are used. All
nodes, layers, attention, tokens and both complete pair exits are retained.
No other model or pose frontend is added.

Eight original saved calibration inputs are executed on the board through an
augmented original FP32 graph. Its outputs retain both original exits and add
the four actual Norm activations: frame0/global0 norm1 before QKV and norm2
before FC1. A block-entry tensor is not used as a substitute for FC1 input.

For each input channel, use alpha 0.5 and

```text
s = 2 ** rint(0.5*log2(max(activation_absmax, 1e-5))
             - 0.5*log2(max(weight_absmax_over_output_channels, 1e-5)))
gamma' = gamma / s
beta'  = beta / s
W_T'   = W_T * s[:, None]
```

Exactly 12 original graph initializers change. The original checkpoint file
does not change; transformed graph parameters are not literally unchanged.
Each inverse transformation reconstructs original parameter bytes. More
importantly, both complete FP32 outputs on both real inputs are byte-identical
to the original graph under ORT 1.16.0 with graph optimization disabled.
Parameter reversibility alone would not prove floating-point equivalence.
One evaluation input has no image SHA overlap with the eight calibration pairs.

The activation/weight balancing principle is described by the
[SmoothQuant paper](https://arxiv.org/abs/2211.10438) and its
[reference scaling implementation](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py).
This implementation applies a power-of-two restriction to the original VGGT
ONNX parameters; it imports no implementation, checkpoint or additional model
from that project. Quantization uses the same RKNN 2.3.2 W8A8/channel/normal
configuration and original eight input files as B.

## Runtime and evidence

All actual calibration, FP32 checking, compilation and NPU execution happened
serially on RK3588S. Host operations inspect source and saved arrays only.
FP32 checking and compilation used 300 MHz without changing clocks. The two
NPU jobs used the previously authorized controller at 1 GHz, core mask 7,
with a 180-second timeout and 75 C limit. All 15 controller samples and 156
per-call platform samples reported 1 GHz; maximum observed temperature was
53.615 C. Both jobs restored the original 300 MHz/userspace/min/max state.

All three routes use FP32 input with `pass_through=0` and two FP32 outputs with
`want_float=1`, even though B/C query as affine INT8. Timing includes input API,
run, output API, owned copies, release and finite checks. Hashing, NPZ saving,
platform queries, initialization and profiling are outside that component
time. This component is an older two-block partition and cannot stand in for
the current four retained groups or complete input-to-pose latency.

The normal job completed 72 calls and released all 144 outputs; the separate
flags=8 job completed six calls and released all 12 outputs. Each job created
and destroyed three contexts. Repeated outputs are byte-identical. Profiles
have 54 NPU operator rows and three CPU I/O rows per graph; B/C NPU rows are
INT8-tagged, including the eight original linear operators. Operator-profile
times are not substituted for ordinary API timings.

B and C have identical queried input/output quantizer descriptors. Saved-array
affine roundtrips alone give about 4.13% input, 2.96% frame-output and 3.40%
global-output L2 error, without clipping on the image-disjoint input. These
are isolated ties-to-even rounding diagnostics, **not** propagated inference
error bounds or proof of the SDK's precise rounding convention. Internal Norm
scaling does not change these measured external quantizers.

The frozen probe retains a known failure-reporting limitation: if `outputs_get`
fails with partial buffers, its aggregate acquired/released counters need not
match. It still attempts release and marks the job failed. No such failure
occurred in these runs; the successful counters and saved outputs are audited.

- [Summary](../../../../results/original_power2_smoothing_v78.json)
- [FP32 check](check-result.json), [complete check receipt](check.json), [compiler receipt](compile.json)
- [Normal measurements](normal-result.json), [separate profiles](profile-result.json)
- [Normal controller](normal-control.json), [profile controller](profile-control.json)
- [Saved quantizer/profile analysis](saved_quantizer_profile_analysis.json)
- [Independent saved-result audit](saved_result_audit.json)

## Reproduction

The directory contains the exact executed checker, compiler and probe sources.
The checker/compiler require the pinned v55 original pair graph, v56 calibration
manifest and arrays, v53 reference captures, original checkpoint provenance,
and the private board toolchain described by their source checks. The NPU probe
requires the original typed ABI, vendor header and SDK, whose hashes it checks.
Weights, ONNX/RKNN graphs, SDK binaries, activation arrays and scales are private
artifacts and are not distributed here; their independent licenses still apply.

[board-commands.json](board-commands.json) records the exact board commands;
the normal/profile shell launchers record their child commands. They use the
historical private experiment paths. For a new run, stage these sources into
a new board directory and choose unused result/control paths: runners reject
an existing output directory. Complete the checker first, bind its new result
SHA and checker-source SHA into the compiler call, then bind the new compiler
receipt SHA into the probe. Do not reuse old receipt hashes for regenerated
artifacts. Use the bounded frequency controller for the normal and profile
launchers; the probe itself only verifies frequency and temperature.

To reproduce the secondary analysis of private saved arrays, with no model
or SDK execution:

```sh
python3 analyze_saved.py --evidence /path/to/private/v78 --output /tmp/v78-analysis.json
```

The input layout is `check-r1/`, `normal-r1/` and `profile-r1/`, as named in the
receipts. Public metadata and hashes permit identity checks; raw-array
recomputation requires access to those private captures.

For the independent audit, place `audit_saved_results.py` in that private
evidence root beside the frozen sources and run it there. It verifies saved
array identities, fixed-gate errors, repeats, counters and controller reports
without importing or executing a model.
