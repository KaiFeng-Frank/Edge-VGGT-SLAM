# Original native518 basis quantization

The tested mixed-precision policies are rejected. A/F produce nonfinite global
outputs; finite B still has large conditional error. Keeping its full attention
path FP16 reduces that error but does not pass the unchanged precision gate.
No complete-model speed or quality gain follows from this component screen.

The same six real TUM inputs start before original frame0, after all 24 DINO
layers per image. Four calibration pairs are `[0,8]`, `[16,24]`, `[48,56]`,
`[64,72]`; two heldouts are `[96,104]`, `[112,120]`. All source-image identities
are disjoint. A is original frame0/global0, F folds affine/LayerScale with
identity basis, and B retains the mean-preserving basis between these two
blocks. All 36 original learned values, native tokens, keys and operations remain.

All 18 same-input CPU FP32 calls passed relative L2 ≤1e-5 / maximum error
≤1e-3; independent strict diagnostics also passed. Saved-only audit verified
all 36 endpoints. These conditional two-block references are not full original
Torch predictions or trajectory references.

Manual W8A8 calibration used only four inputs. All three step1 artifacts and
step2 builds completed; initial 180-second A and 600-second B deadlines and a
missing isolated PyYAML dependency were preserved as clean failures before
successful retries. Offline calibration peak RSS was 12,323–12,397 MiB on the
current 16 GB board. This is neither inference memory nor 8 GB calibration proof.

Compiled tables reveal A/F's INT8 canonical frame output sharing the global
reshape buffer. B has a separate FP16 boundary and CPU transpose. Six residual
Adds being FP16 did not protect the whole residual stream. Normal/profile calls
on calibration pair0 reproduced bytes, including NaNs, with all owners closed.
The production finite guard remained; diagnostics retained invalid owned copies
only after SDK release, without changing values.

| Pair0 policy | Frame relative L2 vs original A FP32 | Global relative L2 | Global finite |
| --- | ---: | ---: | --- |
| Original A FP16 | 0.1725% | 0.6073% | Yes |
| Mixed A / F | 4.62% / 4.81% | Unavailable | No |
| Mixed B | 0.6741% | 20.7558% | Yes |
| B, full attention FP16 / MLP INT8 | 0.6689% | 8.6424% | Yes |

The last policy preserves the same B graph and prior calibration. All four new
outputs are finite and flags0/8 byte-identical. Both same-input original A and
own B FP32 comparisons still fail numeric and strict gates; reject this candidate.
It is not a basis cure or proof of the sole error source. Actual flags8 profiling
matches 84 compiled rows: 63 FP16/NPU, 17 INT8/NPU and four FP16/CPU. Its three
FP16 SDPA rows total 391.604 ms, 52.03% of the diagnostic operator sum; this is
neither normal-call latency nor a causal cross-run speed comparison. The
canonical frame-output CPU transpose takes 118.928 ms (15.8% of the diagnostic
sum), motivating consumer-layout review without predicting a removable gain.

An unchanged mixedA dump reproduces both final outputs byte for byte. Independent
audit verifies all 102 dumps and first observes nonfinites at global-input
Reshape057, after the finite frame boundary and before global normalization.
This localizes the observed boundary without proving a particular bitcast,
alias cause or SDK bug. A config-only `remove_reshape=True` discriminator was
rejected for inconsistency with step1, producing no RKNN model or runtime.
Mixed-precision runtime heldouts remain unused; the FP32 checks did use all six.

Capture balanced 12 contexts/48 runs/48 buffers. The original calibration
profile balanced 10/10/20; the full-attention follow-up balanced 2/2/4 and its
controller observed 58.230°C, restoring the prior 1 GHz configuration. CPU
preparation used 70°C/63°C cooling and a 75°C hard limit without frequency
writes. NPU diagnostics used bounded restoring 1 GHz/75°C/180-second controls.
Debug/profile time is not deployment performance. No map, trajectory,
fresh-camera realtime/P95, stability or 8 GB acceptance is claimed.

Reproduction requires private pinned original sources/checkpoint, compiled
DINO groups, R126 graphs, calibrated artifacts and input captures. These are
not shipped or automatically downloaded. Recorded board launchers require
fresh output paths and the exact staged dependencies; source identities are
listed separately. See the [aggregate record](../../../../results/original_native518_basis_quantization_v128.json).
Repository code, original weights and vendor dependencies have separate licenses.
No captured arrays, weight-bearing graphs or vendor binaries are included.
