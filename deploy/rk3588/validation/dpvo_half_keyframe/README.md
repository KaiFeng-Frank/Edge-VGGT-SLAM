# DPVO half-resolution keyframe threshold trial

The actual original-weight CPU trial changes only the half-resolution keyframe
threshold from 15 to 7.5. It improves the three primary errors against the
original half CPU run, but still misses the 10–20% quality-loss target against
full FAST. The candidate remains research and is not adopted. Original VGGT
defaults, M16/L4/R8/O4, motion probe 2, original weights, full Update/BA, sampling
RNG, cached undistortion, failure boundaries and offline EOF are preserved.

See the combined [actual v44 quality receipts](../../../../results/dpvo_half_keyframe_v44.json).

## Actual quality and coverage

Both comparisons use the same 158 common fresh source observations and 128
eligible pairs from the original 162 all-source, one-second RPE endpoint plan.
Selection uses actual fresh identities before GT association or error filtering.
Ratios below are candidate/reference RMSE on each common population; lower is
better.

| Reference | ATE | Rotation RPE | Translation RPE at GT evaluation scale | Additional translation-direction diagnostic |
| --- | ---: | ---: | ---: | ---: |
| Cached half CPU, keyframe 15 | 0.87016 | 0.95114 | 0.90593 | 1.24050 |
| Full FAST CPU | 1.54145 | 1.60995 | 1.57433 | 1.32610 |

The headline 10% and 20% limits apply to ATE, rotation RPE and translation RPE
RMSE. Direction remains an additional diagnostic, not a new acceptance standard.
All three primary ratios fail both limits against full FAST. Coverage is reported
separately and is not hidden by common-frame filtering.

| Actual saved run | Fresh / requested | First fresh source index | Used / fixed RPE pairs | Input-active elapsed | Fresh / full replay denominator |
| --- | ---: | ---: | ---: | ---: | ---: |
| Half keyframe 7.5 trial | 158 / 192 | 34 | 128 / 162 | 114.157 s | 1.384 Hz |
| Cached half keyframe 15 baseline | 158 / 192 | 34 | 128 / 162 | 107.043 s | 1.476 Hz |
| Full FAST CPU baseline | 179 / 192 | 13 | 149 / 162 | 348.556 s | 0.514 Hz |

These are separate serial, unpaced replay histories. They do not establish a
causal speed ratio, physical-camera frequency or capture/queue-age guarantee.
The full FAST receipt reports its original CPU ORT encoder; the half runs use
original Torch CPU FP32. EOF performs 12 offline Updates and never contributes
fresh poses. The trial reports 395.414 MiB peak RSS and 3.295 s offline EOF.

## Why 7.5, and what was isolated

Pinned original DPVO `projective_ops.flow_mag` and the frozen stream compare
Euclidean displacements of projected feature-grid XY coordinates. Image K is
divided by original RES=4 once. For corresponding rays, depths and SE3 poses,
half-resolution focal lengths and principal points give
`feature_xy_half = 0.5 * feature_xy_full - 0.25 / 4`. The offset cancels in
displacements, so geometric flow scales by 0.5. This supports `15 * 0.5 = 7.5`
without selecting a threshold from GT errors. It does not make resized learned
features, sampled patches or the changed workload numerically equivalent to
full FAST.

In the saved original half run, 62 of 157 keyframe checks fall in `[7.5,15)`.
The first is source observation 54. The real trial verifies that all frame
fields for observations 0–53 match the cached CPU baseline exactly after
removing only keys ending `_seconds` or `_monotonic`. At 54 both runs report
the same flow, 8.138959407806396: baseline deletes observation 53 and the trial
retains it. The pose published for 54 still matches because its Update occurs
before keyframe deletion. Later graph and pose changes come from the actual
feedback replay; the 62 saved checks are not a predicted retained-frame count.
Total keyframe deletions change from 155 to 148.

The first complete initialization capture at observation 34 was also reread:
all six inputs and three original Torch outputs have matching shape, dtype,
finite domain and raw bytes against the cached half CPU capture. Later captures
are not paired with old gold after feedback diverges. Close, source stability
and original-weight stability are checked from saved metadata, not by rerunning
the network or SDK.

## Remaining source evidence

Motion probe is the median norm of learned Update delta, which is added to
feature-grid projected coordinates. Its units are feature pixels, but learned
responses need not scale exactly after image resizing. Both half runs reject
27 startup observations and first emit fresh poses at index 34; full FAST
rejects six and first emits at 13. Twenty of the 27 rejected half probes have
saved values in `[1,2)`. This motivates a separate motion-threshold-only CPU
trial. It does not predict its admissions: accepting a frame changes the next
probe reference and initialization graph. No motion-threshold trial is executed
or adopted here.

Full versus half also changes M48 to M16, L11/R16/O7 to L4/R8/O4, keyframe index
4 to 2, image information and the physical support of fixed 3×3 patches and
radius-3 correlation. The remaining quality gap cannot be assigned solely to
the threshold from this trial.

## Reproduce source and saved-result checks

From this directory, run the generated tests with NumPy and the unchanged
sibling source directories present:

```sh
python3 -B test_v44_half_keyframe.py
python3 -B test_v44_half_keyframe_evaluation.py
python3 -B run-v44-half-keyframe-cpu-board.py --help
python3 -B evaluate-v44-half-keyframe.py --help
```

`DPVO_V44_VALIDATION_ROOT` may select the directory containing the sibling
`dpvo_half_workload/` and `dpvo_geometry/` trees. The eight tests use generated
arrays/JSON only; no Torch, ORT, SDK, native library, BA or real image execution.

The board driver keeps the pinned v41 cached CPU driver's existing CLI paths and
adds required `--baseline-driver` and `--baseline-policy` paths. Supply the actual
deployment paths for the original byte-pinned sources, external checkpoint,
correlation library/build receipt and data; the repository directory is not a
board deployment manifest. The new policy and driver are independently pinned,
and the driver rejects an undeclared second runtime policy change.

The saved evaluator requires completed SHA-pinned v44, v41 and full FAST runs,
their original manifests, both policies, the trial driver, the frozen v39
evaluator and v34 math, and SHA-pinned TUM GT. It reads each actual schema without
rewriting it. It reuses the frozen fresh/EOF/ratio and trajectory math functions,
reports both common-population comparisons, individual all-source coverage,
fixed endpoint coverage, first divergence and unavailable metrics. GT is only
post-run evaluation: one nearest association within 20 ms and one positive Sim3
per population, with no segment fit or runtime GT input.

Sources and check receipts are listed in [source-freeze.json](source-freeze.json).
No checkpoint, image, NPZ, compiled library or model binary is published here.
DPVO-derived math and lifecycle attribution: Princeton Vision & Learning Lab,
MIT; see [LICENSE](LICENSE).
