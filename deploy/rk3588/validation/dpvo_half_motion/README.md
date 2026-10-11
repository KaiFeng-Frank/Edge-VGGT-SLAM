# DPVO half-resolution initial-motion threshold trial

This independent CPU research trial changes initial motion rejection from
`probe < 2.0` to `probe < 1.0`. It starts from the original SHA-pinned half
policy, keeping keyframe threshold 15, M16/L4/R8/O4, image/K preprocessing,
original 94 state-dict entries, shared sampling/depth RNG, complete Update/BA and EOF.
It does not stack the v44 keyframe7.5 trial. Original VGGT defaults are unchanged.
The actual RK3588S CPU replay completed all192 observations and produced181
fresh poses. Earlier initialization improved coverage, while all three main
errors exceeded the20% limit against full FAST. The candidate is not adopted.
See [the actual result card](../../../../results/dpvo_half_motion_v45.json).

## Actual CPU replay and saved evaluation

The pinned run SHA is
`c6a3ef8a5a063fc8dfa1319841ea782f38c9cc1822ce7058aa312cec4665f7e3`.
Its original schema is `v45_half_motion1_original_Torch_CPU_research_replay`,
with status `completed_unpaced_half_motion1_Torch_CPU_research_not_realtime`.
The first fresh observation moved from34 to11. Its initial eight admissions
are0,1,2,3,5,8,9,11; four initial observations were rejected. The three actual
populations keep their individual coverage visible:

| Run | Fresh/all192 | First fresh index | Used/all162 one-second endpoints |
| --- | ---: | ---: | ---: |
| Candidate half motion1/KF15 | 181/192 (94.2708%) | 11 | 151/162 (93.2099%) |
| Original half motion2/KF15 | 158/192 (82.2917%) | 34 | 128/162 (79.0123%) |
| Original full FAST | 179/192 (93.2292%) | 13 | 149/162 (91.9753%) |

Common-fresh populations are selected from the actual source identities before
GT/error filtering. These are separate pairwise comparisons, with different
common populations:

| Reference | Common fresh / endpoint pairs | ATE RMSE change | Rotation RPE RMSE change | Translation RPE RMSE change |
| --- | ---: | ---: | ---: | ---: |
| Original half CPU | 158 / 128 | +7.2214% | -21.4205% | +13.1601% |
| Original full FAST CPU | 179 / 149 | +119.9226% | +30.7296% | +105.4787% |

The three main metrics pass the20% numerical limit against the already reduced
half workload and fail its10% limit. Against full FAST, both10% and20% limits
fail; the coverage increase does not repair this quality gap. Additional
translation-direction RMSE changes are -12.0706% versus half and +51.3653%
versus full. Direction is reported as a diagnostic, not an added acceptance gate.

Active unpaced replay took119.443248043s:181 fresh divided by this interval
is1.515364Hz. Fresh software admission-to-publication P95 was780.780036ms;
peak RSS was368.9375MiB. These clocks include read/preprocessing, first-use
construction, original neural/geometry work, capture I/O and publication.
EOF12 offline Updates took a separate3.580708026s and do not count as fresh.
The original half/full runs have different thermal and timing histories;
these measurements do not establish a causal speed change. The20Hz and
P95<=100ms goals remain unmet.

Saved checks reread six actual capture NPZ files containing27 arrays at
observations11,127,191. File/array hashes, shapes, dtypes, finite/domain checks
and full callback identities match their descriptors. The first capture is
the first complete initialization callback at11, followed by the remaining
initialization Updates; it is not the final published pose or old observation34
gold. The original Torch callback AST equals the frozen cached parent.
All192 source rows,94 weight entries, initial RNG and half terminal RNG
digests, source stability, context/map/encoder closure and empty cleanup were
checked from saved metadata. Raw RNG state was not reconstructed and no model
or native library was rerun by this review.

## Source change and loading guards

`dpvo_stream_motion1.py` is an independent copy of frozen
`../dpvo_stream/dpvo_stream.py`. Exactly two lines change: the rejection limit
and the truthful reason `initial_motion_below_1`. All other source bytes,
including original source pins, numerical validation, geometric/BA code,
transactional state, frame ownership and EOF reconstruction, remain identical.
The original stream is not modified.

The driver derives from the frozen v41 cached CPU driver. The existing
`--stream-module` selects the candidate, while the additional required
`--baseline-stream-module` selects the original source. Before neural loading,
both sources are checked by SHA and the candidate must equal the exact two-line
transformation of the original. `--approved-stream-sha256` must equal the fixed
approved candidate SHA; it cannot authorize an arbitrary module. The frozen
parent's original stream declaration and upstream source/manifest guards remain
checked, and there is no parent-module monkeypatch or import override.

| Artifact | SHA-256 |
| --- | --- |
| Frozen original stream | `295233513c1b670c953db25bbf4bfad8bdb72b2001f40ac3e1bf5d17083f9bde` |
| Candidate motion1 stream | `bcc7cea80c758b49c2abef1a4bd961ff130a0144fe2d0f7e5db76009754ca8ce` |
| Original half policy | `cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a` |
| Candidate policy | `ca0f1632f73765c913cd707c03df54d45b03d136f45e33b6cc034ac85cb2e7b4` |
| Candidate driver | `3b0a5b809cab8277e1275bb0b3d4bacd99c14622e9b4091d980286276457543a` |

The entire runtime `StreamConfig` equals the original half configuration.
Metadata declares the changed hardcoded rejection threshold separately. The
driver pins the original cached driver/policy as additional provenance and
rejects a second policy change, including keyframe7.5.

## Motivation and saved-check rules

Original Update delta is added to projected feature-grid coordinates, and
motion probe takes the median norm of that delta. The unit is a feature pixel;
image intrinsics are divided by original RES=4 exactly once. Keeping motion2 at
half resolution doubles the equivalent original-image displacement range.
Choosing `2 * 0.5 = 1` follows this unit change, without GT parameter selection.
Learned network responses are not claimed to scale exactly with image resizing.

The frozen cached half CPU run's first probe, at source observation1, is
1.0345186591148376. Actual saved comparison confirms source0 is identical
after removing clocks, then the same probe at1 is rejected by motion2 and
accepted by motion1. Neither result is fresh at1. Once a frame is admitted,
later references, graphs, deltas and trajectory must come from a real feedback
replay; old saved probes do not predict later admissions.

`evaluate-v45-half-motion.py` requires actual completed v45, original v41 half
CPU and full FAST CPU receipts, their original manifests and SHA pins. It
recognizes each actual schema without rewriting it, verifies the independent
stream source change and checks source0/nonclock equality plus the first
admission decision. It reuses frozen v39 fresh/EOF/ratio helpers and v34 math,
reports both common-fresh comparisons and each run's full192-source coverage,
and keeps the all-source one-second RPE endpoint plan fixed before success
filtering. Post-run GT association stays nearest within20ms and one-to-one;
each population gets one positive Sim3 evaluation fit. EOF12 offline Updates
and reconstructed/rejected/initialization rows never become fresh observations.

The main 10% and 20% error limits stay on ATE, rotation RPE and translation RPE
RMSE. Translation direction is an additional diagnostic. Missing ratios are
not passes, and individual source/endpoint coverage remains visible. Original
weights/RNG and cleanup are checked from saved metadata, not a model rerun.

New `gold` captures denote the actual same-call original Torch output of this
motion1 path. They do not denote an old baseline teacher. Old full Update gold
is not reused after admission changes, even if a new capture has the same shape
or a similar observation number.

## Source-only checks and execution

From this directory with NumPy and the unchanged sibling source directories:

```sh
python3 -B test_v45_half_motion.py
python3 -B test_v45_half_motion_evaluation.py
python3 -B run-v45-half-motion-cpu-board.py --help
python3 -B evaluate-v45-half-motion.py --help
```

`DPVO_V45_VALIDATION_ROOT` can select the directory containing the original
sibling validation trees. Ten generated/fake tests cover strict threshold
bounds, depth draws on rejection, nonfresh initialization, failure before state
commit, exact source guards, true source clocks, first admission divergence and
the unchanged primary gates. These tests do not execute a model, SDK, native
library, BA or real image processing.

The board driver retains the original v41 CLI paths and additionally requires
`--baseline-driver`, `--baseline-policy`, `--baseline-stream-module` and
`--approved-stream-sha256`. Use deployment-manifest paths and the pinned
external checkpoint, native correlation library/build receipt and original
dataset. Repository paths do not imply board deployment paths. Only the root
operator executes the actual board CPU replay.

The saved evaluator adds `--candidate-stream-module` and
`--baseline-stream-module` to the v44 evaluator's saved-input CLI. It requires
a fresh output path and completed SHA-pinned receipts; it does not run neural
code to manufacture missing results. The 20Hz/P95<=100ms goal, original FP32
gates and no-adoption boundary remain unchanged. A CPU coverage/quality trial
alone does not validate that end-to-end goal, physical capture, mapping or
endurance.

See [source-loader-audit-r1.json](source-loader-audit-r1.json),
[static-generated-tests-r1.json](static-generated-tests-r1.json) and
[actual-motion1-evaluation-r1.json](actual-motion1-evaluation-r1.json),
[actual-motion1-saved-evidence-r1.json](actual-motion1-saved-evidence-r1.json),
[audit-v45-motion-saved.py](audit-v45-motion-saved.py) and
[source-freeze.json](source-freeze.json). No model, checkpoint, image, NPZ or
compiled binary is published here. DPVO-derived source attribution: Princeton
Vision & Learning Lab, MIT; see [LICENSE](LICENSE).
