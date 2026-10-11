# Native transport / cache-closed query AB (v44)

The board driver `run-v44-native-closed-cache-ab-board.py` uses the frozen v43
native full-state harness and v42 saved-state helpers. A and B share the native
FP16 packer, session class, metadata callback, nine compiled models and exactly
the same source-pin reporting dictionaries. Independent empty caches are
created before any SDK context. Only B replaces the live-hit `closed` query
with the locked raw-field read described in `README.md`.

## Actual board result

The completed RK3588S experiment uses ABBA twice per state. All 24 output NPZ
files and 72 saved arrays match the first A call for the corresponding state
bitwise; the independent saved-data review passes **1,533 checks**. The same
source-pin reporting payload is used in both methods. Warm callback wall
means exclude each method's first call per state, leaving three calls:

| Saved observation | Native / original cache | Native / closed-query cache |
| --- | ---: | ---: |
| 34 | 83.106 ms | 70.597 ms |
| 127 | 105.655 ms | 97.252 ms |
| 191 | 107.248 ms | 94.043 ms |

Each method completes 56 SDK calls and 56 output releases, closes six used
contexts, and finishes with zero model/DMA bytes. Both methods together
record 72 native attribute queries and 336 native input-pack records. Source
and input hashes remain unchanged and cleanup has no errors. The actual run
SHA256 is `1a2132b70849fe823e3d85617d588f2d1dd5626326adfa4c14c0e718d2f7faf4`.

This is an interleaved fixed-state component experiment. It does not measure
the removed deepcopy in isolation or control the complete thermal/frequency
history; SDK run intervals also differ. The candidate remains unadopted and
has not been tested in a complete recurrent online replay. Original Torch
FP32 rejection, trajectory-quality rejection and the real-time targets are
unchanged.

The [published evidence](../../../../results/dpvo_cache_closed_flag_v44.json)
contains the actual run and the independent saved-array review, including
source and lifecycle hashes. The source protocol is bound separately by
`harness-source-freeze-r1.json`; no execution-source bytes changed after the
measured run.

It retains all six immutable original inputs and complete-edge output order
for observations 34, 127 and 191, ABBA twice per state, 24 output NPZ files and
72 output array descriptors. The inherited checks require byte equality to
the first A call for each state, finite outputs, unchanged inputs, native SDK
success and copy/release/close lifecycle evidence. A failure is preserved in
`run.json`. The three original saved inputs are rejected-feedback states,
not original CPU teachers; bit equality cannot reverse that rejection.

Use every argument of `run-v43-native-full-ab-board.py` unchanged, then add:

```text
--native-harness PATH_TO_FROZEN_run-v43-native-full-ab-board.py
--native-harness-sha256 4c056f07828db7b5fa74914148e5a196840af2275988eb4f4104a459488fa9c2
--closed-flag-helper PATH_TO_v44_closed_flag_cache.py
--closed-flag-helper-sha256 014685c3b7736070bf3b1a1e3a333a928468e013cf454e4acac4ee53da1354b2
--diagnostic-allow-closed-flag
```

The original `--diagnostic-allow-fp32-rejected` and
`--diagnostic-allow-native-fp16` remain required. The output must be a new
directory under the board experiments root. The board guard runs before SDK
or native-library load and output-directory creation; CPU affinity is 4–7
and restored in `finally`. All pinned source/input/model bytes are checked
again after close. Eight generated/fake harness checks passed without any
real inference; `harness-source-freeze-r1.json` binds these sources.

The measured callback, full-graph, cache/group and SDK intervals are nested.
Warm summaries omit the first call per method per state, leaving three calls;
they are not a full trajectory or controlled thermal benchmark. This driver
does not evaluate a new observation, run BA, commit VO state, establish 20 Hz,
validate trajectory accuracy or adopt the cache candidate. DPVO-derived
helpers retain the adjacent Princeton MIT notice. No models, SDK or data are
distributed in this source package.
