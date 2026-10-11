# Cache-hit closed flag query (v44 research)

`v44_closed_flag_cache.py` replaces one operand in the frozen v37 cache:
`self._live[key].metrics()['closed']` becomes
`session_closed(self._live[key])`. The latter holds the same session lock and
returns the same original `_metrics['closed']` operand. The inherited
`execute()` already holds the cache lock, so cache → session lock order stays
unchanged. There is no unlocked flag read or cached/stale boolean.

The new subclass overrides only `_get`. Its body is validated against the
SHA-pinned original AST after exactly that expression substitution. All other
methods remain the same inherited function objects: constructor, `_usage`,
`_evict`, `execute`, `close`, `clear`, and `metrics`. In particular, full metrics
copies used for cold init, model/DMA budgets, history, destruction and public
reporting remain. Invalid/closed sessions cannot become cache hits. SDK/native
packing, models, complete group topology, output ownership and thresholds do
not change. A custom session `metrics()` override is rejected by the binding.

## API

After loading the already pinned native runtime, and before constructing a
new empty candidate cache:

```python
candidate_runtime = closed_flag_helper.bind_runtime(
    native_runtime,
    parent_runtime_path=frozen_v37_runtime_path,
    diagnostic_allow_closed_flag=True,
)
cache = candidate_runtime.BoundedContextCache(specs, factory,
    max_contexts=9, max_model_bytes=original_host_budget,
    max_dma_bytes=original_DMA_budget)
```

The returned API preserves `GroupedSession`, native pack functions and
`execute_full_graph` identities. The old runtime and its class/globals are not
mutated. Existing factory/close source checks must include the new runtime's
`source_pins`; caller pins this helper before import. This helper creates no
SDK context and must not migrate live contexts. The parent v37 runtime remains
`55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d`.

Ten generated/fake contract tests passed. They use tiny synthetic output
arrays and metadata, not real model tensors or SDK calls. Tests accept
`V44_PARENT_RUNTIME` for the byte-pinned parent; repository placement defaults
to sibling `dpvo_grouped_update/v37_grouped_rknn.py`.

The v43 native AB saved clocks identified nested cache/group overhead, without
timing this individual deepcopy. The separate v44 board experiment now
compares native/original-cache against the same native/candidate-cache, with
identical callback, nine model specs and source-pin dictionaries. It uses
three fixed states and two ABBA cycles. All 24 complete outputs and 72 saved
arrays match bitwise. Warm callback wall means, three calls per method per
state, are **83.106→70.597 / 105.655→97.252 / 107.248→94.043 ms** for
observations 34/127/191. The independent saved-data auditor reports 1,533
checks passed. Both methods close all six used contexts, each with 56 SDK
calls and releases; model/DMA bytes return to zero and cleanup has no errors.

These are same-process interleaved component observations, not a controlled
thermal benchmark or a direct measurement of the removed copy alone. SDK
run intervals also differ across calls. The cache candidate has **not** been
integrated into a complete recurrent replay. Numerical acceptance against
original Torch, trajectory accuracy, 20 Hz and default adoption are not
established. Historical FP32/quality rejection remains. See [AB protocol and
actual evidence](AB.md) for the measured boundary; execution sources are
unchanged by this documentation update.

DPVO-derived code is Copyright (c) 2022 Princeton University, MIT; see the
byte-preserved adjacent LICENSE. Source licensing does not grant separate
checkpoint or compiled-model rights. No weights, SDK or dataset is included.
