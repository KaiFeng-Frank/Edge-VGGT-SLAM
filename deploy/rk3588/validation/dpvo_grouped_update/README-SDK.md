# Complete source-batch Update SDK research

This diagnostic preserves the original DPVO 50 Update parameter values and complete dependency groups. Each compiled graph uses its actual `[B,C,D,48]` shape; no sources, targets or patches are padded or dropped. Default VGGT execution is unchanged. This is a different-model research path, with no ORB, LK or PnP frontend.

The frozen grouped CPU/ORT export is numerically rejected at `rtol=1e-4, atol=1e-5`. Both compiler and hardware initialization require an explicit diagnostic flag to use it. `candidate_accepted` remains false even if an individual component passes. The SDK result is never submitted into VO state by this module.

## Compile one actual shape

Use the existing isolated RKNN Toolkit 2.3.2 / Python 3.11 toolchain. Compilation writes only into a new output directory, sets its working directory and temporary directory there, and performs no model inference. Host compilation hashes the export manifest and selected ONNX; unavailable board source paths are explicitly declarations retained in that manifest.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TOOLCHAIN_SITE" python -B compile-v37-grouped-update.py   --manifest "$EXPORT/manifest.json" --manifest-sha256 "$MANIFEST_SHA"   --model-key B8_D8_M48 --diagnostic-allow-fp32-rejected   --output-dir "$NEW_COMPILE"
```

The original configuration remains RK3588 / FP16 / optimization level 3 / `do_quantization=False`. Normal SDK execution and profiling are separate contexts/processes, using core mask 7 and initialization flags 0 or 8.

## Reusable hardware interface

`validate_export_manifest(...)` and `validate_compilation(...)` check the pinned export, compiler, contracts and model chain. The board owner rechecks the large original source/input provenance before and after the experiment, outside forward timing.

`GroupedSession.open_board(model_path, model_sha256, inputs, outputs, *, abi_path, header_path, library_path, log_helper_path, export_strict_FP32_gate_passed, diagnostic_allow_fp32_rejected=False, pins=None, collect_perf_detail=False, log_directory=None)` creates one persistent fixed-shape context. The ABI, private SDK header/library and native log helper are SHA-bound. Header/library files are not redistributed here.

`session.execute(feed, reference=None)` returns `(owned_values, record)`. Feed keys are `net`, `inp`, `corr`; SDK output keys are `net_out`, `delta`, `weight`. All logical arrays are C-order FP32 channel-first. Queried NHWC input attributes trigger actual CF→NHWC packing and request `fmt=1`, with `pass_through=0`. Unknown layouts or padded native inputs are rejected. Native output stride padding is recorded; `outputs_get(want_float=1)` must still return exactly `4*N` logical bytes, copied to owned CF arrays before release.

The record separates execution status, finite/domain validity and the unchanged strict gate. Invalid raw outputs remain available for diagnostic preservation; an `E RKNN` log is an execution failure even when C API return codes are zero. `close()` is idempotent; metrics remain readable and failed destruction cannot claim memory freed.

`BoundedContextCache(specs, factory, max_contexts=4, max_model_bytes=128<<20, max_dma_bytes=512<<20)` serializes one context per exact shape with LRU eviction. Actual DMA is known after initialization, so transient allocation is recorded separately from the resident bound. `execute_full_graph(...)` stages all true source batches and restores original edge order only after every group succeeds and the snapshot identity remains current. It never commits VO state. The hardware LRU path across all 14 shapes has not been tested.

## Actual first complete graph

Root ran the captured initial graph `B8_D8_M48`, all 3,072 original edges: four ordinary calls and one separate profiling call. The ten saved CF/full-order NPZ files independently verify exact inverse edge restoration and bitwise equality across all five calls. Ordinary warm SDK forward, including owned packing and I/O, averaged **174.289 ms**; original-order scatter took another 3.44–3.47 ms. This excludes loading, graph planning, context initialization and reference checks.

Strict FP32 comparison to the original full Update failed for all three outputs: relative L2 `0.00152831 / 0.00184839 / 0.00153985` for net/delta/weight. Profiling reported 50 internal NPU nodes, three internal CPU nodes (two Exp and one Min), and six CPU I/O nodes. The two contexts closed cleanly; all five output releases were recorded. This does not validate recurrent feedback, the other 13 shapes, trajectory quality or the complete 20 Hz / 100 ms path.

## Saved-array audit and synthetic checks

The independent auditor uses only NumPy, JSON and file hashes. It uses a separate lexicographic inverse permutation rather than the driver scatter implementation.

```bash
python -B audit-v37-grouped-initial.py   --normal-dir "$NORMAL" --profile-dir "$PROFILE" --export-dir "$EXPORT"   --capture-dir "$ORIGINAL_CAPTURE" --compile-dir "$COMPILE"   --sdk-dir "$HELPERS" --driver "$HELPERS/run-v37-grouped-initial-board.py"   --output "$NEW_REVIEW.json"
```

The 33 generated-array / fake-SDK tests do not load a native library or execute a real model. Tests prefer repository-relative helpers, and accept `V37_GROUPED_HELPERS`, `V37_ABI_HELPER` and `V37_GROUPED_ADAPTER` overrides. They cover transport layout, copy-before-release, invalid data/logs, source lineage, cache cleanup, complete edge restoration and stale/failed graph rejection. Source topology and adapted DPVO behavior are attributed to Princeton-VL/DPVO commit `0ac95b656d1fda91c271d2a106460d19ad966fc7` under the adjacent MIT license.
