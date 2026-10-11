# Immutable per-frame correlation cache (v36 research)

This independent candidate changes feature preparation and cache ownership.
It retains the original DPVO weights, edge order, radius 3, levels 1 and 4,
FP32 coordinates, zero boundaries, channel dot products before bilinear
combination, and level as the innermost output dimension. No model, BA,
ORB/LK/PnP, device or production default is changed. Actual ARM build, four-case saved-feature AB and complete M48 FAST replay have now completed. The AB audit passed 1,337 checks; the complete replay audit passed 1,607 checks. [Actual receipts and boundaries](../../../../results/dpvo_frame_cache_v36.json).

The frozen v35 callback rebuilt a concatenated source/target context whenever
the referenced frame-ID sets changed. That repeated scans, layout conversion
and 4x4 pooling for historical frames. This candidate prepares each immutable
frame once and supplies a small current-order handle table to each query.
Source `kk // M`, patch `kk % M`, target `jj` and every edge remain unchanged.
The C++ translation unit includes the byte-frozen v27 parent implementation
to reuse its preparation, pooling and scalar/NEON dot operations exactly.

## Build and source lineage

Keep the frozen `dpvo_corr.cpp` and `dpvo_corr.h` together in `PARENT_DIR`.
Verify them with `dpvo_frame_corr.verify_parent_sources(PARENT_DIR)` before
compilation and bind these hashes, the candidate files, compiler command and
resulting library SHA into the actual board build receipt:

| Parent | SHA256 |
|---|---|
| dpvo_corr.cpp | 3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990 |
| dpvo_corr.h | 1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932 |

```sh
c++ -std=c++17 -O3 -fPIC -shared -fopenmp \
  -fno-fast-math -ffp-contract=off \
  -I "$PARENT_DIR" dpvo_frame_corr.cpp -o libdpvo_frame_corr.so
```

Use the root's isolated board compiler and a new output path. No fast-math or
FP contraction flags are allowed. The library retains v27 exports for the
controlled old/new comparator, and adds the v36 per-frame C ABI in the new
header. The Python loader requires an explicit actual library SHA, ABI and
statistics-size checks. The embedded parent declaration is not a substitute
for the external before/after build-source hash verification.

## FAST callback and ownership

```python
correlator = PerFrameCorrelation(
    library, library_sha256=actual_library_sha,
    patch_count=48, threads=4, max_frames=64, max_bytes=512 << 20)
corr = correlator(coords, ii=ii, jj=jj, kk=kk, frames=frames)
```

`coords` is FP32 `[1,E,2,3,3]`, nonempty `ii/jj/kk` are int64 `[E]`, and
`frames` is the current retained record sequence, including any tentative
motion-probe frame. Each unique, stable `frame_id` is bound to the same
read-only C-order FP32 `gmap[M,128,3,3]` and `fmap[128,H,W]` ndarray objects.
Their values and all aliases must remain immutable for the retained lifetime.
The native snapshot is independent; the cache strongly retains the original
objects to avoid Python object-ID reuse. Replacing a resident frame's arrays
under the same ID raises. There is no per-query content hashing.

Only newly referenced frames are prepared. Removed frames are destroyed on
the next query. Capacity pressure can evict an unreferenced cached frame;
the current query's referenced set must fit the frame and byte budgets or the
query is rejected before new allocations. All payload estimates include the
owned packed patches and both target levels. Shared caller input references,
Python metadata and each owned query output are separate from that budget.
Query, eviction, `reset()` and `close()` hold one RLock, so close waits for an
in-flight synchronous query. Returned arrays survive later calls and close.
Call `reset()` for a new sequence/feature geometry; `close()` is idempotent.
Closed metrics retain cumulative counters and report zero cached payload.
Histories are bounded and report dropped history counts.

The native kernel checks every produced value for nonfinite/overflow. Invalid
coordinates, indices, geometry, generation or capacities fail explicitly;
there is no old-result or alternate-backend fallback. The frozen stream's
independent result validation remains unchanged.

## Measurement and complete replay

Metrics provide each call's `preparation_seconds`, `native_query` (including
handle/buffer packing and allocation), kernel/scan time, validation/lock time,
eviction time, cache hit/miss counts and payload bytes. The callback's internal
`total_seconds` ends before receipt bookkeeping; the root should measure the
entire callback and fresh-frame path externally. Construction/library hashing
and final close are separate; compare cold startup and all misses, not only
warm kernel time. Metadata/history copies are outside the internal timer.

For the board test, first interleave the frozen concatenated v27 callback and
this candidate on the same saved features and original changing graphs. Check
all correlation elements at the unchanged `rtol=1e-4, atol=1e-5` gate, repeat
determinism, removal/current-slot reindex, payload limits and close. Then root
may wire this callback into a new isolated v35-derived complete FAST diagnostic
driver while leaving the M48 RNG, rejected NPU encoder authorization, original
Torch Update, native BA, EOF handling and saved-trajectory evaluation intact.
Keep every source/model/input pin and the old driver evidence. Do not change
the frozen v35 driver or relabel its old library pin as the new library.

The historical fresh correlation outer mean was about 290 ms (prepare about
100 ms, query about 124 ms, with nesting/other costs). These observations
motivate reuse; they do not predict a measured new speed or 20 Hz acceptance.
The query kernel, stream result copies/validation, Update and BA still remain.
The host tests use only generated tiny arrays and generic scalar compilation,
with frozen-v27 byte comparison plus an independent NumPy fixed-gate oracle.
ARM NEON and complete replay must be measured separately.

## Measured v36 result

The same 192 input images produced 179 fresh poses in 269.215 s (0.665 Hz), with fresh software processing P95 2.170 s and separate EOF work of 22.809 s. Per-observation poses and lifecycle/BA diagnostics, and EOF reconstruction match v35 exactly after excluding timing fields. Peak process RSS was 949.781 MiB on the 16 GB board. The cache prepared 192 frames, recorded 2,365 hits, destroyed all 192 handles and retained zero bytes at close; peak native payload was 166,820,864 bytes.

Online accumulated cache preparation changed from historical 18.076 s to 1.791 s, while original Update changed from 147.293 s to 155.975 s. Separate thermal histories prevent a controlled speedup claim. The original encoder FP32 rejection remains, and this is not real-time, physical-camera, dense-map, 30-minute, 8 GB or original-VGGT accuracy acceptance. See [engineering record 57](../../../../docs/ENGINEERING_NOTES.md) for timing nesting, ownership and comparison scope.
