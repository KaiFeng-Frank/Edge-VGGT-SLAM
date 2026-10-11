# V27 CPU correlation research helper

This implements only the pinned DPVO forward correlation arithmetic. It contains
no neural weights, BA, estimated pose, trajectory or full VO runner. Sources derive
from DPVO commit `0ac95b656d1fda91c271d2a106460d19ad966fc7` (MIT; see LICENSE).

Compile **on the target board**, in a new directory, with its isolated C++ toolchain:

```sh
g++ -std=c++17 -O3 -fPIC -shared -fopenmp -fno-fast-math -ffp-contract=off \
  dpvo_corr.cpp -o libdpvo_corr.so
```

The host test library is synthetic-only and must not be deployed as an ARM binary.
No fast-math or contracted multiply-add is permitted. AArch64 uses conditional
NEON multiply/add and a four-lane reduction; other hosts use scalar channel sums.
FP32 sum order can differ from Torch/CUDA. The fixed numerical comparison is
`rtol=1e-4, atol=1e-5`; neither CUDA byte identity nor ARM speed is established by
the host synthetic tests. The shared library needs libstdc++ and OpenMP libgomp.
Each region uses the requested `num_threads`; the runtime's thread limit/dynamic
settings can restrict actual teams. No process affinity or global OpenMP setting
is changed. Caller sets its observed configuration, including `OMP_DYNAMIC=FALSE`.

```python
from dpvo_corr import PreparedCorrelation, correlate_once

# All arrays are CPU NumPy; gmap/fmap/coords are strictly FP32, kk/jj int64.
# gmap: [B,Npatch,C,PH,PW]; fmap: [B,Nview,C,H,W], C divisible by8, H/W>=4.
with PreparedCorrelation(library, gmap, fmap, threads=4,
                         cache_key="explicit-feature-generation-SHA",
                         library_sha256=actual_library_sha) as corr:
    preparation = corr.preparation
    # coords [B,E,2,PH,PW] xy, kk/jj [E], same edges for all batches.
    output, timing = corr.correlate(coords, kk, jj,
                                   cache_key="explicit-feature-generation-SHA")
    metrics = corr.metrics()
# output is owned and valid after another query or close. close is idempotent.

output, full_timing = correlate_once(library, gmap, fmap, coords, kk, jj,
                                    threads=4, cache_key="generation",
                                    library_sha256=actual_library_sha)
```

Preparation stores independent channel-contiguous gmap and target snapshots,
plus an independent level4 4x4 average-pool pyramid with floor spatial dimensions.
Arrays borrowed during preparation/query must remain unmodified until that call
returns. Later mutations to original features do not update the context. Changed
feature values require a new context and generation key; pointer/shape equality
is never used as a cache validity test. Caller manages the key from source values.
Python rejects mismatched keys, dtypes, shapes and unsupported radius. C validates
element lengths, bounds, nonfinite inputs, int32-safe FP32 coordinates, aliases,
arithmetic overflow and a 512MiB context/output safety budget. Failures raise;
there is no previous-output fallback. Calls and close on one context are locked.

For each level1/4, FP32 coordinates are divided first, then 8x8 integer channel
dot products are formed with zero padding. Four FP32 bilinear terms produce 7x7
correlations. The owned result `[B,E,98*PH*PW]` flattens
`offset_x, offset_y, patch_y, patch_x, level`; level is innermost. There is no
channel normalization. Output is not a pose or a learned confidence.

`preparation.total_seconds` includes library loading/hash, validation, allocation,
packing/layout and pooling; `native` subdivides scan, layout, pool and total.
Query `total_seconds` includes locking, NumPy packing, validation, output allocation,
FFI and both-level computation until the output is ready, before metrics bookkeeping.
It excludes preparation, hashes/array statistics and Update. `native_call_seconds`
and `native.kernel_seconds` are separate diagnostics. `correlate_once.total_seconds`
includes preparation, query and destruction. Cached preparation must be reported
separately; warm cached time is not the same scope as v26's per-call pooling.

Synthetic tests compile a temporary host library and use only tiny generated data:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B -m pytest -q -p no:cacheprovider \
  tests/test_dpvo_corr.py tests/test_corr_numpy_oracle.py
```

All real captured-feature, Update and performance comparisons must run separately
on the board. Full call timing and returned arrays are needed to assess usefulness;
the kernel timing alone does not establish an end-to-end improvement or 20Hz VO.

Run those tests from the repository root with NumPy, pytest, a C++17 compiler and
OpenMP installed. They build a temporary host library from these source files.

The historical board experiment is driven by `../run-v27-dpvo-corr-board.py`.
Its `--help` lists external v26 source/capture/motion/checkpoint inputs and the
native library/build-receipt inputs. These external assets must match the pinned
v26 receipts. `../build-v27-dpvo-corr-board.py` retains the exact experiment build
entry and expects the isolated toolchain under the recorded board experiment
path; it is not a general toolchain installer. The published result records
compiler/package hashes, flags, initial linker lookup failure and repaired smoke
receipt. None of those external packages or compiled binaries are bundled.
