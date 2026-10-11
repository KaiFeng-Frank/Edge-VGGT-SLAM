# Native FP16 conversion candidate

The actual RK3588S conversion check passed all nine queried shapes and the saved observation34 B2D6 feed. Its three saved native arrays independently match NumPy IEEE RN bits, including 1,152 positive-zero corr padding elements. FPCR was 0 before and after; 18 finite boundary cases, six numeric rejections and capacity/alias guards passed. This is the executed AArch64 NEON conversion, separate from the 21 generated/fake host protocol tests.

The actual single-group SDK test used the same saved B2D6 feed in two normal contexts, ordered ABBAABBA. All eight saved outputs (24 FP32 arrays) are bitwise identical to the first ordinary FP32/pass0 SDK result. Excluding the first call of each method, the three warm execution wall means are 12.654 ms for A and 10.186 ms for native B (19.505% lower). Input-set means are 2.797 and 0.072 ms; complete pack means are 0.900 and 0.568 ms. Run API means are 8.154 and 8.768 ms. Two contexts destroyed and eight outputs released, with no reported cleanup errors. The independent audit reread every output artifact and recomputed warm means.

The SDK comparison validates this one input and B2D6 model only. No original Torch teacher was used, the historical strict FP32 rejection remains, no recurrent VO state was committed, and no whole Update, online replay, quality or 20Hz claim follows from it. `pass_through=1` uses ordinary SDK input submission; zero copy is not claimed. Source pins, actual results and scope are recorded in [the v43 results](../../../../results/dpvo_native_input_v43.json).

This research module converts stable, aligned, C-contiguous FP32 CF `[B,C,D,16]` to owned `uint16` IEEE binary16 bits `[B,ceil(C/8),D,16,8]`. It supports B1/B2, D3..7, C384/C882, with six positive-zero padding channels for corr. The actual nine-model command8/10 query is pinned to `89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642`; command1 describes a different layout. Generated B2D3 conversion support does not add an unqueried model to the catalog.

`pack_blocked_fp16(value, attribute, native_library)` validates the actual native attribute, allocates an independent contiguous buffer, and calls the single C conversion. It does not create a whole-array NumPy FP16 temporary or silently copy a noncontiguous input. Retain the returned buffer through SDK input submission, run, output copying and output release. A failed call raises and returns no partial buffer; the C destination is unspecified on error and must never be submitted. Do not mutate the input during the synchronous call.

The C kernel uses an 8-channel by 4-spatial tile, transpose and `vcvt_f16_f32`. Bit checks reject every NaN/Inf before conversion; the converted half exponent rejects overflow without clipping. Finite FP32 values immediately above 65504 are accepted if they round to finite 65504. Padding contains positive-zero bits. The explicit byte capacity, source element count and disjoint pointer ranges are checked before writing.

FPCR is read, never written. The supported state has round-to-nearest, FZ=FZ16=0, AHP=DN=0, no alternate handling and no enabled FP traps; unsupported states return ENOTSUP. The conversion may set normal cumulative FPSR exception flags. This does not demonstrate equivalence to the SDK's original typed FP32 conversion: same-state SDK A/B is a separate required measurement, and this module does not call the SDK.

Root builds `libv43_native_fp16_pack.so` on the board with the existing isolated GCC toolchain. Example flags are `-O3 -std=c11 -shared -fPIC -march=armv8.2-a+fp16 -fno-fast-math -ffp-contract=off`. No host compilation, model execution or system settings are required. The header provides C linkage if included by C++ code; the source is C11. Pin the actual compiler, source/header and resulting library hashes in the build receipt.

```python
library = load_native_library(library_path, expected_library_sha256)
native_bits = pack_blocked_fp16(CF_fp32, queried_native_attribute, library)
```

`check-v43-native-pack-board.py` executes only conversion. It verifies queried command8/10 agreement, exact RN boundary bits (tie-to-even, gradual subnormals, signed zero and 65504), overflow/NaN/Inf rejection, capacity/alias guards, input preservation and independent output ownership. All timed samples include validation, allocation and the complete C pack; input generation, NumPy reference and comparison are outside the measured interval. Each repetition packs the FP32 input anew. It loads no RKNN SDK and performs zero model calls.

```sh
python3 check-v43-native-pack-board.py \
  --library libv43_native_fp16_pack.so --library-sha256 LIBRARY_SHA256 \
  --query-json query.json --model-key B2_D6_M16 --repeat 3 \
  --feed grouped-CF-inputs.npz --feed-sha256 FEED_SHA256 \
  --output-dir new-conversion-result
```

The saved feed must contain exactly `net`, `inp`, `corr` as FP32 CF arrays matching the selected queried model. Omitting `--feed` runs only generated checks. `--all-queried-shapes` checks all nine actual attribute sets. The output `conversion.json` states conversion-only scope and preserves any failure; a successful saved-feed check also writes `saved-feed-native-blocked-fp16.npz` with file and array hashes. These converted inputs are neither network outputs nor original Torch teachers.

Host tests use generated arrays and a fake ABI surrogate. Passing them establishes wrapper, attribute, error, ownership and driver protocol behavior. The separate actual board conversion receipt covers the NEON kernel in the queried shapes and supported FPCR state. Conversion-only success or matching NumPy bits alone does not establish SDK transport acceptance, network equivalence, SLAM quality, 20Hz performance or candidate adoption.

`extract-v43-saved-group.py` reuses the SHA-pinned original source planner and `pack_batch` to extract the first complete B2D6 group from the pinned actual observation34 capture. It preserves source slots `[2,5]`, all 192 selected edges and six original array hashes, without importing a model. `audit-v43-saved-conversion.py` verifies the actual converted input NPZ; `audit-v43-saved-single-ab.py` rereads the eight actual SDK output NPZ files. These audits use saved data only and load neither the packing library nor SDK. Supply the frozen receipts and arrays through CLI paths; large arrays and compiled libraries are excluded from this public source directory.

Original DPVO attribution: Princeton Vision & Learning Lab, commit `0ac95b656d1fda91c271d2a106460d19ad966fc7`, MIT license in the adjacent `LICENSE`. The packing code does not replace the original DPVO Update mathematics or weights.

## Complete saved-graph SDK A/B

The subsequent board test ran three complete saved NPU-feedback states, observations 34/127/191 with 704/800/752 edges. A and B used the same frozen v43 metadata-only callback, source-group planner, nine-model catalogue and compiled weights; B changed only the input transport. A submitted owned FP32 NHWC buffers with pass0. B queried each context's command8/10 native attributes, packed owned FP16 NC1HWC2 C2=8 buffers and submitted pass1. Both retain the original synchronous run, FP32 CF output copy before release, and complete-source scatter to the original edge order.

Each state used ABBA twice. All 24 saved full-graph outputs were bitwise identical to the first A result for their state. Each method recorded 56 SDK run calls, 56 output releases and six destroyed contexts; current DMA and model-host bytes were zero after close, with no reported cleanup errors. The independent saved-array review is linked through [the v43 results](../../../../results/dpvo_native_input_v43.json).

The table reports actual callback wall means in milliseconds. The first call of each method for each state is excluded, leaving three warm samples per table cell; that exclusion does not imply each excluded call initialized a context.

| Saved observation | A: FP32/pass0 | B: native FP16/pass1 |
| --- | ---: | ---: |
| 34 | 94.828 | 71.467 |
| 127 | 105.123 | 94.093 |
| 191 | 116.358 | 91.406 |

The nested sum of input-set intervals fell from 20.578/23.711/26.226 ms to 0.427/0.558/0.598 ms for these states. Callback wall, complete-graph wall and summed SDK intervals describe nested scopes and must not be added. These are three fixed saved states with warm n=3 and observed, uncontrolled temperature/frequency conditions. They do not establish online native-transport integration, trajectory quality, statistical significance or 20Hz operation. No original CPU teacher, recurrent feedback or BA ran in this harness. Historical strict FP32 and quality rejections remain; `candidate_accepted` stays false and the default VGGT model is unchanged.

`v43_native_fp16_grouped.py` privately binds the pinned v37/v40 runtime's input pack function, retaining the frozen output, lifecycle and cache implementation. It never changes the shared parent source or silently falls back to CPU. Native-layout queries occur once per initialized context; one pinned packing library is loaded lazily and reused by the private runtime. Each measured pack includes native-attribute validation, allocation and the C conversion. `run-v43-native-single-ab-board.py` is the prerequisite single B2D6 experiment. `run-v43-native-full-ab-board.py` requires that actual successful single-group receipt, the successful conversion receipt, the native query, library/build identity and two explicit diagnostic flags before SDK execution.

To reproduce the full saved-state experiment, first stage the SHA-pinned nine compiled models, their export/compile ancestors and the three original input captures on the RK3588S. Supply the original v43 board configuration externally: its paths name those staged files, and the expected configuration SHA is `7cb492b839a8529698687e6224868cf3a3d4de5e0546f945747963282290b622`. It is distinct from the older v41/v42 public configuration examples. The SDK header, runtime library, model binaries and raw input/output arrays are external artifacts, excluded from this source package.

Use `--help` for the required path/SHA pairs and diagnostic flags:

```sh
python3 -B deploy/rk3588/validation/dpvo_native_input/run-v43-native-full-ab-board.py --help
```

The required ancestors are the sibling `dpvo_update_performance/run-v42-callback-ab-board.py` and `dpvo_online_replay/v43_online_factory.py`, the external v43 board configuration, this transport and packing module, the native library/build receipt, the command8/10 query, the successful conversion receipt and the successful single-group AB receipt. Their frozen identities are recorded in [the v43 results](../../../../results/dpvo_native_input_v43.json). `--inputs-dir` names the original `vggt-v41-online-bothnpu-r2` directory containing its pinned `run.json` and `update-034/127/191-input.npz`; the source run SHA is `78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b`. Both `--diagnostic-allow-fp32-rejected` and `--diagnostic-allow-native-fp16` are mandatory. The output directory must be new and under the board experiments directory.

The added transport and full-harness tests contain 20 and seven generated/fake cases respectively, with no real SDK or model execution. Run them without the external raw artifacts:

```sh
python3 -B -m pytest -q -p no:cacheprovider \
  deploy/rk3588/validation/dpvo_native_input/test_v43_native_transport.py \
  deploy/rk3588/validation/dpvo_native_input/test_v43_native_full_ab.py
```
