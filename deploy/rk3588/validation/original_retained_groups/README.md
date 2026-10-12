# Original VGGT aggregator groups at retained consumers

This diagnostic composes the original 24 frame blocks and 24 global blocks
into four graphs: pairs **0–4, 5–11, 12–17 and 18–23**. Every original node,
initializer, learned parameter tensor and token computation remains. Only
the original global-to-next-frame views are added between parent graphs.
The first three groups return their final frame/global values. The last
returns those values plus the unnormalized **global20** target value.

The runtime keeps original image normalization, DINO, special tokens and
positions. It performs the four original CPU `torch.cat` operations for
retained layers **4, 11, 17 and 23**, and the original target-token view,
clone and normalization. Unconsumed concatenation allocations may be
omitted; original block computations may not. The fixed images are
`[1,2,3,84,112]`; graph inputs are `[2,53,1024]`, with frame outputs of the
same shape and global outputs `[1,106,1024]`.

**Constructor detail:** the standalone Aggregator default `target_layer=20`
is overridden by the original `VGGT.__init__(target_layer=21)`. The original
loop increments its global index before capturing, so the actual loaded
model consumes zero-based **global20**, not global19. The third output must
come from global20 while that same value continues to frame21. Keep the
original CPU `.view(2,53,1024).clone()` and division by
`norm(dim=-1, keepdim=True) + 1e-8`. Auditing only the Aggregator default gives
the wrong consumer. Earlier first-group 0–4 diagnostics do not exercise this
target boundary.

## Prerequisites and fixed identities

Run actual original-model, ORT, RKNN compilation and inference only on
RK3588S, sequentially. Use the production environment for the model and
replay, and the existing isolated RKNN Toolkit2 **2.3.2** environment for
ONNX checks and compilation. No command below changes system clocks.
Every generated output directory must be new. All paths below refer to the
board installation, from its repository root.

| Input or source | SHA256 |
| --- | --- |
| Original v55 24-pair `compiled.json` | `1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42` |
| Original v53 `capture.json` | `4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67` |
| Original v56 `capture.json` | `a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c` |
| `compose_all_retained_groups.py` | `e6fa508e0b03a011a1ccb484f2a46847ffd49fc1bf6ae11ecfeb8bd45f835f60` |
| `check_compile_retained_groups.py` | `21d958ef385eaa419e81bef04303f07ffb5b7bb32a4854cd5d17ffba6247680b` |
| `assemble_compiled_manifest.py` | `0122bddc8ce988423dea6fd01a67fbaa2ba374e1a2dd54e6b221d9a3408f43dd` |

The sole checkpoint is the original `model.pt`, **5,026,874,952 bytes**, SHA256
`d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`.
Its loader checks the identity before deserialization. Captures and
weight-bearing ONNX/RKNN files must already exist privately; none are bundled.

```bash
set -e
BASE=/mnt/vggt-data/vggt-slam2
cd "$BASE/src"
. deploy/rk3588/env.sh
EXP="$BASE/experiments"
V=deploy/rk3588/validation/original_retained_groups
PARENT="$EXP/vggt-v55-pair-graphs-r1/compiled.json"
PARENT_SHA=1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42
GROUP="$EXP/vggt-v67-retained-groups-reproduce-new"
TOOL_PY="$EXP/vggt-v49-toolchain-venv-r1/bin/python"
RUN_PY="$BASE/venv/bin/python"
tool_python() {
  env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
    LD_LIBRARY_PATH="$BASE/sysroot/usr/lib/aarch64-linux-gnu" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$TOOL_PY" "$@"
}
```

The historical `vggt-v67-retained-groups-r1` directory already exists; use
the new directory above rather than overwriting its evidence.

## Compose, check each group, then compile

```bash
tool_python "$V/compose_all_retained_groups.py" \
  --manifest "$PARENT" --manifest-sha256 "$PARENT_SHA" \
  --output-dir "$GROUP"
EXPORT_SHA=$(sha256sum "$GROUP/export.json" | cut -d ' ' -f 1)
```

The recorded static export SHA is
`41014fa4724d668da40842b8af1a95f2f7e89e4f8c37c72b2022f9aafe229f5c`.
Pass the hash of the exact export used to every later stage. Composition
checks that inverse namespacing recovers every parent node and initializer
protobuf byte-for-byte, and copies original position files without changing
their bytes. This is an FP32 graph preservation check, not proof of compiled
FP16 boundary equivalence.

For each group index **0, 1, 2, 3**, run check before its compile. Stop on a
failed command or failed cleanup; allow the board to cool between jobs.
The following shows one group; repeat with the other indices and new
`check1..3` / `compile1..3` directories.

```bash
i=0
tool_python "$V/check_compile_retained_groups.py" \
  --mode check --group-index "$i" \
  --manifest "$GROUP/export.json" --sha256 "$EXPORT_SHA" \
  --parent-manifest "$PARENT" \
  --capture-dir "$EXP/vggt-v53-capture-r1" \
  --calibration-dir "$EXP/vggt-v56-calibration-r1" \
  --output-dir "$GROUP/check$i"
CHECK_SHA=$(sha256sum "$GROUP/check$i/check.json" | cut -d ' ' -f 1)
tool_python "$V/check_compile_retained_groups.py" \
  --mode compile --group-index "$i" \
  --manifest "$GROUP/export.json" --sha256 "$EXPORT_SHA" \
  --parent-manifest "$PARENT" \
  --check-report "$GROUP/check$i/check.json" --check-report-sha256 "$CHECK_SHA" \
  --output-dir "$GROUP/compile$i"
```

Checks use board CPU ORT with `ORT_DISABLE_ALL` and one intra/inter thread.
Every group output, including the last group's target, must match the actual
sequential parent ONNX chain byte-for-byte before compilation. Later parent
graphs consume the preceding computed output, never a reset captured input.
The v53 input has original Torch FP32 exit references; the second v56 input
has no saved exit references. Missing references and any strict original
`rtol=1e-4, atol=1e-5` failures remain explicit in receipts. The saved v56
input is an additional check case, not a new quantization calibration run.

Compilation uses `rk3588`, `float_dtype=float16`, optimization level 3,
identity preprocessing, pruning/reshape-removal/weight-quantization/flash
attention disabled; build options are `do_quantization=False, dataset=None,
auto_hybrid=False`. Inspect all four SDK return codes and release/cleanup,
environment restoration and unchanged input hashes. If a process is
interrupted, retain its failed directory and use a new directory for retry.

## Assemble the runtime manifest

All successful check and compile directories must be children of `GROUP`.
The assembler requires their explicit hashes and preserves the complete
export metadata. It adds relative `check_receipt`, `compile_receipt` and
`rknn` identities to each group. A compiled manifest remains
`compiled_not_quality_accepted`.

```bash
CHECK_SHAS=()
COMPILE_SHAS=()
COMPILE_SUBDIRS=(compile0 compile1 compile2 compile3)
for i in 0 1 2 3; do
  CHECK_SHAS+=("$(sha256sum "$GROUP/check$i/check.json" | cut -d ' ' -f 1)")
  COMPILE_SHAS+=("$(sha256sum "$GROUP/${COMPILE_SUBDIRS[$i]}/result.json" | cut -d ' ' -f 1)")
done
tool_python "$V/assemble_compiled_manifest.py" \
  --manifest "$GROUP/export.json" --manifest-sha256 "$EXPORT_SHA" \
  --check-sha256 "${CHECK_SHAS[@]}" --compile-sha256 "${COMPILE_SHAS[@]}" \
  --compile-subdirs "${COMPILE_SUBDIRS[@]}" --output "$GROUP/compiled.json"
GROUP_SHA=$(sha256sum "$GROUP/compiled.json" | cut -d ' ' -f 1)
```

The historical run selects `compile0 compile1 compile2-r2 compile3`,
preserving the interrupted `compile2`. Change `COMPILE_SUBDIRS` to those
names when assembling that run; only successful pinned receipts are
eligible. Directory selections must be single safe relative names. No
graph, SDK configuration or precision change is implied by a retry directory.

## Complete original-model comparison and paced replay

Use the existing verified NEON library and original DINO/depth manifests.
The 1 GiB runtime cap applies **per artifact**; it is not a total RAM or
8 GB operation claim. All four contexts must initialize and validate before
the Aggregator forward is replaced. Original parameter/source/position
checks, runtime mutation guards and owned SDK outputs remain. Bypassed inner
hooks, similarity, attention visualization and unsupported shapes fail;
there is no silent CPU fallback. Install all backends before the tail cache
and close the cache first.

```bash
NEON="$EXP/vggt-v56-neon-build-r1/fp16_neon.so"
NEON_SHA=d08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9
DINO="$EXP/vggt-v58-dino-groups-r1/compiled.json"
DINO_SHA=a602bcdaab3f2e1ab51d49feb300ff3af76c07fe334489c983d647859f3f9493
DEPTH="$EXP/vggt-v54-depth-graphs-r1/compiled.json"
DEPTH_SHA=9a6e5435df543b8484a4fc50da4b065995f32153233548829234dd149fbc7135
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$RUN_PY" deploy/rk3588/validation/original_full_aggregator/compare_complete_model.py \
  --manifest "$GROUP/compiled.json" --manifest-sha256 "$GROUP_SHA" \
  --paired-aggregator --retained-aggregator --save-aggregator-outputs \
  --half-io-library "$NEON" --half-io-library-sha256 "$NEON_SHA" \
  --dino-manifest "$DINO" --dino-manifest-sha256 "$DINO_SHA" --grouped-dino \
  --depth-manifest "$DEPTH" --depth-manifest-sha256 "$DEPTH_SHA" \
  --reference-run "$EXP/vggt-v53-complete-ab-r1" \
  --model-threads 1 --output "$EXP/original-retained-comparison-new"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$RUN_PY" scripts/replay_original_vggt.py \
  --rgb-index "$BASE/datasets/rgbd_dataset_freiburg1_xyz/rgb.txt" \
  --checkpoint "$BASE/weights/model.pt" --image-size 112 --model-threads 1 \
  --rknn-manifest "$GROUP/compiled.json" --rknn-manifest-sha256 "$GROUP_SHA" \
  --rknn-paired-aggregator --rknn-retained-aggregator --rknn-camera-head \
  --half-io-library "$NEON" --half-io-library-sha256 "$NEON_SHA" \
  --rknn-dino-manifest "$DINO" --rknn-dino-manifest-sha256 "$DINO_SHA" --rknn-grouped-dino \
  --rknn-depth-manifest "$DEPTH" --rknn-depth-manifest-sha256 "$DEPTH_SHA" \
  --output "$EXP/original-retained-replay-new"
```

The comparator uses 17 fixed real timestamps and 16 overlapping pairs with
the original model and SL4. It saves five canonical prediction arrays,
poses, all four retained tensors and normalized target tokens privately;
aggregator saves occur after timed service. To reuse a completed matched
CPU baseline, the example supplies `--reference-run DIRECTORY`; input
indices, timestamps, shapes and saved array identities must match. Compare the same complete
outputs and trajectory with the preceding paired FP16 backend as well as
original CPU FP32. Sequential branch timings are not an interleaved thermal
comparison and are not paced input latency.

Replay measures fresh-observation publication, the single-slot queue, full
model plus SL4, and map/pose file publication including EOF drain. Do not
restrict the outer process to A76 CPUs: the decoder needs CPU0. The model
uses its existing process-local affinity. Inspect actual fresh timestamps,
end-to-end P95, map cadence, dropped-source counts and all context/output
release counters. Compilation or finite arrays do not establish strict
FP32 equivalence, trajectory quality, real-time rate, 30-minute stability
or 8 GB operation. No candidate full-model result is claimed by this recipe.

Checkpoints, arrays, ONNX/RKNN and SDK/NEON binaries remain private. The
repository code license does not grant rights to the original weights; see
[model provenance](../../../../MODEL_LICENSES.md).
