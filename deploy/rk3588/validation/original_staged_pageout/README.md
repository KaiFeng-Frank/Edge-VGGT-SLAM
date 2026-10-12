# Original VGGT staged checkpoint residency

`--checkpoint-pageout` is an explicit, **default-off** experiment. After each
successful original backend installation, it advises the process's verified
clean `model.pt` mapping with `MADV_PAGEOUT` (21): Aggregator, camera, DINO,
then depth. This reduces overlap between resident checkpoint pages and the
next backend's startup allocations. All original parameters, Tensor/storage
identities, virtual addresses, versions, layers, tokens and forward operations
remain; later original parameter checks and CPU operations can fault pages
back in normally.

The fixed 16-pair comparison preserved all **160 saved arrays byte-for-byte**
(80 prediction arrays and 80 retained/target arrays) and the poses file against
the existing FP16 branch. Process high-water RSS was **7169.87109375 →
3920.0625 MiB**. The separate original-timestamp replay published **18 fresh
poses / 798 source frames**, **0.6214327 Hz**, input-to-publication **P95
2.030057 s**, with peak RSS **3920.39453125 MiB**. These are process-memory
and short replay observations, not 8 GB-board acceptance. The original CPU
FP32 strict comparisons remain failed; this experiment does not establish new
trajectory quality, camera latency, 1 Hz, or long-run stability.

There is **no speedup conclusion**. The complete comparisons were sequential,
not interleaved; their service means were 1.593269 s and 1.589965 s. Replay
startup includes four advice calls and its startup memory snapshot. Its final
pre-cleanup snapshot is separately timed diagnostic work after publication;
there are no per-frame smaps reads in the replay. Comparator memory samples
occur after timed service. Active sampled RSS, post-context-cleanup RSS and
the process high-water mark are different scopes. RSS/PSS and SDK DMA counters
must not be summed as independent physical memory.

## Required original artifacts and source versions

Run all actual model, ORT, RKNN and advice operations **on RK3588S**, sequentially.
Use the existing verified original checkpoint and compiled artifacts; this
experiment does not rebuild or quantize them. Preparation is documented in
the [retained groups](../original_retained_groups/README.md),
[grouped DINO](../original_grouped_dino/README.md) and
[camera partition](../original_camera_partition/README.md) recipes.

The comparator requires retained plus paired Aggregator, grouped original
DINO, native FP16 I/O, camera split4, depth logits, and a completed matched
`--reference-run`. Replay requires the same backends, `--rknn-camera-head`,
and the default patch cache; `--no-patch-cache` is incompatible. Missing
dependencies fail before model loading. The CPU reference branch does not
perform pageout.

The measured baseline used the comparator from commit
`0f01c2d64fa2a92750f4853074effd2c0d91a789`, SHA256
`1e1d28e11f2f476b875593d272526bf816ac3049fa3b69309acaf10a08229927`.
The candidate used the current staged source:

| Source | SHA256 |
| --- | --- |
| `vggt_slam/checkpoint_residency.py` | `1948fa3f49814d8fc1025516025df79adc7b1bd0e0a54a86f842b1590f445777` |
| `compare_complete_model.py` | `75b81c6b8dd2c9fcb96f76bdb5fecb5d15e5c7794a161deee8aa4f323d020e7c` |
| `scripts/replay_original_vggt.py` | `eb793e40e126753c4c4714547b0cae9e102c4f3da7f93d6071dbb8474f3821e4` |

The [baseline](baseline-command.json), [candidate](candidate-command.json)
and [replay](replay-command.json) command records retain the exact executed
arguments. The examples below use new output and monitor-stage names; do not
overwrite the historical `r1` evidence. They assume a board checkout with the
repository history available to extract the historical comparator.

```bash
set -e
BASE=/mnt/vggt-data/vggt-slam2
cd "$BASE/src"
. deploy/rk3588/env.sh
EXP="$BASE/experiments"
RUN_PY="$BASE/venv/bin/python"
MONITOR=deploy/rk3588/validation/original_staged_pageout/run_board_stage.py
COMPARE=deploy/rk3588/validation/original_full_aggregator/compare_complete_model.py
GROUP="$EXP/vggt-v67-retained-groups-r1/compiled.json"
GROUP_SHA=771a9b82007d2feefd65ebbf3f4b0fb27070b79ff7f6d0faab1a8c866da90e43
DINO="$EXP/vggt-v58-dino-groups-r1/compiled.json"
DINO_SHA=a602bcdaab3f2e1ab51d49feb300ff3af76c07fe334489c983d647859f3f9493
DEPTH="$EXP/vggt-v54-depth-graphs-r1/compiled.json"
DEPTH_SHA=9a6e5435df543b8484a4fc50da4b065995f32153233548829234dd149fbc7135
NEON="$EXP/vggt-v56-neon-build-r1/fp16_neon.so"
NEON_SHA=d08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9
COMMON=(--manifest "$GROUP" --manifest-sha256 "$GROUP_SHA"
  --retained-aggregator --paired-aggregator
  --half-io-library "$NEON" --half-io-library-sha256 "$NEON_SHA"
  --model-threads 1 --model-cpus 4 5 6 7
  --dino-manifest "$DINO" --dino-manifest-sha256 "$DINO_SHA" --grouped-dino
  --depth-manifest "$DEPTH" --depth-manifest-sha256 "$DEPTH_SHA"
  --reference-run "$EXP/vggt-v53-complete-ab-r1"
  --save-aggregator-outputs --camera-split4)
BASELINE_SCRIPT="$EXP/original-staged-baseline-0f01c2d-new.py"
test ! -e "$BASELINE_SCRIPT"
git show 0f01c2d64fa2a92750f4853074effd2c0d91a789:"$COMPARE" > "$BASELINE_SCRIPT"
test "$(sha256sum "$BASELINE_SCRIPT" | cut -d ' ' -f 1)" = \
  1e1d28e11f2f476b875593d272526bf816ac3049fa3b69309acaf10a08229927
```

Run the baseline, then the candidate, keeping the same original inputs and
backends. The read-only monitor requires the observed NPU frequency to remain
300 MHz, terminates at 75°C or 180 seconds, and does not change clocks.
Use unique stage names when repeating it.

```bash
"$RUN_PY" -B "$MONITOR" --stage reproduce-baseline-new --timeout 180 -- \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$RUN_PY" -B -u "$BASELINE_SCRIPT" \
  "${COMMON[@]}" --output "$EXP/original-staged-baseline-new"
"$RUN_PY" -B "$MONITOR" --stage reproduce-candidate-new --timeout 180 -- \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$RUN_PY" -B -u "$COMPARE" \
  "${COMMON[@]}" --checkpoint-pageout --output "$EXP/original-staged-pageout-new"
"$RUN_PY" -B "$MONITOR" --stage reproduce-replay-new --timeout 180 -- \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$RUN_PY" -B -u scripts/replay_original_vggt.py \
  --rgb-index "$BASE/datasets/rgbd_dataset_freiburg1_xyz/rgb.txt" \
  --checkpoint "$BASE/weights/model.pt" --output "$EXP/original-staged-replay-new" \
  --image-size 112 --model-threads 1 --model-cpus 4 5 6 7 \
  --rknn-manifest "$GROUP" --rknn-manifest-sha256 "$GROUP_SHA" \
  --rknn-paired-aggregator --rknn-retained-aggregator --rknn-camera-head \
  --rknn-dino-manifest "$DINO" --rknn-dino-manifest-sha256 "$DINO_SHA" --rknn-grouped-dino \
  --rknn-depth-manifest "$DEPTH" --rknn-depth-manifest-sha256 "$DEPTH_SHA" \
  --half-io-library "$NEON" --half-io-library-sha256 "$NEON_SHA" \
  --rknn-camera-split4 --checkpoint-pageout
```

Do not constrain the outer replay process to A76 CPUs: its decoder needs CPU0.
The model affinity is specified separately above. Replay counts each unique
new observation once; its seed is not a fresh-pose update. Latency extends
from software arrival through decode, the single-slot queue, the complete
original model and SL4, dense-map saving, and atomic pose/map publication.
Throughput includes the final EOF drain, but excludes startup.

## Safety and evidence boundaries

The original loader verifies the fixed checkpoint size and SHA before
deserialization. The helper binds stable pre/post-load stat identity, pinned
loader sources and the process's own maps/smaps device, inode, offsets and
page-aligned extent. Only clean private file mappings qualify; anonymous,
dirty, swapped, locked, huge or device mappings are rejected. Tensor/storage
addresses and versions plus VMA identity are checked before and afterward
without rereading tensor values. Each stage's receipt reference is persisted
before advice. Any error retains its receipt and triggers normal backend
cleanup, with no alternative advice, global cache operation or silent fallback.

Inspect the four `checkpoint-pageout-{stage}.json` receipts, complete output
byte equality, map/pose timestamps, active residency/fault observations, and
all 12 context lifecycles and the available output-release counters. Separate
release counters were not saved for every backend. A successful advice return does
not by itself prove reclaimed bytes. Later checks must still read and verify
the original parameters. Pageout cannot undo an already reached high-water
mark; the lower measured peak here comes from intervening between installs.

The generated tests exercise fake advice and backend lifecycles; they do not
execute a model, RKNN SDK or `madvise`. Checkpoints, captured arrays, ONNX/RKNN
and native libraries remain private. Repository code licensing and original
weight licensing are separate; see [model provenance](../../../../MODEL_LICENSES.md).

See the [curated measurements](../../../../results/original_staged_pageout_v74.json),
[saved complete comparison](complete-staged-audit.json),
[independent memory audit](saved-complete-memory-audit.json),
[independent replay audit](saved-replay-audit.json), and
[source review](source-review.json). The replay audit checked 18 map chunks,
253,976 points, 19 pose rows including the seed, and 294 successful SDK runs.
