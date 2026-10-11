# Original frame/global pairs

This backend composes each adjacent original frame and global block into one
RKNN graph. Each of its 24 calls returns both original outputs; the original
Aggregator still executes its loop, intermediate concatenations and target
token selection. No layers, tokens or parameters are removed.

Starting from the complete original 48-block manifest, run on RK3588S in the
compiler environment:

```sh
python -B export_original_pairs.py \
  --manifest ORIGINAL_48_BLOCKS/compiled.json \
  --manifest-sha256 ORIGINAL_MANIFEST_SHA256 \
  --output-dir NEW_PAIR_DIRECTORY
python -B ../original_full_aggregator/compile_original_blocks.py \
  --manifest NEW_PAIR_DIRECTORY/export.json \
  --manifest-sha256 EXPORTED_MANIFEST_SHA256
```

The exporter reuses the pinned original composer and preserves each parent
graph plus its original C-order view. A new directory is required. All graphs
and compiled artifacts contain original weights and must remain private.

For complete replay, pass `--rknn-paired-aggregator` together with
`--rknn-manifest NEW_PAIR_DIRECTORY/compiled.json` and its SHA256. Original
DINO, CameraHead and DPT options remain available. For the matched complete
model comparison, use `--paired-aggregator` with the pair manifest and the
existing DINO/depth/reference-run options.

Frame forward makes one SDK call and copies both outputs before releasing SDK
buffers. Global forward accepts only the same unmodified contiguous view of
that frame output, then consumes the corresponding pending global output.
Wrong order, a copied or changed input, unsupported bindings and SDK errors
stop execution. The backend does not reuse outputs across observations.

Install before the original DINO tail cache and close that cache first.
Generated service checks cover ownership and cleanup; numerical and timing
claims require complete board runs. FP16 execution does not imply equivalence
to the original FP32 reference or real-time acceptance. The original weight
license remains separate from the repository code license.
