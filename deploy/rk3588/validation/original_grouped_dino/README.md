# Original VGGT DINO graph grouping

This optional backend runs the original 24 DINO layers in six groups of four. Every original node and all 336 block parameter tensors remain in the exported graphs; token preparation and final normalization retain their original CPU FP32 implementation. The original CameraHead, DPT, aggregator and SL4 path are unchanged. No auxiliary model or pose frontend is added.

The backend replaces the DINO `forward_features` block loop and returns its original dictionary. It does not fabricate intermediate block outputs. Block/submodule and global module hooks that grouping would bypass are rejected. Outer DINO hooks and the actually executed token preparation and normalization remain active. Install before the original tail cache and close the cache first.

The fixed input is one or two RGB images at 84×112. Each graph accepts one `[1,53,1024]` token tensor; both images run on a cold pair, and a valid tail-cache hit reuses only the unchanged old image. FP16 I/O uses the existing verified NEON implementation. All original layers and tokens still execute.

Use the existing [paired aggregator replay command](../original_fused_aggregator/README.md) with the [FP16 I/O options](../original_half_io/README.md), replace the DINO manifest with the six-group `compiled.json`, and add `--rknn-grouped-dino`. The matched complete-model comparator accepts `--grouped-dino`. Both options require the DINO manifest and FP16 library. Runtime manifests and all weight-bearing files stay private.

Graph production runs only on RK3588S using the existing isolated RKNN Toolkit2 2.3.2 environment:

1. `compose_original_dino_group.py` combines the unchanged original layer 0–3 ONNX files from the pinned 24-layer manifest, using `--boundary-mode fp32`.
2. `check_compile_dino_group_r2.py --mode check` writes to the group directory's `ort-check-r2` subdirectory, comparing both original captured inputs against the original Torch FP32 output and the four separate ONNX graphs. Then `--mode compile` writes the diagnostic FP16 RKNN artifact.
3. `build_all_dino_groups.py` takes the parent manifest and pinned group0 export/check/compile receipts, reuses group0 and sequentially builds groups 4–7 through 20–23 in isolated processes. Supply all corresponding `--*-sha256` arguments and a new `--output-dir`.
4. Run the full comparator on the original matched inputs and the complete timestamp replay before using the candidate. `probe_dino_group_board.py` is only a component diagnostic with a fixed group0 artifact identity.

Original Torch FP32 strict failures remain in the records. Byte-identical ONNX composition, successful compilation or component timing does not establish numerical equivalence, trajectory quality or real-time performance. The public result record reports complete-model and replay evidence separately. The code license grants no additional rights to the original model weights; see [model provenance](../../../../MODEL_LICENSES.md).
