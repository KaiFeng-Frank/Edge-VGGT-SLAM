# Original VGGT execution boundary

The implementation separates original VGGT execution from submap construction and SL4 optimization. Hardware implementations may provide the same model contract; this separation does not authorize a different learned model or pose source. [Project constraints](AGENTS.md).

```mermaid
flowchart LR
    A[Source images and original preprocessing] --> B[FP32 RGB image batch]
    B --> C[Original VGGT: DINO, frame and global attention]
    C --> D[Original CameraHead and DPT]
    D --> E[Original pose codec, depth and confidence]
    E --> F[Solver submaps and original constraints]
    F --> G[SL4 graph optimization]
```

## Execution contract

`ModelExecutor` receives already-preprocessed, C-contiguous CPU FP32 RGB arrays `[S,3,H,W]` in `[0,1]`. Frame order and original pixel geometry remain unchanged. Source indices and timestamps are separate metadata, without FP32 conversion. The original loader defaults to 518 pixels wide, with height rounded to the 14-pixel patch grid and its original crop/padding rules.

Canonical results are finite CPU FP32 arrays: OpenCV world-to-camera extrinsics `[S,3,4]`, intrinsics `[S,3,3]` in pixels, depth `[S,H,W,1]` and confidence `[S,H,W]`. Original pose encodings `[S,9]` may also be preserved. `TorchVGGTExecutor` uses the original VGGT call and original pose decoder. Caller pixels are read-only; result storage must survive until consumers finish. Any future SDK adapter must copy retained outputs before releasing SDK buffers and propagate device failures explicitly.

The original VGGT similarity request remains a model capability where implemented. It is not a retrieval system. Additional-model retrieval/semantic implementations have been removed; Solver supports only `max_loops=0`. Enabling loops or passing legacy auxiliary-model arguments must be rejected before preprocessing, state construction or inference. **Natural loop closure is currently unavailable and has not been replaced or validated.**

## State and geometry

Solver retains submap images, local camera matrices, dense points and confidence, then adds original consecutive-submap constraints and optimizes the SL4 graph. Changing execution backend cannot change those equations or state ownership. No additional traditional pose frontend, interpolated pose or reused output supplies fresh poses.

The model's local world-to-camera matrices and the graph's optimized projective homographies are different objects. `get_projection_matrix()` returns the inverse graph homography. Submap camera export additionally applies intrinsics, normalizes the projective matrix and uses original camera decomposition. Export keeps the latest pose for an overlap timestamp; duplicate graph nodes are not separate fresh observations. A projective SL4 world is not automatically a metric SE3 world.

The optional exact tail cache retains only the previous main batch's cloned DINO patch tokens. Content, shape, dtype and fixed-model/backend guards determine reuse. Each new pair still recomputes camera/register-token roles, frame/global attention, CameraHead and DPT. Cache/reference comparisons need independent Solver/map/graph owners and owned same-call captures. Global-attention outputs cannot be reused merely because one source image overlaps.

## Validation status

Original CPU execution and individual NPU operators have measured evidence. A complete accepted RKNN VGGT backend, sustained 20 Hz, complete-latency P95, trajectory-quality preservation, bounded long-run memory and 30-minute operation remain unvalidated. Canonical FP32 comparisons, double-precision graph comparisons, literal bit equality and timing each have separate meanings. No component measurement is promoted to whole-system acceptance. [Measurements](MEASUREMENTS.json), [engineering notes](docs/ENGINEERING_NOTES.md).

The [repository license](LICENSE) covers repository source. The [original checkpoint and pinned implementation terms](MODEL_LICENSES.md) remain independent and apply to weight-bearing compiled models as well.
