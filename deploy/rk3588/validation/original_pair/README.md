# Original VGGT frame0 → global0 execution probe

This diagnostic keeps the complete first frame and global blocks of the original VGGT checkpoint. It introduces only the original C-order reshape from `[2,437,1024]` to `[1,874,1024]`. It does not implement the complete model or a pose frontend. Actual tensors come from the same two-image TUM capture at actual image shape `[2,3,252,336]`.

The static exporter verifies both frozen parent graphs and their board CPU checks, preserves 36 original parameter tensors (100,788,224 FP32 bytes), and validates every parent node and initializer after reversing the namespace. The 87-node pair contains 86 parent nodes plus one reshape. Original normalization, RoPE, exact GELU, LayerScale and residuals remain.

## Execution sequence

1. Use `export-v47-frame-global-pair.py --help` for the frozen capture, parent export and parent CPU-check inputs. Static composition and ONNX checker may run on the host; they do not execute the captured model.
2. On RK3588S, source the deployment `env.sh`. Make ONNX Runtime 1.20.1 available in that process; it is an external dependency, not bundled here. Run `check-v47-pair-onnx-board.py` with the new pair directory, exact export-receipt SHA, contracts path/SHA and a new output directory. This performs four CPU-only ORT runs and compares every final element against the original CPU global0 reference with fixed `rtol=1e-4, atol=1e-5`.
3. After the CPU check, run `compile-v47-pair.py` in the isolated RKNN 2.3.2 / Torch 2.4.0 / NumPy 1.26.4 environment. Both cwd and `TMPDIR` must be on `/dev/shm`. Supply the exact CPU receipt SHA and the original `compile-v12-rknn-block.py`. This uses only config/load/build/export, `do_quantization=False`; no host inference or calibration is performed.
4. On RK3588S, run `run-v47-pair-rknn-board.py` with the compiled receipt/model SHA and the original `run-v12-rknn-block-board.py`. The exact SDK 2.3.2 header/runtime are external prerequisites. The explicit diagnostic flag records FP32 failures without accepting the candidate. A separate invocation and output directory with `--collect-perf-detail` collects operator placement.

Each command's `--help` lists required paths. Board cwd must be the deployed project root; output directories are newly created under `experiments`. Do not overwrite an earlier receipt. All private artifacts are bound by SHA. The pair export receipt is `149450ffc675ce59950c6f89cf84adb41e388cd050eeb731b6d09a6fe16e9fe1`; graph SHA is `2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc`. Checkpoints, captured arrays, ONNX/RKNN files and SDK binaries are not distributed.

## Verification and limits

The source freeze covers 23 small generated-graph/fake-runtime checks, not actual model execution. Run them with NumPy, ONNX and pytest:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  deploy/rk3588/validation/original_pair/test_v47_frame_global_pair.py \
  deploy/rk3588/validation/original_pair/test_v47_pair_runtime_contracts.py
```

The SDK path validates distinct rank-three input and output shapes, floating identity descriptors, logical byte counts and the pinned ABI. It performs blocking run/get, copies the output before release, destroys the context and restores process affinity. Normal IO timing includes input submission, run, retrieval, owned copy and release; init, error calculations, file saving and hashing are excluded. Profile clocks belong to separate calls and cannot be added to normal API clocks.

The old separate-block records feed CPU frame0 output into global0. Their sum is not a measured NPU chain. A proper sequential comparison must feed the actual preceding NPU frame0 output into global0. The original DINO, remaining 23 frame/global pairs, CameraHead, DPT, geometry and queues remain outside this probe. See the [new measured evidence](../../../../results/original_pair_v48.json) and [engineering record](../../../../docs/ENGINEERING_NOTES.md).
