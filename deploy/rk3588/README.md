# Original VGGT deployment on RK3588S

This directory supports VGGT-SLAM2 with its original pretrained VGGT model only. Real inference, tracing/export validation and calibration must run on RK3588S. Host tools may inspect sources/static graphs, audit saved arrays and run small generated checks. Sustained 20 Hz, complete-latency P95, full-model numerical/trajectory quality, 30-minute stability and 8 GB operation have not been established.

Use the pinned source and checkpoint in [UPSTREAMS.json](../../UPSTREAMS.json). Install the original implementation from the repository root:

```sh
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt
git -C third_party/vggt checkout 6e6e16107b88e8e76c751826af10d4295d87ecd2
git -C third_party/vggt apply ../../deploy/rk3588/vggt-spark.patch
```

The only supported weight file is original `facebook/VGGT-1B` `model.pt`, 5,026,874,952 bytes, SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. The [download helper](download_weights.py) verifies this identity. Weights are external and are not licensed by repository BSD terms; see [checkpoint provenance](../../MODEL_LICENSES.md).

The tested runtime uses Linux aarch64, Python 3.11.17, CPU PyTorch 2.3.1, torchvision 0.18.1 and an external GTSAM build exposing SL4. [Dependency manifests](dependencies/README.md) and the [offline installer](install-offline.sh) describe externally supplied packages; binaries and wheels are not committed.

Additional-model retrieval and semantic paths have been removed. There is no replacement loop implementation: use `max_loops=0`; unsupported loop/legacy auxiliary-model requests are rejected before loading. The runtime adds no separate traditional pose frontend. Consult `python main.py --help` for current dataset arguments.

Original VGGT operator diagnostics are under `validation/`. Each board helper requires its declared source, checkpoint, input and artifact pins. A whole model, a single DINO block, CameraHead and an attention block have different timing and quality scopes. Use the [original-model evidence index](../../README.md) and [engineering notes](../../docs/ENGINEERING_NOTES.md); component compilation does not remove recorded FP32 failures or prove complete-model real-time performance.

RKNN diagnostics need matching licensed external SDK headers and `librknnrt.so`; tested API/runtime is 2.3.2. XNNPACK diagnostics also require matching external headers/runtime. These files are not in a fresh clone. Keep actual ABI queries, complete I/O timing, output ownership and resource cleanup in every hardware receipt. Do not change global clocks, governors, cooling, drivers or network settings without authorization.
