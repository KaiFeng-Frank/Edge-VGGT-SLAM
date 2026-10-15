# Connected original native518 residual basis

One real `[32,40]` image pair passes all 12 fixed numeric FP32 gates through
the complete original 48-block aggregator and both original DPT frame chunks
on RK3588S. Each group consumes the actual preceding global output; no saved
reference activation is inserted between groups. This extends the separate
[DPT consumer gate](../original_native518_depth_basis/README.md).

The run starts at the captured original DINO output `[2,1041,1024]`, with
images originally prepared as `[2,3,392,518]`. It retains all original tokens,
attention keys and layers. Four groups cover pairs `0–4`, `5–11`, `12–17` and
`18–23`; original frame/global parameter identities total 864 before deriving
the folded weights. Original positions, Q/K normalization, RoPE and Erf
nonlinearities remain in their original computation coordinates.

The mean-preserving orthogonal residual basis persists across all four groups.
Frame-first/global-second retained features feed the already checked folded
DPT projections directly. Only the final two camera token-0 rows are restored
to canonical coordinates. Raw global20 is restored on a separate side branch;
the continuation to pair21 still consumes its basis value. Its original target
clone/normalization and CameraHead refinements are outside this gate.

The unchanged numeric gate requires relative L2 ≤1e-5 and maximum absolute
error ≤1e-3. All nine group exits, the camera input boundary and two raw DPT
logits pass. Worst relative L2 is `4.69435e-6` at raw global20; worst absolute
error is `1.77383e-4` at frame0 logits. Independent elementwise diagnostics
(`rtol=1e-4, atol=1e-5`) fail at some later exits and both logits. These outputs
are not byte-identical to the original captured reference.

Independent saved-only audits verified all four conditional group checks and
the connected input/output dependencies, frame/global concatenations, eight
DPT input hashes, nine group arrays, two camera-row arrays and two logits.
Five sequential CPU ORT sessions performed six calls and closed cleanly.
The controlled CPU stage lasted 294.113 seconds including source validation,
model loading, I/O and thermal pauses; it stayed below 70.23°C and wrote no
frequency. This duration is not NPU inference performance.

The first group0 FP16 compilation reached its 900-second CPU limit and was
killed after the cleanup grace period, without an RKNN artifact or a final
SDK-release receipt. Its process group exited. No NPU speed or precision
result follows from that failed compilation.

Reproduction requires the private pinned original source, checkpoint provenance,
four weight-bearing group graphs, DPT graph and matched captures. None is
shipped or automatically downloaded. `sources.json` records the tested board
workers and launchers. First reproduce the four conditional checks and DPT
checks; then stage `run_connected_r1.sh` as `run.sh` beside their pinned helpers.
The recorded connected CPU controller command was:

```sh
BASE=/mnt/vggt-data/vggt-slam2
STAGE=/dev/shm/edge-vggt-v130-connected-basis-r1-source
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH="$BASE/sysroot/usr/lib/aarch64-linux-gnu" \
  "$BASE/venv/bin/python" -B -u "$STAGE/run_sampled_cpu_stage_r126.py" \
  --expected-npu-hz 1000000000 --stage native518basis130connectedr1 \
  --timeout 900 -- /bin/sh "$STAGE/run.sh"
```

Use fresh stage/output paths and update launcher arguments for another run.
This gate does not execute DINO, CameraHead, pose codec, depth postactivation
or SL4; it does not establish trajectory quality, realtime rate, NPU deployment
or physical 8 GB operation. See the [aggregate record](../../../../results/original_native518_connected_basis_v130.json).
Repository code, original model weights and vendor dependencies have separate licenses.
