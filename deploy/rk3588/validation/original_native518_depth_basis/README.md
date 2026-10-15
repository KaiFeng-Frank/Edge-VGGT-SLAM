# Original native DPT basis consumer

This board CPU test passes the unchanged numeric FP32 gate for the complete
original depth head in mean-preserving basis coordinates. Both real frame
chunks also pass the independent elementwise folded-versus-baseline check.
This is a consumer equivalence result, not a quantization or speed result.

Four original patch-token inputs have shape `[1,1036,2048]`; raw logits are
`[1,2,392,518]` per frame. The two channels remain the original depth and
confidence logits. Original `exp`/`expp1` activation remains outside the graph.
The inputs come from the pinned original CPU continuation whose final outputs
were bitwise equal to the original complete capture; they are not outputs
from the newly mixed-precision aggregator.

For `T=diag(R1024,R1024)`, the consumer receives `u=zT`. Bare LayerNorm keeps
its original epsilon `1e-5`. Shared affine parameters are folded into each
original projection as `W_fold=(W*gamma)T`, `b_fold=W*beta+b` using FP64 offline
arithmetic followed by FP32 storage. All 62 original learned values are
verified against their captured identities before deriving replacements.
Exactly 10 initializers change. All 79 node protos, 66 other initializer
protos, five positions, resize geometry and residual alias behavior stay
byte-identical. Restoring those 10 parent tensors restores the entire parent
ModelProto exactly; no inverse-basis operator is added to the depth graph.

| Frame | Folded / baseline relative L2 | Maximum absolute error | Strict outliers |
| --- | ---: | ---: | ---: |
| 0 | 2.93531e-7 | 4.38690e-5 | 0 |
| 1 | 3.32407e-7 | 5.34058e-5 | 0 |

The fixed numeric gate is relative L2 ≤1e-5 and maximum absolute error ≤1e-3.
Elementwise `rtol=1e-4, atol=1e-5` is a separate diagnostic. Comparisons with
original captured Torch logits pass the numeric gate but have 15/18 baseline
and 11/30 folded elementwise outliers. Outputs are not byte-identical, and
this record does not claim strict equality to Torch.

Independent saved-only audit verified all six NPY files, recomputed all six
comparisons, and checked source, graph, original-parameter and receipt links.
Four sequential CPU ORT calls closed their four sessions. Each frame's CPU
stage took about 22 seconds including preparation and I/O. The recorded
600-second CPU guards wrote no frequency, observed 1 GHz, stayed below
67.461°C and left no worker process group. These durations are not NPU
performance measurements. No RKNN compilation, consumer quantization,
activated-depth comparison, whole-model map/trajectory or realtime acceptance
was performed here.

Reproduction requires private pinned original graph, reference chunks and
source provenance; none is shipped or automatically downloaded. The reviewed
`run_export_r1.sh`, `run_check0_r1.sh` and `run_check1_r1.sh` record the tested
worker arguments. Stage their pinned worker/proof bundle and invoke the existing
CPU controller for export then each check. For another run, use fresh source
and output paths and update launcher paths explicitly. `sources.json` lists
executed identities and the public-only auditor path adapter.
Stage `run_export_r1.sh` as `run.sh` beside the pinned worker, proof and
controller sources. The recorded export controller was invoked as follows; the recorded check launchers use the same controller with fresh paths:

```sh
BASE=/mnt/vggt-data/vggt-slam2
STAGE=/dev/shm/edge-vggt-v130-depth-basis-export-r1-source
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH="$BASE/sysroot/usr/lib/aarch64-linux-gnu" \
  "$BASE/venv/bin/python" -B -u "$STAGE/run_sampled_cpu_stage_r126.py" \
  --expected-npu-hz 1000000000 --stage native518depthbasis130exportr1 \
  --timeout 600 -- /bin/sh "$STAGE/run.sh"
```

See the [aggregate record](../../../../results/original_native518_depth_basis_v130.json).
Repository code, original weights and vendor dependency licenses are separate.
