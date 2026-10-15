# Experimental original native518 SLAM integration

This board-only executor implements the existing `ModelExecutor` boundary and
feeds fresh prepared pixels through the original `Solver`, submaps and SL4
backend. It accepts exactly two RGB frames `[2,3,392,518]` after original 518
preprocessing. Each call executes all 24 original DINO layers per frame, all 48
aggregator blocks, four camera refinements, both DPT frames and the original
pose codec. Outputs use the canonical singular `extrinsic`/`intrinsic` names.
The production backend default is unchanged; this remains a validation path.

The existing private RK3588S installation is a prerequisite: original weights,
the frozen `vggt-v100-source-r1` helpers/config, all linked DINO/aggregator/head
compile receipts and RKNN payloads, runtime 2.3.2, the board Python environment,
bounded controller, TUM RGB files and saved v100 full-model reference must exist at
their pinned paths. These artifacts are not shipped here; this folder does
not automatically download or rebuild them. The original checkpoint is
5,026,874,952 bytes, SHA256
`d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`,
verified before deserialization. Helpers, original sources and compiled
artifacts also have independent identity checks.

| Staged source/config | SHA256 |
| --- | --- |
| `original_native518_executor.py` | `8923992b323f805c03d8e70441952b65aefe9f8a70a3cfc2241ac14b704b563f` |
| `run_native518_slam_pair_chain.py` | `d42ccc8d26c6f56e0abdd23c2e509a5c9355d3008c8d9f9fdfc735857f252d15` |
| `full-native-sequential-config-r1.json` | `7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b` |

After staging the verified sources in `vggt-v109-source-r1`, the recorded command
below launches the worker through the tested bounded controller: 1 GHz, at most
180 seconds, 75°C limit, then restoration of the previous clock/governor state.
`run_pair_chain.sh` sets the process-local library path, thread environment and
exact executor source pin. For another run, change its worker `--output-dir`
and the controller `--output` below to fresh paths before launching.

```bash
BASE=/mnt/vggt-data/vggt-slam2
S109="$BASE/experiments/vggt-v109-source-r1"
PRODPY="$BASE/venv/bin/python"
"$PRODPY" -B "$BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py" \
  --execute-frequency-test --frequency-hz 1000000000 \
  --temperature-limit-c 75 --timeout-seconds 180 \
  --launcher "$S109/run_pair_chain.sh" \
  --output "$BASE/experiments/vggt-v109-pair-chain-control-r1"
```

The recorded board run processed TUM indices `[32,40]` then `[40,48]` with
`max_loops=0`. The first pair's five canonical arrays matched the previous
complete v100 NPU run byte-for-byte. The second pair shared the exact overlap
pixels and introduced a new frame. Original scale estimation and SL4 optimization
produced two submaps, four nodes/factors, three unique pose timestamps and a
609,168-point PLY. All 20 contexts, 36 runs and 46 output buffers were balanced.
Read-to-pose-publication took 36.7352 s and 34.3896 s, including diagnostic saves;
this is an unpaced two-pair check, not realtime/P95, ATE/RPE, 30-minute stability
or 8 GB acceptance. Byte parity with v100 is separate from CPU FP32 equivalence.
See the [recorded result](../../../../results/original_native518_slam_integration_v109.json).

Repository code and original model-weight licenses remain separate; see
[model provenance](../../../../MODEL_LICENSES.md).
