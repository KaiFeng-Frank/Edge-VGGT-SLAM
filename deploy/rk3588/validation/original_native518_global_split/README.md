# Experimental complete native518 global0 split

This check preserves the complete original global0 block: 53 nodes, all 44
initializers and 18 learned parameter tensors. Exact cuts contain 20 prefix
nodes, five original attention nodes and 28 suffix nodes; their union restores
the original parent protobuf. Board FP32 composition was byte-identical to
the parent and passed the fixed original-reference gate.

Route A runs the original complete RKNN block with core mask 7. Route B runs
prefix 7, actual prefix Q/K/V through concurrent 6/5/5 attention heads on
masks 1/2/4, then suffix 7 with the original input residual. Every head keeps
all 2082 queries/keys and dimension 64. No CPU reference intermediates are
substituted. Prefix closes before the head contexts; all head workers close
and join before suffix initialization. At most one weighted context is live.

Each call times its complete fresh-context lifecycle: model/helper/library
hash checks, initialization, input/layout copies, SDK IO, owned copies/release,
worker joins, destruction and finite/residual-integrity checks. SDK logical
rank3/rank4 formats are queried and validated explicitly. The ordinary flags0
ABBA trial makes two calls per phase, eight total. The first invocation of
each route is reported separately; three subsequent lifecycle samples per
route form the means. Every call constructs and destroys its contexts.

Private prerequisites must already exist: original checkpoint/source provenance,
native input/output captures, exact export/check/compile receipts and graphs,
R112 head artifacts/runtime, frozen S100/S99 owned IO helpers, RKNN 2.3.2,
and board Python/sysroot. This folder supplies validation source only; weights,
captured arrays, compiled graphs and SDK binaries are not shipped. It supplies
no automatic download/build workflow. Code and model-weight licenses remain
separate; see [model provenance](../../../../MODEL_LICENSES.md).

| Source | SHA256 |
| --- | --- |
| `export_global0_prefix_suffix.py` | `ac3d9e8a7f3ff2d628f843a374c9eba7c5ab6a35a99e8465a39f84cb491d486c` |
| `check_compile_global0_split_board.py` | `990b4e97cdfc1552285f08df1a1154b9ff2ae8a9a63190a7c939cc1e7eae3178` |
| `run_global0_split_abba.py` | `9b35689eed8f282b7f77022721f83a80e84b7abd046d6b589ebd2e54bbbad820` |

After staging verified files in S114, use the bounded controller. The launcher
sets process-local library/thread settings and exact receipt pins. For a new
run, change its worker output directory and controller output to fresh paths.

```bash
BASE=/mnt/vggt-data/vggt-slam2
S114="$BASE/experiments/vggt-v114-source-r1"
PRODPY="$BASE/venv/bin/python"
CONTROL="$BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py"
"$PRODPY" -B "$CONTROL" --execute-frequency-test --frequency-hz 1000000000 \
  --temperature-limit-c 75 --timeout-seconds 180 \
  --launcher "$S114/run_board_abba.sh" \
  --output "$BASE/experiments/vggt-v114-abba-control-new"
```

Observed full-lifecycle means were A **0.644933 s**, B **1.399067 s**:
B was **2.1693×**, or **116.93% slower**. All eight complete outputs were
byte-identical across routes/repeats. Both failed the unchanged CPU FP32 gate
(relative L2 0.00570641, maximum absolute error 0.0171948). All 24 contexts,
24 runs and 32 output buffers balanced; the controller restored 300 MHz.

Recorded A IO averaged 0.416827 s. B prefix IO, head dispatch/join and suffix
IO averaged 0.261989 + 0.217172 + 0.153397 = 0.632558 s. B initialization
averaged 0.494101 s and destruction/shutdown 0.247930 s. Removing those terms
arithmetically is not a measured persistent-context schedule: partitioning
also changes compiler boundaries and data movement. Constructor hashing read
37,594,711 logical bytes per A call versus 77,226,777 per B call; these counts
are not measured DDR traffic or cold-page-cache costs.

This tested schedule is rejected without production integration. It does not
establish that other head-parallel schedules are impossible, or demonstrate
full-model/SLAM quality or realtime operation. See the
[compact result](../../../../results/original_native518_global_split_v114.json).
