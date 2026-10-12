# Original VGGT frame/global FC1 output partition

This board diagnostic rewrites only the two FC1 MatMul/Add operations in the
original frame0/global0 pair. Each branch retains the full K=1024 reduction;
output channels N=4096 are split into two or four parts, then concatenated
before the unchanged original exact GELU. All 106 tokens, attention, residuals,
other parameter values, and the original two SDK outputs remain present.

At 300 MHz and core mask 7, the confirmation run used two distinct saved real
inputs and ten ABCCBA cycles per input. Excluding the first call per input and
graph leaves 38 calls per graph:

| Graph | Complete pair mean | Reduction |
| --- | ---: | ---: |
| Original | 39.019 ms | — |
| Two FC1 output parts | 37.850 ms | 3.00% |
| Four FC1 output parts | 37.914 ms | 2.83% |

The earlier, smaller run is retained. The order remains deterministic and the
inputs cover only this pair; these measurements do not establish gains across
the full backbone. Neither candidate is installed in the model runtime.

All eight FP32 ONNX candidate/original output checks are byte-identical. Four
checks with available original Torch references also pass the existing
`rtol=1e-4, atol=1e-5` gate; the second input has no saved Torch reference.
NPU outputs change relative to the original FP16 graph and fail the original
FP32 strict gate. No tolerance was relaxed.

Separate actual profiles show each FC1 changing from a single-core
ConvExGelu to two/four three-core Conv operations, Concat, and a single-core
exGelu. Branch count does not prove concurrent branch execution. Profile
times are excluded from the normal benchmark. No extra SDK output is added.

All actual ORT, compilation, and SDK calls run on the RK3588S. The host only
inspects source and audits saved arrays. Private graphs, arrays, and SDK files
are required to reproduce this diagnostic and are not distributed.

```sh
# Run on the prepared board, with fresh output directories.
# Compiler: existing v49 toolchain environment, SDK 2.3.2, one CPU thread.
python build_pair_fc1_splits_board.py --chunks 2 --output-dir SPLIT2_DIR
python build_pair_fc1_splits_board.py --chunks 4 --output-dir SPLIT4_DIR

# Runtime: existing board environment; use each exact result.json SHA256.
python probe_pair_fc1_splits_board.py \
  --compiled SPLIT2_DIR SPLIT2_RESULT_SHA256 \
  --compiled SPLIT4_DIR SPLIT4_RESULT_SHA256 \
  --rounds 10 --output-dir NORMAL_DIR

# Separate profiling process and output directory.
python probe_pair_fc1_splits_board.py \
  --compiled SPLIT2_DIR SPLIT2_RESULT_SHA256 \
  --compiled SPLIT4_DIR SPLIT4_RESULT_SHA256 \
  --profile --output-dir PROFILE_DIR
```

The compiler environment unsets `PYTHONPATH`, `PYTHONHOME`, and `VIRTUAL_ENV`,
uses the existing sysroot library path, and sets `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`. Run on CPUs 4–7. `run_board_stage.py` records child
termination and read-only frequency/temperature samples, stops its own process
group above 75 C or its deadline, and does not change clocks or cooling.
The initial probe source is retained to reproduce the first normal/profile
receipts; the revised probe only adds the configurable cycle count.

[Measurements, provenance, and limits](../../../../results/original_pair_partition_v70.json).
