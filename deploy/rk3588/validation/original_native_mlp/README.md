# Original VGGT native MatMul diagnostic

This frozen board diagnostic uses the sole original VGGT checkpoint's frame0
MLP, with both frames and all 53 tokens per frame. It adds no model or production
backend. All Torch, ORT, SDK and compiler execution occurred on RK3588S.

At 300 MHz, one NPU core, one Torch thread on CPUs 4–7, the matched rank3
compiled MLP took **15.139 ms** and the persistent native MatMul implementation
took **33.878 ms**, excluding each route's first call in an ABBAABBA sequence.
There are only three subsequent observations per route on one real input.
This native implementation is **123.77% slower** and is not adopted.

The native route packs the two original FP16 weights once in
`[N/16,K/32,16,32]` order, keeps separate FC1/FC2 contexts, uses normal A/C
layouts, and returns FP32 matrix outputs. Original FP32 bias, original Torch
GELU and eval dropout remain on CPU. Native M is exactly 106, without padding.
The mean original CPU GELU cost in the final comparison is 19.363 ms.
Each B is uploaded once; the existing allocator also binds its initial zero B,
so there are two setup binds, not one. Complete timings include conversions,
input synchronization, execution, owned output copies and release.

Native relative L2 error against the original MLP is 0.00013938; the fused
compiled route is 0.00511318 on this input. Both fail the strict original FP32
elementwise gate (`rtol=1e-4`, `atol=1e-5`). These are component observations,
not full-model precision or trajectory acceptance. The original Block
reconstruction is byte-identical to its saved original full-forward output;
the FP32 rank3 ONNX graph passes the original MLP gate before this comparison.

## Preserved failed baseline

`compile_mlp_board.py` is the first, rank2 diagnostic. Its first GELU Mul has
reversed operands compared with the frozen production graph pattern. The
compiler leaves Erf and a Transpose on CPU, and changes the FC1 layout. Its
43.093 ms versus 31.725 ms native comparison is **not evidence of improvement
over the existing production path**. The raw run and its hashes are retained.

`compile_mlp_rank3_board.py` restores original `[2,53,1024]` shapes and the frozen
GELU operand order. Its compiler log assigns ConvExGelu and FC2 Conv to NPU.
Both changes were made together; the experiment does not isolate which change
caused each compiler decision. The two FP32 ONNX outputs are byte-identical on
the saved input. The faster compiled result reverses the comparison outcome.
An isolated MLP still lacks the original surrounding LayerScale/residual
consumers, so it is not a complete production-block benchmark.

## Files and execution conditions

- `native_mlp_rank3_probe_board.py` verifies the pinned original capture,
  reconstructs its original Block, hooks the real MLP input, then runs
  Torch/native and optionally compiled/native comparisons. The final run uses
  `--threads 1 --cpu-cores 4 5 6 7 --core-mask 1` and the pinned NEON library.
- `compiled_mlp_rank3.py` binds the existing typed SDK ABI and owns one fixed
  rank3 graph context. Its public `run()` accepts flattened `[106,1024]` dense
  storage without changing original token order.
- `check_mlp_rank3_onnx_board.py` compares the rank3 ONNX output with the saved
  original Torch MLP output using board ORT with optimizations disabled.
- `compile_mlp_rank3_board.py` exports the original four learned tensors and
  exact Erf equation, then performs a non-quantized FP16 build with SDK 2.3.2.

These are exact experiment sources with frozen private capture/artifact paths
and hashes. The board root is `/mnt/vggt-data/vggt-slam2`; the compiler Python is
`experiments/vggt-v49-toolchain-venv-r1/bin/python`, and actual model/SDK Python
is `venv/bin/python`. Compiler execution unsets `PYTHONPATH`, `PYTHONHOME` and
`VIRTUAL_ENV`, while retaining the project's sysroot library path. All runs use
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Use `--help` for each script's inputs.
The final probe's optional helper SHA is
`f055e45eec073557e85f00f6ce3fab30e6ee212c1f32d928d9ac0c40e6055bc3`.

Output directories must be new. To reproduce elsewhere, create a separate
private experiment tree and explicitly update path/hash pins for locally
verified artifacts; do not overwrite retained evidence. Checkpoint, captured
arrays, ONNX/RKNN artifacts and SDK libraries are not included in this repository.
Their licensing remains separate from repository code.

The seven monitored stages finished at 300 MHz, with maximum sampled temperature
58.230 C. Native and compiled lifecycle counters reconcile. No clock, governor,
cooling or default execution path was changed. There is no new full-model
replay or real-time acceptance from this diagnostic.

[Measurements, source identities and raw-receipt hashes](../../../../results/original_native_mlp_v68.json).
