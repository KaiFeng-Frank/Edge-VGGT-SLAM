# Small-denominator reciprocal experiment

This optional DPVO encoder experiment changes ten pinned spatial normalization
expressions from `centered / denominator` to `centered * (1 / denominator)`.
The denominators contain 480 elements altogether, while the corresponding
centered tensors contain 16,926,720. The original 65 initializer protobufs,
unselected nodes, graph metadata and input/output contracts remain byte-exact;
one nonlearned FP32 scalar is added. Division and reciprocal multiplication
are not assumed bitwise equivalent or quantization equivalent.

The spatial parent already fails the original CPU FP32 gate. Every stage
retains that failure, and no candidate is adopted. The default VGGT path and
original weight-loading behavior are unchanged. DPVO code attribution is MIT;
this does not grant separate rights to external neural weights.

The sequence is:

1. Run `rewrite-v38-spatial-reciprocal.py` for static graph/identity checks.
2. Execute `run-v38-spatial-reciprocal-ort-board.py` on RK3588S with the original
   saved three-frame inputs and references. Both graphs use ORT_DISABLE_ALL,
   the CPU provider and four A76 threads. Exactly six forwards are diagnostic;
   the candidate must pass `rtol=1e-4, atol=1e-5` against its current parent.
3. Run `compile-v38-reciprocal-int8-board.py` in the isolated board Toolkit2
   environment. Calibration executes real model operations and belongs on the
   board. The same frozen 32 rows, 64 file identities and PTQ configuration
   must match the prior spatial PTQ receipt.
4. Query actual SDK I/O, then run ordinary and separate profile diagnostics.
   Never infer scales, strides, operator placement or precision from compilation
   success. All diagnostic outputs are separately saved and compared.

Each board entry point requires `--diagnostic-allow-fp32-rejected`, an absent
output directory and pinned local inputs. The historical `orb-v12-dino-block`
directory is only an ORT dependency location; no ORB frontend runs here. No
new images are decoded or selected in the CPU comparison, and calibration
selection does not use GT or candidate outputs.

[Actual results and saved evidence](../../../../results/dpvo_reciprocal_encoder_v38.json)
record the graph, CPU, calibration and deployment stages separately. Model
weights, ONNX/RKNN files, SDK files and image/feature arrays are private
experiment artifacts, excluded from this repository. Component measurements
do not establish a fresh-pose rate, camera latency, map quality or endurance.
