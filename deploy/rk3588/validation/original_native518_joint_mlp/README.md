# Original native518 joint MLP screen

R129 preserved the original frame0/global0 equations and all 36 original parameter
tensors, but rejected the current FP16 candidate as an effective acceleration.
The matched A/J/J/A test measured **853.10 ms for original A and 843.93 ms
for the joint graph**. The 1.08% descriptive difference failed the declared
10% component screen. It is not a sustained-throughput or full-model result.

Only global0 changes: the four existing FC2 K4 partitions receive their
matching 1024 FC1 output columns, original bias slices and five-node Erf
GELU trees. All 4096 hidden neurons and complete native518 token sets remain.
Frame0, FC2 weights, the three left-associated partial sums, final bias,
LayerScale and residuals retain their original values and arithmetic.
The actual export passed the original tensor-value and whole-parent inverse proofs.

Six real pre-frame0 inputs from original NPU DINO24 were checked on board.
An independent saved-array audit confirmed all 12 FP32 canonical outputs
byte-equal to same-input original A references (relative L2 and max error 0).
The unchanged numeric gates are relative L2 ≤ 1e-5 and max error ≤ 1e-3;
strict allclose (`rtol=1e-4`, `atol=1e-5`) is an independent diagnostic.
These are conditional two-block references, not full-model Torch accuracy.

Source order interleaves each FC1/GELU tile with its original FC2 consumer.
The actual compiler table instead lists **all four FC1/ConvExGelu operations
before any FC2**, in branch order 3,0,2,1. Thus the intended interleaving did
not survive compilation. Compiler rows are not hardware timing counters.
SDK-reported internal memory remained **92,124,480 bytes** for both routes;
no hidden-lifetime or physical memory reduction was established.

Normal measurements use core1, flags0, four fresh A/J/J/A contexts and two
heldouts per context: eight calls and 16 owned outputs, with no hidden warmup.
The call timer includes input preparation/SDK conversion, set/run/get, owned
copies, release and finite checks. Context lifecycle, saves and comparisons
are separate. Independent saved-array auditing found all 16 outputs byte-equal across
A/J and their repeats, while both
routes still fail the fixed FP32 numerical gates. Defaults are not promoted.

The parent matches the currently selected R110/v100 baseline, which excludes
the R105 frame-FC2 K4 candidate. R104–R106 already tested that transformation;
this screen does not benchmark the best previously tested frame-FC2 layout.

All four contexts, eight runs and 16 output buffers closed cleanly. The 180 s,
75°C controller observed at most 63.769°C and restored the prior 1 GHz NPU
state. Peak worker RSS and SDK allocation fields do not establish 8GB fitness.

Reproduction requires private pinned R126/R128 graphs, captures, references
and original A RKNN artifacts. The board-only export/check, FP16 compiler and
normal runtime use fresh directories and recorded controllers. No artifacts
are automatically downloaded; no weights, arrays, model binaries or SDK are
included. Code, weights and vendor-library licenses remain separate.

See the [verified result summary](../../../../results/original_native518_joint_mlp_v129.json).
