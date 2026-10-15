# Original native518 residual-basis graph review

This is a **source review and generated algebra check**, not a converted VGGT
model or a board performance result. The selected research hypothesis changes
the residual coordinate system across the original transformer chain to make
low-precision representation less sensitive to isolated channel magnitudes.
It preserves the original real-valued function and dense arithmetic count;
finite-precision equivalence and any useful speed or quality gain remain untested.
See the [review record](../../../../results/original_native518_graph_basis_v125.json).

## The graph transformation

For row vectors, use `z = x R`, where
`R = ((J4 - 2 I4) / 2) tensor-power 5`. This fixed 1024-dimensional matrix is
orthogonal, symmetric and preserves the all-ones vector. Therefore the
**original bare LayerNorm**, with its mean subtraction and unchanged epsilon,
satisfies `LN0(x R) = LN0(x) R` in real arithmetic. An arbitrary orthogonal
matrix does not have this property.

For each original pre-norm residual branch, fold the original affine terms
and the basis into adjacent linear operations:

```text
Incoming row-vector linear:
    W_new = R.T @ diag(norm_gamma) @ W
    b_new = norm_beta @ W + b

Outgoing row-vector linear, including original LayerScale:
    W_new = W @ diag(layer_scale) @ R
    b_new = (b * layer_scale) @ R
```

The input formulas apply to QKV and FC1. The output formulas apply to attention
projection and FC2. Norm1/norm2 must then use bare normalization; applying the
original affine parameters again would be incorrect. Torch stores linear
weights as `[out, in]`; the transposed formulas are recorded in the JSON.
Original QKV and FC1 results stay in their original coordinates, so q/k
normalization, RoPE, softmax and Erf are not moved through the rotation.

The same basis must persist across frame/global blocks and compiled groups.
Adding forward/inverse rotations at every block defeats the intended graph
transformation. Retained frame/global features use `diag(R, R)` and must be
restored to their canonical coordinates before existing consumers, unless
those consumers are separately proven to absorb the inverse. In particular,
CameraHead's input-dependent adaptive scales and gates cannot be treated as
constant linear weights. Original heads, layers, tokens and refinements remain.

## Evidence and next gate

The [standalone checker](check_mean_preserving_basis.py) uses only Python scalar
arithmetic and generated values. It checks the exact integer base identities,
mean/norm preservation and LayerNorm commutation up to dimension 1024, as well
as affine-bias, LayerScale and residual-fold orientation. A counterexample
rejects arbitrary orthogonal LayerNorm commutation. The largest generated
float64 discrepancy is approximately `1.1e-14`; this is **not** a VGGT FP32
accuracy measurement. The [frozen receipt](mean-preserving-basis-algebra.json)
records all scopes. To rerun, copy the checker into a fresh scratch directory;
it refuses to overwrite an existing receipt.

First validate a complete original frame0/global0 pair on identical real
inputs in FP32, including inverse-transformed outputs and unchanged internal
QKV/FC1 coordinates. Include an `R=I` control with the same affine folding to
separate its effects from rotation. Only then assess actual RKNN placement, quantization and
complete multi-block timing, with separate calibration and evaluation data.
Folded bias rounding, weight underflow and output conversion are explicit
risks. All transformed weight-bearing artifacts remain private and derive
solely from the pinned original checkpoint.

Rotation-assisted quantization has a
[published precedent](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html).
No external implementation, kernels, checkpoint or reported speed is imported.
This LayerNorm-specific construction requires its own numerical and hardware
validation. It does not yet address the measured SDPA bottleneck or establish
any full-model, trajectory, 20 Hz, P95, map-cadence or 8 GB acceptance.
