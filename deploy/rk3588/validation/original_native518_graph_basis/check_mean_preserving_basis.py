#!/usr/bin/env python3
"""Generated scalar algebra checks, not a VGGT/model/SDK execution.

Use row vectors. R = ((J4 - 2 I4)/2) tensor-power k, D = 4**k.
The integer base proves orthogonality and preservation of the all-ones vector.
Floating examples check fold orientation, affine bias and LayerScale placement.
No weights, captured tensors, model packages or numerical libraries are read.
"""
import hashlib
import json
import math
from pathlib import Path
import random


def rotate(x):
    """Apply the symmetric regular Hadamard in O(D log D) scalar arithmetic."""
    n = len(x)
    size = n
    while size > 1 and size % 4 == 0:
        size //= 4
    assert size == 1 and n >= 4
    y = list(x)
    stride = 1
    while stride < n:
        for start in range(0, n, 4 * stride):
            for offset in range(stride):
                ids = [start + offset + j * stride for j in range(4)]
                values = [y[j] for j in ids]
                half_sum = math.fsum(values) / 2
                for j, old in zip(ids, values):
                    y[j] = half_sum - old
        stride *= 4
    return y


def norm0(x, eps):
    mean = math.fsum(x) / len(x)
    centered = [v - mean for v in x]
    denominator = math.sqrt(math.fsum(v * v for v in centered) / len(x) + eps)
    return [v / denominator for v in centered]


def linear(x, w, b):
    assert len(w) == len(x) and all(len(row) == len(b) for row in w)
    return [math.fsum(x[i] * w[i][j] for i in range(len(x))) + b[j]
            for j in range(len(b))]


def max_error(a, b):
    assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b))


def run():
    base = [[1 - 2 * int(i == j) for j in range(4)] for i in range(4)]
    assert [sum(row) for row in base] == [2] * 4
    assert [[sum(base[k][i] * base[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)] == [[4 * int(i == j) for j in range(4)] for i in range(4)]
    rng = random.Random(125)
    errors = dict(involution=0.0, mean=0.0, squared_norm_relative=0.0,
                  bare_LayerNorm_commutation=0.0, input_affine_fold=0.0,
                  output_LayerScale_fold=0.0, residual_addition=0.0)
    checked = 0
    for d in (4, 16, 64, 256, 1024):
        assert rotate([1.0] * d) == [1.0] * d
        for eps in (1e-6, 1e-5):
            for _ in range(4):
                x = [rng.uniform(-20, 20) for _ in range(d)]
                z = rotate(x)
                errors['involution'] = max(errors['involution'], max_error(rotate(z), x))
                errors['mean'] = max(errors['mean'], abs(math.fsum(x) - math.fsum(z)) / d)
                norm = math.fsum(v * v for v in x)
                errors['squared_norm_relative'] = max(errors['squared_norm_relative'],
                    abs(norm - math.fsum(v * v for v in z)) / max(norm, 1.0))
                errors['bare_LayerNorm_commutation'] = max(errors['bare_LayerNorm_commutation'],
                    max_error(norm0(z, eps), rotate(norm0(x, eps))))
                checked += 1

    fold_cases = 0
    for d in (4, 16):
        for eps in (1e-6, 1e-5):
            for _ in range(8):
                x = [rng.uniform(-4, 4) for _ in range(d)]
                gamma = [rng.uniform(-2, 2) for _ in range(d)]
                beta = [rng.uniform(-2, 2) for _ in range(d)]
                w = [[rng.uniform(-2, 2) for _ in range(7)] for _ in range(d)]
                b = [rng.uniform(-2, 2) for _ in range(7)]
                # R.T @ diag(gamma) @ W: rotate each input-channel column.
                columns = [rotate([gamma[i] * w[i][j] for i in range(d)]) for j in range(7)]
                folded_w = [[columns[j][i] for j in range(7)] for i in range(d)]
                folded_b = linear(beta, w, b)
                original = linear([n * g + t for n, g, t in zip(norm0(x, eps), gamma, beta)], w, b)
                transformed = linear(norm0(rotate(x), eps), folded_w, folded_b)
                errors['input_affine_fold'] = max(errors['input_affine_fold'], max_error(original, transformed))

                # Independent arbitrary branch coordinates, not a network forward.
                h = [rng.uniform(-2, 2) for _ in range(7)]
                wo = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(7)]
                bo = [rng.uniform(-2, 2) for _ in range(d)]
                ls = [rng.uniform(-2, 2) for _ in range(d)]
                original_branch = [a * s for a, s in zip(linear(h, wo, bo), ls)]
                wo_new = [rotate([a * s for a, s in zip(row, ls)]) for row in wo]
                bo_new = rotate([a * s for a, s in zip(bo, ls)])
                new_branch = linear(h, wo_new, bo_new)
                errors['output_LayerScale_fold'] = max(errors['output_LayerScale_fold'],
                    max_error(rotate(original_branch), new_branch))
                errors['residual_addition'] = max(errors['residual_addition'], max_error(
                    rotate([a + b for a, b in zip(x, original_branch)]),
                    [a + b for a, b in zip(rotate(x), new_branch)]))
                fold_cases += 1

    assert all(v < 1e-10 for v in errors.values()), errors
    # Arbitrary orthogonal matrices do NOT in general commute with LayerNorm.
    x = [1.0, 2.0, 4.0, 8.0]
    sign = [-1.0, 1.0, 1.0, 1.0]
    counterexample = max_error(norm0([a * b for a, b in zip(x, sign)], 1e-6),
        [a * b for a, b in zip(norm0(x, 1e-6), sign)])
    assert counterexample > 0.1
    # Algebraic illustration only; not a real activation distribution or calibration.
    impulse = [1024.0] + [0.0] * 1023
    spread = rotate(impulse)
    assert max(abs(v) for v in spread) == 32.0
    return {
        'schema': 'v125_generated_mean_preserving_residual_basis_algebra',
        'status': 'passed_generated_scalar_algebra_only',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'row_vector_convention': True,
        'basis': 'R = ((J4 - 2 I4)/2) tensor-power k; D=4**k',
        'integer_base_gram': 'S.T*S=4I',
        'integer_base_row_and_column_sums': 2,
        'Kronecker_identity': 'R.T*R=I, R*ones=ones, R.T=R',
        'LayerNorm_identity': 'LN0(xR;epsilon)=LN0(x;epsilon)R; same epsilon and dimension',
        'input_linear_fold': "W_in'=R.T*diag(gamma)*W_in; b_in'=beta*W_in+b_in",
        'output_linear_fold': "W_out'=W_out*diag(layer_scale)*R; b_out'=b_out*diag(layer_scale)*R",
        'bare_norm_example_count': checked,
        'affine_and_residual_example_count': fold_cases,
        'dimensions': [4, 16, 64, 256, 1024],
        'epsilon_values': [1e-6, 1e-5],
        'maximum_generated_float64_discrepancies': errors,
        'arbitrary_orthogonal_LayerNorm_counterexample_max_abs': counterexample,
        'synthetic_impulse_only_max_abs_before_after': [1024.0, 32.0],
        'real_activation_or_quantization_improvement_measured': False,
        'actual_VGGT_or_model_execution': False,
        'checkpoint_or_captured_array_access': False,
        'Torch_ONNX_SDK_GPU_or_board_execution': False,
        'actual_FP32_or_FP16_equivalence_verified': False,
        'compiler_placement_performance_trajectory_or_realtime_accepted': False,
        'limits': [
            'Algebra is over real arithmetic; finite precision and folded bias rounding change.',
            'Proof is for LayerNorm with original mean subtraction, not replacing it with RMSNorm.',
            'QKV internals, q/k normalization, RoPE, softmax and Erf remain in original coordinates.',
            'Every residual-stream producer/consumer must use the same R or explicit inverse.',
            'Retained features, final normalization, target-token export and heads require separate boundary handling.',
            'The dense MAC count is unchanged; benefit would require better measured low-precision behavior.',
            'A synthetic impulse is not evidence that real VGGT outliers or quantization errors improve.',
        ],
    }


if __name__ == '__main__':
    report = run()
    destination = Path(__file__).with_name('mean-preserving-basis-algebra.json')
    assert not destination.exists(), 'Keep prior evidence; choose a fresh directory for reruns'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(destination),
                      'max_discrepancies': report['maximum_generated_float64_discrepancies']}))
