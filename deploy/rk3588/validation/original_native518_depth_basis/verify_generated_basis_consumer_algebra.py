#!/usr/bin/env python3
"""Generated small-matrix proof checks only; no checkpoint, NN or SDK loading.

For row vectors z and T=diag(R,R), u=zT. R is the existing symmetric,
mean-preserving orthogonal basis. The original DPT consumer is
    (LN0(z) diag(gamma) + beta) W.T + bias.
Its equivalent consumer in u coordinates is
    LN0(u) (T.T diag(gamma) W.T) + beta W.T + bias.
LN0 keeps the original epsilon. This removes full feature inverse transforms
at this consumer, not any original layer or token computation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def norm0(x, eps):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)


def run():
    # This 64-channel fixture is an algebra check, not reduced-resolution or
    # reduced-channel execution of the original model.
    r = np.ones((1, 1), dtype=np.float64)
    base = (np.ones((4, 4)) - 2 * np.eye(4)) / 2
    for _ in range(3): r = np.kron(r, base)
    n = r.shape[0]
    t = np.zeros((2 * n, 2 * n))
    t[:n, :n] = r; t[n:, n:] = r
    assert np.array_equal(t, t.T)
    assert np.array_equal(t @ t.T, np.eye(2 * n))
    assert np.array_equal(t @ np.ones(2 * n), np.ones(2 * n))
    rng = np.random.default_rng(130)
    gamma = rng.normal(size=2 * n)
    beta = rng.normal(size=2 * n)
    weight = rng.normal(size=(23, 2 * n))
    bias = rng.normal(size=23)
    # Camera selection is inherited from the original consumer, after every
    # aggregator token was computed. No aggregator token is omitted here.
    z = rng.normal(size=(2, 7, 2 * n))
    u = z @ t
    selected_inverse = u[:, 0, :] @ t.T
    full_inverse_selected = (u @ t.T)[:, 0, :]
    selection_error = float(np.max(np.abs(selected_inverse - full_inverse_selected)))
    assert selection_error < 1e-12, selection_error
    rows = []
    for eps in (1e-6, 1e-5):
        for label, x in [('random', z), ('constant', np.ones_like(z)), ('near_constant', 2 + z * 1e-7)]:
            rotated = x @ t
            original = (norm0(x, eps) * gamma + beta) @ weight.T + bias
            folded_weight = (weight * gamma) @ t
            folded_bias = beta @ weight.T + bias
            candidate = norm0(rotated, eps) @ folded_weight.T + folded_bias
            error = float(np.max(np.abs(candidate - original)))
            assert error < 1e-8, (eps, label, error)
            rows.append(dict(epsilon=eps, fixture=label, max_abs_error=error))
    # Negative control: original camera AdaLN's channel-wise modulation is
    # not invariant under a general basis rotation. Keep its trunk canonical.
    scale = rng.normal(size=2 * n)
    a = (z * scale) @ t
    b = (z @ t) * scale
    modulation_error = float(np.max(np.abs(a - b)))
    assert modulation_error > 1e-3
    return dict(schema='generated_basis_consumer_algebra_v130_design_check',
        status='generated_algebra_passed_original_graph_not_exported_or_accepted',
        original_model_values_or_saved_captures_loaded=False, NN_or_SDK_executed=False,
        generated_fixture_dimensions=dict(frame_channels=n, combined_channels=2*n, tokens=7, frames=2),
        generated_fixture_is_not_original_model_execution=True,
        orthogonality_symmetry_and_mean_preservation_exact=True,
        DPT_projection_checks=rows,
        camera_original_token_selection_commutes_with_inverse=True,
        camera_selection_float64_max_abs_error=selection_error,
        camera_selection_float64_byte_equal=bool(np.array_equal(selected_inverse, full_inverse_selected)),
        selection_check_note='The initial byte-equality assertion failed for the different matrix batch shapes; the selected numerical kernels were not inspected. This generated check establishes bounded error, not bitwise execution equivalence.',
        camera_channelwise_modulation_noncommutation_max_abs=modulation_error,
        decisions=['Carry the existing basis into DPT LayerNorm plus projection; fold its affine parameters and inverse basis into the original projection weights and bias.',
                   'Restore only the original camera consumer tokens before its token_norm; keep adaptive modulation and the camera trunk in canonical coordinates.'],
        required_next_evidence=['Board transformation and provenance proof for all original consumer parameters.',
                                'Same-input original FP32 consumer outputs, then actual compiled placement.',
                                'Complete model precision, latency and trajectory validation.'],
        full_model_speed_quality_or_realtime_accepted=False)


if __name__ == '__main__':
    result = run()
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target = Path(__file__).with_name('generated-basis-consumer-algebra.json')
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], max_abs_error=max(x['max_abs_error'] for x in result['DPT_projection_checks']))))
