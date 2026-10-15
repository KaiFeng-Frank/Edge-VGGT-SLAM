#!/usr/bin/env python3
"""Compare already-saved calibration pair0 arrays; no inference or heldout use."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PINS = {
    'calibration-profile-r2/result.json': '4013651f463a39475560b7c2e0544b3ce0fb813ece5a73e343cae9299e84d91b',
    'fp32-batch0-r1/result.json': '70caf6de746831cf7950fb4971359a1341d2dd4d845f3f3eb5a1da0e1b32072c',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder, row):
    p = folder / row['file']
    assert p.resolve().is_relative_to(folder.resolve())
    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    v = np.load(p, allow_pickle=False)
    assert v.dtype == np.float32 and list(v.shape) == row['shape'] and v.flags.c_contiguous
    assert hashlib.sha256(v.tobytes()).hexdigest() == row['raw_sha256']
    return v


def compare(actual, reference):
    assert np.isfinite(reference).all() and actual.shape == reference.shape
    finite = bool(np.isfinite(actual).all())
    result = dict(finite=finite, nan_count=int(np.isnan(actual).sum()),
                  positive_inf_count=int(np.isposinf(actual).sum()), negative_inf_count=int(np.isneginf(actual).sum()))
    if finite:
        d = actual.astype(np.float64) - reference.astype(np.float64)
        den = float(np.linalg.norm(reference.astype(np.float64)))
        assert den > 0
        rel, maximum = float(np.linalg.norm(d) / den), float(np.abs(d).max())
        result.update(relative_l2=rel, max_abs=maximum,
                      fixed_FP32_gate_passed=rel <= 1e-5 and maximum <= 1e-3)
    else:
        result.update(relative_l2=None, max_abs=None, fixed_FP32_gate_passed=False)
    return result


def main():
    for name, digest in PINS.items():
        assert sha(ROOT / name) == digest
    runtime = json.loads((ROOT / 'calibration-profile-r2/result.json').read_text())
    fp32 = json.loads((ROOT / 'fp32-batch0-r1/result.json').read_text())
    assert runtime['input_id'] == 'pair0' and runtime['source_pair_indices'] == [0, 8]
    assert runtime['input'] == fp32['records'][0]['input']
    assert runtime['checkpoint'] == fp32['checkpoint'] and fp32['FP32_gate_passed']
    refs = {r['variant']: r for r in fp32['checks'] if r['input_id'] == 'pair0'}
    assert set(refs) == {'A', 'F', 'B'}
    assert all(r['split'] == 'calibration' and r['source_pair_indices'] == [0, 8] for r in refs.values())
    baseline = {r['name']: load(ROOT / 'fp32-batch0-r1', r) for r in refs['A']['outputs']}
    rows = []
    for call in runtime['calls']:
        if call['init_flags'] != 0:
            continue
        variant = call['route'][-1] if call['route'].startswith('mixed') else 'A'
        own = {r['name']: load(ROOT / 'fp32-batch0-r1', r) for r in refs[variant]['outputs']}
        for r in call['outputs']:
            actual = load(ROOT / 'calibration-profile-r2', r)
            rows.append(dict(route=call['route'], endpoint=r['name'], output_sha256=r['sha256'], FP32_comparison_graph=variant,
                             vs_original_A=compare(actual, baseline[r['name']]),
                             vs_comparison_graph_FP32=compare(actual, own[r['name']])))
    assert len(rows) == 10
    report = dict(schema='v128_calibration_pair0_saved_accuracy_diagnostic', source_sha256=sha(Path(__file__)),
                  receipt_pins=PINS, input_id='pair0', source_pair_indices=[0, 8], split='calibration',
                  host_NN_SDK_or_board_executed=False, heldouts_accessed=False, candidate_accepted=False,
                  relative_l2_max=1e-5, max_abs_max=1e-3, rows=rows,
                  limitations=['Calibration pair only; not heldout or trajectory quality.',
                               'A/F/B actual internal precision boundaries differ; this is not an isolated basis comparison.',
                               'The upstream captured input contains original NPU DINO error; not end-to-end FP32.',
                               'Joint MLP is compared with original A FP32, not a separately loaded joint-MLP reference.'])
    out = ROOT / 'calibration-pair0-saved-accuracy-r2.json'
    with out.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(file=out.name, sha256=sha(out), rows=rows)))


if __name__ == '__main__':
    main()
