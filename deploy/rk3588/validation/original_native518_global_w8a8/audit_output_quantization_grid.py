#!/usr/bin/env python3
"""Saved-data nearest output-grid bound, not a quantizer or neural simulation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


root = Path(sys.argv[1])
r = json.loads((root / 'abba-r1/result.json').read_text())
capture = json.loads((root / 'capture-r2/result.json').read_text())
rows = []
for call in r['calls'][2:4]:
    assert call['route'] == 'B'
    record = next(v for v in capture['records'] if v['id'] == call['input_id'])
    ref_path, out_path = root / 'capture-r2' / record['reference']['file'], root / 'abba-r1' / call['output']['file']
    assert sha(ref_path) == record['reference']['sha256'] and sha(out_path) == call['output']['sha256']
    ref, actual = np.load(ref_path, allow_pickle=False), np.load(out_path, allow_pickle=False)
    descriptor, = call['resources']['output_descriptors']
    assert descriptor['type'] == 2 and descriptor['quantization_type'] == 2
    levels = ((np.arange(-128, 128, dtype=np.float32) - descriptor['zero_point']) * np.float32(descriptor['scale'])).astype(np.float64)
    assert np.all(np.diff(levels) > 0)

    def closest(v):
        high = np.searchsorted(levels, v.astype(np.float64)).clip(0, 255)
        low = (high - 1).clip(0, 255)
        return np.where(np.abs(levels[low] - v) <= np.abs(levels[high] - v), levels[low], levels[high])

    reference = ref.astype(np.float64)
    error_floor = closest(ref) - reference
    actual_grid_error = actual.astype(np.float64) - closest(actual)
    rows.append(dict(id=call['input_id'], reference_sha256=sha(ref_path), output_sha256=sha(out_path),
                     output_scale=descriptor['scale'], output_zero_point=descriptor['zero_point'],
                     original_reference_outside_grid_range=int(np.count_nonzero((ref < levels[0]) | (ref > levels[-1]))),
                     nearest_output_grid_relative_l2=float(np.linalg.norm(error_floor) / np.linalg.norm(reference)),
                     actual_output_relative_l2=float(np.linalg.norm(actual.astype(np.float64) - reference) / np.linalg.norm(reference)),
                     actual_output_max_distance_to_grid=float(np.max(np.abs(actual_grid_error))),
                     actual_output_exactly_on_float32_dequantized_grid=bool(np.all(actual_grid_error == 0))))
report = dict(schema='v115_saved_output_grid_bound', auditor_sha256=sha(Path(__file__)),
              runtime_sha256=sha(root / 'abba-r1/result.json'), capture_sha256=sha(root / 'capture-r2/result.json'),
              scope='Pure saved-array nearest-point calculation on the 256 queried signed-affine INT8 output levels. No model, SDK, calibration or device execution.',
              interpretation='Closest output levels bound error for this fixed output representation. This does not identify internal error sources or predict a float-boundary model result.',
              cases=rows)
(root / 'output-grid-bound.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(rows))
