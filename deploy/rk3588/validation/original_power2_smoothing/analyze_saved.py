#!/usr/bin/env python3
"""Inspect saved tensors and SDK reports only; never load a model or runtime."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.evidence
    normal = json.loads((root/'normal-r1/result.json').read_text())
    profile = json.loads((root/'profile-r1/result.json').read_text())
    checked = json.loads((root/'check-r1/result.json').read_text())
    rows = []
    for case in checked['cases']:
        for item, desc in zip([case['saved_input']]+case['baseline_outputs'], normal['queries'][1]['descriptors']):
            path = root/'check-r1'/item['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            x = np.load(path, allow_pickle=False).astype(np.float64)
            assert desc['type'] == 2 and np.isfinite(x).all()
            q = np.rint(x/desc['scale'])+desc['zero_point']
            clipped = np.clip(q, -128, 127)
            y = (clipped-desc['zero_point'])*desc['scale']
            rows.append(dict(input=case['tag'], tensor=desc['name'], scale=desc['scale'],
                zero_point=desc['zero_point'], elements=x.size,
                clipped_elements=int(np.count_nonzero(q != clipped)),
                relative_l2=float(np.linalg.norm(y-x)/np.linalg.norm(x)),
                max_abs=float(np.max(np.abs(y-x)))))
    profiles = []
    for call in profile['calls']:
        text = call['profile']['raw_text']
        assert hashlib.sha256(text.encode()).hexdigest() == call['profile']['raw_sha256']
        ops = []
        for line in text.splitlines():
            m = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.+?)\s+\d+/\d+/\d+\s+(\d+)\s+(.+)$', line)
            if m:
                ops.append(dict(id=int(m[1]), op=m[2], dtype=m[3], target=m[4],
                    microseconds=int(m[6]), name=m[7].split()[-1]))
        total = int(re.search(r'Total Operator Elapsed Per Frame Time\(us\): (\d+)', text)[1])
        assert sum(x['microseconds'] for x in ops) == total
        counts = collections.Counter((x['dtype'], x['target']) for x in ops)
        linear = [x for x in ops if x['op'].startswith('Conv') and 'matmul' in x['name']]
        assert len(linear) == 8
        profiles.append(dict(input=call['input'], role=call['role'], operator_count=len(ops),
            total_microseconds=total,
            placement_counts={f'{a}/{b}':v for (a,b),v in counts.items()},
            original_linear_ops=linear, profile_sha256=call['profile']['raw_sha256']))
    report = dict(schema='v78_saved_array_quantizer_and_profile_diagnostic',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        host_model_execution=False, sdk_execution=False,
        input_and_output_descriptors_B_C_identical=normal['queries'][1]['descriptors'] == normal['queries'][2]['descriptors'],
        isolated_roundtrip_convention='float64 rint ties-to-even, affine clamp INT8, for saved tensors only; not claimed SDK exact rounding or inference/trajectory error, not a propagated-error bound',
        isolated_roundtrips=rows, profiles=profiles)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(isolated_roundtrips=len(rows), profiles=len(profiles), all_profile_sums_verified=True)))


if __name__ == '__main__':
    main()
