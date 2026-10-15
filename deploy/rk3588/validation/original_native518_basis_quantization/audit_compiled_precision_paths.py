#!/usr/bin/env python3
"""Read compiler tables only; no model payloads, SDK or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path):
    operators, memories = [], []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        text = re.sub(r'^\w RKNN: \[[^]]+\] ', '', line)
        match = re.match(r'^(\d+)\s+(\w+)\s+(FLOAT16|FLOAT|INT8|INT16|INT32|INT64|UINT8)\s+(NPU|CPU|GPU)\s+', text)
        if match:
            parts = re.split(r'\s{2,}', text.strip())
            assert len(parts) in (8, 9) and ':' in parts[-1], (path, line_no, parts)
            operators.append(dict(id=int(match[1]), op=match[2], dtype=match[3], target=match[4],
                                  name=parts[-1], source_line=line_no))
        memory = re.match(r'^(\d+)\s+(\w+)\s+(\S+)\s+(FLOAT16|FLOAT|INT8|INT16|INT32|INT64|UINT8)\s+.*?\|\s+(0x\w+)\s+(0x\w+)\s+(0x\w+)\s*$', text)
        if memory:
            memories.append(dict(user_id=int(memory[1]), op=memory[2], tensor=memory[3], dtype=memory[4],
                                 start=int(memory[5], 16), end=int(memory[6], 16), size=int(memory[7], 16), source_line=line_no))
    assert operators and len({r['id'] for r in operators}) == len(operators)
    return operators, memories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = dict(schema='v128_actual_compiler_precision_path_audit', script_sha256=sha(Path(__file__)),
                  host_NN_or_SDK_executed=False, compiled_metadata_only=True, variants={},
                  inference_finiteness_or_numerical_accuracy_established=False,
                  calibration_policy_is_not_actual_precision_proof=True)
    for variant in 'AFB':
        log = a.root / ('vggt-v83-native518basis128step2' + variant.lower() + 'r2-r1.log')
        ops, mem = parse(log)
        residuals = [r for r in ops if r['op'] == 'Add' and
                     re.search(r':(?:frame0/batch[01]|global0)/(?:residual1|output)$', r['name'])]
        assert len(residuals) == 6 and all(r['dtype'] == 'FLOAT16' for r in residuals)
        bridge = [r for r in mem if r['tensor'] == 'frame_output']
        alias = [(x, y) for x in bridge for y in bridge
                 if x['op'] == 'OutputOperator' and y['op'] == 'Reshape' and
                 (x['start'], x['end'], x['size']) == (y['start'], y['end'], y['size'])]
        if variant in 'AF':
            assert len(alias) == 1 and all(r['dtype'] == 'INT8' for r in alias[0])
        selected_ops = [r for r in ops if any(s in r['name'] for s in
                       ('frame_output', '__global_input', '__frame_basis', '__v126_restore_frame_output'))]
        selected_mem = [r for r in mem if any(s in r['tensor'] for s in
                       ('frame_output', '__global_input', '__frame_basis'))]
        report['variants'][variant] = dict(log=dict(file=log.name, bytes=log.stat().st_size, sha256=sha(log)),
            compiled_operator_count=len(ops), six_FP16_residual_Adds=residuals, bridge_operators=selected_ops,
            bridge_memory_rows=selected_mem, shared_INT8_public_exit_and_global_input_buffer=bool(alias),
            all_operator_types_and_targets=ops,
            interpretation=('The global residual path traverses the same INT8 buffer exposed as frame_output; '
                            'FP16 residual Add operators do not protect the complete path.' if alias else
                            'The basis fork has a different compiled precision boundary; '
                            'comparing its errors with A/F cannot isolate the basis transform alone.'))
    report['decision'] = 'Hold heldout candidate acceptance until the intended precision path is established.'
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=a.output.name, sha256=sha(a.output),
                         A_F_bridge_INT8=True, heldout_acceptance=False)))


if __name__ == '__main__':
    main()
