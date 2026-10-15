#!/usr/bin/env python3
"""Audit saved compiler tables for the original B attention precision policy."""
import hashlib
import importlib.util
import json
from pathlib import Path
from collections import Counter

W = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = W / 'audit_compiled_precision_paths.py'
    assert sha(parser) == 'edc62eb8169e629a8c4eacc039e3f7428828553b8eebab35786ccb8811a57a7d'
    spec = importlib.util.spec_from_file_location('_v128_compiler_table_parser', parser)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    receipt = W / 'B-attention-step2-r1/result.json'
    assert sha(receipt) == '3c952601fa1cad925c37c1dd4a9a5edb169dea2ab6356ab5abd8c91fa5004317'
    compiled = json.loads(receipt.read_text())
    assert compiled['script_sha256'] == '95e2e56c2c090e5f5de389466e704aa39353972044c96c372b320ec8ff47c8dc'
    assert compiled['selector_count'] == 80 and compiled['sdk_released'] and not compiled['cleanup_errors']
    log = W / 'vggt-v83-native518basis128battentionr1-r1.log'
    operators, memory = module.parse(log)
    branches = []
    for prefix in ('frame0/batch0/', 'frame0/batch1/', 'global0/'):
        rows = [r for r in operators if r['name'].split(':', 1)[1].startswith(prefix)]
        attention = [r for r in rows if any(k in r['name'] for k in
                     ('/norm1#', '/attn.', '/qkv_', '/query_rope', '/key_rope', '/attention_'))]
        assert len(attention) == 15 and all(r['dtype'] == 'FLOAT16' and r['target'] == 'NPU' for r in attention)
        assert len([r for r in attention if r['op'] == 'exSDPAttention']) == 1
        projection = [r for r in attention if '/attn.proj.matmul' in r['name']]
        assert len(projection) == 1 and projection[0]['op'] == 'ConvAdd'
        mlp = [r for r in rows if '/norm2#' in r['name'] or '/mlp.' in r['name'] and r['op'] != 'exDataConvert']
        assert mlp and all(r['dtype'] == 'INT8' and r['target'] == 'NPU' for r in mlp)
        assert len([r for r in mlp if r['op'] == 'ConvExGelu']) == 1
        exit_rows = [r for r in rows if r['name'] == 'Add:' + prefix + 'output']
        assert len(exit_rows) == 1 and exit_rows[0]['dtype'] == 'FLOAT16'
        branches.append(dict(prefix=prefix, attention_rows=attention, MLP_rows=mlp,
                             projection_residual_fusion=projection[0], final_residual_Add=exit_rows[0]))
    bridge = [r for r in operators if '__global_input' in r['name']]
    assert len(bridge) == 2 and all(r['dtype'] == 'FLOAT16' and r['target'] == 'NPU' for r in bridge)
    basis = [r for r in operators if r['name'].startswith('Conv:__v126_')]
    assert len(basis) == 3 and all(r['dtype'] == 'FLOAT16' for r in basis)
    cpu = [r for r in operators if r['target'] == 'CPU']
    assert any(r['name'] == 'Transpose:frame_output_mm_tp' for r in cpu)
    report = dict(schema='v128_B_attention_actual_compiler_metadata_audit', source_sha256=sha(Path(__file__)),
                  parser_sha256=sha(parser), compile_receipt_sha256=sha(receipt),
                  log=dict(file=log.name, sha256=sha(log)), operator_count=len(operators),
                  dtype_target_counts={k + '/' + v: n for (k, v), n in Counter((r['dtype'], r['target']) for r in operators).items()},
                  branches=branches, frame_to_global_bridge=bridge, basis_boundaries=basis, CPU_rows=cpu,
                  all_operators=operators, compiled_memory_rows=memory,
                  host_NN_SDK_or_board_execution=False, candidate_accepted=False,
                  runtime_finiteness_accuracy_speed_or_8GB_established=False,
                  conclusions=['Compiled attention rows are FLOAT16; three projection/residual additions are fused ConvAdd.',
                               'MLP normalization, FC1/GELU and FC2 rows remain INT8; final residual Adds are FLOAT16.',
                               'The frame-to-global view rows remain FLOAT16; canonical frame transpose remains CPU.',
                               'Operator dtype labels and relative buffer offsets do not prove internal accumulator precision or numerical quality.'])
    out = W / 'B-attention-compiled-placement.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(file=out.name, sha256=sha(out), operators=len(operators), counts=report['dtype_target_counts'])))


if __name__ == '__main__':
    main()
