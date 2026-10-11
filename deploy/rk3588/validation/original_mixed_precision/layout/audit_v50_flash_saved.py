#!/usr/bin/env python3
"""Read saved original-pair Flash flag off/on compile/profile evidence only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from v50_original_mlp_scope_contracts import PAIR_EXPORT_SHA, PAIR_GRAPH_SHA, require, sha

COMPILER_SHA = '709544077448718fe866e8e288810ac43110fb2afc98bd3a39cde5d6d3a965b0'
RUNNER_SHA = '10b553b7c2a005310c6b6d00a5f6442668e6597b9f1a0d7ee119f88fd349e9a6'
PARENT_SHA = '33fc5deb749c8b3ceaa47271e389f5fbbfc2dba6db93aa19887b2fc41e0e7882'
CAPTURE_SHA = 'bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc'
PROFILE_PARSER_SHA = '061b15948907ab1b1317c1c25315e7683f26b191b04345558cf670eff20b111e'
ROLES = ('ordinary', 'flash')
CONFIG = {'target_platform': 'rk3588', 'float_dtype': 'float16', 'optimization_level': 3,
          'mean_values': None, 'std_values': None, 'quant_img_RGB2BGR': False,
          'model_pruning': False, 'remove_reshape': False, 'quantize_weight': False}
ROW = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.+?)\s+'
                 r'(\d+/\d+/\d+)\s+(\d+)\s+(.+?)\s*$')
PREFIX = re.compile(r'^D RKNN:\s*\[[^\]]+\]\s*')


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(re.fullmatch(r'[0-9a-f]{64}', expected) is not None and sha(path) == expected,
            'Saved artifact SHA mismatch: ' + str(path))
    pins[str(path)] = expected
    return path


def identity(row, offset=0):
    return (row['id'] + offset, row['op'], row['data_type_token'], row['target'], row['name'],
            ' '.join(row['input_and_output_shapes_text'].split()))


def board_float_sum(values):
    """Reproduce board CPython3.11 sum; newer host versions compensate floats."""
    result = 0.0
    for value in values:
        result += value
    return result


def compile_tables(raw):
    text = re.sub(r'\x1b\[[0-9;]*m', '', raw)
    tables, current = [], None
    flags = []
    for line in text.splitlines():
        if 'enable_flash_attention = ' in line:
            flags.append(int(re.search(r'enable_flash_attention = (\d+)', line)[1]))
        if 'ID   OpType' in line and 'Target' in line:
            current = []
            tables.append(current)
            continue
        if 'ID  User' in line:
            current = None
            continue
        if current is None:
            continue
        row = PREFIX.sub('', line)
        if row == line or not re.match(r'^\s*\d+\s', row):
            continue
        m = ROW.fullmatch(row)
        require(m is not None, 'Unparsed numeric final compiler operator: ' + line)
        current.append({'id': int(m[1]), 'op': m[2], 'data_type_token': m[3], 'target': m[4],
            'input_and_output_shapes_text': m[5], 'DDR_NPU_total_cycles': [int(v) for v in m[6].split('/')],
            'RW_KB': int(m[7]), 'name': m[8]})
    require(len(tables) == 2 and flags == [0, 1], 'Exactly two off/on C compiler tables/flags required')
    for table in tables:
        require([v['id'] for v in table] == list(range(54)), 'Incomplete final compiler table')
        require(table[0]['op'] == 'InputOperator' and table[-1]['op'] == 'OutputOperator', 'IO boundaries changed')
    return tables, flags


def check_compile(directory, expected, log, log_sha, pins):
    file = pin(directory / 'compile.json', expected, pins)
    row = json.loads(file.read_text())
    require(row['schema'] == 'v50_original_pair_flash_config_board_compile'
        and row['status'] == 'both_compiled_not_runtime_validated' and row['script_sha256'] == COMPILER_SHA
        and row['pair_graph_sha256'] == PAIR_GRAPH_SHA and row['pair_export_sha256'] == PAIR_EXPORT_SHA
        and row['machine'] == 'aarch64' and 'rockchip,rk3588' in row['board']
        and row['actual_calibration_performed'] is False and row['runtime_API_called'] is False
        and row['host_inference_performed'] is False and row['candidate_accepted'] is False
        and row['source_and_inputs_unchanged'] is True and row['environment_restored'] is True
        and row['cwd_restored'] is True and row['cleanup_errors'] == []
        and row['source_and_input_hashes'] == row['source_and_input_hashes_after']
        and row['new_heldout_ORT_status'] == 'numerical_rejected', 'Clean unchanged original compile evidence required')
    require(row['versions'] == {'rknn-toolkit2': '2.3.2', 'torch': '2.2.0', 'numpy': '1.26.4',
        'onnx': '1.16.1', 'onnxruntime': '1.16.0', 'protobuf': '3.20.3'}, 'Isolated toolkit version set changed')
    models = {}
    for role, flag in zip(ROLES, (False, True)):
        item = row['variants'][role]
        require(item['status'] == 'compiled_not_runtime_validated' and item['released'] is True
            and item['release_code'] is None and item['configuration'] == dict(CONFIG, enable_flash_attention=flag)
            and item['build_options'] == {'do_quantization': False, 'dataset': None, 'auto_hybrid': False}
            and [v['name'] for v in item['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
            and all(v['status'] == 'completed' and type(v['code']) is int and v['code'] == 0 for v in item['operations']),
            'Flag-only completed nonquantized compile operations required')
        output = item['output']
        require(output['file'] == role + '/pair.rknn', 'Unexpected model output file')
        model = pin(directory / output['file'], output['sha256'], pins)
        require(model.stat().st_size == output['bytes'], 'Compiled artifact byte count mismatch')
        models[role] = model
    require(models[ROLES[0]].stat().st_size == models[ROLES[1]].stat().st_size, 'Component artifact extents differ')
    differences = 0
    with models[ROLES[0]].open('rb') as first, models[ROLES[1]].open('rb') as second:
        while left := first.read(1 << 20):
            right = second.read(len(left))
            differences += sum(a != b for a, b in zip(left, right))
    compiler_log = pin(log, log_sha, pins)
    tables, flags = compile_tables(compiler_log.read_text())
    return row, tables, {'model_bytes_each': models[ROLES[0]].stat().st_size,
        'different_binary_byte_count': differences, 'same_final_operator_records': tables[0] == tables[1],
        'same_final_operator_identity_shape_dtype_placement': [identity(v) for v in tables[0]] == [identity(v) for v in tables[1]],
        'C_compiler_flag_values_observed': flags,
        'internal_target_counts_per_variant': {role: dict(Counter(v['target'] for v in table[1:-1]))
                                                for role, table in zip(ROLES, tables)},
        'internal_dtype_tokens_per_variant': {role: dict(Counter(v['data_type_token'] for v in table[1:-1]))
                                               for role, table in zip(ROLES, tables)},
        'byte_differences_prove_changed_math_or_flash_kernel': False,
        'entire_compiled_binary_noop_proven': False}


def check_run(directory, expected, compiled, compile_sha, normal, pins):
    file = pin(directory / 'run.json', expected, pins)
    row = json.loads(file.read_text())
    require(row['schema'] == 'v50_original_pair_flash_SDK_AB' and row['status'] == 'completed_diagnostic_not_adopted'
        and row['script_sha256'] == RUNNER_SHA and row['parent_source_sha256'] == PARENT_SHA
        and row['compile_sha256'] == compile_sha and row['original_graph_sha256'] == PAIR_GRAPH_SHA
        and row['capture_sha256'] == CAPTURE_SHA and row['candidate_accepted'] is False
        and row['strict_gate'] == {'rtol': 1e-4, 'atol': 1e-5}
        and row['normal_benchmark'] is normal and row['profile_separate_fresh_contexts'] is (not normal)
        and row['successful_SDK_calls'] == 16 and row['requested_sdk_call_count'] == 16
        and len(row['calls']) == len(row['sdk_calls']) == 16 and row['unmeasured_warmup_calls'] == 0
        and row['cleanup_errors'] == [] and row['affinity_restored'] is True and row['environment_restored'] is True
        and row['source_and_inputs_unchanged'] is True
        and row['source_and_input_hashes'] == row['source_and_input_hashes_after'],
        'Completed source-pinned clean sixteen-call Flash SDK evidence required')
    require(set(row['context_records']) == set(ROLES), 'Two explicit contexts required')
    for role in ROLES:
        context = row['context_records'][role]
        require(context['init_flags'] == (0 if normal else 8) and context['context_destroyed'] is True
            and context['tensor_attributes_validated'] is True and context['core_mask'] == 7
            and context['driver_version'] == '0.7.2' and context['sdk_version'].startswith('2.3.2')
            and context['integer_logical_IO_observed'] is False
            and row['compiled_artifacts'][role] == compiled['variants'][role]['output'],
            'Floating typed IO/contexts/artifact provenance mismatch')
    schedule = []
    for sample in range(2):
        for iteration in range(4):
            order = ROLES if (sample + iteration) % 2 == 0 else ROLES[::-1]
            schedule.extend(('evaluation_' + f'{sample:02d}', iteration, role) for role in order)
    require([(v['sample_id'], v['round_index'], v['role']) for v in row['calls']] == schedule,
            'Actual off/on sixteen-call AB/BA schedule changed')
    for index, call in enumerate(row['calls']):
        sdk = row['sdk_calls'][index]
        require(call['sdk_call_index'] == sdk['sdk_call_index'] == index and sdk['status'] == 'completed'
            and sdk['role'] == call['role'] and sdk['sample_id'] == call['sample_id']
            and sdk['sample_call_index'] == call['round_index'] and sdk['synchronous'] is True
            and sdk['input_pass_through'] == 0 and sdk['input_api_type_enum'] == 0 and sdk['input_format'] == 3
            and sdk['want_float'] == 1 and sdk['output_buffer_released'] is True
            and sdk['output_owned_copy_before_release'] is True
            and sdk['owned_output_pointer'] != sdk['SDK_output_pointer']
            and sdk['output_bytes'] == 3579904 and sdk['runtime_frame_id'] == sdk['output_frame_id']
            and sdk['input_raw_sha256_before'] == sdk['input_raw_sha256_after'] == call['input_raw_sha256'],
            'Actual typed synchronous IO ownership/release/input identity mismatch')
        require(call['original_FP32_comparison']['rtol'] == 1e-4 and
                call['original_FP32_comparison']['atol'] == 1e-5, 'Fixed original FP32 threshold changed')
        output = call['output']
        path = pin(directory / output['file'], output['sha256'], pins)
        require(path.stat().st_size == output['bytes'], 'Saved SDK output file extent changed')
    for role in ROLES:
        repeat = [v for v in row['calls'] if v['role'] == role and v['round_index'] > 0]
        require(len(repeat) == 6 and row['summaries'][role]['mean_denominator'] == 6 and
            row['summaries'][role]['subsequent_IO_mean_seconds'] == board_float_sum(v['io_total_seconds'] for v in repeat) / 6,
            'Repeat denominator/mean mismatch')
    return row


def profile_tables(directory, run, parser_path, compiled_tables, pins):
    parser_path = pin(parser_path, PROFILE_PARSER_SHA, pins)
    spec = importlib.util.spec_from_file_location('_v50_reused_v49_profile_parser', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    tables = {role: [] for role in ROLES}
    for call in run['calls']:
        detail = call['separate_profile_detail']
        file = pin(directory / detail['file'], detail['saved_text_sha256'], pins)
        require(sha(file) == detail['raw_sha256'] and file.stat().st_size == detail['bytes'],
                'Actual raw SDK operator detail bytes/hash mismatch')
        rows, total = parser.parse(file.read_text())
        role_index = ROLES.index(call['role'])
        require([identity(v) for v in rows] == [identity(v, 1) for v in compiled_tables[role_index]],
                'Actual profile identity/shape/dtype/placement differs from compiled table')
        tables[call['role']].append({'sample_id': call['sample_id'], 'sample_call_index': call['round_index'],
            'rows': rows, 'exact_table_total_microseconds': total, 'file': file.name, 'sha256': sha(file)})
    summaries = {role: parser.summarize_tables(values) for role, values in tables.items()}
    same = [identity(v) for v in tables['ordinary'][0]['rows']] == [identity(v) for v in tables['flash'][0]['rows']]
    return tables, summaries, same


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('compile-dir', 'compiler-log', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('compile-sha256', 'compiler-log-sha256'):
        parser.add_argument('--' + key, required=True)
    for mode in ('normal', 'profile'):
        parser.add_argument('--' + mode + '-dir', type=Path)
        parser.add_argument('--' + mode + '-sha256')
    parser.add_argument('--profile-parser', type=Path)
    args = parser.parse_args()
    for mode in ('normal', 'profile'):
        require((getattr(args, mode + '_dir') is None) == (getattr(args, mode + '_sha256') is None),
                'Saved mode directory and independent SHA must be supplied together')
    pins = {}
    compiled, compiler_tables, compiler = check_compile(args.compile_dir, args.compile_sha256,
        args.compiler_log, args.compiler_log_sha256, pins)
    report = {'schema': 'v50_original_pair_flash_saved_compile_profile_independent_audit',
        'status': 'saved_compile_audited_runtime_pending', 'script_sha256': sha(__file__),
        'actual_compile_receipt_sha256': args.compile_sha256, 'actual_compiler_log_sha256': args.compiler_log_sha256,
        'compile_summary': compiler, 'compiled_final_operator_tables': dict(zip(ROLES, compiler_tables)),
        'actual_model_ORT_SDK_compiler_calibration_executed_by_auditor': False, 'board_access_by_auditor': False,
        'candidate_accepted': False, 'whole_binary_noop_or_whole_model_quality_or_real_time_proven': False,
        'limitations': ['Only saved original-pair component evidence is audited; no actual jobs are run by this auditor',
            'Internal operator IO dtype/placement/identity does not expose every arithmetic instruction or low-level kernel',
            'Binary size/SHA/byte differences alone cannot prove changed numerical computation',
            'Compiler logger appends second build to first verbose file; final operator tables come from full stdout',
            'Full stdout has asynchronous interleaving in a weights-table record; this audit parses final operator tables, not a reconstructed full weight allocation table',
            'Normal SDK IO and profile operator clocks use separate processes/contexts and denominators']}
    normal = None
    if args.normal_dir is not None:
        normal = check_run(args.normal_dir, args.normal_sha256, compiled, args.compile_sha256, True, pins)
        report['actual_normal_receipt_sha256'] = args.normal_sha256
        report['normal_SDK_IO_separate_from_profile'] = normal['summaries']
    if args.profile_dir is not None:
        require(args.profile_parser is not None, 'SHA-pinned previously validated profile parser required')
        profile = check_run(args.profile_dir, args.profile_sha256, compiled, args.compile_sha256, False, pins)
        tables, summaries, same = profile_tables(args.profile_dir, profile, args.profile_parser, compiler_tables, pins)
        report.update(status='saved_compile_profile_audited_diagnostic_not_adopted',
            actual_profile_receipt_sha256=args.profile_sha256, profile_operator_identity_same_off_on=same,
            profile_tables=tables, profile_summaries=summaries,
            profile_FP32_gate_summaries=profile['summaries'], profile_operator_tables_checked=16)
        if normal is not None:
            report['profile_output_raw_SHA_matches_same_input_normal_outputs'] = all(
                p['output']['raw_sha256'] == n['output']['raw_sha256']
                for p, n in zip(profile['calls'], normal['calls']))
            report['profile_contexts_separate_from_normal'] = True
    require(all(sha(path) == expected for path, expected in pins.items()), 'Saved artifacts changed during audit')
    report['saved_sources_and_inputs_unchanged'] = True
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'report_sha256': sha(args.output),
        'same_compile_operator_records': compiler['same_final_operator_records'],
        'different_binary_bytes': compiler['different_binary_byte_count'],
        'profile_same_operator_identity': report.get('profile_operator_identity_same_off_on'),
        'candidate_accepted': False}))


if __name__ == '__main__':
    main()
