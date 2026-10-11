#!/usr/bin/env python3
"""Audit saved original-pair hybrid compiler/profile records; never execute a model.

All private inputs are supplied through CLI paths. Only pure SHA-pinned contracts
and the existing saved-profile parser are imported. No executor or board access.
The full report stays private; --public-output emits counts and conclusions only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics

GRAPH_SHA = '2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CAPTURE_SHA = 'bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc'
LAYOUT_SHA = '95dd66db2194e3a3c90fb6b854bad94d3ebab231117c074c50f73222b15094c2'
COMPILE_SHA = '60e046c89442a5e0fc2cb5b793cf7184d48ec38810a2e46ba9215ae549bfd50c'
COMPILE_LOG_SHA = '42dd647d6b5f544b0849a7768a4b5f932f9b17bfa63b7080272c4108ba20eb16'
ORDINARY_COMPILE_SHA = 'd19b34f90ee1c7d41724fd36fb4ba808a880863f57ce58f053cc292d7e705d34'
ORDINARY_LOG_SHA = 'be8c61773750d2c89bf391ba32bff939170cc3c43fae72d22462054769e8ab73'
MODEL_SHA = 'a502ceb2ae936550d56efe94c4fa4ff385b6d4530a0eaa62ac7ba0d57830b16d'
ORDINARY_MODEL_SHA = 'da0b26174ea54657d8352fb3b02acfbf4dc6ab5690f5855229e3516dbce83976'
COMPILER_SHA = '3ca43c492c6ee00a2cb20d7d22e9cdd9d27f570e892ea6e56881cea7b121a224'
CONTRACTS_SHA = '4d599517db9cbc438c1b095bb97c0dd74ea923cc6eaf613a8d32cd594b9a9101'
R1_CONTRACTS_SHA = 'd92103b618fd186605269d151f4cd480429b26570aa5b466c5f0f90ca01075df'
POLICY_SHA = 'd250f505c7ed4c4764d13ccf1e5b79b9b412a68266cb9f795ef51be8b4db3935'
STEP1_SHA = '7c2a101be488dded5086c6b418480f459bc14a62f9269991c1d88a6892ec8133'
RUNNER_SHA = '8138d93dc642e50ba1df1466e8a505deb1fbe12b64199d3959ae008c4e6e52e9'
PARENT_SHA = '33fc5deb749c8b3ceaa47271e389f5fbbfc2dba6db93aa19887b2fc41e0e7882'
PROFILE_PARSER_SHA = '061b15948907ab1b1317c1c25315e7683f26b191b04345558cf670eff20b111e'
NORMAL_SHA = '6532b30b12ee714d5ea37ea58d0c71ff2168d4a40c9b96b7eb71ee8dea6eb199'
PROFILE_SHA = 'caefb4cdd49ee22beaa3e49c2d9ea253f0f34b03b8dabcc69774bb3c2f7d07ba'
ROLES = ('ordinary', 'hybrid')
PREFIX = re.compile(r'^D RKNN:\s*\[[^\]]+\]\s*')
OP = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.+?)\s+'
                r'(\d+/\d+/\d+)\s+(\d+)\s+(.+?)\s*$')
ALLOCATION = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.+?)\s*\|\s*'
                        r'(0x[0-9a-f]+)[ *]+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s*$')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    need(re.fullmatch(r'[0-9a-f]{64}', expected) is not None and sha(path) == expected,
         'Saved evidence hash mismatch: ' + str(path))
    pins[str(path)] = expected
    return path


def owned_path(directory, relative):
    directory = Path(directory).resolve(strict=True)
    rel = Path(relative)
    need(not rel.is_absolute() and '..' not in rel.parts, 'Artifact path must be receipt-owned')
    path = (directory / rel).resolve(strict=True)
    need(path.is_relative_to(directory), 'Artifact escaped receipt directory')
    return path


def source(path, expected, pins, name):
    path = pin(path, expected, pins)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(row, offset=0, dtype=True):
    fields = (row['id'] + offset, row['op'], row['target'], row['name'],
              ' '.join(row['input_and_output_shapes_text'].split()))
    return fields + ((row['data_type_token'],) if dtype else ())


def board_sum(values):
    result = 0.0
    for value in values:
        result += value
    return result


def repair_exact_interleave(raw):
    """Repair one known stdout/stderr insertion, never discard an unparsed row."""
    success = '{"status": "compiled_not_runtime_validated", "receipt_sha256": "' + COMPILE_SHA + '", "error": null}'
    inserted = 'I rknn building done.\n' + success + '\n'
    need(raw.count(inserted) == 1, 'Expected exact source-pinned logger insertion once')
    start = raw.index(inserted)
    need(raw[max(0, start - 18):start].endswith('D RKNN: [14:20:')
         and raw[start + len(inserted):].startswith('54.140] 48  Reshape'),
         'Known insertion does not interrupt the expected timestamp/constant row')
    repaired = raw[:start] + raw[start + len(inserted):]
    return repaired, {'removed_only_known_stdout_status_insertion': True,
        'original_log_sha256': COMPILE_LOG_SHA, 'inserted_text_sha256': hashlib.sha256(inserted.encode()).hexdigest(),
        'inserted_text_bytes': len(inserted.encode()), 'insertion_byte_offset': len(raw[:start].encode()),
        'repaired_log_sha256': hashlib.sha256(repaired.encode()).hexdigest(),
        'restored_constant_row': '48 Reshape global0/attention_heads_flatten_rs_i1_sw INT64 (4)',
        'raw_evidence_modified': False}


def parse_tables(raw, allocations=True):
    """Parse every numeric row in each final table, preserving printed dtypes."""
    operators, native, weights = [], [], []
    section, current = None, None
    for line in re.sub(r'\x1b\[[0-9;]*m', '', raw).splitlines():
        if 'ID   OpType' in line and 'Target' in line:
            section = 'ops'; current = []; operators.append(current); continue
        if 'ID  User' in line and 'NativeShape' in line:
            section = 'native'; continue
        if 'ID  User' in line and 'OrigShape' in line and 'NativeShape' not in line:
            section = 'weights'; continue
        text = PREFIX.sub('', line)
        if text == line or not re.match(r'^\s*\d+\s+', text):
            continue
        if section in ('native', 'weights') and not allocations:
            continue
        if section == 'ops':
            m = OP.fullmatch(text)
            need(m is not None, 'Unparsed compiler operator: ' + line)
            current.append({'id': int(m[1]), 'op': m[2], 'data_type_token': m[3], 'target': m[4],
                'input_and_output_shapes_text': m[5], 'DDR_NPU_total_cycles': [int(v) for v in m[6].split('/')],
                'RW_KB': int(m[7]), 'name': m[8]})
        elif section in ('native', 'weights'):
            m = ALLOCATION.fullmatch(text)
            need(m is not None, 'Unparsed compiler allocation: ' + line)
            start, end, size = (int(m[i], 16) for i in (6, 7, 8))
            need(end - start == size and size > 0, 'Inconsistent printed allocation extent')
            row = {'operator_id': int(m[1]), 'operator': m[2], 'tensor': m[3], 'data_type_token': m[4],
                'shape_and_format_text': m[5], 'allocation_start': start, 'allocation_end': end,
                'allocation_bytes': size}
            if section == 'weights':
                shape = ast.literal_eval(m[5])
                if type(shape) is int:
                    shape = (shape,)
                need(isinstance(shape, tuple) and all(type(v) is int and v > 0 for v in shape),
                     'Constant shape is not a positive integer tuple')
                row['shape'] = list(shape); row['logical_elements'] = math.prod(shape)
            (native if section == 'native' else weights).append(row)
        else:
            raise ValueError('Numeric compiler row outside a recognized table')
    need(operators, 'Missing final compiler table')
    for table in operators:
        need([v['id'] for v in table] == list(range(len(table))) and len(table) >= 2,
             'Incomplete/noncontiguous final operator table')
        need(table[0]['op'] == 'InputOperator' and table[-1]['op'] == 'OutputOperator',
             'Expected one logical input/output pair')
    return operators, native, weights


def compile_evidence(args, pins):
    compiled = json.loads(pin(args.compile_dir / 'compile.json', COMPILE_SHA, pins).read_text())
    contracts = source(args.step2_contracts, CONTRACTS_SHA, pins, '_v50_saved_r2_contracts')
    pin(contracts.R1_SOURCE, R1_CONTRACTS_SHA, pins)
    contracts.check_step2_compile(compiled, COMPILER_SHA, CONTRACTS_SHA, POLICY_SHA, STEP1_SHA, MODEL_SHA)
    model = pin(owned_path(args.compile_dir, compiled['output']['file']), MODEL_SHA, pins)
    need(model.stat().st_size == compiled['output']['bytes'] == 48050420, 'Hybrid model extent changed')
    captured = json.loads(pin(args.capture_manifest, CAPTURE_SHA, pins).read_text())
    need(compiled['capture_samples'] == captured['samples'], 'Compile/capture sample metadata differ')
    dataset = pin(args.compile_dir / 'dataset.txt', compiled['dataset']['sha256'], pins)
    need(dataset.stat().st_size == 728 and dataset.read_bytes() ==
         ''.join(v['path'] + '\n' for v in compiled['dataset']['rows']).encode(), 'Eight-row dataset bytes changed')
    need([v['sample_id'] for v in compiled['dataset']['rows']] == [f'calibration_{i:02d}' for i in range(8)]
         and compiled['dataset']['heldout_samples_excluded'] == ['evaluation_00', 'evaluation_01'],
         'Calibration/evaluation exclusion changed')
    pin(args.original_graph, GRAPH_SHA, pins)
    pin(args.layout_audit, LAYOUT_SHA, pins)
    ordinary = json.loads(pin(args.ordinary_compile_dir / 'compile.json', ORDINARY_COMPILE_SHA, pins).read_text())
    need(ordinary['pair_graph_sha256'] == GRAPH_SHA and ordinary['candidate_accepted'] is False
         and ordinary['source_and_inputs_unchanged'] is True and ordinary['cleanup_errors'] == []
         and ordinary['source_and_input_hashes'] == ordinary['source_and_input_hashes_after'],
         'Original nonquantized compilation provenance changed')
    item = ordinary['variants']['ordinary']
    need(item['status'] == 'compiled_not_runtime_validated' and item['released'] is True
         and item['configuration']['enable_flash_attention'] is False
         and item['build_options'] == {'do_quantization': False, 'dataset': None, 'auto_hybrid': False}
         and all(o['status'] == 'completed' and type(o['code']) is int and o['code'] == 0 for o in item['operations']),
         'Clean nonquantized comparison required')
    model = pin(owned_path(args.ordinary_compile_dir, item['output']['file']), ORDINARY_MODEL_SHA, pins)
    need(model.stat().st_size == item['output']['bytes'] == 55280948, 'Ordinary model extent changed')
    raw = pin(args.compiler_log, COMPILE_LOG_SHA, pins).read_text()
    repaired, repair = repair_exact_interleave(raw)
    tables, native, weights = parse_tables(repaired)
    other_tables, _, _ = parse_tables(pin(args.ordinary_compiler_log, ORDINARY_LOG_SHA, pins).read_text(), allocations=False)
    need(len(tables) == 1 and len(tables[0]) == 58 and len(native) == 101 and len(weights) == 56,
         'Actual hybrid table/allocation counts differ')
    need(len(other_tables) == 2 and all(len(t) == 54 for t in other_tables), 'Ordinary off/on tables incomplete')
    hybrid = tables[0]; nonquant = other_tables[0]
    clean = [row for row in hybrid if row['op'] != 'exDataConvert']
    need(len(clean) == len(nonquant), 'Existing compiled operators removed/added outside conversions')
    changes = []
    for i, (old, new) in enumerate(zip(nonquant, clean)):
        aligned = dict(new, id=old['id'])
        need(identity(old, dtype=False) == identity(aligned, dtype=False),
             'Existing operator name/type/shape/target changed beyond precision allocation')
        if old['data_type_token'] != new['data_type_token']:
            changes.append({'name': new['name'], 'from': old['data_type_token'], 'to': new['data_type_token']})
    need(len(changes) == 2 and all('mlp.fc1' in c['name'] and c['from'] == 'FLOAT16' and c['to'] == 'INT8'
                                 for c in changes), 'Unexpected existing operator dtype changes')
    need(len([r for r in hybrid if r['op'] == 'exDataConvert']) == 4, 'Expected four explicit boundary conversions')
    internal = hybrid[1:-1]
    need(Counter(r['target'] for r in internal) == {'NPU': 56}
         and Counter(r['data_type_token'] for r in internal) == {'FLOAT16': 52, 'INT8': 4},
         'Unexpected hybrid final placement/dtype labels')
    learned = {}
    for role in ('frame0', 'global0'):
        values = {}
        for suffix, dtype in [('mlp.fc1.weight_transposed', 'INT8'), ('mlp.fc1.bias', 'INT32'),
                              ('mlp.fc2.weight_transposed', 'FLOAT16'), ('mlp.fc2.bias', 'FLOAT')]:
            found = [w for w in weights if w['tensor'] == role + '/' + suffix]
            need(len(found) == 1 and found[0]['data_type_token'] == dtype, 'Final MLP constant dtype changed')
            values[suffix] = found[0]
        fc2_id = values['mlp.fc2.weight_transposed']['operator_id']
        fc2_inputs = [n for n in native if n['operator_id'] == fc2_id and
                      n['tensor'] in (role + '/gelu_tp-rs__float16', role + '/residual1-rs_tp')]
        need(len(fc2_inputs) == 2 and all(n['data_type_token'] == 'FLOAT16' for n in fc2_inputs),
             'FC2 GELU/residual boundary allocation is not floating')
        values['fc2_native_input_allocations'] = fc2_inputs
        learned[role] = values
    return compiled, ordinary, captured, {'hybrid': hybrid, 'ordinary': nonquant}, {
        'final_operator_count': 58, 'native_allocation_rows': 101, 'constant_allocation_rows': 56,
        'all_existing_54_operator_types_names_shapes_targets_retained': True,
        'existing_operator_dtype_changes': changes, 'added_conversion_operator_count': 4,
        'internal_target_counts': dict(Counter(r['target'] for r in internal)),
        'internal_dtype_labels': dict(Counter(r['data_type_token'] for r in internal)),
        'constant_dtype_labels': dict(Counter(w['data_type_token'] for w in weights)),
        'native_dtype_labels': dict(Counter(n['data_type_token'] for n in native)),
        'MLP_constants_and_native_boundaries': learned, 'known_logger_interleave_repair': repair,
        'weights_table_precision_equals_all_arithmetic_precision': False}, native, weights


def check_run(directory, expected, normal, compiled, ordinary, captured, pins):
    row = json.loads(pin(directory / 'run.json', expected, pins).read_text())
    want = {'schema': 'v50_original_pair_hybrid_SDK_AB', 'status': 'completed_diagnostic_not_adopted',
        'script_sha256': RUNNER_SHA, 'parent_source_sha256': PARENT_SHA,
        'ordinary_compile_sha256': ORDINARY_COMPILE_SHA, 'hybrid_compile_sha256': COMPILE_SHA,
        'hybrid_source_sha256': COMPILER_SHA, 'hybrid_contracts_sha256': CONTRACTS_SHA,
        'hybrid_policy_sha256': POLICY_SHA, 'step1_sha256': STEP1_SHA,
        'original_graph_sha256': GRAPH_SHA, 'capture_sha256': CAPTURE_SHA,
        'candidate_accepted': False, 'full_model_run': False, 'real_time_validated': False,
        'whole_model_quality_validated': False, 'strict_gate': {'rtol': 1e-4, 'atol': 1e-5},
        'normal_benchmark': normal, 'profile_separate_fresh_contexts': not normal,
        'successful_SDK_calls': 16, 'requested_sdk_call_count': 16, 'unmeasured_warmup_calls': 0,
        'contexts_live_together': True, 'cleanup_errors': [], 'affinity_restored': True,
        'environment_restored': True, 'source_and_inputs_unchanged': True,
        'hybrid_requested_floating_IO_observed': True}
    need(all(type(row.get(k)) is type(v) and row.get(k) == v for k, v in want.items()),
         'Actual complete unchanged hybrid runtime contract mismatch')
    need(row['source_and_input_hashes'] == row['source_and_input_hashes_after']
         and row['original_checkpoint'] == captured['checkpoint']
         and captured['checkpoint']['sha256'] == CHECKPOINT_SHA, 'Original source/checkpoint changed')
    need(row['compiled_artifacts'] == {'ordinary': ordinary['variants']['ordinary']['output'],
                                      'hybrid': compiled['output']}, 'Runtime artifacts differ from compile')
    need(set(row['context_records']) == set(ROLES), 'Two typed contexts required')
    for context in row['context_records'].values():
        need(context['context_destroyed'] is True and context['destroy_code'] == 0
             and context['init_flags'] == (0 if normal else 8) and context['init_code'] == 0
             and context['core_mask'] == 7 and context['core_mask_code'] == 0
             and context['sdk_version'].startswith('2.3.2') and context['driver_version'] == '0.7.2'
             and context['actual_logical_IO_floating'] is True, 'Typed fresh-context lifecycle mismatch')
        for direction, shape in [('input', [2, 437, 1024]), ('output', [1, 874, 1024])]:
            attr = context[direction + '_attr']
            need(attr['index'] == 0 and attr['name'] == direction and attr['shape'] == shape
                 and attr['n_elems'] == 894976 and attr['size'] == 1789952 and attr['format'] == 3
                 and attr['type'] == 1 and attr['quantization_type'] == 2 and attr['scale'] == 1.0
                 and attr['zero_point'] == 0 and attr['pass_through'] == 0,
                 'Actual FP16 identity-affine logical IO descriptor changed')
    evaluations = {s['sample_id']: s for s in captured['samples'] if s['split'] == 'evaluation'}
    schedule = []
    for sample in range(2):
        for iteration in range(4):
            order = ROLES if (sample + iteration) % 2 == 0 else ROLES[::-1]
            schedule.extend((f'evaluation_{sample:02d}', iteration, role) for role in order)
    need(len(row['calls']) == len(row['sdk_calls']) == 16
         and [(c['sample_id'], c['round_index'], c['role']) for c in row['calls']] == schedule,
         'Expected sixteen measured AB/BA calls')
    context_counts = Counter()
    previous_end = 0
    for i, (call, sdk) in enumerate(zip(row['calls'], row['sdk_calls'])):
        sample = evaluations[call['sample_id']]
        need(call['input_raw_sha256'] == sample['artifacts']['input.npy']['raw_sha256']
             and call['reference_raw_sha256'] == sample['artifacts']['output.npy']['raw_sha256'],
             'Original input/Torch reference identity changed')
        need(sdk['sdk_call_index'] == call['sdk_call_index'] == i and sdk['role'] == call['role']
             and sdk['sample_id'] == call['sample_id'] and sdk['sample_call_index'] == call['round_index']
             and sdk['context_call_index'] == context_counts[call['role']]
             and sdk['first_in_context'] is (context_counts[call['role']] == 0), 'Call/context indices changed')
        context_counts[call['role']] += 1
        need(sdk['status'] == 'completed' and sdk['synchronous'] is True
             and all(sdk[k] == 0 for k in ('inputs_set_code', 'run_code', 'outputs_get_code', 'outputs_release_code'))
             and sdk['input_api_type'] == 'FP32' and sdk['input_api_type_enum'] == 0
             and sdk['input_pass_through'] == 0 and sdk['input_format'] == 3 and sdk['want_float'] == 1
             and sdk['input_shape'] == [2, 437, 1024] and sdk['output_shape'] == [1, 874, 1024]
             and sdk['input_bytes'] == sdk['output_bytes'] == sdk['expected_FP32_output_bytes'] == 3579904
             and sdk['output_owned_copy_before_release'] is True and sdk['output_buffer_released'] is True
             and sdk['owned_output_pointer'] > 0 and sdk['SDK_output_pointer'] > 0
             and sdk['owned_output_pointer'] != sdk['SDK_output_pointer']
             and sdk['runtime_frame_id'] == sdk['output_frame_id']
             and sdk['manual_quantization_or_dequantization'] is False
             and sdk['input_raw_sha256_before'] == sdk['input_raw_sha256_after'] == call['input_raw_sha256'],
             'Actual typed SDK conversion/ownership/release contract mismatch')
        begin, end = sdk['io_begin_monotonic_ns'], sdk['io_end_monotonic_ns']
        need(previous_end <= begin < end and (end - begin) / 1e9 == call['io_total_seconds'] == sdk['io_total_seconds'],
             'Overlapping or inconsistent synchronous SDK IO clock')
        phase_names = ('input_set_seconds', 'run_api_seconds', 'outputs_get_seconds',
                       'output_validation_and_copy_seconds', 'pre_release_bookkeeping_seconds', 'outputs_release_seconds')
        need(all(math.isfinite(sdk[k]) and sdk[k] >= 0 for k in phase_names)
             and sum(round(sdk[k] * 1e9) for k in phase_names) == end - begin,
             'SDK measured phase clocks do not sum exactly to IO nanoseconds')
        previous_end = end
        comparison = call['original_FP32_comparison']
        need(comparison['rtol'] == 1e-4 and comparison['atol'] == 1e-5
             and comparison['strict_FP32_gate_passed'] is False and comparison['outside_tolerance_elements'] > 0,
             'Original elementwise rejection was hidden/threshold changed')
        out = call['output']; path = pin(owned_path(directory, out['file']), out['sha256'], pins)
        need(path.stat().st_size == out['bytes'] == 3580032 and out['shape'] == [1, 874, 1024]
             and out['dtype'] == 'float32' and out['tensor_bytes'] == 3579904
             and comparison['output_raw_sha256'] == out['raw_sha256'], 'Saved output extent/metadata mismatch')
        need(('separate_profile_detail' in call) is (not normal), 'Profile and normal call modes mixed')
    for role in ROLES:
        repeats = [c for c in row['calls'] if c['role'] == role and c['round_index'] > 0]
        need(len(repeats) == 6 and row['summaries'][role]['mean_denominator'] == 6
             and row['summaries'][role]['subsequent_IO_mean_seconds'] == board_sum(c['io_total_seconds'] for c in repeats) / 6
             and row['summaries'][role]['strict_original_FP32_gate_passed'] is False,
             'Normal/profile repeat mean or rejection mismatch')
    return row


def run(args):
    pins = {}
    compiled, ordinary, captured, final_ops, compile_summary, native, weights = compile_evidence(args, pins)
    normal = check_run(args.normal_dir, NORMAL_SHA, True, compiled, ordinary, captured, pins)
    profile = check_run(args.profile_dir, PROFILE_SHA, False, compiled, ordinary, captured, pins)
    parser = source(args.profile_parser, PROFILE_PARSER_SHA, pins, '_v50_reused_saved_profile_parser')
    tables = {role: [] for role in ROLES}
    for call in profile['calls']:
        detail = call['separate_profile_detail']
        path = pin(owned_path(args.profile_dir, detail['file']), detail['raw_sha256'], pins)
        need(path.stat().st_size == detail['bytes'] and detail['raw_sha256'] == detail['saved_text_sha256']
             and detail['query_command'] == 3, 'Raw profile hash/extent/query mismatch')
        rows, total = parser.parse(path.read_text())
        need([identity(r) for r in rows] == [identity(r, 1) for r in final_ops[call['role']]],
             'Actual profile type/name/shape/dtype/target differs from final compiler table')
        tables[call['role']].append({'sample_id': call['sample_id'], 'sample_call_index': call['round_index'],
            'rows': rows, 'exact_table_total_microseconds': total, 'sha256': sha(path), 'file': path.name})
    summaries = {role: parser.summarize_tables(values) for role, values in tables.items()}
    need(all(n['output']['raw_sha256'] == p['output']['raw_sha256']
             for n, p in zip(normal['calls'], profile['calls'])), 'Normal/profile output metadata identity changed')
    focus = {}
    for role, summary in summaries.items():
        focus[role] = {}
        for block in ('frame0', 'global0'):
            named = [v for v in summary['profile_named_operators'] if block + '/' in v['name']]
            selected = [v for v in named if v['op'] in ('ConvExGelu', 'exDataConvert') or 'mlp.fc2' in v['name']]
            focus[role][block] = {v['name']: {'op': v['op'], 'data_type_token': v['data_type_token'],
                'target': v['target'], 'mean_all_eight_microseconds': v['mean_all_eight_microseconds']} for v in selected}
    conversions = sum(v['mean_all_eight_microseconds'] for block in focus['hybrid'].values()
                      for v in block.values() if v['op'] == 'exDataConvert')
    normal_means = {role: normal['summaries'][role]['subsequent_IO_mean_seconds'] for role in ROLES}
    need(all(sha(path) == expected for path, expected in pins.items()), 'Saved evidence changed during audit')
    report = {'schema': 'v50_original_pair_hybrid_saved_operator_independent_audit',
        'status': 'saved_compile_profile_checked_diagnostic_not_adopted', 'script_sha256': sha(__file__),
        'actual_executor_or_board_access_by_auditor': False, 'candidate_accepted': False,
        'compiled_final_operator_tables': final_ops, 'native_allocation_rows': native, 'constant_allocation_rows': weights,
        'compile_summary': compile_summary, 'profile_tables': tables, 'profile_summaries': summaries,
        'named_MLP_and_conversion_means': focus, 'conversion_operator_mean_all_eight_microseconds': conversions,
        'separate_normal_SDK_IO_seconds': normal_means,
        'normal_IO_change_percent': (normal_means['hybrid'] / normal_means['ordinary'] - 1) * 100,
        'profile_SDK_IO_seconds_separate': profile['summaries'],
        'all_32_output_files_hashed_no_NPY_numeric_arrays_loaded_by_this_auditor': True,
        'normal_profile_output_raw_hash_metadata_equal': True, 'saved_sources_and_inputs_unchanged': True,
        'evidence': pins}
    compact_profiles = {role: {key: summary[key] for key in (
        'operator_rows_per_call', 'internal_rows_per_call', 'internal_target_counts', 'boundary_target_counts',
        'internal_data_type_token_counts', 'all_eight_operator_mean_microseconds',
        'six_repeat_operator_mean_microseconds', 'original_subregions')} for role, summary in summaries.items()}
    public_weights = {block: {key: {k: value[k] for k in ('data_type_token', 'shape', 'logical_elements', 'allocation_bytes')}
        for key, value in values.items() if key != 'fc2_native_input_allocations'}
        for block, values in compile_summary['MLP_constants_and_native_boundaries'].items()}
    public = {'schema': 'v50_original_pair_hybrid_public_operator_audit', 'status': report['status'],
        'script_sha256': report['script_sha256'], 'actual_executor_or_board_access_by_auditor': False,
        'evidence_sha256': {'original_pair_graph': GRAPH_SHA, 'original_checkpoint': CHECKPOINT_SHA,
            'static_original_layout_audit': LAYOUT_SHA, 'capture': CAPTURE_SHA, 'hybrid_compile': COMPILE_SHA,
            'ordinary_compile': ORDINARY_COMPILE_SHA, 'hybrid_compiler_stdout': COMPILE_LOG_SHA,
            'normal': NORMAL_SHA, 'profile': PROFILE_SHA, 'runner': RUNNER_SHA,
            'step2_contracts': CONTRACTS_SHA, 'profile_parser': PROFILE_PARSER_SHA, 'precision_policy': POLICY_SHA},
        'compile_summary': {k: v for k, v in compile_summary.items() if k != 'MLP_constants_and_native_boundaries'},
        'final_MLP_constant_labels': public_weights, 'FC2_GELU_and_residual_native_inputs_FLOAT16': True,
        'profiles': compact_profiles, 'profiles_checked': 16,
        'all_profile_rows_match_final_compiler_identity_shape_dtype_target': True,
        'all_profile_microseconds_sum_exactly_to_each_sdk_TOTAL': True,
        'named_MLP_and_conversion_means': focus, 'conversion_mean_all_eight_microseconds': conversions,
        'separate_normal_SDK_IO': {'repeat_calls_per_implementation': 6, 'means_seconds': normal_means,
            'mean_IO_change_percent': report['normal_IO_change_percent'], 'strict_FP32_gate_passed': False,
            'fixed_gate': {'rtol': 1e-4, 'atol': 1e-5}, 'original_Torch_FP32_reference_only': True},
        'all_32_output_file_SHA_extent_metadata_checked': True, 'SDK_output_copy_owned_before_release': True,
        'logical_IO_FP16_identity_affine_observed': True, 'API_FP32_pass_through_0_want_float_1': True,
        'normal_and_profile_fresh_contexts_separate': True, 'source_and_input_SHA_before_after_identical': True,
        'candidate_accepted': False, 'full_model_or_SLAM_quality_or_realtime_validated': False,
        'limitations': [
            'Only the complete original frame0/global0 pair is covered; the original whole model and SLAM are not run by this probe',
            'Compiler/profile dtype tokens and printed constant allocations do not reveal every arithmetic instruction precision',
            'INT32 FC1 bias allocation is 32768 bytes for 4096 logical elements; packing/metadata do not change its printed INT32 label',
            'FC2 ConvAdd includes fused LayerScale and residual semantics; named MLP times are not pure original MLP times',
            'Conversion operator dtype labels describe their input side; INT8-to-FLOAT16 is not an INT8 output boundary',
            'Normal SDK IO, profile SDK IO and profile operator clocks are separate measured ranges and must not be summed',
            'Two development evaluation inputs do not establish trajectory accuracy, P95, a camera or fresh-pose throughput',
            'Saved output arrays are independently checked by a separate numeric audit; this source hashes files and validates metadata only',
            'All original elementwise FP32 rejections and the failed step2-r1 selector attempt remain retained']}
    return report, public


def tiny_checks():
    prefix = 'D RKNN: [00:00:00.000] '
    header = prefix + 'ID   OpType DataType Target InputShape OutputShape Cycles RW FullName\n'
    rows = [prefix + '0 InputOperator FLOAT16 CPU \\ (1) 0/0/0 0 InputOperator:input\n',
            prefix + '1 OutputOperator FLOAT16 CPU (1) \\ 0/0/0 0 OutputOperator:output\n']
    example = header + ''.join(rows)
    need(len(parse_tables(example)[0][0]) == 2, 'Generated complete table failed')
    negative = [example.replace('1 OutputOperator', '2 OutputOperator'),
                example.replace('FLOAT16 CPU (1)', 'BROKEN (1)'),
                example + prefix + 'ID  User Tensor DataType OrigShape | Start End Size\n' +
                prefix + '1 Conv test.weight FLOAT16 (2) | 0x0 0x4 0x2\n']
    for item in negative:
        try:
            parse_tables(item)
        except ValueError:
            pass
        else:
            raise ValueError('Malformed generated compiler table accepted')
    return {'schema': 'v50_hybrid_saved_parser_generated_checks', 'checks_passed': 4,
        'actual_model_ORT_SDK_compiler_calibration_board_access': False,
        'checks': ['complete generated table', 'noncontiguous ID rejected', 'unparsed operator rejected',
                   'allocation end-start mismatch rejected']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny-checks', action='store_true')
    names = ('compile-dir', 'ordinary-compile-dir', 'compiler-log', 'ordinary-compiler-log',
             'normal-dir', 'profile-dir', 'capture-manifest', 'step2-contracts', 'profile-parser',
             'original-graph', 'layout-audit', 'output', 'public-output')
    for name in names:
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.tiny_checks:
        print(json.dumps(tiny_checks(), allow_nan=False)); return
    for name in names:
        if getattr(args, name.replace('-', '_')) is None:
            parser.error('--' + name + ' is required for saved-data audit')
    need(args.output != args.public_output and not args.output.exists() and not args.public_output.exists(),
         'Exclusive new private/public outputs required')
    report, public = run(args)
    for file, value in ((args.output, report), (args.public_output, public)):
        with file.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': report['status'], 'private_sha256': sha(args.output),
        'public_sha256': sha(args.public_output), 'candidate_accepted': False}))


if __name__ == '__main__':
    main()
