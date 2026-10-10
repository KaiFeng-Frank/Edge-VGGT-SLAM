#!/usr/bin/env python3
"""Independently verify saved v23 MLP artifacts, without inference or SDK calls.

NumPy is the only required non-stdlib dependency. --check-onnx additionally
parses the four ONNX graphs/weights with ONNX; it never executes those graphs.
An audit may pass consistency checks while the original FP32 accuracy gate
fails. That distinction and the observed CPU/NPU placement remain explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np

sys.dont_write_bytecode = True
CAPTURE_SHA = '3664f5c932602a3c6538ea6013ea0389625c42bdcb7dcf10a9fbceb9304abeb1'
EXPORTERS = {
    'r1': '56476c724cde537fd35466be0b41d300592821be59f2a840519be6c117c5146e',
    'r2': '31602681e0c6ee5650baf66784764124926b8666ca75e4529bda71f98d54585a',
}
DRIVERS = {
    'r1': 'a3ea513431ae14a9fca6c3b5cd9073b847c3b1908a0e132e5d497e5810a34fd4',
    'r2': 'b328df00e0178b91a82c06dc840920159c88a35d0d1d8500126fb786ae90b331',
}
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
NAMES = ('baseline', 'shard_0', 'shard_1', 'shard_2')
INTERVALS = ((0, 4096), (0, 1344), (1344, 2688), (2688, 4096))
SHAPE = (1, 874, 1024)
SCHEDULE = ['A', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A']
RTOL, ATOL = 1e-4, 1e-5
PROFILE_ROW = re.compile(
    r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU)\s+(.*?)\s+(.*?)\s+'
    r'(\d+)/(\d+)/(\d+)\s+(\d+)\s+(.*?)\s+'
    r'([\d.]+%/[\d.]+%/[\d.]+%)\s+(\d+)\s+(.+)$')


class Checks:
    def __init__(self):
        self.count = 0

    def need(self, condition, message):
        self.count += 1
        if not condition:
            raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def local_file(directory, name, checks):
    checks.need(isinstance(name, str) and Path(name).name == name and name not in ('', '.', '..'),
                'Artifact must have a simple local filename')
    path = directory / name
    checks.need(path.is_file(), 'Missing artifact: ' + name)
    return path


def artifact(directory, item, checks, *, size_key='bytes'):
    path = local_file(directory, item['file'], checks)
    checks.need(sha(path) == item['sha256'], 'File SHA mismatch: ' + path.name)
    checks.need(path.stat().st_size == item[size_key], 'File size mismatch: ' + path.name)
    return path


def array_record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def exact(actual, expected, checks, message):
    checks.need(actual.shape == expected.shape and actual.dtype == expected.dtype
                and actual.tobytes(order='C') == expected.tobytes(order='C'), message)


def compare(actual, expected, checks):
    checks.need(actual.shape == expected.shape == SHAPE
                and actual.dtype == expected.dtype == np.float32, 'Comparison shape/dtype mismatch')
    checks.need(bool(np.isfinite(actual).all() and np.isfinite(expected).all()), 'Nonfinite array')
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    difference = a - b
    a_norm, b_norm = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    checks.need(a_norm > 0 and b_norm > 0, 'All-zero output/reference rejected')
    close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
    return dict(finite=True, output_all_zero=False, element_count=int(close.size),
                failed_elements=int(np.count_nonzero(~close)), elementwise_gate_passed=bool(close.all()),
                rtol=RTOL, atol=ATOL, byte_equal=actual.tobytes() == expected.tobytes(),
                max_abs_error=float(np.abs(difference).max()),
                rmse=float(np.sqrt(np.mean(difference * difference))),
                relative_l2=float(np.linalg.norm(difference) / b_norm),
                cosine_similarity=float(np.sum(a * b) / (a_norm * b_norm)),
                output_raw_sha256=array_record(actual)['sha256'])


def agree(actual, reported, checks):
    for name in ('finite', 'output_all_zero', 'element_count', 'failed_elements',
                 'elementwise_gate_passed', 'rtol', 'atol', 'output_raw_sha256'):
        checks.need(actual[name] == reported[name], 'Reported comparison mismatch: ' + name)
    for name in ('max_abs_error', 'rmse', 'relative_l2', 'cosine_similarity'):
        checks.need(math.isclose(actual[name], reported[name], rel_tol=1e-9, abs_tol=2e-14),
                    'Reported numeric comparison mismatch: ' + name)


def combine(parts, bias):
    result = parts[0].copy()
    for value in parts[1:]:
        np.add(result, value, out=result)
    if len(parts) == 3:
        np.add(result, bias, out=result)
    return result


def check_onnx(path, state, interval, partial, variant, checks):
    """Only parse tensors/nodes. No original-weight computation."""
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    checks.need(all(t.data_location != onnx.TensorProto.EXTERNAL for t in model.graph.initializer),
                'External ONNX data is not allowed')
    checks.need(len(model.graph.input) == len(model.graph.output) == 1
                and model.graph.input[0].name == 'input' and model.graph.output[0].name == 'output',
                'ONNX input/output names mismatch')
    for info in [model.graph.input[0], model.graph.output[0]]:
        tensor = info.type.tensor_type
        checks.need(tensor.elem_type == onnx.TensorProto.FLOAT
                    and [d.dim_value for d in tensor.shape.dim] == list(SHAPE), 'ONNX IO shape/type mismatch')
    initializers = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    start, end = interval
    embedded = {'mlp.fc1.weight': state['mlp.fc1.weight'][start:end],
                'mlp.fc1.bias': state['mlp.fc1.bias'][start:end],
                'mlp.fc2.weight': state['mlp.fc2.weight'][:, start:end]}
    if variant == 'r2':
        embedded['mlp.fc1.weight'] = embedded['mlp.fc1.weight'].T
        embedded['mlp.fc2.weight'] = embedded['mlp.fc2.weight'].T
    if not partial:
        embedded['mlp.fc2.bias'] = state['mlp.fc2.bias']
    else:
        checks.need('mlp.fc2.bias' not in initializers, 'Partial graph repeats FC2 bias')
    for name, value in embedded.items():
        exact(initializers[name], value, checks, 'ONNX original parameter slice differs: ' + name)
    constants = {'gelu_one': np.array(1, np.float32), 'gelu_half': np.array(.5, np.float32)}
    if variant == 'r2':
        constants['gelu_sqrt_two'] = np.array(math.sqrt(2), np.float32)
        nodes = [('MatMul', ['input', 'mlp.fc1.weight'], ['mlp.fc1.matmul']),
                 ('Add', ['mlp.fc1.matmul', 'mlp.fc1.bias'], ['mlp.fc1']),
                 ('Div', ['mlp.fc1', 'gelu_sqrt_two'], ['gelu_scaled_input']),
                 ('Erf', ['gelu_scaled_input'], ['gelu_erf']),
                 ('Add', ['gelu_erf', 'gelu_one'], ['gelu_erf_plus_one']),
                 ('Mul', ['mlp.fc1', 'gelu_erf_plus_one'], ['gelu_before_half']),
                 ('Mul', ['gelu_before_half', 'gelu_half'], ['gelu'])]
        nodes += ([('MatMul', ['gelu', 'mlp.fc2.weight'], ['output'])] if partial else
                  [('MatMul', ['gelu', 'mlp.fc2.weight'], ['mlp.fc2.matmul']),
                   ('Add', ['mlp.fc2.matmul', 'mlp.fc2.bias'], ['output'])])
    else:
        constants.update(gelu_inv_sqrt_two=np.array(1 / math.sqrt(2), np.float32),
                         flat_shape=np.array([874, 1024], np.int64), output_shape=np.array(SHAPE, np.int64))
        nodes = [('Reshape', ['input', 'flat_shape'], ['rows']),
                 ('Gemm', ['rows', 'mlp.fc1.weight', 'mlp.fc1.bias'], ['fc1']),
                 ('Mul', ['fc1', 'gelu_inv_sqrt_two'], ['gelu_scaled']),
                 ('Erf', ['gelu_scaled'], ['gelu_erf']),
                 ('Mul', ['fc1', 'gelu_half'], ['gelu_half_input']),
                 ('Add', ['gelu_erf', 'gelu_one'], ['gelu_erf_plus_one']),
                 ('Mul', ['gelu_half_input', 'gelu_erf_plus_one'], ['gelu']),
                 ('Gemm', ['gelu', 'mlp.fc2.weight'] + ([] if partial else ['mlp.fc2.bias']), ['fc2']),
                 ('Reshape', ['fc2', 'output_shape'], ['output'])]
    checks.need(set(initializers) == set(embedded) | set(constants), 'Unexpected ONNX initializer')
    for name, value in constants.items():
        exact(initializers[name], value, checks, 'ONNX constant differs: ' + name)
    checks.need([(n.op_type, list(n.input), list(n.output)) for n in model.graph.node] == nodes,
                'ONNX exact GELU/linear graph differs')
    for node in model.graph.node:
        expected_attrs = dict(alpha=1., beta=1., transA=0, transB=1) if node.op_type == 'Gemm' else {}
        checks.need(not node.domain and {a.name: helper.get_attribute_value(a) for a in node.attribute}
                    == expected_attrs, 'Unexpected ONNX node attributes/domain')


def load_inputs(args, checks):
    capture_file = args.capture / 'capture.json'
    checks.need(sha(capture_file) == CAPTURE_SHA, 'Frozen original capture SHA differs')
    capture = read_json(capture_file)
    checks.need(capture['status'] == 'captured_original_FP32_stages_bitwise_bound'
                and capture['sources_and_inputs_unchanged'] is True, 'Original stage capture not verified')
    loaded = {}
    for name, meta in [('activations.npz', 'arrays'), ('mlp-state.npz', 'global0_state')]:
        path = artifact(args.capture, dict(file=name, **capture['artifacts'][name]), checks)
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]
                checks.need(array_record(value) == capture[meta][key], 'Captured raw tensor mismatch: ' + key)
                loaded[key] = value
    graphs = []
    for i, name in enumerate(NAMES):
        directory = args.artifacts_root / name
        path = directory / 'export.json'
        export = read_json(path)
        checks.need(export['status'] == 'exported_and_structurally_checked'
                    and export['script_sha256'] == EXPORTERS[args.variant]
                    and export['capture_sha256'] == CAPTURE_SHA, 'Export identity mismatch')
        checks.need(export['input_shape'] == export['output_shape'] == list(SHAPE)
                    and export['hidden_slice'] == list(INTERVALS[i])
                    and export['includes_fc2_bias'] == (i == 0), 'Hidden slice/row/bias contract differs')
        checks.need(export['original_parameter_metadata'] == {k: capture['global0_state'][k] for k in
                    ('mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias')},
                    'Original parameter provenance differs')
        graph = artifact(directory, export['onnx'], checks)
        for filename, original_key in [('input.npy', 'norm2'), ('reference.npy', 'mlp_output'),
                                       ('fc2-bias.npy', 'mlp.fc2.bias')]:
            saved = export['artifacts'][filename]
            a = np.load(artifact(directory, saved, checks), allow_pickle=False)
            checks.need(array_record(a) == saved['array'], 'Export raw array metadata differs')
            exact(a, loaded[original_key], checks, 'Export original array bytes differ: ' + filename)
        if args.check_onnx:
            check_onnx(graph, loaded, INTERVALS[i], i != 0, args.variant, checks)
        compile_dir = args.artifacts_root / (name + '-compile')
        compile_file = compile_dir / 'compile.json'
        compiled = read_json(compile_file)
        checks.need(compiled['status'] == 'compiled_not_runtime_validated'
                    and compiled['script_sha256'] == COMPILER_SHA
                    and compiled['input']['sha256'] == sha(graph)
                    and compiled['configuration'] == dict(target_platform='rk3588', float_dtype='float16',
                                                         optimization_level=3, do_quantization=False),
                    'Compiled artifact source/configuration differs')
        model = artifact(compile_dir, dict(file='block.rknn', **compiled['output']), checks, size_key='size')
        graphs.append(dict(name=name, export_sha256=sha(path), onnx_sha256=sha(graph),
                           compile_sha256=sha(compile_file), model_sha256=sha(model),
                           hidden_slice=list(INTERVALS[i]), includes_fc2_bias=i == 0,
                           original_parameter_slices_and_nodes_checked=args.check_onnx))
    ort_path = args.ort / 'run.json'
    ort = read_json(ort_path)
    check_run_identity(ort, args, capture, graphs, checks, mode='ort', profile=False)
    checks.need(ort['provider'] == 'CPUExecutionProvider' and ort['graph_optimization'] == 'ORT_DISABLE_ALL'
                and ort['exact_graph_reference_gate_passed'] is True, 'Board CPU ORT gate not verified')
    refs = []
    checks.need(len(ort['output_arrays']) == 4, 'Exactly four saved ORT outputs required')
    for name, saved in zip(NAMES, ort['output_arrays']):
        checks.need(saved['file'] == name + '.npy', 'ORT output order differs')
        value = np.load(artifact(args.ort, saved, checks), allow_pickle=False)
        checks.need(array_record(value) == saved['array'], 'ORT raw tensor SHA mismatch')
        refs.append(value)
    summed = combine(refs[1:], loaded['mlp.fc2.bias'])
    comparisons = {n: compare(a, b, checks) for n, a, b in [
        ('baseline_vs_original', refs[0], loaded['mlp_output']),
        ('shards_vs_original', summed, loaded['mlp_output']), ('shards_vs_baseline', summed, refs[0]) ]}
    for name, values in comparisons.items():
        agree(values, ort['comparisons'][name], checks)
        checks.need(values['elementwise_gate_passed'], 'Original board ORT FP32 gate failed')
    return capture, loaded, graphs, refs, dict(receipt_sha256=sha(ort_path), comparisons=comparisons)


def check_run_identity(run, args, capture, graphs, checks, *, mode, profile):
    checks.need(run['status'] == 'completed' and run['mode'] == mode
                and run['script_sha256'] == DRIVERS[args.variant] and run['runtime_sha256'] == RUNTIME_SHA,
                'Runtime receipt/driver identity mismatch')
    checks.need(run['capture_sha256'] == CAPTURE_SHA and run['export_sha256'] == [g['export_sha256'] for g in graphs]
                and run['selected_pair'] == capture['selected_pair'] and run['checkpoint'] == capture['checkpoint'],
                'Source pair/weights/exports differ')
    checks.need(run['profile'] == profile and run['normal_benchmark_timing'] == (not profile)
                and run['shape'] == list(SHAPE) and run['hidden_neurons_dropped'] == 0,
                'Runtime mode/shape/hidden coverage differs')
    checks.need(run['cleanup_errors'] == [] and run['sources_and_inputs_unchanged'] is True
                and run['affinity_restored'] is True, 'Runtime cleanup/source consistency failed')


def parse_profile(call, checks):
    detail = call['perf_detail']
    checks.need(hashlib.sha256(detail['raw_text'].encode()).hexdigest() == detail['raw_sha256'],
                'Embedded profile SHA mismatch')
    rows = []
    for line in detail['raw_text'].splitlines():
        match = PROFILE_ROW.match(line)
        if match:
            group = match.groups()
            rows.append(dict(id=int(group[0]), op=group[1], target=group[3], time_us=int(group[9]),
                             workload=group[11], name=group[13]))
    checks.need(bool(rows) and [r['id'] for r in rows] == list(range(1, len(rows) + 1)),
                'Incomplete/duplicate SDK profile rows')
    return rows


def audit_run(directory, log, profile, args, capture, data, graphs, refs, stable, checks):
    receipt = directory / 'run.json'
    run = read_json(receipt)
    check_run_identity(run, args, capture, graphs, checks, mode='npu', profile=profile)
    checks.need(len(run['contexts']) == 4 and run['schedule'] == SCHEDULE and len(run['rounds']) == 10,
                'Context/call schedule differs')
    checks.need(run['cpu_affinity'] == [4, 5, 6, 7], 'Recorded CPU affinity differs')
    for i, context in enumerate(run['contexts']):
        checks.need(context['index'] == i and context['core_mask'] == [7, 1, 2, 4][i]
                    and context['model_sha256'] == graphs[i]['model_sha256']
                    and context['onnx_sha256'] == graphs[i]['onnx_sha256']
                    and context['export_sha256'] == graphs[i]['export_sha256'], 'Context/model binding mismatch')
        checks.need(context['destroy_code'] == 0 and context['context_destroyed'] is True,
                    'Owned context was not destroyed')
        for key in ('input_attr', 'output_attr'):
            desc = context[key]
            checks.need(desc['shape'] == list(SHAPE) and desc['n_elems'] == math.prod(SHAPE)
                        and desc['format'] == 3 and desc['type'] in (0, 1)
                        and desc['size'] == math.prod(SHAPE) * (4 if desc['type'] == 0 else 2),
                        'Logical rank-three SDK tensor/layout descriptor differs')
    checks.need(run['sum_context_dma_bytes'] == sum(c['memory_bytes']['total_dma_allocated_size']
                                                   for c in run['contexts']), 'Context DMA sum differs')
    checks.need(any(Path(p).name == 'run.json' and digest == sha(args.ort / 'run.json')
                    for p, digest in run['source_and_input_hashes'].items()), 'Original ORT receipt not pinned by run')
    for name, reference in zip(NAMES, refs):
        saved = args.ort / (name + '.npy')
        checks.need(any(p.endswith('/' + name + '.npy') and digest == sha(saved)
                        for p, digest in run['source_and_input_hashes'].items()), 'ORT array not pinned by run')
    artifacts, comparisons, profile_sets, first = [], {}, {}, {}
    array_count = sdk_count = 0
    for index, row in enumerate(run['rounds']):
        kind = row['variant']
        checks.need(row['index'] == index and kind == SCHEDULE[index]
                    and row['phase'] == ('warmup' if index < 2 else 'measured'), 'Warmup/call ordering differs')
        checks.need(len(row['calls']) == (1 if kind == 'A' else 3), 'Wrong partial output count')
        path = artifact(directory, row['artifact'], checks)
        with np.load(path, allow_pickle=False) as archive:
            checks.need(set(archive.files) == {'mlp', 'block', *['partial_' + str(i) for i in range(len(row['calls']))]},
                        'NPZ output schema differs')
            partials = [archive['partial_' + str(i)] for i in range(len(row['calls']))]
            actual, block = archive['mlp'], archive['block']
        for partial_index, (partial, call) in enumerate(zip(partials, row['calls'])):
            context_index = 0 if kind == 'A' else partial_index + 1
            checks.need(call['context_index'] == context_index and call['core_mask'] == [7, 1, 2, 4][context_index]
                        and call['input_format'] == 3 and call['input_pass_through'] == 0 and call['want_float'] == 1,
                        'Call identity/layout differs')
            checks.need(call['synchronous'] is True and call['output_buffer_released'] is True
                        and call['cleanup_errors'] == [] and call['perf_run_query_code'] == 0
                        and call['output_bytes'] == partial.nbytes == math.prod(SHAPE) * 4,
                        'Synchronous call/output ownership failed')
            checks.need(call['begin_monotonic'] <= call['end_monotonic']
                        and call['end_monotonic'] - call['begin_monotonic'] + 1e-8 >= call['io_total_seconds'],
                        'Call clock bracket differs')
            values = compare(partial, refs[context_index], checks)
            agree(values, call['versus_board_ORT'], checks)
            if context_index not in stable:
                stable[context_index] = partial.copy()
            exact(partial, stable[context_index], checks, 'Repeated output bytes changed')
            comparisons['context_' + str(context_index) + '_vs_ORT'] = values
            if profile:
                profile_sets.setdefault(context_index, []).append(parse_profile(call, checks))
        reconstructed = combine(partials, data['mlp.fc2.bias'])
        exact(actual, reconstructed, checks, 'Ordered sum/once-bias output differs')
        exact(block, actual * data['ls2.gamma'] + data['residual1'], checks, 'Original LS2/residual bytes differ')
        for key, value, expected in [('mlp_vs_original', actual, data['mlp_output']),
                                      ('block_vs_original', block, data['block_output'])]:
            values = compare(value, expected, checks)
            agree(values, row[key], checks)
            comparisons[kind + '_' + key] = values
        if kind not in first:
            first[kind] = actual.copy()
        agree(compare(actual, first[kind], checks), row['repeat'], checks)
        exact(actual, first[kind], checks, 'Same-variant outputs changed')
        checks.need(math.isclose(row['stage_endpoint_seconds'], sum(row[k] for k in
                    ('pack_seconds', 'execution_join_seconds', 'reduction_crop_seconds', 'ls2_residual_seconds')),
                    rel_tol=0, abs_tol=1e-12), 'Stage endpoint component sum differs')
        checks.need(math.isclose(row['mlp_endpoint_seconds'] + row['ls2_residual_seconds'],
                                row['stage_endpoint_seconds'], rel_tol=0, abs_tol=1e-12), 'MLP/stage endpoint differs')
        artifacts.append(dict(file=path.name, sha256=sha(path), variant=kind, phase=row['phase']))
        array_count += 2 + len(partials)
        sdk_count += len(partials)
    variant = compare(first['B'], first['A'], checks)
    agree(variant, run['variant_comparison'], checks)
    summary = {}
    for kind in ('A', 'B'):
        times = [r['stage_endpoint_seconds'] for r in run['rounds'] if r['phase'] == 'measured' and r['variant'] == kind]
        checks.need(len(times) == 4 and times == run['measured_summary'][kind]['values']
                    and statistics.mean(times) == run['measured_summary'][kind]['mean'], 'Measured timing summary differs')
        summary[kind] = dict(count=4, values_ms=[1000 * v for v in times], mean_ms=1000 * statistics.mean(times))
    clean_log = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', log.read_text())
    errors = [line for line in clean_log.splitlines() if re.search(
        r'(^|\s)E\s+RKNN|normalize error|unsupported src layout', line, re.I)]
    checks.need(not errors, 'SDK error lines found in actual log')
    coverage = []
    for i, sets in profile_sets.items():
        signature = [(r['op'], r['target'], r['name']) for r in sets[0]]
        checks.need(len(sets) == 5 and all([(r['op'], r['target'], r['name']) for r in rows] == signature
                                         for rows in sets), 'Profile call count/operator topology differs')
        internal_cpu = sorted({r['op'] for r in sets[0] if r['target'] == 'CPU'
                               and r['op'] not in ('InputOperator', 'OutputOperator')})
        expected_cpu = ['Erf', 'Transpose'] if args.variant == 'r1' else []
        checks.need(internal_cpu == expected_cpu, 'Actual CPU operator placement differs from experiment variant')
        operators = sorted({r['op'] for r in sets[0]})
        times = {op: statistics.mean(sum(r['time_us'] for r in rows if r['op'] == op) for rows in sets) / 1000
                 for op in operators}
        coverage.append(dict(context=i, core_mask=[7, 1, 2, 4][i], actual_calls=5,
                             operators_per_call=len(sets[0]), target_counts=dict(Counter(r['target'] for r in sets[0])),
                             internal_CPU_types=internal_cpu,
                             compute_workloads={op: sorted({r['workload'] for rows in sets for r in rows if r['op'] == op})
                                                for op in operators if op.startswith('Conv')},
                             separate_profile_mean_operator_ms=times))
    return dict(receipt_sha256=sha(receipt), log_sha256=sha(log), sdk_error_lines=[],
                schedule=SCHEDULE, warmup_count=2, measured_summary=summary, SDK_calls=sdk_count,
                saved_arrays_checked=array_count, artifacts=artifacts, independent_comparisons=comparisons,
                variant_comparison=variant, all_outputs_finite_nonzero=True,
                ordered_sum_once_bias_byte_exact=True, LS2_residual_byte_exact=True,
                repeats_byte_exact_across_normal_and_profile=True,
                context_count=4, all_contexts_destroyed=True, all_output_buffers_released=True,
                sum_context_DMA_bytes=run['sum_context_dma_bytes'], memory_final=run['memory_final'],
                profile_operator_coverage=coverage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('r1', 'r2'), default='r2')
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--ort', type=Path, required=True)
    parser.add_argument('--normal', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--normal-log', type=Path)
    parser.add_argument('--profile-log', type=Path)
    parser.add_argument('--check-onnx', action='store_true', help='Static graph/initializer parsing only; requires ONNX')
    parser.add_argument('--output', type=Path, required=True, help='New audit JSON file; never overwrite raw artifacts')
    args = parser.parse_args(argv)
    roots = [getattr(args, name).resolve(strict=True) for name in ('artifacts_root', 'capture', 'ort', 'normal', 'profile')]
    for name, root in zip(('artifacts_root', 'capture', 'ort', 'normal', 'profile'), roots):
        setattr(args, name, root)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parent) or any(output.is_relative_to(r) for r in roots):
        parser.error('Output must be a new file outside raw input/source directories')
    normal_log = args.normal_log or Path(str(args.normal) + '.log')
    profile_log = args.profile_log or Path(str(args.profile) + '.log')
    if output in (normal_log.resolve(), profile_log.resolve()):
        parser.error('Output must not overwrite a raw log')
    checks = Checks()
    report = dict(kind='independent_v23_saved_MLP_audit', variant=args.variant, status='failed',
                  script_sha256=sha(__file__), audit_consistency_passed=False,
                  host_model_ORT_or_SDK_execution=False, board_jobs_started=False,
                  production_changed=False, full_model_run=False, realtime_validated=False, quality_validated=False)
    try:
        capture, data, graphs, refs, ort = load_inputs(args, checks)
        stable = {}
        normal = audit_run(args.normal, normal_log, False, args, capture, data, graphs, refs, stable, checks)
        profiled = audit_run(args.profile, profile_log, True, args, capture, data, graphs, refs, stable, checks)
        strict_gate = all(normal['independent_comparisons'][key]['elementwise_gate_passed'] for key in
                          ('A_mlp_vs_original', 'B_mlp_vs_original', 'A_block_vs_original', 'B_block_vs_original'))
        all_npu_internal = all(not c['internal_CPU_types'] for c in profiled['profile_operator_coverage'])
        report.update(status='consistent_evidence_FP32_gate_passed' if strict_gate else 'consistent_evidence_FP32_gate_failed',
                      audit_consistency_passed=True, capture_receipt_sha256=CAPTURE_SHA,
                      input_shape=list(SHAPE), input_raw_sha256=array_record(data['norm2'])['sha256'],
                      graphs=graphs, ort=ort, runs=dict(normal=normal, profile=profiled),
                      internal_ops_all_NPU=all_npu_internal,
                      execution_type='NPU_internal_graph_with_CPU_IO_and_FP32_reduction' if all_npu_internal else 'mixed_CPU_NPU',
                      original_FP32_gate_passed=strict_gate, maximum_persistent_contexts=4,
                      SDK_calls_checked=normal['SDK_calls'] + profiled['SDK_calls'],
                      saved_arrays_checked=normal['saved_arrays_checked'] + profiled['saved_arrays_checked'],
                      checks=checks.count, failures=0,
                      source_scope='Array/export/model files independently rehashed. Historical source/image/checkpoint consistency is receipt evidence, not a current host rehash.',
                      timing_scope='Copy + submit/join + synchronous SDK IO + ordered FP32 sum + one FC2 bias + crop + LS2/residual; excludes init/metrics/hash/file IO. Profile runs separate.',
                      limitations=['One fixed original global0 MLP input. No full CameraHead/model/SLAM throughput claim.',
                                   'Finite repeatable output is not a model-quality gate; strict original FP32 gate reported separately.',
                                   'Hidden partition preserves real-arithmetic formula but changes floating-point reduction.',
                                   'SDK memory queries are not a universal process RSS lower bound.'])
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}', checks=checks.count, failures=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], audit_consistency_passed=report['audit_consistency_passed'],
                          output=str(output), sha256=sha(output), checks=report['checks'])))
    return 0 if report['audit_consistency_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
