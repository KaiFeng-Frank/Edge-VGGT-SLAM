#!/usr/bin/env python3
"""Board-only complete M48 grouped Update Torch/ONNX/CPU ORT diagnostic.

Derived topology/math: Princeton-VL/DPVO 0ac95b656d1fda91c271d2a106460d19ad966fc7,
MIT Copyright (c) 2022 Princeton University; see adjacent LICENSE. Frozen CF
arithmetic and all original weights are reused. No SDK, BA, VO or adoption here.
The original full-Update gold is the actual once-called capture, not a replay of
its output as a model. Equal-degree sources use their TRUE batch size, no padding.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2/experiments')
CAPTURE_HELPER_SHA = '51b9d830762c1c1f701ca38a13378b874240a04f3d7e98181fa397ae1aa8a5d2'
CAPTURE_AUDITOR_SHA = '29041c24c5eb39d46f1f6bd8b2a55d924ba5cb5f8fc54eb3bea752f8505278fa'
PARENT_SHA = '84b8f02e14c0695c79d5c47319ff47ec4c7a8dcf53a3b38d905e03bc1efde6a8'
CHECKPOINT_SHA = '30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
UPDATE_PINS = {
    'source_frame_batches.py': '0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1',
    'source_frame_plan.py': '04e240c2c35bd1359dbe5649fb038490a0a7ff99654d8d112ad7c83acc87ac2e',
    'source_frame_update_channel_first.py': '7d89d91c64158d34f72b17091ebeaf06369ef71cb330495f6c3ec39092035c57',
}
NEURAL_PINS = {
    'dpvo_probe_common.py': 'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9',
    'dpvo_cpu_ops.py': '15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32',
    'manifest.json': 'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375',
}
INPUTS = ('net', 'inp', 'corr')
OUTPUTS = ('net_out', 'delta', 'weight')
CANONICAL_OUTPUTS = ('net', 'delta', 'weight')
OBSERVATIONS = (13, 127, 191)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def board_guard():
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Real execution is RK3588-board only')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588 device tree required')


def compare(actual, reference):
    require(actual.dtype == reference.dtype == np.float32 and actual.shape == reference.shape,
            'Same FP32 shape required for fixed comparison')
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        return dict(finite=False, gate_passed=False, failed_elements=None, max_abs=None,
                    relative_L2=None, bitwise_equal=False, rtol=1e-4, atol=1e-5)
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = a-b
    failed = int(np.count_nonzero(np.abs(error) > 1e-5+1e-4*np.abs(b)))
    norm = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, gate_passed=failed == 0, failed_elements=failed,
                max_abs=float(np.abs(error).max(initial=0)),
                relative_L2=float(np.linalg.norm(error.ravel())/norm) if norm else None,
                bitwise_equal=actual.tobytes() == reference.tobytes(), rtol=1e-4, atol=1e-5)


def model_key(batch):
    return 'B%d_D%d_M%d' % (len(batch.source_slots), batch.degree, batch.patches)


def contracts(batch):
    return ([dict(name=name, dtype='float32', shape=list(batch.shape(c)))
             for name, c in zip(INPUTS, (384, 384, 882))],
            [dict(name=name, dtype='float32', shape=list(batch.shape(c)))
             for name, c in zip(OUTPUTS, (384, 2, 2))])


def grouped_run(adapter, plan, arrays, execute, identity):
    """One full staged update. Collect references only; hashes/save are outside.

    Outer time includes input finite validation, immutable snapshot copy,
    packing, every callback/owned output copy and restore/validation. Callback
    intervals exclude fullgraph packing/scatter and are not the outer total.
    """
    calls = []
    def callback(batch, feeds):
        begin = time.perf_counter()
        values = execute(batch, feeds)
        elapsed = time.perf_counter()-begin
        calls.append(dict(batch=batch, feeds=feeds, outputs=values, callback_seconds=elapsed))
        return values
    begin = time.perf_counter()
    outputs = adapter.execute_grouped_update(plan, *(arrays[k] for k in INPUTS), callback,
                  snapshot_identity=identity, current_identity=lambda: identity)
    return outputs, calls, time.perf_counter()-begin


def validate_full_outputs(outputs, edges):
    require(set(outputs) == set(CANONICAL_OUTPUTS), 'Three fullgraph canonical outputs required')
    finite = True
    for name, channels in zip(CANONICAL_OUTPUTS, (384, 2, 2)):
        value = outputs[name]
        require(value.dtype == np.float32 and value.shape == (1, edges, channels), 'Fullgraph output shape/dtype: '+name)
        finite = finite and bool(np.isfinite(value).all())
    weight = outputs['weight']
    return dict(finite=finite, weight_domain_valid=bool(np.isfinite(weight).all()
                and np.all((weight >= 0) & (weight <= 1))))


def initializer_coverage(graph, parameters, onnx):
    """Only original FP32 parameter bytes with exact reshape transforms."""
    choices = {}
    for name, value in parameters.items():
        variants = [('identity', value)]
        if value.ndim == 2:
            variants.append(('conv1x1_reshape', value.reshape(*value.shape, 1, 1)))
        if value.ndim == 1:
            variants.append(('channel_affine_reshape', value.reshape(1, -1, 1, 1)))
        for transform, array in variants:
            key = (tuple(array.shape), str(array.dtype), meta(array)['tensor_sha256'])
            choices.setdefault(key, []).append(dict(parameter=name, transform=transform))
    covered, mapping = set(), []
    for tensor in graph.graph.initializer:
        require(not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL,
                'External parameter data is prohibited')
        value = onnx.numpy_helper.to_array(tensor)
        matches = choices.get((tuple(value.shape), str(value.dtype), meta(value)['tensor_sha256']), [])
        require(bool(matches), 'Initializer not bound to original parameter: '+tensor.name)
        covered.update(match['parameter'] for match in matches)
        mapping.append(dict(initializer=tensor.name, array=meta(value), original_matches=matches))
    require(covered == set(parameters) and len(parameters) == 50, 'All original 50 parameters must be covered')
    return dict(original_parameter_count=50, all_original_parameters_covered=True, mapping=mapping)


def graph_contract(value):
    require(value.type.tensor_type.elem_type == 1, 'Fixed FP32 ONNX IO required')
    dims = value.type.tensor_type.shape.dim
    require(all(dim.HasField('dim_value') and dim.dim_value > 0 for dim in dims), 'No symbolic/padded ONNX dimension')
    return dict(name=value.name, dtype='float32', shape=[dim.dim_value for dim in dims])


def validate_receipts(capture, parent, audit, *, capture_sha, parent_sha, manifest_sha):
    """Pure metadata validation, usable without reading a model or real pixels."""
    require(capture['schema'] == 'v36_full_M48_original_Update_capture'
            and capture['status'] == 'completed_original_M48_capture_not_performance_or_candidate'
            and capture['script_sha256'] == CAPTURE_HELPER_SHA and capture['parent_runner_sha256'] == PARENT_SHA,
            'Completed frozen original M48 capture required')
    require(capture['capture_observations'] == list(OBSERVATIONS) and capture['captured'] == 3
            and [case['observation_id'] for case in capture['records']] == list(OBSERVATIONS),
            'All three predeclared observations required')
    require(capture['cleanup_errors'] == [] and capture['original_loader_and_stream_class_restored'] is True
            and capture['parent_runner_source_unchanged'] is True
            and capture['original_Update_replaced'] is False and capture['original_weights_or_RNG_changed'] is False,
            'Capture restoration/source/original callback gate failed')
    require(audit['schema'] == 'v36_saved_M48_capture_independent_audit'
            and audit['status'] == 'passed_saved_capture_identity_not_neural_equivalence'
            and audit['auditor_sha256'] == CAPTURE_AUDITOR_SHA
            and audit['input_bindings']['capture_sha256'] == capture_sha
            and audit['input_bindings']['parent_run_sha256'] == parent_sha
            and audit['input_bindings']['parent_input_manifest_sha256'] == manifest_sha
            and audit['input_bindings']['adapter_sha256'] == UPDATE_PINS['source_frame_batches.py']
            and bool(audit['checks']) and all(check['passed'] for check in audit['checks']),
            'Passed independent capture audit required')
    require(parent['schema'] == 'v35_DPVO_original_FAST_dense_spatial_RKNN_encoder_diagnostic_replay'
            and parent['status'].startswith('completed_') and parent['script_sha256'] == PARENT_SHA
            and parent['cleanup_errors'] == [] and parent['original_weights_unchanged'] is True
            and parent['config']['patches_per_frame'] == 48
            and parent['original_weights_before'] == parent['original_weights_after']
            and parent['sources_and_inputs_unchanged'] is True and parent['source_pins'] == parent['source_hashes_after'],
            'Original full Update parent weight/source chain differs')
    require(capture['original_parent_receipt']['run_sha256'] == parent_sha
            and capture['original_parent_receipt']['input_manifest_sha256'] == parent['input_manifest_sha256'] == manifest_sha
            and capture['original_call_attempts'] == capture['original_call_successes'] == len(parent['update_calls']),
            'Original parent manifest/callback-count binding differs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('capture-dir', 'parent-run-dir', 'capture-audit', 'update-source-dir', 'v26-source-dir',
                 'checkpoint', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('capture-sha256', 'capture-audit-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    board_guard()  # Before output, receipt/array reads, model/Torch/ONNX/ORT import.
    out = args.output_dir.resolve()
    require(out.is_relative_to(BASE) and out != BASE, 'New isolated board experiment output required')
    protected = [value.resolve() for key, value in vars(args).items() if key != 'output_dir' and isinstance(value, Path)]
    require(not any(out == path or out.is_relative_to(path) or path.is_relative_to(out) for path in protected),
            'Output must not overlap source/reference/checkpoint')
    out.mkdir(exist_ok=False)
    started, prior = time.monotonic(), os.sched_getaffinity(0)
    pins, sessions = {}, {}
    neural, wrapped, before, torch_calls = None, None, None, []
    report = dict(schema='v36_full_M48_grouped_Update_CPU_ONNX_diagnostic', status='failed_execution',
        script_sha256=sha(__file__), capture_sha256=args.capture_sha256, capture_audit_sha256=args.capture_audit_sha256,
        candidate_accepted=False, actual_board_execution=True, host_model_inference=False,
        original_VGGT_default_unchanged=True, different_model_from_VGGT=True, NPU_used=False,
        BA_executed=False, VO_state_committed=False, realtime_validated=False, ORB_LK_PnP_used=False,
        original_full_Update_reexecuted=False, source_member_padding=False, neighbor_or_edge_removal=False,
        terminal_sigmoid='stable', strict_FP32_thresholds=dict(rtol=1e-4, atol=1e-5),
        strict_FP32_thresholds_relaxed=False, output_clipping=False, models={}, cases=[], cleanup_errors=[])
    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, 'Pinned source/input SHA differs: '+str(path))
        pins[str(path)] = expected
        return path
    def saved(item):
        require(Path(item['file']).name == item['file'] and not (args.capture_dir/item['file']).is_symlink(),
                'Safe original artifact basename required')
        path = pin(args.capture_dir/item['file'], item['sha256'])
        require(path.stat().st_size == item['bytes'], 'Original artifact byte count differs')
        with np.load(path, allow_pickle=False) as archive:
            values = {key: np.array(archive[key], copy=True, order='C') for key in archive.files}
        require({key: meta(value) for key, value in values.items()} == item['arrays'], 'Original artifact raw SHA/shape differs')
        require(all(np.isfinite(value).all() for value in values.values()), 'Finite original capture required')
        return values
    def save(filename, values):
        path = out/filename
        with path.open('xb') as stream:
            np.savez(stream, **values)
        return dict(file=filename, bytes=path.stat().st_size, sha256=sha(path), arrays={key: meta(value) for key, value in values.items()})
    try:
        pin(__file__, report['script_sha256'])
        capture = json.loads(pin(args.capture_dir/'capture.json', args.capture_sha256).read_text())
        audit = json.loads(pin(args.capture_audit, args.capture_audit_sha256).read_text())
        parent_file = args.parent_run_dir/'run.json'
        parent = json.loads(pin(parent_file, capture['original_parent_receipt']['run_sha256']).read_text())
        manifest_file = pin(args.parent_run_dir/'input-manifest.json', capture['original_parent_receipt']['input_manifest_sha256'])
        validate_receipts(capture, parent, audit, capture_sha=args.capture_sha256,
                          parent_sha=sha(parent_file), manifest_sha=sha(manifest_file))
        require(parent_file.stat().st_size == capture['original_parent_receipt']['run_bytes'], 'Original parent receipt byte count differs')
        for path, digest in parent['source_pins'].items():
            pin(path, digest)  # Board-only source rehash, never required by host static tests.
        for name, digest in UPDATE_PINS.items():
            pin(args.update_source_dir/name, digest)
        for name, digest in NEURAL_PINS.items():
            pin(args.v26_source_dir/name, digest)
        pin(args.checkpoint, CHECKPOINT_SHA)
        require({4, 5, 6, 7} <= prior, 'Four A76 CPUs required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import torch
        import onnx
        import onnxruntime as ort
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        adapter = module(args.update_source_dir/'source_frame_batches.py', '_v36_grouped_original_plan')
        cf = module(args.update_source_dir/'source_frame_update_channel_first.py', '_v36_grouped_frozen_CF')
        common = module(args.v26_source_dir/'dpvo_probe_common.py', '_v36_grouped_original_loader')
        neural, loader = common.load_neural(args.v26_source_dir/'official', args.v26_source_dir/'manifest.json',
                          args.checkpoint, CHECKPOINT_SHA, args.v26_source_dir/'dpvo_cpu_ops.py',
                          cpu_ops_sha256=NEURAL_PINS['dpvo_cpu_ops.py'])
        for name, value in loader['source_hashes'].items():
            pin(args.v26_source_dir/'official'/name, value['sha256'])
        before = {key: meta(value.detach().numpy()) for key, value in neural.state_dict().items()}
        require(len(before) == 94 and before == parent['original_weights_before'], 'Original 94 complete weights differ')
        parameters = {key: value.detach().numpy() for key, value in neural.update.state_dict().items()}
        param_meta = {key: meta(value) for key, value in parameters.items()}
        require(len(parameters) == 50 and {'update.'+key: value for key, value in param_meta.items()}
                == {key: value for key, value in before.items() if key.startswith('update.')}, 'Original Update 50 raw parameters differ')
        wrapped = cf.SourceFrameUpdateChannelFirst(neural.update, terminal_sigmoid='stable').eval()
        hook = wrapped.register_forward_hook(lambda _m, _a, _o: torch_calls.append(1))
        report.update(original_weights_before=before, original_Update_parameters=param_meta,
                      original_binding=wrapped.verify_original_binding(neural.update), loader=loader,
                      cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=4, torch_interop_threads=1,
                      torch_version=torch.__version__, onnx_version=onnx.__version__, ORT_version=ort.__version__,
                      input_layout='true_source_batch,channels,sorted_target,patch', patches_per_frame=48,
                      OMP_WAIT_POLICY=os.environ.get('OMP_WAIT_POLICY'), GOMP_SPINCOUNT=os.environ.get('GOMP_SPINCOUNT'))
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
            options.add_session_config_entry(key, '0')
        gates, ordinary_calls, trace_calls = [], 0, 0
        for record in capture['records']:
            observation, edges = record['observation_id'], record['edges']
            require(record['phase'] == 'online' and record['step_within_observation'] == 0
                    and record['before_BA'] is True and record['original_called_once'] is True
                    and record['callback_outputs_returned_unmodified'] is True and record['patches_per_frame'] == 48,
                    'Actual selected original once-called pre-BA gold required')
            arrays, gold = saved(record['inputs_artifact']), saved(record['original_gold_artifact'])
            require(validate_full_outputs(gold, edges) == dict(finite=True, weight_domain_valid=True), 'Original complete gold domain')
            call_index = record['actual_original_call_index']
            require(parent['update_calls'][call_index]['index'] == call_index
                    and parent['update_calls'][call_index]['edges'] == edges, 'Gold original callback identity differs')
            input_meta = {key: meta(arrays[key]) for key in INPUTS}
            plan = adapter.build_degree_plan(*(arrays[key] for key in ('ii', 'jj', 'kk')), patches_per_frame=48)
            captured_identity = record['snapshot_identity']
            identity = adapter.SnapshotIdentity(captured_identity['graph']['revision'], captured_identity['geometry']['revision'],
                                               captured_identity['features']['revision'], captured_identity['recurrent_revision'])
            case = dict(observation_id=observation, edges=edges, snapshot_identity=captured_identity,
                        original_call_index=call_index, original_inputs_artifact=record['inputs_artifact'],
                        original_gold_artifact=record['original_gold_artifact'], original_callback_seconds=record['original_callback_seconds'],
                        degree_plan=plan.metadata(), groups=[])
            report['cases'].append(case)
            def execute_torch(batch, feeds):
                arguments = tuple(torch.from_numpy(feeds[key]) for key in INPUTS)
                wrapped.validate_inputs(*arguments)
                with torch.no_grad():
                    values = wrapped(*arguments)
                return {key: value.detach().numpy().copy(order='C') for key, value in zip(CANONICAL_OUTPUTS, values)}
            torch_full, torch_records, torch_seconds = grouped_run(adapter, plan, arrays, execute_torch, identity)
            ordinary_calls += len(torch_records)
            case['grouped_Torch_total_seconds'] = torch_seconds
            case['grouped_Torch_artifact'] = save('obs-%d-grouped-Torch.npz' % observation, torch_full)
            case['Torch_vs_original_full_gold'] = {key: compare(torch_full[key], gold[key]) for key in CANONICAL_OUTPUTS}
            gates.extend(item['gate_passed'] for item in case['Torch_vs_original_full_gold'].values())
            for group_index, entry in enumerate(torch_records):
                batch, feeds = entry['batch'], entry['feeds']
                key = model_key(batch)
                group = dict(index=group_index, model_key=key, topology=batch.metadata(),
                             canonical_edge_indices=meta(batch.canonical_edge_indices),
                             inputs_artifact=save('obs-%d-group-%02d-inputs.npz' % (observation, group_index), feeds),
                             Torch_callback_seconds=entry['callback_seconds'])
                case['groups'].append(group)
                if key not in report['models']:
                    path = out/(key+'.onnx')
                    arguments = tuple(torch.from_numpy(feeds[name]) for name in INPUTS)
                    export_options = dict(input_names=list(INPUTS), output_names=list(OUTPUTS), opset_version=17, do_constant_folding=True)
                    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
                        export_options['dynamo'] = False
                    before_trace_count, begin = len(torch_calls), time.perf_counter()
                    with torch.no_grad():
                        torch.onnx.export(wrapped, arguments, str(path), **export_options)
                    export_seconds = time.perf_counter()-begin
                    actual_trace_calls = len(torch_calls)-before_trace_count
                    trace_calls += actual_trace_calls
                    graph = onnx.load(path, load_external_data=False)
                    onnx.checker.check_model(graph, full_check=True)
                    counts = Counter(node.op_type for node in graph.graph.node)
                    wanted_inputs, wanted_outputs = contracts(batch)
                    require([graph_contract(value) for value in graph.graph.input] == wanted_inputs
                            and [graph_contract(value) for value in graph.graph.output] == wanted_outputs, 'Exact true-B fixed ONNX contract differs')
                    axes = [onnx.helper.get_attribute_value(attribute) for node in graph.graph.node if node.op_type == 'Softmax'
                            for attribute in node.attribute if attribute.name == 'axis']
                    require(counts['Conv'] == 21 and counts['Softmax'] == 2 and sorted(axes) == [2, 3]
                            and not {'Transpose', 'Gather', 'GatherElements', 'GatherND', 'Where', 'Pad', 'Gemm', 'MatMul'} & set(counts),
                            'Unchanged original CF dependency graph required')
                    report['models'][key] = dict(model=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)),
                        inputs=wanted_inputs, outputs=wanted_outputs, batch=len(batch.source_slots), degree=batch.degree, patches=48,
                        example_observation_id=observation, example_group_index=group_index,
                        parameter_coverage=initializer_coverage(graph, parameters, onnx), ONNX_operator_counts=dict(counts),
                        export_seconds=export_seconds, actual_export_trace_forward_calls=actual_trace_calls,
                        external_data_used=False, padding=False, numerical_validation_completed=False)
                    begin = time.perf_counter()
                    sessions[key] = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                    session = sessions[key]
                    require(session.get_providers() == ['CPUExecutionProvider'], 'CPU ORT provider only')
                    for actual, wanted in ((session.get_inputs(), wanted_inputs), (session.get_outputs(), wanted_outputs)):
                        require([dict(name=value.name, dtype='float32' if value.type == 'tensor(float)' else value.type,
                                      shape=list(value.shape)) for value in actual] == wanted, 'ORT actual IO contract differs')
                    report['models'][key]['ORT_session_seconds'] = time.perf_counter()-begin
                # No model/weight/input values may mutate during trace.
                wrapped.verify_original_binding(neural.update)
            def execute_ort(batch, feeds):
                values = sessions[model_key(batch)].run(list(OUTPUTS), feeds)
                return {key: np.array(value, copy=True, order='C') for key, value in zip(CANONICAL_OUTPUTS, values)}
            ort_full, ort_records, ort_seconds = grouped_run(adapter, plan, arrays, execute_ort, identity)
            case['grouped_ORT_total_seconds'] = ort_seconds
            case['grouped_ORT_artifact'] = save('obs-%d-grouped-ORT.npz' % observation, ort_full)
            case['ORT_vs_original_full_gold'] = {key: compare(ort_full[key], gold[key]) for key in CANONICAL_OUTPUTS}
            case['ORT_vs_grouped_Torch'] = {key: compare(ort_full[key], torch_full[key]) for key in CANONICAL_OUTPUTS}
            gates.extend(item['gate_passed'] for comparison in (case['ORT_vs_original_full_gold'], case['ORT_vs_grouped_Torch'])
                         for item in comparison.values())
            for group, t_entry, o_entry in zip(case['groups'], torch_records, ort_records):
                batch, values = o_entry['batch'], o_entry['outputs']
                group['ORT_callback_seconds'] = o_entry['callback_seconds']
                local_gold = {key: adapter.pack_batch(gold[key], batch) for key in CANONICAL_OUTPUTS}
                group['Torch_vs_original_group_gold'] = {key: compare(t_entry['outputs'][key], local_gold[key]) for key in CANONICAL_OUTPUTS}
                group['ORT_vs_original_group_gold'] = {key: compare(values[key], local_gold[key]) for key in CANONICAL_OUTPUTS}
                group['ORT_vs_grouped_Torch'] = {key: compare(values[key], t_entry['outputs'][key]) for key in CANONICAL_OUTPUTS}
                require({key: meta(o_entry['feeds'][key]) for key in INPUTS} == group['inputs_artifact']['arrays'],
                        'Torch and ORT do not share identical group inputs')
            case['Torch_domain'], case['ORT_domain'] = validate_full_outputs(torch_full, edges), validate_full_outputs(ort_full, edges)
            require(case['Torch_domain'] == case['ORT_domain'] == dict(finite=True, weight_domain_valid=True), 'Finite/domain grouped outputs required')
            require({key: meta(arrays[key]) for key in INPUTS} == input_meta, 'Original canonical inputs mutated')
            wrapped.verify_original_binding(neural.update)
            print(json.dumps(dict(observation_id=observation, groups=len(plan.batches), true_batch_shapes=[list(batch.shape(384)) for batch in plan.batches],
                                  strict_gates_so_far=all(gates))), flush=True)
            del arrays, gold, torch_full, ort_full, torch_records, ort_records
        report.update(status='completed_grouped_Update_numerical_accepted_not_adopted' if all(gates) else 'completed_grouped_Update_numerical_rejected',
                      strict_FP32_gate_passed=all(gates), strict_fullgraph_comparison_count=len(gates),
                      outputs_finite=True, weight_domain_valid=True, ordinary_grouped_Torch_forward_calls=ordinary_calls,
                      actual_export_trace_forward_calls=trace_calls, actual_total_Torch_wrapper_forward_calls=len(torch_calls),
                      grouped_ORT_calls=sum(len(case['groups']) for case in report['cases']),
                      ORT_graph_optimization='ORT_DISABLE_ALL', original_parameters_unchanged=False,
                      numerical_gate_scope='All three complete captured graphs restored to original E order; Torch vs original gold and ORT vs both gold/Torch, plus every true source-group comparison.',
                      timing_scope='One call per method/group. Complete grouped time includes immutable full-input snapshot, finite scan, packing, callback validation/forward/owned copy and scatter. Group callbacks are nested. Model load, export traces, ORT session construction, hashes, save and error statistics excluded.',
                      limitations=['Only once-called gold at observations 13/127/191 before BA; no original full Update or encoder rerun.',
                                   'CF Conv, channel norm and stable terminal Sigmoid preserve original real arithmetic/50 parameters but can change FP32 rounding.',
                                   'No SDK B>1, placement, quantization, recurrent VO submission, BA, trajectory, sustained throughput or 20Hz acceptance.'])
        for model in report['models'].values():
            model['numerical_validation_completed'] = True
    except Exception as error:
        report.update(status='failed_execution', error=repr(error), traceback=traceback.format_exc())
    finally:
        if 'hook' in locals():
            try:
                hook.remove()
            except Exception as error:
                report['cleanup_errors'].append(repr(error))
        sessions.clear()
        session = None
        report['ORT_session_references_released'] = True
        if neural is not None and before is not None:
            try:
                after = {key: meta(value.detach().numpy()) for key, value in neural.state_dict().items()}
                report['original_weights_after'] = after
                report['original_parameters_unchanged'] = after == before
                require(after == before, 'Original parameters changed')
                if wrapped is not None:
                    wrapped.verify_original_binding(neural.update)
            except Exception as error:
                report['cleanup_errors'].append(repr(error))
        try:
            report.update(source_pins=pins, source_hashes_after={path: sha(path) for path in pins})
            report['sources_and_inputs_unchanged'] = report['source_pins'] == report['source_hashes_after']
            require(report['sources_and_inputs_unchanged'], 'Pinned source/input changed')
        except Exception as error:
            report['cleanup_errors'].append(repr(error))
        try:
            os.sched_setaffinity(0, prior)
            report['affinity_restored'] = os.sched_getaffinity(0) == prior
            require(report['affinity_restored'], 'CPU affinity restoration failed')
        except Exception as error:
            report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']:
            report['status'] = 'failed_execution'
        report.update(actual_driver_seconds=time.monotonic()-started,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'strict_FP32_gate_passed', 'outputs_finite', 'weight_domain_valid', 'error', 'cleanup_errors')}), flush=True)
    return 0 if report['status'] == 'completed_grouped_Update_numerical_accepted_not_adopted' else 2 if report['status'] == 'completed_grouped_Update_numerical_rejected' else 1


if __name__ == '__main__':
    raise SystemExit(main())
