#!/usr/bin/env python3
"""Board-only original M16 complete-source Update Torch/ONNX/CPU ORT research.

Reuse Princeton-VL/DPVO 0ac95b656d1fda91c271d2a106460d19ad966fc7,
MIT Copyright (c) 2022 Princeton University, and the frozen v37 CF adapter.
All 50 original Update parameters and every captured edge are retained.
This exports only actual equal-degree TRUE source batches; no padding, BA,
VO state submission, NPU execution, default replacement or adoption occurs.
Pure load_capture/validate_run functions do not import Torch/ONNX/ORT.
"""
import sys
sys.dont_write_bytecode = True

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
import time
import traceback

import numpy as np

BASE = Path('/mnt/vggt-data/vggt-slam2/experiments')
PARENT_NAME = 'export-v36-grouped-update-board.py'
PARENT_SHA256 = '473bd9441c983c08aa3ab3a97a7f14b82da8f3d508690403b08468ba25d2dc72'
RUN_SHA256 = 'dd9f37524b999c1e586182a92c226481fcfc7f4da851d5e88d62d82be1029ecd'
RUNNER_SHA256 = '562008c8650404d78c4974eed0bcfab347f7db6ff0671ac8b4e7f05fa6c92ba2'
POLICY_SHA256 = 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
INPUTS = ('net', 'inp', 'corr')
OUTPUTS = ('net_out', 'delta', 'weight')
CANONICAL_OUTPUTS = ('net', 'delta', 'weight')
PATCHES = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def utilities(update_source_dir):
    """Load only the exact frozen utility implementation; its main is not used."""
    path = Path(update_source_dir)/PARENT_NAME
    require(sha(path) == PARENT_SHA256, 'Frozen v37 parent exporter SHA differs')
    name = '_v39_M16_parent_' + PARENT_SHA256
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def validate_run(run):
    """Pure metadata gate; board absolute source pins are declarations here."""
    require(run['schema'] == 'v39_DPVO_half_original_Torch_CPU_research_replay'
            and run['status'] == 'completed_unpaced_half_Torch_CPU_research_not_realtime'
            and run['script_sha256'] == RUNNER_SHA256 and run['policy_sha256'] == POLICY_SHA256,
            'Frozen completed original M16 CPU run required')
    require(run['frames_completed'] == 192 and run['seed'] == 20261011
            and run['diagnostic_capture_enabled'] is True and run['cleanup_errors'] == []
            and run['affinity_restored'] is True and run['original_weights_unchanged'] is True
            and run['original_weights_before'] == run['original_weights_after']
            and len(run['original_weights_before']) == 94
            and run['sources_and_inputs_unchanged'] is True
            and run['source_pins'] and run['source_pins'] == run['source_hashes_after'],
            'Complete original weight/source/capture lifecycle differs')
    wanted = dict(patches_per_frame=16, removal_window=8, optimization_window=4,
                  patch_lifetime=4, keyframe_index=2, capacity=64,
                  loop_closure=False, classic_loop_closure=False)
    require(all(run['config'].get(key) == value for key, value in wanted.items()),
            'Frozen M16/L4/R8/O4/KF2 configuration differs')
    captures, calls = run['update_captures'], run['update_calls']
    full = [row for row in calls if row['stage'] == 'stream' and row['edges'] > PATCHES]
    require(full and len(captures) == 3, 'All three predeclared complete graphs required')
    observations = [full[0]['frame_index'], 127, 191]
    require(observations[0] < 127 and [row['frame_index'] for row in captures] == observations,
            'First actual full initialization plus observations 127/191 required')
    for record in captures:
        index, edges = record['update_index'], record['edges']
        require(type(index) is int and 0 <= index < len(calls) and type(edges) is int
                and edges > PATCHES and edges % PATCHES == 0
                and record['entire_active_graph'] is True and record['same_call_original_Torch'] is True,
                'Complete same-call original Torch capture required')
        call = calls[index]
        selected = next(row for row in full if row['frame_index'] == record['frame_index'])
        require(call['index'] == selected['index'] == index and call['stage'] == 'stream'
                and call['frame_index'] == record['frame_index'] and call['edges'] == edges,
                'Capture must be the first full Update at its original observation')
    return captures


def read_artifact(directory, item):
    filename = item['file']
    require(type(filename) is str and Path(filename).name == filename
            and filename not in ('', '.', '..'), 'Safe original artifact basename required')
    path = Path(directory)/filename
    require(not path.is_symlink() and path.is_file() and sha(path) == item['sha256']
            and path.stat().st_size == item['bytes'], 'Original artifact file SHA/bytes differ')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True, order='C') for key in archive.files}
    require({key: meta(value) for key, value in arrays.items()} == item['arrays'],
            'Original artifact raw SHA/dtype/shape differs')
    require(all(np.isfinite(value).all() for value in arrays.values()), 'Finite captured values required')
    return arrays


def validate_arrays(arrays, gold, edges):
    require(set(arrays) == set(INPUTS + ('ii', 'jj', 'kk')) and set(gold) == set(CANONICAL_OUTPUTS),
            'Exact original full input/gold keys required')
    for name, channels in zip(INPUTS, (384, 384, 882)):
        value = arrays[name]
        require(value.dtype == np.float32 and value.shape == (1, edges, channels)
                and value.flags.c_contiguous and np.isfinite(value).all(), 'Original input FP32 shape/finite: '+name)
    for name in ('ii', 'jj', 'kk'):
        value = arrays[name]
        limit = 64*PATCHES if name == 'kk' else 64
        require(value.dtype == np.int64 and value.shape == (edges,)
                and np.all((value >= 0) & (value < limit)), 'Original topology int64/range: '+name)
    require(np.array_equal(arrays['ii'], arrays['kk']//PATCHES), 'Original ii == kk//M16 identity differs')
    for name, channels in zip(CANONICAL_OUTPUTS, (384, 2, 2)):
        value = gold[name]
        require(value.dtype == np.float32 and value.shape == (1, edges, channels)
                and np.isfinite(value).all(), 'Original gold FP32 shape/finite: '+name)
    require(np.all((gold['weight'] >= 0) & (gold['weight'] <= 1)), 'Original gold weight domain differs')


def build_plan(adapter, arrays, call):
    plan = adapter.build_degree_plan(*(arrays[key] for key in ('ii', 'jj', 'kk')),
            patches_per_frame=PATCHES, frame_capacity=64, max_target_degree=7)
    degrees = {source: batch.degree for batch in plan.batches for source in batch.source_slots}
    require(sorted(degrees) == call['source_slots']
            and [degrees[key] for key in sorted(degrees)] == call['source_degrees']
            and [degrees[key]*PATCHES for key in sorted(degrees)] == call['source_patch_counts'],
            'Captured original source membership/degree differs from full graph')
    return plan


def load_capture(run_dir, expected_run_sha256, update_source_dir):
    """NumPy-only saved-input validator. Return run, records, pins.

    Each record has capture metadata, inputs, gold, and the actual degree plan.
    Source paths from the original board receipt are NOT dereferenced on host.
    The real board main rehashes all of them before and after execution.
    """
    root, source = Path(run_dir), Path(update_source_dir)
    run_file = root/'run.json'
    require(expected_run_sha256 == RUN_SHA256 and sha(run_file) == expected_run_sha256,
            'Exact original v39 run SHA required')
    run = json.loads(run_file.read_text())
    captures = validate_run(run)
    manifest = root/'input-manifest.json'
    require(sha(manifest) == run['input_manifest_sha256'], 'Original source input manifest SHA differs')
    parent = utilities(source)
    pins = {str(run_file.resolve()): expected_run_sha256,
            str(manifest.resolve()): run['input_manifest_sha256'],
            str((source/PARENT_NAME).resolve()): PARENT_SHA256}
    for name, digest in parent.UPDATE_PINS.items():
        path = source/name
        require(sha(path) == digest, 'Frozen grouping/CF source SHA differs: '+name)
        pins[str(path.resolve())] = digest
    adapter = parent.module(source/'source_frame_batches.py', '_v39_M16_frozen_degree_plan')
    records = []
    for captured in captures:
        arrays, gold = read_artifact(root, captured['input']), read_artifact(root, captured['gold'])
        validate_arrays(arrays, gold, captured['edges'])
        plan = build_plan(adapter, arrays, run['update_calls'][captured['update_index']])
        records.append(dict(capture=captured, inputs=arrays, gold=gold, plan=plan))
        for role in ('input', 'gold'):
            pins[str((root/captured[role]['file']).resolve())] = captured[role]['sha256']
    return run, records, pins


def board_guard():
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'RK3588 board execution only')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588 device tree required')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('run-dir', 'update-source-dir', 'v26-source-dir', 'checkpoint', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-sha256', default=RUN_SHA256)
    args = parser.parse_args(argv)
    board_guard()  # Before reading captures, making output or importing model/toolchain.
    out = args.output_dir.resolve()
    protected = [value.resolve() for key, value in vars(args).items() if key != 'output_dir' and isinstance(value, Path)]
    require(out != BASE and out.is_relative_to(BASE) and not any(out == path or out.is_relative_to(path)
            or path.is_relative_to(out) for path in protected), 'New non-overlapping board experiment directory required')
    out.mkdir(exist_ok=False)
    started, prior = time.monotonic(), os.sched_getaffinity(0)
    pins, sessions = {str(Path(__file__).resolve()): sha(__file__)}, {}
    neural, wrapped, weights_before, hook, session = None, None, None, None, None
    torch_calls = []
    report = dict(schema='v39_full_M16_grouped_Update_CPU_ONNX_diagnostic', status='failed_execution',
        script_sha256=sha(__file__), parent_exporter_sha256=PARENT_SHA256, original_run_sha256=args.run_sha256,
        policy_sha256=POLICY_SHA256, patches_per_frame=PATCHES, candidate_accepted=False,
        actual_board_execution=True, host_model_inference=False, original_VGGT_default_unchanged=True,
        different_model_from_VGGT=True, NPU_used=False, BA_executed=False, VO_state_committed=False,
        realtime_validated=False, ORB_LK_PnP_used=False, original_full_Update_reexecuted=False,
        source_member_padding=False, neighbor_or_edge_removal=False, terminal_sigmoid='stable',
        strict_FP32_thresholds=dict(rtol=1e-4, atol=1e-5), strict_FP32_thresholds_relaxed=False,
        output_clipping=False, original_parameters_unchanged=False, models={}, cases=[], cleanup_errors=[])

    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, 'Pinned board source/input SHA differs: '+str(path))
        pins[str(path)] = expected
        return path

    def save(filename, values):
        path = out/filename
        with path.open('xb') as stream:
            np.savez(stream, **values)
        return dict(file=filename, bytes=path.stat().st_size, sha256=sha(path),
                    arrays={key: meta(value) for key, value in values.items()})

    try:
        parent = utilities(args.update_source_dir)
        run, records, saved_pins = load_capture(args.run_dir, args.run_sha256, args.update_source_dir)
        pins.update(saved_pins)
        for path, digest in run['source_pins'].items():
            pin(path, digest)  # Actual board-only source provenance check, outside numerical timing.
        for name, digest in parent.NEURAL_PINS.items():
            pin(args.v26_source_dir/name, digest)
        pin(args.checkpoint, parent.CHECKPOINT_SHA)
        require({4, 5, 6, 7} <= prior, 'Four A76 CPUs required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import torch
        import onnx
        import onnxruntime as ort
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        adapter = parent.module(args.update_source_dir/'source_frame_batches.py', '_v39_M16_transport')
        cf = parent.module(args.update_source_dir/'source_frame_update_channel_first.py', '_v39_M16_CF')
        common = parent.module(args.v26_source_dir/'dpvo_probe_common.py', '_v39_M16_original_loader')
        neural, loader = common.load_neural(args.v26_source_dir/'official', args.v26_source_dir/'manifest.json',
            args.checkpoint, parent.CHECKPOINT_SHA, args.v26_source_dir/'dpvo_cpu_ops.py',
            cpu_ops_sha256=parent.NEURAL_PINS['dpvo_cpu_ops.py'])
        for name, value in loader['source_hashes'].items():
            pin(args.v26_source_dir/'official'/name, value['sha256'])
        weights_before = {key: meta(value.detach().numpy()) for key, value in neural.state_dict().items()}
        require(len(weights_before) == 94 and weights_before == run['original_weights_before'], 'Original complete 94 weights differ')
        parameters = {key: value.detach().numpy() for key, value in neural.update.state_dict().items()}
        parameter_meta = {key: meta(value) for key, value in parameters.items()}
        require(len(parameters) == 50 and {'update.'+key: value for key, value in parameter_meta.items()}
                == {key: value for key, value in weights_before.items() if key.startswith('update.')}, 'All original 50 Update weights required')
        wrapped = cf.SourceFrameUpdateChannelFirst(neural.update, terminal_sigmoid='stable').eval()
        hook = wrapped.register_forward_hook(lambda _m, _a, _o: torch_calls.append(1))
        report.update(input_manifest_sha256=run['input_manifest_sha256'], original_weights_before=weights_before,
            original_Update_parameters=parameter_meta, original_binding=wrapped.verify_original_binding(neural.update),
            loader=loader, cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=4, torch_interop_threads=1,
            torch_version=torch.__version__, onnx_version=onnx.__version__, ORT_version=ort.__version__,
            input_layout='true_source_batch,channels,sorted_target,patch',
            captured_observations=[row['capture']['frame_index'] for row in records],
            snapshot_identity_scope='Offline immutable saved input; update_index labels are transport-only versions, not captured online graph/geometry revisions.',
            OMP_WAIT_POLICY=os.environ.get('OMP_WAIT_POLICY'), GOMP_SPINCOUNT=os.environ.get('GOMP_SPINCOUNT'))
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
            options.add_session_config_entry(key, '0')
        gates, ordinary_calls, trace_calls = [], 0, 0
        for loaded in records:
            record, arrays, gold, plan = loaded['capture'], loaded['inputs'], loaded['gold'], loaded['plan']
            observation, edges, call_index = record['frame_index'], record['edges'], record['update_index']
            identity = adapter.SnapshotIdentity(call_index, call_index, call_index, call_index)
            original_meta = {key: meta(value) for key, value in arrays.items()}
            case = dict(observation_id=observation, frame_index=observation, edges=edges, original_call_index=call_index,
                original_inputs_artifact=record['input'], original_gold_artifact=record['gold'],
                original_callback_seconds=run['update_calls'][call_index]['seconds'], degree_plan=plan.metadata(),
                snapshot_identity=dict(saved_update_index=call_index, online_revisions_available=False), groups=[])
            report['cases'].append(case)

            def execute_torch(batch, feeds):
                arguments = tuple(torch.from_numpy(feeds[key]) for key in INPUTS)
                wrapped.validate_inputs(*arguments)
                with torch.no_grad():
                    values = wrapped(*arguments)
                return {key: value.detach().numpy().copy(order='C') for key, value in zip(CANONICAL_OUTPUTS, values)}

            torch_full, torch_records, torch_seconds = parent.grouped_run(adapter, plan, arrays, execute_torch, identity)
            ordinary_calls += len(torch_records)
            case.update(grouped_Torch_total_seconds=torch_seconds,
                grouped_Torch_artifact=save('obs-%d-grouped-Torch.npz' % observation, torch_full),
                Torch_vs_original_full_gold={key: parent.compare(torch_full[key], gold[key]) for key in CANONICAL_OUTPUTS})
            gates.extend(row['gate_passed'] for row in case['Torch_vs_original_full_gold'].values())
            for group_index, entry in enumerate(torch_records):
                batch, feeds = entry['batch'], entry['feeds']
                key = parent.model_key(batch)
                group = dict(index=group_index, model_key=key, topology=batch.metadata(),
                    canonical_edge_indices=meta(batch.canonical_edge_indices),
                    inputs_artifact=save('obs-%d-group-%02d-inputs.npz' % (observation, group_index), feeds),
                    Torch_callback_seconds=entry['callback_seconds'])
                case['groups'].append(group)
                if key not in report['models']:
                    path = out/(key+'.onnx')
                    wanted_inputs, wanted_outputs = parent.contracts(batch)
                    arguments = tuple(torch.from_numpy(feeds[name]) for name in INPUTS)
                    export_options = dict(input_names=list(INPUTS), output_names=list(OUTPUTS), opset_version=17, do_constant_folding=True)
                    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
                        export_options['dynamo'] = False
                    before_trace_count, begin = len(torch_calls), time.perf_counter()
                    with torch.no_grad():
                        torch.onnx.export(wrapped, arguments, str(path), **export_options)
                    export_seconds, actual_trace_calls = time.perf_counter()-begin, len(torch_calls)-before_trace_count
                    trace_calls += actual_trace_calls
                    graph = onnx.load(path, load_external_data=False)
                    onnx.checker.check_model(graph, full_check=True)
                    counts = Counter(node.op_type for node in graph.graph.node)
                    axes = [onnx.helper.get_attribute_value(a) for node in graph.graph.node if node.op_type == 'Softmax'
                            for a in node.attribute if a.name == 'axis']
                    require([parent.graph_contract(value) for value in graph.graph.input] == wanted_inputs
                            and [parent.graph_contract(value) for value in graph.graph.output] == wanted_outputs,
                            'Exact actual true-B/M16 ONNX contract differs')
                    require(counts['Conv'] == 21 and counts['Softmax'] == 2 and sorted(axes) == [2, 3]
                            and not {'Transpose', 'Gather', 'GatherElements', 'GatherND', 'Where', 'Pad', 'Gemm', 'MatMul'} & set(counts),
                            'Frozen complete-source CF dependency graph differs')
                    report['models'][key] = dict(model=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)),
                        inputs=wanted_inputs, outputs=wanted_outputs, batch=len(batch.source_slots), degree=batch.degree,
                        patches=PATCHES, example_observation_id=observation, example_group_index=group_index,
                        parameter_coverage=parent.initializer_coverage(graph, parameters, onnx), ONNX_operator_counts=dict(counts),
                        export_seconds=export_seconds, actual_export_trace_forward_calls=actual_trace_calls,
                        external_data_used=False, padding=False, numerical_validation_completed=False)
                    begin = time.perf_counter()
                    sessions[key] = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                    session = sessions[key]
                    require(session.get_providers() == ['CPUExecutionProvider'], 'Only CPU ORT allowed')
                    for actual, wanted in ((session.get_inputs(), wanted_inputs), (session.get_outputs(), wanted_outputs)):
                        require([dict(name=v.name, dtype='float32' if v.type == 'tensor(float)' else v.type,
                                      shape=list(v.shape)) for v in actual] == wanted, 'Actual ORT IO contract differs')
                    report['models'][key]['ORT_session_seconds'] = time.perf_counter()-begin
                wrapped.verify_original_binding(neural.update)

            def execute_ort(batch, feeds):
                values = sessions[parent.model_key(batch)].run(list(OUTPUTS), feeds)
                return {key: np.array(value, copy=True, order='C') for key, value in zip(CANONICAL_OUTPUTS, values)}

            ort_full, ort_records, ort_seconds = parent.grouped_run(adapter, plan, arrays, execute_ort, identity)
            case.update(grouped_ORT_total_seconds=ort_seconds,
                grouped_ORT_artifact=save('obs-%d-grouped-ORT.npz' % observation, ort_full),
                ORT_vs_original_full_gold={key: parent.compare(ort_full[key], gold[key]) for key in CANONICAL_OUTPUTS},
                ORT_vs_grouped_Torch={key: parent.compare(ort_full[key], torch_full[key]) for key in CANONICAL_OUTPUTS})
            gates.extend(row['gate_passed'] for comparison in (case['ORT_vs_original_full_gold'], case['ORT_vs_grouped_Torch'])
                         for row in comparison.values())
            require(len(torch_records) == len(ort_records) == len(case['groups']), 'Every complete source group required')
            for group, t_entry, o_entry in zip(case['groups'], torch_records, ort_records):
                batch = o_entry['batch']
                group['ORT_callback_seconds'] = o_entry['callback_seconds']
                require({key: meta(o_entry['feeds'][key]) for key in INPUTS} == group['inputs_artifact']['arrays'],
                        'Torch/ORT group inputs differ')
                local_gold = {key: adapter.pack_batch(gold[key], batch) for key in CANONICAL_OUTPUTS}
                for label, actual, reference in (
                        ('Torch_vs_original_group_gold', t_entry['outputs'], local_gold),
                        ('ORT_vs_original_group_gold', o_entry['outputs'], local_gold),
                        ('ORT_vs_grouped_Torch', o_entry['outputs'], t_entry['outputs'])):
                    group[label] = {key: parent.compare(actual[key], reference[key]) for key in CANONICAL_OUTPUTS}
                    gates.extend(row['gate_passed'] for row in group[label].values())
            case['Torch_domain'], case['ORT_domain'] = parent.validate_full_outputs(torch_full, edges), parent.validate_full_outputs(ort_full, edges)
            require(case['Torch_domain'] == case['ORT_domain'] == dict(finite=True, weight_domain_valid=True), 'Grouped finite/domain gate failed')
            require({key: meta(value) for key, value in arrays.items()} == original_meta, 'Saved original input/topology mutated')
            wrapped.verify_original_binding(neural.update)
            print(json.dumps(dict(observation_id=observation, edges=edges, groups=len(plan.batches),
                actual_shapes=[list(batch.shape(384)) for batch in plan.batches], strict_gates_so_far=all(gates))), flush=True)
        report.update(status='completed_grouped_Update_numerical_accepted_not_adopted' if all(gates) else 'completed_grouped_Update_numerical_rejected',
            strict_FP32_gate_passed=all(gates), strict_comparison_count=len(gates), outputs_finite=True,
            weight_domain_valid=True, ordinary_grouped_Torch_forward_calls=ordinary_calls,
            actual_export_trace_forward_calls=trace_calls, actual_total_Torch_wrapper_forward_calls=len(torch_calls),
            grouped_ORT_calls=sum(len(case['groups']) for case in report['cases']), ORT_graph_optimization='ORT_DISABLE_ALL',
            numerical_gate_scope='All three captured complete graphs and every group; Torch vs same-call original gold and ORT vs both original gold and grouped Torch.',
            timing_scope='Outer grouped time includes immutable input copy, finite scan, packing, callbacks/owned outputs and full E scatter. Callback times are nested. Loading/export/session initialization/hashes/saving/error statistics are outside.',
            limitations=['Actual saved M16 initialization and observations 127/191 only; original full Update gold is not reexecuted.',
                'Frozen CF Conv/channel norm/stable Sigmoid retains real arithmetic and original 50 weights but may change FP32 rounding.',
                'No SDK placement, recurrent VO submission, BA, trajectory, sustained throughput or 20Hz acceptance.'])
        for model in report['models'].values():
            model['numerical_validation_completed'] = True
    except Exception as error:
        report.update(status='failed_execution', error=repr(error), traceback=traceback.format_exc())
    finally:
        if hook is not None:
            try:
                hook.remove()
            except Exception as error:
                report['cleanup_errors'].append(repr(error))
        sessions.clear()
        session = None
        report['ORT_session_references_released'] = True
        if neural is not None and weights_before is not None:
            try:
                after = {key: meta(value.detach().numpy()) for key, value in neural.state_dict().items()}
                report.update(original_weights_after=after, original_parameters_unchanged=after == weights_before)
                require(after == weights_before, 'Original parameters changed')
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
            require(report['affinity_restored'], 'Affinity restoration failed')
        except Exception as error:
            report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']:
            report['status'] = 'failed_execution'
        report.update(actual_driver_seconds=time.monotonic()-started,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'strict_FP32_gate_passed', 'outputs_finite',
        'weight_domain_valid', 'error', 'cleanup_errors')}), flush=True)
    return 0 if report['status'] == 'completed_grouped_Update_numerical_accepted_not_adopted' else 2 if report['status'] == 'completed_grouped_Update_numerical_rejected' else 1


if __name__ == '__main__':
    raise SystemExit(main())
