#!/usr/bin/env python3
"""Board-only export of two missing, complete B1/D4-or-D6/M16 Update shapes.

Original parameters and frozen v37 CF arithmetic are retained. Sources are
preselected from the accepted v39 capture, never from error/GT measurements.
No SDK, BA, recurrent submission, source padding, target/patch split or adoption.
Princeton-VL/DPVO 0ac95b656d1fda91c271d2a106460d19ad966fc7, MIT,
Copyright (c) 2022 Princeton University. Weights/SDK licenses remain separate.
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
PARENT_NAME = 'export-v39-m16-grouped-update-board.py'
PARENT_SHA256 = '0dc3d9e49645cd3e472fa344434bca0aa9360b62d6d37fc3e4d71fefa67d0842'
MANIFEST_SHA256 = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
RUN_SHA256 = 'dd9f37524b999c1e586182a92c226481fcfc7f4da851d5e88d62d82be1029ecd'
INPUTS, OUTPUTS = ('net','inp','corr'), ('net_out','delta','weight')
CANONICAL = ('net','delta','weight')
DEGREES = (4,6)


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''): digest.update(block)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def load_parent(path):
    require(sha(path) == PARENT_SHA256,'Frozen v39 exporter SHA differs')
    spec = importlib.util.spec_from_file_location('_v41_frozen_M16_exporter',path)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complete_sources(adapter, plan, case):
    """Pure topology selection: first source of each predeclared B2D4/B2D6."""
    require(case['observation_id'] == case['original_call_index'] == 34 and case['edges'] == plan.edges == 704
        and case['degree_plan'] == plan.metadata(),'Exact original initialization graph required')
    result = []
    for degree in DEGREES:
        group_index, group = next((i,g) for i,g in enumerate(case['groups'])
                                 if g['model_key'] == f'B2_D{degree}_M16')
        batch = plan.batches[group_index]
        require(batch.metadata() == group['topology'] and batch.degree == degree and batch.patches == 16
            and len(batch.source_slots) == 2 and meta(batch.canonical_edge_indices) == group['canonical_edge_indices'],
            'Accepted parent complete source group differs')
        indices = np.array(batch.canonical_edge_indices[:1],copy=True,order='C')
        require(indices.shape == (1,degree,16) and np.unique(indices).size == degree*16,
                'Complete D times 16 source indices required')
        indices.flags.writeable = False
        selected = adapter.DegreeBatch(degree,16,batch.source_slots[:1],batch.target_slots[:1],indices)
        result.append((selected,group_index,group))
    return result


def validate_outputs(values, batch):
    require(set(values) == set(CANONICAL),'Three canonical CF outputs required')
    finite = all(values[name].dtype == np.float32 and values[name].shape == batch.shape(channels)
        and np.isfinite(values[name]).all() for name,channels in zip(CANONICAL,(384,2,2)))
    weight = values['weight']
    return dict(finite=bool(finite),weight_domain_valid=bool(np.isfinite(weight).all()
                and np.all((weight >= 0)&(weight <= 1))))


def board_guard():
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64','RK3588 board execution only')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 device tree required')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('parent-exporter','run-dir','source-export-dir','update-source-dir','v26-source-dir','checkpoint','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(argv)
    board_guard()  # Before model imports, captures, output files or affinity mutation.
    out = args.output_dir.resolve()
    protected = [v.resolve() for k,v in vars(args).items() if k != 'output_dir']
    require(out != BASE and out.is_relative_to(BASE) and not out.exists()
        and not any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected),
        'New non-overlapping board output directory required')
    out.mkdir(exist_ok=False)
    begin,prior = time.monotonic(),os.sched_getaffinity(0)
    pins,sessions,hook,session,neural,wrapped,before = {},{},None,None,None,None,None
    forwards = []
    report = dict(schema='v41_missing_M16_single_source_Update_CPU_ONNX_diagnostic',status='failed_execution',
        script_sha256=sha(__file__),parent_exporter_sha256=PARENT_SHA256,
        source_M16_manifest_sha256=MANIFEST_SHA256,original_run_sha256=RUN_SHA256,patches_per_frame=16,
        candidate_accepted=False,original_VGGT_default_unchanged=True,host_model_inference=False,
        actual_board_execution=True,NPU_used=False,BA_executed=False,VO_state_committed=False,realtime_validated=False,
        original_full_Update_reexecuted=False,source_member_padding=False,neighbor_or_edge_removal=False,
        original_parameters_unchanged=False,strict_FP32_thresholds=dict(rtol=1e-4,atol=1e-5),
        strict_FP32_thresholds_relaxed=False,output_clipping=False,terminal_sigmoid='stable',models={},cases=[],cleanup_errors=[],
        selection='Observation34 original call34; first complete source in frozen B2D4 and B2D6 source-slot order, fixed before numerical comparisons; no GT selection.')
    def pin(path,expected):
        path = Path(path).resolve();require(sha(path) == expected,'Source/input SHA differs: '+str(path))
        pins[str(path)] = expected;return path
    def save(name,values):
        path = out/name
        with path.open('xb') as stream: np.savez(stream,**values)
        return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in values.items()})
    try:
        pin(__file__,report['script_sha256']);pin(args.parent_exporter,PARENT_SHA256)
        v39 = load_parent(args.parent_exporter);v37 = v39.utilities(args.update_source_dir)
        source_file = pin(args.source_export_dir/'manifest.json',MANIFEST_SHA256)
        source_export = json.loads(source_file.read_text())
        require(source_export['schema'] == 'v39_full_M16_grouped_Update_CPU_ONNX_diagnostic'
            and source_export['status'] == 'completed_grouped_Update_numerical_accepted_not_adopted'
            and source_export['script_sha256'] == PARENT_SHA256 and source_export['strict_FP32_gate_passed']
            and source_export['candidate_accepted'] is False and source_export['original_parameters_unchanged'],
            'Original accepted M16 source export required')
        run,records,capture_pins = v39.load_capture(args.run_dir,RUN_SHA256,args.update_source_dir)
        pins.update(capture_pins)
        for source in (run,source_export):
            require(source['source_pins'] == source['source_hashes_after'] and source['sources_and_inputs_unchanged'],
                    'Frozen source declaration differs')
            for path,digest in source['source_pins'].items():pin(path,digest)
        for name,digest in v37.NEURAL_PINS.items():pin(args.v26_source_dir/name,digest)
        pin(args.checkpoint,v37.CHECKPOINT_SHA)
        require({4,5,6,7} <= prior,'Four A76 CPUs required');os.sched_setaffinity(0,{4,5,6,7})
        import torch
        import onnx
        import onnxruntime as ort
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        adapter = v37.module(args.update_source_dir/'source_frame_batches.py','_v41_original_CF_transport')
        cf = v37.module(args.update_source_dir/'source_frame_update_channel_first.py','_v41_original_CF_math')
        common = v37.module(args.v26_source_dir/'dpvo_probe_common.py','_v41_original_neural_loader')
        neural,loader = common.load_neural(args.v26_source_dir/'official',args.v26_source_dir/'manifest.json',
            args.checkpoint,v37.CHECKPOINT_SHA,args.v26_source_dir/'dpvo_cpu_ops.py',
            cpu_ops_sha256=v37.NEURAL_PINS['dpvo_cpu_ops.py'])
        for name,value in loader['source_hashes'].items():pin(args.v26_source_dir/'official'/name,value['sha256'])
        before = {k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
        require(len(before) == 94 and before == run['original_weights_before'] == source_export['original_weights_before'],
                'Original complete 94 state bytes differ')
        parameters = {k:v.detach().numpy() for k,v in neural.update.state_dict().items()}
        parameter_meta = {k:meta(v) for k,v in parameters.items()}
        require(len(parameters) == 50 and parameter_meta == source_export['original_Update_parameters'],
                'All original 50 Update parameters required')
        wrapped = cf.SourceFrameUpdateChannelFirst(neural.update,terminal_sigmoid='stable').eval()
        hook = wrapped.register_forward_hook(lambda _m,_a,_o:forwards.append(1))
        report.update(v37_exporter_sha256=v39.PARENT_SHA256,input_manifest_sha256=run['input_manifest_sha256'],
            policy_sha256=run['policy_sha256'],original_weights_before=before,original_Update_parameters=parameter_meta,
            original_binding=wrapped.verify_original_binding(neural.update),loader=loader,
            cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,
            torch_version=torch.__version__,onnx_version=onnx.__version__,ORT_version=ort.__version__)
        captured = next(r for r in records if r['capture']['frame_index'] == 34)
        parent_case = next(c for c in source_export['cases'] if c['observation_id'] == 34)
        selections = complete_sources(adapter,captured['plan'],parent_case)
        options = ort.SessionOptions();options.intra_op_num_threads,options.inter_op_num_threads = 4,1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for key in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):options.add_session_config_entry(key,'0')
        gates,ordinary,trace_calls = [],0,0
        for batch,group_index,source_group in selections:
            key = v37.model_key(batch);feeds = {name:adapter.pack_batch(captured['inputs'][name],batch) for name in INPUTS}
            gold = {name:adapter.pack_batch(captured['gold'][name],batch) for name in CANONICAL}
            parent_feed = v39.read_artifact(args.source_export_dir,source_group['inputs_artifact'])
            for name in INPUTS:require(np.array_equal(feeds[name],parent_feed[name][:1]),'Exact complete source feed differs')
            pin(args.source_export_dir/source_group['inputs_artifact']['file'],source_group['inputs_artifact']['sha256'])
            input_before = {name:meta(v) for name,v in feeds.items()}
            binding = dict(parent_manifest_sha256=MANIFEST_SHA256,parent_model_key=source_group['model_key'],
                observation_id=34,original_call_index=34,parent_group_index=group_index,
                selected_source_slot=batch.source_slots[0],selected_target_slots=list(batch.target_slots[0]),
                canonical_edge_indices=meta(batch.canonical_edge_indices))
            case = dict(model_key=key,observation_id=34,original_call_index=34,edges=batch.degree*16,
                source_binding=binding,topology=batch.metadata(),original_full_inputs_artifact=captured['capture']['input'],
                original_full_gold_artifact=captured['capture']['gold'],
                inputs_artifact=save(key+'-inputs.npz',feeds),original_gold_artifact=save(key+'-original-source-gold.npz',gold),
                source_indices_artifact=save(key+'-source-indices.npz',dict(canonical_edge_indices=batch.canonical_edge_indices)))
            report['cases'].append(case)
            arguments = tuple(torch.from_numpy(feeds[name]) for name in INPUTS)
            wrapped.validate_inputs(*arguments);started = time.perf_counter()
            with torch.no_grad(): torch_values = wrapped(*arguments)
            case['Torch_forward_seconds'] = time.perf_counter()-started;ordinary += 1
            actual = {name:np.array(value.detach().numpy(),copy=True,order='C') for name,value in zip(CANONICAL,torch_values)}
            case['Torch_artifact'] = save(key+'-Torch.npz',actual)
            case['Torch_domain'] = validate_outputs(actual,batch)
            case['Torch_vs_original_group_gold'] = {n:v37.compare(actual[n],gold[n]) for n in CANONICAL}
            path = out/(key+'.onnx');wanted_inputs,wanted_outputs = v37.contracts(batch)
            export_options = dict(input_names=list(INPUTS),output_names=list(OUTPUTS),opset_version=17,do_constant_folding=True)
            if 'dynamo' in inspect.signature(torch.onnx.export).parameters:export_options['dynamo'] = False
            count_before,started = len(forwards),time.perf_counter()
            with torch.no_grad():torch.onnx.export(wrapped,arguments,str(path),**export_options)
            export_seconds = time.perf_counter()-started;count = len(forwards)-count_before;trace_calls += count
            graph = onnx.load(path,load_external_data=False);onnx.checker.check_model(graph,full_check=True)
            counts = Counter(node.op_type for node in graph.graph.node)
            axes = [onnx.helper.get_attribute_value(attr) for node in graph.graph.node if node.op_type == 'Softmax'
                    for attr in node.attribute if attr.name == 'axis']
            require([v37.graph_contract(v) for v in graph.graph.input] == wanted_inputs
                and [v37.graph_contract(v) for v in graph.graph.output] == wanted_outputs
                and counts['Conv'] == 21 and counts['Softmax'] == 2 and sorted(axes) == [2,3]
                and not {'Transpose','Gather','GatherElements','GatherND','Where','Pad','Gemm','MatMul'} & set(counts),
                'Original complete-source CF ONNX contract differs')
            report['models'][key] = dict(model=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)),
                inputs=wanted_inputs,outputs=wanted_outputs,batch=1,degree=batch.degree,patches=16,source_binding=binding,
                parameter_coverage=v37.initializer_coverage(graph,parameters,onnx),ONNX_operator_counts=dict(counts),
                export_seconds=export_seconds,actual_export_trace_forward_calls=count,external_data_used=False,
                padding=False,numerical_validation_completed=False)
            session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);sessions[key] = session
            require(session.get_providers() == ['CPUExecutionProvider'],'Only CPU ORT allowed')
            for rows,wanted in ((session.get_inputs(),wanted_inputs),(session.get_outputs(),wanted_outputs)):
                require([dict(name=v.name,dtype='float32' if v.type == 'tensor(float)' else v.type,shape=list(v.shape))
                         for v in rows] == wanted,'ORT actual IO differs')
            started = time.perf_counter();ort_values = session.run(list(OUTPUTS),feeds)
            case['ORT_forward_seconds'] = time.perf_counter()-started
            ort_arrays = {n:np.array(v,copy=True,order='C') for n,v in zip(CANONICAL,ort_values)}
            case['ORT_artifact'] = save(key+'-ORT.npz',ort_arrays);case['ORT_domain'] = validate_outputs(ort_arrays,batch)
            case['ORT_vs_original_group_gold'] = {n:v37.compare(ort_arrays[n],gold[n]) for n in CANONICAL}
            case['ORT_vs_grouped_Torch'] = {n:v37.compare(ort_arrays[n],actual[n]) for n in CANONICAL}
            require(case['Torch_domain'] == case['ORT_domain'] == dict(finite=True,weight_domain_valid=True),
                    'Source outputs must be finite/domain-valid')
            for label in ('Torch_vs_original_group_gold','ORT_vs_original_group_gold','ORT_vs_grouped_Torch'):
                gates.extend(v['gate_passed'] for v in case[label].values())
            require({n:meta(v) for n,v in feeds.items()} == input_before,'Source feeds mutated')
            wrapped.verify_original_binding(neural.update);report['models'][key]['numerical_validation_completed'] = True
        report.update(status='completed_missing_M16_source_numerical_accepted_not_adopted' if all(gates)
            else 'completed_missing_M16_source_numerical_rejected',strict_FP32_gate_passed=all(gates),
            strict_comparison_count=len(gates),outputs_finite=True,weight_domain_valid=True,
            ordinary_Torch_forward_calls=ordinary,actual_export_trace_forward_calls=trace_calls,
            actual_total_Torch_wrapper_forward_calls=len(forwards),ORT_calls=len(report['cases']),
            ORT_graph_optimization='ORT_DISABLE_ALL',
            timing_scope='Torch/ORT calls include neural execution only; NumPy conversion, hashes, saving, graph export/session initialization are separate. These are numerical checks, not a controlled performance benchmark.',
            limitations=['Only the two fixed complete source batches; other catalogue and D1 motion-probe paths are separate.',
                'All original weights/math retained, but changed batch shape may change FP32 rounding; fixed gates decide component acceptance.',
                'No SDK placement, full online VO recurrence, BA, trajectory, sustained throughput or 20Hz acceptance.'])
    except Exception as error:report.update(status='failed_execution',error=repr(error),traceback=traceback.format_exc())
    finally:
        if hook is not None:
            try:hook.remove()
            except Exception as error:report['cleanup_errors'].append(repr(error))
        sessions.clear();session = None;report['ORT_session_references_released'] = True
        if neural is not None and before is not None:
            try:
                after = {k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
                report.update(original_weights_after=after,original_parameters_unchanged=after == before)
                require(after == before,'Original parameters changed')
                if wrapped is not None:wrapped.verify_original_binding(neural.update)
            except Exception as error:report['cleanup_errors'].append(repr(error))
        try:
            report.update(source_pins=pins,source_hashes_after={p:sha(p) for p in pins})
            report['sources_and_inputs_unchanged'] = report['source_pins'] == report['source_hashes_after']
            require(report['sources_and_inputs_unchanged'],'Pinned input/source changed')
        except Exception as error:report['cleanup_errors'].append(repr(error))
        try:
            os.sched_setaffinity(0,prior);report['affinity_restored'] = os.sched_getaffinity(0) == prior
            require(report['affinity_restored'],'Affinity restore failed')
        except Exception as error:report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']:report['status'] = 'failed_execution'
        report.update(actual_driver_seconds=time.monotonic()-begin,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','strict_FP32_gate_passed','error','cleanup_errors')}),flush=True)
    return 0 if report['status'] == 'completed_missing_M16_source_numerical_accepted_not_adopted' else 2 if report['status'] == 'completed_missing_M16_source_numerical_rejected' else 1


if __name__ == '__main__':raise SystemExit(main())
