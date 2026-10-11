#!/usr/bin/env python3
"""Board-only original-weight teacher and complete B1/D3/M16 EOF export.

Original parameters and frozen v37 CF arithmetic are retained. Saved failed
v41 NPU-feedback inputs get one new original-Torch full-graph teacher on board.
This teacher is not an original CPU trajectory gold. The only D3 source keeps
all three targets and all16 patches; selection uses topology, not error or GT.
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
SOURCE_M16_MANIFEST_SHA256 = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
SOURCE_RUN_SHA256 = '78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b'
SOURCE_FAILURE_SHA256 = 'f7fecc2da09ee74da72da4d30af99bb874149a5e6f6961a199a28c29f91cb072'
SOURCE_FAILED_INPUT_SHA256 = 'b0954b8dd84118d9749e7baeca15df83c59985059a0407cf1970521e8acb18ca'
SOURCE_BEFORE_TERMINATION_SHA256 = 'd879eece461d1c20cd52388bcde97ba29f196b468b9398aac49eec6a9a85313e'
INPUTS, OUTPUTS = ('net','inp','corr'), ('net_out','delta','weight')
CANONICAL = ('net','delta','weight')
MODEL_KEY = 'B1_D3_M16'


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
    spec = importlib.util.spec_from_file_location('_v42_frozen_M16_exporter',path)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_run_record(run, failure):
    """Saved declarations only; raw files and topology are checked separately."""
    require(run['schema'] == 'v41_cached_undistort_online_RKNN_research_replay'
        and run['status'] == 'failed' and run['frames_completed'] == 192
        and 'B1_D3_M16' in run['error'] and run['active_stage'] == 'termination_offline'
        and run['online_metrics']['calls'][-1] == failure,
        'Exact EOF missing-shape failure and owning run required')
    require(failure['index'] == 203 and failure['route'] == 'full'
        and failure['status'] == 'failed_no_partial_fullgraph' and failure['edges'] == 640
        and not failure['fullgraph_returned'] and not failure['outputs_committed_by_callback']
        and 'SDK_execution' not in failure and 'B1_D3_M16' in failure['error'],
        'Failed complete640 graph must have returned no partial SDK result')
    require(run['cleanup_errors'] == [] and run['affinity_restored']
        and run['sources_and_inputs_unchanged'] and run['source_pins'] == run['source_hashes_after']
        and run['original_weights_unchanged'] and len(run['original_weights_before']) == 94
        and run['original_weights_before'] == run['original_weights_after']
        and run['candidate_accepted'] is False and run['historical_SDK_strict_FP32_rejected'] is True
        and run['SDK_activity']['Update_NPU_used'] is True and run['SDK_activity']['encoder_NPU_used'] is True,
        'Original run weights/source/cleanup and rejected NPU-feedback provenance differ')
    require(run['config']['patches_per_frame'] == 16 and run['config']['capacity'] == 64
        and run['snapshot_before_termination']['metadata']['initialized'] is True,
        'M16 initialized original snapshot required')


def load_failed_capture(directory, v39, adapter):
    """NumPy-only provenance loader; never imports a model or runs geometry."""
    root = Path(directory).resolve()
    files = {'run.json':SOURCE_RUN_SHA256,
        'online-sdk/update-000203-failed.json':SOURCE_FAILURE_SHA256,
        'online-sdk/update-000203-failed-inputs.npz':SOURCE_FAILED_INPUT_SHA256,
        'before-termination.npz':SOURCE_BEFORE_TERMINATION_SHA256}
    pins = {}
    for filename, digest in files.items():
        path = root/filename
        require(path.resolve().is_relative_to(root) and not path.is_symlink()
            and path.is_file() and sha(path) == digest, 'Frozen EOF artifact SHA differs: '+filename)
        pins[str(path.resolve())] = digest
    run = json.loads((root/'run.json').read_text())
    failure = json.loads((root/'online-sdk/update-000203-failed.json').read_text())
    validate_run_record(run, failure)
    require(failure['failure_artifact']['file'] == 'update-000203-failed-inputs.npz'
        and failure['failure_artifact']['sha256'] == SOURCE_FAILED_INPUT_SHA256
        and run['snapshot_before_termination']['artifact']['file'] == 'before-termination.npz'
        and run['snapshot_before_termination']['artifact']['sha256'] == SOURCE_BEFORE_TERMINATION_SHA256,
        'Owning record must bind each original artifact')
    inputs = v39.read_artifact(root/'online-sdk', failure['failure_artifact'])
    snapshot = v39.read_artifact(root, run['snapshot_before_termination']['artifact'])
    require(set(inputs) == set(INPUTS+('ii','jj','kk')), 'Exact failed-input keys required')
    for name, channels in zip(INPUTS,(384,384,882)):
        value = inputs[name]
        require(value.dtype == np.float32 and value.shape == (1,640,channels)
            and value.flags.c_contiguous and np.isfinite(value).all(), 'Complete failed FP32 input differs: '+name)
    for name in ('ii','jj','kk'):
        value = inputs[name]
        require(value.dtype == np.int64 and value.shape == (640,), 'Failed int64 topology differs: '+name)
    require(np.array_equal(inputs['ii'],inputs['kk']//16), 'Failed source-patch mapping differs')
    for name in ('net','ii','jj','kk'):
        require(np.array_equal(inputs[name],snapshot[name]), 'Failed input differs from pre-EOF snapshot: '+name)
    require(snapshot['poses'].shape == (1,12,7) and snapshot['patches'].shape == (1,192,3,3,3),
        'Pre-EOF retained source population differs')
    manifest = root/'input-manifest.json'
    require(sha(manifest) == run['input_manifest_sha256'], 'Original image manifest SHA differs')
    pins[str(manifest.resolve())] = run['input_manifest_sha256']
    plan = adapter.build_degree_plan(*(inputs[n] for n in ('ii','jj','kk')),
        patches_per_frame=16,frame_capacity=64,max_target_degree=7,max_sources_per_batch=2)
    return run,failure,inputs,plan,pins


def complete_source(plan, failure):
    """One complete D3 source, independently sorted by target then patch."""
    require(plan.edges == 640 and plan.metadata() == failure['degree_plan'],
        'Every failed graph edge and original degree-plan identity required')
    selected = [batch for batch in plan.batches if batch.degree == 3]
    require(len(selected) == 1, 'Exactly one actual D3 source required; no synthetic B2 graph')
    batch = selected[0]
    require(batch.source_slots == (11,) and batch.target_slots == ((9,10,11),)
        and batch.patches == 16 and batch.canonical_edge_indices.shape == (1,3,16)
        and np.unique(batch.canonical_edge_indices).size == 48,
        'Complete source11 with targets9/10/11 and all16 patches required')
    return batch


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
    for name in ('parent-exporter','run-dir','update-source-dir','v26-source-dir','checkpoint','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--diagnostic-allow-fp32-rejected',action='store_true')
    args = parser.parse_args(argv)
    board_guard()  # Before model imports, captures, output files or affinity mutation.
    require(args.diagnostic_allow_fp32_rejected,
        'Explicit diagnostic flag required for historical rejected NPU-feedback inputs')
    out = args.output_dir.resolve()
    protected = [v.resolve() for k,v in vars(args).items() if k != 'output_dir' and isinstance(v,Path)]
    require(out != BASE and out.is_relative_to(BASE) and not out.exists()
        and not any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected),
        'New non-overlapping board output directory required')
    out.mkdir(exist_ok=False)
    begin,prior = time.monotonic(),os.sched_getaffinity(0)
    pins,sessions,hook,session,neural,wrapped,before = {},{},None,None,None,None,None
    forwards = []
    report = dict(schema='v42_EOF_M16_D3_single_source_Update_CPU_ONNX_diagnostic',status='failed_execution',
        script_sha256=sha(__file__),parent_exporter_sha256=PARENT_SHA256,
        source_M16_manifest_sha256=SOURCE_M16_MANIFEST_SHA256,source_run_sha256=SOURCE_RUN_SHA256,
        original_run_sha256=SOURCE_RUN_SHA256,source_failure_sha256=SOURCE_FAILURE_SHA256,
        source_failed_input_sha256=SOURCE_FAILED_INPUT_SHA256,
        source_before_termination_sha256=SOURCE_BEFORE_TERMINATION_SHA256,patches_per_frame=16,
        candidate_accepted=False,original_VGGT_default_unchanged=True,host_model_inference=False,
        actual_board_execution=True,NPU_used=False,BA_executed=False,VO_state_committed=False,realtime_validated=False,
        original_full_Update_reexecuted=False,original_full_Update_forward_calls=0,
        teacher_reference_kind='new_original_Torch_on_saved_NPU_feedback_input',
        original_CPU_trajectory_gold_used=False,historical_SDK_strict_FP32_rejected=True,
        diagnostic_allow_fp32_rejected=True,source_member_padding=False,neighbor_or_edge_removal=False,
        original_parameters_unchanged=False,strict_FP32_thresholds=dict(rtol=1e-4,atol=1e-5),
        strict_FP32_thresholds_relaxed=False,output_clipping=False,terminal_sigmoid='stable',models={},cases=[],cleanup_errors=[],
        selection='Frozen EOF callback203 full640 edges; unique complete source11 targets9/10/11, all16 patches, from topology before comparisons; no GT selection.')
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
        for name,digest in v37.UPDATE_PINS.items():pin(args.update_source_dir/name,digest)
        adapter = v37.module(args.update_source_dir/'source_frame_batches.py','_v42_original_CF_transport')
        run,failure,full_inputs,plan,capture_pins = load_failed_capture(args.run_dir,v39,adapter)
        pins.update(capture_pins)
        for path,digest in run['source_pins'].items():pin(path,digest)
        source_manifest = next((Path(path) for path,digest in run['source_pins'].items()
            if digest == SOURCE_M16_MANIFEST_SHA256),None)
        require(source_manifest is not None, 'Original M16 source manifest pin missing')
        original_export = json.loads(source_manifest.read_text())
        require(original_export['original_parameters_unchanged'] and original_export['strict_FP32_gate_passed']
            and original_export['original_weights_before'] == run['original_weights_before'],
            'Original94 weight ancestry differs')
        batch = complete_source(plan,failure)
        full_input_before = {name:meta(value) for name,value in full_inputs.items()}
        for name,digest in v37.NEURAL_PINS.items():pin(args.v26_source_dir/name,digest)
        pin(args.checkpoint,v37.CHECKPOINT_SHA)
        require({4,5,6,7} <= prior,'Four A76 CPUs required');os.sched_setaffinity(0,{4,5,6,7})
        import torch
        import onnx
        import onnxruntime as ort
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        cf = v37.module(args.update_source_dir/'source_frame_update_channel_first.py','_v42_original_CF_math')
        common = v37.module(args.v26_source_dir/'dpvo_probe_common.py','_v42_original_neural_loader')
        neural,loader = common.load_neural(args.v26_source_dir/'official',args.v26_source_dir/'manifest.json',
            args.checkpoint,v37.CHECKPOINT_SHA,args.v26_source_dir/'dpvo_cpu_ops.py',
            cpu_ops_sha256=v37.NEURAL_PINS['dpvo_cpu_ops.py'])
        for name,value in loader['source_hashes'].items():pin(args.v26_source_dir/'official'/name,value['sha256'])
        before = {k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
        require(len(before) == 94 and before == run['original_weights_before'] == original_export['original_weights_before'],
                'Original complete 94 state bytes differ')
        parameters = {k:v.detach().numpy() for k,v in neural.update.state_dict().items()}
        parameter_meta = {k:meta(v) for k,v in parameters.items()}
        require(len(parameters) == 50 and parameter_meta == original_export['original_Update_parameters'],
                'All original 50 Update parameters required')
        wrapped = cf.SourceFrameUpdateChannelFirst(neural.update,terminal_sigmoid='stable').eval()
        hook = wrapped.register_forward_hook(lambda _m,_a,_o:forwards.append(1))
        report.update(v37_exporter_sha256=v39.PARENT_SHA256,input_manifest_sha256=run['input_manifest_sha256'],
            policy_sha256=run['policy_sha256'],original_weights_before=before,original_Update_parameters=parameter_meta,
            original_binding=wrapped.verify_original_binding(neural.update),loader=loader,
            cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,
            torch_version=torch.__version__,onnx_version=onnx.__version__,ORT_version=ort.__version__)
        teacher_arguments = tuple(torch.from_numpy(full_inputs[name]) for name in INPUTS+('ii','jj','kk'))
        teacher_started = time.perf_counter()
        with torch.inference_mode():
            teacher_net,(teacher_delta,teacher_weight,unused) = neural.update(
                *teacher_arguments[:3],None,*teacher_arguments[3:])
        teacher_seconds = time.perf_counter()-teacher_started
        require(unused is None, 'Original full Update auxiliary return differs')
        teacher = {name:np.array(value.detach().numpy(),copy=True,order='C')
                   for name,value in zip(CANONICAL,(teacher_net,teacher_delta,teacher_weight))}
        v39.validate_arrays(full_inputs,teacher,640)
        require({name:meta(value) for name,value in full_inputs.items()} == full_input_before,
            'Original teacher mutated frozen inputs')
        report.update(original_full_Update_reexecuted=True,original_full_Update_forward_calls=1,
            original_full_Update_forward_seconds=teacher_seconds,
            full_teacher_artifact=save('new-same-input-full640-Torch-teacher.npz',teacher),
            source_failure_artifact=failure['failure_artifact'],
            source_before_termination_artifact=run['snapshot_before_termination']['artifact'],
            source_failure_snapshot_identity=failure['snapshot_identity'],full_source_degree_plan=plan.metadata())
        options = ort.SessionOptions();options.intra_op_num_threads,options.inter_op_num_threads = 4,1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for key in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):options.add_session_config_entry(key,'0')
        gates,ordinary,trace_calls = [],0,0
        for batch in (batch,):
            key = v37.model_key(batch);require(key == MODEL_KEY,'Only actual B1D3 permitted')
            feeds = {name:adapter.pack_batch(full_inputs[name],batch) for name in INPUTS}
            gold = {name:adapter.pack_batch(teacher[name],batch) for name in CANONICAL}
            input_before = {name:meta(v) for name,v in feeds.items()}
            binding = dict(parent_run_sha256=SOURCE_RUN_SHA256,parent_failure_sha256=SOURCE_FAILURE_SHA256,
                before_termination_sha256=SOURCE_BEFORE_TERMINATION_SHA256,
                failed_update_index=203,selected_source_slot=11,selected_target_slots=[9,10,11],
                canonical_edge_indices=meta(batch.canonical_edge_indices),
                teacher_reference_kind=report['teacher_reference_kind'],original_CPU_trajectory_gold_used=False)
            case = dict(model_key=key,observation_id=191,original_call_index=203,edges=48,
                source_binding=binding,topology=batch.metadata(),
                original_full_inputs_artifact=failure['failure_artifact'],
                full_original_teacher_artifact=report['full_teacher_artifact'],
                inputs_artifact=save(key+'-inputs.npz',feeds),
                original_gold_artifact=save(key+'-new-original-Torch-source-teacher.npz',gold),
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
        require(len(gates) == 9 and len(report['models']) == len(report['cases']) == 1, 'One complete source with nine fixed comparisons required')
        report.update(status='completed_EOF_M16_D3_numerical_accepted_not_adopted' if all(gates)
            else 'completed_EOF_M16_D3_numerical_rejected',strict_FP32_gate_passed=all(gates),
            strict_comparison_count=len(gates),outputs_finite=True,weight_domain_valid=True,
            ordinary_Torch_forward_calls=ordinary,actual_export_trace_forward_calls=trace_calls,
            actual_total_Torch_wrapper_forward_calls=len(forwards),ORT_calls=len(report['cases']),
            ORT_graph_optimization='ORT_DISABLE_ALL',
            timing_scope='Torch/ORT calls include neural execution only; NumPy conversion, hashes, saving, graph export/session initialization are separate. These are numerical checks, not a controlled performance benchmark.',
            limitations=['Only actual B1D3 complete EOF source; no B2D3 graph, full catalogue/online recurrence validation or D1 NPU claim.',
                'Teacher is a new original-Torch evaluation of saved NPU-feedback inputs, not the original CPU trajectory gold; inherited RKNN rejection is retained.',
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
    return 0 if report['status'] == 'completed_EOF_M16_D3_numerical_accepted_not_adopted' else 2 if report['status'] == 'completed_EOF_M16_D3_numerical_rejected' else 1


if __name__ == '__main__':raise SystemExit(main())
