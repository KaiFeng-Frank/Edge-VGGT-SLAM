#!/usr/bin/env python3
"""Board-only connected original 48-block FP32 basis continuation and DPT check.

Only group0's original captured DINO boundary is supplied. Every later group
consumes the actual preceding basis global output. Original references are
comparison targets and never replace intermediate activations. Two original
DPT frame chunks consume the actual retained frame-first/global-second basis
concats. This does not execute DINO, CameraHead, pose codec, NPU or image loading.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time
import traceback

B = Path('/mnt/vggt-data/vggt-slam2')
EXPORTER_SHA = 'da34d47ad391e8e30a4162b2d1d3cedc992221b8d6c84e59dbd94c2f385f8443'
GROUP_CHECKER_SHA = '869aa1be6d3f448e208409daebae26185dcbff2cc6813524b6d22fb66e0787ce'
ORIGINAL_CHECKER_SHA = 'b58c7c83bfc652d859c2472517b8162457f79951a8f5a9773a519e9c2171c616'
BASIS_SHA = '73affeb0b78a794ed214cadcf0307d164ef74f79d52b779bc579e906a1089b70'
DEPTH_SOURCE_SHA = '57d888a469581af4c41af58ee858835ba216c98e9c26154a014812e8aeb9bdf7'
DEPTH_EXPORT_SHA = '0040819ec2418235f4f77b49f13943c53aa8a2704ef403bfb63d6b0a8ce6623f'
DEPTH_GRAPH_SHA = '309ec0c60f0b10346de970b064334d790b196fd708b9863f3b822294cb695396'
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
SCHEMA = 'v130_original_native518_connected_basis_groups_FP32_check'
SUCCESS = 'checked_connected_original_48_blocks_and_DPT_fixed_FP32_gate_passed'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def module(path, digest, name):
    require(sha(path) == digest, 'Frozen helper differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def validate_group(exported, checked, index, export_sha, reference, C):
    indices, bindings, sources = C.spec(index)
    coordinates = dict(frame_output='basis_R', global_output='basis_R')
    if index == 3:
        coordinates['target_global_output'] = 'canonical'
    require(exported['schema'] == 'v130_original_native518_retained_basis_group_export'
        and exported['status'] == 'exported_original_basis_group_not_numerically_accepted'
        and exported['script_sha256'] == EXPORTER_SHA and exported['group_index'] == index
        and exported['indices'] == indices and exported['checkpoint'] == C.CHECKPOINT
        and exported['source_hashes'] == reference['source_hashes']
        and exported['original_weights_capture_sha256'] == C.WEIGHTS_SHA
        and exported['native_capture_sha256'] == C.PREFIX_SHA
        and exported['input_coordinates'] == ('canonical' if index == 0 else 'basis_R')
        and exported['output_coordinates'] == coordinates
        and exported['selected_block_count'] == 2*len(indices)
        and exported['original_parameter_value_count'] == 36*len(indices)
        and exported['total_required_blocks'] == 48
        and all(exported['fixed_gate'][k] == v for k,v in C.GATE.items())
        and exported['source_and_inputs_unchanged'] and not exported['cleanup_errors'],
        'Original full-group export lineage differs')
    require(checked['schema'] == 'v130_original_native518_retained_basis_group_FP32_check'
        and checked['status'] == 'checked_original_basis_group_fixed_FP32_gate_passed'
        and checked['script_sha256'] == GROUP_CHECKER_SHA and checked['group_index'] == index
        and checked['indices'] == indices and checked['checkpoint'] == C.CHECKPOINT
        and checked['source_hashes'] == reference['source_hashes']
        and checked['export_receipt_sha256'] == export_sha and checked['exporter_sha256'] == EXPORTER_SHA
        and checked['reference_receipt_sha256'] == C.REFERENCE_SHA
        and checked['graph'] == exported['onnx'] and checked['input_coordinates'] == exported['input_coordinates']
        and checked['output_coordinates'] == coordinates and checked['fixed_gate'] == GATE
        and checked['conditional_group_reference_input'] and not checked['rolling_full_chain_checked']
        and checked['fixed_FP32_gate_passed'] and checked['sessions_created'] == checked['sessions_dropped'] == checked['calls_completed'] == 1
        and checked['source_and_inputs_unchanged'] and checked['affinity_unchanged'] and checked['environment_unchanged']
        and not checked['cleanup_errors'] and len(checked['checks']) == len(bindings)
        and [v['name'] for v in checked['checks']] == [v['name'] for v in bindings]
        and all(v['comparison']['fixed_gate_passed'] for v in checked['checks']),
        'Passing same-graph conditional FP32 group check required')
    case = reference['groups'][index]
    require(case['output_sources'] == sources
        and [{k:v[k] for k in ('index','name','shape','dtype')} for v in case['outputs']] == bindings,
        'Original retained/global20 reference ports differ')
    return bindings, coordinates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports', type=Path, nargs=4, required=True)
    p.add_argument('--export-sha256s', nargs=4, required=True)
    p.add_argument('--conditional-checks', type=Path, nargs=4, required=True)
    p.add_argument('--conditional-check-sha256s', nargs=4, required=True)
    for name in ('exporter-source','group-checker-source','original-checker','basis-source','depth-source','depth-export'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--depth-checks', type=Path, nargs=2, required=True)
    p.add_argument('--depth-check-sha256s', nargs=2, required=True)
    p.add_argument('--reference', type=Path, default=B/'experiments/vggt-v100-retained-reference-r1/capture.json')
    p.add_argument('--head-reference', type=Path, default=B/'experiments/vggt-v99-head-reference-r1/capture.json')
    p.add_argument('--source-root', type=Path, default=B/'src')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64'
        and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'Board-only execution')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage(out).free >= 128 << 20, '128MiB free output storage required')
    environment, affinity = dict(os.environ), set(os.sched_getaffinity(0))
    handlers, watched, session, C = {}, {}, None, None
    started = time.monotonic()
    report = dict(schema=SCHEMA, status='running', script_sha256=sha(__file__), fixed_gate=GATE,
        fixed_gate_definition='relative_l2 <= 1e-5 AND max_abs <= 1e-3; strict allclose is an independent diagnostic',
        strict_allclose_is_diagnostic=True, original_model_only=True, actual_execution_on_board_only=True,
        scope='Connected original 48-block FP32 aggregator plus two DPT raw-logit frame chunks from one captured original DINO boundary',
        original_DINO_or_image_loading_executed=False, CameraHead_or_pose_codec_executed=False,
        RKNN_GPU_or_calibration_executed=False, checkpoint_deserialized=False,
        intermediate_reference_activations_injected=False, candidate_accepted=False,
        full_model_quality_realtime_or_8GB_accepted=False, system_or_frequency_writes=False,
        caller_owns_CPU_thermal_and_timeout_controller=True,
        timings_include_session_loading_validation_IO_and_possible_controller_pauses=True,
        maximum_simultaneous_ORT_sessions=1, sessions_created=0, sessions_dropped=0,
        session_creation_attempts=0, calls_completed=0, actual_aggregator_blocks_evaluated=0,
        groups=[], depth_chunks=[], checks=[], cleanup_errors=[])

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = out/'result.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temporary.replace(out/'result.json')
        print(stage, flush=True)

    def interrupted(signum, frame):
        raise InterruptedError('Controller signal '+str(signum))

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, interrupted)
        require(affinity and affinity <= {4,5,6,7}, 'Caller must restrict CPU to available A76 cores')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        C = module(a.original_checker, ORIGINAL_CHECKER_SHA, '_v130_connected_original_checker')
        R = module(a.basis_source, BASIS_SHA, '_v130_connected_basis')
        Q = module(a.group_checker_source, GROUP_CHECKER_SHA, '_v130_connected_conditional_checker')
        D = module(a.depth_source, DEPTH_SOURCE_SHA, '_v130_connected_depth_source')
        for path,digest in [(Path(__file__),report['script_sha256']), (a.original_checker,ORIGINAL_CHECKER_SHA),
                           (a.basis_source,BASIS_SHA), (a.group_checker_source,GROUP_CHECKER_SHA),
                           (a.exporter_source,EXPORTER_SHA), (a.depth_source,DEPTH_SOURCE_SHA)]:
            C.pin(path,digest,watched)
        rp = C.pin(a.reference,C.REFERENCE_SHA,watched)
        hp = C.pin(a.head_reference,D.CAPTURE_SHA,watched)
        dp = C.pin(a.depth_export,DEPTH_EXPORT_SHA,watched)
        reference, head, depth = [json.loads(v.read_text()) for v in (rp,hp,dp)]
        Q.validate_reference(reference,C)
        original_depth_export = C.pin(depth['parent_export']['file'],D.PARENT_EXPORT_SHA,watched)
        D.validate_lineage(json.loads(original_depth_export.read_text()),head)
        require(reference['head_capture_sha256'] == D.CAPTURE_SHA
            and reference['source_images'] == head['source_images']
            and reference['source_hashes'] == head['source_hashes']
            and [reference['groups'][i]['original_retained_concat'] for i in range(4)] ==
                [head['retained_tokens']['tokens_'+str(i)] for i in (4,11,17,23)],
            'Retained and head reference cases differ')
        require(depth['schema'] == 'v130_original_native518_DPT_basis_consumer_export'
            and depth['status'] == D.EXPORT_SUCCESS and depth['script_sha256'] == DEPTH_SOURCE_SHA
            and depth['checkpoint'] == C.CHECKPOINT and depth['source_hashes'] == reference['source_hashes']
            and depth['head_reference']['sha256'] == D.CAPTURE_SHA
            and depth['all62_original_learned_values_verified'] and depth['source_and_inputs_unchanged']
            and depth['environment_restored'] and depth['affinity_restored'] and not depth['cleanup_errors']
            and depth['graph']['sha256'] == DEPTH_GRAPH_SHA and depth['graph']['bytes'] == 246272013
            and depth['graph']['input_coordinates'] == 'basis T'
            and depth['graph']['output_coordinates'] == 'canonical raw logits'
            and depth['proof']['all79_original_node_protos_byte_identical']
            and depth['proof']['inverse_entire_model_proto_byte_equal'], 'Reviewed complete folded DPT export required')
        depth_graph = C.artifact(dp.parent,depth['graph'],watched)
        for index,(path,digest) in enumerate(zip(a.depth_checks,a.depth_check_sha256s)):
            checked = json.loads(C.pin(path,digest,watched).read_text())
            require(checked['schema'] == 'v130_original_native518_DPT_basis_consumer_check'
                and checked['status'] == D.CHECK_SUCCESS and checked['script_sha256'] == DEPTH_SOURCE_SHA
                and checked['frame_index'] == index and checked['export_receipt']['sha256'] == DEPTH_EXPORT_SHA
                and checked['head_reference']['sha256'] == D.CAPTURE_SHA and checked['graph'] == depth['graph']
                and checked['FP32_gate_passed'] and checked['comparisons']['folded_vs_baseline']['fixed_gate_passed']
                and checked['sessions_created'] == checked['sessions_dropped'] == checked['calls_completed'] == 2
                and checked['source_and_inputs_unchanged'] and checked['affinity_restored'] and checked['environment_restored']
                and not checked['cleanup_errors'], 'Passing folded DPT check for each original frame required')
        root = a.source_root.resolve(strict=True)
        for relative,digest in reference['source_hashes'].items():
            path = (root/relative).resolve(strict=True)
            require(path.is_relative_to(root), 'Original source escaped source root')
            C.pin(path,digest,watched)
        bundles = []
        for index,(ep,eh,cp,ch) in enumerate(zip(a.exports,a.export_sha256s,a.conditional_checks,a.conditional_check_sha256s)):
            ep = C.pin(ep,eh,watched); cp = C.pin(cp,ch,watched)
            exported,checked = json.loads(ep.read_text()),json.loads(cp.read_text())
            bindings,coordinates = validate_group(exported,checked,index,eh,reference,C)
            graph = C.artifact(ep.parent,exported['onnx'],watched)
            bundles.append((graph,bindings,coordinates))
        report.update(checkpoint=C.CHECKPOINT, source_hashes=reference['source_hashes'],
            source_images=reference['source_images'], source_pair_indices=reference['source_pair_indices'],
            original_reference=dict(file=str(rp),sha256=C.REFERENCE_SHA),
            head_reference=dict(file=str(hp),sha256=D.CAPTURE_SHA),
            depth_export=dict(file=str(dp),sha256=DEPTH_EXPORT_SHA,graph=depth['graph']),
            export_receipts=[dict(file=str(p),sha256=h) for p,h in zip(a.exports,a.export_sha256s)],
            conditional_checks=[dict(file=str(p),sha256=h) for p,h in zip(a.conditional_checks,a.conditional_check_sha256s)],
            depth_conditional_checks=[dict(file=str(p),sha256=h) for p,h in zip(a.depth_checks,a.depth_check_sha256s)],
            source_pins={str(p):h for p,(h,_) in watched.items() if p.suffix == '.py'})
        import numpy as np
        import onnxruntime as ort
        report['versions'] = dict(numpy=np.__version__,onnxruntime=ort.__version__,python=platform.python_version())
        options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        report['ORT'] = dict(providers=['CPUExecutionProvider'],optimization='ORT_DISABLE_ALL',
            intra_op_num_threads=1,inter_op_num_threads=1,execution_mode='ORT_SEQUENTIAL')

        def save(name,value):
            path = out/(name+'.npy')
            with path.open('xb') as stream:
                np.save(stream,value,allow_pickle=False)
            return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),**R.metadata(value))

        def compare(actual,expected):
            return R.comparison(actual,expected,np)

        def open_session(graph,inputs,outputs):
            nonlocal session
            require(session is None, 'Only one ORT session may be live')
            report['session_creation_attempts'] += 1
            session = ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
            report['sessions_created'] += 1
            require(session.get_providers() == ['CPUExecutionProvider']
                and [(v.name,v.shape,v.type) for v in session.get_inputs()] == inputs
                and [(v.name,v.shape,v.type) for v in session.get_outputs()] == outputs, 'Full-token logical ORT ABI differs')

        def drop_session():
            nonlocal session
            if session is not None:
                session = None; gc.collect(); report['sessions_dropped'] += 1

        current = C.tensor(np,rp.parent,reference['groups'][0]['input'],watched)
        report['supplied_original_group0_input'] = reference['groups'][0]['input']
        retained = {}
        for index,(graph,bindings,coordinates) in enumerate(bundles):
            case = reference['groups'][index]
            row = dict(group_index=index,indices=C.GROUPS[index],input_coordinates='canonical' if index == 0 else 'basis_R',
                input=R.metadata(current),input_origin='original_captured_DINO_boundary' if index == 0 else 'actual_previous_group_global_output_reshape',
                graph_sha256=a.export_sha256s[index],raw_outputs=[])
            # This field is the graph digest, not the export-receipt digest.
            row['graph_sha256'] = json.loads(a.exports[index].read_text())['onnx']['sha256']
            report['groups'].append(row); progress('starting_actual_group_'+str(index))
            tick = time.monotonic()
            open_session(graph,[('input',C.FRAME,'tensor(float)')],[(v['name'],v['shape'],'tensor(float)') for v in bindings])
            try:
                values = [np.array(v,copy=True,order='C') for v in session.run([b['name'] for b in bindings],{'input':current})]
                report['calls_completed'] += 1
            finally:
                drop_session()
            row['session_load_forward_drop_seconds_including_possible_pauses'] = time.monotonic()-tick
            require(len(values) == len(bindings), 'Complete original retained outputs required')
            for value,binding,original in zip(values,bindings,case['outputs']):
                name = binding['name']
                artifact = save('group'+str(index)+'-'+name,value)
                entry = dict(name=name,coordinates=coordinates[name],artifact=artifact,original_reference=original)
                row['raw_outputs'].append(entry); progress('saved_group'+str(index)+'_'+name)
                require(value.dtype == np.float32 and list(value.shape) == binding['shape'] and artifact['finite'], 'Finite complete FP32 group output required')
                restored = value if coordinates[name] == 'canonical' else np.ascontiguousarray(R.rotate64(value,-1,np),dtype=np.float32)
                expected = C.tensor(np,rp.parent,original,watched)
                entry.update(restored_metadata=R.metadata(restored),comparison=compare(restored,expected))
                report['checks'].append(dict(stage='group',group_index=index,name=name,comparison=entry['comparison']))
                del restored,expected
            retained[index] = np.concatenate((values[0].reshape(1,2,1041,1024),values[1].reshape(1,2,1041,1024)),axis=-1)
            row['actual_retained_frame_first_global_second_basis_concat'] = R.metadata(retained[index])
            if index < 3:
                current = values[1].reshape(C.FRAME)
                row['next_input'] = R.metadata(current)
                require(row['next_input']['raw_sha256'] == row['raw_outputs'][1]['artifact']['raw_sha256'], 'Rolling global reshape changed bytes')
            else:
                report['canonical_global20_target'] = row['raw_outputs'][2]
            report['actual_aggregator_blocks_evaluated'] += 2*len(C.GROUPS[index])
            progress('completed_actual_group_'+str(index))

        last = retained[3][:,:,0,:]
        camera_basis = np.ascontiguousarray(last)
        camera = np.concatenate((np.ascontiguousarray(R.rotate64(last[...,:1024],-1,np),dtype=np.float32),
                                 np.ascontiguousarray(R.rotate64(last[...,1024:],-1,np),dtype=np.float32)),axis=-1)
        expected_camera = C.tensor(np,hp.parent,head['artifacts']['camera_input'],watched)
        camera_comparison = compare(camera,expected_camera)
        report['camera_input'] = dict(basis_rows=save('camera-token0-basis',camera_basis),
            canonical_rows=save('camera-token0-canonical',camera),comparison=camera_comparison,
            only_final_two_token0_rows_restored=True,original_reference=head['artifacts']['camera_input'],
            CameraHead_four_refinements_not_executed=True)
        report['checks'].append(dict(stage='camera_input_boundary',comparison=camera_comparison))
        del expected_camera,camera,camera_basis,last,current,values
        progress('restored_final_camera_token0_rows')

        open_session(depth_graph,[(n,D.INPUT_SHAPE,'tensor(float)') for n in D.INPUT_NAMES],
            [('logits',D.OUTPUT_SHAPE,'tensor(float)')])
        try:
            for frame in range(2):
                chunk = next(v for v in head['depth_chunks'] if v['frame_index'] == frame)
                supplied = {name:np.ascontiguousarray(retained[i][:,frame,5:,:]) for i,name in enumerate(D.INPUT_NAMES)}
                row = dict(frame_index=frame,input_coordinates='basis T=diag(R,R)',input_origin='actual_connected_retained_basis_outputs',
                    original_patch_start_idx=5,full_1036_original_patch_tokens=True,
                    inputs={name:dict(**R.metadata(value),source_group=i,source_concat_raw_sha256=report['groups'][i]['actual_retained_frame_first_global_second_basis_concat']['raw_sha256'])
                            for i,(name,value) in enumerate(supplied.items())})
                report['depth_chunks'].append(row); progress('starting_connected_DPT_frame_'+str(frame))
                tick = time.monotonic()
                logits = np.array(session.run(['logits'],supplied)[0],copy=True,order='C')
                report['calls_completed'] += 1
                row['ORT_forward_seconds_including_possible_pauses'] = time.monotonic()-tick
                row['raw_logits'] = save('depth-frame'+str(frame)+'-raw-logits',logits)
                progress('saved_connected_DPT_frame_'+str(frame))
                require(logits.dtype == np.float32 and list(logits.shape) == D.OUTPUT_SHAPE and row['raw_logits']['finite'], 'Finite complete original DPT logits required')
                expected = C.tensor(np,hp.parent,chunk['raw_logits'],watched)
                row.update(original_reference=chunk['raw_logits'],comparison=compare(logits,expected))
                report['checks'].append(dict(stage='DPT_raw_logits',frame_index=frame,comparison=row['comparison']))
                del logits,expected,supplied
        finally:
            drop_session()
        report['fixed_FP32_gate_passed'] = all(v['comparison']['fixed_gate_passed'] for v in report['checks'])
        report['resource_counts_balanced'] = report['sessions_created'] == report['sessions_dropped'] == 5 and report['calls_completed'] == 6
        require(report['actual_aggregator_blocks_evaluated'] == 48 and len(report['groups']) == 4
            and len(report['depth_chunks']) == 2 and len(report['checks']) == 12 and report['resource_counts_balanced'], 'Complete connected original continuation counts differ')
        report['status'] = SUCCESS if report['fixed_FP32_gate_passed'] else 'checked_connected_original_FP32_gate_failed_all_outputs_preserved'
    except BaseException as error:
        report.update(status='failed_preserved_available_evidence',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        raise
    finally:
        if session is not None:
            session = None; gc.collect(); report['sessions_dropped'] += 1
        for number,previous in handlers.items():
            signal.signal(number,previous)
        try:
            os.sched_setaffinity(0,affinity)
            report['affinity_restored'] = set(os.sched_getaffinity(0)) == affinity
        except BaseException as error:
            report['affinity_restored'] = False; report['cleanup_errors'].append('affinity: '+repr(error))
        os.environ.clear(); os.environ.update(environment)
        report['environment_restored'] = dict(os.environ) == environment
        try:
            report['source_and_inputs_unchanged'] = C is not None and all(C.identity(path) == before and sha(path) == digest
                for path,(digest,before) in watched.items())
            require(report['source_and_inputs_unchanged'],'Pinned source/input/graph changed')
        except BaseException as error:
            report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('pins: '+repr(error))
        if report['cleanup_errors']:
            report['status'] = 'failed_cleanup_or_pinned_identity'
        report['ORT_sessions_dropped'] = report['sessions_created'] == report['sessions_dropped']
        report['process_cpu_seconds'] = dict(user=resource.getrusage(resource.RUSAGE_SELF).ru_utime,system=resource.getrusage(resource.RUSAGE_SELF).ru_stime)
        progress('terminal')
    require(report.get('fixed_FP32_gate_passed') and not report['cleanup_errors'],'Connected original fixed FP32 gate or cleanup failed; outputs preserved')
    print(json.dumps(dict(status=report['status'],checks=report['checks'])),flush=True)


if __name__ == '__main__':
    main()
