#!/usr/bin/env python3
"""Check one complete original native518 retained group against pinned CPU arrays.

Adapted from Edge-VGGT-SLAM's BSD-2-Clause retained-group validator. Original
weights have their separate license. This checks one real two-frame activation
case, with no intermediate substitution; it is not full-chain quality or speed
acceptance. Scratch and owned outputs stay under --output-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import tempfile
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
SCHEMA = 'v100_original_native518_retained_group_check_compile'
CHECK_SUCCESS = 'checked_original_FP32_native_group_not_quality_accepted'
COMPILE_SUCCESS = 'compiled_original_native_group_not_quality_accepted'
FRAME, GLOBAL, IMAGE = [2, 1041, 1024], [1, 2082, 1024], [2, 3, 392, 518]
GROUPS = [list(range(0, 5)), list(range(5, 12)), list(range(12, 18)), list(range(18, 24))]
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
REFERENCE_SHA = 'b95d06c154c37b2e9fde74bd682e494cbffe54224688405f1fcaf0b9a5518b71'
WEIGHTS_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
PREFIX_SHA = '4ebbc4adaaed5e3c3cf8558b4947ec3fc090d56e82f8c61c9509951eb34dd35d'
EXPORTER_SHA = 'b0ac4e8070db7f0cfdd3f2713e2ce43c28fc7847b8da79e64cc6b73108f5c269'
GRAPH_HELPER_SHA = '3ff8b0d0515ff44ea3625bd2c6529d63a0bed3487d409128ee4b443922a14bfd'
COMPOSER_SHA = 'e6fa508e0b03a011a1ccb484f2a46847ffd49fc1bf6ae11ecfeb8bd45f835f60'
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3)
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'TMPDIR')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def identity(path):
    st = Path(path).stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def pin(path, digest, watched, size=None):
    path = Path(path).resolve(strict=True)
    before = identity(path)
    require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Malformed SHA256')
    require(path.is_file() and (size is None or before[2] == size) and sha(path) == digest and identity(path) == before,
            'Pinned source/artifact changed: ' + str(path))
    watched[path] = (digest, before)
    return path


def artifact(root, row, watched):
    root = Path(root).resolve(strict=True)
    relative = Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Artifact must be capture-owned relative path')
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), 'Artifact escaped its receipt directory')
    return pin(path, row['sha256'], watched, row['bytes'])


def atomic(path, report):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def spec(index):
    indices = GROUPS[index]
    bindings = [dict(index=0, name='frame_output', shape=FRAME, dtype='float32'),
                dict(index=1, name='global_output', shape=GLOBAL, dtype='float32')]
    sources = [dict(index=i, name=b['name'], pair_index=indices[-1], role=role)
               for i, (b, role) in enumerate(zip(bindings, ('frame', 'global')))]
    if index == 3:
        bindings.append(dict(index=2, name='target_global_output', shape=GLOBAL, dtype='float32'))
        sources.append(dict(index=2, name='target_global_output', pair_index=20, role='global'))
    return indices, bindings, sources


def validate_metadata(exported, reference, weights, index):
    indices, bindings, sources = spec(index)
    require(reference['schema'] == 'v100_original_native518_retained_group_reference_capture'
        and reference['status'] == 'completed_original_native518_retained_group_references'
        and reference['checkpoint'] == CHECKPOINT and reference['image_shape'] == IMAGE
        and reference['frame_shape'] == FRAME and reference['global_shape'] == GLOBAL
        and reference['weights_capture_sha256'] == WEIGHTS_SHA and reference['prefix_capture_sha256'] == PREFIX_SHA
        and reference['original_model_only'] is True and reference['original_retained_concat_halves_preserved'] is True
        and reference['original_target_clone_and_norm_executed'] is True
        and reference['source_and_inputs_unchanged'] is True and reference['cleanup_errors'] == []
        and [g['indices'] for g in reference['groups']] == GROUPS, 'Exact completed original native reference required')
    require(exported['schema'] == 'v100_original_native518_retained_aggregator_group_ONNX_export'
        and exported['status'] == 'static_original_native_group_not_FP32_or_RKNN_accepted'
        and exported['script_sha256'] == EXPORTER_SHA and exported['checkpoint'] == CHECKPOINT
        and exported['image_shape'] == IMAGE and exported['group_index'] == index and exported['indices'] == indices
        and exported['source_hashes'] == reference['source_hashes'] and exported['boundary_mode'] == 'fp32'
        and exported['original_weights_capture_sha256'] == WEIGHTS_SHA and exported['native_capture_sha256'] == PREFIX_SHA
        and exported['graph_helper_sha256'] == GRAPH_HELPER_SHA and exported['retained_composer_sha256'] == COMPOSER_SHA
        and exported['source_and_inputs_unchanged'] is True and exported['verification_errors'] == []
        and all(exported['required_next_gate'][k] == v for k, v in GATE.items())
        and exported['selected_block_count'] == 2 * len(indices)
        and exported['original_parameter_value_count'] == 36 * len(indices)
        and exported['total_required_blocks'] == 48 and exported['retained_consumer_indices'] == [4, 11, 17, 23]
        and exported['original_target_tokens_global_index'] == 20
        and exported['original_target_token_clone_and_norm_must_be_retained'] is True
        and len(exported['blocks']) == 1, 'Complete original native group export required')
    row, case = exported['blocks'][0], reference['groups'][index]
    require(row['role'] == 'retained_pair_group' and row['index'] == index and row['indices'] == indices
        and row['input_name'] == 'input' and row['input_shape'] == FRAME and row['output_bindings'] == bindings
        and row['public_output_sources'] == case['output_sources'] == sources
        and case['input']['shape'] == FRAME and case['input']['name'] == 'input'
        and [{k:r[k] for k in ('index','name','shape','dtype')} for r in case['outputs']] == bindings
        and row['added_reshape_count'] == row['added_shape_initializer_count'] == len(indices)-1
        and row['original_parameter_value_count'] == 36 * len(indices)
        and row['target_global_output_source_pair_index'] == (20 if index == 3 else None)
        and row['target_global_output_is_unnormalized'] is (True if index == 3 else None)
        and [p['index'] for p in row['parents']] == indices
        and [p['index'] for p in row['parent_preservation_proofs']] == indices
        and all(p['inverse_namespace_recovers_every_node_and_initializer_protobuf_bytewise'] is True
                for p in row['parent_preservation_proofs'])
        and all(row[k] is True for k in ('all_original_parent_nodes_attributes_and_initializer_payloads_retained',
            'all_original_parameter_values_preserved_by_parent_composition', 'original_FP32_parent_graph_math_preserved',
            'only_original_interpair_reshapes_added', 'all_original_frame_and_global_computation_tensors_retained',
            'only_retained_consumer_outputs_public', 'original_retained_concats_required')), 'Native group ABI or preservation proof differs')
    require(weights['checkpoint'] == CHECKPOINT
        and weights['status'] == 'completed_original_FP32_full_forward_and_48_block_capture', 'Pinned original 48-block parameter capture required')
    original = {(r['role'], r['index']):r for r in weights['blocks']}
    require(len(original) == len(weights['blocks']) == 48
        and set(original) == {(role,i) for role in ('frame','global') for i in range(24)}, 'All48 original parameter metadata required')
    expected = {str(i):{role:{n:r['raw_sha256'] for n,r in original[role,i]['parameters'].items()}
                          for role in ('frame','global')} for i in indices}
    require(row['parameter_raw_sha256'] == expected
        and all(p['parameter_raw_sha256'] == expected[str(p['index'])] for p in row['parents'])
        and all(len(expected[str(i)][role]) == 18 for i in indices for role in ('frame','global')),
        'Original learned parameter hashes differ')
    require(all(row[k] == exported['onnx'][k] for k in ('file','bytes','sha256')), 'Graph metadata disagrees')
    return row, case


def load_bundle(args, watched, report):
    export_path = pin(args.export, args.export_sha256, watched)
    reference_path = pin(args.reference, REFERENCE_SHA, watched)
    weights_path = pin(args.weights_capture, WEIGHTS_SHA, watched)
    exported, reference, weights = [json.loads(p.read_text()) for p in (export_path, reference_path, weights_path)]
    row, case = validate_metadata(exported, reference, weights, args.group_index)
    graph = artifact(export_path.parent, row, watched)
    root = args.source_root.resolve(strict=True)
    for relative, digest in reference['source_hashes'].items():
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), 'Original source escaped source root')
        pin(path, digest, watched)
    indices, bindings, sources = spec(args.group_index)
    report.update(checkpoint=CHECKPOINT, group_index=args.group_index, indices=indices,
        input_name='input', input_shape=FRAME, image_shape=IMAGE, output_bindings=bindings,
        public_output_sources=sources, source_hashes=reference['source_hashes'],
        export_receipt=str(export_path), export_receipt_sha256=args.export_sha256,
        reference_receipt=str(reference_path), reference_receipt_sha256=REFERENCE_SHA,
        weights_capture_sha256=WEIGHTS_SHA, graph_sha256=row['sha256'],
        onnx=dict(file=str(graph), bytes=row['bytes'], sha256=row['sha256']),
        original_parameter_value_count=row['original_parameter_value_count'],
        selected_block_count=2*len(indices), total_required_blocks=48,
        retained_consumer_indices=[4,11,17,23], original_target_tokens_global_index=20,
        original_target_clone_and_norm_required=True,
        frame_layout=exported['frame_layout'], global_fc2_parts=exported['global_fc2_parts'],
        original_reference_input=case['input'], original_reference_outputs=case['outputs'],
        original_target_reference_has_no_prior_native_tensor_comparison=reference['new_target_reference_without_prior_numerical_comparison'])
    return graph, reference_path.parent, case


def tensor(np, root, row, watched):
    value = np.load(artifact(root, row, watched), allow_pickle=False)
    require(value.dtype == np.float32 and list(value.shape) == row['shape'] and value.flags.c_contiguous
        and row['dtype'] == 'float32' and row['tensor_bytes'] == value.nbytes and np.isfinite(value).all()
        and hashlib.sha256(memoryview(value).cast('B')).hexdigest() == row['raw_sha256'], 'Original FP32 array identity differs')
    return value


def metrics(np, actual, reference):
    expected = reference.astype(np.float64)
    delta = actual.astype(np.float64) - expected
    relative = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-30))
    maximum = float(np.max(np.abs(delta)))
    return dict(relative_l2=relative, max_abs=maximum,
        fixed_gate_pass=relative <= GATE['relative_l2_max'] and maximum <= GATE['max_abs_max'],
        elements=int(actual.size), rmse=float(np.sqrt(np.mean(delta*delta))),
        bitwise_equal_original_CPU=actual.tobytes() == reference.tobytes())


def run_check(graph, root, case, output, watched, report):
    import numpy as np
    import onnxruntime as ort
    report['versions'] = dict(numpy=np.__version__, onnxruntime=ort.__version__)
    report['ORT'] = dict(providers=['CPUExecutionProvider'], optimization='ORT_DISABLE_ALL',
        intra_op_num_threads=1, inter_op_num_threads=1, execution_mode='ORT_SEQUENTIAL')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = None
    try:
        x = tensor(np, root, case['input'], watched)
        session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
        require(session.get_providers() == ['CPUExecutionProvider']
            and [(v.name,v.shape,v.type) for v in session.get_inputs()] == [('input',FRAME,'tensor(float)')]
            and [(v.name,v.shape,v.type) for v in session.get_outputs()] ==
                [(b['name'],b['shape'],'tensor(float)') for b in report['output_bindings']], 'Exact native FP32 ORT ABI required')
        report['ORT_run_attempts'] = 1
        atomic(output/'check.json', report)
        wall, cpu = time.perf_counter(), time.process_time()
        values = session.run([b['name'] for b in report['output_bindings']], {'input':x})
        report.update(ORT_call_wall_seconds_including_possible_controller_pauses=time.perf_counter()-wall,
            ORT_call_process_cpu_seconds=time.process_time()-cpu)
        require(len(values) == len(report['output_bindings']), 'All original retained outputs required')
        for value, original, binding in zip(values, case['outputs'], report['output_bindings']):
            require(value.dtype == np.float32 and list(value.shape) == binding['shape']
                and value.flags.c_contiguous and np.isfinite(value).all(), 'Finite native FP32 ORT output required')
            reference = tensor(np, root, original, watched)
            path = output / (binding['name']+'-ort.npy')
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            saved = dict(**binding, file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                tensor_bytes=value.nbytes, raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())
            report['checks'].append(dict(index=binding['index'], name=binding['name'],
                original_reference=original, actual_output=saved, **metrics(np,value,reference)))
            atomic(output/'check.json', report)
            print(json.dumps(report['checks'][-1]), flush=True)
            del reference
        report['original_FP32_gate_passed'] = all(c['fixed_gate_pass'] for c in report['checks'])
        report['status'] = CHECK_SUCCESS if report['original_FP32_gate_passed'] else 'numerical_rejected'
    finally:
        session = None
        report['sessions_dropped'] = True


def parser(description=__doc__):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-sha256', required=True)
    p.add_argument('--reference', type=Path, default=BASE/'experiments/vggt-v100-retained-reference-r1/capture.json')
    p.add_argument('--weights-capture', type=Path, default=BASE/'experiments/vggt-v53-capture-r1/capture.json')
    p.add_argument('--source-root', type=Path, default=BASE/'src')
    p.add_argument('--group-index', type=int, choices=range(4), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cpu-affinity', type=int, nargs='+', default=[4,5,6,7])
    return p


def execute(args, mode, operation, script_path):
    compatible = Path('/proc/device-tree/compatible')
    require(platform.machine() == 'aarch64' and compatible.exists()
        and b'rockchip,rk3588' in compatible.read_bytes(), 'Actual ORT/SDK execution is restricted to RK3588/S')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    old_cwd, old_affinity = Path.cwd(), os.sched_getaffinity(0)
    old_env, old_tmp = {k:os.environ.get(k) for k in THREAD_ENV}, tempfile.tempdir
    old_handlers, watched = {}, {}
    report = dict(schema=SCHEMA, mode=mode, status='running', script_sha256=sha(script_path),
        checkpoint=CHECKPOINT, fixed_gate=GATE, original_fixed_FP32_gate_preserved=True,
        actual_execution_on_board_only=True, original_model_only=True,
        full_checkpoint_deserialized=False, performance_measurement=False,
        full_chain_execution=False, component_original_activation_case_only=True,
        no_capture_intermediate_substitution=True, candidate_accepted=False,
        quality_realtime_or_8GB_validated=False, system_or_frequency_writes=False,
        checks=[], operations=[], cleanup_errors=[], source_and_inputs_unchanged=False)
    started = time.perf_counter()
    def terminated(signum, frame):
        raise InterruptedError('External controller signal '+str(signum))
    try:
        require(args.cpu_affinity and set(args.cpu_affinity) <= {4,5,6,7}
            and set(args.cpu_affinity) <= old_affinity, 'Available process-local A76 affinity required')
        os.sched_setaffinity(0, set(args.cpu_affinity))
        for name in THREAD_ENV[:-1]:
            os.environ[name] = '1'
        os.chdir(args.output_dir)
        os.environ['TMPDIR'] = str(args.output_dir)
        tempfile.tempdir = str(args.output_dir)
        for number in (signal.SIGINT,signal.SIGTERM):
            old_handlers[number] = signal.signal(number,terminated)
        pin(script_path, report['script_sha256'], watched)
        graph, root, case = load_bundle(args, watched, report)
        atomic(args.output_dir/'result.json', report)
        operation(graph, root, case, args.output_dir, watched, report)
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        try:
            require(watched and all(identity(p) == before and sha(p) == digest for p,(digest,before) in watched.items()),
                    'A pinned source/graph/reference changed during execution')
            report['source_and_inputs_unchanged'] = True
        except BaseException as error:
            report['cleanup_errors'].append('Pins: '+str(error))
        try:
            os.chdir(old_cwd)
            os.sched_setaffinity(0,old_affinity)
            tempfile.tempdir = old_tmp
            for k,v in old_env.items():
                if v is None: os.environ.pop(k,None)
                else: os.environ[k]=v
            for number,handler in old_handlers.items():
                signal.signal(number,handler)
            report.update(affinity_restored=os.sched_getaffinity(0)==old_affinity,
                environment_restored=Path.cwd()==old_cwd and tempfile.tempdir==old_tmp
                    and all(os.environ.get(k)==v for k,v in old_env.items()))
            require(report['affinity_restored'] and report['environment_restored'], 'Process environment restore failed')
        except BaseException as error:
            report['cleanup_errors'].append('Environment: '+str(error))
        if report['cleanup_errors']:
            report['status']='failed_cleanup'
        report.update(elapsed_seconds_including_controller_pauses_IO_and_validation=time.perf_counter()-started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            source_and_input_hashes={str(p):d for p,(d,_) in watched.items()})
        atomic(args.output_dir/'result.json',report)
        if mode=='check': atomic(args.output_dir/'check.json',report)
        print(json.dumps({k:report.get(k) for k in ('status','group_index','original_FP32_gate_passed','rknn','error','cleanup_errors')}),flush=True)
    return 0 if report['status'] in (CHECK_SUCCESS,COMPILE_SUCCESS) else 1


def main():
    args=parser().parse_args()
    # Resolve caller-relative inputs before entering the private scratch directory.
    for name in ('export','reference','weights_capture','source_root'):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    return execute(args,'check',run_check,Path(__file__).resolve())


if __name__ == '__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
