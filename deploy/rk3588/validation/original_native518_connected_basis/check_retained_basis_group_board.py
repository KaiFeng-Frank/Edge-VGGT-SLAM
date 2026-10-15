#!/usr/bin/env python3
"""Board-only, conditional FP32 check of one complete original basis group.

Original reference inputs are used once at the group boundary. No intermediate
activation is injected. Basis outputs and their restored canonical counterparts
are both preserved. This is not a rolling full-chain or NPU acceptance test.
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
import signal
import sys
import time
import traceback

B = Path('/mnt/vggt-data/vggt-slam2')
CHECKER_SHA = 'b58c7c83bfc652d859c2472517b8162457f79951a8f5a9773a519e9c2171c616'
BASIS_SHA = '73affeb0b78a794ed214cadcf0307d164ef74f79d52b779bc579e906a1089b70'
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)


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
    require(sha(path) == digest, 'Frozen source differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def validate_reference(reference, C):
    require(reference['schema'] == 'v100_original_native518_retained_group_reference_capture'
        and reference['status'] == 'completed_original_native518_retained_group_references'
        and reference['checkpoint'] == C.CHECKPOINT and reference['image_shape'] == C.IMAGE
        and reference['frame_shape'] == C.FRAME and reference['global_shape'] == C.GLOBAL
        and reference['weights_capture_sha256'] == C.WEIGHTS_SHA
        and reference['prefix_capture_sha256'] == C.PREFIX_SHA
        and reference['original_model_only'] and reference['original_retained_concat_halves_preserved']
        and reference['original_target_clone_and_norm_executed']
        and reference['source_and_inputs_unchanged'] and not reference['cleanup_errors']
        and [r['indices'] for r in reference['groups']] == C.GROUPS,
        'Exact clean original native retained references required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-sha256', required=True)
    p.add_argument('--exporter-source', type=Path, required=True)
    p.add_argument('--exporter-sha256', required=True)
    p.add_argument('--original-checker', type=Path, required=True)
    p.add_argument('--basis-source', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=B/'experiments/vggt-v100-retained-reference-r1/capture.json')
    p.add_argument('--source-root', type=Path, default=B/'src')
    p.add_argument('--group-index', type=int, choices=range(4), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64'
        and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'Board-only execution')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    require(os.statvfs(out).f_bavail * os.statvfs(out).f_frsize >= 128 << 20, '128MiB free output storage required')
    affinity, environment = set(os.sched_getaffinity(0)), dict(os.environ)
    handlers, watched, session = {}, {}, None
    started = time.perf_counter()
    report = dict(schema='v130_original_native518_retained_basis_group_FP32_check', status='running',
        script_sha256=sha(__file__), group_index=a.group_index, fixed_gate=GATE,
        original_model_only=True, actual_execution_on_board_only=True,
        conditional_group_reference_input=True, intermediate_activations_injected=False,
        rolling_full_chain_checked=False, RKNN_GPU_or_calibration_executed=False,
        full_checkpoint_deserialized=False, candidate_accepted=False,
        quality_realtime_or_8GB_accepted=False, system_or_frequency_writes=False,
        sessions_created=0, sessions_dropped=0, calls_completed=0, checks=[], cleanup_errors=[])

    def record():
        tmp = out/'result.tmp'; tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(out/'result.json')

    def interrupted(signum, frame):
        raise InterruptedError('Controller signal ' + str(signum))

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, interrupted)
        require(affinity <= {4,5,6,7} and affinity, 'Caller must restrict CPU affinity to available A76 cores')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            require(os.environ.get(key) == '1', 'Caller must restrict numerical threads')
        C = module(a.original_checker, CHECKER_SHA, '_v130_original_group_checker')
        R = module(a.basis_source, BASIS_SHA, '_v130_original_basis_algebra')
        for path, digest in [(Path(__file__),report['script_sha256']), (a.original_checker,CHECKER_SHA),
                             (a.basis_source,BASIS_SHA),(a.exporter_source,a.exporter_sha256)]:
            C.pin(path, digest, watched)
        ep = C.pin(a.export, a.export_sha256, watched)
        rp = C.pin(a.reference, C.REFERENCE_SHA, watched)
        exported, reference = json.loads(ep.read_text()), json.loads(rp.read_text())
        validate_reference(reference, C)
        indices, bindings, sources = C.spec(a.group_index)
        coordinates = dict(frame_output='basis_R', global_output='basis_R')
        if a.group_index == 3:
            coordinates['target_global_output'] = 'canonical'
        require(exported['schema'] == 'v130_original_native518_retained_basis_group_export'
            and exported['status'] == 'exported_original_basis_group_not_numerically_accepted'
            and exported['script_sha256'] == a.exporter_sha256
            and exported['group_index'] == a.group_index and exported['indices'] == indices
            and exported['checkpoint'] == C.CHECKPOINT
            and exported['source_hashes'] == reference['source_hashes']
            and exported['original_weights_capture_sha256'] == C.WEIGHTS_SHA
            and exported['native_capture_sha256'] == C.PREFIX_SHA
            and exported['input_coordinates'] == ('canonical' if a.group_index == 0 else 'basis_R')
            and exported['output_coordinates'] == coordinates
            and exported['selected_block_count'] == 2*len(indices)
            and exported['original_parameter_value_count'] == 36*len(indices)
            and exported['total_required_blocks'] == 48
            and all(exported['fixed_gate'][k] == v for k,v in C.GATE.items())
            and exported['source_and_inputs_unchanged'] and not exported['cleanup_errors'],
            'Complete original basis group export lineage differs')
        case = reference['groups'][a.group_index]
        require(case['output_sources'] == sources
            and [{k:v[k] for k in ('index','name','shape','dtype')} for v in case['outputs']] == bindings,
            'Original retained and global20 bindings differ')
        root = a.source_root.resolve(strict=True)
        for relative, digest in reference['source_hashes'].items():
            path = (root/relative).resolve(strict=True)
            require(path.is_relative_to(root), 'Original source escaped source root')
            C.pin(path,digest,watched)
        graph = C.artifact(ep.parent, exported['onnx'], watched)
        import numpy as np
        import onnxruntime as ort
        report.update(checkpoint=C.CHECKPOINT, indices=indices, source_hashes=reference['source_hashes'],
            export_receipt_sha256=a.export_sha256, exporter_sha256=a.exporter_sha256,
            reference_receipt_sha256=C.REFERENCE_SHA, graph=exported['onnx'],
            input_coordinates=exported['input_coordinates'],output_coordinates=coordinates,
            versions=dict(numpy=np.__version__,onnxruntime=ort.__version__),
            ORT=dict(providers=['CPUExecutionProvider'], optimization='ORT_DISABLE_ALL',
                     intra_op_num_threads=1,inter_op_num_threads=1,execution_mode='ORT_SEQUENTIAL'))
        canonical = C.tensor(np,rp.parent,case['input'],watched)
        supplied = canonical if a.group_index == 0 else np.ascontiguousarray(R.rotate64(canonical,-1,np),dtype=np.float32)
        report['original_input'] = case['input']; report['supplied_input'] = R.metadata(supplied)
        options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        record()
        session = ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
        report['sessions_created'] = 1
        require(session.get_providers() == ['CPUExecutionProvider']
            and [(v.name,v.shape,v.type) for v in session.get_inputs()] == [('input',C.FRAME,'tensor(float)')]
            and [(v.name,v.shape,v.type) for v in session.get_outputs()] ==
                [(b['name'],b['shape'],'tensor(float)') for b in bindings], 'Original full-token ORT ABI differs')
        tick = time.perf_counter()
        values = session.run([b['name'] for b in bindings], {'input':supplied})
        report.update(calls_completed=1, ORT_call_seconds_including_possible_controller_pauses=time.perf_counter()-tick)
        session = None; gc.collect(); report['sessions_dropped'] = 1
        require(len(values) == len(bindings), 'All retained outputs required')

        def save(name, value):
            path = out/(name+'.npy')
            with path.open('xb') as stream:
                np.save(stream,value,allow_pickle=False)
            return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),**R.metadata(value))

        for value, binding, original in zip(values,bindings,case['outputs']):
            require(value.dtype == np.float32 and list(value.shape) == binding['shape']
                and value.flags.c_contiguous and np.isfinite(value).all(), 'Finite complete FP32 output required')
            name = binding['name']
            restored = value if coordinates[name] == 'canonical' else np.ascontiguousarray(R.rotate64(value,-1,np),dtype=np.float32)
            expected = C.tensor(np,rp.parent,original,watched)
            report['checks'].append(dict(name=name,coordinates=coordinates[name],original_reference=original,
                raw_output=save(name+'-raw',value),canonical_output=save(name+'-canonical',restored),
                comparison=R.comparison(restored,expected,np)))
            record()
        report['fixed_FP32_gate_passed'] = all(v['comparison']['fixed_gate_passed'] for v in report['checks'])
        report['status'] = ('checked_original_basis_group_fixed_FP32_gate_passed' if report['fixed_FP32_gate_passed']
                            else 'checked_original_basis_group_fixed_FP32_gate_failed')
    except BaseException as error:
        report.update(status='failed_preserved_evidence', error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        raise
    finally:
        if session is not None:
            session = None; gc.collect(); report['sessions_dropped'] = report['sessions_created']
        for number, previous in handlers.items():
            signal.signal(number, previous)
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            affinity_unchanged=set(os.sched_getaffinity(0)) == affinity,environment_unchanged=dict(os.environ) == environment)
        try:
            report['source_and_inputs_unchanged'] = all(C.identity(path) == before and sha(path) == digest
                for path,(digest,before) in watched.items())
        except BaseException as error:
            report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append(str(error))
        if not all(report[k] for k in ('source_and_inputs_unchanged','affinity_unchanged','environment_unchanged')):
            report['cleanup_errors'].append('Source/input/process state differed')
            report['status'] = 'failed_cleanup_preserved_evidence'
        record()
    require(report['fixed_FP32_gate_passed'] and not report['cleanup_errors'], 'Original fixed FP32 gate or cleanup failed')
    print(json.dumps(dict(status=report['status'],checks=[dict(name=c['name'],comparison=c['comparison']) for c in report['checks']])),flush=True)


if __name__ == '__main__':
    main()
