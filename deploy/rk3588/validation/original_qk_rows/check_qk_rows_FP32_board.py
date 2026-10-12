#!/usr/bin/env python3
"""RK3588-only fixed FP32 gate for original frame0 Q/K row layout; no SDK build."""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

BASE = Path('/mnt/vggt-data/vggt-slam2')
HELPER_SHA = '4094e393d0504954a67993c24e82c5534862393930f60db3ec64c2c3e4098b76'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
SCHEMA = 'v86_original_VGGT_pair00_frame_QK_rows_FP32_check'
PASSED = 'passed_original_FP32_ONNX_equivalence_not_quality_accepted'
SHAPE = [2, 53, 1024]
BINDINGS = [dict(index=0, name='frame_output', shape=SHAPE, dtype='float32'),
            dict(index=1, name='global_output', shape=[1, 106, 1024], dtype='float32')]
PROOF_FLAGS = ('all_original_initializers_protobuf_unchanged',
    'all_original_parameter_values_byte_equal', 'all_unrelated_nodes_protobuf_unchanged',
    'original_IO_protobuf_unchanged', 'global0_protobuf_unchanged',
    'original_qk_affine_and_LayerNorm_attributes_preserved',
    'original_RoPE_values_and_rotation_preserved',
    'all_original_tokens_and_attention_reductions_preserved')
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def load_helper(path):
    require(sha(path) == HELPER_SHA, 'Frozen reference helper changed')
    spec = importlib.util.spec_from_file_location('_v86_original_pair_reference', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def run_check(c, original, candidate, source_cases, output, report):
    import numpy as np
    import onnxruntime as ort
    report['versions'] = dict(python=sys.version, numpy=np.__version__, onnxruntime=ort.__version__)
    report['ORT'] = dict(optimization='ORT_DISABLE_ALL', intra_op_num_threads=1,
        inter_op_num_threads=1, providers=['CPUExecutionProvider'])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions, session = [], None
    try:
        for path in (original, candidate):
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            sessions.append(session)
            session.disable_fallback()
            require(session.get_providers() == ['CPUExecutionProvider']
                and [(v.name, v.shape, v.type) for v in session.get_inputs()] == [('input', SHAPE, 'tensor(float)')]
                and [(v.name, v.shape, v.type) for v in session.get_outputs()]
                    == [(b['name'], b['shape'], 'tensor(float)') for b in BINDINGS], 'ORT provider or complete pair ABI differs')
        report['ORT_sessions_created'] = len(sessions)
        for source in source_cases:
            root, tag = Path(source['directory']), source['tag']
            x = c.tensor(np, root, source['input'], SHAPE)
            input_hash = hashlib.sha256(memoryview(x).cast('B')).hexdigest()
            case = dict(tag=tag, capture_manifest_sha256=source['capture_manifest_sha256'],
                source_images=source['source_images'], input_origin=source['input_origin'],
                input=c.save_tensor(np, output, tag + '-input.npy', x, dict(name='input', shape=SHAPE)),
                original_outputs=[], candidate_outputs=[], original_Torch_outputs=None, checks=[])
            report['cases'].append(case)
            results = []
            for label, session in zip(('original', 'candidate'), sessions):
                tick = time.perf_counter()
                results.append(session.run([b['name'] for b in BINDINGS], {'input': x}))
                case[label + '_ORT_CPU_seconds'] = time.perf_counter() - tick
                require(hashlib.sha256(memoryview(x).cast('B')).hexdigest() == input_hash, 'ORT changed the shared real input')
            references = source['original_outputs'][:2] if source['original_outputs'] else None
            if references is not None:
                case['original_Torch_outputs'] = []
            for index, binding in enumerate(BINDINGS):
                a, b = results[0][index], results[1][index]
                for label, value in (('original', a), ('candidate', b)):
                    case[label + '_outputs'].append(c.save_tensor(np, output, f'{tag}-{label}-{index}.npy', value, binding))
                check = dict(name=binding['name'], versus_original_FP32_ONNX=c.compare(np, b, a),
                    byte_equal_to_original=a.tobytes() == b.tobytes(), versus_original_Torch=None)
                if references is not None:
                    reference = c.tensor(np, root, references[index], binding['shape'])
                    case['original_Torch_outputs'].append(c.save_tensor(np, output, f'{tag}-original-Torch-{index}.npy', reference, binding))
                    check['versus_original_Torch'] = {label: c.compare(np, value, reference)
                        for label, value in (('original', a), ('candidate', b))}
                case['checks'].append(check)
            c.write(output / 'check.json', report)
        checks = [v for case in report['cases'] for v in case['checks']]
        require(len(checks) == 4, 'Exactly four complete-output comparisons required')
        report.update(all_candidate_vs_original_FP32_ONNX_strict_passed=all(v['versus_original_FP32_ONNX']['strict_pass'] for v in checks),
            all_candidate_vs_original_FP32_ONNX_byte_equal=all(v['byte_equal_to_original'] for v in checks),
            original_Torch_reference_case_count=sum(case['original_Torch_outputs'] is not None for case in report['cases']),
            original_Torch_references_cover_all_cases=False,
            all_original_Torch_strict_passed_where_reference_available=all(v['versus_original_Torch'][label]['strict_pass']
                for v in checks if v['versus_original_Torch'] is not None for label in ('original', 'candidate')))
        require(report['all_candidate_vs_original_FP32_ONNX_strict_passed'], 'Fixed original FP32 ONNX tolerance failed; compilation blocked')
        report['status'] = PASSED
    finally:
        sessions.clear()
        session = None
        report['sessions_dropped'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('rewrite-receipt', 'rewrite-source', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--rewrite-receipt-sha256', required=True)
    p.add_argument('--rewrite-source-sha256', required=True)
    p.add_argument('--parent-manifest', type=Path, default=BASE / 'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--capture-dir', type=Path, default=BASE / 'experiments/vggt-v53-capture-r1')
    p.add_argument('--calibration-dir', type=Path, default=BASE / 'experiments/vggt-v56-calibration-r1')
    p.add_argument('--reference-helper', type=Path, default=BASE / 'src/deploy/rk3588/validation/original_pair_group/check_compile_pair_group.py')
    args = p.parse_args()
    compatible = Path('/proc/device-tree/compatible')
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64'
        and compatible.is_file() and b'rockchip,rk3588' in compatible.read_bytes(), 'Actual ORT execution is restricted to RK3588')
    c = load_helper(args.reference_helper)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0)
    environment = {name: os.environ.get(name) for name in THREAD_ENV}
    cwd, started = Path.cwd(), time.perf_counter()
    report = dict(schema=SCHEMA, status='running', script_sha256=sha(__file__), helper_sha256=HELPER_SHA,
        original_parent_manifest_sha256=PARENT_SHA, original_pair_onnx_sha256=GRAPH_SHA,
        rewrite_receipt_sha256=args.rewrite_receipt_sha256, rewrite_source_sha256=args.rewrite_source_sha256,
        strict_rtol=1e-4, strict_atol=1e-5, input_shape=SHAPE, output_bindings=BINDINGS,
        candidate_accepted=False, quality_accepted=False, realtime_validated=False, original_model_only=True,
        Torch_model_loaded=False, SDK_executed=False, cases=[], operations=[], cleanup_errors=[], sessions_dropped=False,
        cpu_affinity_before=sorted(affinity), byte_equality_is_not_compile_gate=True,
        timing_scope='CPU ORT reference diagnostics; not NPU or complete-model performance')
    watched = [Path(__file__).resolve(), args.reference_helper.resolve()]
    pins = {str(path): sha(path) for path in watched}
    c.write(output / 'result.json', report)
    try:
        os.sched_setaffinity(0, {4, 5, 6, 7})
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        require(report['cpu_affinity'] == [4, 5, 6, 7], 'Matched board CPU affinity unavailable')
        for name in THREAD_ENV:
            os.environ[name] = '1'
        parent_path = c.pin(args.parent_manifest, PARENT_SHA)
        source_path = c.pin(args.rewrite_source, args.rewrite_source_sha256)
        receipt_path = c.pin(args.rewrite_receipt, args.rewrite_receipt_sha256)
        parent, rewrite = json.loads(parent_path.read_text()), json.loads(receipt_path.read_text())
        row, proof = parent['blocks'][0], rewrite['proof']
        require(parent['schema'] == 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest'
            and parent['checkpoint'] == c.CHECKPOINT and row['index'] == 0 and row['role'] == 'pair'
            and row['sha256'] == GRAPH_SHA and row['input_name'] == 'input' and row['input_shape'] == SHAPE
            and row['output_bindings'] == BINDINGS, 'Exact original pair00 provenance/ABI required')
        require(rewrite['schema'] == 'v86_original_pair00_frame_QK_rows_ONNX_rewrite'
            and rewrite['status'] == 'candidate_written_not_FP32_accepted'
            and rewrite['original_parent_manifest_sha256'] == PARENT_SHA
            and rewrite['original_pair_onnx_sha256'] == GRAPH_SHA
            and rewrite['original_checkpoint'] == c.CHECKPOINT and rewrite['parent'] == row
            and rewrite['script_sha256'] == args.rewrite_source_sha256
            and rewrite['source_and_inputs_unchanged'] is True, 'Pinned completed rewrite lineage differs')
        require(proof['original_parameter_tensors_verified'] == 36 and proof['original_initializer_count'] == 63
            and all(proof[k] is True for k in PROOF_FLAGS)
            and set(proof['named_node_diff']) == {'removed', 'modified', 'added'}, 'Original computation/parameter preservation proof differs')
        original, candidate = c.artifact(parent_path.parent, row), c.artifact(receipt_path.parent, rewrite['candidate'])
        require(sha(original) == GRAPH_SHA, 'Original pair graph differs')
        watched.extend((parent_path, source_path, receipt_path, original, candidate))
        source_cases = c.real_cases(args.capture_dir.resolve(strict=True), args.calibration_dir.resolve(strict=True), parent, watched)
        require([v['tag'] for v in source_cases] == ['v53', 'v56-calib00'], 'Exactly two distinct frozen real inputs required')
        pins = {str(path): sha(path) for path in watched}
        report.update(candidate_onnx_sha256=rewrite['candidate']['sha256'], candidate=rewrite['candidate'],
            checkpoint=parent['checkpoint'], parent=row, graph_proof=proof, source_hashes=parent['source_hashes'],
            source_and_input_hashes=pins)
        c.write(output / 'result.json', report)
        run_check(c, original, candidate, source_cases, output, report)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        report['environment_restored'] = Path.cwd() == cwd and all(os.environ.get(k) == v for k, v in environment.items())
        try:
            os.sched_setaffinity(0, affinity)
            report['cpu_affinity_after'] = sorted(os.sched_getaffinity(0))
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            require(report['affinity_restored'], 'Original CPU affinity was not restored')
        except BaseException as exc:
            report['cleanup_errors'].append(f'affinity restore: {type(exc).__name__}: {exc}')
        changed = []
        for path, digest in pins.items():
            try:
                require(sha(path) == digest, 'Pinned file changed')
            except BaseException as exc:
                changed.append(f'{path}: {type(exc).__name__}: {exc}')
        report['source_and_inputs_unchanged'] = not changed
        report['cleanup_errors'].extend(changed)
        if not report['environment_restored']:
            report['cleanup_errors'].append('Original environment/cwd was not restored')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['seconds'] = time.perf_counter() - started
        c.write(output / 'check.json', report)
        final = dict(report, check_receipt_sha256=sha(output / 'check.json'))
        c.write(output / 'result.json', final)
        print(json.dumps(dict(status=report['status'], check_receipt_sha256=final['check_receipt_sha256'],
            result_sha256=sha(output / 'result.json'))), flush=True)
    return int(report['status'] != PASSED)


if __name__ == '__main__':
    raise SystemExit(main())
