#!/usr/bin/env python3
"""Board FP32 composition gate, then nonquantizing FP16 compile of two cuts.

The middle is the unchanged original 12-node ONNX attention/layout graph.
Only the prefix and suffix are compiled. No checkpoint deserialization.
"""
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import weakref

BASE = Path('/mnt/vggt-data/vggt-slam2')
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
COMPILER_SOURCE_SHA = 'ade8f932c267bdc2a9ff5854e305d87456984630025aa34c0c448499f5a6a81b'
CHECK_SCHEMA = 'v120_original_native518_DINO0_cuts_FP32_check'
CHECK_SUCCESS = 'checked_original_DINO0_complete_composition_not_NPU_quality_accepted'
COMPILE_SCHEMA = 'v120_original_native518_DINO0_cuts_FP16_compile'
COMPILE_SUCCESS = 'compiled_original_DINO0_cuts_not_quality_accepted'
FIXED_GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3)
STRICT_COMPARISON = dict(rtol=1e-4, atol=1e-5)
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)
PROOF_KEYS = ('original_node_union_exactly_once', 'whole_parent_reassembled_protobuf_equal',
    'full_initializer_union_original_protobuf_equal', 'all14_original_learned_parameters_verified',
    'original_middle_12_nodes_preserved', 'original_residual_input_preserved_in_suffix',
    'all1041_queries_keys_16_heads_64_dimensions', 'original_exact_Erf_GELU_preserved')
CHECK_IDS = ('parent_vs_original', 'prefix_QKV_vs_original', 'middle_attention_vs_original',
    'composed_vs_original', 'composed_vs_parent')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def raw(value):
    import numpy as np
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def atomic(path, row):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def module(path):
    spec = importlib.util.spec_from_file_location('_v120_frozen_exact_cut_exporter', path)
    require(spec is not None and spec.loader is not None, 'Exporter source cannot be loaded')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compare(actual, reference):
    import numpy as np
    require(actual.dtype == reference.dtype == np.float32 and actual.shape == reference.shape and
        np.isfinite(actual).all() and np.isfinite(reference).all(), 'Complete finite FP32 comparison required')
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    d = a - b
    relative = float(np.linalg.norm(d) / max(float(np.linalg.norm(b)), np.finfo(np.float64).tiny))
    maximum = float(np.abs(d).max())
    outside = int((np.abs(d) > STRICT_COMPARISON['atol'] + STRICT_COMPARISON['rtol'] * np.abs(b)).sum())
    return dict(relative_l2=relative, max_abs=maximum, strict_pass=outside == 0,
        strict_rtol=STRICT_COMPARISON['rtol'], strict_atol=STRICT_COMPARISON['atol'],
        outside_count=outside, byte_equal=actual.tobytes() == reference.tobytes(),
        fixed_gate_passed=relative <= FIXED_GATE['relative_l2_max'] and maximum <= FIXED_GATE['max_abs_max'])


def array_file(path, row, watched):
    import numpy as np
    path = Path(path).resolve(strict=True)
    require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Saved array file differs: ' + str(path))
    watched[path] = row['sha256']
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float32 and list(value.shape) == row['shape'] and row['dtype'] == 'float32' and
        value.nbytes == row['tensor_bytes'] and raw(value) == row['raw_sha256'] and np.isfinite(value).all(),
        'Saved complete FP32 tensor differs: ' + str(path))
    return value


def child(root, file):
    relative = Path(file)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Receipt-relative artifact required')
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), 'Artifact escapes its receipt directory')
    return path


def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict' and not node.args:
        return {k.arg: literal(k.value) for k in node.keywords if k.arg is not None}
    return ast.literal_eval(node)


def original_config(path):
    values = [n.value for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in n.targets)]
    require(len(values) == 1 and literal(values[0]) == CONFIG, 'Frozen original FP16 configuration differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('check', 'compile'), required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-sha256', required=True)
    p.add_argument('--exporter-source', type=Path, default=Path(__file__).with_name('export_dino0_cuts.py'))
    p.add_argument('--exporter-source-sha256', required=True)
    p.add_argument('--capture', type=Path, default=BASE / 'experiments/vggt-v100-dino0-intermediates-r1/capture.json')
    p.add_argument('--capture-sha256', default=CAPTURE_SHA)
    p.add_argument('--check', type=Path)
    p.add_argument('--check-sha256')
    p.add_argument('--compiler-source', type=Path, default=BASE / 'experiments/vggt-v99-source-r1/compile_native_block_r1.py')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    exporter_path = Path(a.exporter_source).resolve(strict=True)
    require(sha(exporter_path) == a.exporter_source_sha256, 'Exporter source must be pinned before import')
    E = module(exporter_path)
    E.board_guard()
    out = E.fresh_output(a.output_dir)
    started = time.perf_counter()
    old_env, old_affinity, old_cwd = dict(os.environ), set(os.sched_getaffinity(0)), Path.cwd()
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    watched, session_drops, release_codes = {}, [], []
    r = dict(schema=CHECK_SCHEMA if a.mode == 'check' else COMPILE_SCHEMA, mode=a.mode, status='failed',
        script_sha256=sha(__file__), exporter_source_sha256=a.exporter_source_sha256,
        original_model_only=True, checkpoint_deserialized=False, fixed_gate=FIXED_GATE,
        strict_comparison=STRICT_COMPARISON, original_FP32_gate_passed=False, candidate_accepted=False,
        full_model_or_SLAM_or_realtime_validated=False, cleanup_errors=[], calls=[], sdk_released=False,
        ORT_execution=a.mode == 'check', SDK_build_execution=a.mode == 'compile', GPU_execution=False,
        calibration_executed=False, quantization_applied=False, system_or_frequency_writes=False)
    def progress(stage):
        r.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        atomic(out / 'result.json', r)
        print(json.dumps(dict(stage=stage)), flush=True)
    def pin(path, digest, size=None):
        path = E.pin(path, digest, size)
        watched[path] = digest
        return path
    def interrupt(number, frame):
        raise InterruptedError('External bounded stage terminated worker')
    try:
        for s in old_handlers:
            signal.signal(s, interrupt)
        require({4, 5, 6, 7} <= old_affinity, 'A76 CPU availability differs')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        pin(__file__, r['script_sha256'])
        pin(a.exporter_source, a.exporter_source_sha256)
        ep = pin(a.export, a.export_sha256)
        e = json.loads(ep.read_text())
        require(e['schema'] == E.SCHEMA and e['status'] == E.SUCCESS and
            e['script_sha256'] == a.exporter_source_sha256 and e['checkpoint'] == E.CHECKPOINT and
            e['source_and_inputs_unchanged'] and not e['cleanup_errors'] and
            all(e['proof'][k] is True for k in PROOF_KEYS) and e['proof']['original_parent_nodes'] == 31 and
            e['proof']['original_parent_initializers'] == 23, 'Successful exact original cut export required')
        parent = pin(e['parent']['file'], E.PARENT_SHA, E.PARENT_BYTES)
        oe = pin(e['original_export_receipt']['file'], E.ORIGINAL_EXPORT_SHA)
        cp = pin(a.capture, CAPTURE_SHA)
        require(a.capture_sha256 == CAPTURE_SHA, 'Exact frozen R100 original CPU capture required')
        c = json.loads(cp.read_text())
        require(c['schema'] == 'v100_original_native518_DINO0_intermediate_capture' and
            c['status'] == 'completed_original_DINO0_intermediates' and c['checkpoint'] == E.CHECKPOINT and
            c['calls'] == 1 and c['source_and_inputs_unchanged'] and c['hooks_removed'] and
            c['input_unchanged'] and c['original_parameter_state_unchanged'] and
            not c['source_class_forward_methods_overridden'] and not c['cleanup_errors'] and
            c['final_comparison']['byte_equal_v99'] and c['final_comparison']['raw_sha256_equal_v99'] and
            c['original_parameters'] == e['original_parameter_metadata'], 'Clean same-original CPU reference required')
        import numpy as np
        import onnx
        models, expected_rows, proof, learned = E.split_graphs(onnx.load_model(parent), json.loads(oe.read_text()))
        require(e['proof'] == proof and e['original_parameter_metadata'] == learned and
            [row['key'] for row in e['models']] == ['prefix', 'middle', 'suffix'], 'Export proof/ordering differs')
        paths = {}
        for row, expected in zip(e['models'], expected_rows):
            require(all(row[k] == expected[k] for k in expected) and
                row['onnx'] == {k: row[k] for k in ('file', 'bytes', 'sha256')}, 'Original cut descriptor differs')
            path = pin(child(ep.parent, row['file']), row['sha256'], row['bytes'])
            require(path.read_bytes() == models[row['key']].SerializeToString(), 'Original cut protobuf differs')
            paths[row['key']] = path
        del models
        refs = {}
        for key, shape in (('input', E.INPUT), ('output', E.INPUT), ('attn.qkv', E.QKV), ('attn.proj.input', E.INPUT)):
            row = c['artifacts'][key]
            require(row['shape'] == list(shape), 'Original reference shape differs')
            refs[key] = array_file(child(cp.parent, row['file']), row, watched)
        r.update(parent=e['parent'], export_receipt=dict(file=str(ep), sha256=a.export_sha256),
            capture=dict(file=str(cp), sha256=CAPTURE_SHA), checkpoint=E.CHECKPOINT,
            models=e['models'], proof=e['proof'], original_parameter_metadata=e['original_parameter_metadata'],
            original_source_hashes=e['original_source_hashes'])
        actual = {}
        if a.mode == 'check':
            require(os.statvfs(out).f_bavail * os.statvfs(out).f_frsize >= 128 << 20, 'Need 128 MiB free for gate evidence')
            import onnxruntime as ort
            r['ORT_version'] = ort.__version__
            def run(path, inputs, outputs):
                session = None
                reference = None
                try:
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = options.inter_op_num_threads = 1
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                    reference = weakref.ref(session)
                    require([v.name for v in session.get_inputs()] == list(inputs) and
                        [v.name for v in session.get_outputs()] == list(outputs), 'FP32 graph IO order differs')
                    result = session.run(list(outputs), inputs)
                    require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in result),
                        'Actual ORT output must remain finite FP32 without casting')
                    return [np.array(v, copy=True, order='C') for v in result]
                finally:
                    session = None
                    gc.collect()
                    dropped = reference is None or reference() is None
                    session_drops.append(dropped)
                    require(dropped, 'ORT session still retained')
            progress('FP32_parent')
            actual['parent'] = run(parent, {'input': refs['input']}, ('output',))[0]
            progress('FP32_prefix')
            actual['prefix'] = run(paths['prefix'], {'input': refs['input']}, ('attn.qkv',))[0]
            progress('FP32_original_middle12')
            actual['middle'] = run(paths['middle'], {'attn.qkv': actual['prefix']}, ('attention_heads_flatten',))[0]
            progress('FP32_suffix_actual_middle')
            actual['composed'] = run(paths['suffix'], {'input': refs['input'],
                'attention_heads_flatten': actual['middle']}, ('output',))[0]
            saved = {}
            for key, value in actual.items():
                file = out / (key + '-output.npy')
                with file.open('xb') as stream:
                    np.save(stream, value, allow_pickle=False)
                saved[key] = dict(file=file.name, bytes=file.stat().st_size, sha256=sha(file),
                    raw_sha256=raw(value), tensor_bytes=value.nbytes, shape=list(value.shape),
                    dtype=str(value.dtype), finite=bool(np.isfinite(value).all()))
            r['actual_outputs'] = saved
        else:
            require(a.check is not None and a.check_sha256 is not None, 'Pinned successful board check required before SDK compile')
            gp = pin(a.check, a.check_sha256)
            g = json.loads(gp.read_text())
            require(g['schema'] == CHECK_SCHEMA and g['status'] == CHECK_SUCCESS and g['mode'] == 'check' and
                g['script_sha256'] == r['script_sha256'] and g['exporter_source_sha256'] == a.exporter_source_sha256 and
                g['export_receipt'] == r['export_receipt'] and g['capture'] == r['capture'] and
                g['models'] == e['models'] and g['proof'] == e['proof'] and g['fixed_gate'] == FIXED_GATE and
                g['strict_comparison'] == STRICT_COMPARISON and g['original_FP32_gate_passed'] and
                g['ORT_sessions_dropped'] and g['source_and_inputs_unchanged'] and not g['cleanup_errors'] and
                all(g[k] for k in ('affinity_restored', 'environment_restored', 'cwd_restored')),
                'Successful same-source complete board FP32 gate required')
            require(set(g['actual_outputs']) == {'parent', 'prefix', 'middle', 'composed'}, 'All actual FP32 outputs required')
            for key, row in g['actual_outputs'].items():
                actual[key] = array_file(child(gp.parent, row['file']), row, watched)
            r['check_receipt'] = dict(file=str(gp), sha256=a.check_sha256)
        checks = dict(parent_vs_original=compare(actual['parent'], refs['output']),
            prefix_QKV_vs_original=compare(actual['prefix'], refs['attn.qkv']),
            middle_attention_vs_original=compare(actual['middle'], refs['attn.proj.input']),
            composed_vs_original=compare(actual['composed'], refs['output']),
            composed_vs_parent=compare(actual['composed'], actual['parent']))
        r['checks'] = checks
        require(tuple(checks) == CHECK_IDS and all(v['fixed_gate_passed'] for v in checks.values()), 'Unchanged original FP32 gate failed')
        if a.mode == 'compile':
            for key in CHECK_IDS:
                old = g['checks'][key]
                require(old['fixed_gate_passed'] and old['strict_pass'] == checks[key]['strict_pass'] and
                    old['outside_count'] == checks[key]['outside_count'] and old['byte_equal'] == checks[key]['byte_equal'] and
                    math.isclose(old['relative_l2'], checks[key]['relative_l2'], rel_tol=1e-10, abs_tol=1e-14) and
                    math.isclose(old['max_abs'], checks[key]['max_abs'], rel_tol=1e-12, abs_tol=1e-14),
                    'Saved board FP32 metric differs')
        r['original_FP32_gate_passed'] = True
        r['FP32_composition_uses_actual_prefix_and_middle_not_teacher_substitution'] = True
        if a.mode == 'check':
            r['status'] = CHECK_SUCCESS
        else:
            compiler_source = pin(a.compiler_source, COMPILER_SOURCE_SHA)
            original_config(compiler_source)
            require(os.statvfs(out).f_bavail * os.statvfs(out).f_frsize >= 1 << 30, 'Need 1 GiB free for owned compile scratch')
            os.chdir(out)
            os.environ['TMPDIR'] = str(out)
            require(importlib.metadata.version('rknn-toolkit2') == '2.3.2', 'Exact SDK 2.3.2 required')
            from rknn.api import RKNN
            r.update(configuration=CONFIG, build_options=BUILD, compiler_source_sha256=COMPILER_SOURCE_SHA,
                SDK_version='2.3.2', SDK_verbose=False, scratch_cwd=str(out), scratch_TMPDIR=str(out))
            compiled = []
            for row in e['models']:
                if row['key'] == 'middle':
                    continue
                progress('compile_' + row['key'])
                sdk = None
                try:
                    sdk = RKNN(verbose=False, verbose_file=str(out / (row['key'] + '-sdk.log')))
                    operations = (('config', lambda: sdk.config(**CONFIG)),
                        ('load_onnx', lambda: sdk.load_onnx(model=str(paths[row['key']]))),
                        ('build', lambda: sdk.build(**BUILD)),
                        ('export_rknn', lambda: sdk.export_rknn(str(out / (row['key'] + '.rknn')))))
                    for name, operation in operations:
                        tick = time.perf_counter()
                        code = operation()
                        r['calls'].append(dict(model=row['key'], operation=name, returncode=code,
                            seconds=time.perf_counter() - tick))
                        require(code == 0, 'SDK ' + name + ' failed for ' + row['key'])
                    file = out / (row['key'] + '.rknn')
                    item = dict(row)
                    item['rknn'] = dict(file=file.name, bytes=file.stat().st_size, sha256=sha(file))
                    compiled.append(item)
                finally:
                    if sdk is not None:
                        try:
                            code = sdk.release()
                            release_codes.append(code)
                            require(code in (None, 0), 'SDK release failed')
                        except BaseException as error:
                            r['cleanup_errors'].append('SDK release: ' + str(error))
                        sdk = None
            require(len(compiled) == 2 and len(release_codes) == 2 and not r['cleanup_errors'], 'Both original cuts must compile and release')
            r.update(models=compiled, middle_not_compiled=True, sdk_released=True, status=COMPILE_SUCCESS)
    except BaseException as error:
        r.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        r['ORT_sessions_dropped'] = all(session_drops)
        r['ORT_sessions_created'] = len(session_drops)
        r['sdk_release_codes'] = release_codes
        for path, digest in watched.items():
            if not path.is_file() or sha(path) != digest:
                r['cleanup_errors'].append('Pinned source/input changed: ' + str(path))
        r['source_and_inputs_unchanged'] = not r['cleanup_errors']
        try:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)
            os.sched_setaffinity(0, old_affinity)
            for number, handler in old_handlers.items():
                signal.signal(number, handler)
            r.update(cwd_restored=Path.cwd() == old_cwd, environment_restored=dict(os.environ) == old_env,
                affinity_restored=set(os.sched_getaffinity(0)) == old_affinity)
            require(r['cwd_restored'] and r['environment_restored'] and r['affinity_restored'], 'Process state restoration failed')
        except BaseException as error:
            r['cleanup_errors'].append('Process restoration: ' + str(error))
        if r['cleanup_errors']:
            r['status'] = 'failed_cleanup'
        progress('finished')
    return 0 if r['status'] in (CHECK_SUCCESS, COMPILE_SUCCESS) else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
