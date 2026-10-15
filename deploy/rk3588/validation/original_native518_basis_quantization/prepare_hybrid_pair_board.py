#!/usr/bin/env python3
"""Board-only manual hybrid step1 for one unchanged gated A/F/B native pair.

Only four calibration inputs are passed to SDK; two heldouts are checked and
excluded. This generates the SDK's real intermediate files for later selector
inspection. It neither selects FLOAT16 internal names nor builds/runs RKNN.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time
import traceback

COMPILER_SHA = '2bbf180a9a70444ba592b7898fa23b96b11f36da3859f07cc5b435f0a25acb11'
CHECKER_SHA = 'b0f4bd489d1e6ec8eb28869f609d7e48469aa9d19b2120d7906cd9284059bc4c'
DESIGN_SHA = 'fad42daf567a1ca3d0064d932e3890f1de19367fcc54c7e4b1480e8ca89c0e61'
STEP1 = dict(proposal=False, proposal_dataset_size=1, custom_hybrid=None, rknn_batch_size=None)
SUCCESS = 'generated_manual_hybrid_step1_not_compiled_or_accepted'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('A', 'F', 'B'), required=True)
    p.add_argument('--capture-receipts', type=Path, nargs=3, required=True)
    p.add_argument('--capture-sha256', nargs=3, required=True)
    p.add_argument('--check-receipts', type=Path, nargs=3, required=True)
    p.add_argument('--check-sha256', nargs=3, required=True)
    p.add_argument('--checker-source', type=Path, required=True)
    p.add_argument('--compiler-source', type=Path, required=True)
    p.add_argument('--boundary-design', type=Path, required=True)
    p.add_argument('--export-receipt', type=Path, default=Path('/tmp/edge-vggt-v126-basis-export-r2/result.json'))
    p.add_argument('--original-check-receipt', type=Path, default=Path('/tmp/edge-vggt-v126-basis-check-r2/result.json'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--minimum-free-mib', type=int, default=1024)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'SDK calibration is RK3588S-board-only')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False); (out / 'tmp').mkdir()
    source = Path(__file__).resolve(); watched = {source: sha(source)}
    old_env, old_cwd, old_affinity = dict(os.environ), Path.cwd(), set(os.sched_getaffinity(0))
    handlers, sdk = {}, None; began = time.monotonic()
    report = dict(schema='v128_original_native518_basis_pair_manual_hybrid_step1', status='running', script_sha256=watched[source],
        variant=a.variant, actual_calibration_on_board_only=True, runtime_inference_invoked=False,
        RKNN_build_or_step2_or_export_invoked=False, original_ONNX_modified=False, candidate_accepted=False,
        protected_internal_selectors_selected=False, protected_precision_placement_established=False,
        system_or_frequency_writes=False, calls=[], cleanup_errors=[], sdk_initialized=False, sdk_released=False)

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - began, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        t = out / 'result.tmp'; t.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); t.replace(out / 'result.json')
        print(stage, flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(isinstance(digest, str) and len(digest) == 64 and sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
        require(path not in watched or watched[path] == digest, 'Conflicting file identity')
        watched[path] = digest; return path

    def module(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

    def relative(folder, file):
        require(isinstance(file, str) and not Path(file).is_absolute() and '..' not in Path(file).parts, 'Receipt-relative artifact required')
        return folder / file

    progress('preflight')
    try:
        def interrupted(number, frame): raise InterruptedError('Controller signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require({4, 5, 6, 7} <= old_affinity and a.minimum_free_mib >= 0, 'A76 affinity/storage request differs')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', TMPDIR=str(out / 'tmp'))
        os.chdir(out)
        C = module(a.compiler_source, COMPILER_SHA, '_v128_frozen_compile_authority')
        V = module(a.checker_source, CHECKER_SHA, '_v128_frozen_metadata_validator')
        design = json.loads(pin(a.boundary_design, DESIGN_SHA).read_text())
        configuration = dict(C.CONFIG, quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
        require(configuration == design['minimal_config'], 'Exact documented W8A8 config differs')
        efile = pin(a.export_receipt, V.EXPORT_SHA); ofile = pin(a.original_check_receipt, V.CHECK_SHA)
        export, original_check = json.loads(efile.read_text()), json.loads(ofile.read_text())
        V.validate_original_pair(export, original_check)
        records, receipt_rows = [], []
        for batch, (cap_arg, cap_sha, chk_arg, chk_sha) in enumerate(zip(a.capture_receipts, a.capture_sha256, a.check_receipts, a.check_sha256)):
            cap_file, chk_file = pin(cap_arg, cap_sha), pin(chk_arg, chk_sha)
            cap, chk = json.loads(cap_file.read_text()), json.loads(chk_file.read_text()); V.validate_capture(cap)
            require(cap['batch_index'] == chk['batch_index'] == batch and chk['schema'] == 'v128_original_native518_captured_pair_FP32_check' and
                chk['status'] == V.SUCCESS and chk['script_sha256'] == CHECKER_SHA and chk['capture_receipt']['sha256'] == cap_sha and
                chk['capture_receipt']['script_sha256'] == V.CAPTURE_SOURCE_SHA and chk['records'] == cap['records'] and chk['models'] == export['models'] and
                chk['export_receipt']['sha256'] == V.EXPORT_SHA and chk['original_check_receipt']['sha256'] == V.CHECK_SHA and
                chk['fixed_gate'] == V.GATE and chk['strict_allclose_is_diagnostic'] and chk['FP32_gate_passed'] and chk['all6_calls_completed'] and
                chk['saved_array_count'] == 12 and chk['ORT_sessions_dropped'] and chk['source_and_inputs_unchanged'] and
                chk['affinity_restored'] and chk['environment_restored'] and not chk['cleanup_errors'], 'Clean matched original six-call FP32 check required')
            require([(v['input_id'], v['variant']) for v in chk['checks']] == [(r['id'], k) for r in cap['records'] for k in ('A', 'F', 'B')], 'Same inputs/all variants required')
            for row in chk['checks']:
                require(row['complete'] and row['input_unchanged'] and row['fixed_gate_passed'] and len(row['outputs']) == 2 and
                    set(row['comparisons_vs_A']) == {'frame_output', 'global_output'}, 'Incomplete same-input FP32 call')
                for m in row['comparisons_vs_A'].values():
                    require(m['fixed_gate_passed'] and math.isfinite(m['relative_l2']) and math.isfinite(m['max_abs']) and
                        m['relative_l2'] <= V.GATE['relative_l2_max'] and m['max_abs'] <= V.GATE['max_abs_max'], 'Unchanged numeric FP32 gate failed')
            require(chk['source_hashes'] == cap['source_hashes'] == export['source_hashes'] and chk['original_parameters'] == export['original_parameters'], 'Original math/parameter lineage differs')
            for row in cap['records']:
                item = copy.deepcopy(row); item['absolute_input_file'] = str(pin(relative(cap_file.parent, row['input']['file']), row['input']['sha256'], row['input']['bytes']))
                for image in item['source_images']: pin(image['file'], image['sha256'], image['bytes'])
                records.append(item)
            receipt_rows.append(dict(batch_index=batch, capture=dict(file=str(cap_file), sha256=cap_sha), check=dict(file=str(chk_file), sha256=chk_sha)))
        require([r['id'] for r in records] == ['pair' + str(i) for i in range(6)] and [r['split'] for r in records] == ['calibration'] * 4 + ['heldout'] * 2, 'Four calibration/two heldout contract differs')
        require(len({v['sha256'] for r in records for v in r['source_images']}) == 12 and len({r['input']['raw_sha256'] for r in records}) == 6, 'PNG or captured input partitions overlap')
        import numpy as np
        for row in records:
            x = np.load(row['absolute_input_file'], allow_pickle=False)
            require(x.dtype == np.float32 and list(x.shape) == V.FRAME and x.flags.c_contiguous and np.isfinite(x).all() and
                x.nbytes == row['input']['tensor_bytes'] and hashlib.sha256(x.tobytes(order='C')).hexdigest() == row['input']['raw_sha256'], 'Canonical original calibration/evaluation input differs')
            del x
        dataset = out / 'dataset.txt'; dataset.write_text(''.join(r['absolute_input_file'] + '\n' for r in records[:4]))
        watched[dataset] = sha(dataset)
        require(set(dataset.read_text().splitlines()) == {r['absolute_input_file'] for r in records[:4]} and
                not set(dataset.read_text().splitlines()) & {r['absolute_input_file'] for r in records[4:]}, 'Heldouts must not enter SDK calibration')
        model = next(m for m in export['models'] if m['key'] == a.variant)
        graph_file = pin(relative(efile.parent, model['onnx']['file']), model['onnx']['sha256'], model['onnx']['bytes'])
        import onnx
        graph = onnx.load_model(str(graph_file), load_external_data=False)
        require(all(t.data_location != onnx.TensorProto.EXTERNAL for t in graph.graph.initializer), 'Self-contained original graph required')
        onnx.checker.check_model(graph, full_check=True)
        edges = [dict(index=i, name=n.name, op_type=n.op_type, inputs=list(n.input), outputs=list(n.output)) for i, n in enumerate(graph.graph.node)]
        index_file = out / 'public-graph-semantic-index.json'
        index = dict(schema='v128_public_graph_semantic_index_not_internal_selector_map', variant=a.variant,
            graph_sha256=model['onnx']['sha256'], input_bindings=model['input_bindings'], output_bindings=model['output_bindings'], nodes=edges,
            initializer_shapes={t.name: list(t.dims) for t in graph.graph.initializer},
            residual_add_role_candidates=[v for v in edges if v['op_type'] == 'Add'],
            basis_boundary_nodes=[v for v in edges if v['name'].startswith('__v126_')],
            protected_role_policy=design['common_A_F_B_precision_policy'],
            internal_names_not_assumed=True, selector_mapping_requires_actual_generated_model=True)
        index_file.write_text(json.dumps(index, indent=2) + '\n'); watched[index_file] = sha(index_file)
        del graph, edges; gc.collect()
        versions = {name: importlib.metadata.version(name) for name in ('rknn-toolkit2', 'onnx', 'numpy')}
        require(versions['rknn-toolkit2'].split('+')[0] == '2.3.2', 'RKNN toolkit2.3.2 required')
        free = shutil.disk_usage(out).free; require(free >= a.minimum_free_mib * (1 << 20), 'Insufficient private SDK scratch space')
        report.update(configuration=configuration, configuration_delta={k: configuration[k] for k in configuration.keys() - C.CONFIG.keys()},
            configuration_authority_source_sha256=COMPILER_SHA, checker_source_sha256=CHECKER_SHA, boundary_design_sha256=DESIGN_SHA,
            original_export_receipt_sha256=V.EXPORT_SHA, original_check_receipt_sha256=V.CHECK_SHA, three_clean_capture_check_receipts=receipt_rows,
            all18_new_FP32_calls_passed=True, checkpoint=export['checkpoint'], original_parameters=export['original_parameters'],
            graph=dict(file=str(graph_file), **{k: model['onnx'][k] for k in ('bytes', 'sha256')}), input_bindings=model['input_bindings'], output_bindings=model['output_bindings'],
            calibration_records=records[:4], heldout_records=records[4:], calibration_evaluation_disjoint=True, heldouts_excluded_from_dataset=True,
            dataset=dict(file=dataset.name, bytes=dataset.stat().st_size, sha256=sha(dataset), line_count=4, input_count=1, canonical_unrotated_inputs=True),
            public_semantic_index=dict(file=index_file.name, bytes=index_file.stat().st_size, sha256=sha(index_file)), versions=versions,
            minimum_free_bytes=a.minimum_free_mib * (1 << 20), pre_SDK_free_bytes=free, step1_options=dict(dataset=str(dataset), **STEP1))
        progress('all_original_gates_verified_before_SDK_import')
        import rknn.api as api
        api_file = Path(api.__file__).resolve(); watched[api_file] = sha(api_file)
        signature = str(inspect.signature(api.RKNN.hybrid_quantization_step1)); report['actual_step1_signature'] = signature
        require(set(STEP1) | {'dataset'} <= set(inspect.signature(api.RKNN.hybrid_quantization_step1).parameters), 'Documented step1 interface unavailable')
        sdk = api.RKNN(verbose=False, verbose_file=str(out / 'sdk.log')); report['sdk_initialized'] = True
        for name, action in [('config', lambda: sdk.config(**configuration)), ('load_onnx', lambda: sdk.load_onnx(model=str(graph_file))),
                             ('hybrid_quantization_step1', lambda: sdk.hybrid_quantization_step1(dataset=str(dataset), **STEP1))]:
            call = dict(operation=name, status='running'); report['calls'].append(call); progress(name + '_start')
            tick = time.monotonic(); code = action(); call.update(returncode=code, seconds=time.monotonic() - tick, status='returned')
            require(code in (0, None), 'SDK ' + name + ' failed: ' + str(code)); progress(name + '_completed')
        report.update(status=SUCCESS, all_SDK_operations_succeeded=True)
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        if sdk is not None:
            report['sdk_release_called'] = True
            try: progress('SDK_release_start')
            except BaseException as error: report['cleanup_errors'].append('Pre-release receipt: ' + str(error))
            try:
                code = sdk.release(); report['sdk_release_code'] = code
                require(code in (0, None), 'SDK release failed'); report['sdk_released'] = True
            except BaseException as error: report['cleanup_errors'].append('SDK release: ' + str(error))
        try:
            files = [p for p in out.rglob('*') if p.is_file() and p.name not in ('result.json', 'result.tmp')]
            report['generated_files'] = [dict(file=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(files)]
            if report['status'] == SUCCESS:
                require(all(any(v['file'].endswith(suffix) for v in report['generated_files']) for suffix in ('.model', '.data', '.quantization.cfg')), 'Expected real step1 model/data/cfg files absent')
        except BaseException as error: report['cleanup_errors'].append('Generated files: ' + str(error))
        for label, action in [('cwd', lambda: os.chdir(old_cwd)), ('affinity', lambda: os.sched_setaffinity(0, old_affinity))]:
            try: action()
            except BaseException as error: report['cleanup_errors'].append(label + ': ' + str(error))
        os.environ.clear(); os.environ.update(old_env)
        for number, handler in handlers.items(): signal.signal(number, handler)
        try: report['source_and_inputs_unchanged'] = all(p.is_file() and sha(p) == digest for p, digest in watched.items())
        except BaseException as error:
            report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity: ' + str(error))
        report.update(cwd_restored=Path.cwd() == old_cwd, affinity_restored=set(os.sched_getaffinity(0)) == old_affinity,
            environment_restored=dict(os.environ) == old_env, pinned_files={str(p): digest for p, digest in watched.items()})
        if report['cleanup_errors'] or not all(report[k] for k in ('cwd_restored', 'affinity_restored', 'environment_restored', 'source_and_inputs_unchanged')) or (sdk is not None and not report['sdk_released']):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS and report['sdk_released'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
