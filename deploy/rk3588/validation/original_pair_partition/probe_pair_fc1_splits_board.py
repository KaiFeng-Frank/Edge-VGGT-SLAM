#!/usr/bin/env python3
"""Original pair00 versus FC1 split2/split4; reuse the published half-IO probe."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
HELPER_SHA = 'f3801ba5b92c1043aab2d3ebc6c0ad3258b8633c25e216b1298fbb1e0c9c38a0'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
PARENT_ONNX_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
BUILD_SCRIPT_SHA = '5004ca5376873259658d840f6f8ace67756571452fc0a3e97c66218d1c3b071d'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_source(path, digest, name, pins):
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(value == digest, 'Frozen helper changed: ' + str(path))
    pins[Path(path).resolve()] = digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--parent-manifest', type=Path, default=BASE / 'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--compiled', nargs=2, action='append', required=True, metavar=('DIRECTORY', 'RECEIPT_SHA'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--rounds', type=int, choices=range(1, 21), default=2,
                   help='Normal ABCCBA rounds per input; profile always uses ABC')
    a = p.parse_args()
    require(len(a.compiled) == 2, 'Exactly split2 and split4 compiled candidates required')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    pins, contexts, api, base = {}, [], None, None
    affinity = os.sched_getaffinity(0)
    r = dict(schema='v70_original_pair00_FC1_splits_FP16_IO_probe', status='failed',
        original_model_only=True, candidate_accepted=False, full_model_or_realtime_validated=False,
        profile=a.profile, init_flags=8 if a.profile else 0, core_mask=7,
        normal_benchmark_timing=not a.profile, system_clocks_changed=False,
        expected_frequency_hz=300000000, frequency_scope='Before/after SDK-call samples only',
        routes={'A': 'original_pair00', 'B': 'split2', 'C': 'split4'},
        schedule=list('ABC' if a.profile else 'ABCCBA' * a.rounds), hidden_warmup_calls=0,
        normal_rounds=None if a.profile else a.rounds,
        first_exclusion_scope='First call per input per graph; all subsequent calls in normal mode are warm',
        calls=[], queries=[], cleanup_errors=[], contexts_created=0, contexts_destroyed=0,
        counters=dict(sdk_run_attempts=0, sdk_runs_succeeded=0, outputs_acquired=0, outputs_released=0),
        timing_scope='One complete two-output pair per call; NEON conversion, SDK run, owned half copy, release, widening and finite checks included; platform reads/hash/NPZ save/profile queries excluded')
    try:
        h = load_source(a.repo / 'deploy/rk3588/validation/original_pair_group/probe_original_pair_group.py',
                        HELPER_SHA, '_v70_half_IO', pins)
        r['script_sha256'] = h.sha(__file__); r['helper_sha256'] = HELPER_SHA
        pin, artifact = lambda path, digest: h.pin(path, digest, pins), lambda root, row: h.artifact(root, row, pins)
        base = load_source(a.repo / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', h.ABI_SHA, '_v70_ABI', pins)
        r['board'] = base.require_board()
        require({4, 5, 6, 7} <= affinity, 'Original CPU cores unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7}); r['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        import numpy as np
        parent_path = pin(a.parent_manifest, PARENT_SHA)
        parent = json.loads(parent_path.read_text()); original = parent['blocks'][0]
        require(original['role'] == 'pair' and original['index'] == 0 and original['sha256'] == PARENT_ONNX_SHA,
                'Fixed original pair00 graph required')
        expected = original['output_bindings']
        require(expected == [dict(index=i, name=role + '_output', shape=list(shape), dtype='float32')
            for i, (role, shape) in enumerate((('frame', h.SHAPE), ('global', (1, 106, 1024))))], 'Original two-output ABI differs')
        models = [('A', artifact(parent_path.parent, original['rknn']))]
        candidates = {}
        for directory, digest in a.compiled:
            directory = Path(directory).resolve(strict=True)
            row = json.loads(pin(directory / 'result.json', digest).read_text())
            require(row['schema'] == 'v70_original_VGGT_pair00_FC1_split_build'
                and row['status'] == 'compiled_not_quality_accepted' and row['chunks'] in (2, 4)
                and row['script_sha256'] == BUILD_SCRIPT_SHA
                and row['original_parent_manifest_sha256'] == PARENT_SHA
                and row['original_pair_onnx_sha256'] == PARENT_ONNX_SHA and row['parent'] == original
                and row['configuration'] == CONFIG and row['build_options'] == BUILD
                and row['released'] is True and row['release_code'] in (None, 0)
                and not row['cleanup_errors'] and row['source_and_inputs_unchanged'] and row['environment_restored'],
                'Completed original nonquantized compile provenance differs')
            require([x['name'] for x in row['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
                and all(x['code'] in (None, 0) and x['status'] == 'completed' for x in row['operations']), 'Incomplete compile operations')
            checked = json.loads(pin(directory / 'check.json', row['check_receipt_sha256']).read_text())
            require(checked['schema'] == 'v70_original_VGGT_pair00_FC1_split_ONNX_check'
                and checked['status'] == 'passed_original_FP32_ONNX_equivalence_not_quality_accepted'
                and checked['chunks'] == row['chunks'] and checked['script_sha256'] == BUILD_SCRIPT_SHA
                and checked['original_parent_manifest_sha256'] == PARENT_SHA
                and checked['original_pair_onnx_sha256'] == PARENT_ONNX_SHA
                and checked['candidate_onnx_sha256'] == row['onnx']['sha256']
                and checked['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] is True
                and not checked['cleanup_errors']
                and checked['cases'] == row['cases'] and len(checked['cases']) == 2,
                'Passed same-candidate original two-case ONNX gate required')
            require(all(len(case['checks']) == 2 and all(x['versus_baseline_FP32_ONNX']['strict_pass'] is True
                for x in case['checks']) for case in checked['cases']), 'Original output FP32 ONNX gate failed')
            artifact(directory, row['onnx'])
            require(row['chunks'] not in candidates, 'Duplicate candidate partition')
            candidates[row['chunks']] = (directory, row, checked)
        require(set(candidates) == {2, 4}, 'Both split2 and split4 required')
        for chunks, role in ((2, 'B'), (4, 'C')):
            directory, row, _ = candidates[chunks]
            models.append((role, artifact(directory, row['rknn'])))
        case_inputs = []
        for first, second in zip(candidates[2][2]['cases'], candidates[4][2]['cases']):
            require(first['tag'] == second['tag'] and first['saved_input']['raw_sha256'] == second['saved_input']['raw_sha256'], 'Candidates must use the same real inputs')
            artifact(candidates[4][0], second['saved_input'])
            path = artifact(candidates[2][0], first['saved_input']); x = np.load(path, allow_pickle=False)
            require(x.shape == h.SHAPE and x.dtype == np.float32 and x.flags.c_contiguous and np.isfinite(x).all()
                and hashlib.sha256(x.tobytes()).hexdigest() == first['saved_input']['raw_sha256'], 'Real pair input identity differs')
            refs = [np.load(artifact(first['directory'], item), allow_pickle=False) for item in first['original_outputs']] if first['original_outputs'] is not None else None
            case_inputs.append((first, x, refs))
        require([c[0]['tag'] for c in case_inputs] == ['v53', 'v56-calib00']
            and case_inputs[0][2] is not None and len(case_inputs[0][2]) == 2 and case_inputs[1][2] is None
            and case_inputs[0][0]['saved_input']['raw_sha256'] != case_inputs[1][0]['saved_input']['raw_sha256'], 'Two distinct real cases; only first has original Torch references')
        r['input_cases'] = [c[0] for c in case_inputs]
        r['parent_manifest_sha256'] = PARENT_SHA
        for relative, digest in parent['source_hashes'].items(): pin(a.repo / relative, digest)
        pin(a.repo / 'deploy/rk3588/reference/rknn_api_v2.3.2.h', h.HEADER_SHA)
        api = base.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so', h.SDK_SHA))))
        neon = ct.CDLL(str(pin(BASE / 'experiments/vggt-v56-neon-build-r1/fp16_neon.so', h.NEON_SHA)))
        for name in ('f32_to_f16', 'f16_to_f32'):
            fn = getattr(neon, name); fn.argtypes, fn.restype = [ct.c_void_p, ct.c_void_p, ct.c_size_t], None
        routes = {}
        for role, model in models:
            context = base.CONTEXT(); contexts.append(context)
            file = ct.create_string_buffer(os.fsencode(model))
            code = api.rknn_init(ct.byref(context), ct.cast(file, ct.c_void_p), 0, r['init_flags'], None)
            r['contexts_created'] += int(bool(context.value)); base.check_call(code, 'rknn_init')
            require(bool(context.value), 'Empty context')
            base.check_call(api.rknn_set_core_mask(context, 7), 'rknn_set_core_mask')
            version = base.query(api, context, 5, base.SDKVersion())
            require(re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()), 'SDK2.3.2 required')
            counts = base.query(api, context, 0, base.InputOutputNum())
            require((counts.n_input, counts.n_output) == (1, 2), 'Exactly the original two outputs required')
            records = []
            for command, bindings in ((1, [dict(index=0, name='input', shape=list(h.SHAPE))]), (2, expected)):
                for binding in bindings:
                    attr = base.query(api, context, command, base.TensorAttr(index=binding['index']))
                    saved_shape = base.SHAPE
                    try: base.SHAPE = tuple(binding['shape']); desc = base.validate_attr(attr)
                    finally: base.SHAPE = saved_shape
                    require(desc['index'] == binding['index'] and desc['name'] == binding['name'] and desc['type'] == 1
                            and desc['format'] == 3 and desc['pass_through'] == 0, 'Dense original FP16 descriptor differs')
                    records.append(desc)
            mem = base.query(api, context, 6, base.MemSize())
            r['queries'].append(dict(role=role, model_sha256=h.sha(model), input=records[0], outputs=records[1:],
                SDK_version=bytes(version.api_version).decode(), driver_version=bytes(version.drv_version).decode(),
                memory=dict(weight_bytes=int(mem.total_weight_size), internal_bytes=int(mem.total_internal_size), dma_bytes=int(mem.total_dma_allocated_size))))
            routes[role] = context
        for case, x, refs in case_inputs:
            first = {}
            for sequence, role in enumerate(r['schedule']):
                before = h.sample(); require(before['actual_frequency_hz'] == 300000000, 'NPU is not at original300MHz')
                y, timing = h.execute(base, api, neon, np, routes[role], x, expected, r['counters'], r['cleanup_errors'])
                after = h.sample(); require(after['actual_frequency_hz'] == 300000000, 'NPU frequency changed')
                cold = role not in first
                if cold: first[role] = y
                path = a.output_dir / f'{case["tag"]}-{sequence:02d}-{role}.npz'
                with path.open('xb') as stream: np.savez(stream, output0=y[0], output1=y[1])
                call = dict(input=case['tag'], sequence=sequence, role=role, first_for_input_role=cold,
                    component_seconds=timing['route_seconds'], timing=timing, platform_before=before, platform_after=after,
                    original_FP32=[h.compare(np, v, ref) for v, ref in zip(y, refs)] if refs else None,
                    same_input_first_A=[h.compare(np, v, ref) for v, ref in zip(y, first['A'])],
                    repeat_first_role=[h.compare(np, v, ref) for v, ref in zip(y, first[role])],
                    artifact=dict(file=path.name, bytes=path.stat().st_size, sha256=h.sha(path)))
                if a.profile: call['profile'] = base.perf_detail(api, routes[role])
                r['calls'].append(call)
            require(hashlib.sha256(x.tobytes()).hexdigest() == case['saved_input']['raw_sha256'], 'Real input mutated')
        runs = 6 if a.profile else 12 * a.rounds
        require(r['counters'] == dict(sdk_run_attempts=runs, sdk_runs_succeeded=runs, outputs_acquired=2*runs, outputs_released=2*runs), 'SDK run/output invariant failed')
        if not a.profile:
            r['summary'] = {role: dict(warm_count=len([c for c in r['calls'] if c['role'] == role and not c['first_for_input_role']]),
                mean_component_seconds=float(np.mean([c['component_seconds'] for c in r['calls'] if c['role'] == role and not c['first_for_input_role']])),
                per_input={case['tag']: float(np.mean([c['component_seconds'] for c in r['calls'] if c['role'] == role and c['input'] == case['tag'] and not c['first_for_input_role']])) for case, _, _ in case_inputs}) for role in 'ABC'}
        r['status'] = 'completed_component_diagnostic_not_quality_accepted'
    except BaseException as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for context in reversed(contexts):
            if context.value:
                try:
                    base.check_call(api.rknn_destroy(context), 'rknn_destroy'); context.value = 0; r['contexts_destroyed'] += 1
                except BaseException as exc: r['cleanup_errors'].append('Context cleanup: ' + str(exc))
        r['live_contexts'] = sum(bool(x.value) for x in contexts)
        try: r['source_and_inputs_unchanged'] = bool(pins) and all(h.sha(path) == digest for path, digest in pins.items())
        except BaseException as exc: r['source_and_inputs_unchanged'] = False; r['cleanup_errors'].append('Identity: ' + str(exc))
        r['affinity_restored'] = False
        try: os.sched_setaffinity(0, affinity); r['affinity_restored'] = os.sched_getaffinity(0) == affinity
        except BaseException as exc: r['cleanup_errors'].append('Affinity: ' + str(exc))
        if r['cleanup_errors'] or r['live_contexts'] or not r['source_and_inputs_unchanged'] or not r['affinity_restored']:
            r['status'] = 'failed_cleanup_or_identity'
        temp = a.output_dir / 'result.json.tmp'; temp.write_text(json.dumps(r, indent=2, allow_nan=False) + '\n')
        temp.replace(a.output_dir / 'result.json')
        print(json.dumps({k: r.get(k) for k in ('status', 'error', 'summary', 'contexts_created', 'contexts_destroyed', 'cleanup_errors')}), flush=True)
    return 0 if r['status'] == 'completed_component_diagnostic_not_quality_accepted' else 1


if __name__ == '__main__':
    sys.exit(main())
