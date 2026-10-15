#!/usr/bin/env python3
"""Independent saved NumPy audit of the complete DINO0 cut/shared-attention ABBA.

No ONNX, Torch, ORT, SDK, GPU or board imports/calls. Run only after the actual
export, FP32 check, compile, complete block outputs and control are pulled.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path('/dev/shm/edge-vggt-v120-review')
PRIOR = Path('/path/to/research-artifacts/v117-native518-attention-dump-r1')
SHARED_PRIOR = Path('/path/to/research-artifacts/v119-native518-shared-attention-r1')
DRIVER_SHA = 'b62356ad5f3256b0f6c447965fe61e7fb158f3c7748db2d83756341599226140'
EXPORTER_SHA = '3b61eddfd7cae104d0ad716b2b06b24d234dcea8f3ef94be0230e528d277752c'
EXPORT_SHA = '4290248a74a73e129b931192203a3aac01473542b26314b1b22f81f3afc0e4f7'
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
RUNTIME_SHA = '82ec9d4ad838c84dcd02309a9da7d73db68b77d30525d1308f3e482907844968'
BASE_RAW = 'e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554'
OLD_QKV_RAW = '1f09065df446dbbc56ca249b96156f7ec2c78b87555461c3ed33852e40f91936'
CF_SHA = '03e9a5797ca21c542e467a68057f7290f0a881a689b273db74f7f7d5d3dea94d'
SHARED_RESULT_SHA = '552cf6bb4e7a18448a9f34095105957ef6fd310674e81fa461c44d796a2d4dc0'
SHAPE, QKV = (1, 1041, 1024), (1, 1041, 3072)
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def raw(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def compare(value, reference):
    require(value.shape == reference.shape, 'Comparison shape differs')
    v, z = value.astype(np.float64), reference.astype(np.float64)
    delta = v - z
    zn, vn, dn = (float(np.linalg.norm(a)) for a in (z, v, delta))
    bad = np.abs(delta) > 1e-5 + 1e-4 * np.abs(z)
    relative = dn / zn if zn else (0.0 if not dn else None)
    maximum = float(np.abs(delta).max())
    return dict(relative_l2=relative, max_abs=maximum, l2_error=dn,
        reference_l2_norm=zn, cosine=float(np.vdot(v, z) / (vn * zn)) if vn and zn else None,
        strict_rtol=1e-4, strict_atol=1e-5, strict_pass=not bool(bad.any()),
        outside_count=int(bad.sum()), byte_equal=value.tobytes() == reference.tobytes(),
        fixed_gate_passed=relative is not None and relative <= 1e-5 and maximum <= 1e-3)


def agree(actual, reported):
    for key in ('relative_l2', 'max_abs', 'strict_pass', 'outside_count', 'byte_equal'):
        left, right = actual[key], reported[key]
        if key in ('relative_l2', 'max_abs') and left is not None:
            require(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-14), 'Reported metric differs: ' + key)
        else:
            require(left == right, 'Reported metric differs: ' + key)
    if 'fixed_gate_passed' in reported:
        require(actual['fixed_gate_passed'] == reported['fixed_gate_passed'], 'Reported fixed gate differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--prior', type=Path, default=PRIOR)
    p.add_argument('--shared-prior', type=Path, default=SHARED_PRIOR)
    p.add_argument('--driver-sha256', default=DRIVER_SHA)
    p.add_argument('--output', type=Path, default=ROOT / 'shared-block-saved-audit.json')
    a = p.parse_args()
    files, checked = [], dict(block=0, FP32_gate=0)

    def record(path, digest=None):
        actual = sha(path)
        require(digest is None or actual == digest, 'Pinned identity differs: ' + str(path))
        files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=actual))
        return path

    def receipt(path, digest=None):
        return json.loads(record(path, digest).read_text())

    def load(base, row, kind=None):
        require(Path(row['file']).name == row['file'], 'NPY must be a relative basename')
        path = base / row['file']
        require(path.stat().st_size == row['bytes'], 'NPY file size differs')
        record(path, row['sha256'])
        v = np.load(path, allow_pickle=False)
        require(list(v.shape) == row['shape'] and str(v.dtype) == row['dtype'] and
            np.isfinite(v).all() and raw(v) == row['raw_sha256'], 'NPY shape/dtype/finite/raw identity differs')
        if 'tensor_bytes' in row:
            require(v.nbytes == row['tensor_bytes'], 'Tensor byte count differs')
        files[-1].update(raw_sha256=raw(v), shape=list(v.shape), dtype=str(v.dtype), finite=True)
        if kind:
            checked[kind] += 1
        return v

    for name, digest in (('run_dino0_shared_block_board.py', a.driver_sha256),
            ('export_dino0_cuts.py', EXPORTER_SHA), ('shared_attention_runtime.py', RUNTIME_SHA)):
        ast.parse(record(a.root / name, digest).read_text())
    data = receipt(a.root / 'block-r1/result.json')
    control = receipt(a.root / 'control-r1/control.json')
    exported = receipt(a.root / 'export-r1/export.json', EXPORT_SHA)
    gated = receipt(a.root / 'check-r1/result.json', data['check_sha256'])
    compiled = receipt(a.root / 'compile-r1/result.json', data['compile_sha256'])
    require(data['script_sha256'] == a.driver_sha256 and data['runtime_sha256'] == RUNTIME_SHA and
        data['schema'] == 'v120_original_native518_complete_DINO0_shared_attention_ABBA' and
        data['status'] == 'completed_original_DINO0_component_ABBA_not_full_model_accepted' and
        data['export_sha256'] == EXPORT_SHA and not data['cleanup_errors'] and
        all(data[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored',
            'original_input_unchanged', 'all31_original_nodes_accounted', 'original_input_used_for_suffix_residual')) and
        data['reference_QKV_or_attention_substituted'] is False and data['candidate_accepted'] is False,
        'Clean complete original component execution required')
    require(exported['script_sha256'] == EXPORTER_SHA and exported['proof'] == data['original_graph_proof'] and
        [r['node_count'] for r in exported['models']] == [3, 12, 16] and
        [r['initializer_count'] for r in exported['models']] == [4, 6, 13] and
        len(exported['original_parameter_metadata']) == 14, 'Original complete cut/source proof differs')
    for row in exported['models']:
        path = a.root / 'export-r1' / row['file']
        require(path.stat().st_size == row['bytes'], 'Cut file size differs')
        record(path, row['sha256'])
    for r in (gated, compiled):
        require(r['export_receipt']['sha256'] == EXPORT_SHA and r['capture']['sha256'] == CAPTURE_SHA and
            r['fixed_gate'] == GATE and r['original_FP32_gate_passed'] and not r['cleanup_errors'] and
            all(r[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'cwd_restored')),
            'Linked clean original FP32 gate and compile required')
    require(compiled['check_receipt']['sha256'] == data['check_sha256'] and compiled['sdk_released'] and
        gated['ORT_sessions_dropped'] and compiled['models'][0]['key'] != compiled['models'][1]['key'],
        'Compilation/session ownership linkage differs')
    capture = receipt(a.prior / 'old-reference-r100/capture.json', CAPTURE_SHA)
    require(capture['final_comparison']['byte_equal_v99'] and capture['final_comparison']['raw_sha256_equal_v99'] and
        data['original_input'] == capture['artifacts']['input'] and
        data['original_output_reference'] == capture['artifacts']['output'], 'Matched original CPU reference differs')
    refs = {k: load(a.prior / 'old-reference-r100', capture['artifacts'][k])
        for k in ('input', 'output', 'attn.qkv', 'attn.proj.input')}
    x, ref = refs['input'], refs['output']
    require(x.shape == ref.shape == SHAPE and x.dtype == ref.dtype == np.float32, 'Native complete input/output differs')
    fp = {k: load(a.root / 'check-r1', row, 'FP32_gate') for k, row in gated['actual_outputs'].items()}
    gate_pairs = dict(parent_vs_original=(fp['parent'], ref), prefix_QKV_vs_original=(fp['prefix'], refs['attn.qkv']),
        middle_attention_vs_original=(fp['middle'], refs['attn.proj.input']), composed_vs_original=(fp['composed'], ref),
        composed_vs_parent=(fp['composed'], fp['parent']))
    fp_checks = {k: compare(*v) for k, v in gate_pairs.items()}
    for k, value in fp_checks.items():
        agree(value, gated['checks'][k]); agree(value, compiled['checks'][k])
        require(value['fixed_gate_passed'], 'Saved original FP32 composition fixed gate failed: ' + k)

    rows = data['calls']
    require(len(rows) == 8 and [r['index'] for r in rows] == list(range(8)) and
        ''.join(r['route'] for r in rows) == 'ABBAABBA', 'Complete ordered ABBAABBA required')
    values, timings, comparisons = {}, [], []
    first_b = next(r for r in rows if r['route'] == 'B')
    qkv = load(a.root / 'block-r1', first_b['actual_prefix_QKV'], 'block')
    attention = load(a.root / 'block-r1', first_b['actual_attention'], 'block')
    require(qkv.shape == QKV and attention.shape == SHAPE and qkv.dtype == attention.dtype == np.float32,
        'Actual complete candidate prefix/attention boundary differs')
    half_roundtrip = qkv.astype(np.float16).astype(np.float32).tobytes() == qkv.tobytes()
    for r in rows:
        route = r['route']
        require(r['first_for_route'] == (r['index'] == (0 if route == 'A' else 1)), 'First-route exclusion differs')
        v = load(a.root / 'block-r1', r['output'], 'block')
        require(v.shape == SHAPE and v.dtype == np.float32, 'Complete output shape/dtype differs')
        if route in values:
            require(v.tobytes() == values[route].tobytes(), 'Within-route repeat byte equality failed')
        else:
            values[route] = v
        if route == 'A':
            require(raw(v) == BASE_RAW, 'Original fused baseline bytes changed')
        else:
            require(r['actual_prefix_QKV_raw_sha256'] == raw(qkv) and
                r['actual_attention_raw_sha256'] == raw(attention) and
                r['prefix_QKV_Half_roundtrip_byte_equal'] == half_roundtrip,
                'Actual repeated prefix/attention/roundtrip boundary identity differs')
        whole, update = compare(v, ref), compare(v - x, ref - x)
        agree(whole, r['vs_original_CPU']); agree(update, r['residual_update_vs_original_CPU'])
        comparisons.append(dict(index=r['index'], route=route, whole_output=whole, residual_update=update))
        t = dict(index=r['index'], route=route, complete_block_seconds=r['complete_block_seconds'])
        if route == 'A':
            t['fused_boundary_seconds'] = r['stages']['fused']['complete_boundary_seconds']
            subtotal = t['fused_boundary_seconds']
        else:
            stages = r['stages']; heads = stages['attention']['heads']
            require([h['head'] for h in heads] == list(range(16)), 'All16 actual attention heads required')
            for key in ('prefix', 'suffix'):
                t[key + '_boundary_seconds'] = stages[key]['complete_boundary_seconds']
            t['complete_attention_seconds'] = stages['attention']['complete_attention_seconds']
            t['shared_softmax_bridge_seconds'] = sum(h['shared_softmax']['complete_bridge_seconds'] for h in heads)
            t['contained_GPU_kernel_seconds'] = sum(h['shared_softmax']['GPU_kernel_seconds'] for h in heads)
            for key in ('QK_pack_upload_bind_seconds', 'QK_SDK_run_seconds', 'PV_upload_bind_seconds',
                    'PV_SDK_run_seconds', 'PV_sync_owned_finite_seconds'):
                t[key] = sum(h[key] for h in heads)
            for key in ('QK_C_FROM_DEVICE_seconds', 'PV_A_prewrite_TO_DEVICE_seconds', 'finite_flag_reset_seconds',
                    'kernel_enqueue_seconds', 'kernel_finish_seconds', 'finite_flag_read_check_seconds',
                    'PV_A_postwrite_FROM_DEVICE_seconds', 'profiling_and_event_release_seconds'):
                t['shared_' + key] = sum(h['shared_softmax'][key] for h in heads)
            for h in heads:
                s = h['shared_softmax']; c = s['GPU_event_clocks_ns']
                require(s['finite_error_bits'] == 0 and c['queued'] <= c['submit'] <= c['start'] <= c['end'] and
                    math.isclose(s['GPU_kernel_seconds'], (c['end'] - c['start']) / 1e9, rel_tol=1e-12),
                    'GPU event/finite evidence differs')
            subtotal = t['prefix_boundary_seconds'] + t['complete_attention_seconds'] + t['suffix_boundary_seconds']
        for name in (('fused',) if route == 'A' else ('prefix', 'suffix')):
            for key in ('input_prepare_and_finite_seconds', 'inputs_set_seconds', 'SDK_run_seconds',
                    'outputs_get_seconds', 'all_owned_copies_seconds', 'outputs_release_seconds', 'output_finite_seconds'):
                t[name + '_' + key] = r['stages'][name][key]
        t['outer_other_seconds'] = t['complete_block_seconds'] - subtotal
        require(t['outer_other_seconds'] >= 0, 'Nested stages exceed complete boundary')
        timings.append(t)
    require(checked == dict(block=10, FP32_gate=4) and
        {p.name for p in (a.root / 'block-r1').glob('*.npy')} ==
        {r['output']['file'] for r in rows} | {first_b['actual_prefix_QKV']['file'], first_b['actual_attention']['file']},
        'All10 block NPYs and4 FP32 gate NPYs required')
    indices = {route: [r['index'] for r in rows if r['route'] == route and not r['first_for_route']] for route in 'AB'}
    means = {route: {k: statistics.mean(timings[i][k] for i in ids) for k in timings[ids[0]]
        if k not in ('index', 'route')} for route, ids in indices.items()}
    for route in 'AB':
        require(len(indices[route]) == 3 and math.isclose(means[route]['complete_block_seconds'],
            data['subsequent_three_means_seconds'][route], rel_tol=1e-12), 'Reported complete block mean differs')
    expected = dict(rknn_matmul_create=2, rknn_matmul_set_core_mask=2, rknn_create_mem=6,
        rknn_matmul_set_io_mem=262, rknn_mem_sync=512, rknn_matmul_run=128, rknn_destroy_mem=6, rknn_matmul_destroy=2)
    require(data['MatMul_attempts'] == data['MatMul_successes'] == expected, 'Actual128-product native ledger differs')
    weights = data['weighted_resources']
    require(set(weights) == {'baseline', 'prefix', 'suffix'}, 'All3 weighted owners required')
    for r in weights.values():
        require(r['closed'] and not r['failed'] and r['resource_counts_balanced'] and not r['cleanup_errors'] and
            r['live_contexts'] == r['live_output_buffers'] == 0 and r['contexts_created'] == r['contexts_destroyed'] == 1 and
            r['run_attempts'] == r['runs_completed'] == 4 and r['output_buffers_acquired'] == r['output_buffers_released'] == 4,
            'Weighted complete input/output/context ownership differs')
    clean, g = data['shared_cleanup'], data['GPU_after_close']
    require(clean['native_close_permitted_after_GPU_release'] and not clean['cleanup_errors'] and
        clean['calls_attempted'] == clean['calls_completed'] == 4 and len(clean['resources']) == 2 and
        all(not r['cleanup_errors'] and r['live_context'] == r['live_memories'] == 0 for r in clean['resources']),
        'Native allocations released before/without safe GPU release')
    require(g['fully_released'] and g['queue_idle'] and g['closed'] and not g['failed'] and not g['cleanup_errors'] and
        all(g[k] == 64 for k in ('calls_attempted', 'calls_completed', 'events_created', 'events_released')) and
        g['objects_created'] == g['objects_released'] == dict(Context=1, CommandQueue=1, Program=1, Kernel=1, MemObject=3) and
        g['live_OpenCL_objects'] == g['live_events'] == g['live_owned_DMA_fds'] == 0, 'GPU ownership/event ledger differs')
    lifecycle = sum(data['initialization_seconds'].values()) + data['destroy_seconds'] + sum(r['complete_block_seconds'] for r in rows)
    require(math.isclose(lifecycle, data['summed_eight_call_lifecycle_seconds'], rel_tol=1e-12), 'Full init/call/destroy accounting differs')
    require(control['child_returncode'] == 0 and not control['restore_errors'] and control['baseline'] == control['restored'] and
        control['frequency_governor_limits_restored'] and control['restored']['cur_freq'] == '300000000' and
        all(r['actual_frequency_hz'] == 1000000000 for r in control['observations']), 'Controller frequency/policy restoration differs')
    maximum = max(v for r in control['observations'] for v in r['temperatures_c'].values())
    require(maximum < 75, 'Sampled temperature exceeded controller hard stop')
    same_qkv = dict(available=False, actual_prefix_matches_old_QKV=raw(qkv) == OLD_QKV_RAW,
        note='Only compare to the old same-QKV CPU counterfactual if actual new prefix bytes match; no forced pairing.')
    if same_qkv['actual_prefix_matches_old_QKV']:
        cf = receipt(a.prior / 'qkv-counterfactual-r100/result.json', CF_SHA)
        z = load(a.prior / 'qkv-counterfactual-r100', cf['artifacts']['attn.proj.input'])
        same_qkv.update(available=True, actual_attention_vs_original_CPU_same_QKV=compare(attention, z))
        old = receipt(a.shared_prior / 'shared-attention-r1/result.json', SHARED_RESULT_SHA)
        require(old['QKV_raw_sha256'] == raw(qkv), 'Prior shared-attention comparison uses different QKV')
        z = load(a.shared_prior / 'shared-attention-r1', old['calls'][1]['output'])
        require(attention.tobytes() == z.tobytes(), 'Actual new attention differs from frozen R119 same-QKV B')
        same_qkv['actual_attention_byte_equal_frozen_R119_shared_output'] = True
    am, bm = (means[k]['complete_block_seconds'] for k in 'AB')
    result = dict(schema='v120_independent_saved_complete_DINO0_shared_attention_audit', status='passed',
        auditor_sha256=sha(__file__), execution='Saved NumPy, JSON and AST only; no ONNX/NN/SDK/GPU/board calls',
        checked_NPYs=checked, saved_FP32_composition_gate=fp_checks, all8_output_metric_records_recomputed=True,
        within_each_route_all4_outputs_byte_equal=True, A_matches_frozen_original_fused_output=True,
        actual_prefix_QKV_raw_sha256=raw(qkv), actual_attention_raw_sha256=raw(attention),
        actual_prefix_Half_roundtrip_byte_equal=half_roundtrip,
        actual_prefix_vs_original_CPU=compare(qkv, refs['attn.qkv']),
        actual_attention_vs_original_CPU_original_QKV=compare(attention, refs['attn.proj.input']),
        same_actual_QKV_optional_control=same_qkv, route_output_raw_sha256={k: raw(v) for k, v in values.items()},
        whole_output_vs_original_CPU={k: compare(v, ref) for k, v in values.items()},
        residual_update_vs_original_CPU={k: compare(v-x, ref-x) for k, v in values.items()},
        B_vs_A=compare(values['B'], values['A']), per_call_comparisons=comparisons,
        subsequent_indices=indices, subsequent_three_mean_seconds=means, complete_block_call_timings=timings,
        component_complete_block_mean_reduction_percent=(am-bm)/am*100,
        initialization_seconds=data['initialization_seconds'], destroy_seconds=data['destroy_seconds'],
        summed_eight_call_lifecycle_seconds=lifecycle,
        final_ledger=dict(weighted_contexts_created_destroyed=3, weighted_runs=12, weighted_output_buffers_acquired_released=12,
            native_MatMul_contexts_created_destroyed=2, native_memories_created_destroyed=6, native_products=128,
            GPU_softmax_calls_events_created_released=64, GPU_owned_objects_created_released=7, live=0),
        controller=dict(exit_code=0, observed_NPU_hz=1000000000, restored_NPU_hz=300000000,
            exact_policy_restored=True, max_sampled_temperature_c=maximum),
        limits=['One original native518 DINO0 block and one input; no full-model/trajectory/realtime acceptance',
            'Actual prefix QKV feeds attention and original input feeds suffix residual; output evidence plus frozen source review, not new execution',
            'Only first B boundary arrays saved; subsequent prefix/attention identity is recorded raw hashes',
            'Residual updates use FP32 y-x as in executed driver; all error reductions/norms computed in FP64',
            'Whole block and intermediate differences are cumulative; no local operator attribution without same-input pairing',
            'Persistent-context call timing excludes initialization/teardown; their measured costs are reported separately',
            'GPU event kernel times are contained in bridge/attention wall time and must not be added twice',
            'GPU simple_ondemand and sampled endpoints do not prove fixed kernel frequency or physical DDR traffic'], files=files)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], checked_NPYs=checked, A_mean=am, B_mean=bm,
        B_whole_relative_l2=result['whole_output_vs_original_CPU']['B']['relative_l2'],
        B_update_relative_l2=result['residual_update_vs_original_CPU']['B']['relative_l2'])))


if __name__ == '__main__':
    main()
