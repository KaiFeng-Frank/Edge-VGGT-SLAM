#!/usr/bin/env python3
"""Independent saved-only audit of two CPU controls and native attention ABBA.

JSON, NumPy arrays and source AST only; no Torch, model, SDK, ORT or board calls.
Relative errors always identify their reference; FP16 output storage does not
prove the internal accumulation precision of either CPU or native MatMul.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

R = Path('/dev/shm/edge-vggt-v118-review')
D117 = Path('/path/to/research-artifacts/v117-native518-attention-dump-r1')
CPU_SHA = 'e9a64c20658e3c5a25ba9026d06d3a4d6904ab938b154e6008dcfd026435a0d1'
NATIVE_SHA = '6302f2f2efa236b317ce5985923a21897945906b1d68cd21006c9422618c92d2'
COPY_SHA = '422903841eda6a4dcd83dda8b9a13aad2406d49bd1504ff8c975d7b557d45425'
CF_SHA = '03e9a5797ca21c542e467a68057f7290f0a881a689b273db74f7f7d5d3dea94d'
DUMP_SHA = '565054e051afd2565bcd07d6487e02cedeca62456e09e1214021f30c9c4f46d4'
TAP_SHA = '1f89eac69b85ec84ea76f9e8ca0a50ca7d4c5f19620c4159f41eaf6f3661225a'
QKV_RAW = '1f09065df446dbbc56ca249b96156f7ec2c78b87555461c3ed33852e40f91936'
NATIVE_RAW = '00545f97739c7b0a0748918cf766fcd808a9060e2edf6772ec0e88ed502e4fff'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def raw(v):
    return hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()


def compare(actual, ref):
    a, b = actual.astype(np.float64), ref.astype(np.float64); d = a-b
    an, rn = float(np.linalg.norm(a)), float(np.linalg.norm(b)); bad = np.abs(d) > 1e-5+1e-4*np.abs(b)
    return dict(relative_l2=float(np.linalg.norm(d))/rn if rn else None, max_abs=float(np.abs(d).max()),
        cosine=float(np.sum(a*b)/(an*rn)) if an and rn else None, strict_rtol=1e-4, strict_atol=1e-5,
        strict_pass=not bool(bad.any()), outside_count=int(bad.sum()), byte_equal=actual.dtype == ref.dtype and actual.tobytes() == ref.tobytes())


def agree(actual, reported):
    for key, value in reported.items():
        if key not in actual: continue
        if isinstance(value, float): require(math.isclose(actual[key], value, rel_tol=1e-10, abs_tol=1e-12), 'Reported metric differs: ' + key)
        else: require(actual[key] == value, 'Reported metric differs: ' + key)


def flat(heads):
    require(heads.shape == (1, 16, 1041, 64), 'Complete original BHND required')
    return np.ascontiguousarray(heads.transpose(0, 2, 1, 3).reshape(1, 1041, 1024))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=R); p.add_argument('--archive117', type=Path, default=D117)
    p.add_argument('--output', type=Path, default=R / 'saved-sdpa-audit.json')
    a = p.parse_args(); root, prior = a.root, a.archive117; files = []; array_count = 0
    def receipt(path, digest=None):
        actual = sha(path); require(digest is None or actual == digest, 'Receipt identity differs: ' + str(path))
        files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=actual)); return json.loads(path.read_text())
    def load(path, row, counted=False):
        nonlocal array_count
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Array file identity differs: ' + str(path))
        v = np.load(path, allow_pickle=False)
        require(list(v.shape) == row['shape'] and str(v.dtype) == row['dtype'] and np.isfinite(v).all() and raw(v) == row['raw_sha256'], 'Array raw/shape/dtype/finite differs')
        if 'tensor_bytes' in row: require(v.nbytes == row['tensor_bytes'], 'Tensor byte count differs')
        files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=row['sha256'], raw_sha256=raw(v), shape=list(v.shape), dtype=str(v.dtype), finite=True))
        array_count += int(counted); return v
    def source(name, digest):
        path = root / name; require(sha(path) == digest, 'Executed source differs: ' + name)
        ast.parse(path.read_text()); files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=digest)); return ast.parse(path.read_text())
    source('cpu_sdpa_same_qkv_board.py', CPU_SHA); parent_tree = source('run_native_attention_board.py', NATIVE_SHA); copy_tree = source('run_native_attention_copy_abba.py', COPY_SHA)
    keys = {'ABI_SHA','SDK_SHA','NEON_SHA','ATTENTION_SHA'}
    def constants(tree):
        return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in keys}
    dependencies = constants(parent_tree); require(dependencies == constants(copy_tree), 'Runtime/ABI/conversion source pins changed between native routes')
    static_copies = {'ABI_SHA':Path('/dev/shm/edge-vggt-v50/public-snapshot/vggt_slam/rknn_linear.py'),
                     'SDK_SHA':Path('/path/to/research-artifacts/v57/librknnrt.so'),
                     'ATTENTION_SHA':Path('/dev/shm/edge-vggt-v111-native-trajectory/original-vggt-source/vggt/layers/attention.py')}
    for key,path in static_copies.items():
        require(sha(path)==dependencies[key], 'Available source/SDK copy identity differs: '+key)
    parent_math = next(n for n in ast.walk(parent_tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'head')
    copy_math = next(n for n in ast.walk(copy_tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'head')
    require(ast.dump(parent_math) == ast.dump(copy_math), 'ABBA per-head attention mathematics changed')
    cpu = receipt(root / 'same-qkv-r1/result.json'); native = receipt(root / 'native-attention-r1/result.json'); copy = receipt(root / 'copy-abba-r1/result.json')
    cf = receipt(prior / 'qkv-counterfactual-r100/result.json', CF_SHA); d = receipt(prior / 'dump-r1/result.json', DUMP_SHA)
    tap = receipt(prior / 'old-reference-r100/npu-result.json', TAP_SHA)
    require(cf['baseline_all_points_byte_equal_original_CPU'] and cf['QKV_substitution_calls'] == 1 and cf['cleanup_errors'] == [] and d['final_baseline_byte_equal'], 'Original same-QKV references required')
    ref = load(prior / 'qkv-counterfactual-r100/attn_proj_input.npy', cf['artifacts']['attn.proj.input'])
    drow = next(r for r in d['dump_files'] if r['file'].startswith('008_'))
    z = load(prior / 'dump-r1/dumps' / drow['file'], drow); fused = np.ascontiguousarray(z[:, :, 0, :].transpose(0, 2, 1))
    trow = tap['calls'][0]['artifact']; npz = prior / 'old-reference-r100/npu-call0.npz'
    require(sha(npz) == trow['sha256'] and npz.stat().st_size == trow['bytes'], 'Actual QKV NPZ identity differs')
    with np.load(npz, allow_pickle=False) as archive: qkv = np.array(archive['attn.qkv'], copy=True, order='C')
    require(raw(qkv) == QKV_RAW and qkv.shape == (1, 1041, 3072) and qkv.dtype == np.float32 and np.isfinite(qkv).all() and
            qkv.astype(np.float16).astype(np.float32).tobytes() == qkv.tobytes(), 'Actual QKV Half roundtrip must preserve all bytes')
    files.append(dict(file=str(npz), bytes=npz.stat().st_size, sha256=trow['sha256']))
    for data, digest in ((cpu, CPU_SHA), (native, NATIVE_SHA), (copy, COPY_SHA)):
        require(data['script_sha256'] == digest and data['status'].startswith('completed') and data['cleanup_errors'] == [] and
                all(data[k] is True for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored')), 'Successful clean actual receipt required')
    require(cpu['actual_QKV_raw_sha256'] == native['QKV_raw_sha256'] == copy['QKV_raw_sha256'] == QKV_RAW and
            cpu['actual_QKV_Half_roundtrip_byte_equal'] and native['QKV_Half_roundtrip_byte_equal'] and copy['QKV_Half_roundtrip_byte_equal'], 'All routes must use the identical actual QKV')
    ref_heads = np.ascontiguousarray(ref.reshape(1, 1041, 16, 64).transpose(0, 2, 1, 3))
    fused_heads = np.ascontiguousarray(fused.reshape(1, 1041, 16, 64).transpose(0, 2, 1, 3)); cpu_results = {}
    require([r['name'] for r in cpu['calls']] == ['original_CPU_FP32', 'original_CPU_float16'], 'Exactly two distinct CPU controls required')
    for row in cpu['calls']:
        require(row['attempts'] == 1 and row['completed'] and row['input_unchanged'] and row['QKV_strides'] == [[3197952,64,3072,1]]*3, 'Original CPU call/strides required')
        v = load(root / 'same-qkv-r1' / row['output']['file'], row['output'], True); cpu_results[row['name']] = v
        require(raw(flat(v)) == row['output']['consumer_raw_sha256'], 'CPU consumer reshape identity differs')
        agree(compare(v, ref_heads), row['versus_same_QKV_original_CPU_FP32']); agree(compare(v, fused_heads), row['versus_actual_NPU_SDPA'])
    fp32, half = cpu_results.values(); require(flat(fp32).tobytes() == ref.tobytes(), 'CPU FP32 must byte-match old same-QKV counterfactual')
    require(fp32.dtype == np.float32 and half.dtype == np.float16 and cpu['original_FP32_reference_byte_gate_passed'], 'Genuine two CPU dtypes required')
    baseline_value = None; run_summaries = {}; call_rows = []
    for label, data, expected_calls in (('native-r1', native, 4), ('copy-abba', copy, 8)):
        require(len(data['calls']) == expected_calls and [r['index'] for r in data['calls']] == list(range(expected_calls)), 'Complete ordered calls required')
        if expected_calls == 8:
            require(''.join(r['route'] for r in data['calls']) == 'ABBAABBA' and data['parent_source_sha256'] == NATIVE_SHA, 'Controlled eight-call ABBA lineage required')
        products = expected_calls*16*2
        expected = dict(rknn_matmul_create=2, rknn_matmul_set_core_mask=2, rknn_create_mem=6, rknn_matmul_set_io_mem=6+products*2,
                        rknn_mem_sync=products*3, rknn_matmul_run=products, rknn_destroy_mem=6, rknn_matmul_destroy=2)
        require(data['SDK_attempts'] == data['SDK_successes'] == expected and data['resource_counts_balanced'] and len(data['resources']) == 2 and
                all(r['live_context'] == r['live_memories'] == 0 and r['cleanup_errors'] == [] for r in data['resources']), 'Persistent2contexts/6memories native ledger differs')
        phases = []
        for row in data['calls']:
            directory = 'native-attention-r1' if expected_calls == 4 else 'copy-abba-r1'
            v = load(root / directory / row['output']['file'], row['output'], True)
            require(v.shape == (1,1041,1024) and v.dtype == np.float32 and raw(v) == NATIVE_RAW, 'Complete frozen native output required')
            if baseline_value is None: baseline_value = v
            require(v.tobytes() == baseline_value.tobytes(), 'All12 native outputs must byte-match')
            agree(compare(v, ref), row['vs_original_CPU_same_QKV']); agree(compare(v, fused), row['vs_unchanged_fused_same_QKV'])
            require([h['head'] for h in row['heads']] == list(range(16)), 'All16 timed heads required')
            phase = dict(complete_attention_seconds=row['complete_attention_seconds'])
            for product in ('QK','PV'):
                for field in ('pack_upload_bind_seconds','SDK_run_seconds','sync_owned_output_finite_seconds','complete_product_seconds'):
                    phase[product+'_'+field] = sum(h[product][field] for h in row['heads'])
                for h in row['heads']:
                    t = h[product]; require(all(math.isfinite(v) and v >= 0 for v in t.values()) and math.isclose(t['complete_product_seconds'],sum(t[k] for k in ('pack_upload_bind_seconds','SDK_run_seconds','sync_owned_output_finite_seconds')),rel_tol=1e-10,abs_tol=1e-12), 'Product timing decomposition differs')
            phase['softmax_crop_scale_probability_store_seconds'] = sum(h['softmax_crop_scale_probability_store_seconds'] for h in row['heads'])
            phase['outer_other_seconds'] = phase['complete_attention_seconds']-phase['QK_complete_product_seconds']-phase['PV_complete_product_seconds']-phase['softmax_crop_scale_probability_store_seconds']
            require(phase['outer_other_seconds'] >= 0, 'Substages exceed complete attention timer')
            phases.append(phase); call_rows.append(dict(run=label,index=row['index'],route=row.get('route'),output_raw_sha256=raw(v),timings_seconds=phase))
        route_indices = {'native': list(range(1,4))} if expected_calls == 4 else {r:[i for i,row in enumerate(data['calls']) if row['route']==r and i != (0 if r=='A' else 1)] for r in 'AB'}
        means = {r:{k:statistics.mean(phases[i][k] for i in ids) for k in phases[0]} for r,ids in route_indices.items()}
        for route, ids in route_indices.items():
            require(len(ids)==3, 'Exclude exactly first of each route, leaving three samples')
            reported = data['steady_three_mean_seconds'] if expected_calls == 4 else data['subsequent_three_means_seconds'][route]
            require(math.isclose(means[route]['complete_attention_seconds'],reported,rel_tol=1e-12), 'Reported complete mean differs')
        lifecycle = data['init_seconds']+data['destroy_seconds']+sum(r['complete_attention_seconds'] for r in data['calls'])
        require(math.isclose(lifecycle,data['complete_four_call_lifecycle_seconds' if expected_calls==4 else 'complete_eight_call_lifecycle_seconds'],rel_tol=1e-12), 'Lifecycle accounting differs')
        run_summaries[label] = dict(calls=expected_calls,SDK_products=products,contexts=2,memories=6,all_released=True,first_complete_seconds=data['calls'][0]['complete_attention_seconds'],
                                   init_seconds=data['init_seconds'],destroy_seconds=data['destroy_seconds'],summed_lifecycle_seconds=lifecycle,subsequent_indices=route_indices,subsequent_mean_seconds=means)
    controls = {}
    for label, name in (('native-r1','native-attention-control-r1'),('copy-abba','copy-abba-control-r1')):
        control = receipt(root / name / 'control.json')
        require(control['child_returncode']==0 and control['restore_errors']==[] and control['baseline']==control['restored'] and control['frequency_governor_limits_restored'] and
                control['restored']['cur_freq']=='300000000' and all(r['actual_frequency_hz']==1000000000 for r in control['observations']), 'Native controller policy/frequency restore differs')
        maximum = max(v for r in control['observations'] for v in r['temperatures_c'].values()); require(maximum<75,'Observed thermal stop threshold exceeded')
        controls[label] = dict(exit_code=0,observed_all_NPU_samples_hz=1000000000,sampled_max_temperature_c=maximum,restored_NPU_hz=300000000,restored_exact_policy=True)
    cpu_log = (root / 'cpu-control-terminal.log').read_text().splitlines(); require(cpu_log[0] == cpu_log[-1] == '300000000', 'CPU launcher before/after NPU reads differ')
    copy_means = run_summaries['copy-abba']['subsequent_mean_seconds']; am,bm = [copy_means[r]['complete_attention_seconds'] for r in 'AB']
    require(array_count==14,'Exactly14 new arrays must be checked')
    result = dict(schema='v118_independent_saved_same_QKV_CPU_native_attention_audit',status='passed',auditor_sha256=sha(__file__),execution='Saved NumPy/JSON and source AST only; no NN, Torch, ORT, SDK, GPU or board calls',
        checked_new_arrays=14,CPU_arrays=2,native_arrays=12,reported_comparison_records_independently_recomputed=28,
        source_dependency_pins=dependencies,static_copies_rehashed=sorted(static_copies),NEON_library_payload_not_available_to_this_auditor=True,
        all_native_output_bytes_identical=True,native_output_raw_sha256=NATIVE_RAW,
        CPU_FP32_byte_equal_old_counterfactual=True,actual_QKV_Half_roundtrip_byte_equal=True,copy_ABBA_per_head_math_AST_equal=True,
        CPU_Half_vs_same_QKV_CPU_FP32=compare(half,fp32),fused_NPU_vs_same_QKV_CPU_FP32=compare(fused_heads,fp32),fused_NPU_vs_same_QKV_CPU_Half=compare(fused_heads,half),
        native_vs_same_QKV_CPU_FP32=compare(baseline_value,ref),native_vs_same_QKV_CPU_Half=compare(baseline_value,flat(half)),native_vs_unchanged_fused_NPU=compare(baseline_value,fused),
        CPU_single_observed_calls_seconds={r['name']:r['observed_wall_seconds'] for r in cpu['calls']},native_runs=run_summaries,
        copy_ABBA_mean_reduction_percent=(am-bm)/am*100,call_timings=call_rows,controls=controls,
        CPU_control=dict(NPU_before_after_read_hz=300000000,thermal_trace_collected=False,SDK_contexts=0),
        limits=['Single original DINO0 case; no complete-model/trajectory acceptance','CPU and native Half inputs/output representations do not prove identical internal accumulation precision',
                'Timing phases include sync/copy/finite or crop/scale/softmax/Half storage; not isolated memcpy or exp timings','Initialization/destruction are separate from each persistent-context attention call',
                'Tiny six-sample ABBA difference is not established practical improvement; no device parallelism, bandwidth or full-pipeline speed claim'],files=files)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],arrays=array_count,Half_L2=result['CPU_Half_vs_same_QKV_CPU_FP32']['relative_l2'],native_L2=result['native_vs_same_QKV_CPU_FP32']['relative_l2'],ABBA_reduction_percent=result['copy_ABBA_mean_reduction_percent'])))


if __name__=='__main__':
    main()
