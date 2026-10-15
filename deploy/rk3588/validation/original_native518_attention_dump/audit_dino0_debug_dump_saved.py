#!/usr/bin/env python3
"""Saved NumPy-only boundary checks of the unchanged DINO0 debug run.

No Torch, model, ORT, SDK or board execution. Layout permutations are accepted
only after exact byte identities against the pinned logical QKV and three splits.
The same-QKV CPU SDPA counterfactual localizes an execution boundary, not a
specific internal QK, softmax or PV implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

R = Path('/dev/shm/edge-vggt-v117-review')
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
TAP_SHA = '1f89eac69b85ec84ea76f9e8ca0a50ca7d4c5f19620c4159f41eaf6f3661225a'
CF_SHA = '03e9a5797ca21c542e467a68057f7290f0a881a689b273db74f7f7d5d3dea94d'
SOURCE_SHA = '1084270777ce931147f29fc39510ef6ff19563fa24ae367e4c689434ff351fab'
MODEL_SHA = 'fe8acac49eb10b1215edf1d4c348b69da3ddfed53f16c2e202b0e3f6428b32c0'
FINAL_RAW = 'e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554'


def require(ok, text):
    if not ok: raise ValueError(text)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def raw(v):
    return hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()


def comparison(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64); d = a-b
    den = float(np.linalg.norm(b)); an = float(np.linalg.norm(a)); bad = np.abs(d) > 1e-5+1e-4*np.abs(b)
    return dict(relative_l2=float(np.linalg.norm(d))/den if den else None, max_abs=float(np.abs(d).max()),
                cosine=float(np.sum(a*b)/(an*den)) if an and den else None, strict_rtol=1e-4, strict_atol=1e-5,
                strict_pass=not bool(bad.any()), outside_count=int(bad.sum()), byte_equal=actual.dtype == reference.dtype and actual.tobytes() == reference.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=R / 'dump-r1')
    p.add_argument('--old-reference', type=Path, default=Path('/dev/shm/edge-vggt-v116-review/reference-r100'))
    p.add_argument('--additional-reference', type=Path, default=R / 'reference-r100')
    p.add_argument('--counterfactual-dir', type=Path, default=R / 'qkv-counterfactual-r100')
    p.add_argument('--control', type=Path, default=R / 'dump-control-r1/control.json')
    p.add_argument('--output', type=Path, default=R / 'debug-dump-saved-audit.json')
    a = p.parse_args(); files = []
    def receipt(path, digest=None):
        require(digest is None or sha(path) == digest, 'Receipt pin differs: ' + str(path)); files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=sha(path)))
        return json.loads(path.read_text())
    def load(path, row):
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'NPY file identity differs: ' + str(path))
        v = np.load(path, allow_pickle=False)
        require(list(v.shape) == row['shape'] and str(v.dtype) == row['dtype'] and np.isfinite(v).all() and raw(v) == row['raw_sha256'], 'NPY raw metadata differs: ' + str(path))
        files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=row['sha256'], raw_sha256=raw(v), shape=list(v.shape), dtype=str(v.dtype), finite=True))
        return v
    report = receipt(a.run_dir / 'result.json'); cpu = receipt(a.old_reference / 'capture.json', CAPTURE_SHA)
    extra = receipt(a.additional_reference / 'capture.json', CAPTURE_SHA); tap = receipt(a.old_reference / 'npu-result.json', TAP_SHA)
    cf = receipt(a.counterfactual_dir / 'result.json', CF_SHA); control = receipt(a.control)
    require(report['script_sha256'] == SOURCE_SHA and report['model_sha256'] == MODEL_SHA and report['status'] == 'completed_debug_dump_final_byte_equal_not_root_cause_proven' and
            report['init_flags'] == 0 and report['core_mask'] == 1 and report['forward_calls'] == 1 and report['graph_recompiled_or_outputs_appended'] is False,
            'Exact ordinary untapped run required')
    require(sha(R / 'run_dino0_debug_dump.py') == SOURCE_SHA, 'Executed source identity differs')
    require(tap['final_baseline_byte_equal_all_calls'] is True and tap['capture_sha256'] == CAPTURE_SHA and
            cf['capture_sha256'] == CAPTURE_SHA and cf['tap_result_sha256'] == TAP_SHA and cf['QKV_substitution_calls'] == 1 and
            cf['baseline_all_points_byte_equal_original_CPU'] is True and cf['model_parameter_state_unchanged'] is True and cf['hooks_removed'] is True and cf['cleanup_errors'] == [],
            'Same-QKV untouched downstream CPU reference lineage required')
    refs = {n: load(a.old_reference / cpu['artifacts'][n]['file'], cpu['artifacts'][n]) for n in ('input', 'norm1', 'attn.qkv', 'attn.proj.input', 'output')}
    for name in ('norm2.input', 'norm2', 'mlp.act'):
        refs[name] = load(a.additional_reference / extra['artifacts'][name]['file'], extra['artifacts'][name])
    npu_row = tap['calls'][0]['artifact']; npu_path = a.old_reference / 'npu-call0.npz'
    require(sha(npu_path) == npu_row['sha256'] and npu_path.stat().st_size == npu_row['bytes'], 'Prior NPU NPZ file differs')
    with np.load(npu_path, allow_pickle=False) as z:
        npu = {n: np.array(z[n], copy=True, order='C') for n in z.files}
    require(all(raw(v) == npu_row['arrays'][n]['raw_sha256'] and np.isfinite(v).all() and v.dtype == np.float32 for n, v in npu.items()), 'Prior NPU raw identities differ')
    files.append(dict(file=str(npu_path), bytes=npu_path.stat().st_size, sha256=npu_row['sha256']))
    rows = report['dump_files']; require(len(rows) == 18 and set(r['file'] for r in rows) == {f.name for f in (a.run_dir / 'dumps').glob('*.npy')}, 'All18 dump files required')
    dumps = {r['file']: load(a.run_dir / 'dumps' / r['file'], r) for r in rows}
    def stage(prefix):
        selected = [v for n, v in dumps.items() if n.startswith(prefix)]; require(len(selected) == 1, 'Unique compiled stage required: ' + prefix)
        return selected[0]
    def feature(v):
        require(v.ndim == 4 and v.shape[0] == v.shape[2] == 1 and v.shape[3] == 1041, 'Compiled feature[B,C,1,N] layout required')
        return np.ascontiguousarray(v[:, :, 0, :].transpose(0, 2, 1))
    def head(v):
        require(v.shape == (16, 64, 1, 1041), 'Compiled head[H,D,1,N] layout required')
        return np.ascontiguousarray(v[:, :, 0, :].transpose(0, 2, 1)[None])
    expected_input = refs['input'].astype(np.float16).astype(np.float32)
    require(np.array_equal(stage('000_'), expected_input) and np.array_equal(feature(stage('002_')), expected_input), 'Input normalization/layout must be exact original Half roundtrip')
    norm1 = feature(stage('003_')); require(np.array_equal(norm1, npu['norm1']), 'Compiled norm1 must byte-match previously final-compatible tap')
    z = stage('004_')[:, :, 0, :]
    qkv_standard = np.ascontiguousarray(z.transpose(0, 2, 1))
    qkv = np.ascontiguousarray(z.reshape(1, 16, 3, 64, 1041).transpose(0, 4, 2, 1, 3).reshape(1, 1041, 3072))
    require(np.array_equal(qkv, npu['attn.qkv']) and not np.array_equal(qkv_standard, npu['attn.qkv']) and
            np.array_equal(stage('004_').reshape(16, 192, 1, 1041), stage('005_')), 'Exact head-major QKV swizzle and following reshape required')
    q, k, v = [head(stage('006_Split_' + name)) for name in ('query', 'key', 'value')]
    rejoined = np.ascontiguousarray(np.stack((q, k, v), axis=2).transpose(0, 3, 2, 1, 4).reshape(1, 1041, 3072))
    require(np.array_equal(rejoined, qkv) and raw(qkv) == cf['actual_NPU_tap_artifact']['arrays']['attn.qkv']['raw_sha256'], 'All3 split inputs must exactly match the CPU counterfactual QKV')
    sdpa = np.ascontiguousarray(head(stage('007_')).transpose(0, 2, 1, 3).reshape(1, 1041, 1024))
    require(np.array_equal(sdpa, feature(stage('008_'))), 'SDPA→consumer flatten must preserve bytes')
    cf_attention = load(a.counterfactual_dir / cf['artifacts']['attn.proj.input']['file'], cf['artifacts']['attn.proj.input'])
    y = load(a.run_dir / report['output']['file'], report['output'])
    require(raw(y) == FINAL_RAW and np.array_equal(y, npu['output']) and np.array_equal(feature(stage('012_')), y) and
            np.array_equal(stage('013_').reshape(1, 1041, 1024), y) and np.array_equal(stage('014_'), y) and np.array_equal(stage('015_'), y), 'All final aliases and ordinary final must match untouched baseline')
    s = report['resources']; require(all(s[key] == 1 for key in ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')) and
            s['live_contexts'] == s['live_output_buffers'] == 0 and s['closed'] is True and s['cleanup_errors'] == report['cleanup_errors'] == [], 'Saved onecontext/onecall/onebuffer ledger required')
    require(control['child_returncode'] == 0 and control['restore_errors'] == [] and control['frequency_governor_limits_restored'] is True and
            control['baseline'] == control['restored'] and control['restored']['cur_freq'] == '300000000', 'Saved controller must restore original300MHz')
    points = dict(norm1=norm1, QKV=qkv, SDPA_consumer=sdpa, residual1_norm2_input=feature(stage('009_')), norm2=feature(stage('010_')), MLP_GELU=feature(stage('011_')), output=y)
    ref_names = ('norm1', 'attn.qkv', 'attn.proj.input', 'norm2.input', 'norm2', 'mlp.act', 'output')
    result = dict(schema='v117_saved_unchanged_DINO0_debug_boundary_audit', status='passed', auditor_sha256=sha(__file__), execution='Saved JSON/NumPy only; no NN/SDK/ORT/GPU/board calls',
        source_sha256=SOURCE_SHA, model_sha256=MODEL_SHA, checked_dump_count=18, dump_file_bytes=sum(r['bytes'] for r in rows),
        final_baseline_byte_equal=True, final_raw_sha256=FINAL_RAW, input_exact_original_Half_roundtrip=True,
        norm1_exact_previous_tap_raw_sha256=raw(norm1), QKV_exact_previous_tap_raw_sha256=raw(qkv),
        QKV_channel_order_verified='[head16,QKV3,dim64] -> [token1041,QKV3,head16,dim64]', three_split_rejoin_byte_equal=True,
        SDPA_consumer_layout_byte_equal=True, same_actual_QKV_CPU_reference_raw_sha256=raw(cf_attention),
        SDPA_vs_original_CPU_on_identical_actual_QKV=comparison(sdpa, cf_attention),
        SDPA_vs_Half_roundtrip_CPU_on_identical_actual_QKV=comparison(sdpa, cf_attention.astype(np.float16).astype(np.float32)),
        cumulative_points_vs_original_CPU={name: comparison(value, refs[ref_name]) for (name, value), ref_name in zip(points.items(), ref_names)},
        whole_output_update_vs_original_CPU=comparison(y.astype(np.float64)-refs['input'], refs['output'].astype(np.float64)-refs['input']),
        saved_resource_ledger_verified=dict(contexts=1, calls=1, output_buffers=1, live=0),
        controller=dict(child_returncode=0, restored_hz=300000000, sampled_max_temperature_c=max(v for row in control['observations'] for v in row['temperatures_c'].values())),
        dump_destination_verified=report['default_dump_unchanged'] and report['default_dump_before']['exists'] is False and report['default_dump_after']['exists'] is False,
        interpretation='For this unchanged final-compatible artifact/input, a difference is established between the verified QKV split inputs and fused exSDP output/storage versus original CPU SDPA on identical QKV. Dumps do not expose QK/softmax/PV internal values or independently identify a faulty suboperation. Norm2/MLP/full-output comparisons are cumulative, not local causal attribution; no trajectory or latency acceptance.',
        files=files)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], SDPA_same_QKV=result['SDPA_vs_original_CPU_on_identical_actual_QKV'], output=a.output.as_posix())))


if __name__ == '__main__':
    main()
