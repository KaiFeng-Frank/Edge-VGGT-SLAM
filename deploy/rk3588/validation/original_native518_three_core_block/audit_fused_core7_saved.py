#!/usr/bin/env python3
"""NumPy-only audit of the original native518 fused DINO0 core7 preflight."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path('/dev/shm/edge-vggt-v124-review')
REFERENCE = Path('/path/to/research-artifacts/v117-native518-attention-dump-r1/old-reference-r100')
RESULT_SHA = 'b01ffed1323d271db006c959adfd4b9c77f5447119b6f211af1c66548a70917b'
CONTROL_SHA = 'ddb615830a18a8d7a3ba973accdad0e7581fcc6707139c47fe5ab9a39a5626b2'
REFERENCE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
MODEL_SHA = 'fe8acac49eb10b1215edf1d4c348b69da3ddfed53f16c2e202b0e3f6428b32c0'
BASELINE_RAW = 'e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--output', type=Path, default=ROOT/'fused-core7-saved-audit.json')
    args = parser.parse_args(); files = []

    def record(path, digest=None):
        actual = sha(path)
        require(digest is None or actual == digest, 'Pinned file differs: '+str(path))
        files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=actual))
        return path

    def receipt(path, digest):
        return json.loads(record(path, digest).read_text())

    def load(base, row):
        require(Path(row['file']).name == row['file'], 'Relative NPY basename required')
        path = record(base/row['file'], row['sha256']); value = np.load(path, allow_pickle=False)
        require(path.stat().st_size == row['bytes'] and list(value.shape) == row['shape'] == [1,1041,1024]
                and str(value.dtype) == row['dtype'] == 'float32' and np.isfinite(value).all()
                and raw(value) == row['raw_sha256'], 'NPY shape/dtype/file/raw/finite differs')
        files[-1].update(raw_sha256=raw(value), shape=list(value.shape), dtype='float32', finite=True)
        return value

    result = receipt(args.root/'fused-core7-r1/result.json', RESULT_SHA)
    control = receipt(args.root/'fused-core7-control-r1/control.json', CONTROL_SHA)
    reference = receipt(args.reference/'capture.json', REFERENCE_SHA)
    original = load(args.reference, reference['artifacts']['output'])
    launcher = record(args.root/'run_fused_core7_preflight.sh').read_text()
    require('--model-sha256 '+MODEL_SHA in launcher and '--core-mask 7' in launcher
            and '--input-sha256 aae907d5105f68d4b6d74bb733871e0971fb2c639ec8be8cb46a82176a706aa8' in launcher,
            'Exact same original compiled model/input launch required')
    require(result['schema'] == 'original_native518_single_block_normal_IO_diagnostic'
            and result['status'] == 'completed_component_not_quality_accepted' and result['role'] == 'dino'
            and result['requested_core_mask'] == 7 and result['init_flags'] == 0
            and result['set_core_mask_code'] == result['init_code'] == result['destroy_code'] == 0
            and len(result['calls']) == 4 and result['hidden_warmup_calls'] == 0,
            'Completed original fused core7 preflight required')
    comparisons = []; first = None
    for index, call in enumerate(result['calls']):
        require(call['index'] == index and call['status'] == 'completed' and call['outputs_release_code'] == 0
                and call['phase'] == ('first_recorded_warmup' if index == 0 else 'steady')
                and call['output']['owned_copy_before_release'], 'Actual call phase/ownership differs')
        value = load(args.root/'fused-core7-r1', call['output'])
        if first is None:
            first = value
        require(raw(value) == BASELINE_RAW and value.tobytes() == first.tobytes()
                and call['byte_equal_to_first'], 'Core7 output differs from canonical core1 or repeat')
        x, y = value.astype(np.float64), original.astype(np.float64)
        error = x-y; maximum = float(np.abs(error).max())
        relative = float(np.linalg.norm(error.ravel())/np.linalg.norm(y.ravel()))
        outside = int(np.count_nonzero(np.abs(error) > 1e-5+1e-4*np.abs(y)))
        reported = call['reference_comparison']
        require(reported['strict_rtol'] == 1e-4 and reported['strict_atol'] == 1e-5
                and reported['strict_pass'] == (outside == 0) and reported['max_abs'] == maximum
                and math.isclose(relative, reported['relative_L2'], rel_tol=1e-12), 'Reported FP32 precision differs')
        require(all(call[k]['npu_hz'] == 1000000000 for k in ('platform_before','platform_after'))
                and call['complete_boundary_seconds'] >= call['timings']['SDK_run_seconds'] > 0,
                'Normal complete boundary/platform differs')
        comparisons.append(dict(index=index, phase=call['phase'], complete_seconds=call['complete_boundary_seconds'],
            SDK_run_seconds=call['timings']['SDK_run_seconds'], raw_sha256=raw(value),
            byte_equal_canonical_core1=True, relative_L2=relative, max_abs=maximum,
            strict_outside_elements=outside, strict_pass=outside == 0))
    require({p.name for p in (args.root/'fused-core7-r1').glob('*.npy')}
            == {c['output']['file'] for c in result['calls']}, 'Exactly four saved arrays required')
    counts = result['resources']
    require(counts['contexts_created'] == counts['contexts_destroyed'] == 1
            and counts['runs_completed'] == counts['output_buffers_acquired'] == counts['output_buffers_released'] == 4
            and result['resources_balanced'] and result['live_contexts'] == result['live_output_buffers'] == 0
            and result['cleanup_errors'] == [] and all(result[k] for k in ('pins_unchanged','affinity_restored','environment_restored')),
            'SDK resources/pins/environment not restored')
    require(control['child_returncode'] == 0 and control['restore_errors'] == []
            and control['baseline'] == control['restored'] and control['frequency_governor_limits_restored']
            and control['restored']['cur_freq'] == '300000000'
            and all(c['actual_frequency_hz'] == 1000000000 for c in control['observations']), 'Control restoration differs')
    steady = [v['complete_seconds'] for v in comparisons[1:]]
    mean = statistics.mean(steady)
    require(math.isclose(mean, result['summary']['steady_mean_seconds'], rel_tol=1e-12)
            and result['summary']['steady_count'] == 3 and result['summary']['all_repeat_byte_equal'], 'Steady summary differs')
    audit = dict(schema='v124_independent_fused_DINO0_core7_saved_audit', status='verified_original_fused_core7_preflight',
        auditor_sha256=sha(__file__), execution='Saved JSON/NumPy/text only; no neural/preprocessing/SDK/board calls',
        model_sha256_from_pinned_launch=MODEL_SHA, requested_core_mask=7, set_core_mask_code=0,
        all4_outputs_byte_equal_canonical_core1=True, canonical_raw_sha256=BASELINE_RAW,
        calls=comparisons, first_seconds=comparisons[0]['complete_seconds'], steady_count=3,
        steady_mean_seconds=mean, steady_range_seconds=[min(steady),max(steady)],
        steady_SDK_run_mean_seconds=statistics.mean(c['SDK_run_seconds'] for c in comparisons[1:]),
        SDK_ledger=dict(contexts_created=1,contexts_destroyed=1,runs_completed=4,
                        output_buffers_acquired=4,output_buffers_released=4), SDK_cleanup_verified=True,
        external_control=dict(child_returncode=0, exact_policy_restored=True, restored_NPU_hz=300000000,
            max_sampled_temperature_c=max(t for c in control['observations'] for t in c['temperatures_c'].values())),
        timing_scope=result['timing_scope'],
        limits=['Preflight uses a resident context; init/destroy and save/hash/reference checks are outside each call timer',
                'Four identical-input calls do not establish continuous throughput or P95',
                'Canonical core1 byte parity does not establish original FP32, full-model trajectory or realtime quality',
                'Historical core1 timings are not matched ABBA measurements; later candidate comparison must use core7'], files=files)
    args.output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'], outputs=4, all_byte_equal=True,
        seconds=[c['complete_seconds'] for c in comparisons], steady_mean_seconds=mean,
        reference_relative_L2=comparisons[0]['relative_L2'])))


if __name__ == '__main__':
    main()
