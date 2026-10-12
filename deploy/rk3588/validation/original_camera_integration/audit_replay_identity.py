#!/usr/bin/env python3
"""Small saved-receipt supplement to the unchanged replay auditor; no SDK."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINS = {
    'replay-r1/results.json': 'a7406aed51c1ed8bde8239ef33210dc620235fc72aedf1d0040d6bfa499c0893',
    'complete-candidate-r1/result.json': '4644c425a238f672dd92a0319348d2a10de756151e5484a9c2eb65686616e84a',
    'vggt-v71-replay-control-r1.json': '1d215bda5dedb5b13dc39ef424900dd33af06a335673c5b4c83b76d4ba3e8df7',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    for name, digest in PINS.items():
        require(sha(ROOT / name) == digest, 'Frozen receipt changed: ' + name)
    replay = json.loads((ROOT / 'replay-r1/results.json').read_text())
    complete = json.loads((ROOT / 'complete-candidate-r1/result.json').read_text())
    control = json.loads((ROOT / 'vggt-v71-replay-control-r1.json').read_text())
    camera = replay['camera_rknn']
    original_camera = next(iter(complete['branches'].values()))['camera_rknn']
    require(replay['rknn_camera_split4'] is True and complete['camera_split4'] is True
            and camera['model_sha256'] == '0c216630b608ddd72ae288385c886d20d5e002b85fad3f741ed1308360bd7f51',
            'Same split4 candidate required')
    identity_keys = ('camera_graph_variant', 'model_sha256', 'baseline_model_sha256', 'compile_receipt_sha256',
                     'check_receipt_sha256', 'verified_original_parameters', 'input_attrs', 'output_attr', 'core_mask',
                     'original_GELU_nodes_protobuf_unchanged', 'compiled_rounding_equivalence_validated', 'compiled_graph_changed')
    require(all(camera[k] == original_camera[k] for k in identity_keys)
            and camera['verified_original_parameters'] == 69 and camera['compiled_rounding_equivalence_validated'] is False,
            'Camera compile/source/I/O identity differs from complete candidate')
    require(control['status'] == 'completed' and control['returncode'] == 0 and control['cleanup_errors'] == []
            and control['frequency_written'] is False and control['samples']
            and all(s['npu_hz'] == 300000000 for s in control['samples']), 'Replay control failed or sampled frequency changed')
    temperatures = [v for sample in control['samples'] for v in sample['temperatures_c'].values()]
    require(all(math.isfinite(v) for v in temperatures)
            and all(math.isfinite(replay[k]) and replay[k] > 0 for k in ('startup_seconds', 'peak_rss_mib')),
            'Nonfinite thermal/startup/RSS record')
    report = dict(schema='v71_saved_replay_camera_identity_and_monitor_audit', status='passed_saved_checks',
        source_sha256=sha(Path(__file__)), input_pins=PINS, replay_saved_audit_sha256=sha(ROOT / 'replay_saved_audit.json'),
        camera_split4=True, camera_model_sha256=camera['model_sha256'], same_camera_complete_identity=True,
        original_camera_parameter_count=69, compiled_rounding_equivalence_validated=False,
        startup_seconds=replay['startup_seconds'], peak_RSS_MiB=replay['peak_rss_mib'],
        monitor_samples=len(control['samples']), all_sampled_NPU_hz=300000000, no_frequency_write_recorded=True,
        maximum_sampled_temperature_c=max(temperatures), stage_wall_seconds=control['seconds'],
        limits=['Saved receipts only; no model/SDK/board execution or independent observation of hardware between samples.',
                'RSS is historical process accounting, not independent physical memory or proof of 8GB operation.',
                'The replay selects different timestamps; fixed-input ATE/RPE does not establish this trajectory quality.'])
    output = ROOT / 'replay_identity_monitor_audit.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], output_sha256=sha(output))))
