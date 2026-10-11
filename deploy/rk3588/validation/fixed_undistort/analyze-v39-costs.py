"""Saved JSON timing analysis only; no model, image, SDK or BA execution.

Clocks within correlation and stream are nested. This script subtracts their
outer clocks rather than adding them again. Remaining time is observational,
not a hardware lower bound or a measured result of replacing two components.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def stats(values):
    assert values and all(math.isfinite(v) for v in values)
    return dict(count=len(values), mean_ms=fmean(values) * 1000,
                p50_ms=quantile(values, .5) * 1000,
                p95_ms=quantile(values, .95) * 1000,
                min_ms=min(values) * 1000, max_ms=max(values) * 1000)


def analyze(run):
    updates, corr, ba = run['update_calls'], run['correlation']['calls'], run['BA_calls']
    assert len(updates) == len(corr)
    for i, (update, call) in enumerate(zip(updates, corr)):
        assert update['index'] == call['index'] == i
        assert update['edges'] == call['edges']
    captured = {c['frame_index'] for c in run['update_captures']}
    populations = {
        'all_fresh_including_initialization': [f for f in run['frames']
                                              if f['result']['fresh_pose_valid']],
        'post_initialization_single_Update': [f for f in run['frames']
            if f['result']['fresh_pose_valid'] and f['result']['updates'] == 1],
        'post_initialization_without_full_Update_capture': [f for f in run['frames']
            if f['result']['fresh_pose_valid'] and f['result']['updates'] == 1
            and f['index'] not in captured],
    }
    result = {}
    for label, frames in populations.items():
        rows = []
        for frame in frames:
            index = frame['index']
            us = [u for u in updates if u['stage'] == 'stream' and u['frame_index'] == index]
            bs = [b for b in ba if b['stage'] == 'stream' and b['frame_index'] == index]
            assert len(bs) == frame['result']['updates']
            # Initialization has a motion probe followed by twelve graph Updates.
            assert len(us) == len(bs) + int(frame['result']['motion_probe'] is not None)
            cs = [corr[u['index']] for u in us]
            row = dict(
                preprocess=frame['decode_undistort_crop_resize_seconds'],
                encoder=frame['encode_seconds'], stream=frame['stream_seconds'],
                publication=frame['publication_seconds'],
                admission_to_publication=frame['admission_to_publication_seconds'],
                Update=sum(u['seconds'] for u in us),
                correlation=sum(c['total_seconds'] for c in cs),
                BA_two_iterations=sum(b['seconds'] for b in bs),
                Update_capture_and_metadata=sum(u['capture_and_metadata_seconds'] for u in us),
            )
            row['other_stream'] = row['stream'] - sum(row[k] for k in
                ('Update', 'correlation', 'BA_two_iterations', 'Update_capture_and_metadata'))
            row['remaining_excluding_encoder_Update_capture'] = sum(row[k] for k in
                ('preprocess', 'correlation', 'BA_two_iterations', 'other_stream', 'publication'))
            row['stream_without_Update_capture'] = sum(row[k] for k in
                ('correlation', 'BA_two_iterations', 'other_stream'))
            row['correlation_plus_BA'] = row['correlation'] + row['BA_two_iterations']
            assert row['other_stream'] >= 0
            assert abs(row['admission_to_publication'] - sum(row[k] for k in
                ('preprocess', 'encoder', 'stream', 'publication'))) < 1e-8
            for name, extract in (
                ('corr_validation_and_lock', lambda c: c['validation_and_lock_seconds']),
                ('corr_eviction', lambda c: c['eviction_seconds']),
                ('corr_prepare', lambda c: c['preparation_seconds']),
                ('corr_query_outer', lambda c: c['native_query']['total_seconds']),
                ('corr_native_kernel', lambda c: c['native_query']['native']['kernel_seconds']),
                ('corr_native_input_scan', lambda c: c['native_query']['native']['scan_seconds']),
            ):
                row[name] = sum(extract(c) for c in cs)
            rows.append(row)
        result[label] = dict(
            count=len(rows), source_indices=[f['index'] for f in frames],
            stats={k: stats([row[k] for row in rows]) for k in rows[0]},
            initialization_and_EOF_excluded=label != 'all_fresh_including_initialization',
        )
    return dict(
        schema='v40_saved_v39_M16_CPU_component_cost_analysis',
        status='saved_timing_consistency_passed_not_an_acceleration_experiment',
        scope='Saved JSON arithmetic only; no image/model/ORT/SDK/BA execution.',
        populations=result,
        metadata=dict(source_observations=len(run['frames']),
            completed_correlation_and_Update_calls=len(corr),
            stream_neural_calls=sum(u['stage'] == 'stream' for u in updates),
            EOF_neural_calls=sum(u['stage'] == 'termination_offline' for u in updates),
            full_capture_source_indices=sorted(captured), OpenCV_version=run['OpenCV_version']),
        limits=[
            'Correlation inner clocks are descriptive subsets of its outer clock, not added to it.',
            'The residual includes reprojection, context gather, copies, finite checks, keyframe flow and graph mutations; these were not separately timed.',
            'Update outer time includes original Torch execution plus input/output checks and copies; it is not pure neural compute.',
            'The apparent remaining cost is measured on a serial CPU replay, not an immutable lower bound after NPU offload.',
            'No sum of component P95 values is presented as end-to-end P95.',
            'Progress-file and console writes after publication are outside each admission-to-publication clock but inside replay elapsed time.',
            'No physical camera, 20 Hz, trajectory acceptance or default model change claim.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.run.read_bytes()
    result = analyze(json.loads(raw))
    result['run_sha256'] = hashlib.sha256(raw).hexdigest()
    result['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'run_sha256': result['run_sha256'],
                      'populations': {k:v['count'] for k,v in result['populations'].items()}}))


if __name__ == '__main__':
    main()
