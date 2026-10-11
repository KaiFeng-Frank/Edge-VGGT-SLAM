#!/usr/bin/env python3
"""Read saved motion1 captures and metadata; execute no neural/native/BA code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

RUN_SHA = 'c6a3ef8a5a063fc8dfa1319841ea782f38c9cc1822ce7058aa312cec4665f7e3'
HALF_SHA = 'c1f0a4636601fb1cd702ca57018af5fd937ecdbb9e722b0293428810bf66d05d'
FULL_SHA = 'a0f18920b156f9f819630b591fa71af25f5e869a9a160ccd5c7d4b471218162e'
EVALUATION_SHA = '6f3009ce17ca920d3669d61748bdeacc95049c92bd9c766514d2a38a5cb8078b'
DRIVER_SHA = '3b0a5b809cab8277e1275bb0b3d4bacd99c14622e9b4091d980286276457543a'
PARENT_SHA = '18ad8f7bc3cc48790d2329b35372fd77473942c698cd3e7e291298194e498c2d'


def require(ok, reason):
    if not bool(ok):
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(directory, expected):
    path = Path(directory) / 'run.json'
    require(sha(path) == expected, 'Run SHA differs')
    return json.loads(path.read_text())


def callback_ast(path):
    main = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    update = next(n for n in main.body
                  if isinstance(n, ast.Try)).body
    function = next(n for n in update if isinstance(n, ast.FunctionDef) and n.name == 'update')
    return ast.dump(function, include_attributes=False)


def read_capture(root, descriptor, shapes, dtypes):
    name = descriptor['file']
    require(Path(name).name == name and name.endswith('.npz'), 'Capture path escape')
    path = root / name
    require(path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256'],
            'Capture file SHA/size differs: ' + name)
    records = {}
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(shapes) == set(descriptor['arrays']),
                'Capture keys differ')
        values = {key: archive[key] for key in archive.files}
    for key, value in values.items():
        raw_sha = hashlib.sha256(value.tobytes(order='C')).hexdigest()
        actual = dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                      tensor_sha256=raw_sha)
        require(value.shape == shapes[key] and str(value.dtype) == dtypes[key] and
                actual == descriptor['arrays'][key] and np.isfinite(value).all(),
                'Capture shape/dtype/SHA/finite differs: ' + name + ':' + key)
        records[key] = dict(**actual, finite=True, min=float(value.min()), max=float(value.max()))
    return values, dict(file=name, bytes=path.stat().st_size, sha256=sha(path), arrays=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run-dir', 'half-baseline-run-dir', 'full-baseline-run-dir',
                'evaluation', 'driver', 'baseline-driver', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh output path required')
    require(sha(args.evaluation) == EVALUATION_SHA and sha(args.driver) == DRIVER_SHA and
            sha(args.baseline_driver) == PARENT_SHA, 'Evaluation/driver source SHA differs')
    run = read_run(args.run_dir, RUN_SHA)
    half = read_run(args.half_baseline_run_dir, HALF_SHA)
    full = read_run(args.full_baseline_run_dir, FULL_SHA)
    evaluation = json.loads(args.evaluation.read_text())
    require(run['frames_completed'] == len(run['frames']) == 192 and
            run['schema'] == 'v45_half_motion1_original_Torch_CPU_research_replay' and
            run['status'] == 'completed_unpaced_half_motion1_Torch_CPU_research_not_realtime',
            'Actual completed research run required')
    same_callback = callback_ast(args.driver) == callback_ast(args.baseline_driver)
    require(same_callback and evaluation['source_result_first_divergence']['trial_isolation_verified'],
            'Frozen Torch capture callback/source isolation differs')
    plans = []
    for root, value in ((args.run_dir, run), (args.half_baseline_run_dir, half),
                        (args.full_baseline_run_dir, full)):
        require(sha(root / 'input-manifest.json') == value['input_manifest_sha256'], 'Manifest SHA differs')
        plans.append(json.loads((root / 'input-manifest.json').read_text())['frames'])
    require(len(plans[0]) == 192 and plans[0] == plans[1] == plans[2], '192 source rows differ')
    require(run['random_generator_before'] == half['random_generator_before'] == full['random_generator_before']
            and run['random_generator_after'] == half['random_generator_after']
            and run['random_depth_draw_calls'] == half['random_depth_draw_calls'] == 192,
            'Saved shared RNG metadata differs')
    require(len(run['original_weights_before']) == 94 and run['original_weights_unchanged'] and
            run['original_weights_before'] == run['original_weights_after'] ==
            half['original_weights_before'] == full['original_weights_before'] and
            run['source_pins'] == run['source_hashes_after'] and run['sources_and_inputs_unchanged']
            and run['cleanup_errors'] == [] and run['affinity_restored'], 'Saved stability/cleanup differs')
    require([row['frame_index'] for row in run['update_captures']] == [11, 127, 191],
            'Actual capture identities differ')
    captures = []
    for descriptor in run['update_captures']:
        frame, index, edges = (descriptor[key] for key in ('frame_index', 'update_index', 'edges'))
        callback = run['update_calls'][index]
        require(callback['index'] == index and callback['frame_index'] == frame and
                callback['stage'] == 'stream' and callback['edges'] == edges and
                descriptor['entire_active_graph'] is True and descriptor['same_call_original_Torch'] is True,
                'Same actual callback identity differs')
        shapes = dict(net=(1, edges, 384), inp=(1, edges, 384), corr=(1, edges, 882),
                      ii=(edges,), jj=(edges,), kk=(edges,))
        dtypes = {key: 'float32' if key in ('net', 'inp', 'corr') else 'int64' for key in shapes}
        inputs, input_record = read_capture(args.run_dir, descriptor['input'], shapes, dtypes)
        output_shapes = dict(net=(1, edges, 384), delta=(1, edges, 2), weight=(1, edges, 2))
        outputs, output_record = read_capture(args.run_dir, descriptor['gold'], output_shapes,
                                             {key: 'float32' for key in output_shapes})
        sources, counts = np.unique(inputs['ii'], return_counts=True)
        require(np.array_equal(inputs['ii'], inputs['kk'] // 16) and (inputs['ii'] >= 0).all() and
                (inputs['jj'] >= 0).all() and (inputs['kk'] >= 0).all() and (counts % 16 == 0).all()
                and sources.tolist() == callback['source_slots'] and counts.tolist() == callback['source_patch_counts']
                and (counts // 16).tolist() == callback['source_degrees'], 'Saved source/patch routing differs')
        require(((outputs['weight'] >= 0) & (outputs['weight'] <= 1)).all(), 'Output weight domain differs')
        captures.append(dict(frame_index=frame, update_index=index, edges=edges,
                             same_call_original_Torch_reported=True, entire_active_graph_reported=True,
                             callback_stage='stream', source_slots=sources.tolist(),
                             source_degrees=(counts // 16).tolist(), input=input_record, gold=output_record,
                             original_callback_AST_identical=True, old_gold_comparison=False))
    first = next(row['index'] for row in run['frames'] if row['result']['fresh_pose_valid'])
    require(first == 11 and run['encoder_metrics']['closed'] and
            run['correlation']['closed'] and run['correlation']['current_context_bytes'] == 0 and
            run['correlation']['current_cached_frames'] == 0 and run['undistort_cache']['closed'] and
            run['undistort_cache']['current_map_bytes'] == 0, 'Saved lifecycle differs')
    result = dict(schema='v45_motion1_saved_capture_source_RNG_audit', status='passed_saved_only',
        audit_source_sha256=sha(__file__), run_sha256=RUN_SHA, evaluation_sha256=EVALUATION_SHA,
        half_baseline_run_sha256=HALF_SHA, full_baseline_run_sha256=FULL_SHA,
        candidate_driver_sha256=DRIVER_SHA, original_cached_driver_sha256=PARENT_SHA,
        actual_run_schema=run['schema'], actual_run_status=run['status'],
        same_original_192_source_rows=True, first_fresh_source_index=first,
        first_admitted_source_indices=[row['index'] for row in run['frames'][:12] if row['result']['admitted']],
        initialization_rejected_source_indices=[row['index'] for row in run['frames'] if
                                                 row['result']['reason'] == 'initial_motion_below_1'],
        source_result_first_divergence=evaluation['source_result_first_divergence'],
        random_generator_before=run['random_generator_before'], random_generator_after=run['random_generator_after'],
        random_depth_draw_calls=run['random_depth_draw_calls'],
        half_terminal_RNG_metadata_equal=True, raw_RNG_state_array_reread=False,
        original94_weight_metadata_stable=True, sources_stable_in_saved_metadata=True,
        cleanup_errors=run['cleanup_errors'], all_contexts_closed_in_saved_metadata=True,
        reread_NPZ_files=6, reread_arrays=27, captures=captures,
        capture_semantics='Each gold is the output of its own actual original Torch callback on this new motion1 path. Observation11 is the first full initialization callback, not old observation34 gold or the last of its12 initialization Updates.',
        limits=['Array bytes/SHA, shape, dtype, finite/domain and same-call descriptors were reread. Original Torch callback source AST matches the frozen parent.',
                'No teacher equivalence, old capture comparison, model rerun, or trajectory acceptance inferred from captures.',
                'Source/weight/RNG/close facts are saved metadata; raw RNG state or live SDK/context was not reconstructed.'],
        model_forward_calls=0, ORT_calls=0, SDK_calls=0, native_calls=0, BA_calls=0, images_read=0, board_jobs=0)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=result['status'], reread_NPZ_files=6, reread_arrays=27,
                          audit_sha256=sha(args.output))))


if __name__ == '__main__':
    main()
