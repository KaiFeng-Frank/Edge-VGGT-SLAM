"""Saved-array audit only: no model, SDK, OpenCL or board access."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(np, current, reference):
    assert current.shape == reference.shape and current.dtype == reference.dtype
    delta = current.astype(np.float64) - reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    return dict(byte_equal=current.tobytes() == reference.tobytes(),
                changed_elements=int(np.count_nonzero(delta)),
                max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta.ravel()) / denominator) if denominator else None,
                reference_all_zero=denominator == 0)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--capture-sha256', required=True)
    ap.add_argument('--pair-result', type=Path, required=True)
    ap.add_argument('--pair-result-sha256', required=True)
    ap.add_argument('--pair-case', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    assert sha(args.capture) == args.capture_sha256
    assert sha(args.pair_result) == args.pair_result_sha256
    capture = json.loads(args.capture.read_text())
    pair = json.loads(args.pair_result.read_text())
    result = dict(schema='v83_saved_callback_repeat_boundary_audit',
                  model_SDK_OpenCL_or_board_executed=False,
                  capture_sha256=args.capture_sha256,
                  pair_result_sha256=args.pair_result_sha256,
                  pair_case=args.pair_case, roles={}, pair_outputs={}, artifacts=[],
                  source_of_failure_established=False,
                  scope='Same-call saved memory boundaries and first two repeated callback/pair outputs; no recomputed attention')
    rows = capture['captures']
    assert len(rows) == 4 and all(row['status'] == 'returned' for row in rows)
    for role in ('frame', 'global'):
        selected = sorted((row for row in rows if row['role'] == role), key=lambda row: row['role_callback_index'])
        assert [row['role_callback_index'] for row in selected] == [0, 1]
        arrays = []
        for row in selected:
            artifact = row['artifact']
            path = args.capture.parent / artifact['file']
            assert path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256']
            with np.load(path, allow_pickle=False) as archive:
                owned = {key: archive[key].copy() for key in archive.files}
            assert set(owned) == set(artifact['arrays'])
            for key, value in owned.items():
                info = artifact['arrays'][key]
                assert list(value.shape) == info['shape'] and str(value.dtype) == info['dtype']
                assert value.nbytes == info['bytes'] and np.isfinite(value).all()
                assert hashlib.sha256(value.tobytes()).hexdigest() == info['raw_sha256']
            arrays.append(owned)
            result['artifacts'].append(dict(file=str(path), sha256=artifact['sha256'], arrays=len(owned)))
        result['roles'][role] = dict(
            repeated={key: compare(np, arrays[1][key], arrays[0][key]) for key in arrays[0]},
            CPU_output_vs_GPU_import_readback=[compare(np, a['CPU_output_after_FROM_DEVICE'], a['GPU_proj_input']) for a in arrays],
            CPU_input_vs_GPU_import_readback=[{key: compare(np, a['CPU_input_' + str(i).zfill(2)], a[key])
                                               for i, key in enumerate(('GPU_q', 'GPU_k', 'GPU_v'))} for a in arrays],
            actual_program=[row['GPU_bridge_metrics']['actual_SDK_program'] for row in selected])
    calls = [row for row in pair['calls'] if row['case'] == args.pair_case and row['route'] == 'B']
    assert len(calls) == 2
    outputs = []
    for row in calls:
        artifact = row['saved_output']
        path = args.pair_result.parent / artifact['file']
        assert sha(path) == artifact['sha256'] and path.stat().st_size == artifact['bytes']
        with np.load(path, allow_pickle=False) as archive:
            owned = {key: archive[key].copy() for key in archive.files}
        assert all(np.isfinite(value).all() for value in owned.values())
        outputs.append(owned)
        result['artifacts'].append(dict(file=str(path), sha256=artifact['sha256'], arrays=len(owned)))
    result['pair_outputs'] = {key: compare(np, outputs[1][key], outputs[0][key]) for key in outputs[0]}
    frame = result['roles']['frame']['repeated']
    result['observations'] = dict(
        frame_GPU_input_and_all_internal_saved_arrays_byte_stable=all(
            value['byte_equal'] for key, value in frame.items() if key.startswith('GPU_')),
        frame_callback_CPU_and_GPU_output_agree_all=all(
            x['byte_equal'] for x in result['roles']['frame']['CPU_output_vs_GPU_import_readback']),
        whole_pair_frame_repeated_output_byte_stable=result['pair_outputs']['frame']['byte_equal'],
        global_GPU_inputs_byte_stable=all(result['roles']['global']['repeated'][key]['byte_equal']
            for key in ('GPU_q', 'GPU_k', 'GPU_v')),
        first_CPU_QKV_pre_sync_all_zero=all(frame['CPU_input_' + str(i).zfill(2)]['reference_all_zero'] for i in range(3)))
    result['status'] = 'saved_hash_shape_finite_and_comparisons_verified'
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], observations=result['observations'],
                         report_sha256=sha(args.output)), indent=2))


if __name__ == '__main__':
    main()
