#!/usr/bin/env python3
"""Commit saved complete original VGGT predictions through the unchanged SL4.

Board-only saved-result reconstruction; never loads or calls a model/SDK.
Offline commit time is not camera input-to-pose latency or live throughput.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
KEYS = ('pose_enc', 'extrinsic', 'intrinsic', 'depth', 'depth_conf')
CHECKPOINT = {'bytes': 5026874952, 'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'}
SOURCE_PINS = {
    'vggt_slam/model_execution.py': 'dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87',
    'vggt_slam/solver.py': '33b5b4d816cd08f0cb9857f89c9e3fc770ad0c2ab3c9dc938864f00cefe86905',
    'vggt_slam/submap.py': 'fddccafee4e9f32452ce044b80669bd48f5fe5f331a980986a6091e7fce92e63',
    'vggt_slam/graph.py': 'dda8fb7c947c3c39baa7d190e2942cd4b9f26fe1557d7605fdc8651f27a5ab48',
    'vggt_slam/map.py': '2d7b374101248519d921e16e272063ff6f7e001c7af0073a7a7c13c14ac79143',
    'vggt_slam/scale_solver.py': 'dc36b11d4d9953222a3fc2f3aef274acc9e9cc6dc61cb22d86b1e6721d9bd691',
    'vggt_slam/slam_utils.py': 'aeba669bcb263f24e6bd93e5a176bc028e2022770499bd5ea1c3eeeebbbd5008',
    'vggt_slam/runtime.py': 'f36bb5b87fbdbf03984ea3e1466e6ee27865a540ae3e5abd4ee07a45538e23f5',
    'vggt_slam/frame_overlap.py': 'bbb78f549a34b58eaf27556e6dda703468cc0518de4b936f7f886882d151afb8',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class FrozenPredictionExecutor:
    """One saved original prediction, bound to the exact ordered RGB request."""
    device = 'cpu'

    def __init__(self, execution, np, images, outputs, expected_raw_sha256):
        self.execution, self.np = execution, np
        self.capabilities = execution.ExecutionCapabilities('saved-original-VGGT', False)
        self.images, self.outputs, self.expected = images, outputs, expected_raw_sha256
        self.calls = 0
        require(tuple(images.shape) == (2, 3, 84, 112), 'Original112 RGB shape differs')
        execution.validate_images(images)
        require(hashlib.sha256(images.tobytes()).hexdigest() == self.expected, 'Prepared RGB SHA differs')
        require(set(outputs) == set(KEYS), 'The five original canonical outputs are required')
        execution.validate_predictions(outputs, images, compute_similarity=False)
        require(all(value.flags.owndata for value in outputs.values()), 'Prediction storage must be owned')

    def predict(self, images, *, compute_similarity=False):
        require(compute_similarity is False and self.calls == 0, 'Only one ordinary saved call is allowed')
        require(images is self.images and hashlib.sha256(images.tobytes()).hexdigest() == self.expected,
                'Saved prediction does not match this ordered prepared request')
        self.calls += 1
        return self.execution.validate_predictions(self.outputs, images, compute_similarity=False)


def validate_chain(cases):
    """Reject disjoint calibration pairs before constructing any SLAM graph."""
    require(bool(cases), 'No complete pairs')
    for index, case in enumerate(cases):
        require(case['case_index'] == index and case['pair'] == index + 1, 'Pair order changed')
        require(len(case['source_indices']) == len(case['image_paths']) == len(case['timestamps']) == 2,
                'Exactly two original source frames are required')
        require(case['source_indices'][0] < case['source_indices'][1]
                and all(math.isfinite(v) for v in case['timestamps'])
                and case['timestamps'][0] < case['timestamps'][1], 'Source frame order differs')
        if index:
            previous = cases[index - 1]
            require(previous['source_indices'][1] == case['source_indices'][0]
                    and previous['image_paths'][1] == case['image_paths'][0]
                    and previous['timestamps'][1] == case['timestamps'][0],
                    'SL4 needs an identical previous-tail/current-first overlap; disjoint pairs rejected')


def commit_chain(rows, output_dir, *, solver, execution, np):
    """Parent may reuse this ordered commit with its already owned unique Solver.

    Each row: case (original IDs/paths/timestamps), images (original prepared
    owned RGB), outputs (five owned canonical arrays), prepared_sha256.
    It invokes only the saved executor, original Submap/SL4 and map publication.
    """
    validate_chain([row['case'] for row in rows])
    output_dir = Path(output_dir)
    chunks = output_dir / 'map_chunks'
    chunks.mkdir()
    manifest = dict(format='vggt_camera_local_dense_chunks_with_SL4_graph', units='model_units',
                    chunks=[], nodes={}, revision=0, metric_scale_validated=False, rectification=None)
    updates = []
    for row in rows:
        case, images, outputs = row['case'], row['images'], row['outputs']
        executor = FrozenPredictionExecutor(execution, np, images, outputs, row['prepared_sha256'])
        begin = time.monotonic()
        prediction = solver.run_predictions(case['image_paths'], executor, max_loops=0, prepared_images=images)
        current = solver.current_working_submap
        require(current.frame_ids == case['timestamps'] and executor.calls == 1, 'Submap source identity differs')
        solver.add_points(prediction)
        solver.graph.optimize()
        graph_error = float(solver.graph.graph.error(solver.graph.values))
        require(math.isfinite(graph_error), 'Nonfinite SL4 graph error')
        pose = np.asarray(current.get_all_poses_world(solver.graph)[-1])
        require(pose.shape == (4, 4) and np.isfinite(pose).all(), 'Nonfinite original SL4 pose')
        saved, counts = {}, []
        for index in range(len(current.pointclouds)):
            mask = current.conf_masks[index] > current.conf_threshold
            points = np.asarray(current.pointclouds[index][mask], np.float32)
            colors = np.asarray(current.colors[index][mask], np.uint8)
            require(np.isfinite(points).all() and points.shape == (len(points), 3), 'Nonfinite map chunk')
            saved[f'points_{index}'], saved[f'colors_{index}'] = points, colors
            counts.append(len(points))
        node = current.get_id()
        path = chunks / f'submap_{node:06d}.npz'
        np.savez(path, **saved)
        manifest['chunks'].append(dict(path=f'map_chunks/{path.name}', submap_id=node,
                                       nodes=[node + i for i in range(len(counts))], point_counts=counts))
        for first, submap in sorted(solver.map.submaps.items()):
            for index in range(len(submap.pointclouds)):
                H = np.asarray(solver.graph.get_homography(first + index), np.float64)
                require(H.shape == (4, 4) and np.isfinite(H).all(), 'Invalid SL4 graph homography')
                manifest['nodes'][str(first + index)] = H.tolist()
        manifest['revision'] += 1
        temporary = output_dir / 'poses.tum.tmp'
        solver.map.write_poses_to_file(str(temporary), solver.graph, kitti_format=False)
        temporary.replace(output_dir / 'poses.tum')
        atomic_json(output_dir / 'map.json', manifest)
        updates.append(dict(case_index=case['case_index'], pair=case['pair'], source_indices=case['source_indices'],
                            timestamps=case['timestamps'], graph_error=graph_error, pose=pose.tolist(),
                            offline_commit_and_publication_seconds=time.monotonic() - begin, executor_calls=1))
        require(hashlib.sha256(images.tobytes()).hexdigest() == row['prepared_sha256'], 'Original RGB mutated by commit')
    return dict(updates=updates, ordered_commits=len(updates), original_model_forward_calls=0,
                saved_executor_calls=len(updates), graph_nodes=int(solver.graph.values.size()),
                map_chunks=len(manifest['chunks']), point_count=sum(sum(c['point_counts']) for c in manifest['chunks']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-result', type=Path, required=True)
    parser.add_argument('--source-result-sha256', required=True)
    parser.add_argument('--route', choices=('serial', 'parallel'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=BASE / 'src')
    args = parser.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'SL4 reconstruction is permitted only on RK3588')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0)
    report = dict(schema='v79_original_VGGT_ordered_saved_SL4_commit', status='failed',
                  script_sha256=sha(__file__), source_result_sha256=args.source_result_sha256, route=args.route,
                  original_model_only=True, original_model_or_SDK_executed=False, candidate_accepted=False,
                  timing_scope='Offline saved-prediction SL4 and publication; not live input-to-pose latency or throughput',
                  realtime_validated=False, cleanup_errors=[])
    begin = time.monotonic()
    pins = {args.source_result.resolve(): args.source_result_sha256}
    try:
        require(sha(args.source_result) == args.source_result_sha256, 'Source result SHA differs')
        source = json.loads(args.source_result.read_text())
        require(source['schema'] == 'v79_original_VGGT_full_parallel_diagnostic'
                and source['status'] == 'completed_not_quality_acceptance' and source['original_model_only'] is True
                and source['checkpoint'] == CHECKPOINT, 'Original full-model source identity/status differs')
        cases = source['cases']
        validate_chain(cases)
        for relative, expected in SOURCE_PINS.items():
            path = (args.repo / relative).resolve(strict=True)
            require(sha(path) == expected, f'Original SLAM source changed: {relative}')
            pins[path] = expected
        sys.path.insert(0, str(args.repo.resolve(strict=True)))
        import numpy as np
        from vggt_slam import model_execution as execution
        from vggt_slam.solver import Solver
        require(Path(execution.__file__).resolve() == (args.repo / 'vggt_slam/model_execution.py').resolve(),
                'Executor import escaped pinned source')
        os.sched_setaffinity(0, {6, 7})
        jobs = source['routes'][args.route]['jobs']
        require(len(jobs) == len(cases), 'Exactly one prediction per source pair required')
        ordered = sorted(jobs, key=lambda job: job['case_index'])
        require([job['case_index'] for job in ordered] == list(range(len(cases))), 'Missing/duplicate prediction')
        rows = []
        for case, job in zip(cases, ordered):
            require(job['image_shape'] == [2, 3, 84, 112], 'Original prepared grid differs')
            require([image['path'] for image in job['source_images']] == case['image_paths'], 'Image identities differ')
            for image in job['source_images']:
                path = Path(image['path']).resolve(strict=True)
                require(sha(path) == image['sha256'], 'Original image bytes changed')
                pins[path] = image['sha256']
            artifact = job['artifact']
            path = (args.source_result.parent / artifact['file']).resolve(strict=True)
            require(path.is_relative_to(args.source_result.parent.resolve()) and path.stat().st_size == artifact['bytes']
                    and sha(path) == artifact['sha256'], 'Saved prediction artifact differs')
            pins[path] = artifact['sha256']
            with np.load(path, allow_pickle=False) as archive:
                require(set(archive.files) == set(KEYS) | {'images'}, 'Saved prediction fields differ')
                arrays = {key: archive[key].copy() for key in archive.files}
            for key, array in arrays.items():
                meta = artifact['arrays'][key]
                require(list(array.shape) == meta['shape'] and str(array.dtype) == meta['dtype']
                        and np.isfinite(array).all() and hashlib.sha256(array.tobytes()).hexdigest() == meta['raw_sha256'],
                        f'Saved array metadata differs: {key}')
            rows.append(dict(case=case, images=arrays.pop('images'), outputs=arrays,
                             prepared_sha256=job['preprocessed_images_raw_sha256']))
        solver = Solver(25., device='cpu', image_size=112, max_loops=0, offline=True, enable_viewer=False)
        report.update(commit_chain(rows, args.output_dir, solver=solver, execution=execution, np=np))
        report['status'] = 'completed_saved_SL4_not_realtime_acceptance'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        try:
            require(all(sha(path) == expected for path, expected in pins.items()), 'Source or saved artifact changed')
            report['source_and_input_hashes_unchanged'] = True
        except BaseException as exc:
            report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
        try:
            os.sched_setaffinity(0, affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
        except BaseException as exc:
            report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
        report['seconds_including_saved_validation'] = time.monotonic() - begin
        if report['cleanup_errors']:
            report['status'] = 'failed'
        atomic_json(args.output_dir / 'result.json', report)
    return 0 if report['status'] == 'completed_saved_SL4_not_realtime_acceptance' else 1


if __name__ == '__main__':
    raise SystemExit(main())
