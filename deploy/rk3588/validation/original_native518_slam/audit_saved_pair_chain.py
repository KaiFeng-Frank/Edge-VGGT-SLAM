#!/usr/bin/env python3
"""Audit saved predictions/PNG/SL4/PLY/JSON only; no NN, ORT, SDK or GPU calls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

R = Path(__file__).resolve().parent
RUNNER_SHA = 'd42ccc8d26c6f56e0abdd23c2e509a5c9355d3008c8d9f9fdfc735857f252d15'
EXECUTOR_SHA = '8923992b323f805c03d8e70441952b65aefe9f8a70a3cfc2241ac14b704b563f'
REFERENCE_SHA = '4af7b9c4cbd38dbfd54c14b1308dd6035da25941f98f858a5ad13b781c3c8af6'
ALIASES = {'pose_enc':'pose_enc','extrinsic':'extrinsics','intrinsic':'intrinsics','depth':'depth','depth_conf':'depth_conf'}
COUNTERS = ('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')
PER_CALL = dict(zip(COUNTERS,(10,10,18,23,23)))
TOTALS = dict(zip(COUNTERS,(20,20,36,46,46)))
NAMES = ['1305031103.243216.png','1305031103.511333.png','1305031103.775342.png']


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4 << 20), b''): h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, default=R/'pair-chain-r1/result.json')
    p.add_argument('--result-sha256', required=True)
    p.add_argument('--control', type=Path, default=R/'control-r1/control.json')
    p.add_argument('--control-sha256', required=True)
    p.add_argument('--reference', type=Path, default=R/'original-full-model-reference/result.json')
    p.add_argument('--source-images', type=Path, default=R/'source-images')
    p.add_argument('--output', type=Path, default=R/'saved-pair-chain-audit.json')
    a = p.parse_args(); env = dict(os.environ); pins = {}; verified = []
    out = dict(schema='v109_independent_saved_native518_SL4_chain_audit', status='failed', auditor_sha256=sha(__file__),
        saved_data_only=True, model_ORT_SDK_GPU_execution=False, trajectory_quality_realtime_map_cadence_30min_or_8GB_accepted=False)
    try:
        os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        import numpy as np
        def pin(path, digest):
            path = Path(path).resolve(strict=True)
            require(sha(path) == digest, 'File SHA differs: '+str(path)); pins[path] = digest
            return path
        def receipt(path, digest):
            path = pin(path, digest); return json.loads(path.read_text()), path.parent
        def artifact(root, row):
            rel = Path(row['file']); require(not rel.is_absolute() and '..' not in rel.parts, 'Relative artifact required')
            path = pin(root/rel, row['sha256'])
            require(path.is_relative_to(root) and path.stat().st_size == row['bytes'], 'Artifact size/path differs')
            return path
        def array(root, row, dtype):
            path = artifact(root,row); value = np.load(path, allow_pickle=False)
            require(value.dtype == dtype and list(value.shape) == row['shape'] and value.flags.c_contiguous and np.isfinite(value).all()
                and hashlib.sha256(value.tobytes()).hexdigest() == row['raw_sha256'], 'Array dtype/shape/raw/finite differs')
            verified.append(dict(file=str(path), sha256=row['sha256'], raw_sha256=row['raw_sha256'], shape=list(value.shape), finite=True))
            return value
        pin(R/'run_native518_slam_pair_chain.py',RUNNER_SHA); pin(R/'original_native518_executor.py',EXECUTOR_SHA)
        result, rd = receipt(a.result,a.result_sha256); control, _ = receipt(a.control,a.control_sha256)
        reference, refdir = receipt(a.reference,REFERENCE_SHA)
        require(result['schema'] == 'v109_original_native518_two_pair_SL4_chain'
            and result['status'] == 'completed_original_native518_two_pair_SL4_chain_not_quality_or_realtime_accepted'
            and result['script_sha256'] == RUNNER_SHA and result['executor_sha256'] == EXECUTOR_SHA
            and result['max_loops'] == 0 and result['image_size'] == 518 and result['source_indices'] == [32,40,48]
            and result['prepared_image_shape'] == [2,3,392,518] and len(result['pairs']) == 2
            and result['cleanup_errors'] == [] and result['source_and_inputs_unchanged']
            and result['environment_restored'] and result['affinity_restored'] and result['sys_path_restored'], 'Complete pinned native SL4 chain required')
        sources = result['source_images']; stamps = [float(Path(n).stem) for n in NAMES]
        require([Path(v['path']).name for v in sources] == NAMES and [v['timestamp'] for v in sources] == stamps
            and len({v['sha256'] for v in sources}) == 3, 'Three original chronological source identities required')
        for row in sources:
            path = pin(a.source_images/Path(row['path']).name,row['sha256'])
            require(path.stat().st_size == row['bytes'], 'Source PNG size differs')
        refs = {name:array(refdir,reference['calls'][0]['artifacts'][alias],np.float32)[0] for name,alias in ALIASES.items()}
        per_pair = []; pixel_rows = []; all_values = []; per_call_totals = []
        for i, row in enumerate(result['pairs']):
            require(row['index'] == i and row['source_indices'] == [32,40,48][i:i+2]
                and row['timestamps'] == stamps[i:i+2] and [Path(p).name for p in row['image_paths']] == NAMES[i:i+2]
                and row['prepared_image_shape'] == [2,3,392,518] and row['graph_nodes'] == row['graph_factors'] == 2*(i+1)
                and row['map_submaps'] == i+1 and row['published_unique_timestamps'] == i+2
                and math.isfinite(row['graph_error']), 'Pair identity/graph counts differ')
            call = row['executor_call']; raw = row['prepared_frame_raw_sha256']; pixel_rows.append(raw)
            require(call['index'] == i and call['status'] == 'completed' and call['input_frame_raw_sha256'] == raw
                and raw[0] != raw[1] and call['actual_fresh_caller_pixels'] and not call['fixed_reference_pixel_substitution']
                and not call['image_or_feature_cache_used'] and not call['intermediate_reference_substitution'], 'Fresh pixel receipt/source proof differs')
            values = {name:array(rd,meta,np.float32) for name,meta in row['outputs'].items()}; all_values.append(values)
            require(set(values) == set(ALIASES), 'Five original canonical predictions required')
            parity = {name:bool(np.array_equal(value,refs[name]) and value.tobytes() == refs[name].tobytes()) for name,value in values.items()} if i == 0 else None
            require(parity is None or all(parity.values()), 'First complete output differs from v100 reference[0]')
            for name,value in values.items():
                require(call['final_outputs'][name]['raw_sha256'] == hashlib.sha256(value.tobytes()).hexdigest(), 'Executor/output raw identity differs')
            owners = [v for name,v in call['resources'].items() if name != 'heads']+call['resources']['heads']['contexts']
            totals = {key:sum(v[key] for v in owners) for key in COUNTERS}
            require(totals == PER_CALL == call['SDK_totals'] and all(v['closed'] and not v['live_contexts']
                and not v['live_output_buffers'] and not v['cleanup_errors'] for v in owners), 'Per-pair owner ledger differs')
            per_call_totals.append(totals)
            timing_keys = ('prediction_seconds','identity_parity_and_prediction_save_seconds','points_scale_factors_LM_seconds','publication_seconds')
            require(all(math.isfinite(row[k]) and row[k] >= 0 for k in timing_keys)
                and math.isclose(sum(row[k] for k in timing_keys),row['image_read_to_original_pose_publication_seconds'],abs_tol=1e-9), 'Complete diagnostic timing attribution differs')
            per_pair.append(dict(index=i, timestamps=row['timestamps'], first_pair_v100_byte_equal=parity,
                prediction_seconds=row['prediction_seconds'], original_SL4_and_points_seconds=row['points_scale_factors_LM_seconds'],
                image_read_to_original_pose_publication_seconds=row['image_read_to_original_pose_publication_seconds'],
                graph_error_internal_tree_consistency_only=row['graph_error'], SDK_totals=totals))
        require(pixel_rows[0][1] == pixel_rows[1][0] and pixel_rows[1][1] not in pixel_rows[0], 'Actual pixel overlap or fresh target receipt differs')
        z = result['executor_resources']; totals = {k:sum(v[k] for v in per_call_totals) for k in COUNTERS}
        require(totals == TOTALS == z['SDK_totals'] and z['closed'] and not z['live_contexts'] and not z['live_output_buffers']
            and z['calls_attempted'] == z['calls_completed'] == 2 and z['cleanup_errors'] == [], 'Cumulative original executor ledger differs')
        metadata = json.loads(artifact(rd,result['map_manifest_artifact']).read_text())
        require(metadata['submaps'] == 2 and metadata['graph_nodes'] == metadata['graph_factors'] == 4
            and metadata['unique_timestamps'] == stamps and [v['node_ids'] for v in metadata['nodes']] == [[0,1],[2,3]]
            and [v['frame_timestamps'] for v in metadata['nodes']] == [stamps[:2],stamps[1:]], 'Original map metadata differs')
        homographies = array(rd,metadata['homographies'],np.float64)
        require(homographies.shape == (4,4,4), 'Four original homographies required')
        trajectory = np.loadtxt(artifact(rd,result['poses_artifact']))
        require(trajectory.shape == (3,8) and np.isfinite(trajectory).all() and np.all(np.diff(trajectory[:,0]) > 0)
            and np.allclose(trajectory[:,0],stamps,rtol=0,atol=1e-7), 'Original three unique TUM timestamps differ')
        ply = artifact(rd,result['map_PLY_artifact'])
        with ply.open('rb') as f:
            lines = []
            while True:
                line = f.readline(); require(line and len(lines) < 100,'Invalid PLY header')
                lines.append(line.decode('ascii').strip())
                if lines[-1] == 'end_header': break
            offset = f.tell()
        require(lines[:2] == ['ply','format binary_little_endian 1.0'] and
            [v for v in lines if v.startswith('property ')] == ['property double '+v for v in ('x','y','z')]+['property uchar '+v for v in ('red','green','blue')], 'Original PointCloud PLY layout differs')
        count = int(next(v.split()[-1] for v in lines if v.startswith('element vertex ')))
        expected_points = sum(sum(v['filtered_point_counts']) for v in metadata['nodes'])
        dtype = np.dtype([(v,'<f8') for v in ('x','y','z')]+[(v,'u1') for v in ('red','green','blue')])
        require(count == expected_points == sum(sum(v['current_filtered_points']) for v in result['pairs'])
            and ply.stat().st_size == offset+count*dtype.itemsize, 'PLY vertex ledger/size differs')
        vertices = np.memmap(ply,dtype=dtype,mode='r',offset=offset,shape=(count,))
        require(all(np.isfinite(vertices[v]).all() for v in ('x','y','z')), 'Saved map coordinates nonfinite')
        observations = control['observations']
        require(control['child_returncode'] == 0 and control['restore_errors'] == [] and control['frequency_governor_limits_restored']
            and control['restored'] == control['baseline'] and control['requested_frequency_hz'] == 1000000000
            and observations and all(v['actual_frequency_hz'] == 1000000000 for v in observations), 'Clock/controller restoration differs')
        temps = [v for row in observations for v in row['temperatures_c'].values()]
        require(max(temps) < 75, 'Observed thermal limit exceeded')
        out.update(status='passed_independent_saved_native518_SL4_chain_audit', first_pair_all5_raw_byte_equal_v100=True,
            source_images_independently_rehashed=True, shared_image_and_new_pixel_receipt_consistency=True,
            prepared_pixel_SHA_independently_recomputed=False, source_images=sources, verified_arrays=verified, pairs=per_pair,
            SDK_totals=totals, graph_nodes=4, graph_factors=4, submaps=2, unique_timestamps=trajectory[:,0].tolist(),
            saved_homography_determinants=np.linalg.det(homographies).tolist(), trajectory_quaternion_norms=np.linalg.norm(trajectory[:,4:8],axis=1).tolist(),
            PLY_vertices=count, PLY_bytes=ply.stat().st_size, PLY_coordinates_finite=True,
            controller_observations=len(observations), observed_peak_temperature_c=max(temps), NPU_restored_hz=int(control['restored']['cur_freq']),
            limits=['Two sequential overlapping original native518 pairs; no paced fresh-stream/P95/20Hz or trajectory-quality acceptance.',
                'Prepared RGB arrays were not saved; rawpixel hashes are checked across board runner/executor receipts, not re-preprocessed on host.',
                'The no-loop tree graph residual checks internal factor consistency, not pose accuracy or metric map scale.'])
    except BaseException as exc: out.update(error=type(exc).__name__+': '+str(exc))
    finally:
        out['file_sha256'] = {str(p):h for p,h in pins.items()}
        changed = [str(p) for p,h in pins.items() if not p.is_file() or sha(p) != h]
        out.update(saved_files_unchanged=not changed, changed_files=changed)
        if changed: out['status'] = 'failed'
        os.environ.clear(); os.environ.update(env)
        with a.output.open('x') as f: json.dump(out,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:out.get(k) for k in ('status','error','first_pair_all5_raw_byte_equal_v100','PLY_vertices','SDK_totals')}))
    return 0 if out['status'].startswith('passed_') else 1


if __name__ == '__main__': raise SystemExit(main())
