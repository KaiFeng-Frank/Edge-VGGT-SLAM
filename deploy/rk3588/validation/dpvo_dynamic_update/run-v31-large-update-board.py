#!/usr/bin/env python3
"""RK3588-only next-step DPVO Update comparison on two saved E23520 states.

Reconstruct original retained-frame features, not a trajectory. Original Torch
Update and dynamic CPU ORT receive the same nonzero hidden state/context/corr.
Each repeat resets its inputs; no BA, admission, pose generation or fallback.
DPVO source/math: MIT, Copyright (c) 2022 Princeton Vision & Learning Lab.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import numpy as np

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2/experiments')
CONTRACT_SHA = '45ba417a63a813a3af0c33ed080fc404d9eb7052ddae309cb0dba02d4216c45f'
V30_RUNNER_SHA = '22c5af18f8009524e348f7c400a2a8b72b0dffb55176b27d82c5ec22231dde86'
ENCODER_SHA = 'fc6fc662613a720cfb779a360a63906a593a96ff08addecb9d01cfd8767ca7a8'
TOPOLOGY_SHA = '92c459ed444ea7086f3b838db5333ded8d10bb8374e54ecd6c1f93b030dee536'
DYNAMIC_SHA = '2e313911b4cf2ac18373cbd3b72dbfe81644918b92c4ffee32f1109e238aa692'
RESULT_NAMES = ('net', 'delta', 'weight')
GRAPH_NAMES = ('ii', 'jj', 'kk')
ORDERS = (('Torch', 'ORT'), ('ORT', 'Torch'), ('Torch', 'ORT'))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for value in iter(lambda: file.read(1 << 20), b''):
            digest.update(value)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def require_board():
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64',
            'Actual feature/model execution is RK3588 board-only')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'RK3588 device tree required')


def centers_from_patches(patches, views=17, count=96):
    require(isinstance(patches, np.ndarray) and patches.dtype == np.float32
            and patches.shape == (1, views * count, 3, 3, 3)
            and np.isfinite(patches).all(), 'Finite saved FP32 patch contract required')
    xy = patches[0, :, :2, 1, 1]
    require(np.equal(xy, np.floor(xy)).all(), 'Saved centers must be exact integers')
    require(((xy[:, 0] >= 1) & (xy[:, 0] <= 150)
             & (xy[:, 1] >= 1) & (xy[:, 1] <= 114)).all(), 'Saved centers must be interior')
    offsets = np.arange(-1, 2, dtype=np.float32)
    require(np.array_equal(patches[0, :, 0],
                           np.broadcast_to(xy[:, 0, None, None] + offsets[None, None, :],
                                           (views * count, 3, 3)))
            and np.array_equal(patches[0, :, 1],
                               np.broadcast_to(xy[:, 1, None, None] + offsets[None, :, None],
                                               (views * count, 3, 3))), 'Original xy patch grid changed')
    return np.ascontiguousarray(xy.reshape(views, count, 2), dtype=np.int64)


def validate_snapshot(values, views=17, count=96, edges=23520):
    shapes = dict(poses=(1, views, 7), patches=(1, views * count, 3, 3, 3),
                  intrinsics=(1, views, 4), net=(1, edges, 384))
    for name, shape in shapes.items():
        a = values[name]
        require(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all(),
                'Saved neural/geometry shape/type/finite differs: ' + name)
    for name in GRAPH_NAMES:
        a = values[name]
        require(a.dtype == np.int64 and a.shape == (edges,), 'Saved graph contract differs: ' + name)
    ii, jj, kk = (values[k] for k in GRAPH_NAMES)
    require(edges > 0 and np.array_equal(ii, kk // count)
            and (ii >= 0).all() and (ii < views).all() and (jj >= 0).all()
            and (jj < views).all() and (kk >= 0).all() and (kk < views * count).all(),
            'Saved original graph index/patch ownership differs')
    return centers_from_patches(values['patches'], views, count)


def output_checks(values, edges):
    require(isinstance(values, dict) and tuple(values) == RESULT_NAMES,
            'Exactly ordered net/delta/weight outputs required')
    result = {}
    for name, channels in zip(RESULT_NAMES, (384, 2, 2)):
        value = values[name]
        require(isinstance(value, np.ndarray) and value.dtype == np.float32
                and value.shape == (1, edges, channels), 'Output shape/type differs: ' + name)
        result[name] = dict(finite=bool(np.isfinite(value).all()),
                            nonfinite_elements=int(np.count_nonzero(~np.isfinite(value))))
    weight = values['weight']
    finite = result['weight']['finite']
    return dict(arrays=result, finite=all(v['finite'] for v in result.values()),
                weight_domain_passed=bool(finite and ((weight >= 0) & (weight <= 1)).all()),
                weight_min=float(weight.min()) if finite else None,
                weight_max=float(weight.max()) if finite else None)


def compare_outputs(actual, expected):
    edges = expected['net'].shape[1]
    checks = output_checks(actual, edges)
    reference = output_checks(expected, edges)
    rows = {}
    for key in RESULT_NAMES:
        a, b = actual[key], expected[key]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.all():
            difference = a.astype(np.float64) - b.astype(np.float64)
            failures = int(np.count_nonzero(np.abs(difference) > 1e-5 + 1e-4 * np.abs(b.astype(np.float64))))
            denominator = float(np.linalg.norm(b.astype(np.float64).ravel()))
            error_norm = float(np.linalg.norm(difference.ravel()))
            rows[key] = dict(failed_elements=failures, max_abs=float(np.abs(difference).max()),
                             relative_l2=error_norm / denominator if denominator else None,
                             RMS=float(np.sqrt(np.mean(difference * difference))))
        else:
            rows[key] = dict(failed_elements=int(a.size), max_abs=None, relative_l2=None, RMS=None)
        rows[key].update(rtol=1e-4, atol=1e-5, bitwise_equal=a.tobytes(order='C') == b.tobytes(order='C'))
    passed = checks['finite'] and reference['finite'] and checks['weight_domain_passed'] \
        and reference['weight_domain_passed'] and all(r['failed_elements'] == 0 for r in rows.values())
    return dict(outputs=rows, actual=checks, reference=reference, gate_passed=bool(passed))


def timed_torch(torch, update, inputs):
    """Complete input-copy, original class call and owned output-copy time."""
    begin = time.monotonic()
    arrays = tuple(inputs[k].copy(order='C') for k in ('net', 'context', 'corr', *GRAPH_NAMES))
    tensors = tuple(torch.from_numpy(v) for v in arrays)
    ready = time.monotonic()
    with torch.inference_mode():
        net, (delta, weight, unused) = update(*tensors[:3], None, *tensors[3:])
        require(unused is None, 'Original unused flow output changed')
        executed = time.monotonic()
        output = {k: v.detach().cpu().numpy().copy(order='C')
                  for k, v in zip(RESULT_NAMES, (net, delta, weight))}
    end = time.monotonic()
    return output, dict(total_seconds=end - begin, feed_copy_tensor_seconds=ready - begin,
                        original_Update_seconds=executed - ready, output_copy_seconds=end - executed)


def alternating_comparison(inputs, execute, save, record=None):
    """Tiny-testable scheduler; same original saved net for all six calls."""
    calls = []
    representatives = {}
    fingerprints = {}
    gates = []
    if record is not None:
        record.update(calls=calls, comparisons=gates, representative_outputs=representatives)
    for repeat, order in enumerate(ORDERS):
        pair = {}
        for method in order:
            try:
                values, timing = execute(method, inputs)
            except Exception as error:
                calls.append(dict(method=method, repeat=repeat, order=list(order),
                                  execution_failed=True, error=type(error).__name__ + ': ' + str(error)))
                raise
            checks = output_checks(values, len(inputs['kk']))
            metadata = {k: meta(v) for k, v in values.items()}
            if method not in fingerprints:
                fingerprints[method] = metadata
                artifact = save(method, values)
            else:
                artifact = None
            exact = metadata == fingerprints[method]
            calls.append(dict(method=method, repeat=repeat, order=list(order),
                              phase='first' if repeat == 0 else 'warm', timing=timing,
                              outputs=metadata, output_checks=checks,
                              repeat_bitwise_equal=exact, saved_artifact=artifact))
            pair[method] = values
            if repeat == 0:
                representatives[method] = artifact
        gate = compare_outputs(pair['ORT'], pair['Torch'])
        gates.append(dict(repeat=repeat, comparison=gate))
        del pair
    accepted = all(g['comparison']['gate_passed'] for g in gates) \
        and all(c['repeat_bitwise_equal'] for c in calls)
    timing = {}
    for method in ('Torch', 'ORT'):
        times = [c['timing']['total_seconds'] for c in calls if c['method'] == method]
        timing[method] = dict(first_seconds=times[0], warm_seconds=times[1:],
                              warm_mean_seconds=float(np.mean(times[1:])))
    return dict(calls=calls, comparisons=gates, representative_outputs=representatives,
                numerical_gate_passed=bool(accepted), timing=timing)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot-dir', 'dataset-dir', 'v30-runner', 'encoder-module', 'source-dir',
                 'checkpoint', 'geometry', 'native-helper', 'library', 'build-receipt',
                 'encoder-model', 'encoder-export', 'encoder-ort', 'update-manifest', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--contract', type=Path, default=Path(__file__).with_name('state-reconstruction-contract.json'))
    parser.add_argument('--dynamic-module', type=Path, default=Path(__file__).with_name('v31_dynamic_update.py'))
    parser.add_argument('--topology-module', type=Path, default=Path(__file__).with_name('v31_update_topology.py'))
    parser.add_argument('--update-manifest-sha256', required=True)
    parser.add_argument('--save-inputs', action='store_true')
    args = parser.parse_args(argv)
    require_board()  # Before output creation, image imports or actual model loading.
    out = args.output_dir.resolve()
    require(out.is_relative_to(BASE), 'New isolated board experiment output required')
    require(not any(p.resolve() == out or p.resolve().is_relative_to(out) for key, p in
                    vars(args).items() if key != 'output_dir' and isinstance(p, Path)),
            'Output would overlap an input')
    require(not any(out.is_relative_to(p.resolve()) for p in
                    (args.dataset_dir, args.snapshot_dir, args.source_dir)), 'Output lies within an input directory')
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    prior = os.sched_getaffinity(0)
    pins = {}
    encoder = correlation = dynamic = neural = None
    weights_before = None
    report = dict(schema='v31_DPVO_saved_large_Update_comparison', status='failed_execution',
                  script_sha256=sha(__file__), actual_board_execution=True, production_changed=False,
                  model_name='DPVO', original_DPVO_weights=True, different_model_from_VGGT=True,
                  original_VGGT_weights_used=False, ORB_used=False, NPU_used=False, BA_executed=False,
                  valid_pose_output=False, realtime_validated=False, states=[], reconstructed_frames=[],
                  cleanup_errors=[], order=[list(v) for v in ORDERS], fixed_gate=dict(rtol=1e-4, atol=1e-5))

    def pin(path, digest):
        path = Path(path).resolve()
        require(sha(path) == digest, 'Source/artifact SHA differs: ' + str(path))
        pins[str(path)] = digest
        return path

    def save(name, arrays):
        path = out / name
        np.savez(path, **arrays)
        return dict(file=name, bytes=path.stat().st_size, sha256=sha(path),
                    arrays={k: meta(v) for k, v in arrays.items()})

    try:
        pin(__file__, report['script_sha256'])
        contract = json.loads(pin(args.contract, CONTRACT_SHA).read_text())
        require(contract['schema'] == 'v31_DPVO_saved_state_reconstruction_static_contract'
                and len(contract['actual_saved_states']) == 2, 'Frozen reconstruction contract required')
        pin(args.v30_runner, V30_RUNNER_SHA)
        pin(args.encoder_module, ENCODER_SHA)
        pin(args.topology_module, TOPOLOGY_SHA)
        pin(args.dynamic_module, DYNAMIC_SHA)
        v30 = module(args.v30_runner, '_v31_frozen_v30_large')
        for path, digest in ((args.geometry, v30.GEOMETRY), (args.native_helper, v30.NATIVE),
                             (args.library, v30.LIBRARY), (args.build_receipt, v30.BUILD),
                             (args.checkpoint, v30.CHECKPOINT)):
            pin(path, digest)
        build = json.loads(args.build_receipt.read_text())
        require(build['status'] == 'compiled_and_synthetic_smoke_passed'
                and build['output']['sha256'] == v30.LIBRARY, 'Original native build chain differs')
        for name, digest in v30.PINS.items():
            pin(args.source_dir / name, digest)
        source_manifest = json.loads((args.source_dir / 'manifest.json').read_text())
        official = (args.source_dir / 'official').resolve()
        for row in source_manifest['retrieved_files']:
            path = (official / row['path']).resolve()
            require(path.is_relative_to(official), 'Official source path escaped')
            if row.get('available_in_pinned_tree') is False:
                require('sha256' not in row and not path.exists(), 'Absent official source unexpectedly present')
            else:
                pin(path, row['sha256'])
        for name, digest in contract['source_sha256'].items():
            if name.startswith('official/'):
                path = args.source_dir / name
            else:
                require(name.startswith('v30/'), 'Unknown reconstruction source domain')
                path = args.v30_runner.parent / name.removeprefix('v30/')
            pin(path, digest)
        run = json.loads(pin(args.snapshot_dir / 'run.json', contract['provenance']['v30_run_sha256']).read_text())
        require(run['status'] == 'completed_unpaced_fresh_frame_CPU_VO_research_not_realtime'
                and run['original_weights_unchanged'] and run['sources_and_inputs_unchanged']
                and not run['cleanup_errors'] and run['script_sha256'] == V30_RUNNER_SHA,
                'Original saved run lifecycle differs')
        source_inputs = json.loads(pin(args.snapshot_dir / 'input-manifest.json',
                                      contract['provenance']['input_manifest_sha256']).read_text())
        require(source_inputs['schema'] == 'v30_first_contiguous_TUM_RGB_frames', 'Original source manifest required')
        snapshots = []
        centers = []
        for row in contract['actual_saved_states']:
            saved = run[row['role']]
            item = saved['artifact']
            require(all(item[k] == row[k] for k in ('file', 'bytes', 'sha256', 'arrays')),
                    'Run/contract saved artifact differs')
            path = pin(args.snapshot_dir / row['file'], row['sha256'])
            require(path.stat().st_size == row['bytes'], 'Snapshot size differs')
            with np.load(path, allow_pickle=False) as z:
                values = {k: z[k].copy(order='C') for k in z.files}
            require({k: meta(v) for k, v in values.items()} == row['arrays'], 'Snapshot raw metadata differs')
            md = saved['metadata']
            for key in ('retained_observation_ids', 'retained_frame_ids', 'initialized'):
                require(md[key] == row[key], 'Retained source metadata differs: ' + key)
            selected = [source_inputs['frames'][i] for i in row['retained_observation_ids']]
            require(selected == row['retained_sources'] and [r['frame_id'] for r in selected] == row['retained_frame_ids'],
                    'Original retained observation identity/order differs')
            current = validate_snapshot(values)
            center_meta = row['patch_centers_int64_metadata']
            require(meta(current)['tensor_sha256'] == center_meta['tensor_sha256']
                    and list(current.shape) == center_meta['shape'], 'Saved center bytes differ')
            for a in values.values():
                a.flags.writeable = False
            snapshots.append(values)
            centers.append(current)
        require(np.array_equal(centers[0], centers[1])
                and all(np.array_equal(snapshots[0][k], snapshots[1][k])
                        for k in (*GRAPH_NAMES, 'intrinsics'))
                and contract['actual_saved_states'][0]['retained_sources'] == contract['actual_saved_states'][1]['retained_sources'],
                'Feature sharing requires identical centers/source order/K/graph')
        require({4, 5, 6, 7} <= prior, 'Four A76 CPUs required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import torch
        import cv2
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        common = module(args.source_dir / 'dpvo_probe_common.py', '_v31_original_large_neural')
        geo = module(args.geometry, '_v31_saved_geometry')
        native = module(args.native_helper, '_v31_saved_native')
        enc = module(args.encoder_module, '_v31_saved_encoder')
        topo = module(args.topology_module, '_v31_large_topology')
        dyn = module(args.dynamic_module, '_v31_large_dynamic')
        load_started = time.monotonic()
        neural, loader = common.load_neural(official, args.source_dir / 'manifest.json', args.checkpoint,
                                           v30.CHECKPOINT, args.source_dir / 'dpvo_cpu_ops.py',
                                           cpu_ops_sha256=v30.PINS['dpvo_cpu_ops.py'])
        weights_before = {k: meta(v.detach().numpy()) for k, v in neural.state_dict().items()}
        require(len(weights_before) == 94 and weights_before == run['original_weights_before'],
                'Original complete weights differ')
        report.update(loader=loader, original_weights_before=weights_before,
                      neural_load_seconds=time.monotonic() - load_started, cpu_affinity=sorted(os.sched_getaffinity(0)),
                      torch_threads=4, torch_interop_threads=1, torch_version=torch.__version__,
                      numpy_version=np.__version__, OpenCV_version=cv2.__version__,
                      reconstruction_contract_sha256=CONTRACT_SHA, dynamic_source_sha256=DYNAMIC_SHA)
        for path, digest in ((args.encoder_model, enc.MODEL_SHA), (args.encoder_export, enc.EXPORT_SHA),
                             (args.encoder_ort, enc.ORT_SHA)):
            pin(path, digest)
        encoder = enc.BoardORTEncoder(args.encoder_model, args.encoder_export, args.encoder_ort)
        manifest = json.loads(pin(args.update_manifest, args.update_manifest_sha256).read_text())
        require(manifest['original_weights_before'] == weights_before, 'Dynamic graph original94 weights differ')
        dynamic = dyn.DynamicORTUpdate(args.update_manifest, args.update_manifest_sha256, topo)
        for path, digest in dynamic.pins.items():
            pin(path, digest)
        correlation = v30.CompactCorrelation(native.PreparedCorrelation, args.library)
        K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], np.float64)
        D = np.array([.2624, -.9531, -.0054, .0026, 1.1633], np.float64)
        expected_K = np.array([517.3, 516.5, 302.6, 247.3], np.float32) / np.float32(4)
        require(np.array_equal(snapshots[0]['intrinsics'], np.broadcast_to(expected_K, (1, 17, 4))),
                'Saved feature calibration differs from original preprocessing')
        frames = []
        contexts = []
        reconstruction_started = time.monotonic()
        for slot, item in enumerate(contract['actual_saved_states'][0]['retained_sources']):
            path = (args.dataset_dir / item['relative_path']).resolve()
            require(path.is_relative_to(args.dataset_dir.resolve()), 'RGB input escaped dataset')
            pin(path, item['sha256'])
            require(path.stat().st_size == item['bytes'], 'RGB input size differs')
            begin = time.monotonic()
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            require(bgr is not None and bgr.dtype == np.uint8 and bgr.shape == (480, 640, 3), 'Original BGR shape differs')
            cropped = np.ascontiguousarray(cv2.undistort(bgr, K, D)[8:-8, 16:-16])
            decoded = time.monotonic()
            value = encoder.encode(cropped, centers=centers[0][slot].copy(), generator=None, patch_count=96)
            encoded = time.monotonic()
            require(np.array_equal(value['patches'][:, :2], snapshots[0]['patches'][0, slot * 96:(slot + 1) * 96, :2]),
                    'Reconstructed patch xy differs; original depth remains untouched')
            for key in ('fmap', 'gmap'):
                value[key].flags.writeable = False
            frames.append(dict(frame_id=item['frame_id'], fmap=value['fmap'], gmap=value['gmap']))
            contexts.append(value['context'])
            report['reconstructed_frames'].append(dict(slot=slot, source=item,
                decode_undistort_crop_seconds=decoded - begin, encoder_complete_seconds=encoded - decoded,
                encoder_timings=value['timing'], arrays={k: meta(value[k]) for k in ('fmap', 'gmap', 'context', 'centers')},
                source_pixels=enc.frame_input_metadata(cropped), saved_xy_match=True))
        context_all = np.ascontiguousarray(np.concatenate(contexts, axis=0)[None])
        del contexts
        report.update(reconstruction_seconds=time.monotonic() - reconstruction_started,
                      context_all=meta(context_all), saved_features_byte_equality_certificate_available=False)
        for index, (row, state) in enumerate(zip(contract['actual_saved_states'], snapshots)):
            record = dict(role=row['role'], snapshot_sha256=row['sha256'], edges=len(state['kk']))
            report['states'].append(record)
            begin = time.monotonic()
            coords = geo.reproject(*(state[k] for k in ('poses', 'patches', 'intrinsics', *GRAPH_NAMES)))
            projected = time.monotonic()
            corr = correlation(coords, **{k: state[k] for k in GRAPH_NAMES}, frames=tuple(frames))
            correlated = time.monotonic()
            context = np.ascontiguousarray(context_all[:, state['kk'], :])
            gathered = time.monotonic()
            inputs = dict(net=state['net'], context=context, corr=corr, **{k: state[k] for k in GRAPH_NAMES})
            record.update(reproject_seconds=projected - begin, correlation_complete_seconds=correlated - projected,
                          context_gather_seconds=gathered - correlated, input_arrays={k: meta(v) for k, v in inputs.items()},
                          coordinates=meta(coords), original_state_net_reset_each_call=True,
                          historical_target_and_weight_are_not_reference=True)
            if args.save_inputs:
                record['inputs_artifact'] = save('state-%d-inputs.npz' % index, dict(inputs, coords=coords))
            original_input_metadata = record['input_arrays']

            def execute(method, same_inputs):
                if method == 'Torch':
                    return timed_torch(torch, neural.update, same_inputs)
                begin = time.monotonic()
                result = dynamic(*(same_inputs[k] for k in ('net', 'context', 'corr', *GRAPH_NAMES)))
                end = time.monotonic()
                return result, dict(total_seconds=end - begin, runtime_breakdown=dynamic.metrics()['calls'][-1])

            record.update(alternating_comparison(inputs, execute,
                          lambda method, values: save('state-%d-%s.npz' % (index, method.lower()), values), record))
            require({k: meta(v) for k, v in inputs.items()} == original_input_metadata, 'Neural inputs mutated')
            record['inputs_unchanged'] = True
            print(json.dumps(dict(event='state_completed', role=row['role'], gate=record['numerical_gate_passed'],
                                  timing=record['timing']), allow_nan=False), flush=True)
            del inputs, coords, corr, context
        accepted = all(r['numerical_gate_passed'] for r in report['states'])
        report.update(status='completed_numerical_accepted' if accepted else 'completed_numerical_rejected',
                      candidate_accepted=accepted, dynamic_large_shape_gate_exercised=True)
    except Exception as error:
        report.update(error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc(),
                      candidate_accepted=False)
    finally:
        for name, handle in (('dynamic', dynamic), ('encoder', encoder), ('correlation', correlation)):
            if handle is None:
                continue
            try:
                if hasattr(handle, 'check_sources'):
                    report[name + '_sources_after'] = handle.check_sources()
            except Exception as error:
                report['cleanup_errors'].append(name + ' sources: ' + type(error).__name__ + ': ' + str(error))
            try:
                handle.close()
                report[name + '_metrics'] = handle.metrics()
            except Exception as error:
                report['cleanup_errors'].append(name + ' close: ' + type(error).__name__ + ': ' + str(error))
        if neural is not None:
            report['original_weights_after'] = {k: meta(v.detach().numpy()) for k, v in neural.state_dict().items()}
            report['original_weights_unchanged'] = report['original_weights_after'] == weights_before
            if not report['original_weights_unchanged']:
                report['cleanup_errors'].append('Original neural weights changed')
        try:
            report['source_hashes_after'] = {path: sha(path) for path in pins}
            report['sources_and_inputs_unchanged'] = report['source_hashes_after'] == pins
            require(report['sources_and_inputs_unchanged'], 'Pinned source/input changed')
        except Exception as error:
            report['cleanup_errors'].append('Source recheck: ' + str(error))
        try:
            os.sched_setaffinity(0, prior)
            report['affinity_restored'] = os.sched_getaffinity(0) == prior
            require(report['affinity_restored'], 'CPU affinity restore failed')
        except Exception as error:
            report['cleanup_errors'].append('Affinity restore: ' + str(error))
        if report['cleanup_errors']:
            report.update(status='failed_execution', candidate_accepted=False)
        report.update(source_pins=pins, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      actual_driver_elapsed_seconds=time.monotonic() - started,
                      timing_scope='Same-process AB/BA/AB per state. Update includes input copies, conversion, original/dynamic execution and owned output copies; ORT additionally includes topology/cache and checks. Image reconstruction, projection, full native correlation preparation/query, context gather, hashes/statistics and file I/O are separately recorded or outside Update time. No thermal control or full-VO latency claim.',
                      limitations=['Reconstructed singleton encoder features were not retained in v30; source pixels/centers are bound, but old feature byte equality cannot be certified.',
                                   'Snapshot target/weight are historical prior-step outputs, not the next-step reference.',
                                   'Only two fixed actual E23520 states, not recurrent streaming, BA, trajectory accuracy or 20Hz acceptance.',
                                   'Dynamic ORT is CPU execution; no NPU execution or GPU inference occurs.'])
        (out / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], candidate_accepted=report.get('candidate_accepted', False),
                          output=str(out / 'run.json'), sha256=sha(out / 'run.json')), allow_nan=False), flush=True)
    return 0 if report['status'] == 'completed_numerical_accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
