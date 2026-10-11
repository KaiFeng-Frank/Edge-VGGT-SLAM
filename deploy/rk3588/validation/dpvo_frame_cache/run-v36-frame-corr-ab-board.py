#!/usr/bin/env python3
"""Board-only saved-three-frame old/new correlation ABBA; no neural forward/BA."""
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
CAPTURE_SHA = 'e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088'
MOTION_SHA = '55a7db19d1a9bfbf986da62ed999fc8b122add0fca84f01681e6ad0b82ba4e86'
PARENT_REPLAY_SHA = '84b8f02e14c0695c79d5c47319ff47ec4c7a8dcf53a3b38d905e03bc1efde6a8'
OLD_LIBRARY_SHA = '829d1b42fa24d28b1e280b882c0a9e8556b0e7cdbe02f7489eeef15ba3581bca'
OLD_HELPER_SHA = 'a6c37bf4b840b492736d5c5916be6e2c0e8680693d01d0fe3a5fe62702ecb779'
CANDIDATE_HELPER_SHA = '9743962325ba1c4ada689769eaf800a617ba1f46f4602062a6dab3b23b5bdf48'
BUILD_SCRIPT_SHA = 'b1f2553aa0e7786dd1343288ea67dd05d5b6e9f470c88693a6134d5d82756d2d'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def meta(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes,
                tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def board_guard():
    require(platform.machine() == 'aarch64' and platform.system() == 'Linux', 'RK3588 board only')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588 device tree required')


def comparison(actual, expected):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32,
            'Equal FP32 comparison shape/type required')
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return dict(finite=False, gate_passed=False, failed_elements=None, max_abs=None,
                    relative_L2=None, bitwise_equal=False, rtol=1e-4, atol=1e-5)
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    d = a-b
    failed = int(np.count_nonzero(np.abs(d) > 1e-5+1e-4*np.abs(b)))
    n = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, failed_elements=failed, gate_passed=failed == 0,
                max_abs=float(np.abs(d).max(initial=0)),
                relative_L2=float(np.linalg.norm(d.ravel())/n) if n else None,
                bitwise_equal=actual.tobytes() == expected.tobytes(), rtol=1e-4, atol=1e-5)


def load_artifact(folder, item, array_metadata, pin):
    name = item['file']
    require(isinstance(name, str) and Path(name).name == name, 'Safe artifact basename required')
    path = pin(Path(folder)/name, item['sha256'])
    require(path.stat().st_size == item['bytes'], 'Artifact byte count differs')
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k].copy(order='C') for k in z.files}
    require(set(values) == set(array_metadata), 'Artifact array keys differ')
    for key, value in values.items():
        require(meta(value) == array_metadata[key] and np.isfinite(value).all(), 'Artifact raw array metadata differs: '+key)
        value.flags.writeable = False
    return values


def remap_graph(graph, permutation, patch_count, remove=None):
    """Retain source edge order and actual xy; only CURRENT frame slots change."""
    original_count = len(permutation)+(1 if remove is not None else 0)
    inverse = np.full(original_count, -1, np.int64)
    inverse[np.asarray(permutation, np.int64)] = np.arange(len(permutation), dtype=np.int64)
    keep = (inverse[graph['ii']] >= 0) & (inverse[graph['jj']] >= 0)
    selected = np.flatnonzero(keep)
    ii, jj = inverse[graph['ii'][keep]], inverse[graph['jj'][keep]]
    kk = ii*patch_count + graph['kk'][keep] % patch_count
    result = dict(ii=np.ascontiguousarray(ii), jj=np.ascontiguousarray(jj), kk=np.ascontiguousarray(kk),
                  coords=np.ascontiguousarray(graph['coords'][:, keep]))
    for value in result.values():
        value.flags.writeable = False
    return result, selected


def round_order(index):
    require(type(index) is int and 0 <= index < 4, 'Four rounds only')
    return ('old', 'new') if index % 2 == 0 else ('new', 'old')


def validate_candidate_build(build, library_sha):
    require(build['schema'] == 'v36_immutable_frame_correlation_native_build'
            and build['status'] == 'compiled_and_synthetic_smoke_passed'
            and build['script_sha256'] == BUILD_SCRIPT_SHA
            and build['cleanup_errors'] == [] and build['sources_and_inputs_unchanged'] is True,
            'Successful unchanged candidate build with cleanup required')
    require(build['output']['sha256'] == library_sha
            and build['candidate_source_sha256']['dpvo_frame_corr.py'] == CANDIDATE_HELPER_SHA,
            'Candidate build/library/helper chain differs')
    require(build['smoke']['after_close']['closed']
            and build['smoke']['after_close']['current_context_bytes'] == 0, 'Build smoke cache not closed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-dir', 'motion-dir', 'parent-dir', 'old-library', 'old-build',
                 'old-replay-driver', 'candidate-dir', 'candidate-library', 'candidate-build', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('old-build-sha256', 'candidate-library-sha256', 'candidate-build-sha256'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args(argv)
    board_guard()  # No saved real arrays, library or outputs loaded before this.
    out = a.output_dir.resolve()
    require(out.is_relative_to(BASE), 'New isolated board output required')
    out.mkdir(exist_ok=False)
    prior = os.sched_getaffinity(0)
    pins, caches = {}, {}
    report = dict(schema='v36_saved_array_frame_correlation_AB_probe', status='failed_execution',
                  script_sha256=sha(__file__), actual_board_execution=True, candidate_accepted=False,
                  neural_forward_executed=False, BA_executed=False, ORB_LK_PnP_used=False,
                  realtime_validated=False, cases=[], cleanup_errors=[])
    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, 'Pinned file differs: '+str(path))
        pins[str(path)] = expected
        return path
    def save(filename, values):
        path = out/filename
        np.savez(path, **values)
        return dict(file=filename, bytes=path.stat().st_size, sha256=sha(path), arrays={k: meta(v) for k, v in values.items()})
    try:
        pin(__file__, report['script_sha256'])
        old_build = json.loads(pin(a.old_build, a.old_build_sha256).read_text())
        require(old_build['status'] == 'compiled_and_synthetic_smoke_passed'
                and old_build['output']['sha256'] == OLD_LIBRARY_SHA, 'Frozen v27 native build differs')
        pin(a.old_library, OLD_LIBRARY_SHA)
        for name, digest in old_build['source_sha256'].items():
            pin(a.parent_dir/name, digest)
        pin(a.parent_dir/'dpvo_corr.py', OLD_HELPER_SHA)
        pin(a.old_replay_driver, PARENT_REPLAY_SHA)
        new_build = json.loads(pin(a.candidate_build, a.candidate_build_sha256).read_text())
        validate_candidate_build(new_build, a.candidate_library_sha256)
        pin(a.candidate_library, a.candidate_library_sha256)
        for name, digest in new_build['candidate_source_sha256'].items():
            pin(a.candidate_dir/name, digest)
        for name, digest in new_build['parent_source_sha256'].items():
            pin(a.parent_dir/name, digest)
        candidate = module(a.candidate_dir/'dpvo_frame_corr.py', '_v36_frame_cache_AB')
        require(candidate.PARENT_SOURCE_PINS == new_build['parent_source_sha256'], 'Frozen parent compile identity differs')
        old_native = module(a.parent_dir/'dpvo_corr.py', '_v36_old_native_AB')
        old_driver = module(a.old_replay_driver, '_v36_old_stream_AB')
        captured = json.loads(pin(a.capture_dir/'capture.json', CAPTURE_SHA).read_text())
        motion = json.loads(pin(a.motion_dir/'motion.json', MOTION_SHA).read_text())
        require(captured['status'] == 'completed_encoder_cpu_probe_not_VO'
                and motion['status'] == 'completed_dynamic_neural_probe_not_VO'
                and motion['capture_sha256'] == CAPTURE_SHA, 'Frozen three-frame capture/motion chain differs')
        require(captured['sources_and_inputs_unchanged'] and motion['sources_and_inputs_unchanged'], 'Reference source changed')
        patch = load_artifact(a.motion_dir, motion['patches'], motion['patches']['arrays'], pin)
        require(patch['gmap'].shape == (1, 288, 128, 3, 3), 'Original 3x96 saved source patches required')
        records = []
        for i in range(3):
            row = next(r for r in captured['calls'] if r['frame_index'] == i)
            values = load_artifact(a.capture_dir, row['artifact'], row['arrays'], pin)
            require(values['fmap'].shape == (1, 1, 128, 116, 152), 'Original captured feature geometry differs')
            g, f = patch['gmap'][0, i*96:(i+1)*96], values['fmap'][0, 0]
            g.flags.writeable = f.flags.writeable = False
            records.append(dict(frame_id=motion['frame_ids'][i], gmap=g, fmap=f))
        scenarios = []
        for row in motion['scenarios']:
            graph = load_artifact(a.motion_dir, row['graph'], row['graph']['arrays'], pin)
            reference = load_artifact(a.motion_dir, row['calls'][0]['artifact'], row['calls'][0]['artifact']['arrays'], pin)['corr']
            require(graph['ii'].dtype == graph['jj'].dtype == graph['kk'].dtype == np.int64
                    and graph['coords'].dtype == np.float32 and np.array_equal(graph['ii'], graph['kk']//96), 'Saved original graph ownership differs')
            scenarios.append((graph, reference))
        first = next(s for s in scenarios if s[0]['kk'].size == 96)
        full = next(s for s in scenarios if s[0]['kk'].size == 864)
        require(np.all(first[0]['ii'] == 0) and np.all(first[0]['jj'] == 1), 'Original first-motion graph differs')
        reordered, select = remap_graph(full[0], [2, 0, 1], 96)
        deleted, subset = remap_graph(full[0], [0, 2], 96, remove=1)
        cases = [dict(name='first_motion_two_frames', frames=records[:2], graph=first[0], reference=first[1], prepared=2, resident=2),
                 dict(name='grow_to_three_frames', frames=records, graph=full[0], reference=full[1], prepared=3, resident=3),
                 dict(name='reorder_slots_same_three_frames', frames=[records[i] for i in (2, 0, 1)], graph=reordered,
                      reference=full[1][:, select], prepared=3, resident=3),
                 dict(name='delete_middle_slot_preserve_edge_order', frames=[records[0], records[2]], graph=deleted,
                      reference=full[1][:, subset], prepared=3, resident=2)]
        require({4, 5, 6, 7} <= prior, 'Four A76 CPUs required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)), native_threads=4,
                      OMP_WAIT_POLICY=os.environ.get('OMP_WAIT_POLICY'), GOMP_SPINCOUNT=os.environ.get('GOMP_SPINCOUNT'),
                      capture_sha256=CAPTURE_SHA, motion_sha256=MOTION_SHA,
                      candidate_build_sha256=a.candidate_build_sha256,
                      candidate_library_sha256=a.candidate_library_sha256,
                      candidate_helper_sha256=CANDIDATE_HELPER_SHA, parent_library_sha256=OLD_LIBRARY_SHA,
                      source_frames=[dict(frame_id=r['frame_id'], fmap=meta(r['fmap']), gmap=meta(r['gmap'])) for r in records])
        t = time.monotonic()
        caches['old'] = old_driver.CompactCorrelation(old_native.PreparedCorrelation, a.old_library, patch_count=96)
        old_constructor = time.monotonic()-t
        t = time.monotonic()
        caches['new'] = candidate.PerFrameCorrelation(a.candidate_library, library_sha256=a.candidate_library_sha256,
                                                     patch_count=96, threads=4, max_frames=64, max_bytes=512 << 20)
        report['constructor_seconds'] = dict(old=old_constructor, new=time.monotonic()-t)
        for case in cases:
            graph = {k: case['graph'][k] for k in ('ii', 'jj', 'kk', 'coords')}
            row = dict(name=case['name'], frame_ids_current=[f['frame_id'] for f in case['frames']],
                       graph=save(case['name']+'-graph.npz', graph),
                       original_CPU_reference=save(case['name']+'-original-corr.npz', dict(corr=case['reference'])),
                       expected_prepared_frames=case['prepared'], expected_resident_frames=case['resident'], calls=[])
            report['cases'].append(row)
            representatives, repeats = {}, {}
            for repetition in range(4):
                outputs = {}
                for mode in round_order(repetition):
                    t = time.monotonic()
                    corr = caches[mode](graph['coords'], ii=graph['ii'], jj=graph['jj'], kk=graph['kk'], frames=tuple(case['frames']))
                    end = time.monotonic()
                    metadata = meta(corr)  # Hash/save/statistics deliberately outside timer.
                    record = dict(mode=mode, round=repetition, timing_role='cold_graph' if repetition == 0 else 'warm_same_graph',
                                  begin_monotonic=t, end_monotonic=end, full_callback_seconds=end-t, array=metadata,
                                  versus_original_CPU=comparison(corr, case['reference']))
                    if mode not in repeats:
                        repeats[mode] = metadata
                        record['artifact'] = save(case['name']+'-'+mode+'.npz', dict(corr=corr))
                    record['repeat_bitwise_equal'] = metadata == repeats[mode]
                    require(record['repeat_bitwise_equal'], 'Repeated saved graph correlation changed')
                    metrics = caches[mode].metrics()
                    record['internal_callback_record'] = metrics['calls'][-1]
                    if mode == 'new':
                        record.update(prepared_frames=metrics['prepared_frames'], resident_frames=metrics['current_cached_frames'],
                                      resident_bytes=metrics['current_context_bytes'])
                        require(metrics['prepared_frames'] == case['prepared'] and metrics['current_cached_frames'] == case['resident'],
                                'Incremental preparation/residency contract failed')
                    row['calls'].append(record)
                    outputs[mode] = corr
                compare = comparison(outputs['new'], outputs['old'])
                for record in row['calls'][-2:]:
                    record['old_new_comparison'] = compare
                print(json.dumps(dict(case=row['name'], round=repetition,
                                      old_new_bitwise_equal=compare['bitwise_equal'], fixed_gate=compare['gate_passed'])), flush=True)
            row['warm_means_seconds'] = {mode: sum(c['full_callback_seconds'] for c in row['calls']
                                                  if c['mode'] == mode and c['round'] > 0)/3 for mode in ('old', 'new')}
        report['all_fixed_FP32_gates_passed'] = all(r['versus_original_CPU']['gate_passed'] and r['old_new_comparison']['gate_passed']
                                                       for c in report['cases'] for r in c['calls'])
        report['old_new_bitwise_equal'] = all(r['old_new_comparison']['bitwise_equal'] for c in report['cases'] for r in c['calls'])
        report['incremental_preparation_checks_passed'] = True
        report.update(status='completed_saved_array_correlation_AB_probe' if report['all_fixed_FP32_gates_passed'] else 'completed_numerical_rejected',
                      timing_scope='Four rounds per graph alternate old/new then new/old; full callback includes misses, native prep, current-order packing, query and wrapper bookkeeping. Library construction and final close separately measured. File hashes/stats/save, capture CNN, Update, BA and full VO excluded.',
                      limitations=['Three frozen actual captured feature maps only; no neural forward/BA/full replay.',
                                   'Original saved patches use M96; FAST M48 full replay remains separate.',
                                   'Short interleaved probe without thermal control; no sustained or end-to-end 20Hz claim.'])
    except Exception as exc:
        report.update(status='failed_execution', error=repr(exc), traceback=traceback.format_exc())
    finally:
        report['cache_lifecycle'] = {}
        for mode, cache in caches.items():
            try:
                before = cache.metrics()
                t = time.monotonic()
                cache.close()
                close_seconds = time.monotonic()-t
                after = cache.metrics()
                if mode == 'new':
                    require(after['closed'] and after['current_context_bytes'] == 0, 'New frame cache not released')
                else:
                    require(not after['current_context_open'] and all(m['closed'] and m['current_context_bytes'] == 0
                            for m in after['closed_contexts']), 'Old concatenated cache not released')
                report['cache_lifecycle'][mode] = dict(before_close=before, after_close=after, close_seconds=close_seconds)
            except Exception as exc:
                report['cleanup_errors'].append(repr(exc))
        try:
            after = {path: sha(path) for path in pins}
            report.update(source_pins=pins, source_hashes_after=after, sources_and_inputs_unchanged=after == pins)
            require(after == pins, 'Pinned source/saved array changed')
        except Exception as exc:
            report['cleanup_errors'].append(repr(exc))
        try:
            os.sched_setaffinity(0, prior)
            report['affinity_restored'] = os.sched_getaffinity(0) == prior
            require(report['affinity_restored'], 'Affinity restoration failed')
        except Exception as exc:
            report['cleanup_errors'].append(repr(exc))
        if report['cleanup_errors']:
            report['status'] = 'failed_execution'
        report['peak_RSS_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (out/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'all_fixed_FP32_gates_passed', 'old_new_bitwise_equal', 'error', 'cleanup_errors')}), flush=True)
    return 0 if report['status'] == 'completed_saved_array_correlation_AB_probe' else 1


if __name__ == '__main__':
    raise SystemExit(main())
