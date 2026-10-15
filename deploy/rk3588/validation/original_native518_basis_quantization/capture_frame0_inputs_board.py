#!/usr/bin/env python3
"""Capture two real original native518 frame0 inputs, before any aggregator.

Batch indices 0/1/2 partition four calibration and two heldout pairs. Only the
four frozen original DINO groups execute on NPU, then the original final norm
and pair-role token concatenation run. This is data capture, not performance
or quality acceptance. The caller owns the bounded 1GHz/75C/180s controller.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'; EXP = BASE / 'experiments'; S100 = EXP / 'vggt-v100-source-r1'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
CONFIG_SHA = '7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
PAIRS_SHA = '18288334f5309b59c6eacdf8a95c8141fae6a453cf05305d8d7d8f88b13d81f2'
LINEAGE_SHA = 'f46edc183442c69af3dd314c9a1e096dca42f0590881d51c6fd8c7390df9617e'
RGB_SHA = '532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3'
PREFIX_SHA = 'cbb80f6559ed8856f05382899b7e1025b2637f08108ae6c35abb9d6954a78b07'
OWNED_SHA = '0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
HELPER_SHA = '56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
PAIRS = [[0, 8], [16, 24], [48, 56], [64, 72], [96, 104], [112, 120]]
FRAME, DINO = (2, 1041, 1024), (1, 1041, 1024)
COUNTERS = ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')
STABLE = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_gid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
SUCCESS = 'completed_original_native518_frame0_input_subbatch_capture'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, row):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-index', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--pair-pins', type=Path, required=True)
    p.add_argument('--dino-lineage', type=Path, required=True)
    p.add_argument('--config', type=Path, default=S100 / 'full-native-sequential-config-r1.json')
    p.add_argument('--repo', type=Path, default=SOURCE)
    p.add_argument('--frozen-source-dir', type=Path, default=S100)
    p.add_argument('--runtime-helper', type=Path, default=EXP / 'vggt-v99-source-r1/run_native_block_r1.py')
    p.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--timeout-seconds', type=float, default=175.)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'Board-only capture; no host NN/SDK execution')
    require(0 < a.timeout_seconds <= 175., 'External 180-second controller required')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); old_env, old_affinity, old_path = dict(os.environ), set(os.sched_getaffinity(0)), sys.path[:]
    source = Path(__file__).resolve(); watched = {source: sha(source)}
    handlers, owners, prefix = {}, [], None
    report = dict(schema='v128_original_native518_frame0_input_capture', status='running', script_sha256=watched[source],
        batch_index=a.batch_index, checkpoint=CHECKPOINT, image_size=518, pixel_shape=[2, 3, 392, 518], logical_shape=list(FRAME),
        original_model_only=True, actual_execution_on_board_only=True, component_data_capture_only=True,
        original_DINO_layers_per_image=24, original_images_per_pair=2, original_aggregator_blocks_executed=0,
        camera_or_DPT_executed=False, CPU_DINO_block_forwards_executed=0, CPU_original_pair_reference_executed=False,
        performance_or_quality_acceptance=False, candidate_accepted=False, system_or_frequency_writes=False,
        boundary='Actual original NPU DINO24 -> original final norm -> original camera/register roles -> BEFORE frame0',
        reuse_previous_tail=False, NPU_core_mask=7, NPU_init_flags=0, expected_NPU_hz=1000000000,
        maximum_live_NPU_contexts=1, expected_SDK_totals=dict(zip(COUNTERS, (4, 4, 16, 16, 16))),
        planned_all_pairs=PAIRS, planned_calibration_pair_count=4, planned_heldout_pair_count=2,
        records=[], SDK_calls=[], platform_observations=[], cleanup_errors=[], storage_limit_bytes=20000000)

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        atomic(out / 'result.json', report); print(stage, flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(len(digest) == 64 and sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned artifact differs: ' + str(path))
        watched[path] = digest; return path

    def load_source(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

    def observe(stage):
        require(time.monotonic() - started < a.timeout_seconds, 'Capture deadline reached')
        sample = H.sample()
        require(sample['npu_hz'] == 1000000000 and sample['temperatures_c'] and max(sample['temperatures_c'].values()) < 75., 'External frequency/thermal guard differs')
        report['platform_observations'].append(dict(stage=stage, **sample))

    def meta(value):
        require(value.dtype == np.float32 and value.flags.c_contiguous and np.isfinite(value).all(), 'Owned finite contiguous FP32 required')
        return dict(shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes, finite=True, raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())

    def seal():
        return [(name, id(v), v.data_ptr(), v._version, str(v.dtype), str(v.device)) for name, v in prefix.dino.named_parameters()] + \
            [(name, id(v), v.data_ptr(), v._version, str(v.dtype), str(v.device)) for name, v in (('camera', prefix.camera), ('register', prefix.register))]

    def checkpoint_stats(stat):
        return {key: getattr(stat, key) for key in STABLE + ('st_atime_ns', 'st_blocks', 'st_blksize') if hasattr(stat, key)}

    def verify_checkpoint(final=False):
        initial, current = checkpoint_stats(prefix.checkpoint_stat), checkpoint_stats(prefix.checkpoint.stat())
        require(all(initial[k] == current[k] for k in STABLE), 'Original checkpoint stable metadata changed')
        report.setdefault('checkpoint_stat_observations', []).append(dict(initial=initial, current=current,
            differing_fields=[k for k in initial if initial[k] != current[k]], stable_fields_equal=True))
        if final:
            digest = sha(prefix.checkpoint); after = checkpoint_stats(prefix.checkpoint.stat())
            report.update(checkpoint_final_sha256=digest, checkpoint_stat_after_final_SHA=after)
            require(digest == CHECKPOINT['sha256'] and all(initial[k] == after[k] for k in STABLE), 'Original checkpoint final SHA/identity differs')
            report['checkpoint_final_SHA_verified'] = True

    progress('preflight')
    try:
        def interrupted(number, frame): raise InterruptedError('Controller termination signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require({4, 5, 6, 7} <= old_affinity, 'A76 CPUs4..7 required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        config = json.loads(pin(a.config, CONFIG_SHA).read_text())
        pairs = json.loads(pin(a.pair_pins, PAIRS_SHA).read_text())
        lineage = json.loads(pin(a.dino_lineage, LINEAGE_SHA).read_text())
        require(pairs['sorted_rgb_count'] == 798 and [v['indices'] for v in pairs['pairs']] == PAIRS and
                [v['split'] for v in pairs['pairs']] == ['calibration'] * 4 + ['heldout'] * 2, 'Six predeclared disjoint pairs required')
        image_rows = [v for pair in pairs['pairs'] for v in pair['images']]
        require(len({v['index'] for v in image_rows}) == len({v['file'] for v in image_rows}) == len({v['sha256'] for v in image_rows}) == 12, 'PNG/source partitions overlap')
        root = Path(image_rows[0]['file']).parent.parent
        rgb_file = pin(root / 'rgb.txt', RGB_SHA)
        rgb = [line.split() for line in rgb_file.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        require(len(rgb) == 798, 'Original TUM RGB index differs')
        for image in image_rows:
            require((root / rgb[image['index']][1]).resolve() == Path(image['file']).resolve(), 'Declared index/PNG mapping differs')
            pin(image['file'], image['sha256'], image['bytes'])
        selected = pairs['pairs'][2 * a.batch_index:2 * a.batch_index + 2]
        require(lineage['config_sha256'] == CONFIG_SHA and len(lineage['rows']) == len(config['dino']) == 4 and lineage['all_four_DINO_models_present_and_SHA_match'], 'Four original DINO models required')
        P = load_source(a.frozen_source_dir / 'native_original_prefix.py', PREFIX_SHA, '_v128_original_prefix')
        O = load_source(a.frozen_source_dir / 'native_heads_runtime.py', OWNED_SHA, '_v128_frozen_owned_IO')
        H = load_source(a.runtime_helper, HELPER_SHA, '_v128_platform_IO')
        pin(a.library, O.SDK_SHA); pin(a.repo / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', O.ABI_SHA)
        models, exports = [], []
        for group, row in enumerate(lineage['rows']):
            require(row['group'] == group and row['compile'] == config['dino'][group], 'Original ordered model identity differs')
            compiled = json.loads(pin(row['compile']['file'], row['compile']['sha256']).read_text())
            export = json.loads(pin(row['export']['file'], row['export']['sha256']).read_text())
            require(compiled['schema'] == 'original_native518_single_block_FP16_compile' and compiled['status'] == 'compiled_not_quality_accepted' and
                    compiled['role'] == 'dino' and compiled['logical_shape'] == list(DINO) and compiled['sdk_released'] and not compiled['cleanup_errors'] and
                    not compiled['build_options']['do_quantization'] and not compiled['configuration']['model_pruning'], 'Original clean nonquantizing DINO compile required')
            require(export['group_index'] == group and export['indices'] == list(range(6 * group, 6 * group + 6)) and
                    export['checkpoint'] == CHECKPOINT and export['graph']['sha256'] == compiled['onnx']['sha256'] and
                    export['proof']['original_parameter_count'] == 84 and export['proof']['namespace_inverse_protobuf_checks_passed'] and
                    export['source_and_inputs_unchanged'], 'Original six-layer graph/84-tensor proof differs')
            require(Path(row['model']['file']).resolve() == (Path(row['compile']['file']).parent / compiled['rknn']['file']).resolve() and
                    all(row['model'][key] == compiled['rknn'][key] for key in ('sha256', 'bytes')), 'Compiled model binding differs')
            pin(row['model']['file'], row['model']['sha256'], row['model']['bytes']); models.append(row['model']); exports.append(export)
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        observe('prefix_initialization'); prefix = P.OriginalPrefix(repo=a.repo)
        require(prefix.native['checkpoint'] == prefix.reference['checkpoint'] == CHECKPOINT, 'Original prefix checkpoint differs')
        for path, digest in prefix.pins.items():
            if Path(path) != prefix.checkpoint: watched[Path(path).resolve()] = digest
        for group, export in enumerate(exports):
            for index in export['indices']:
                bindings = export['proof']['parameter_bindings'][str(index)]
                require(set(bindings) == set(prefix.reference['parameters'][str(index)]) and
                        all(v['original_raw_sha256'] == prefix.reference['parameters'][str(index)][name]['raw_sha256'] for name, v in bindings.items()), 'Original DINO learned tensor identity differs')
        original_seal = seal()
        role_tokens = {name: meta(v.detach().contiguous().numpy()) for name, v in (('camera_token', prefix.camera), ('register_token', prefix.register))}
        report.update(versions=dict(torch=torch.__version__, numpy=np.__version__, torch_threads=1, torch_interop_threads=1),
            prefix_source_sha256=PREFIX_SHA, original_prefix_parameters=prefix.reference['token_preparation_and_final_norm_parameters'],
            original_role_token_parameters=role_tokens, original_DINO_parameter_count=336, upstream_models=copy.deepcopy(lineage['rows']),
            checkpoint_verified_before_deserialization=True, full_six_pair_PNG_partitions_disjoint=True,
            pinned_pair_manifest_sha256=PAIRS_SHA, pinned_dino_lineage_sha256=LINEAGE_SHA,
            source_hashes=prefix.native['source_hashes'])
        encoded = []
        for pair in selected:
            observe('prepare_' + pair['id'])
            with torch.inference_mode():
                pixels = prefix.preprocess([v['file'] for v in pair['images']], image_size=518, mode='crop')
                require(tuple(pixels.shape) == (2, 3, 392, 518), 'Native original image geometry differs')
                pixels_meta = meta(np.ascontiguousarray(pixels.numpy()))
                normalized = ((pixels.unsqueeze(0) - prefix.mean) / prefix.std).view(2, 3, 392, 518)
                frames = [np.array(prefix.dino.prepare_tokens_with_masks(normalized[j:j+1].contiguous(), masks=None).numpy(), copy=True, order='C') for j in range(2)]
            require(all(v.shape == DINO for v in frames), 'All native DINO tokens required')
            encoded.append(frames)
            report['records'].append(dict(id=pair['id'], split=pair['split'], source_pair_indices=pair['indices'], source_images=pair['images'],
                preprocessed_pixels=pixels_meta, prepared_DINO_inputs=[meta(v) for v in frames], DINO_group_outputs=[]))
            del pixels, normalized, frames
        progress('inputs_prepared')
        for group, model in enumerate(models):
            observe('DINO_group_' + str(group)); owner = None
            try:
                try:
                    owner = O.OwnedContext(model['file'], model['sha256'], input_shapes=[list(DINO)], output_shapes=[list(DINO)],
                        input_names=['input'], output_names=['output'], core_mask=7, flags=0, source_root=a.repo, library=a.library,
                        runtime_helper=a.runtime_helper, timeout_ms=30000)
                except BaseException as error:
                    owner = getattr(error, 'owned_context', None)
                    if owner is not None: owners.append(owner)
                    raise
                owners.append(owner)
                for pair_index, frames in enumerate(encoded):
                    for frame in range(2):
                        observe('DINO%d_%s_frame%d' % (group, selected[pair_index]['id'], frame))
                        frames[frame] = owner.run([frames[frame]])[0]
                        require(frames[frame].shape == DINO, 'Actual original DINO group output shape differs')
                        report['SDK_calls'].append(dict(group=group, pair_id=selected[pair_index]['id'], frame=frame, output=meta(frames[frame])))
                    report['records'][pair_index]['DINO_group_outputs'].append(dict(group=group, indices=list(range(6*group, 6*group+6)), frames=[meta(v) for v in frames]))
            finally:
                if owner is not None: owner.close()
            progress('DINO_group_' + str(group) + '_closed')
        require(all(owner.closed for owner in owners), 'All NPU contexts must close before final norm/concat')
        verify_checkpoint()
        for pair_index, frames in enumerate(encoded):
            observe('finish_' + selected[pair_index]['id'])
            value = prefix.finish_dino(frames)
            require(value.shape == FRAME and value.flags.owndata, 'Owned original frame0 input required')
            row = meta(value); file = out / (selected[pair_index]['id'] + '-frame0-input.npy')
            require(sum(f.stat().st_size for f in out.iterdir() if f.is_file()) + value.nbytes + 65536 < report['storage_limit_bytes'], 'Subbatch storage bound reached')
            with file.open('xb') as stream: np.save(stream, value, allow_pickle=False)
            row.update(file=file.name, bytes=file.stat().st_size, sha256=sha(file)); watched[file] = row['sha256']
            report['records'][pair_index].update(input=row, original_final_norm_and_role_concat_executed=True,
                input_is_before_original_frame0=True, actual_four_DINO_groups_executed=True)
            del value; progress(selected[pair_index]['id'] + '_saved')
        require(seal() == original_seal and role_tokens == {name: meta(v.detach().contiguous().numpy()) for name, v in (('camera_token', prefix.camera), ('register_token', prefix.register))}, 'Original learned state changed')
        require(len({row['input']['raw_sha256'] for row in report['records']}) == 2, 'Subbatch captured inputs repeat')
        report.update(status=SUCCESS, all16_SDK_calls_completed=len(report['SDK_calls']) == 16, model_state_unchanged=True,
            all_inputs_before_frame0=True, two_pair_subbatch_complete=True, original_final_norm_calls=4,
            saved_array_count=2, saved_array_bytes=sum(v['input']['bytes'] for v in report['records']))
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        for owner in reversed(owners):
            if not owner.closed:
                try: owner.close()
                except BaseException as error: report['cleanup_errors'].append('SDK close: ' + str(error))
        metrics = [owner.metrics() for owner in owners]
        report.update(owner_metrics=metrics, SDK_totals={key: sum(m[key] for m in metrics) for key in COUNTERS},
            all_NPU_contexts_closed=all(m['closed'] and not m['live_contexts'] and not m['live_output_buffers'] and m['resource_counts_balanced'] for m in metrics))
        if prefix is not None:
            try: verify_checkpoint(final=True)
            except BaseException as error: report['cleanup_errors'].append('Checkpoint verification: ' + str(error))
        for label, restore in (('affinity', lambda: os.sched_setaffinity(0, old_affinity)),):
            try: restore()
            except BaseException as error: report['cleanup_errors'].append(label + ': ' + str(error))
        os.environ.clear(); os.environ.update(old_env); sys.path[:] = old_path
        for number, handler in handlers.items(): signal.signal(number, handler)
        report.update(affinity_restored=set(os.sched_getaffinity(0)) == old_affinity, environment_restored=dict(os.environ) == old_env,
            sys_path_restored=sys.path == old_path, source_and_inputs_unchanged=all(p.is_file() and sha(p) == digest for p, digest in watched.items()),
            pinned_files={str(p): digest for p, digest in watched.items()})
        if report['status'] == SUCCESS and (report['SDK_totals'] != report['expected_SDK_totals'] or not report['all16_SDK_calls_completed'] or not report.get('checkpoint_final_SHA_verified')):
            report['cleanup_errors'].append('Completed subbatch ledger/checkpoint verification incomplete')
        if report['cleanup_errors'] or not report['all_NPU_contexts_closed'] or any(m['cleanup_errors'] for m in metrics) or not all(report[k] for k in ('affinity_restored', 'environment_restored', 'sys_path_restored', 'source_and_inputs_unchanged')):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__':
    raise SystemExit(main())
