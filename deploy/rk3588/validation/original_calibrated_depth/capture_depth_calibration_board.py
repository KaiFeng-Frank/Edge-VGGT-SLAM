#!/usr/bin/env python3
"""Capture original DPT calibration from four saved original FP32 block inputs.

Replays only the original frame/global blocks at 4/11/17/23 and the untouched
depth head. No complete VGGT forward, cache, NPU, export or Solver. Capture
timing includes checkpoint/source checks and private array I/O, not throughput.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
TAIL_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
DEPTH_CAPTURE_SHA = 'fa33dc6a05c2e286ca9c9a17c0f62df21ee2b19da0f3ea693fcff3718bc99033'
DEPTH_MANIFEST_SHA = '9a6e5435df543b8484a4fc50da4b065995f32153233548829234dd149fbc7135'
CONTRACT_SHA = 'efdad006ed594636041e5fdeb315b81ec8154d89d74887fabcf26f77e12bb9ba'
CHECKPOINT = dict(bytes=5026874952,
    sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
LAYERS = (4, 11, 17, 23)
INPUT_NAMES = tuple('tokens_' + str(i) for i in LAYERS)
PAIRS = ((0, 48), (96, 144), (192, 240), (288, 336),
         (384, 432), (480, 528), (576, 624), (672, 720))
RUNTIME_SOURCES = {
    'vggt_slam/runtime.py': 'f36bb5b87fbdbf03984ea3e1466e6ee27865a540ae3e5abd4ee07a45538e23f5',
    'vggt_slam/checkpoint_identity.py': '9fc0b927f6a68ac6ca55194616104febb1c488370593d4be60e8dba5bc336461'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def file_info(path, root):
    return dict(file=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def array_info(value, shape=None):
    import numpy as np
    value = np.ascontiguousarray(value)
    if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError('Original nonempty finite FP32 tensor required')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError('Original tensor shape differs')
    return dict(shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes,
        finite=True, raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def save(root, name, value, shape):
    import numpy as np
    value = np.ascontiguousarray(value); info = array_info(value, shape)
    path = root / name
    with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)
    return dict(**file_info(path, root), **info)


def tensor_state(model):
    return tuple((name, id(t), t._version, t.dtype, t.device, tuple(t.shape), tuple(t.stride()),
                  t.data_ptr(), t.requires_grad) for name, t in
                 list(model.named_parameters()) + list(model.named_buffers()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--checkpoint', type=Path, default=BASE / 'weights/model.pt')
    p.add_argument('--calibration-dir', type=Path, default=BASE / 'experiments/vggt-v56-calibration-r1')
    p.add_argument('--original-tail-dir', type=Path, default=BASE / 'experiments/vggt-v53-capture-r1')
    p.add_argument('--depth-manifest', type=Path, required=True)
    p.add_argument('--original-depth-capture', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    board = next((x.read_bytes() for x in (Path('/proc/device-tree/compatible'),
        Path('/sys/firmware/devicetree/base/compatible')) if x.is_file()), b'')
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or b'rk3588' not in board:
        p.error('Actual original-model capture is Linux RK3588/S board-only')
    repo = args.repo.resolve(strict=True); checkpoint = args.checkpoint.resolve(strict=True)
    calibration = args.calibration_dir.resolve(strict=True); tail_dir = args.original_tail_dir.resolve(strict=True)
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0); old_path = sys.path[:]
    env_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {name: os.environ.get(name) for name in env_names}
    hooks, pins = [], {}; model = state = checkpoint_stat = None
    datasets = []
    report = dict(schema='v85_original_VGGT_depth_PTQ_calibration_capture', status='failed',
        script_sha256=sha(__file__), checkpoint=CHECKPOINT, source_hashes={}, runtime_source_hashes=RUNTIME_SOURCES,
        original_model_only=True, actual_inference_on_board_only=True, NPU_or_cache_installed=False,
        full_VGGT_forward_executed=False, DINO_or_camera_executed=False,
        ORT_or_SDK_or_compiler_or_Solver_executed=False, samples=[], parameters={}, cleanup_errors=[],
        original_depth_head_calls=0, original_raw_logits_calls=0, input_tensor_count=0,
        selected_blocks_forward_count={role + str(layer): 0 for layer in LAYERS for role in ('frame', 'global')},
        all_depth_and_conf_bitwise_equal_to_original_capture=False,
        candidate_accepted=False, full_model_quality_or_realtime_validated=False,
        capture_time_is_inference_performance=False)
    started = time.perf_counter()

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        if (size is not None and path.stat().st_size != size) or sha(path) != digest:
            raise ValueError('Frozen artifact/source changed: ' + str(path))
        pins[path] = digest
        return path

    def load_array(directory, item):
        import numpy as np
        path = (directory / item['file']).resolve(strict=True)
        if not path.is_relative_to(directory): raise ValueError('Array path escaped original capture directory')
        value = np.load(pin(path, item['sha256'], item['bytes']), allow_pickle=False)
        if (list(value.shape) != item['shape'] or str(value.dtype) != item['dtype']
                or value.nbytes != item['tensor_bytes']
                or hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest() != item['raw_sha256']
                or not value.size or not np.isfinite(value).all()):
            raise ValueError('Saved original tensor identity differs')
        return np.ascontiguousarray(value)

    try:
        if not {4, 5, 6, 7} <= affinity: raise ValueError('Original CPU cores 4..7 unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7}); os.environ.update({name: '4' for name in env_names})
        calib_path = pin(calibration / 'capture.json', CALIBRATION_SHA)
        tail_path = pin(tail_dir / 'capture.json', TAIL_SHA)
        depth_path = pin(args.original_depth_capture, DEPTH_CAPTURE_SHA)
        manifest_path = pin(args.depth_manifest, DEPTH_MANIFEST_SHA)
        cap = json.loads(calib_path.read_text()); tail = json.loads(tail_path.read_text())
        depth = json.loads(depth_path.read_text()); manifest = json.loads(manifest_path.read_text())
        if (cap['status'] != 'completed_original_FP32_calibration_capture' or cap['checkpoint'] != CHECKPOINT
                or cap['cleanup_errors'] or cap['full_forward_completed'] != 8 or cap['NPU_or_cache_installed']
                or not cap['model_state_unchanged'] or not cap['source_and_images_unchanged']
                or tail['status'] != 'completed_original_FP32_full_forward_and_48_block_capture'
                or tail['cleanup_errors'] or tail['checkpoint']['sha256'] != CHECKPOINT['sha256']
                or depth['status'] != 'completed_original_FP32_depth_capture' or depth['cleanup_errors']
                or depth['checkpoint'] != CHECKPOINT or depth['full_model_forward_completed'] is not True
                or manifest['schema'] != 'v54_original_VGGT_depth_logits_RKNN_manifest'
                or manifest['checkpoint'] != CHECKPOINT or manifest['capture_sha256'] != DEPTH_CAPTURE_SHA
                or manifest['all_62_original_parameter_values_preserved_in_graph'] is not True
                or manifest['original_inplace_relu_residual_skip_preserved'] is not True
                or manifest['all_five_original_align_corners_resizes_preserved'] is not True
                or manifest['parameters'] != depth['artifacts']['state']['tensors']):
            raise ValueError('Completed fixed original capture/depth lineage required')
        if ([tuple(s['indices']) for s in cap['samples']] != list(PAIRS)
                or [s['sample_id'] for s in cap['samples']] != ['calib%02d' % i for i in range(8)]
                or set(sum(PAIRS, ())) & set(cap['evaluation_indices_excluded'])):
            raise ValueError('Eight disjoint original calibration pairs required')
        source_hashes = {name: digest for name, digest in cap['source_hashes'].items()
                         if name.startswith('third_party/vggt/vggt/')}
        for original in (tail, depth, manifest):
            for name, digest in original['source_hashes'].items():
                if name.startswith('third_party/vggt/vggt/') and source_hashes.get(name) != digest:
                    raise ValueError('Original source lineage differs: ' + name)
        for name, digest in dict(source_hashes, **RUNTIME_SOURCES).items(): pin(repo / name, digest)
        contract_path = pin(repo / 'deploy/rk3588/validation/original_depth/depth_contract.py', CONTRACT_SHA)
        report.update(calibration_capture=file_info(calib_path, calibration),
            original_tail_capture=file_info(tail_path, tail_dir), original_depth_capture=file_info(depth_path, depth_path.parent),
            original_depth_manifest_sha256=DEPTH_MANIFEST_SHA,
            depth_manifest=file_info(manifest_path, manifest_path.parent), contract_source_sha256=CONTRACT_SHA,
            evaluation_indices_excluded=cap['evaluation_indices_excluded'], image_shape=[2, 3, 84, 112],
            architecture=manifest['architecture'], positions=manifest['positions'], source_hashes=source_hashes,
            original_inplace_relu_residual_skip_preserved=True, all_five_original_align_corners_resizes_preserved=True,
            original_five_position_additions_preserved=True, original_activation_and_chunking_preserved=True)
        checkpoint_stat = checkpoint.stat()
        if checkpoint_stat.st_size != CHECKPOINT['bytes']: raise ValueError('Original checkpoint size differs')
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.heads import utils
        spec = importlib.util.spec_from_file_location('_v85_original_depth_contract', contract_path)
        C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        report['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, threads=4, interop_threads=1,
                                cpu_affinity=sorted(os.sched_getaffinity(0)))
        tick = time.perf_counter()
        model = load_model(checkpoint, device='cpu', image_size=112, prune_unused_heads=True, offline=True)
        report.update(model_load_seconds=time.perf_counter() - tick, checkpoint_verified_before_deserialization=True)
        agg = model.aggregator; head = model.depth_head
        if (len(agg.frame_blocks) != 24 or len(agg.global_blocks) != 24 or agg.aa_block_size != 1
                or agg.aa_order != ['frame', 'global'] or agg.patch_start_idx != 5 or head is None
                or model.depth_frames_chunk_size != 1 or C.head_contract(head) != manifest['architecture']
                or head.scratch.output_conv2[1].inplace is not True
                or any(getattr(getattr(head.scratch, 'refinenet' + str(i)), 'resConfUnit' + str(u)).activation.inplace is not True
                       for i in range(1, 5) for u in ((1, 2) if i < 4 else (2,)))
                or any(m.training for m in model.modules())
                or any(t.device.type != 'cpu' or t.dtype != torch.float32 for t in model.parameters())):
            raise ValueError('Unchanged original eval CPU FP32 model and depth configuration required')
        state = tensor_state(model)
        metadata = manifest['parameters']
        if len(metadata) != 62 or set(dict(head.named_parameters())) != set(metadata):
            raise ValueError('All original 62 depth parameter tensors required')
        for name, tensor in head.named_parameters():
            info = array_info(tensor.detach().numpy(), metadata[name]['shape'])
            if {k: info[k] for k in metadata[name]} != metadata[name]:
                raise ValueError('Original depth parameter differs: ' + name)
            report['parameters'][name] = info
        report['original_depth_parameters_verified'] = 62
        selected = {}
        for layer in LAYERS:
            for role in ('frame', 'global'):
                block = getattr(agg, role + '_blocks')[layer]
                row = next(r for r in tail['blocks'] if r['role'] == role and r['index'] == layer)
                eps = {name: obj.eps for name, obj in (('norm1', block.norm1), ('norm2', block.norm2),
                       ('attn.q_norm', block.attn.q_norm), ('attn.k_norm', block.attn.k_norm))}
                if (len(row['parameters']) != 18 or set(dict(block.named_parameters())) != set(row['parameters'])
                        or block.attn.num_heads != row['heads'] or eps != row['epsilons']
                        or block.mlp.act.approximate != row['gelu_approximate']
                        or block.attn.rope.base_frequency != row['rope_frequency']
                        or block.attn.rope.scaling_factor != row['rope_scaling_factor']):
                    raise ValueError('Original selected block configuration differs')
                for name, tensor in block.named_parameters():
                    info = array_info(tensor.detach().numpy(), row['parameters'][name]['shape'])
                    if {k: info[k] for k in row['parameters'][name]} != row['parameters'][name]:
                        raise ValueError('Original selected block parameter differs: ' + role + str(layer) + '.' + name)
                selected[role + str(layer)] = row
        report['original_selected_aggregator_parameters_verified'] = 144
        frame_pos = load_array(tail_dir, selected['frame4']['positions'])
        global_pos = load_array(tail_dir, selected['global4']['positions'])
        if (frame_pos.shape != (2, 53, 2) or global_pos.shape != (1, 106, 2)
                or frame_pos.dtype != np.int64 or not np.array_equal(frame_pos.reshape(1, 106, 2), global_pos)
                or any(row['positions'] != selected[role + '4']['positions'] for role in ('frame', 'global')
                       for key, row in selected.items() if key.startswith(role))):
            raise ValueError('Original frame/global position identity differs')
        report['selected_original_blocks'] = [{k: row[k] for k in ('role', 'index', 'heads', 'epsilons',
            'gelu_approximate', 'rope_frequency', 'rope_scaling_factor', 'parameters', 'positions')} for row in selected.values()]
        norm_inputs, logits = [], []
        def norm_before(module, inputs):
            value = np.array(inputs[0].detach().numpy(), copy=True, order='C')
            array_info(value, (1, 48, 2048))
            if len(norm_inputs) >= 8: raise ValueError('Exactly four original norm inputs per frame required')
            norm_inputs.append(value)
        def logits_after(module, inputs, value):
            value = np.array(value.detach().numpy(), copy=True, order='C')
            array_info(value, (1, 2, 84, 112))
            if len(logits) >= 2: raise ValueError('Exactly two original raw-logit chunks required')
            logits.append(value)
        hooks.append(head.norm.register_forward_pre_hook(norm_before))
        hooks.append(head.scratch.output_conv2.register_forward_hook(logits_after))
        groups = {g['index']: g for g in cap['groups']['aggregator']}
        image_hashes = set()
        for n, original in enumerate(cap['samples']):
            for image in original['images']:
                pin(image['file'], image['sha256'], image['bytes'])
                if image['sha256'] in image_hashes: raise ValueError('Original calibration image duplicated')
                image_hashes.add(image['sha256'])
            images = load_and_preprocess_images([item['file'] for item in original['images']], image_size=112, mode='crop')
            image_info = array_info(images.numpy(), (2, 3, 84, 112))
            if any(image_info[k] != original['preprocessed_input'][k] for k in image_info):
                raise ValueError('Original RGB preprocessing does not reproduce saved input')
            (root / original['sample_id']).mkdir()
            row = dict(sample_id=original['sample_id'], split='calibration', source_pair_indices=original['indices'],
                images=original['images'], preprocessed_input=image_info, frames=[], original_frame_inputs=[],
                original_full_forward_references={key: original['canonical_outputs'][key] for key in ('depth', 'depth_conf')})
            tokens = [None] * 24; norm_inputs.clear(); logits.clear(); tick = time.perf_counter()
            with torch.inference_mode():
                for layer in LAYERS:
                    item = groups[layer]['samples'][n]
                    if (len(groups[layer]['samples']) != 8 or item['sample_id'] != original['sample_id']
                            or item['source_pair_indices'] != original['indices']):
                        raise ValueError('Selected original activation sample order differs')
                    x = load_array(calibration, item['artifact']); array_info(x, (2, 53, 1024))
                    frame, fi, local = agg._process_frame_attention(torch.from_numpy(x), 1, 2, 53, 1024, layer, pos=torch.from_numpy(frame_pos))
                    report['selected_blocks_forward_count']['frame' + str(layer)] += 1
                    glob, gi, global_intermediates = agg._process_global_attention(frame, 1, 2, 53, 1024, layer, pos=torch.from_numpy(global_pos))
                    report['selected_blocks_forward_count']['global' + str(layer)] += 1
                    if fi != layer + 1 or gi != layer + 1 or len(local) != 1 or len(global_intermediates) != 1:
                        raise ValueError('Original selected frame/global call order differs')
                    tokens[layer] = torch.cat([local[0], global_intermediates[0]], dim=-1)
                    array_info(tokens[layer].numpy(), (1, 2, 53, 2048))
                    if array_info(x)['raw_sha256'] != item['artifact']['raw_sha256']:
                        raise ValueError('Saved original frame input mutated')
                    row['original_frame_inputs'].append(dict(index=layer, artifact=item['artifact']))
                depth_output, conf_output = head(tokens, images=images.unsqueeze(0), patch_start_idx=5, frames_chunk_size=1)
                report['original_depth_head_calls'] += 1
            row['selected_blocks_and_depth_seconds_including_hooks'] = time.perf_counter() - tick
            if len(norm_inputs) != 8 or len(logits) != 2:
                raise ValueError('Both original DPT chunks/four norm inputs must complete')
            for frame in range(2):
                frame_row = dict(index=frame, inputs=[])
                for j, layer in enumerate(LAYERS):
                    value = norm_inputs[frame * 4 + j]
                    expected = np.ascontiguousarray(tokens[layer].numpy()[:, frame:frame + 1, 5:].reshape(1, 48, 2048))
                    if value.tobytes() != expected.tobytes(): raise ValueError('Original norm input order/view differs')
                    artifact = save(root, '%s/frame%d-%s.npy' % (original['sample_id'], frame, INPUT_NAMES[j]), value, (1, 48, 2048))
                    frame_row['inputs'].append(dict(name=INPUT_NAMES[j], artifact=artifact))
                frame_row['raw_logits'] = save(root, '%s/frame%d-logits.npy' % (original['sample_id'], frame), logits[frame], (1, 2, 84, 112))
                row['frames'].append(frame_row)
                datasets.append(' '.join(str(root / item['artifact']['file']) for item in frame_row['inputs']))
            for key, value, shape in (('depth', depth_output, (1, 2, 84, 112, 1)), ('depth_conf', conf_output, (1, 2, 84, 112))):
                info = array_info(value.numpy(), shape)
                equal = info['raw_sha256'] == original['canonical_outputs'][key]['raw_sha256']
                row[key + '_bitwise_equal_to_original_capture'] = equal
                row[key] = save(root, '%s/%s.npy' % (original['sample_id'], key), value.numpy()[0], shape[1:])
                if not equal: raise ValueError('Original DPT output does not reproduce full FP32 capture: ' + key)
            if array_info(images.numpy())['raw_sha256'] != image_info['raw_sha256']:
                raise ValueError('Original preprocessed images mutated')
            report['samples'].append(row); report['original_raw_logits_calls'] += 2; report['input_tensor_count'] += 8
            print(json.dumps(dict(completed=n + 1, total=8, sample_id=original['sample_id'])), flush=True)
        # Confirm the five geometry constants from the exact original utilities,
        # after every original DPT call; no NumPy approximation or replacement.
        for name, item in manifest['positions'].items():
            _, channels, height, width = item['shape']
            with torch.inference_mode():
                grid = utils.create_uv_grid(width, height, aspect_ratio=112 / 84, dtype=torch.float32, device=torch.device('cpu'))
                embedding = utils.position_grid_to_embed(grid, channels) * 0.1
                value = embedding.permute(2, 0, 1)[None].numpy()
            info = array_info(value, item['shape'])
            if any(info[k] != item[k] for k in item): raise ValueError('Original position addition differs: ' + name)
        report['original_position_constants_verified'] = 5
        if (len(datasets) != 16 or len(report['samples']) != 8
                or any(len(line.split()) != 4 for line in datasets)
                or any(v != 8 for v in report['selected_blocks_forward_count'].values())):
            raise ValueError('Eight original pairs, 64 block calls and 16 four-input rows required')
        report['all_depth_and_conf_bitwise_equal_to_original_capture'] = True
        report['model_state_unchanged'] = tensor_state(model) == state
        if not report['model_state_unchanged']: raise ValueError('Original parameter/buffer identity changed')
        report['status'] = 'completed_original_FP32_depth_calibration_capture'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['hooks_removed'] = True
        for hook in hooks:
            try: hook.remove()
            except BaseException as exc:
                report['hooks_removed'] = False; report['cleanup_errors'].append('hook: ' + str(exc))
        try:
            report['source_and_inputs_unchanged'] = bool(pins) and all(sha(path) == digest for path, digest in pins.items())
            if not report['source_and_inputs_unchanged']: raise ValueError('Pinned source/input changed')
            if checkpoint_stat is not None:
                after = checkpoint.stat()
                report['checkpoint_stat_unchanged'] = (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) == (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, checkpoint_stat.st_ino, checkpoint_stat.st_dev)
                if not report['checkpoint_stat_unchanged']: raise ValueError('Original checkpoint stat changed')
            if model is not None and state is not None:
                report['model_state_unchanged'] = tensor_state(model) == state
                if not report['model_state_unchanged']: raise ValueError('Original model identity changed')
        except BaseException as exc: report['cleanup_errors'].append('identity: ' + str(exc))
        try:
            os.sched_setaffinity(0, affinity); report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            if not report['affinity_restored']: raise ValueError('Affinity not restored')
        except BaseException as exc: report['cleanup_errors'].append('affinity: ' + str(exc))
        for name, value in old_env.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value
        sys.path[:] = old_path
        report['environment_restored'] = all(os.environ.get(name) == value for name, value in old_env.items())
        report['seconds_including_hash_load_and_capture_IO'] = time.perf_counter() - started
        if report['cleanup_errors']: report['status'] = 'failed'
        if report['status'] == 'completed_original_FP32_depth_calibration_capture':
            dataset = root / 'dataset.txt'
            dataset.write_text(''.join(row + '\n' for row in datasets))
            report['dataset'] = dict(**file_info(dataset, root), line_count=16, input_count=4)
        temporary = root / 'capture.json.tmp'; temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, root / 'capture.json')
    print(json.dumps(dict(status=report['status'], error=report.get('error'), capture_sha256=sha(root / 'capture.json'))), flush=True)
    return 0 if report['status'] == 'completed_original_FP32_depth_calibration_capture' else 1


if __name__ == '__main__':
    raise SystemExit(main())
