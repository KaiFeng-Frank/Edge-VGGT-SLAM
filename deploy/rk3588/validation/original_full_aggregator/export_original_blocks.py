#!/usr/bin/env python3
"""Export all 48 captured original blocks in the ONNX compiler environment.

Pure static construction: no Torch/model/ORT/SDK execution or calibration.
Production capture and its original full-forward references remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import original_block_graph as G


def file_path(directory, item):
    relative = Path(item['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Capture-owned relative artifact required')
    path = (directory / relative).resolve(strict=True)
    if (not path.is_relative_to(directory) or path.stat().st_size != item['bytes']
            or G.sha256(path) != item['sha256']):
        raise ValueError('Captured artifact bytes/SHA changed: ' + str(relative))
    return path


def load_array(directory, item):
    value = np.load(file_path(directory, item), allow_pickle=False)
    if (list(value.shape) != item['shape'] or str(value.dtype) != item['dtype']
            or value.nbytes != item['tensor_bytes'] or G.raw_sha256(value) != item['raw_sha256']
            or not np.isfinite(value).all()):
        raise ValueError('Captured tensor shape/dtype/raw SHA changed')
    return np.ascontiguousarray(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--exporter', type=Path, required=True,
                        help='Frozen v17-r3 source beside its four unchanged static helpers')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.capture_dir.resolve(strict=True)
    receipt = directory / 'capture.json'
    if G.sha256(receipt) != args.capture_sha256:
        raise ValueError('Exact completed original full-forward capture SHA required')
    captured = json.loads(receipt.read_text())
    if (captured.get('schema') != 'v53_original_VGGT_48_block_capture'
            or captured.get('status') != 'completed_original_FP32_full_forward_and_48_block_capture'
            or captured.get('checkpoint') != G.CHECKPOINT or captured.get('image_shape') != list(G.IMAGE_SHAPE)
            or captured.get('full_model_forward_attempts') != 1
            or captured.get('full_model_forward_completed') is not True
            or captured.get('hooks_removed') is not True or captured.get('affinity_restored') is not True
            or captured.get('cleanup_errors') != []
            or captured.get('ORT_or_SDK_or_compiler_executed') is not False
            or captured.get('actual_inference_on_board_only') is not True
            or captured.get('original_model_only') is not True):
        raise ValueError('Clean complete original capture required')
    if captured['builder_source_sha256'] != G.sha256(G.__file__):
        raise ValueError('Use the exact static builder deployed for original capture')
    order = [(role, i) for i in range(24) for role in ('frame', 'global')]
    if [(r['role'], r['index']) for r in captured['blocks']] != order:
        raise ValueError('Complete ordered 24 frame plus 24 global original blocks required')
    rows = {(r['role'], r['index']): r for r in captured['blocks']}
    def state_provider(role, index):
        row = rows[(role, index)]
        with np.load(file_path(directory, row['weights']), allow_pickle=False) as archive:
            if set(archive.files) != set(G.PARAMETER_NAMES) or set(row['parameters']) != set(G.PARAMETER_NAMES):
                raise ValueError('Exact 18 captured original parameter tensors required')
            state = {name: np.ascontiguousarray(archive[name]) for name in G.PARAMETER_NAMES}
        for name, value in state.items():
            info = row['parameters'][name]
            if (value.dtype != np.float32 or list(value.shape) != info['shape']
                    or value.nbytes != info['tensor_bytes'] or G.raw_sha256(value) != info['raw_sha256']):
                raise ValueError('Original captured parameter changed: ' + name)
        print(f'Exporting original {role} block {index:02d}/23', flush=True)
        return state
    def metadata_provider(role, index):
        return rows[(role, index)]
    # Runtime source hashes refer exclusively to the original production src.
    # These two capture-source records remain separately bound and unchanged.
    capture_names = ('capture_original_blocks_board.py', 'original_block_graph.py')
    capture_sources = {name: captured['source_hashes'][name] for name in capture_names}
    source_hashes = {name: value for name, value in captured['source_hashes'].items() if name not in capture_names}
    frame_positions = load_array(directory, rows[('frame', 0)]['positions'])
    global_positions = load_array(directory, rows[('global', 0)]['positions'])
    if not np.array_equal(global_positions, frame_positions.reshape(1, 106, 2)):
        raise ValueError('Original frame/global positions are not the same C-order view')
    manifest = G.export_all_original_blocks(state_provider, metadata_provider, args.output_dir,
        exporter_path=args.exporter, frame_positions=frame_positions,
        rope_cos=load_array(directory, captured['rope_cos']), rope_sin=load_array(directory, captured['rope_sin']),
        checkpoint=captured['checkpoint'], source_hashes=source_hashes)
    manifest.update(capture_sha256=args.capture_sha256, capture_sources=capture_sources,
        export_source_sha256=G.sha256(__file__), graph_builder_source_sha256=G.sha256(G.__file__),
        original_reference_capture_directory=str(directory),
        original_full_forward_references=captured['references'])
    path = args.output_dir.resolve() / 'export.json'
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    if G.sha256(receipt) != args.capture_sha256:
        raise ValueError('Original capture receipt changed during static export')
    print(json.dumps({'status': manifest['status'], 'block_count': len(manifest['blocks']),
                      'export_sha256': G.sha256(path), 'candidate_accepted': False}), flush=True)


if __name__ == '__main__':
    main()
