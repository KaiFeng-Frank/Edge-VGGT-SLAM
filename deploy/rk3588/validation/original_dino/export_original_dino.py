#!/usr/bin/env python3
"""Statically export every captured original DINO layer as one-frame ONNX.

No Torch/model/ORT/SDK execution. All 24 graphs retain every original token,
parameter, attention, exact Erf GELU, LayerScale and residual operation.
Original preparation/position interpolation and final normalization stay in
the original DINO forward, outside these block graphs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import original_dino_graph as G


def artifact(directory, item):
    relative = Path(item['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Capture-owned relative artifact required')
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory) or path.stat().st_size != item['bytes'] or G.sha(path) != item['sha256']:
        raise ValueError('Original captured artifact size/SHA mismatch: ' + str(relative))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--capture-sha256', required=True)
    p.add_argument('--operator-directory', type=Path, default=Path(__file__).resolve().parent.parent,
                   help='Frozen v12 original block and GELU static source directory')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    directory = args.capture_dir.resolve(strict=True)
    receipt = directory / 'capture.json'
    if G.sha(receipt) != args.capture_sha256:
        raise ValueError('Exact completed original DINO capture SHA required')
    captured = json.loads(receipt.read_text())
    if (captured['schema'] != 'v54_original_VGGT_24_DINO_block_capture'
            or captured['status'] != 'completed_original_FP32_24_DINO_block_capture'
            or captured['checkpoint'] != G.CHECKPOINT or captured['image_shape'] != [2, 3, 84, 112]
            or captured['single_image_shape'] != list(G.IMAGE_SHAPE)
            or captured['original_batch_forward_attempts'] != 1 or captured['original_single_forward_attempts'] != 2
            or captured['actual_inference_on_board_only'] is not True or captured['original_model_only'] is not True
            or captured['ORT_or_SDK_or_compiler_executed'] is not False
            or captured['hooks_removed'] is not True or captured['affinity_restored'] is not True
            or captured['environment_restored'] is not True or captured['source_and_inputs_unchanged'] is not True
            or captured['checkpoint_stat_unchanged'] is not True or captured['cleanup_errors'] != []
            or captured['source_hashes'] != captured['source_hashes_after']
            or captured['graph_builder_source_sha256'] != G.sha(G.__file__)
            or captured['capture_sources']['original_dino_graph.py'] != G.sha(G.__file__)
            or captured['capture_sources']['capture_original_dino_board.py'] != captured['script_sha256']):
        raise ValueError('Clean complete original batch plus single-frame DINO capture required')
    if [(row['role'], row['index']) for row in captured['blocks']] != [('dino', i) for i in range(24)]:
        raise ValueError('Every original DINO block 0..23 exactly once required')
    source_hashes = captured['source_hashes']
    if (not source_hashes or any(Path(name).is_absolute() or '..' in Path(name).parts for name in source_hashes)
            or any(not isinstance(value, str) or len(value) != 64 for value in source_hashes.values())):
        raise ValueError('Original source hashes must use production src relative paths')
    G.load_builders(args.operator_directory)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema='v54_original_VGGT_24_DINO_block_ONNX_manifest',
        status='static_export_only_not_runtime_validated', script_sha256=G.sha(__file__),
        graph_builder_source_sha256=G.sha(G.__file__), checkpoint=dict(G.CHECKPOINT),
        source_hashes=dict(source_hashes), capture_sources=dict(captured['capture_sources']),
        operator_builder_sources=dict(G.OPERATOR_BUILDERS), capture_sha256=args.capture_sha256,
        image_shape=list(G.IMAGE_SHAPE), capture_image_shape=captured['image_shape'],
        depth=24, embed_dim=1024, dim=1024, heads=16, frames=1, tokens=53,
        patch_size=14, patch_start_idx=5, num_register_tokens=4, cls_tokens=1, chunked_blocks=False,
        original_non_block_parameter_raw_sha256=captured['non_block_parameter_raw_sha256'],
        original_reference_capture_directory=str(directory), original_DINO_references=captured['references'],
        blocks=[], calibration_or_ORT_or_SDK_or_model_execution_by_builder=False,
        candidate_accepted=False, full_NPU_model_or_full_model_quality_or_realtime_validated=False)
    for index, row in enumerate(captured['blocks']):
        if (row['original_batch_shape'] != [2, 53, 1024] or row['single_frame_shape'] != list(G.SHAPE)
                or set(row['references']) != {'batch2', 'single0', 'single1'}):
            raise ValueError('Original batch-two and both one-frame references required for every layer')
        with np.load(artifact(directory, row['weights']), allow_pickle=False) as archive:
            if set(archive.files) != set(G.PARAMETER_NAMES) or set(row['parameters']) != set(G.PARAMETER_NAMES):
                raise ValueError('Every original DINO parameter required')
            weights = {name: np.ascontiguousarray(archive[name]) for name in G.PARAMETER_NAMES}
        for name, value in weights.items():
            info = row['parameters'][name]
            if (value.dtype != np.float32 or list(value.shape) != info['shape'] or value.nbytes != info['tensor_bytes']
                    or G.raw_sha(value) != info['raw_sha256']):
                raise ValueError('Original DINO learned tensor changed: ' + name)
        model = G.build_original_dino_block(weights, index=index, metadata=row,
                                           operator_directory=args.operator_directory)
        path = output / f'dino-{index:02d}.onnx'
        with path.open('xb') as stream:
            stream.write(model.SerializeToString())
        manifest['blocks'].append(dict(role='dino', index=index, file=path.name,
            bytes=path.stat().st_size, sha256=G.sha(path), input_name='input', output_name='output',
            input_shape=list(G.SHAPE), output_shape=list(G.SHAPE), heads=16,
            epsilons=dict(row['epsilons']), gelu_approximate='none', qk_norm=False, rope=None,
            position_included_in_input=True,
            parameter_raw_sha256={name: G.raw_sha(value) for name, value in weights.items()},
            all_14_caller_parameter_values_preserved_in_graph=True,
            original_capture_references=row['references']))
        del model, weights
        print(f'Exported original DINO block {index:02d}: {index + 1}/24 complete', flush=True)
    if G.sha(receipt) != args.capture_sha256:
        raise ValueError('Original capture changed during static export')
    with (output / 'export.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=manifest['status'], blocks=24,
                          export_sha256=G.sha(output / 'export.json'))), flush=True)


if __name__ == '__main__':
    main()
