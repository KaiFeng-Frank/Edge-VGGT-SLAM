#!/usr/bin/env python3
"""Join the 48 original block graphs into 24 frame/global two-output graphs.

Pure static ONNX construction only. Reuse the existing v53 composer, preserve
both complete parent graphs and insert their original C-order reshape. The
original aggregator must still concatenate both outputs and retain its usual
intermediate layers. Numerical acceptance and speed require board execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

COMPOSER_SHA = '3ff8b0d0515ff44ea3625bd2c6529d63a0bed3487d409128ee4b443922a14bfd'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact(directory, row):
    relative = Path(row['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Parent-owned relative artifact required')
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory) or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Original parent artifact bytes/SHA changed: ' + str(relative))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True, help='Original v53 export.json or compiled.json')
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--composer', type=Path,
        default=Path(__file__).resolve().parent.parent / 'original_full_aggregator/original_block_graph.py')
    args = p.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    if sha(manifest_path) != args.manifest_sha256 or sha(args.composer) != COMPOSER_SHA:
        raise ValueError('Exact original manifest and unchanged v53 static composer required')
    spec = importlib.util.spec_from_file_location('_v55_original_pair_composer', args.composer)
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    parent = json.loads(manifest_path.read_text())
    if (parent['schema'] not in ('v53_original_VGGT_48_block_ONNX_manifest', 'v53_original_VGGT_48_block_RKNN_manifest')
            or parent['checkpoint'] != G.CHECKPOINT or parent['image_shape'] != [2, 3, 84, 112]
            or parent['depth'] != 24 or parent['aa_order'] != ['frame', 'global'] or parent['aa_block_size'] != 1
            or [(b['role'], b['index']) for b in parent['blocks']] != [(r, i) for i in range(24) for r in ('frame', 'global')]):
        raise ValueError('Complete unchanged original 48-block manifest required')
    import onnx
    directory = manifest_path.parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = {(row['role'], row['index']): row for row in parent['blocks']}
    for role in ('frame', 'global'):
        position = rows[(role, 0)]['positions']
        source = artifact(directory, position)
        destination = output / position['file']
        if not destination.parent == output:
            raise ValueError('Original v53 position artifact must be a root-level file')
        shutil.copyfile(source, destination)
        if sha(destination) != position['sha256']:
            raise ValueError('Original position bytes changed during copy')
    fields = ('checkpoint', 'source_hashes', 'capture_sources', 'capture_sha256', 'image_shape',
              'depth', 'heads', 'frames', 'tokens', 'dim', 'patch_start_idx', 'aa_order', 'aa_block_size')
    manifest = {field: parent[field] for field in fields if field in parent}
    manifest.update(schema='v55_original_VGGT_24_frame_global_pair_ONNX_manifest',
        status='static_export_only_not_runtime_validated', script_sha256=sha(__file__),
        pair_composer_source_sha256=COMPOSER_SHA,
        original_parent_manifest_sha256=args.manifest_sha256,
        original_parent_manifest_directory=str(directory), blocks=[],
        calibration_or_ORT_or_SDK_or_model_execution_by_builder=False,
        candidate_accepted=False, full_model_quality_or_realtime_validated=False,
        all_original_48_blocks_retained=True, original_aggregator_concat_must_be_retained=True)
    for index in range(24):
        frame, global_ = rows[('frame', index)], rows[('global', index)]
        for row, shape in ((frame, [2, 53, 1024]), (global_, [1, 106, 1024])):
            if (row['input_shape'] != shape or row['output_shape'] != shape or row['heads'] != 16
                    or row['all_18_caller_parameter_values_preserved_in_graph'] is not True
                    or set(row['parameter_raw_sha256']) != set(G.PARAMETER_NAMES)
                    or row['positions'] != rows[(row['role'], 0)]['positions']):
                raise ValueError('Complete original parent shape/parameters/fixed positions differ')
        fpath, gpath = artifact(directory, frame), artifact(directory, global_)
        f = onnx.load(str(fpath), load_external_data=False)
        g = onnx.load(str(gpath), load_external_data=False)
        pair = G.compose_frame_global_pair(f, g, index=index)
        if len(pair.graph.node) != len(f.graph.node) + len(g.graph.node) + 1:
            raise ValueError('Pair must contain both complete parents and one original view')
        bindings = [dict(index=0, name='frame_output', shape=[2, 53, 1024], dtype='float32'),
                    dict(index=1, name='global_output', shape=[1, 106, 1024], dtype='float32')]
        if ([v.name for v in pair.graph.input] != ['input']
                or [v.name for v in pair.graph.output] != [b['name'] for b in bindings]
                or any(v.type.tensor_type.elem_type != 1 for v in (*pair.graph.input, *pair.graph.output))):
            raise ValueError('Original input and both FP32 public outputs must remain visible')
        path = output / f'pair-{index:02d}.onnx'
        with path.open('xb') as stream:
            stream.write(pair.SerializeToString())
        manifest['blocks'].append(dict(role='pair', index=index, file=path.name,
            bytes=path.stat().st_size, sha256=sha(path), input_name='input', input_shape=[2, 53, 1024],
            output_bindings=bindings, heads=16, parents={'frame': frame, 'global': global_},
            positions={'frame': frame['positions'], 'global': global_['positions']},
            parameter_raw_sha256={'frame': frame['parameter_raw_sha256'], 'global': global_['parameter_raw_sha256']},
            all_36_original_parameter_values_preserved_by_parent_composition=True,
            parent_nodes_plus_one_reshape=True, original_aggregator_concat_must_be_retained=True))
        del f, g, pair
        print(f'Composed complete original frame/global pair {index:02d}: {index+1}/24', flush=True)
    if sha(manifest_path) != args.manifest_sha256 or sha(args.composer) != COMPOSER_SHA:
        raise ValueError('Frozen original parent manifest/composer changed during construction')
    with (output / 'export.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=manifest['status'], pair_count=24,
                          export_sha256=sha(output / 'export.json'))), flush=True)


if __name__ == '__main__':
    main()
