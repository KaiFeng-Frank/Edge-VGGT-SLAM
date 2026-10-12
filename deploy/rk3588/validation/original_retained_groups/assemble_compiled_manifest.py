#!/usr/bin/env python3
"""Assemble saved original retained-group receipts; never imports a model or SDK."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
COMPOSER_SHA = 'e6fa508e0b03a011a1ccb484f2a46847ffd49fc1bf6ae11ecfeb8bd45f835f60'
CHECKER_SHA = '21d958ef385eaa419e81bef04303f07ffb5b7bb32a4854cd5d17ffba6247680b'
GROUPS = [list(range(0, 5)), list(range(5, 12)), list(range(12, 18)), list(range(18, 24))]
FRAME, GLOBAL = [2, 53, 1024], [1, 106, 1024]
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)
CHECKPOINT = dict(sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
                  bytes=5026874952)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pinned(root, path, expected, size=None):
    root, path = Path(root).resolve(strict=True), Path(path).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'Artifact must remain inside export root')
    require(type(expected) is str and len(expected) == 64 and sha(path) == expected,
            'Artifact SHA differs: ' + path.name)
    require(size is None or type(size) is int and size > 0 and path.stat().st_size == size,
            'Artifact size differs: ' + path.name)
    return path


def artifact(root, row):
    relative = Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
    return pinned(root, Path(root) / relative, row['sha256'], row['bytes'])


def metadata(root, path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(root), 'Manifest artifact escaped export root')
    return dict(file=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def successful_code(value):
    return value is None or type(value) is int and value == 0


def assemble(manifest_path, manifest_sha256, check_sha256, compile_sha256,
             compile_subdirs=('compile0', 'compile1', 'compile2', 'compile3')):
    require(len(compile_subdirs) == 4 and all(type(s) is str
        and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', s) for s in compile_subdirs),
        'Four safe single relative compile subdirectories required')
    manifest_path = Path(manifest_path).resolve(strict=True)
    root = manifest_path.parent
    pinned(root, manifest_path, manifest_sha256)
    export = json.loads(manifest_path.read_text())
    require(export['schema'] == 'v67_original_VGGT_retained_groups_ONNX_manifest'
        and export['status'] == 'static_export_only_not_runtime_validated'
        and export['script_sha256'] == COMPOSER_SHA and export['original_parent_manifest_sha256'] == PARENT_SHA
        and export['checkpoint'] == CHECKPOINT and export['image_shape'] == [2, 3, 84, 112]
        and export['depth'] == 24 and export['aa_order'] == ['frame', 'global'] and export['aa_block_size'] == 1
        and export['retained_consumer_indices'] == [4, 11, 17, 23]
        and export['original_parameter_value_count'] == 864 and export['all_original_24_pairs_retained'] is True
        and export['original_target_tokens_global_index'] == 20
        and export['original_target_token_clone_and_norm_must_be_retained'] is True
        and export['public_target_output_group_index'] == 3 and export['public_target_output_index'] == 2
        and export['public_target_output_name'] == 'target_global_output'
        and export['public_target_output_shape'] == GLOBAL and export['candidate_accepted'] is False
        and len(export['blocks']) == len(check_sha256) == len(compile_sha256) == 4,
        'Exact original four-group export and global20 contract required')
    result = copy.deepcopy(export)
    watched = {manifest_path: manifest_sha256, Path(__file__).resolve(): sha(__file__)}
    for index, (row, original_row) in enumerate(zip(result['blocks'], export['blocks'])):
        indices = GROUPS[index]
        bindings = [dict(index=0, name='frame_output', shape=FRAME, dtype='float32'),
                    dict(index=1, name='global_output', shape=GLOBAL, dtype='float32')]
        sources = [dict(index=0, name='frame_output', pair_index=indices[-1], role='frame'),
                   dict(index=1, name='global_output', pair_index=indices[-1], role='global')]
        if index == 3:
            bindings.append(dict(index=2, name='target_global_output', shape=GLOBAL, dtype='float32'))
            sources.append(dict(index=2, name='target_global_output', pair_index=20, role='global'))
        require(row['role'] == 'retained_pair_group' and row['index'] == index and row['indices'] == indices
            and row['input_name'] == 'input' and row['input_shape'] == FRAME and row['output_bindings'] == bindings
            and row['public_output_sources'] == sources
            and row['target_global_output_source_pair_index'] == (20 if index == 3 else None)
            and row['target_global_output_is_unnormalized'] is (True if index == 3 else None),
            'Original retained and target output bindings differ')
        graph = artifact(root, row)
        check_path = pinned(root, root / f'check{index}/check.json', check_sha256[index])
        compile_path = pinned(root, root / compile_subdirs[index] / 'result.json', compile_sha256[index])
        k, c = json.loads(check_path.read_text()), json.loads(compile_path.read_text())
        for receipt, mode in ((k, 'check'), (c, 'compile')):
            require(receipt['schema'] == 'v67_original_VGGT_retained_group_check_compile'
                and receipt['mode'] == mode and receipt['group_index'] == index and receipt['indices'] == indices
                and receipt['manifest_sha256'] == manifest_sha256 and receipt['graph_sha256'] == row['sha256']
                and receipt['script_sha256'] == CHECKER_SHA and receipt['original_parent_manifest_sha256'] == PARENT_SHA
                and receipt['input_shape'] == FRAME and receipt['output_bindings'] == bindings
                and receipt['public_output_sources'] == sources
                and receipt['parent_graph_sha256'] == [p['sha256'] for p in row['parents']]
                and receipt['parent_preservation_proofs'] == row['parent_preservation_proofs']
                and receipt['cleanup_errors'] == [] and receipt['candidate_accepted'] is False
                and receipt['quality_accepted'] is False and receipt['realtime_validated'] is False
                and receipt['source_and_inputs_unchanged'] is True
                and bool(receipt['source_and_input_hashes'])
                and receipt['source_and_input_hashes_after'] == receipt['source_and_input_hashes']
                and receipt['environment_restored'] is True and receipt['affinity_restored'] is True,
                'Completed pinned per-group receipt provenance/cleanup differs')
        require(k['status'] == 'checked_separate_ONNX_equivalence_not_quality_accepted'
            and k['all_separate_ONNX_checks_byte_equal'] is True and k['operations'] == []
            and k['sessions_dropped'] is True and k['ORT'] == dict(optimization='ORT_DISABLE_ALL',
                intra_op_num_threads=1, inter_op_num_threads=1, providers=['CPUExecutionProvider'],
                timing_is_not_NPU_performance=True), 'Complete byte-equal ORT chain check required')
        require(c['status'] == 'compiled_not_quality_accepted' and c['check_receipt_sha256'] == check_sha256[index]
            and c['configuration'] == CONFIG and c['build_options'] == BUILD
            and c['toolkit_version'].split('+')[0] == '2.3.2'
            and c['released'] is True and successful_code(c['release_code'])
            and [o['name'] for o in c['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
            and all(o['status'] == 'returned' and successful_code(o['code']) for o in c['operations'])
            and c['cases'] == k['cases']
            and all(c[field] == k[field] for field in ('original_FP32_gate_passed_where_reference_available',
                                                      'original_FP32_references_cover_all_cases')),
            'Successful unchanged nonquant FP16 compile and cleanup required')
        require(len(k['cases']) in (1, 2), 'One or two real captured inputs required')
        for case in k['cases']:
            require(len(case['checks']) == len(bindings)
                and [v['name'] for v in case['checks']] == [b['name'] for b in bindings]
                and all(v['byte_equal_to_separate_FP32_ONNX_chain'] is True for v in case['checks'])
                and case['actual_separate_pair_ORT_calls'] == len(indices)
                and case['actual_original_global_to_frame_views'] == len(indices) - 1
                and case['no_capture_intermediate_substitution'] is True
                and [s['index'] for s in case['chain_steps']] == indices
                and (index != 3 or case['target_global_source_pair_index'] == 20),
                'Complete actual parent chain and retained outputs required')
        model = artifact(compile_path.parent, c['rknn'])
        require(model.is_relative_to(root), 'Compiled model escaped export root')
        row.update(check_receipt=metadata(root, check_path), compile_receipt=metadata(root, compile_path),
                   rknn=metadata(root, model))
        require({key: row[key] for key in original_row} == original_row, 'Export row must be preserved')
        watched.update({graph: original_row['sha256'], check_path: check_sha256[index],
                        compile_path: compile_sha256[index], model: c['rknn']['sha256']})
    require(all(sha(path) == expected for path, expected in watched.items()), 'Inputs changed during assembly')
    result.update(schema='v67_original_VGGT_retained_groups_RKNN_manifest',
        status='compiled_not_quality_accepted', export_manifest_sha256=manifest_sha256,
        manifest_assembler_sha256=watched[Path(__file__).resolve()], compiler_source_sha256=CHECKER_SHA,
        cleanup_errors=[], all_group_check_and_compile_receipts_verified=True, source_and_inputs_unchanged=True,
        candidate_accepted=False, quality_accepted=False, realtime_validated=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--check-sha256', nargs=4, required=True)
    parser.add_argument('--compile-sha256', nargs=4, required=True)
    parser.add_argument('--compile-subdirs', nargs=4, default=['compile0', 'compile1', 'compile2', 'compile3'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.manifest.resolve(strict=True).parent, args.output.resolve()
    require(output.parent == root and not output.exists(), 'New compiled manifest must be in export root')
    result = assemble(args.manifest, args.manifest_sha256, args.check_sha256, args.compile_sha256,
                      args.compile_subdirs)
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    require(not output.exists(), 'Output already exists')
    temporary.rename(output)
    print(json.dumps(dict(status=result['status'], compiled_manifest_sha256=sha(output), groups=4,
                          candidate_accepted=False)), flush=True)


if __name__ == '__main__':
    main()
