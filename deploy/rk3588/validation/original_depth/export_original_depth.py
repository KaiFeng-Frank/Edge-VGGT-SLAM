#!/usr/bin/env python3
"""Private static ONNX DPT export from verified board capture, no inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import depth_contract as C
import original_depth_graph as G


def artifact_path(directory, item):
    relative = Path(item['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Capture-owned relative artifact path required')
    path = (directory / relative).resolve(strict=True)
    if (not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != item['bytes']
            or C.sha256(path) != item['sha256']):
        raise ValueError('Original captured artifact bytes/SHA mismatch')
    return path


def load_archive(directory, item):
    with np.load(artifact_path(directory, item), allow_pickle=False) as archive:
        if set(archive.files) != set(item['tensors']):
            raise ValueError('Captured tensor identities differ')
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for name, value in arrays.items():
        if (value.dtype != np.float32 or not np.isfinite(value).all()
                or C.array_info(value) != item['tensors'][name]):
            raise ValueError('Original tensor shape/dtype/raw SHA mismatch: '+name)
    return arrays


def load_capture(directory, expected_sha256):
    directory = Path(directory).resolve(strict=True)
    receipt = directory / 'capture.json'
    if C.sha256(receipt) != expected_sha256:
        raise ValueError('Original depth capture SHA mismatch')
    report = json.loads(receipt.read_text())
    if (report.get('schema') != 'v54_original_VGGT_depth_capture'
            or report.get('status') != 'completed_original_FP32_depth_capture'
            or report.get('checkpoint') != C.CHECKPOINT or report.get('image_shape') != [1,2,3,84,112]
            or report.get('contract_source_sha256') != C.sha256(C.__file__)
            or report.get('original_model_only') is not True or report.get('actual_inference_on_board_only') is not True
            or report.get('ORT_or_SDK_or_compiler_executed') is not False
            or report.get('full_model_forward_attempts') != 1 or report.get('full_model_forward_completed') is not True
            or report.get('original_depth_head_calls') != 1 or report.get('original_raw_logits_calls') != 2
            or report.get('original_position_utility_calls_after_forward') != 5
            or report.get('model_parameter_identity_unchanged') is not True
            or report.get('hooks_removed') is not True or report.get('affinity_restored') is not True
            or report.get('cleanup_errors') != []):
        raise ValueError('Clean complete original FP32 depth capture required')
    if any(report['source_hashes'].get(name) != expected for name, expected in C.SOURCE_HASHES.items()):
        raise ValueError('Pinned original DPT source identity differs')
    architecture = report['architecture']
    if (architecture['layers'] != list(C.LAYERS) or architecture['norm_epsilon'] != 1e-5
            or architecture['preserved_original_parameter_count'] != 62
            or architecture['frames_chunk_size'] != 1 or architecture['activation'] != 'exp'
            or architecture['conf_activation'] != 'expp1' or architecture['position_ratio'] != .1
            or architecture['input_shape'] != list(C.INPUT_SHAPE) or architecture['output_shape'] != list(C.OUTPUT_SHAPE)):
        raise ValueError('Original fixed DPT architecture differs')
    return directory, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory, captured = load_capture(args.capture_dir, args.capture_sha256)
    state = load_archive(directory, captured['artifacts']['state'])
    positions = load_archive(directory, captured['artifacts']['positions'])
    C.verify_arrays(state, C.PARAMETER_SHAPES)
    C.verify_arrays(positions, C.POSITION_SHAPES)
    graph = G.build_original_depth_logits_onnx(state, norm_epsilon=captured['architecture']['norm_epsilon'],
                                               positional_embeddings=positions)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'original-depth-logits.onnx'
    with path.open('xb') as stream:
        stream.write(graph.SerializeToString())
    report = {'schema': 'v54_original_VGGT_depth_logits_ONNX_manifest',
        'status': 'static_export_only_not_runtime_validated', 'checkpoint': dict(C.CHECKPOINT),
        'architecture': captured['architecture'], 'source_hashes': captured['source_hashes'],
        'capture_sha256': args.capture_sha256, 'capture_script_sha256': captured['script_sha256'],
        'contract_source_sha256': C.sha256(C.__file__), 'graph_builder_source_sha256': C.sha256(G.__file__),
        'export_source_sha256': C.sha256(__file__), 'parameters': captured['artifacts']['state']['tensors'],
        'positions': captured['artifacts']['positions']['tensors'],
        'graph': {'file': path.name, 'bytes': path.stat().st_size, 'sha256': C.sha256(path),
                  'input_names': list(C.INPUT_NAMES), 'input_shapes': {name: list(C.INPUT_SHAPE) for name in C.INPUT_NAMES},
                  'output_name': 'logits', 'output_shape': list(C.OUTPUT_SHAPE), 'opset': 17, 'ir_version': 8},
        'all_62_original_parameter_values_preserved_in_graph': True,
        'original_inplace_relu_residual_skip_preserved': True,
        'all_five_original_align_corners_resizes_preserved': True,
        'model_or_ORT_or_SDK_execution_by_exporter': False,
        'candidate_accepted': False, 'full_model_quality_or_realtime_validated': False}
    if C.sha256(directory/'capture.json') != args.capture_sha256:
        raise ValueError('Original capture receipt changed during static export')
    receipt = output / 'export.json'
    receipt.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'export_sha256': C.sha256(receipt),
                      'graph_sha256': report['graph']['sha256']}), flush=True)


if __name__ == '__main__':
    main()
