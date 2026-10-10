#!/usr/bin/env python3
"""Read-only K/FOV source and saved-array audit, without importing the model."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V17_COMPARE_SHA = '0f740d3a35c472745eccb3f674b411209a2cef6fcd56ba66e44e6274e224979b'


def comparison_module():
    path = HERE / 'compare-v17-resolution-pairs.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != V17_COMPARE_SHA:
        raise ValueError('Frozen resolution comparison dependency changed')
    spec = importlib.util.spec_from_file_location('fov_saved_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = comparison_module()
geometry = comparison.geometry


def decoded_intrinsics(pose_encoding, image_hw):
    """Independent NumPy transcription of the reviewed radian FOV conversion."""
    value = np.asarray(pose_encoding, np.float64)
    if (value.ndim < 1 or value.shape[-1] != 9 or not np.isfinite(value).all()
            or len(image_hw) != 2 or any(type(n) is not int or n <= 0 for n in image_hw)):
        raise ValueError('Invalid pose encoding or actual image dimensions')
    height, width = image_hw
    vertical, horizontal = value[..., 7], value[..., 8]
    if np.any((vertical <= 0) | (vertical >= np.pi) | (horizontal <= 0) | (horizontal >= np.pi)):
        raise ValueError('Positive finite focal reconstruction requires FOV radians in (0, pi)')
    K = np.zeros(value.shape[:-1] + (3, 3), np.float64)
    K[..., 0, 0] = (width / 2.) / np.tan(horizontal / 2.)
    K[..., 1, 1] = (height / 2.) / np.tan(vertical / 2.)
    K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = width / 2., height / 2., 1.
    return K


def function(text, name):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f'Expected one source definition: {name}')
    return nodes[0]


def assert_assignment(node, target, expression):
    expected_target = ast.unparse(ast.parse(target, mode='eval').body)
    expected_value = ast.dump(ast.parse(expression, mode='eval').body, include_attributes=False)
    matched = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
               and any(ast.unparse(t) == expected_target for t in n.targets)]
    if len(matched) != 1 or ast.dump(matched[0].value, include_attributes=False) != expected_value:
        raise ValueError(f'Reviewed source assignment changed: {target}')


def verify_codec_source(text):
    node = function(text, 'pose_encoding_to_extri_intri')
    for target, expression in (
        ('(H, W)', 'image_size_hw'), ('fov_h', 'pose_encoding[..., 7]'),
        ('fov_w', 'pose_encoding[..., 8]'), ('fy', '(H / 2.0) / torch.tan(fov_h / 2.0)'),
        ('fx', '(W / 2.0) / torch.tan(fov_w / 2.0)'),
        ('intrinsics[..., 1, 1]', 'fy'), ('intrinsics[..., 0, 0]', 'fx')):
        assert_assignment(node, target, expression)
    return dict(height_width_order='H,W', pose_encoding_vertical_index=7,
                pose_encoding_horizontal_index=8, angle_unit='radians',
                fy='(H/2)/tan(fov_h/2)', fx='(W/2)/tan(fov_w/2)',
                principal_point_assumption='cx=W/2; cy=H/2, rather than externally calibrated offsets',
                FOV_definition='center-symmetric codec surrogate from focal; off-center edge angles need not equal this',
                no_additional_trained_resolution_factor=True)


def source_contracts():
    paths = dict(runtime='vggt_slam/runtime.py',
        load_fn='third_party/vggt/vggt/utils/load_fn.py',
        pose_enc='third_party/vggt/vggt/utils/pose_enc.py',
        camera_head='third_party/vggt/vggt/heads/camera_head.py',
        head_act='third_party/vggt/vggt/heads/head_act.py',
        vision_transformer='third_party/vggt/vggt/layers/vision_transformer.py',
        patch_embed='third_party/vggt/vggt/layers/patch_embed.py',
        aggregator='third_party/vggt/vggt/models/aggregator.py',
        vggt='third_party/vggt/vggt/models/vggt.py',
        rope='third_party/vggt/vggt/layers/rope.py')
    texts = {name: (ROOT / path).read_text() for name, path in paths.items()}
    hashes = {path: geometry.sha(ROOT / path) for path in paths.values()}
    codec = verify_codec_source(texts['pose_enc'])
    loader = function(texts['load_fn'], 'load_and_preprocess_images')
    assert_assignment(loader, 'target_size', 'int(image_size)')
    resize = [n for n in ast.walk(loader) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == 'resize']
    if len(resize) != 1 or ast.unparse(resize[0].args[0]) != '(new_width, new_height)':
        raise ValueError('Original Pillow resize dimension order changed')
    interp = function(texts['vision_transformer'], 'interpolate_pos_encoding')
    tokens = function(texts['vision_transformer'], 'prepare_tokens_with_masks')
    assert_assignment(interp, 'w0', 'w // self.patch_size')
    assert_assignment(interp, 'h0', 'h // self.patch_size')
    assert_assignment(tokens, '(B, nc, w, h)', 'x.shape')
    if 'kwargs["size"] = (w0, h0)' not in texts['vision_transformer']:
        raise ValueError('Original positional interpolation output grid changed')
    model_load = function(texts['runtime'], 'load_model')
    constructors = [n for n in ast.walk(model_load) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'VGGT']
    if len(constructors) != 1 or any(k.arg == 'img_size' for k in constructors[0].keywords):
        raise ValueError('Loader no longer preserves the original default positional grid')
    vendor = ROOT / 'third_party/vggt'
    unchanged = {}
    for key in ('pose_enc', 'camera_head', 'head_act', 'patch_embed', 'rope'):
        relative = Path(paths[key]).relative_to('third_party/vggt').as_posix()
        old = subprocess.run(['git', '-C', str(vendor), 'show', f'HEAD:{relative}'],
                             check=True, text=True, capture_output=True).stdout
        unchanged[key] = old == texts[key]
        if not unchanged[key]:
            raise ValueError(f'Reviewed unchanged vendored source differs: {relative}')
    old_vit = subprocess.run(['git', '-C', str(vendor), 'show', 'HEAD:vggt/layers/vision_transformer.py'],
                             check=True, text=True, capture_output=True).stdout
    positional_equal = all(ast.get_source_segment(texts['vision_transformer'], function(texts['vision_transformer'], name))
                           == ast.get_source_segment(old_vit, function(old_vit, name))
                           for name in ('interpolate_pos_encoding', 'prepare_tokens_with_masks'))
    if not positional_equal:
        raise ValueError('DINO positional interpolation differs from the vendored original')
    producer_calls = {}
    for name in ('capture-v16-resolution-pairs-board.py', 'capture-v17-resolution-pairs-board.py'):
        tree = ast.parse((HERE / name).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'pose_encoding_to_extri_intri']
        if len(calls) != 1 or ast.unparse(calls[0].args[1]) != 'images.shape[-2:]':
            raise ValueError('Capture stopped using the actual input H,W for K conversion')
        producer_calls[name] = dict(source_sha256=geometry.sha(HERE / name),
                                   conversion_dimensions='images.shape[-2:]')
    head = next(
        n for n in ast.parse(texts['camera_head']).body if isinstance(n, ast.ClassDef) and n.name == 'CameraHead')
    init = next(n for n in head.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    if any(a.arg in ('image_size', 'img_size') for a in init.args.args):
        raise ValueError('CameraHead now has an explicit resolution argument')
    return dict(hashes=hashes, codec=codec, capture_conversion=producer_calls,
                vendor_commit=subprocess.run(['git', '-C', str(vendor), 'rev-parse', 'HEAD'],
                                             check=True, text=True, capture_output=True).stdout.strip(),
                whole_files_byte_equal_to_vendor_HEAD=unchanged,
                DINO_position_functions_byte_equal_to_vendor_HEAD=positional_equal,
                positional_base_grid_implied_by_preserved_518_constructor=[37, 37],
                DINO_dimension_names='w is input tensor H; h is input tensor W; interpolation follows the same order',
                camera_head_has_explicit_image_size_argument=False,
                camera_head_prediction='last-layer camera token; iterative 9D pose/FOV prediction with positive ReLU FOV',
                preprocessing='Pillow resize(new_width,new_height), independent y rounding to 14, no crop for these landscape pairs',
                source_reviews=[dict(path=paths['pose_enc'], line=114), dict(path=paths['load_fn'], line=158),
                    dict(path=paths['vision_transformer'], line=181), dict(path=paths['camera_head'], line=75),
                    dict(path=paths['runtime'], line=85), dict(path=paths['aggregator'], line=505)])


def audit(capture_dirs):
    if set(capture_dirs) != {112, 224, 336, 518}:
        raise ValueError('This audit requires the same three recorded pairs at all four resolutions')
    contracts = source_contracts()
    loaded = {}
    for size, path in capture_dirs.items():
        loaded[size] = (comparison.v16.load_capture(path, size) if size in (112, 518)
                        else comparison.load_capture(path, size, comparison.V17_CAPTURE_SHA))
    reference = loaded[518][0]
    comparison.v16.verify_identity(loaded[112][0], reference)
    comparison.verify_identity(loaded[224][0], reference)
    comparison.verify_identity(loaded[336][0], reference)
    for name, digest in contracts['hashes'].items():
        if name in reference['source_hashes'] and reference['source_hashes'][name] != digest:
            raise ValueError('Current reviewed model source differs from the captured model source')
    maximum_error = 0.
    result = dict(kind='v17_read_only_FOV_dimension_source_and_saved_array_audit', status='passed',
                  auditor_sha256=geometry.sha(__file__), comparison_dependency_sha256=V17_COMPARE_SHA,
                  image_decode_executed=False, model_executed=False, tracking_executed=False,
                  board_operations_executed=False, raw_inputs_modified=False,
                  reviewed_path_scaling_bug_identified=False, training_distribution_cause_validated=False,
                  complete_dataset_accuracy_validated=False,
                  conclusion='Saved CameraHead vertical FOV differs before conversion; no H/W or focal scaling bug identified in the reviewed path',
                  limitations='Only the three fixed pairs; the exact cause inside learned features is not established; no ATE/RPE acceptance or focal rescaling correction is inferred',
                  source_contracts=contracts, resolutions=[])
    for size in (112, 224, 336, 518):
        receipt, pairs, hashes = loaded[size]
        raw_h, raw_w = receipt['rectification']['image_hw']
        physical_K = np.asarray(receipt['rectification']['output_K'], np.float64)
        row = dict(image_size=size, capture_input_hashes=hashes, pairs=[],
                   supplied_rectified_K=physical_K.tolist(),
                   expected_codec_vertical_FOV_degrees=float(np.degrees(2 * np.arctan(raw_h / (2 * physical_K[1, 1])))),
                   expected_codec_horizontal_FOV_degrees=float(np.degrees(2 * np.arctan(raw_w / (2 * physical_K[0, 0])))))
        fy_ratios, verticals = [], []
        for pair in pairs:
            h, w = pair['mapping']['actual_input_hw']
            pose = pair['arrays']['pose_enc'][0]
            reconstructed = decoded_intrinsics(pose, [h, w])
            stored = pair['arrays']['intrinsic'][0].astype(np.float64)
            difference = np.max(np.abs(reconstructed[..., [0, 1], [0, 1]] - stored[..., [0, 1], [0, 1]])
                                / np.abs(stored[..., [0, 1], [0, 1]]))
            if not np.allclose(reconstructed, stored, rtol=1e-6, atol=1e-5):
                raise ValueError('Saved K differs from the independently reconstructed radian FOV K')
            maximum_error = max(maximum_error, float(difference))
            raw_K = np.linalg.inv(pair['A'])[None] @ stored
            fy_ratio = raw_K[:, 1, 1] / physical_K[1, 1]
            vertical = np.degrees(pose[:, 7].astype(np.float64))
            verticals.extend(vertical.tolist()); fy_ratios.extend(fy_ratio.tolist())
            row['pairs'].append(dict(revision=pair['record']['revision'], frame_ids=pair['record']['frame_ids'],
                input_hw=[h, w], predicted_vertical_FOV_degrees=vertical.tolist(),
                predicted_horizontal_FOV_degrees=np.degrees(pose[:, 8].astype(np.float64)).tolist(),
                original_pixel_fy=raw_K[:, 1, 1].tolist(), fy_ratio_to_supplied_K=fy_ratio.tolist(),
                principal_point_delta_to_supplied_K_pixels=(raw_K[:, :2, 2] - physical_K[None, :2, 2]).tolist(),
                independent_codec_max_relative_error=float(difference),
                actual_height_rounding_relative_to_isotropic=float(h / (raw_h * w / raw_w) - 1),
                raw_fy_cancellation='(Hraw/Hmodel)*(Hmodel/2)/tan(fov_h/2) = Hraw/[2*tan(fov_h/2)]',
                half_pixel_translation_affects_focal_diagonal=False))
        row['fy_ratio_range'] = [min(fy_ratios), max(fy_ratios)]
        row['predicted_vertical_FOV_range_degrees'] = [min(verticals), max(verticals)]
        result['resolutions'].append(row)
    result['maximum_independent_intrinsic_codec_relative_error'] = maximum_error
    for size, (_, _, hashes) in loaded.items():
        if any(geometry.sha(Path(capture_dirs[size]) / name) != digest for name, digest in hashes.items()):
            raise ValueError('Saved input changed during audit')
    if any(geometry.sha(ROOT / name) != digest for name, digest in contracts['hashes'].items()):
        raise ValueError('Reviewed source changed during audit')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for size in (112, 224, 336, 518):
        parser.add_argument(f'--capture-{size}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(argv)
    if a.output.exists():
        parser.error('Audit output must be a new JSON file')
    result = audit({size: getattr(a, f'capture_{size}') for size in (112, 224, 336, 518)})
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=a.output.name, sha256=geometry.sha(a.output))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
