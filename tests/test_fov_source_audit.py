"""Pure source parsing and synthetic radian-FOV arrays; no model imports."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit():
    path = ROOT / 'deploy/rk3588/validation/audit-v17-fov-sources.py'
    spec = importlib.util.spec_from_file_location('v17_fov_source_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('size,height', [(112, 84), (224, 168), (336, 252), (518, 392)])
def test_actual_height_and_half_pixel_mapping_recover_focal_without_resolution_factor(audit, size, height):
    A, _ = audit.comparison.pixel_affine([480, 640], [2, 3, height, size], size)
    physical_K = np.array([[517.3, 0., 318.6], [0., 516.5, 255.3], [0., 0., 1.]])
    model_K = A @ physical_K
    pose = np.zeros((2, 9))
    pose[:, 7] = 2 * np.arctan((height / 2) / model_K[1, 1])
    pose[:, 8] = 2 * np.arctan((size / 2) / model_K[0, 0])
    recovered = np.linalg.inv(A)[None] @ audit.decoded_intrinsics(pose, [height, size])
    assert np.allclose(recovered[:, 0, 0], physical_K[0, 0], rtol=1e-12)
    assert np.allclose(recovered[:, 1, 1], physical_K[1, 1], rtol=1e-12)
    # The original camera encoding deliberately assumes a centered principal point.
    assert not np.allclose(recovered[:, :2, 2], physical_K[None, :2, 2])
    assert pose[0, 7] == pytest.approx(2 * np.arctan(480 / (2 * 516.5)))


def test_narrow_vertical_FOV_is_already_an_encoding_bias_not_a_resize_scale_error(audit):
    pose = np.zeros((1, 9)); pose[:, 7], pose[:, 8] = .22, 1.1
    focal = []
    for size, height in [(112, 84), (224, 168), (336, 252), (518, 392)]:
        A, _ = audit.comparison.pixel_affine([480, 640], [2, 3, height, size], size)
        K = np.linalg.inv(A)[None] @ audit.decoded_intrinsics(pose, [height, size])
        focal.append(K[0, 1, 1])
    assert np.allclose(focal, 480 / (2 * np.tan(.22 / 2)), rtol=1e-12)
    assert min(focal) > 4 * 516.5
    # An H/W swap in this landscape input would be 4/3, not the observed 2-4x.
    assert 112 / 84 == pytest.approx(4 / 3)
    assert 392 / (480 * 518 / 640) - 1 == pytest.approx(.009009009009009)


@pytest.mark.parametrize('bad', ['vertical_index', 'horizontal_index', 'height_width', 'fy_uses_width'])
def test_source_contract_rejects_dimension_and_FOV_index_changes(audit, codec_source, bad):
    text = codec_source
    if bad == 'vertical_index': text = text.replace('fov_h = pose_encoding[..., 7]', 'fov_h = pose_encoding[..., 8]')
    if bad == 'horizontal_index': text = text.replace('fov_w = pose_encoding[..., 8]', 'fov_w = pose_encoding[..., 7]')
    if bad == 'height_width': text = text.replace('H, W = image_size_hw', 'W, H = image_size_hw')
    if bad == 'fy_uses_width': text = text.replace('fy = (H / 2.0)', 'fy = (W / 2.0)')
    assert text != codec_source
    with pytest.raises(ValueError, match='source assignment changed'):
        audit.verify_codec_source(text)


@pytest.fixture
def codec_source(audit):
    # Parsing only: no Torch import/execution or dependency on a vendor checkout.
    text = '''def pose_encoding_to_extri_intri(pose_encoding, image_size_hw):
    H, W = image_size_hw
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    fy = (H / 2.0) / torch.tan(fov_h / 2.0)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0)
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 0] = fx
    return intrinsics
'''
    # Every negative starts from a fixture that passes the unchanged contract.
    normal = audit.verify_codec_source(text)
    assert normal['height_width_order'] == 'H,W'
    assert normal['pose_encoding_vertical_index'] == 7
    assert normal['pose_encoding_horizontal_index'] == 8
    return text


def test_current_reviewed_sources_and_capture_HW_order_are_consistent(audit):
    if not (ROOT / 'third_party/vggt/.git').exists():
        pytest.skip('requires pinned VGGT source checkout in third_party/vggt')
    contracts = audit.source_contracts()
    assert contracts['codec']['pose_encoding_vertical_index'] == 7
    assert contracts['codec']['pose_encoding_horizontal_index'] == 8
    assert all(contracts['whole_files_byte_equal_to_vendor_HEAD'].values())
    assert contracts['DINO_position_functions_byte_equal_to_vendor_HEAD']
    assert all(v['conversion_dimensions'] == 'images.shape[-2:]' for v in contracts['capture_conversion'].values())
    assert not contracts['camera_head_has_explicit_image_size_argument']


@pytest.mark.parametrize('bad', [0., np.pi, np.nan, -1.])
def test_invalid_FOV_is_not_a_positive_focal_success(audit, bad):
    pose = np.zeros((1, 9)); pose[:, 7:], pose[:, 7] = 1., bad
    with pytest.raises(ValueError):
        audit.decoded_intrinsics(pose, [84, 112])
