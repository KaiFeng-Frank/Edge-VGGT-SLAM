"""Synthetic pixel/projection checks; no dataset images or model inference."""

import copy
import json

import cv2
import numpy as np
import pytest

from vggt_slam.image_rectification import CalibrationRectifier


def calibration(distortion=None):
    return dict(K=[[52., 0., 31.5], [0., 54., 23.5], [0., 0., 1.]],
                distortion=[0.] * 5 if distortion is None else distortion,
                image_hw=[48, 64], source="synthetic_pinhole_fixture")


def image():
    y, x = np.indices((48, 64))
    return np.stack((x, y, (x * 3 + y * 5) % 256), axis=-1).astype(np.uint8)


def test_identity_keeps_pixels_dimensions_axes_and_owns_output():
    rectifier = CalibrationRectifier(calibration())
    frame = image()
    before = frame.copy()
    result = rectifier.rectify_bgr(frame)
    assert np.array_equal(result, before)
    assert np.array_equal(frame, before)
    assert result.shape == (48, 64, 3) and result.dtype == np.uint8
    assert not np.shares_memory(frame, result)
    metadata = rectifier.metadata()
    assert metadata["output_K"] == metadata["K"]
    assert metadata["output_distortion"] == [0.] * 5
    assert metadata["R"] == np.eye(3).tolist()
    assert metadata["crop"] is None
    assert metadata["map_bytes"] == 48 * 64 * 2 * 4
    json.dumps(metadata, allow_nan=False)


def test_distorted_projection_matches_inverse_remap_and_bgr_pixels():
    config = calibration([.2, -.04, .006, -.008, .01])
    rectifier = CalibrationRectifier(config)
    K, D = np.asarray(config["K"]), np.asarray(config["distortion"])
    uv = np.array([[15, 12], [25, 29], [47, 34], [39, 17]], np.int32)
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T
    distorted = cv2.projectPoints(rays * 3., np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    x, y = uv.T
    # Destination pixels use the unchanged K. Their source locations must be
    # the independently projected distorted measurements of the same rays.
    actual = np.column_stack((rectifier._map_x[y, x], rectifier._map_y[y, x]))
    assert np.allclose(actual, distorted, atol=2e-6, rtol=0)
    result = rectifier.rectify_bgr(image())
    assert np.allclose(result[y, x, :2], distorted, atol=.6, rtol=0)
    assert not np.array_equal(result, image())


def test_outside_image_samples_have_constant_zero_border():
    rectifier = CalibrationRectifier(calibration([2., 0., 0., 0., 0.]))
    result = rectifier.rectify_bgr(np.full((48, 64, 3), 127, np.uint8))
    assert np.array_equal(result[0, 0], [0, 0, 0])
    assert np.array_equal(result[-1, -1], [0, 0, 0])
    assert np.array_equal(result[24, 32], [127, 127, 127])


def test_decode_and_direct_rectification_have_exact_shared_pixels(tmp_path):
    config = calibration([.2, -.04, .006, -.008, .01])
    path = tmp_path / "synthetic_rgb.png"
    original = image()
    assert cv2.imwrite(str(path), original)
    raw_bytes = path.read_bytes()
    reader, mapper = CalibrationRectifier(config), CalibrationRectifier(config)
    direct = reader.rectify_bgr(original)
    assert np.array_equal(reader.decode_bgr(path), direct)
    assert np.array_equal(mapper.decode_bgr(path), direct)
    assert path.read_bytes() == raw_bytes
    assert reader.metadata() == mapper.metadata()


def test_maps_are_precomputed_once_and_configuration_is_copied(monkeypatch):
    original = cv2.initUndistortRectifyMap
    calls = []

    def capture(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(cv2, "initUndistortRectifyMap", capture)
    config = calibration()
    rectifier = CalibrationRectifier(config)
    metadata = rectifier.metadata()
    config["K"][0][0] = 999.
    config["distortion"][0] = 99.
    config["image_hw"][0] = 99
    metadata["K"][0][0] = -1
    metadata["output_distortion"][0] = 99
    rectifier.rectify_bgr(image())
    rectifier.rectify_bgr(image())
    assert len(calls) == 1
    assert rectifier.metadata()["K"][0][0] == 52.
    assert rectifier.metadata()["distortion"][0] == 0.
    assert rectifier.metadata()["output_distortion"][0] == 0.
    for table in (rectifier._map_x, rectifier._map_y):
        assert table.dtype == np.float32 and not table.flags.writeable
    # Only small calibration arrays and two fixed tables survive calls.
    arrays = [v for v in vars(rectifier).values() if isinstance(v, np.ndarray)]
    assert sum(v.nbytes for v in arrays) == 48 * 64 * 8 + 9 * 8 * 2 + 5 * 8


@pytest.mark.parametrize("length", [4, 5, 8, 12, 14])
def test_all_supported_zero_distortion_vectors_preserve_identity(length):
    rectifier = CalibrationRectifier(calibration([0.] * length))
    assert np.array_equal(rectifier.rectify_bgr(image()), image())
    assert rectifier.metadata()["output_distortion"] == [0.] * length


@pytest.mark.parametrize("replacement", [
    {"K": np.eye(2)}, {"K": [[0., 0., 31.5], [0., 54., 23.5], [0., 0., 1.]]},
    {"K": [[52., 0., 31.5], [0., -54., 23.5], [0., 0., 1.]]},
    {"K": [[52., .1, 31.5], [0., 54., 23.5], [0., 0., 1.]]},
    {"K": [[52., 0., 31.5], [.1, 54., 23.5], [0., 0., 1.]]},
    {"K": [[52., 0., 31.5], [0., 54., 23.5], [0., 1., 1.]]},
    {"K": [[float("nan"), 0., 0.], [0., 54., 0.], [0., 0., 1.]]},
    {"K": np.eye(3, dtype=complex)}, {"K": [["52", 0., 0.], [0., 54., 0.], [0., 0., 1.]]},
    {"K": [[True, 0., 0.], [0., 54., 0.], [0., 0., 1.]]},
    {"distortion": [0.] * 3}, {"distortion": np.zeros((5, 1))},
    {"distortion": [float("inf")] + [0.] * 4}, {"distortion": [False] * 5},
    {"distortion": ["0"] * 5}, {"distortion": None},
    {"image_hw": [0, 64]}, {"image_hw": [48, -1]}, {"image_hw": [48., 64]},
    {"image_hw": [True, 64]}, {"image_hw": [48]}, {"image_hw": None},
    {"source": " "}, {"source": None}, {"source": 42},
])
def test_invalid_calibration_is_rejected(replacement):
    config = calibration()
    config.update(replacement)
    with pytest.raises(ValueError, match="Calibration"):
        CalibrationRectifier(config)


@pytest.mark.parametrize("bad", [None, [], {}, {"unknown": 1}])
def test_calibration_requires_exact_schema(bad):
    with pytest.raises(ValueError, match="exactly"):
        CalibrationRectifier(bad)


def test_extra_calibration_key_and_missing_key_are_rejected():
    extra = calibration()
    extra["alpha"] = 1.
    missing = calibration()
    missing.pop("distortion")
    for config in (extra, missing):
        with pytest.raises(ValueError, match="exactly"):
            CalibrationRectifier(config)


@pytest.mark.parametrize("frame", [None, [], np.zeros((48, 64), np.uint8),
    np.zeros((48, 64, 4), np.uint8), np.zeros((64, 48, 3), np.uint8),
    np.zeros((48, 64, 3), np.float32), np.zeros((48, 64, 3), np.int16)])
def test_frame_type_channels_size_and_dtype_are_explicit(frame):
    with pytest.raises(ValueError, match="uint8 HxWx3 BGR"):
        CalibrationRectifier(calibration()).rectify_bgr(frame)


def test_unreadable_or_wrong_size_decoded_image_fails(tmp_path, monkeypatch):
    rectifier = CalibrationRectifier(calibration())
    unreadable = tmp_path / "not_an_image.dat"
    unreadable.write_bytes(b"synthetic invalid image")
    with pytest.raises(OSError, match="could not decode"):
        rectifier.decode_bgr(unreadable)
    monkeypatch.setattr(cv2, "imread", lambda path, flags: np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError, match="matching calibration"):
        rectifier.decode_bgr("synthetic_wrong_size.png")


def test_invalid_maps_and_opencv_map_errors_fail_before_use(monkeypatch):
    monkeypatch.setattr(cv2, "initUndistortRectifyMap",
                        lambda *a: (np.full((48, 64), np.inf, np.float32),
                                    np.zeros((48, 64), np.float32)))
    with pytest.raises(ValueError, match="invalid rectification maps"):
        CalibrationRectifier(calibration())

    def fail(*args):
        raise cv2.error("synthetic invalid camera")

    monkeypatch.setattr(cv2, "initUndistortRectifyMap", fail)
    with pytest.raises(ValueError, match="could not build"):
        CalibrationRectifier(calibration())
