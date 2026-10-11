"""Synthetic-only calibration policy, pixel and random-index checks."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE if (HERE / "generate-v36-dpvo-calibration-board.py").is_file() else (
    HERE.parent / "deploy/rk3588/validation/board_calibration")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = load(SOURCE / "generate-v36-dpvo-calibration-board.py", "_v36_calibration")


@pytest.fixture
def encoder():
    candidates = [Path(os.environ["V30_FRAME_ENCODER"])] if "V30_FRAME_ENCODER" in os.environ else [
        SOURCE.parent / "dpvo_stream/v30_frame_encoder.py",
        Path("/dev/shm/vggt-v30-dpvo-stream/v30_frame_encoder.py")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        pytest.skip("requires pinned v30_frame_encoder.py source, set V30_FRAME_ENCODER")
    assert GENERATOR.sha(path) == GENERATOR.ENCODER_SHA
    return load(path, "_v36_synthetic_frozen_encoder")  # No ORT session/model is created.


def synthetic_rgb_text():
    return "# synthetic metadata only\n" + "\n".join(
        "%d.000000 rgb/%d.000000.png" % (1000 + i, 1000 + i) for i in range(798))


def test_predeclared_32_source_indices_disjoint_from_first192():
    policy = GENERATOR.declared_policy()
    rows = GENERATOR.parse_rgb_index(synthetic_rgb_text())
    chosen = GENERATOR.select_rows(rows, policy)
    assert [r["source_index"] for r in chosen] == [256 + 16 * i for i in range(32)]
    assert chosen[-1]["source_index"] == 752
    assert not set(r["source_index"] for r in chosen).intersection(range(192))
    assert policy["GT_reading_allowed"] is False and policy["independent_scene_claim"] is False


def test_changed_policy_or_truncated_source_is_rejected():
    policy = GENERATOR.declared_policy()
    rows = GENERATOR.parse_rgb_index(synthetic_rgb_text())
    with pytest.raises(ValueError, match="population"):
        GENERATOR.select_rows(rows[:-1], policy)
    policy["calibration_source_indices"][0] = 191
    with pytest.raises(ValueError, match="policy"):
        GENERATOR.select_rows(rows, policy)


@pytest.mark.parametrize("text", [
    "1.000000 /rgb/1.000000.png", "1.000000 rgb/../1.000000.png", "1.000000 depth/1.000000.png",
    "1.000000 rgb/2.000000.png", "nan rgb/nan.png", "2.000000 rgb/2.000000.png\n1.000000 rgb/1.000000.png",
])
def test_rgb_parser_refuses_paths_wrong_identity_or_bad_timestamps(text):
    with pytest.raises(ValueError):
        GENERATOR.parse_rgb_index(text)


class FakeCV2:
    def __init__(self):
        self.calls = []

    def undistort(self, bgr, K, D):
        self.calls.append((K.copy(), D.copy()))
        return bgr.copy()


def test_synthetic_pixel_crop_and_once_normalization_keep_bgr_order(encoder):
    bgr = np.zeros((480, 640, 3), np.uint8)
    bgr[:, :, 1] = 127
    bgr[:, :, 2] = 255
    bgr[8, 16] = [11, 22, 33]
    cv = FakeCV2()
    cropped = GENERATOR.preprocess_bgr(bgr, cv)
    assert cropped.shape == (464, 608, 3) and cropped.flags.c_contiguous
    np.testing.assert_array_equal(cropped[0, 0], [11, 22, 33])
    np.testing.assert_array_equal(cv.calls[0][0], GENERATOR.declared_policy()["original_K"])
    np.testing.assert_array_equal(cv.calls[0][1], GENERATOR.declared_policy()["original_D"])
    normalized = encoder.normalize_bgr(cropped)
    # Follow the original div->multiply->subtract order (not an alternative expression).
    expected = np.float32(2) * (np.array([11, 22, 33], np.float32) / np.float32(255)) - np.float32(.5)
    assert normalized.dtype == np.float32 and normalized.shape == (1, 3, 464, 608)
    assert normalized[0, :, 0, 0].tobytes() == expected.tobytes()
    assert normalized[0, 0, 1, 1] == -.5 and normalized[0, 2, 1, 1] == 1.5


def test_wrong_decoded_dtype_or_size_rejected():
    for image in [np.zeros((480, 640, 3), np.float32), np.zeros((464, 608, 3), np.uint8)]:
        with pytest.raises(ValueError, match="Original decoded"):
            GENERATOR.preprocess_bgr(image, FakeCV2())


def test_original_cpu_random_x_then_y_then_depth_consumption(encoder):
    import torch
    actual = torch.Generator(device="cpu").manual_seed(0)
    reference = torch.Generator(device="cpu").manual_seed(0)
    previous_state = None
    for _ in range(2):
        indices, draws = GENERATOR.sample_indices(encoder, actual, torch)
        initial = reference.get_state().numpy().copy()
        x = torch.randint(1, 151, (1, 96), generator=reference, device="cpu")
        y = torch.randint(1, 115, (1, 96), generator=reference, device="cpu")
        after_xy = reference.get_state().numpy().copy()
        depth = torch.rand((96,), dtype=torch.float32, generator=reference, device="cpu").numpy()
        expected_indices = (y[0].numpy() * np.int64(152) + x[0].numpy())
        assert indices.dtype == np.int64 and indices.tobytes() == expected_indices.tobytes()
        assert draws["consumed_inverse_depth"].tobytes() == depth.tobytes()
        assert draws["state_before"].tobytes() == initial.tobytes()
        assert draws["state_after_centers"].tobytes() == after_xy.tobytes()
        assert draws["state_after_depth"].tobytes() == reference.get_state().numpy().tobytes()
        if previous_state is not None:
            assert draws["state_before"].tobytes() == previous_state
        previous_state = draws["state_after_depth"].tobytes()


def test_source_path_cannot_escape_through_symlink(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"synthetic path test, not an image")
    (root / "link.png").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        GENERATOR.checked_source(root, "link.png")


def test_host_guard_precedes_pixel_loading_and_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(GENERATOR.platform, "machine", lambda: "x86_64")
    args = []
    for name in ("dataset-dir", "rgb-index", "encoder-helper", "policy", "output-dir"):
        args += ["--" + name, str(tmp_path / name)]
    args += ["--policy-sha256", "0" * 64]
    with pytest.raises(ValueError, match="board-only"):
        GENERATOR.main(args)
    assert list(tmp_path.iterdir()) == []


def test_predeclared_pair_array_size_fits_256mib():
    raw = 32 * (1 * 3 * 464 * 608 * 4 + 96 * 8)
    assert raw == 108355584 and raw < GENERATOR.BUDGET
