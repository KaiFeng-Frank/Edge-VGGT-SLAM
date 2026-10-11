"""Synthetic-only guards; never install packages or execute a neural graph."""
import importlib.util
import io
from pathlib import Path
import tarfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE if (HERE / "preflight-v35-arm-toolchain.py").is_file() else (
    HERE.parent / "deploy/rk3588/validation/board_ptq")


def load(name):
    spec = importlib.util.spec_from_file_location("_test_" + name.replace("-", "_"), SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = load("preflight-v35-arm-toolchain")
COMPILE = load("compile-v35-dpvo-int8-board")
INSTALL = load("install-v35-arm-toolchain-board")


def metadata():
    plan = {"schema": "v35_ARM64_RKNN_2_3_2_dependency_plan",
            "packages": [{"name": "torch", "version": "2.2.0"}]}
    kwargs = dict(machine="aarch64", system="Linux", compatible=b"vendor,board\0rockchip,rk3588s\0",
                  python_version=(3, 11, 8), prefix="/synthetic/new", base_prefix="/synthetic/base",
                  config_text="include-system-site-packages = false\n", libc=("glibc", "2.35"),
                  installed={"torch": "2.2.0+cpu", "rknn-toolkit2": "2.3.2"})
    return plan, kwargs


def test_preflight_synthetic_metadata_ready_without_package_imports():
    plan, kwargs = metadata()
    result = PREFLIGHT.evaluate_environment(plan, **kwargs)
    assert result["status"] == "ready_for_board_PTQ_diagnostic"
    assert result["model_inference_executed"] is False


@pytest.mark.parametrize("change", [
    {"machine": "x86_64"}, {"system": "Darwin"}, {"compatible": b"rockchip,rk3568\0"},
    {"compatible": b""}, {"python_version": (3, 10)}, {"prefix": "/synthetic/base"},
    {"config_text": "include-system-site-packages = true"}, {"libc": ("glibc", "2.17")},
    {"libc": ("", "")}, {"installed": {"torch": "2.3.1", "rknn-toolkit2": "2.3.2"}},
])
def test_preflight_rejects_wrong_board_or_dependency_metadata(change):
    plan, kwargs = metadata()
    kwargs.update(change)
    result = PREFLIGHT.evaluate_environment(plan, **kwargs)
    assert result["status"] == "blocked" and result["problems"]


@pytest.mark.parametrize("module", [COMPILE, INSTALL])
@pytest.mark.parametrize("compatible", [b"rockchip,rk3588\0", b"vendor,board\0rockchip,rk3588s\0"])
def test_platform_guard_allows_only_declared_board_family(module, compatible):
    module.board_guard(machine="aarch64", system="Linux", compatible=compatible)
    with pytest.raises(ValueError):
        module.board_guard(machine="aarch64", system="Linux", compatible=b"arm,other-host\0")
    with pytest.raises(ValueError):
        module.board_guard(machine="aarch64", system="Darwin", compatible=compatible)


def test_compile_host_guard_precedes_factory_and_output_side_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(COMPILE.platform, "machine", lambda: "x86_64")
    args = []
    for name in ("dependency-plan", "preflight-script", "frozen-sparse-compiler", "rewrite-dir",
                 "reference-dir", "parent-onnx", "editor", "dataset", "output-dir"):
        args += ["--" + name, str(tmp_path / name)]
    for name in ("dependency-plan", "preflight-script", "dataset"):
        args += ["--" + name + "-sha256", "0" * 64]
    with pytest.raises(ValueError, match="Host PTQ prohibited"):
        COMPILE.main(args)
    assert list(tmp_path.iterdir()) == []


def test_installer_host_guard_precedes_file_checks_network_and_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(INSTALL.platform, "machine", lambda: "x86_64")
    args = []
    for name in ("dependency-plan", "base-python", "sdk-wheel", "gcc-sysroot", "venv-path", "output-dir",
                 "preflight-script"):
        args += ["--" + name, str(tmp_path / name)]
    args += ["--sdk-wheel-sha256", "0" * 64, "--preflight-script-sha256", "0" * 64, "--download-wheels"]
    with pytest.raises(ValueError, match="ARM64 board"):
        INSTALL.main(args)
    assert list(tmp_path.iterdir()) == []


def calibration_arrays():
    image = np.zeros((1, 3, 464, 608), dtype=np.float32)
    indices = np.array([153, 17477, 155, 153] * 24, dtype=np.int64)
    return image, indices


def test_integer_calibration_order_and_repeats_preserved(tmp_path):
    image, indices = calibration_arrays()
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "indices.npy", indices)
    listing = tmp_path / "calibration.txt"
    listing.write_text("image.npy indices.npy\n")
    output = tmp_path / "new"
    output.mkdir()
    pins = {}
    before = indices.tobytes()
    resolved, rows = COMPILE.prepare_dataset(listing, COMPILE.sha(listing), output, pins)
    assert rows[0]["indices_dtype"] == "int64" and rows[0]["index_cast_performed"] is False
    assert np.load(tmp_path / "indices.npy", allow_pickle=False).tobytes() == before
    assert resolved.read_text().split() == [str(tmp_path / "image.npy"), str(tmp_path / "indices.npy")]


@pytest.mark.parametrize("mutation", ["float_indices", "edge_center", "wrong_image_shape", "nonfinite_image", "raw255"])
def test_calibration_contract_rejects_bad_inputs(mutation):
    image, indices = calibration_arrays()
    if mutation == "float_indices":
        indices = indices.astype(np.float32)
    elif mutation == "edge_center":
        indices[0] = 0
    elif mutation == "wrong_image_shape":
        image = image[0]
    elif mutation == "nonfinite_image":
        image[0, 0, 0, 0] = np.nan
    elif mutation == "raw255":
        image[0, 0, 0, 0] = 255
    with pytest.raises(ValueError):
        COMPILE.validate_calibration_arrays(image, indices)


class FakeRKNN:
    def __init__(self, output, *, build_error=False, release_error=False):
        self.output, self.calls = output, []
        self.build_error, self.release_error = build_error, release_error
        self.released = False

    def config(self, **kwargs):
        self.calls.append(("config", kwargs))
        return 0

    def load_onnx(self, **kwargs):
        self.calls.append(("load", kwargs))
        return 0

    def build(self, **kwargs):
        self.calls.append(("build", kwargs))
        if self.build_error:
            raise RuntimeError("synthetic primary build failure")
        return 0

    def export_rknn(self, path):
        Path(path).write_bytes(b"synthetic artifact, not RKNN")
        return 0

    def release(self):
        self.released = True
        if self.release_error:
            raise RuntimeError("synthetic release failure")


def test_fake_ptq_config_keeps_integer_inputs_and_identity_preprocessing(tmp_path, monkeypatch):
    monkeypatch.setattr(COMPILE, "board_guard", lambda: None)
    fake = FakeRKNN(tmp_path)
    report = dict(operations=[], released=False, cleanup_errors=[])
    COMPILE.perform_ptq(lambda **kw: fake, "original-sparse.onnx", "dataset.txt", tmp_path, report)
    assert fake.calls[0][1]["mean_values"] is None and fake.calls[0][1]["std_values"] is None
    assert fake.calls[0][1]["quant_img_RGB2BGR"] is False
    assert fake.calls[1] == ("load", {"model": "original-sparse.onnx"})  # no frozen inputs or casts
    assert fake.calls[2] == ("build", {"do_quantization": True, "dataset": "dataset.txt", "auto_hybrid": False})
    assert report["released"] and fake.released


def test_fake_cleanup_does_not_hide_primary_build_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(COMPILE, "board_guard", lambda: None)
    fake = FakeRKNN(tmp_path, build_error=True, release_error=True)
    report = dict(operations=[], released=False, cleanup_errors=[])
    with pytest.raises(RuntimeError, match="primary build failure"):
        COMPILE.perform_ptq(lambda **kw: fake, "graph", "dataset", tmp_path, report)
    assert fake.released and report["cleanup_errors"]


def test_fake_success_followed_by_release_failure_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(COMPILE, "board_guard", lambda: None)
    fake = FakeRKNN(tmp_path, release_error=True)
    report = dict(operations=[], released=False, cleanup_errors=[])
    with pytest.raises(RuntimeError, match="release failure"):
        COMPILE.perform_ptq(lambda **kw: fake, "graph", "dataset", tmp_path, report)
    assert fake.released and not report["released"]


def test_native_build_environment_is_isolated_and_bounded(tmp_path):
    environment = INSTALL.build_environment(tmp_path / "new", tmp_path / "gcc", tmp_path / "work",
                                            {"PATH": "/original/bin", "PYTHONPATH": "/old/site"})
    assert "PYTHONPATH" not in environment and environment["PYTHONNOUSERSITE"] == "1"
    assert environment["MAX_JOBS"] == "2"
    assert environment["ONNX_OPT_USE_SYSTEM_PROTOBUF"] == "0"
    assert "Ninja" in environment["CMAKE_ARGS"] and "-Dpybind11_DIR=" in environment["CMAKE_ARGS"]
    assert "--sysroot=" in environment["CXXFLAGS"] and environment["CC"].endswith("gcc-11")


@pytest.mark.parametrize("path,kind", [("../escape", "file"), ("/absolute", "file"),
                                       ("onnxoptimizer-0.3.8/link", "symlink")])
def test_source_archive_refuses_traversal_or_links(tmp_path, path, kind):
    archive = tmp_path / "toy.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(path)
        if kind == "symlink":
            member.type, member.linkname = tarfile.SYMTYPE, "/outside"
            stream.addfile(member)
        else:
            member.size = 1
            stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        INSTALL.safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "escape").exists()
