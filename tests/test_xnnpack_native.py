"""Small real-C-API checks on the deployed ABI; no model weights or downloads."""
import platform

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in {"aarch64", "arm64"}
    or torch.__version__.split("+")[0] != "2.3.1",
    reason="Native XNNPACK checks require the deployed ARM Torch 2.3.1 ABI",
)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_native_tail_strides_owned_output_and_restore(dtype, noncontiguous):
    from vggt_slam.xnnpack_linear import install_xnnpack_linear

    torch.manual_seed(382)
    original = torch.nn.Linear(133, 129).eval().requires_grad_(False)
    model = torch.nn.Sequential(original).eval().requires_grad_(False)
    inputs = torch.randn(3, 133, 17).transpose(1, 2)
    if not noncontiguous:
        inputs = inputs.contiguous()
    with torch.inference_mode():
        reference = model(inputs)
        handle = install_xnnpack_linear(model, threads=2, dtype=dtype)
        try:
            actual = model(inputs)
            saved = actual.clone()
            second = model(inputs * 0.5)
            assert actual.shape == reference.shape
            assert actual.dtype == torch.float32
            assert torch.isfinite(actual).all() and torch.isfinite(second).all()
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            relative = (actual - reference).norm() / reference.norm()
            assert float(relative) < (2e-6 if dtype == "float32" else 0.02)
            assert handle.metrics()["xnnpack_calls"] == 2
        finally:
            handle.restore()
            handle.close()
        assert model[0] is original
        assert handle.metrics()["restored"] and handle.metrics()["closed"]
        torch.testing.assert_close(model(inputs), reference, rtol=0, atol=0)
