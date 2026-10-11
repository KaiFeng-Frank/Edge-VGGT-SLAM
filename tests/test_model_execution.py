"""Synthetic execution contract tests: no dataset or original-model inference."""
import builtins
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from vggt_slam import model_execution as execution


def images(count=2):
    return np.linspace(0, 1, count * 3 * 14 * 28, dtype=np.float32).reshape(count, 3, 14, 28)


def prediction(pixels, ratio=None):
    n, _, h, w = pixels.shape
    return {'extrinsic': np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1)),
            'intrinsic': np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            'depth': np.ones((n, h, w, 1), np.float32),
            'depth_conf': np.ones((n, h, w), np.float32),
            'image_match_ratio': ratio}


class NumpyExecutor:
    capabilities = execution.ExecutionCapabilities('synthetic-numpy', True)

    def __init__(self):
        self.calls = []
        self.last = None

    def predict(self, pixels, *, compute_similarity=False):
        self.calls.append((pixels, compute_similarity))
        self.last = prediction(pixels, .95 if compute_similarity else None)
        return self.last

    @property
    def parameters(self):
        raise AssertionError('NumPy executor must not be treated as a Torch model')


def test_numpy_executor_needs_no_torch_parameters_or_pose_codec(monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'torch' or name.startswith('vggt.'):
            raise AssertionError('NumPy execution must not import a model or codec')
        return original_import(name, *args, **kwargs)
    backend = NumpyExecutor()
    pixels = images()
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert execution.as_executor(backend) is backend
    out = execution.execute(backend, pixels)
    assert out['images'] is pixels
    assert out['depth'] is backend.last['depth']  # no unconditional large copy
    assert out['image_match_ratio'] is None
    assert execution.get_execution_device(backend) == 'cpu'


def test_owned_result_remains_stable_after_a_subsequent_call():
    backend = NumpyExecutor()
    first = execution.execute(backend, images(1))
    execution.execute(backend, images(3))
    assert first['depth'].shape == (1, 14, 28, 1)
    np.testing.assert_array_equal(first['depth'], 1)


def test_loop_capability_rejects_before_execution():
    backend = NumpyExecutor()
    backend.capabilities = execution.ExecutionCapabilities('no-loop', False)
    with pytest.raises(execution.UnsupportedExecutionError):
        execution.execute(backend, images(), compute_similarity=True)
    assert backend.calls == []


@pytest.mark.parametrize('bad', ['dtype', 'batch', 'channel', 'grid', 'contiguous', 'nan', 'range'])
def test_bad_request_rejected_before_execution(bad):
    pixels = images()
    if bad == 'dtype': pixels = pixels.astype(np.float64)
    if bad == 'batch': pixels = pixels[None]
    if bad == 'channel': pixels = pixels[:, :2]
    if bad == 'grid': pixels = pixels[..., :13, :]
    if bad == 'contiguous': pixels = pixels[..., ::-1]
    if bad == 'nan': pixels.flat[0] = np.nan
    if bad == 'range': pixels.flat[0] = 1.01
    backend = NumpyExecutor()
    with pytest.raises(ValueError):
        execution.execute(backend, pixels)
    assert backend.calls == []


@pytest.mark.parametrize('key', ['extrinsic', 'intrinsic', 'depth', 'depth_conf'])
@pytest.mark.parametrize('bad', ['missing', 'shape', 'dtype', 'nan'])
def test_canonical_output_shapes_and_finiteness_are_mandatory(key, bad):
    backend = NumpyExecutor()
    def malformed(pixels, **kwargs):
        out = prediction(pixels)
        if bad == 'missing': del out[key]
        if bad == 'shape': out[key] = out[key][None]
        if bad == 'dtype': out[key] = out[key].astype(np.float64)
        if bad == 'nan': out[key].flat[0] = np.nan
        return out
    backend.predict = malformed
    with pytest.raises(ValueError, match=key):
        execution.execute(backend, images())


@pytest.mark.parametrize('ratio', [None, np.nan, np.inf, [0, 1], True])
def test_loop_ratio_is_not_inferred_from_opaque_tokens(ratio):
    backend = NumpyExecutor()
    backend.predict = lambda pixels, **kwargs: {**prediction(pixels, ratio), 'target_tokens': object()}
    with pytest.raises(ValueError, match='image_match_ratio'):
        execution.execute(backend, images(), compute_similarity=True)


def test_original_similarity_ratio_can_exceed_one():
    backend = NumpyExecutor()
    ratio = 1.0043401718139648  # observed original VGGT loop, not a probability
    backend.predict = lambda pixels, **kwargs: prediction(pixels, ratio)
    assert execution.execute(backend, images(), compute_similarity=True)['image_match_ratio'] == ratio


def test_no_fake_loop_ratio_in_normal_result_or_non_pair_loop():
    backend = NumpyExecutor()
    backend.predict = lambda pixels, **kwargs: prediction(pixels, .98)
    with pytest.raises(ValueError, match='Normal'):
        execution.execute(backend, images())
    with pytest.raises(ValueError, match='two-frame'):
        execution.execute(backend, images(3), compute_similarity=True)


def test_input_order_metadata_and_backend_exception_preserved():
    pixels = images()[::-1].copy()
    backend = NumpyExecutor()
    out = execution.execute(backend, pixels, compute_similarity=True)
    assert backend.calls[0] == (pixels, True)
    assert out['images'] is pixels
    backend.predict = lambda values, **kwargs: {**prediction(values), 'images': values[::-1].copy()}
    with pytest.raises(ValueError, match='frame order'):
        execution.execute(backend, pixels)
    failure = RuntimeError('synthetic hardware failure')
    def fail(*args, **kwargs): raise failure
    backend.predict = fail
    with pytest.raises(RuntimeError) as caught:
        execution.execute(backend, pixels)
    assert caught.value is failure


def test_original_preprocessor_and_prepared_numpy_storage(monkeypatch):
    pixels = images()
    calls = []
    def load(paths, image_size):
        calls.append((list(paths), image_size))
        return pixels
    monkeypatch.setitem(sys.modules, 'vggt.utils.load_fn', SimpleNamespace(load_and_preprocess_images=load))
    names = ['new.png', 'old.png']
    assert execution.prepare_rgb_images(names, 28) is pixels
    assert calls == [(names, 28)]
    assert execution.prepare_rgb_images(names, 28, prepared_images=pixels) is pixels
    assert calls == [(names, 28)]


def synthetic_torch_model(torch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = []
            self.tokens = object()
        def forward(self, pixels, compute_similarity=False):
            self.calls.append((pixels, compute_similarity, torch.is_grad_enabled()))
            n, _, h, w = pixels.shape
            return {'pose_enc': torch.zeros((1, n, 9)),
                    'depth': (pixels[:, 0] + 1)[None, ..., None],
                    'depth_conf': torch.ones((1, n, h, w)),
                    'target_tokens': self.tokens, 'images': pixels[None],
                    'image_match_ratio': torch.tensor(.97) if compute_similarity else None}
    return Model()


def decoder(torch):
    def decode(pose, hw):
        n = pose.shape[1]
        assert hw == (14, 28)
        return torch.eye(4)[None, None, :3].repeat(1, n, 1, 1), torch.eye(3)[None, None].repeat(1, n, 1, 1)
    return decode


@pytest.mark.parametrize('count', [1, 2, 3])
def test_torch_original_call_decoder_batch_strip_and_shared_pixels(count):
    torch = pytest.importorskip('torch')
    model = synthetic_torch_model(torch)
    backend = execution.TorchVGGTExecutor(model, pose_decoder=decoder(torch))
    pixels = images(count)
    out = execution.execute(backend, pixels)
    assert model.calls[0][0].data_ptr() == pixels.ctypes.data
    assert model.calls[0][1:] == (False, False)
    assert out['extrinsic'].shape == (count, 3, 4)
    assert out['pose_enc'].shape == (count, 9)
    assert out['target_tokens'] is model.tokens
    np.testing.assert_array_equal(out['depth'][..., 0], pixels[:, 0] + 1)
    assert execution.as_executor(backend) is backend
    assert execution.as_executor(model).capabilities.supports_compute_similarity


def test_torch_loop_uses_original_similarity_not_target_tokens():
    torch = pytest.importorskip('torch')
    model = synthetic_torch_model(torch)
    backend = execution.TorchVGGTExecutor(model, pose_decoder=decoder(torch))
    out = execution.execute(backend, images(), compute_similarity=True)
    assert model.calls[0][1:] == (True, False)
    assert out['image_match_ratio'] == pytest.approx(.97)


def test_unknown_kwargs_do_not_claim_similarity():
    torch = pytest.importorskip('torch')
    class Model(torch.nn.Module):
        def forward(self, pixels, **kwargs):
            pytest.fail('Missing loop capability must be rejected before forward')
    backend = execution.TorchVGGTExecutor(Model())
    assert not backend.capabilities.supports_compute_similarity
    with pytest.raises(execution.UnsupportedExecutionError):
        execution.execute(backend, images(), compute_similarity=True)


@pytest.mark.parametrize('bad', [SimpleNamespace(predict=lambda _: {}), SimpleNamespace(capabilities=True), object()])
def test_incomplete_executor_cannot_silently_become_legacy_model(bad):
    with pytest.raises(TypeError):
        execution.as_executor(bad)
