"""Synthetic integration checks; never load dataset images or real models."""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_lossless_mapper_bridge_preserves_capture_pixels_and_order(monkeypatch):
    from PIL import Image
    from vggt_slam.edge_mapping import prepare_rectified_images

    a = np.arange(18*24*3, dtype=np.uint16).reshape(18, 24, 3).astype(np.uint8)
    b = 255-a
    frames = {'a.png': a, 'b.png': b}
    opened = []
    def load(streams, image_size):
        assert image_size == 112
        opened.extend(streams)
        return [np.array(Image.open(s).convert('RGB')) for s in streams]
    monkeypatch.setitem(sys.modules, 'vggt.utils.load_fn', SimpleNamespace(load_and_preprocess_images=load))
    out = prepare_rectified_images(['b.png', 'a.png'], SimpleNamespace(decode_bgr=frames.__getitem__), 112)
    np.testing.assert_array_equal(out[0], b[..., ::-1])
    np.testing.assert_array_equal(out[1], a[..., ::-1])
    assert all(s.closed for s in opened)
    assert all(np.array_equal(frames[k], v) for k, v in [('a.png', a), ('b.png', b)])


def test_mapper_bridge_releases_streams_after_preprocessing_failure(monkeypatch):
    from vggt_slam.edge_mapping import prepare_rectified_images
    opened = []
    def fail(streams, image_size):
        opened.extend(streams)
        raise RuntimeError('synthetic preprocessing failure')
    monkeypatch.setitem(sys.modules, 'vggt.utils.load_fn', SimpleNamespace(load_and_preprocess_images=fail))
    with pytest.raises(RuntimeError, match='preprocessing failure'):
        prepare_rectified_images(['a'], SimpleNamespace(decode_bgr=lambda _: np.zeros((8, 8, 3), np.uint8)), 14)
    assert opened and all(s.closed for s in opened)


@pytest.fixture
def prepared_solver():
    """Execute the real method with synthetic model/graph services only."""
    torch = pytest.importorskip('torch')
    from vggt_slam import model_execution as execution
    import time
    path = Path(__file__).resolve().parents[1] / 'vggt_slam/solver.py'
    parsed = ast.parse(path.read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Solver')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run_predictions')
    state = SimpleNamespace(inferred=[], names=[])
    class Submap:
        def __init__(self, node):
            self.node = node
        def add_all_frames(self, images):
            self.frames = images
        def set_frame_ids(self, names):
            self.frame_ids = [float(Path(p).stem) for p in names]
        def set_last_non_loop_frame_index(self, index):
            self.last_index = index
        def set_img_names(self, names):
            self.names = list(names)
            state.names.append(list(names))
    class Model:
        def parameters(self):
            yield torch.zeros(1)
        def __call__(self, images, compute_similarity=False):
            state.inferred.append(images)
            n, _, h, w = images.shape
            return {'pose_enc': torch.zeros((1, n, 9)),
                    'depth': (images[:, 0] + 1)[None, ..., None],
                    'depth_conf': torch.ones((1, n, h, w)),
                    'image_match_ratio': .96 if compute_similarity else None}
    def convert(pose, hw):
        n = pose.shape[1]
        return torch.eye(4)[None, None, :3].repeat(1, n, 1, 1), torch.eye(3)[None, None].repeat(1, n, 1, 1)
    def forbidden_read(*args, **kwargs):
        pytest.fail('Prepared images must not reread original files')
    def adapt(model):
        if hasattr(model, 'capabilities'):
            return execution.as_executor(model)
        return execution.TorchVGGTExecutor(model, pose_decoder=convert)
    ns = dict(torch=torch, np=np, time=time, Submap=Submap, DEBUG=False,
              as_executor=adapt, execute=execution.execute,
              prepare_rgb_images=execution.prepare_rgb_images,
              get_execution_device=execution.get_execution_device)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    self = SimpleNamespace(image_size=14, max_loops=0,
                           map=SimpleNamespace(get_largest_key=lambda: None),
                           vggt_timer=nullcontext())
    return ns['run_predictions'], self, Model(), state, torch


def test_prepared_pixels_keep_original_ids_and_feed_only_original_model(prepared_solver):
    run, owner, model, state, torch = prepared_solver
    images = torch.linspace(0, 1, 2*3*14*14).reshape(2, 3, 14, 14)
    names = ['/synthetic/101.25.png', '/synthetic/103.50.png']
    result = run(owner, names, model, prepared_images=images)
    # The execution boundary changes the Python container, not its prepared
    # CPU pixel storage or ordering. The submap retains those same pixels.
    assert state.inferred[0].data_ptr() == images.data_ptr()
    assert owner.current_working_submap.frames.data_ptr() == images.data_ptr()
    assert np.shares_memory(result['images'], images.numpy())
    np.testing.assert_array_equal(result['images'], images.numpy())
    np.testing.assert_array_equal(result['depth'][..., 0], images.numpy()[:, 0] + 1)
    assert owner.current_working_submap.frame_ids == [101.25, 103.50]
    assert state.names == [names] and result['detected_loops'] == []


@pytest.mark.parametrize('bad', ['count', 'dtype', 'nan', 'range', 'shape', 'width', 'auxiliary', 'device'])
def test_invalid_prepared_images_cannot_enter_model(prepared_solver, bad):
    run, owner, model, state, torch = prepared_solver
    images = torch.zeros((2, 3, 14, 14))
    auxiliary = None
    if bad == 'count': images = images[:1]
    if bad == 'dtype': images = images.to(torch.float64)
    if bad == 'nan': images[0, 0, 0, 0] = float('nan')
    if bad == 'range': images[0, 0, 0, 0] = 1.1
    if bad == 'shape': images = images[None]
    if bad == 'width': images = torch.zeros((2, 3, 14, 28))
    if bad == 'auxiliary': auxiliary = object()
    if bad == 'device': images = torch.empty((2, 3, 14, 14), device='meta')
    with pytest.raises(ValueError):
        run(owner, ['101.png', '102.png'], model, 0, auxiliary, None, prepared_images=images)
    assert state.inferred == []


@pytest.mark.parametrize('config', [
    {'max_loops': 1}, {'max_loops': True},
    {'max_loops': 0, 'auxiliary_model': 'unused'},
    {'max_loops': 0, 'unsupported_switch': False},
])
def test_mapping_rejects_before_affinity_imports_or_model_allocation(monkeypatch, config):
    import builtins
    import queue
    from vggt_slam import edge_mapping
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name in ('torch', 'cv2', 'vggt_slam.runtime', 'vggt_slam.solver'):
            pytest.fail('Invalid configuration imported model/graph services')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(edge_mapping.os, 'sched_setaffinity',
                        lambda *args: pytest.fail('Invalid configuration changed affinity'))
    messages = queue.Queue()
    edge_mapping.mapping_worker(config, queue.Queue(), messages)
    terminal = messages.get_nowait()
    assert terminal['kind'] == 'error' and 'ValueError' in terminal['error']
    assert messages.empty()
