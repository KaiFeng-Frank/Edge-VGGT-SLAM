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
    import time
    path = Path(__file__).resolve().parents[1] / 'vggt_slam/solver.py'
    parsed = ast.parse(path.read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Solver')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run_predictions')
    state = SimpleNamespace(inferred=[], retrieval=[], names=[])
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
        def set_all_retrieval_vectors(self, vectors):
            self.vectors = vectors
    class Model:
        def parameters(self):
            yield torch.zeros(1)
        def __call__(self, images):
            state.inferred.append(images)
            return {'pose_enc': torch.zeros((1, len(images), 9))}
    class Retrieval:
        def get_all_submap_embeddings(self, submap):
            state.retrieval.append(submap.frames)
            return np.zeros((len(submap.frames), 1))
        def find_loop_closures(self, *a, **k):
            return []
    def convert(pose, hw):
        n = pose.shape[1]
        return torch.eye(4)[None, None, :3].repeat(1, n, 1, 1), torch.eye(3)[None, None].repeat(1, n, 1, 1)
    def forbidden_read(*args, **kwargs):
        pytest.fail('Prepared images must not reread original files')
    ns = dict(torch=torch, np=np, time=time, Submap=Submap, load_and_preprocess_images=forbidden_read,
              pose_encoding_to_extri_intri=convert, DEBUG=False)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    self = SimpleNamespace(image_size=14, max_loops=1, image_retrieval=Retrieval(),
                           map=SimpleNamespace(get_largest_key=lambda: None),
                           vggt_timer=nullcontext(), clip_timer=nullcontext(), loop_closure_timer=nullcontext(),
                           lc_thres=.8)
    return ns['run_predictions'], self, Model(), state, torch


def test_prepared_pixels_keep_original_ids_and_feed_model_and_retrieval(prepared_solver):
    run, owner, model, state, torch = prepared_solver
    images = torch.linspace(0, 1, 2*3*14*14).reshape(2, 3, 14, 14)
    names = ['/synthetic/101.25.png', '/synthetic/103.50.png']
    result = run(owner, names, model, 1, None, None, prepared_images=images)
    assert state.inferred[0] is images and state.retrieval[0] is images
    assert owner.current_working_submap.frame_ids == [101.25, 103.50]
    assert state.names == [names] and result['detected_loops'] == []


@pytest.mark.parametrize('bad', ['count', 'dtype', 'nan', 'range', 'shape', 'width', 'clip', 'device'])
def test_invalid_prepared_images_cannot_enter_model(prepared_solver, bad):
    run, owner, model, state, torch = prepared_solver
    images = torch.zeros((2, 3, 14, 14))
    clip = None
    if bad == 'count': images = images[:1]
    if bad == 'dtype': images = images.to(torch.float64)
    if bad == 'nan': images[0, 0, 0, 0] = float('nan')
    if bad == 'range': images[0, 0, 0, 0] = 1.1
    if bad == 'shape': images = images[None]
    if bad == 'width': images = torch.zeros((2, 3, 14, 28))
    if bad == 'clip': clip = object()
    if bad == 'device': images = torch.empty((2, 3, 14, 14), device='meta')
    with pytest.raises(ValueError):
        run(owner, ['101.png', '102.png'], model, 0, clip, None, prepared_images=images)
    assert state.inferred == [] and state.retrieval == []
