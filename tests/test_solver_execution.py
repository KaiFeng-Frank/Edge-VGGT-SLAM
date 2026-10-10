"""Real Solver method with synthetic NumPy execution/retrieval/graph services."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from vggt_slam import model_execution as execution


@pytest.fixture
def services():
    torch = pytest.importorskip('torch')
    source = Path(__file__).resolve().parents[1] / 'vggt_slam/solver.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Solver')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run_predictions')
    state = SimpleNamespace(submaps=[], retrieval=[], calls=[], loops=0, ratio=.95, bad=None)

    class Submap:
        def __init__(self, node):
            state.submaps.append(self)
            self.node = node
        def add_all_frames(self, frames): self.frames = frames
        def set_frame_ids(self, names): self.ids = [float(Path(name).stem) for name in names]
        def set_last_non_loop_frame_index(self, index): self.last_index = index
        def set_all_retrieval_vectors(self, values): self.retrieval = values
        def set_img_names(self, names): self.names = list(names)
        def get_frame_at_index(self, index): return self.frames[index]
        def get_img_names_at_index(self, index): return self.names[index]

    match = SimpleNamespace(query_submap_frame=1, detected_submap_id=0, detected_submap_frame=0)
    retrieved = torch.full((3, 14, 14), .2)
    class Retrieval:
        def get_all_submap_embeddings(self, submap):
            state.retrieval.append(submap.frames)
            return np.zeros((len(submap.frames), 1))
        def find_loop_closures(self, *args, **kwargs): return [match]

    class Executor:
        capabilities = execution.ExecutionCapabilities('numpy-test-no-parameters', True)
        def predict(self, pixels, *, compute_similarity=False):
            assert isinstance(pixels, np.ndarray)
            state.calls.append((pixels, compute_similarity))
            n, _, h, w = pixels.shape
            out = {'extrinsic': np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1)),
                   'intrinsic': np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
                   'depth': np.ones((n, h, w, 1), np.float32),
                   'depth_conf': np.ones((n, h, w), np.float32),
                   'image_match_ratio': state.ratio if compute_similarity else None}
            if state.bad == 'normal_shape' and not compute_similarity: out['depth'] = out['depth'][None]
            if state.bad == 'loop_nan' and compute_similarity: out['extrinsic'].flat[0] = np.nan
            if state.bad == 'loop_ratio' and compute_similarity: out['image_match_ratio'] = np.nan
            if state.bad == 'hardware': raise RuntimeError('synthetic device error')
            return out

    def increment(): state.loops += 1
    owner = SimpleNamespace(image_size=14, max_loops=1, image_retrieval=Retrieval(),
        salad_checkpoint=None, offline=True, lc_thres=.8,
        map=SimpleNamespace(get_largest_key=lambda: None,
            get_frames_from_loops=lambda _: [retrieved],
            get_submap=lambda _: SimpleNamespace(get_img_names_at_index=lambda _: '2.png')),
        graph=SimpleNamespace(increment_loop_closure=increment),
        vggt_timer=nullcontext(), clip_timer=nullcontext(), loop_closure_timer=nullcontext())
    ns = dict(torch=torch, np=np, time=time, Submap=Submap, DEBUG=False, colored=lambda s, _: s,
              as_executor=execution.as_executor, execute=execution.execute,
              get_auxiliary_device=execution.get_auxiliary_device,
              require_compute_similarity=execution.require_compute_similarity,
              prepare_rgb_images=execution.prepare_rgb_images)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
    pixels = np.stack((np.full((3, 14, 14), .4, np.float32), np.full((3, 14, 14), .7, np.float32)))
    return ns['run_predictions'], owner, Executor(), state, pixels


def invoke(services, max_loops=1):
    run, owner, backend, state, pixels = services
    return run(owner, ['100.25.png', '102.5.png'], backend, max_loops, None, None, prepared_images=pixels)


def test_no_parameters_numpy_normal_solver_preserves_pixels_and_source_ids(services):
    result = invoke(services, max_loops=0)
    _, owner, _, state, pixels = services
    assert state.calls == [(pixels, False)]
    assert result['images'] is pixels
    assert owner.current_working_submap.frames.data_ptr() == pixels.ctypes.data
    assert owner.current_working_submap.ids == [100.25, 102.5]
    assert state.loops == 0 and state.retrieval == []


@pytest.mark.parametrize('ratio,accepted', [(.949999, False), (.95, True), (1.0043401718139648, True)])
def test_query_then_older_frame_original_similarity_gate(services, ratio, accepted):
    services[3].ratio = ratio
    result = invoke(services)
    _, owner, _, state, _ = services
    assert [loop for _, loop in state.calls] == [False, True]
    np.testing.assert_array_equal(state.calls[1][0][0], .7)
    np.testing.assert_array_equal(state.calls[1][0][1], .2)
    assert result['loop_match_ratio'] == ratio
    assert state.loops == int(accepted)
    if accepted:
        assert len(result['detected_loops']) == 1
        assert result['frames_lc_names'] == ['102.5.png', '2.png']
        assert result['extrinsic_lc'].shape == (2, 3, 4)
        assert result['frames_lc'].device.type == 'cpu'
    else:
        assert result['detected_loops'] == []
        assert len(result['loop_candidate_matches']) == 1
        assert result['loop_rejection_reason'] == 'match_ratio_below_threshold'
        assert 'extrinsic_lc' not in result


def test_missing_loop_capability_has_no_preprocess_submap_or_retrieval_effects(services):
    run, owner, backend, state, _ = services
    backend.capabilities = execution.ExecutionCapabilities('normal-only', False)
    with pytest.raises(execution.UnsupportedExecutionError):
        run(owner, ['missing-file.png'], backend, 1, None, None)
    assert state.submaps == [] and state.retrieval == [] and state.calls == []
    assert not hasattr(owner, 'current_working_submap') and state.loops == 0


@pytest.mark.parametrize('bad', ['normal_shape', 'loop_nan', 'loop_ratio', 'hardware'])
def test_invalid_results_and_backend_errors_never_increment_loop_count(services, bad):
    services[3].bad = bad
    with pytest.raises((ValueError, RuntimeError)):
        invoke(services)
    assert services[3].loops == 0
    if bad in ('normal_shape', 'hardware'):
        assert len(services[3].calls) == 1


def test_explicit_solver_loop_disable_allows_normal_only_executor(services):
    _, owner, backend, _, _ = services
    owner.max_loops = 0
    backend.capabilities = execution.ExecutionCapabilities('normal-only', False)
    assert invoke(services)['detected_loops'] == []
