"""Actual Solver dispatch with generated pixels and fake graph services only."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from vggt_slam import model_execution as execution


SOURCE = Path(__file__).resolve().parents[1] / "vggt_slam/solver.py"


def solver_method(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Solver")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def services():
    torch = pytest.importorskip("torch")
    state = SimpleNamespace(submaps=[], calls=[], bad=None, latest=None)

    class Submap:
        def __init__(self, node):
            state.submaps.append(self)
            self.node = node
        def add_all_frames(self, frames): self.frames = frames
        def set_frame_ids(self, names): self.ids = [float(Path(name).stem) for name in names]
        def set_last_non_loop_frame_index(self, index): self.last_index = index
        def set_img_names(self, names): self.names = list(names)
        def get_last_non_loop_frame_index(self): return self.last_index

    class Executor:
        capabilities = execution.ExecutionCapabilities("numpy-test-no-parameters", False)
        @property
        def parameters(self):
            pytest.fail("Solver must dispatch through the execution boundary")
        def predict(self, pixels, *, compute_similarity=False):
            assert isinstance(pixels, np.ndarray) and compute_similarity is False
            state.calls.append((pixels, compute_similarity))
            n, _, h, w = pixels.shape
            out = {"extrinsic": np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1)),
                   "intrinsic": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
                   "depth": np.ones((n, h, w, 1), np.float32),
                   "depth_conf": np.ones((n, h, w), np.float32),
                   "image_match_ratio": None}
            if state.bad == "shape": out["depth"] = out["depth"][None]
            if state.bad == "dtype": out["intrinsic"] = out["intrinsic"].astype(np.float64)
            if state.bad == "nan": out["extrinsic"].flat[0] = np.nan
            if state.bad == "ratio": out["image_match_ratio"] = .98
            if state.bad == "hardware": raise RuntimeError("synthetic device error")
            return out

    owner = SimpleNamespace(image_size=14, max_loops=0,
        map=SimpleNamespace(
            get_largest_key=lambda: None if state.latest is None else state.latest.node,
            get_latest_submap=lambda: state.latest),
        vggt_timer=nullcontext())
    namespace = dict(torch=torch, np=np, time=time, Submap=Submap,
                     as_executor=execution.as_executor, execute=execution.execute,
                     get_execution_device=execution.get_execution_device,
                     prepare_rgb_images=execution.prepare_rgb_images)
    run = solver_method("run_predictions", namespace)
    pixels = np.stack((np.full((3, 14, 14), .4, np.float32),
                       np.full((3, 14, 14), .7, np.float32)))
    return run, owner, Executor(), state, pixels


def invoke(services, **kwargs):
    run, owner, backend, _, pixels = services
    return run(owner, ["100.25.png", "102.5.png"], backend,
               prepared_images=pixels, **kwargs)


def test_no_parameters_normal_solver_preserves_pixels_source_ids_and_order(services):
    result = invoke(services)
    _, owner, _, state, pixels = services
    assert state.calls == [(pixels, False)]
    assert result["images"] is pixels
    assert owner.current_working_submap.frames.data_ptr() == pixels.ctypes.data
    assert owner.current_working_submap.ids == [100.25, 102.5]
    assert owner.current_working_submap.names == ["100.25.png", "102.5.png"]
    assert result["detected_loops"] == result["loop_candidate_matches"] == []
    assert result["loop_match_ratio"] is result["loop_rejection_reason"] is None


def test_overlap_creates_distinct_submap_nodes_without_reordering_input(services):
    invoke(services)
    run, owner, backend, state, pixels = services
    state.latest = owner.current_working_submap
    second = run(owner, ["102.5.png", "104.75.png"], backend, prepared_images=pixels)
    assert [s.node for s in state.submaps] == [0, 2]
    assert state.submaps[0] is not state.submaps[1]
    assert state.submaps[1].ids == [102.5, 104.75]
    assert second["images"] is pixels


@pytest.mark.parametrize("loop_count", [1, 2, -1, True, 0.0, None])
def test_loop_request_rejects_before_preprocess_model_or_submap(services, loop_count):
    run, owner, _, state, _ = services
    with pytest.raises(ValueError, match="max_loops=0"):
        run(owner, ["missing-file.png"], object(), max_loops=loop_count)
    assert state.submaps == state.calls == []
    assert not hasattr(owner, "current_working_submap")


def test_nonzero_solver_loop_state_rejects_before_dispatch(services):
    services[1].max_loops = 1
    with pytest.raises(ValueError, match="max_loops=0"):
        invoke(services)
    assert services[3].calls == services[3].submaps == []


@pytest.mark.parametrize("kwargs", [
    {"auxiliary_model": object()},
    {"auxiliary_preprocess": object()},
    {"unsupported_checkpoint": "unused"},
    {"unsupported_switch": False},
])
def test_auxiliary_request_rejects_before_preprocess_or_dispatch(services, kwargs):
    run, owner, _, state, _ = services
    with pytest.raises(ValueError, match="Auxiliary"):
        run(owner, ["missing-file.png"], object(), **kwargs)
    assert state.submaps == state.calls == []


def test_empty_legacy_auxiliary_slots_and_none_options_are_harmless(services):
    run, owner, backend, _, pixels = services
    assert run(owner, ["1.png", "2.png"], backend, 0, None, None,
               prepared_images=pixels, unused_option=None)["detected_loops"] == []


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "ratio", "hardware"])
def test_invalid_output_and_backend_failure_propagate_without_fallback(services, bad):
    services[3].bad = bad
    with pytest.raises((ValueError, RuntimeError)):
        invoke(services)
    assert len(services[3].calls) == 1
    assert services[1].map.get_largest_key() is None


@pytest.mark.parametrize("kwargs", [
    {"max_loops": 1}, {"max_loops": True}, {"max_loops": 0.0},
    {"auxiliary_checkpoint": "unused"}, {"unsupported_switch": False},
])
def test_constructor_rejects_before_device_viewer_graph_or_model_services(kwargs):
    def forbidden(*args, **other):
        pytest.fail("Unsupported options must reject before allocating services")
    namespace = {name: forbidden for name in (
        "resolve_device", "validate_image_size", "FrameTracker",
        "GraphMap", "PoseGraph", "Accumulator")}
    constructor = solver_method("__init__", namespace)
    with pytest.raises(ValueError):
        constructor(SimpleNamespace(), 25., **kwargs)


def test_add_points_rejects_loop_constraints_before_any_geometry_or_graph_mutation():
    add_points = solver_method("add_points", {})
    with pytest.raises(ValueError, match="Loop insertion"):
        add_points(SimpleNamespace(), {"detected_loops": [object()]})
