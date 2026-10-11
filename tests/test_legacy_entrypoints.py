"""Legacy-entry tests use mocked models and cameras; no hardware is opened."""

import importlib.util
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def legacy_entrypoint(monkeypatch):
    def load(filename, accepted=None, worker_failure=False, hold_first_worker=False, captures=22, failure_delay=0):
        state = SimpleNamespace(batches=[], model_loads=0, solver_inits=0,
                                camera_kwargs=None, camera_stopped=False, flow_display=[],
                                release_first=threading.Event(), first_completed=threading.Event())
        decisions = iter(accepted) if accepted is not None else None

        class Accumulator:
            total_time = 0.0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeSolver:
            def __init__(self, **kwargs):
                state.solver_inits += 1
                self.submaps = 0
                self.map = SimpleNamespace(get_num_submaps=lambda: self.submaps)
                self.graph = SimpleNamespace(optimize=self.optimize, get_num_loops=lambda: 0)
                self.flow_tracker = SimpleNamespace(compute_disparity=self.disparity)
                self.vggt_timer = self.loop_closure_timer = Accumulator()

            def disparity(self, image, threshold, visualize=False):
                state.flow_display.append(visualize)
                return next(decisions) if decisions is not None else True

            def run_predictions(self, batch, *args, **kwargs):
                assert kwargs.get("max_loops", 0) == 0
                assert all(Path(path).exists() for path in batch)
                state.batches.append(list(batch))
                if worker_failure:
                    time.sleep(failure_delay)
                    raise ValueError("injected inference failure")
                if hold_first_worker and len(state.batches) == 1:
                    assert state.release_first.wait(2.0)
                return {"detected_loops": []}

            def add_points(self, predictions):
                self.submaps += 1

            def optimize(self):
                state.first_completed.set()

            def update_all_submap_vis(self):
                pass

            def update_latest_submap_vis(self):
                pass

        class FakeCamera:
            def __init__(self, **kwargs):
                state.camera_kwargs = kwargs
                self.count = 0

            def start(self):
                pass

            def capture(self):
                self.count += 1
                if self.count >= 10:
                    state.release_first.set()
                if hold_first_worker and self.count >= 11:
                    assert state.first_completed.wait(2.0)
                if self.count > captures:
                    raise KeyboardInterrupt
                time.sleep(0.002)
                return np.zeros((6, 8, 3), dtype=np.uint8)

            def stop(self):
                state.camera_stopped = True

        def load_model(**kwargs):
            state.model_loads += 1
            return object()

        def imwrite(path, frame):
            Path(path).write_bytes(b"mock frame")
            return True

        def forbidden_display(*args):
            raise AssertionError("headless mode opened a local window")

        pyplot = SimpleNamespace()
        modules = {
            "torch": SimpleNamespace(set_num_threads=lambda n: None),
            "torchvision": SimpleNamespace(),
            "torchvision.transforms": SimpleNamespace(),
            "torchvision.transforms.functional": SimpleNamespace(to_pil_image=lambda img: img),
            "tqdm": SimpleNamespace(), "tqdm.auto": SimpleNamespace(tqdm=lambda iterable: iterable),
            "matplotlib": SimpleNamespace(pyplot=pyplot), "matplotlib.pyplot": pyplot,
            "cv2": SimpleNamespace(
                imread=lambda path: np.zeros((6, 8, 3), dtype=np.uint8), imwrite=imwrite,
                imshow=forbidden_display, waitKey=forbidden_display, destroyAllWindows=forbidden_display,
            ),
            "vggt_slam.slam_utils": SimpleNamespace(
                Accumulator=Accumulator, sort_images_by_number=sorted,
                downsample_images=lambda images, factor: images[::factor],
            ),
            "vggt_slam.solver": SimpleNamespace(Solver=FakeSolver),
            "vggt_slam.submap": SimpleNamespace(Submap=object),
            "vggt_slam.runtime": SimpleNamespace(load_model=load_model, resolve_device=lambda device: "cpu"),
            "vggt_slam.cameras": SimpleNamespace(BACKENDS={"v4l2": FakeCamera, "realsense": FakeCamera}),
        }
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        spec = importlib.util.spec_from_file_location(f"legacy_{filename}", ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, state

    return load


@pytest.mark.parametrize("filename", ["main.py", "main_realtime.py"])
@pytest.mark.parametrize("loop_count", ["1", "2", "-1"])
def test_legacy_cli_rejects_unsupported_loop_count(legacy_entrypoint, filename, loop_count):
    module, _ = legacy_entrypoint(filename)
    with pytest.raises(SystemExit) as exc:
        module.parser.parse_args(["--max_loops", loop_count])
    assert exc.value.code == 2
    assert module.parser.parse_args(["--max_loops", "0"]).max_loops == 0
    assert module.parser.parse_args([]).max_loops == 0


@pytest.mark.parametrize("filename", ["main.py", "main_realtime.py"])
@pytest.mark.parametrize("arguments", [["--max_loops", "1"], ["--auxiliary-model", "unused"], ["--run_os"]])
def test_removed_options_reject_before_model_graph_camera_or_output_directory(
        legacy_entrypoint, monkeypatch, tmp_path, filename, arguments):
    module, state = legacy_entrypoint(filename)
    output = tmp_path / "uncreated-output"
    if filename == "main_realtime.py":
        arguments = arguments + ["--keyframe_folder", str(output)]
    monkeypatch.setattr(sys, "argv", [filename] + arguments)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert state.model_loads == state.solver_inits == 0
    assert state.camera_kwargs is None and not output.exists()


def test_empty_offline_input_fails_before_allocating_models(legacy_entrypoint, monkeypatch, tmp_path):
    module, state = legacy_entrypoint("main.py")
    monkeypatch.setattr(sys, "argv", ["main.py", "--image_folder", str(tmp_path)])
    with pytest.raises(ValueError, match="No input images"):
        module.main()
    assert state.model_loads == 0 and state.solver_inits == 0


@pytest.mark.parametrize("last_accepted, expected_sizes", [(False, [3]), (True, [3, 2])])
def test_offline_tail_requires_new_images(legacy_entrypoint, monkeypatch, tmp_path, last_accepted, expected_sizes):
    module, state = legacy_entrypoint("main.py", accepted=[True, True, True, last_accepted])
    for index in range(4):
        (tmp_path / f"frame_{index:04d}.jpg").write_bytes(b"mock image")
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--image_folder", str(tmp_path), "--submap_size", "2",
        "--max_loops", "0", "--headless", "--vis_flow",
    ])
    module.main()
    assert [len(batch) for batch in state.batches] == expected_sizes
    assert not any(state.flow_display)
    if last_accepted:
        assert state.batches[1][0] == state.batches[0][-1]


@pytest.mark.parametrize("captures, failure_delay", [(22, 0), (4, 0.02)])
def test_realtime_worker_error_returns_to_caller_and_stops_camera(legacy_entrypoint, monkeypatch, tmp_path, captures, failure_delay):
    module, state = legacy_entrypoint("main_realtime.py", worker_failure=True,
                                     captures=captures, failure_delay=failure_delay)
    monkeypatch.setattr(sys, "argv", [
        "main_realtime.py", "--camera", "v4l2", "--camera-device", "/dev/video17",
        "--width", "320", "--height", "240", "--fps", "15", "--camera-timeout", "0.5",
        "--headless", "--vis_flow", "--submap_size", "2", "--max_loops", "0",
        "--keyframe_folder", str(tmp_path / "keyframes"),
    ])
    with pytest.raises(RuntimeError, match="SLAM worker failed: injected inference failure") as exc:
        module.main()
    assert isinstance(exc.value.__cause__, ValueError)
    assert state.camera_stopped
    assert len(state.batches) == 1
    assert not module.solver_lock.locked()
    assert not any(state.flow_display)
    assert state.camera_kwargs == dict(width=320, height=240, fps=15, device="/dev/video17", timeout=0.5)


def test_realtime_busy_worker_keeps_committed_overlap(legacy_entrypoint, monkeypatch, tmp_path):
    module, state = legacy_entrypoint("main_realtime.py", hold_first_worker=True)
    monkeypatch.setattr(sys, "argv", [
        "main_realtime.py", "--camera", "v4l2", "--headless", "--submap_size", "2",
        "--max_loops", "0", "--keyframe_folder", str(tmp_path / "keyframes"),
    ])
    module.main()
    assert len(state.batches) >= 2
    assert all(len(batch) == 3 for batch in state.batches)
    for prior, current in zip(state.batches, state.batches[1:]):
        assert current[0] == prior[-1]
    # The first worker holds long enough to drop unsent frames 4, 5, ...,
    # while the overlap from its submitted batch remains frame 3.
    assert Path(state.batches[1][0]).name == "frame_000003.png"
    assert int(Path(state.batches[1][1]).stem.split("_")[-1]) > 4
    assert state.camera_stopped and not module.solver_lock.locked()


def test_realtime_session_without_map_fails(legacy_entrypoint, monkeypatch, tmp_path):
    module, state = legacy_entrypoint("main_realtime.py", captures=2)
    monkeypatch.setattr(sys, "argv", [
        "main_realtime.py", "--camera", "v4l2", "--headless", "--submap_size", "2",
        "--max_loops", "0", "--keyframe_folder", str(tmp_path / "keyframes"),
    ])
    with pytest.raises(RuntimeError, match="No submap completed"):
        module.main()
    assert state.camera_stopped


def test_reconnection_also_observes_worker_failure(legacy_entrypoint):
    module, _ = legacy_entrypoint("main_realtime.py")
    errors = queue.Queue()
    errors.put(ValueError("inference failed while camera disconnected"))
    camera = SimpleNamespace(stop=lambda: None, start=lambda: pytest.fail("reopened a camera after worker failure"))
    with pytest.raises(RuntimeError, match="inference failed while camera disconnected"):
        module.restart_camera(camera, error_queue=errors)
