"""Synthetic weights and fake RKNN ABI only; no original model or hardware execution."""
import copy
import ctypes as ct
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from vggt_slam import rknn_camera as camera


class TinyHead(nn.Module):
    """69 tiny synthetic parameter tensors, never an original CameraHead forward."""
    def __init__(self):
        super().__init__()
        for i in range(68):
            self.register_parameter(f"p{i}", nn.Parameter(torch.tensor([i / 8.], dtype=torch.float32), requires_grad=False))
        seed = torch.arange(-4, 5, dtype=torch.float32).reshape(1, 1, 9)
        self.empty_pose_tokens = nn.Parameter(seed, requires_grad=False)
        self.eval()


class Function:
    def __init__(self, body):
        self.body = body

    def __call__(self, *args):
        return self.body(*args)


class FakeSDK:
    def __init__(self, base, *, reverse=False, fail=None, delay=0., dma=123456):
        self.base, self.reverse, self.fail, self.delay, self.dma = base, reverse, fail, delay, dma
        self.run_count = self.releases = self.destroys = 0
        self.conditions, self.raw_tokens = [], []
        self.active_runs = self.peak_runs = 0
        self.buffer = (ct.c_float * 18)()
        for name in ("init", "query", "set_core_mask", "inputs_set", "run",
                     "outputs_get", "outputs_release", "destroy"):
            setattr(self, "rknn_" + name, Function(getattr(self, name)))

    def init(self, out, filename, size, flags, extend):
        assert size == flags == 0 and extend is None
        assert Path(ct.string_at(filename).decode()).is_file()
        ct.cast(out, ct.POINTER(self.base.CONTEXT)).contents.value = 123
        return -1 if self.fail == "init" else 0

    def attr(self, index, name, shape):
        value = self.base.TensorAttr(index=index, n_dims=len(shape), name=name.encode(),
                                    n_elems=int(np.prod(shape)), size=int(np.prod(shape)) * 2,
                                    type=1, fmt=3, qnt_type=2, zp=0, scale=1.)
        for i, dimension in enumerate(shape):
            value.dims[i] = dimension
        return value

    def query(self, context, command, out, size):
        base = self.base
        if self.fail == "memory_query" and command == 6:
            return -1
        if command == 0:
            value = base.InputOutputNum(2, 1)
        elif command == 5:
            value = base.SDKVersion(api_version=b"2.3.20" if self.fail == "version" else b"2.3.2 (fake)",
                                    drv_version=b"fake")
        elif command in (1, 2):
            index = ct.cast(out, ct.POINTER(base.TensorAttr)).contents.index
            names = ["conditioning_pose", "camera_tokens"] if self.reverse else ["camera_tokens", "conditioning_pose"]
            name = names[index] if command == 1 else "pose_delta"
            shape = (1, 2, 2048) if name == "camera_tokens" else (1, 2, 9)
            value = self.attr(index, name, shape)
            if self.fail == "attr" and name == "pose_delta":
                value.scale = .5
        elif command == 6:
            value = base.MemSize(total_weight_size=120000, total_internal_size=3456,
                                 total_dma_allocated_size=self.dma)
        else:
            raise AssertionError(f"unexpected query {command}")
        assert size == ct.sizeof(value)
        ct.memmove(out, ct.byref(value), size)
        return 0

    def set_core_mask(self, context, mask):
        assert mask in (1, 7)
        return -1 if self.fail == "core" else 0

    def inputs_set(self, context, count, inputs):
        assert count == 2
        raw_index = int(self.reverse)
        cond_index = 1 - raw_index
        assert inputs[raw_index].size == 4096 * 4 and inputs[cond_index].size == 18 * 4
        assert inputs[0].type == inputs[1].type == 0
        assert inputs[0].pass_through == inputs[1].pass_through == 0
        self.conditions.append(np.ctypeslib.as_array(ct.cast(inputs[cond_index].buf,
                               ct.POINTER(ct.c_float)), shape=(18,)).copy().reshape(1, 2, 9))
        self.raw_tokens.append(np.ctypeslib.as_array(ct.cast(inputs[raw_index].buf,
                               ct.POINTER(ct.c_float)), shape=(4096,)).copy().reshape(1, 2, 2048))
        return -1 if self.fail == "inputs" else 0

    def run(self, context, extend):
        self.active_runs += 1
        self.peak_runs = max(self.peak_runs, self.active_runs)
        time.sleep(self.delay)
        self.run_count += 1
        self.active_runs -= 1
        ct.cast(extend, ct.POINTER(self.base.RunExtend)).contents.frame_id = self.run_count
        # Negative FoV delta proves that activation must not be fed back.
        for i in range(18):
            self.buffer[i] = -2 if i % 9 >= 7 else 1
        if self.fail == "nan":
            self.buffer[0] = float("nan")
        return -1 if self.fail == "run" else 0

    def outputs_get(self, context, count, outputs, extend):
        assert count == 1 and outputs[0].want_float == 1 and not outputs[0].is_prealloc
        outputs[0].buf, outputs[0].size = ct.addressof(self.buffer), 72
        if self.fail == "output_size":
            outputs[0].size = 36
        ct.cast(extend, ct.POINTER(self.base.OutputExtend)).contents.frame_id = (
            self.run_count + 1 if self.fail == "frame_id" else self.run_count)
        return -1 if self.fail == "get" else 0

    def outputs_release(self, context, count, outputs):
        self.releases += 1
        for i in range(18):
            self.buffer[i] = -999
        return -1 if self.fail == "release" else 0

    def destroy(self, context):
        self.destroys += 1
        return -1 if self.fail == "destroy" else 0


def synthetic_python_helpers():
    """Load pinned Python ABI/transport only; fake tests do not need SDK headers.

    Production still validates its SDK header and native library. This separate
    synthetic loader supplies no hardware-validation evidence or NPU result.
    """
    path = camera._ROOT / "deploy/rk3588/validation/run-v16-camera-step-board.py"
    camera._require_sha(path, camera._RUNTIME_SHA)
    spec = importlib.util.spec_from_file_location("_synthetic_camera_step_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    base, abi_path = runtime.load_frozen("run-v12-rknn-block-board.py", camera._ABI_SHA,
                                       "_synthetic_camera_ABI")
    step, step_path = runtime.load_frozen("export-v16-camera-step-board.py", camera._EXPORT_SHA,
                                        "_synthetic_camera_step_contract")
    _, old_path, capture, capture_path = step.load_frozen()
    original, original_path = capture.base_capture()
    pins = {str(p): camera._sha(p) for p in
            (path, abi_path, step_path, old_path, capture_path, original_path)}
    return runtime, base, step, capture, original, pins


@pytest.fixture
def contract(tmp_path, monkeypatch):
    helpers = synthetic_python_helpers()
    monkeypatch.setattr(camera, "_load_helpers", lambda: helpers)
    monkeypatch.setattr(helpers[3], "camera_contract", lambda head: {"synthetic": True})
    # Public unit tests also run before fetching the original VGGT vendor tree.
    # Bind fake weights to a real, tiny synthetic source file instead; production
    # retains its complete frozen original-source checks without modification.
    source = tmp_path / "synthetic_camera_source.txt"
    source.write_text("Synthetic fake CameraHead fixture; no model or hardware execution.\n")
    source_pins = {str(source): camera._sha(source)}
    monkeypatch.setattr(helpers[4], "SOURCE_PINS", source_pins)
    monkeypatch.setattr(helpers[3], "EXTRA_PINS", {})
    model = nn.Module()
    model.camera_head = TinyHead()
    head = model.camera_head
    original = json.loads((camera._ROOT / "deploy/rk3588/validation/vggt-v15-camera-capture112.json").read_text())
    exported = json.loads((camera._ROOT / "deploy/rk3588/validation/vggt-v16-camera-step-export112-r1.json").read_text())
    compiled = json.loads((camera._ROOT / "deploy/rk3588/validation/vggt-v16-camera-step-compile112-r1.json").read_text())
    tensors = {name: camera._tensor_info(value) for name, value in head.state_dict().items()}
    original["artifacts"]["camera-state.npz"]["tensors"] = tensors
    original["source_hashes"] = source_pins.copy()
    original["architecture"] = exported["architecture"] = {"synthetic": True}
    original["unique_state_bytes"] = sum(item["bytes"] for item in tensors.values())
    exported["weight_sharing"]["original_unique_parameter_bytes"] = original["unique_state_bytes"] - 36
    exported["weight_sharing"]["parameters"] = [{"parameter": name, "initializer": name,
        "raw_sha256": info["sha256"], "shape": info["shape"]} for name, info in tensors.items() if name != "empty_pose_tokens"]
    exported["outer_iteration_contract"]["empty_weight"] = tensors["empty_pose_tokens"]
    paths = {name: tmp_path / name for name in ("capture.json", "export.json", "compile.json", "model.rknn")}
    paths["model.rknn"].write_bytes(b"synthetic compiled bytes, not a model")

    def rewrite():
        paths["capture.json"].write_text(json.dumps(original))
        exported["capture"]["sha256"] = camera._sha(paths["capture.json"])
        paths["export.json"].write_text(json.dumps(exported))
        compiled["export_receipt_sha256"] = camera._sha(paths["export.json"])
        compiled["output"].update(sha256=camera._sha(paths["model.rknn"]), size=paths["model.rknn"].stat().st_size)
        paths["compile.json"].write_text(json.dumps(compiled))
        return dict(model_path=paths["model.rknn"], model_sha256=camera._sha(paths["model.rknn"]),
                    export_receipt=paths["export.json"], export_sha256=camera._sha(paths["export.json"]),
                    compile_receipt=paths["compile.json"], compile_sha256=camera._sha(paths["compile.json"]),
                    capture_receipt=paths["capture.json"], capture_sha256=camera._sha(paths["capture.json"]))

    args = rewrite()
    return model, args, helpers, (original, exported, compiled), rewrite


def install(contract, **sdk_options):
    model, args, helpers, _, _ = contract
    sdk = FakeSDK(helpers[1], **sdk_options)
    handle = camera.install_rknn_camera(model, **args, _library=sdk)
    return model, handle, sdk


def inputs():
    # Multiple patches, only raw camera-token index zero is used by original head.
    value = torch.full((1, 2, 3, 2048), 99., dtype=torch.float32)
    value[:, :, 0] = 17
    return [value]


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_context_four_steps_bind_by_name_copy_and_original_recurrence(contract, reverse):
    model, handle, sdk = install(contract, reverse=reverse)
    assert model.camera_head is handle and not handle.training
    with torch.inference_mode():
        poses = model.camera_head(inputs(), num_iterations=4)
    assert len(poses) == 4 and all(p.shape == (1, 2, 9) and p.dtype == torch.float32 for p in poses)
    for i, pose in enumerate(poses):
        assert torch.equal(pose[..., :7], torch.full((1, 2, 7), float(i + 1)))
        assert not torch.count_nonzero(pose[..., 7:])
    assert np.array_equal(sdk.conditions[0], np.broadcast_to(np.arange(-4, 5, dtype=np.float32), (1, 2, 9)))
    assert [array[0, 0, 7] for array in sdk.conditions[1:]] == [-2, -4, -6]
    assert all(np.all(raw == 17) for raw in sdk.raw_tokens)
    assert sdk.releases == 4 and all(x == -999 for x in sdk.buffer)
    before = handle.metrics()
    assert before["completed_heads"] == 1 and before["sdk_calls"] == 4 and before["sdk_run_attempts"] == 4
    assert before["sdk_outputs_copied"] == 4 and before["peak_dma_bytes"] == 123456
    assert before["npu_used"] is False and before["library_is_mock"] is True
    handle.close()
    handle.close()
    after = handle.metrics()
    assert sdk.destroys == 1 and after["closed"] and after["current_dma_bytes"] == 0
    assert after["sdk_calls"] == 4 and after["completed_heads"] == 1 and after["peak_dma_bytes"] == 123456
    assert model.camera_head is handle  # The original weights are not retained for restoration.
    after["provenance"]["files"].clear()
    assert handle.metrics()["provenance"]["files"]
    assert torch.equal(poses[0][..., :7], torch.ones((1, 2, 7)))


def test_install_does_not_keep_original_cpu_head_or_mutate_process_settings(contract, monkeypatch):
    head_ref = weakref.ref(contract[0].camera_head)
    # Real module code must not change affinity, Torch threads, or environment.
    monkeypatch.setattr("os.sched_setaffinity", lambda *_: pytest.fail("affinity mutation"))
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: pytest.fail("thread mutation"))
    _, handle, _ = install(contract)
    gc.collect()
    assert head_ref() is None
    assert not list(handle.parameters())
    handle.close()


@pytest.mark.parametrize("change", ["value", "dtype", "missing", "training"])
def test_original_parameter_identity_failure_preserves_cpu_head(contract, change):
    model, args, helpers, _, _ = contract
    old = model.camera_head
    if change == "value":
        old.p0.add_(1)
    elif change == "dtype":
        old.double()
    elif change == "missing":
        del old.p0
    else:
        old.train()
    sdk = FakeSDK(helpers[1])
    with pytest.raises(ValueError):
        camera.install_rknn_camera(model, **args, _library=sdk)
    assert model.camera_head is old and sdk.run_count == sdk.destroys == 0


@pytest.mark.parametrize("bad", ["binding_duplicate", "empty_seed", "compile_link", "source", "file_sha"])
def test_receipt_and_file_identity_rejection_precedes_sdk_initialization(contract, bad):
    model, args, helpers, receipts, rewrite = contract
    cap, exported, compiled = receipts
    if bad == "binding_duplicate":
        exported["weight_sharing"]["parameters"][0] = copy.deepcopy(exported["weight_sharing"]["parameters"][1])
    elif bad == "empty_seed":
        exported["outer_iteration_contract"]["empty_weight"]["sha256"] = "0" * 64
    elif bad == "compile_link":
        compiled["input"]["sha256"] = "0" * 64
    elif bad == "source":
        cap["source_hashes"] = {}
    else:
        args["model_path"].write_bytes(b"changed")
    if bad != "file_sha":
        args = rewrite()
    sdk = FakeSDK(helpers[1])
    with pytest.raises(ValueError):
        camera.install_rknn_camera(model, **args, _library=sdk)
    assert sdk.run_count == sdk.destroys == 0


@pytest.mark.parametrize("fail", ["init", "core", "version", "attr", "memory_query"])
def test_constructor_failure_destroys_partial_context_without_replacing_head(contract, fail):
    model, args, helpers, _, _ = contract
    old = model.camera_head
    sdk = FakeSDK(helpers[1], fail=fail)
    with pytest.raises(camera.RKNNCameraError):
        camera.install_rknn_camera(model, **args, _library=sdk)
    assert sdk.destroys == 1 and sdk.run_count == 0 and model.camera_head is old


def test_dma_budget_failure_destroys_context(contract):
    model, args, helpers, _, _ = contract
    sdk = FakeSDK(helpers[1], dma=4097)
    with pytest.raises(camera.RKNNCameraError, match="DMA"):
        camera.install_rknn_camera(model, **args, max_dma_bytes=4096, _library=sdk)
    assert sdk.destroys == 1


@pytest.mark.parametrize("bad", ["frames", "batch", "dimension", "dtype", "grad", "nan", "iterations", "training", "grad_mode"])
def test_unsupported_forward_is_fatal_and_never_runs_cpu_fallback(contract, bad):
    _, handle, sdk = install(contract)
    token_list, iterations = inputs(), 4
    if bad == "frames":
        token_list = [torch.zeros((1, 3, 1, 2048))]
    elif bad == "batch":
        token_list = [torch.zeros((2, 2, 1, 2048))]
    elif bad == "dimension":
        token_list = [torch.zeros((1, 2, 1, 1024))]
    elif bad == "dtype":
        token_list[0] = token_list[0].half()
    elif bad == "grad":
        token_list[0].requires_grad_(True)
    elif bad == "nan":
        token_list[0][0, 0, 0, 0] = float("nan")
    elif bad == "iterations":
        iterations = 3
    elif bad == "training":
        handle.train()
    mode = torch.enable_grad() if bad == "grad_mode" else torch.inference_mode()
    with mode, pytest.raises(camera.RKNNCameraError):
        handle(token_list, num_iterations=iterations)
    stats = handle.metrics()
    assert stats["failed_heads"] == stats["attempted_heads"] == 1
    assert stats["closed"] and stats["failed"] and not stats["npu_used"]
    assert sdk.run_count == 0 and sdk.destroys == 1


@pytest.mark.parametrize("fail", ["inputs", "run", "get", "frame_id", "output_size", "nan", "release"])
def test_forward_sdk_failure_counts_real_run_then_releases_and_destroys(contract, fail):
    _, handle, sdk = install(contract, fail=fail)
    with torch.inference_mode(), pytest.raises(camera.RKNNCameraError):
        handle(inputs())
    stats = handle.metrics()
    assert stats["closed"] and stats["completed_heads"] == 0 and stats["failed_heads"] == 1
    assert stats["sdk_calls"] == (0 if fail in ("inputs", "run") else 1)
    assert stats["sdk_run_attempts"] == (0 if fail == "inputs" else 1)
    assert sdk.releases == (0 if fail in ("inputs", "run") else 1)
    assert sdk.destroys == 1
    with torch.inference_mode(), pytest.raises(camera.RKNNCameraError, match="closed"):
        handle(inputs())
    handle.close()
    assert sdk.destroys == 1


def test_destroy_failure_reports_unreleased_dma_and_is_idempotent(contract):
    _, handle, sdk = install(contract, fail="destroy")
    with pytest.raises(camera.RKNNCameraError, match="cleanup"):
        handle.close()
    stats = handle.metrics()
    assert stats["closed"] and stats["cleanup_errors"] and stats["current_dma_bytes"] == 123456
    assert stats["context_destroy_attempts"] == 1 and stats["context_destroys"] == 0
    handle.close()
    assert sdk.destroys == 1


def test_multiple_forwards_are_serialized_and_keep_one_context(contract):
    _, handle, sdk = install(contract, delay=.003)
    errors = []
    def invoke():
        try:
            with torch.inference_mode():
                handle(inputs())
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert not errors and not any(thread.is_alive() for thread in threads)
    stats = handle.metrics()
    assert stats["completed_heads"] == 2 and stats["sdk_calls"] == 8
    assert sdk.peak_runs == 1 and stats["context_creates"] == 1
    handle.close()


def test_native_guard_before_library_load_on_host(contract, monkeypatch):
    model, args, helpers, _, _ = contract
    monkeypatch.setattr(helpers[1], "require_board", lambda: (_ for _ in ()).throw(RuntimeError("not RK3588")))
    monkeypatch.setattr(ct, "CDLL", lambda *_: pytest.fail("library loaded on host"))
    with pytest.raises(camera.RKNNCameraError, match="not RK3588"):
        camera.install_rknn_camera(model, **args)


def test_native_library_requires_exact_verified_binary_sha(contract, monkeypatch, tmp_path):
    model, args, helpers, _, _ = contract
    monkeypatch.setattr(helpers[1], "require_board", lambda: "fake rk3588 identity only")
    monkeypatch.setattr(ct, "CDLL", lambda *_: pytest.fail("unverified library loaded"))
    library = tmp_path / "unverified.so"
    library.write_bytes(b"wrong runtime even if its version might say 2.3.2")
    with pytest.raises(camera.RKNNCameraError, match="SHA256 mismatch"):
        camera.install_rknn_camera(model, **args, library_path=library)


def test_install_rejects_second_backend_and_invalid_options(contract):
    model, handle, sdk = install(contract)
    with pytest.raises(ValueError, match="Install once"):
        camera.install_rknn_camera(model, **contract[1], _library=sdk)
    for values in ({"core_mask": 3}, {"run_timeout_ms": 0}, {"max_dma_bytes": 0}):
        with pytest.raises(ValueError):
            camera.install_rknn_camera(nn.Module(), **contract[1], _library=sdk, **values)
    handle.close()
