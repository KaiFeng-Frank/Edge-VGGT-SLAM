"""Generated pixels/features, fake RNG/SDK and one published-receipt check.

No real Torch/ORT/RKNN library/model/calibration/compiler/board execution.
"""
import ast
from contextlib import contextmanager
import copy
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


H = load(HERE / "v41_half_rknn_encoder.py", "_v41_encoder_test")
P = load(VALIDATION / "dpvo_stream/v30_frame_encoder.py", "_v41_sampling_test")
S = load(VALIDATION / "dpvo_half_workload/run-v39-half-int8-board.py", "_v41_half_sdk_test")
A = load(VALIDATION / "run-v12-rknn-block-board.py", "_v41_fake_abi_test")
Q = load(VALIDATION / "dpvo_encoder_int8/query-v37-spatial-int8-io-board.py", "_v41_fake_query_test")
T = load(VALIDATION / "dpvo_sparse_context_sdk/v34_sparse_rknn.py", "_v41_fake_transport_test")


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return FakeTensor(self.value[index])

    def numpy(self):
        return self.value


class FakeGenerator:
    device = SimpleNamespace(type="cpu")

    def __init__(self):
        self.calls = []


@pytest.fixture
def fake_torch(monkeypatch):
    def randint(low, high, shape, *, generator, device):
        assert isinstance(generator, FakeGenerator) and device == "cpu"
        generator.calls.append(dict(low=low, high=high, shape=shape))
        # Interior samples touch both borders and retain duplicate order.
        value = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        value = low + value % (high - low)
        value[0, -1] = high - 1
        return FakeTensor(value)
    fake = SimpleNamespace(Generator=FakeGenerator, randint=randint,
        stack=lambda values, dim: FakeTensor(np.stack([v.value for v in values], axis=dim)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


class FakeSDK:
    def __init__(self):
        self.runs = self.releases = self.destroys = self.inits = 0
        self.allocations = []
        self.bad = None
        self.centers_seen = []

    def rknn_init(self, context, *_):
        self.inits += 1
        ct.cast(context, ct.POINTER(A.CONTEXT)).contents.value = 123
        return 0

    def rknn_set_core_mask(self, context, mask):
        assert mask == 7
        return 0

    def rknn_query(self, context, command, value, _):
        if command == 5:
            item = ct.cast(value, ct.POINTER(A.SDKVersion)).contents
            item.api_version = b"2.3.2-fake"
            item.drv_version = b"fake"
        elif command == 0:
            item = ct.cast(value, ct.POINTER(A.InputOutputNum)).contents
            item.n_input = item.n_output = 2
        elif command == 6:
            ct.cast(value, ct.POINTER(A.MemSize)).contents.total_dma_allocated_size = 12345
        elif command in (1, 2):
            item = ct.cast(value, ct.POINTER(A.TensorAttr)).contents
            i = item.index
            if command == 1 and i == 0:
                shape, name, fmt, dtype = (1, 232, 304, 3), "normalized_bgr", 1, 2
            elif command == 1:
                shape, name, fmt, dtype = (16,), "patch_indices", 3, 8
            elif i == 0:
                shape, name, fmt, dtype = (1, 128, 58, 76), "fmap", 0, 2
                if self.bad == "legacy_shape":
                    shape = (1, 128, 116, 152)
                elif self.bad == "output_nhwc":
                    shape, fmt = (1, 58, 76, 128), 1
            else:
                shape, name, fmt, dtype = (1, 16, 384), "context", 3, 2
            item.n_dims = len(shape)
            for j, d in enumerate(shape):
                item.dims[j] = d
            item.name = name.encode()
            item.fmt, item.type = fmt, dtype
            item.n_elems = int(np.prod(shape))
            item.size = item.n_elems * (8 if dtype == 8 else 1)
            item.qnt_type, item.zp = (0 if dtype == 8 else 2), 0
            item.scale = 1. if dtype == 8 else .0125
        else:
            raise AssertionError(command)
        return 0

    def rknn_inputs_set(self, context, count, inputs):
        assert count == 2
        assert [(v.type, v.fmt, v.pass_through, v.size) for v in inputs] == [
            (0, 1, 0, 232 * 304 * 3 * 4), (8, 3, 0, 16 * 8)]
        raw = np.ctypeslib.as_array(ct.cast(inputs[0].buf, ct.POINTER(ct.c_float)), shape=(232 * 304 * 3,))
        assert np.array_equal(raw.reshape(1, 232, 304, 3), self.expected_image.transpose(0, 2, 3, 1))
        index = np.ctypeslib.as_array(ct.cast(inputs[1].buf, ct.POINTER(ct.c_int64)), shape=(16,)).copy()
        self.centers_seen.append(np.column_stack((index % 76, index // 76)))
        return 0

    def rknn_run(self, context, extension):
        self.runs += 1
        if self.bad == "run_failure":
            return -1
        ct.cast(extension, ct.POINTER(A.RunExtend)).contents.frame_id = self.runs
        return 0

    def rknn_outputs_get(self, context, count, outputs, extension):
        assert count == 2
        ct.cast(extension, ct.POINTER(A.OutputExtend)).contents.frame_id = self.runs
        self.allocations = []
        for i, (name, shape) in enumerate(S.OUTPUTS):
            assert outputs[i].want_float == 1 and outputs[i].is_prealloc == 0
            value = np.full(shape, np.float32(self.runs + i / 4), np.float32)
            if self.bad == "nonfinite" and i == 0:
                value[0, 0, 0, 0] = np.nan
            if self.bad == "zero_context" and i == 1:
                value.fill(0)
            self.allocations.append(value)
            outputs[i].buf, outputs[i].size = value.ctypes.data, value.nbytes
        return 0

    def rknn_outputs_release(self, context, count, outputs):
        self.releases += 1
        for value in self.allocations:
            value.fill(-999)
        return -1 if self.bad == "release_failure" else 0

    def rknn_destroy(self, context):
        self.destroys += 1
        return 0


@pytest.fixture
def encoder_factory(monkeypatch, tmp_path, fake_torch):
    """Hardware guard, disk pins, native library and log capture are explicit fakes.

    The unchanged HalfContext/attr validator/typed transport execute against a
    fake API with generated buffers. Lineage math has a separate static check.
    """
    source = tmp_path / "source"
    source.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    model = source / "model.rknn"
    model.write_bytes(b"fake-model")
    compiled = dict(input=dict(sha256="synthetic_graph"), configuration={})
    probe = dict(schema="v39_DPVO_half_encoder_board_probe",
        status="completed_half_M16_CPU_teacher_and_ORT_diagnostic_not_VO",
        script_sha256=S.PROBE_DRIVER_SHA,
        strict_gate_each_method={"original": True, "reciprocal": True},
        historical_parent_FP32_gate_passed=False,
        historical_parent_rejection_retained=True, candidate_accepted=False)
    for name, value in (("compile.json", compiled), ("probe.json", probe)):
        (source / name).write_text(json.dumps(value))
    monkeypatch.setattr(H, "board_guard", lambda: None)
    monkeypatch.setattr(H, "MODEL_BYTES", len(b"fake-model"))
    monkeypatch.setattr(A, "bind_api", lambda api: api)
    def pin(value, path, expected):
        path = Path(path).resolve()
        value._pins[str(path)] = H.sha(path) if path.is_file() else expected
        return path
    monkeypatch.setattr(H.BoardHalfRKNNEncoder, "_pin", pin)
    sdk_module = SimpleNamespace(HalfContext=S.HalfContext, PROBE_DRIVER_SHA=S.PROBE_DRIVER_SHA,
        COMPILER_SHA=S.COMPILER_SHA, unchanged=lambda p: None,
        validate_compile=lambda *args: None)
    modules = {"_v41_frozen_half_sdk": sdk_module, "_v41_frozen_sampling": P,
        "_v41_frozen_encoder_abi": A, "_v41_frozen_encoder_attr_guard": Q,
        "_v41_frozen_encoder_transport": T}
    def make(bad=None):
        api = FakeSDK()
        api.bad = bad
        api.expected_image = P.normalize_bgr(np.full((232, 304, 3), 17, np.uint8))
        @contextmanager
        def native(directory):
            row = dict(truncated=False, RKNN_error_detected=False, raw_text="", bytes=0)
            yield row
            if api.bad == "native_error" and api.runs and not api.destroys:
                row.update(RKNN_error_detected=True, raw_text="E RKNN fake failure")
        modules["_v41_frozen_encoder_native_log"] = SimpleNamespace(_native_log=native)
        monkeypatch.setattr(H, "load", lambda path, name: modules[name])
        monkeypatch.setattr(H.ct, "CDLL", lambda path: api)
        value = H.BoardHalfRKNNEncoder(model, source / "compile.json", source / "probe.json",
            **{key: source / key for key in ("half_sdk", "query_helper", "transport_helper",
                "abi_helper", "header", "library", "encoder_module", "native_log_helper")},
            log_directory=logs, diagnostic_allow_fp32_rejected=True)
        return value, api
    return make, logs


def test_sparse_patch_math_byte_equal_to_dense_original_and_owned():
    rng = np.random.default_rng(8)
    fmap = rng.uniform(-2, 2, (1, 5, 6, 7)).astype(np.float32)
    imap = rng.uniform(-2, 2, (1, 9, 6, 7)).astype(np.float32)
    centers = np.array([[1, 1], [5, 4], [2, 3], [1, 1]], np.int64)
    sparse = np.ascontiguousarray(imap[0][:, centers[:, 1], centers[:, 0]].T[None])
    expected = P.features_to_patches(fmap, imap, centers)
    actual = H.features_to_sparse_patches(fmap, sparse, centers, P)
    for name in ("fmap", "gmap", "context", "patches", "centers"):
        assert actual[name].shape == expected[name].shape
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].tobytes() == expected[name].tobytes()
        assert actual[name].flags.owndata and actual[name].flags.c_contiguous
    fmap.fill(0)
    sparse.fill(0)
    centers.fill(0)
    assert actual["fmap"].tobytes() == expected["fmap"].tobytes()
    assert actual["context"].tobytes() == expected["context"].tobytes()
    assert actual["centers"].tobytes() == expected["centers"].tobytes()


def test_persistent_half_sdk_correct_typed_inputs_rng_order_and_stable_outputs(encoder_factory):
    factory, _ = encoder_factory
    encoder, api = factory()
    generator = FakeGenerator()
    image = np.full((232, 304, 3), 17, np.uint8)
    a = encoder.encode(image, generator=generator)
    b = encoder.encode(image, generator=generator, patch_count=16)
    assert generator.calls == [dict(low=1, high=75, shape=(1, 16)),
        dict(low=1, high=57, shape=(1, 16))] * 2
    assert api.inits == 1 and api.runs == api.releases == 2
    assert np.array_equal(a["centers"], api.centers_seen[0])
    assert a["fmap"].shape == (128, 58, 76)
    assert a["gmap"].shape == (16, 128, 3, 3)
    assert a["context"].shape == (16, 384)
    assert a["patches"].shape == (16, 3, 3, 3)
    assert np.all(a["fmap"] == 1) and np.all(b["fmap"] == 2)
    assert np.all(a["context"] == 1.25) and np.all(a["patches"][:, 2] == 1)
    for name in ("fmap", "gmap", "context", "patches", "centers"):
        assert a[name].flags.owndata and a[name].flags.c_contiguous
        assert not np.shares_memory(a[name], b[name])
    metrics = encoder.metrics()
    assert metrics["context"]["native_IO_contract_checked"]
    assert metrics["context"]["input_attrs"][1]["shape"] == [16]
    assert metrics["context"]["output_attrs"][1]["shape"] == [1, 16, 384]
    assert metrics["completed_frames"] == metrics["sdk_run_api_calls"] == 2
    assert metrics["candidate_accepted"] is False and metrics["adopted"] is False
    encoder.close()
    encoder.close()
    assert api.destroys == 1 and encoder.metrics()["context"]["context_destroyed"]
    with pytest.raises(ValueError, match="closed"):
        encoder.encode(image, generator=generator)


@pytest.mark.parametrize("bad", ["nonfinite", "zero_context", "native_error", "run_failure", "release_failure"])
def test_execution_failure_closes_and_records_original_error(encoder_factory, bad):
    factory, logs = encoder_factory
    encoder, api = factory()
    api.bad = bad
    with pytest.raises((ValueError, RuntimeError)):
        encoder.encode(np.full((232, 304, 3), 17, np.uint8), generator=FakeGenerator())
    assert encoder.metrics()["closed"] and api.destroys == 1
    assert encoder.metrics()["completed_frames"] == 0
    record = json.loads((logs / "encoder-failure.json").read_text())
    assert record["metrics"]["sdk_run_api_calls"] == 1
    assert record["metrics"]["calls"][0]["completed"] is False
    assert record["metrics"]["candidate_accepted"] is False
    if bad in ("nonfinite", "zero_context", "native_error"):
        assert (logs / "encoder-failure-outputs.npz").is_file()


@pytest.mark.parametrize("bad", ["legacy_shape", "output_nhwc"])
def test_actual_new_model_attrs_reject_legacy_or_ambiguous_outputs(encoder_factory, bad):
    factory, logs = encoder_factory
    with pytest.raises(ValueError):
        factory(bad)
    record = json.loads((logs / "encoder-failure.json").read_text())
    context = record["metrics"]["context"]
    assert len(context["input_attrs"]) == len(context["output_attrs"]) == 2
    assert context["context_destroyed"]
    assert record["metrics"]["sdk_run_api_calls"] == 0


def test_diagnostic_flag_precedes_hardware_guard_and_all_files(monkeypatch):
    monkeypatch.setattr(H, "board_guard", lambda: pytest.fail("must reject before guard"))
    args = {k: "absent" for k in ("half_sdk", "query_helper", "transport_helper", "abi_helper",
                                  "header", "library", "encoder_module", "native_log_helper", "log_directory")}
    with pytest.raises(ValueError, match="Explicit"):
        H.BoardHalfRKNNEncoder("absent", "absent", "absent", **args)


def test_fixed_half_input_contract_no_sdk_attempt_or_rng_on_rejection(encoder_factory):
    factory, _ = encoder_factory
    encoder, api = factory()
    generator = FakeGenerator()
    with pytest.raises(ValueError, match="M16"):
        encoder.encode(np.full((232, 304, 3), 17, np.uint8), generator=generator, patch_count=96)
    assert api.runs == 0 and generator.calls == []
    encoder.close()


def test_frozen_original_rng_source_calls_only_x_then_y_no_depth():
    tree = ast.parse((VALIDATION / "dpvo_stream/v30_frame_encoder.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "random_centers")
    draws = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "randint"]
    assert len(draws) == 2
    assignments = [node for node in fn.body if isinstance(node, ast.Assign)]
    assert assignments[0].targets[0].id == "x" and assignments[1].targets[0].id == "y"
    assert "rand(" not in (HERE / "v41_half_rknn_encoder.py").read_text()


def test_frozen_pin_and_lineage_validation_is_static_only():
    expected = {"dpvo_half_workload/run-v39-half-int8-board.py": H.HALF_SDK_SHA,
        "dpvo_sparse_context_sdk/v34_sparse_rknn.py": H.TRANSPORT_SHA,
        "dpvo_encoder_int8/query-v37-spatial-int8-io-board.py": H.QUERY_SHA,
        "dpvo_stream/v30_frame_encoder.py": H.SAMPLING_SHA,
        "run-v12-rknn-block-board.py": H.ABI_SHA,
        "dpvo_source_frame/v33_rknn_source_frame.py": H.NATIVE_LOG_SHA}
    for filename, digest in expected.items():
        assert H.sha(VALIDATION / filename) == digest
    # Published embedded records reconstruct the original receipt bytes; this
    # verifies static metadata, not a fake SDK call or access to private files.
    published = json.loads((HERE.parents[3] / "results/dpvo_half_workload_v39.json").read_text())
    evidence = published["evidence"]
    compiled, probe = evidence["PTQ"]["record"], evidence["encoder_probe"]["record"]
    for key, record, expected_sha in (("PTQ", compiled, H.COMPILE_SHA),
                                      ("encoder_probe", probe, H.PROBE_SHA)):
        restored = (json.dumps(record, indent=2) + "\n").encode()
        assert hashlib.sha256(restored).hexdigest() == evidence[key]["receipt_sha256"] == expected_sha
    S.validate_compile(compiled, S.COMPILER_SHA, probe, H.MODEL_SHA, H.MODEL_BYTES)
    changed = copy.deepcopy(compiled)
    changed["candidate_accepted"] = True
    with pytest.raises(ValueError, match="Diagnostic"):
        S.validate_compile(changed, S.COMPILER_SHA, probe, H.MODEL_SHA, H.MODEL_BYTES)
