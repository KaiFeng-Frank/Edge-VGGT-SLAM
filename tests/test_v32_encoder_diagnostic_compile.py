"""Synthetic metadata and fake-SDK tests; no model/SDK inference or build."""
import copy
import importlib.util
from pathlib import Path

import pytest


def load_helper():
    here = Path(__file__).resolve().parent
    candidates = (here / "compile-v32-dpvo-encoder-diagnostic.py",
                  here.parent / "deploy/rk3588/validation/dpvo_encoder_opt/compile-v32-dpvo-encoder-diagnostic.py")
    path = next(path for path in candidates if path.is_file())
    spec = importlib.util.spec_from_file_location("_v32_diagnostic_compile_test", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


@pytest.fixture
def helper():
    return load_helper()


def receipts(helper):
    proof = dict(original_initializers_byte_exact=True, non_instance_norm_nodes_byte_exact=True,
                 input_output_protobuf_byte_exact=True, model_header_byte_exact=True,
                 original_initializer_count=44, preserved_non_instance_norm_nodes=91,
                 added_nonlearned_constants=20, original_weights_or_output_divide_by_four_changed=False)
    rewrite = dict(kind="v27_dpvo_encoder_instance_norm_spatial_rewrite", version=1,
                   status="rewritten_and_statically_checked_not_runtime_validated",
                   script_sha256=helper.REWRITER_SHA, input=dict(sha256=helper.PARENT_SHA),
                   output=dict(file="encoders.onnx", bytes=1554634, sha256=helper.CANDIDATE_SHA),
                   external_data_used=False, sources_unchanged_after=True, preservation=proof,
                   rewrites=[{} for _ in range(10)], node_counts=dict(original=101, candidate=201),
                   FP32_gate=dict(rtol=1e-4, atol=1e-5, status="pending_board_ORT"))
    ort = dict(schema="v27_DPVO_InstanceNorm_rewrite_board_ORT", status="failed",
               script_sha256=helper.ORT_CHECKER_SHA, actual_board_execution=True,
               exact_FP32_gate_passed=False,
               error="ValueError('Original FP32 gate failed; do not compile as validated')",
               providers=["CPUExecutionProvider"], graph_optimization="ORT_DISABLE_ALL",
               affinity_restored=True, sources_and_inputs_unchanged=True, cleanup_errors=[],
               source_pins={"synthetic-source": "a" * 64}, source_hashes_after={"synthetic-source": "a" * 64},
               capture_sha256=helper.CAPTURE_SHA, export_sha256=helper.EXPORT_SHA,
               old_ort_sha256=helper.ORIGINAL_ORT_SHA, rewrite_sha256=helper.REWRITE_SHA,
               onnx=copy.deepcopy(rewrite["output"]), calls=[])
    for index, (cpu, old_ort) in enumerate(zip((4, 5, 53), (10, 2, 69))):
        row = dict(frame_index=index)
        for name, count in (("versus_original_CPU", cpu), ("versus_original_ORT", old_ort)):
            row[name] = {key: dict(rtol=1e-4, atol=1e-5, failed_elements=n, gate_passed=n == 0)
                         for key, n in (("fmap", count), ("imap", 0))}
        ort["calls"].append(row)
    return rewrite, ort


def compiled(helper):
    ancestor = helper.validate_lineage(*receipts(helper))
    report = dict(schema=helper.SCHEMA, status=helper.SUCCESS, script_sha256="b" * 64,
                  configuration=dict(helper.CONFIGURATION), ancestor=ancestor,
                  historical_FP32_gate=dict(helper.GATE), diagnostic_only=True, candidate_accepted=False,
                  model_inference_executed=False, production_changed=False, realtime_validated=False,
                  released=True, cleanup_errors=[], sources_and_inputs_unchanged=True,
                  source_pins={"synthetic-source": "a" * 64}, source_hashes_after={"synthetic-source": "a" * 64},
                  static_graph_preservation=ancestor["preservation"], input=dict(sha256=helper.CANDIDATE_SHA),
                  versions={"rknn-toolkit2": "2.3.2", "torch": "2.4.0+cpu", "numpy": "1.26.4"},
                  operations=[dict(name=name, code=0, seconds=0.001) for name in
                              ("config", "load_onnx", "build", "export_rknn")],
                  output=dict(path="/synthetic/block.rknn", size=8, sha256="c" * 64))
    return report, ancestor


def test_exact_rejection_is_preserved_and_runtime_validator_is_pure(helper):
    report, ancestor = compiled(helper)
    helper.validate_compilation(report, ancestor, expected_script_sha256="b" * 64)
    assert ancestor["historical_failed_elements"]["versus_original_CPU"]["fmap"] == [4, 5, 53]
    assert ancestor["failed_ORT_receipt_sha256"] == helper.FAILED_ORT_SHA
    assert report["candidate_accepted"] is False
    assert report["historical_FP32_gate"]["passed"] is False


@pytest.mark.parametrize("mutate", [
    lambda r, o: o.update(exact_FP32_gate_passed=True),
    lambda r, o: o.update(error="ValueError('missing input file')"),
    lambda r, o: o["calls"][0]["versus_original_CPU"]["fmap"].update(rtol=1e-3),
    lambda r, o: r["preservation"].update(original_initializers_byte_exact=False),
    lambda r, o: o.update(rewrite_sha256="0" * 64),
])
def test_reject_wrong_failure_relaxed_gate_or_ancestor(helper, mutate):
    rewrite, ort = receipts(helper)
    mutate(rewrite, ort)
    with pytest.raises(ValueError):
        helper.validate_lineage(rewrite, ort)


@pytest.mark.parametrize("key,value", [("candidate_accepted", True), ("released", False),
                                       ("configuration", {**load_helper().CONFIGURATION, "optimization_level": 2})])
def test_reject_fabricated_acceptance_cleanup_or_configuration(helper, key, value):
    report, ancestor = compiled(helper)
    report[key] = value
    with pytest.raises(ValueError):
        helper.validate_compilation(report, ancestor, expected_script_sha256="b" * 64)


def test_explicit_permission_required_before_output_or_imports(helper, tmp_path):
    output = tmp_path / "new-output"
    arguments = []
    for name in ("onnx", "parent-onnx", "rewrite", "failed-ort", "rewrite-helper", "frozen-compiler", "extractor"):
        arguments.extend(("--" + name, str(tmp_path / name)))
    arguments.extend(("--output-dir", str(output)))
    for name in ("onnx", "rewrite", "failed-ort"):
        arguments.extend(("--" + name + "-sha256", "a" * 64))
    with pytest.raises(ValueError, match="Explicit"):
        helper.main(arguments)
    assert not output.exists()


class FakeSDK:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.closed = False

    def config(self, **kwargs):
        self.calls.append(("config", kwargs))
        return 0

    def load_onnx(self, **kwargs):
        self.calls.append(("load_onnx", kwargs))
        return 0

    def build(self, **kwargs):
        self.calls.append(("build", kwargs))
        return -1 if self.fail == "build" else 0

    def export_rknn(self, path):
        self.calls.append(("export_rknn", path))
        Path(path).write_bytes(b"FAKE-SDK")
        return 0

    def release(self):
        self.closed = True
        if self.fail == "release":
            raise RuntimeError("fake release error")


@pytest.mark.parametrize("failure", [None, "build", "release"])
def test_original_sdk_configuration_and_unconditional_cleanup(helper, tmp_path, failure):
    sdk = FakeSDK(failure)
    report = dict(operations=[], cleanup_errors=[])
    factory = lambda **kwargs: sdk
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            helper.perform_compile(factory, tmp_path / "fake.onnx", tmp_path, report)
    else:
        helper.perform_compile(factory, tmp_path / "fake.onnx", tmp_path, report)
        assert report["output"]["size"] == 8
        assert report["released"] is True
    assert sdk.closed
    assert sdk.calls[0] == ("config", dict(target_platform="rk3588", float_dtype="float16", optimization_level=3))
    assert sdk.calls[2] == ("build", dict(do_quantization=False))
    assert failure != "build" or all(name != "export_rknn" for name, _ in sdk.calls)
    assert (len(report["cleanup_errors"]) == 1) == (failure == "release")
