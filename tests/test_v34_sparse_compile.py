"""Tiny metadata and fake compiler calls only, never compile a real model."""
import copy
import importlib.util
from pathlib import Path

import pytest

ENTRY = Path(__file__).with_name("compile-v34-dpvo-sparse-context.py")
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_sparse_context_sdk/compile-v34-dpvo-sparse-context.py"
SPEC = importlib.util.spec_from_file_location("v34_sparse_compile_test", ENTRY)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def lineage(method):
    proof = {"all_original_initializers_ordered_protobuf_byte_exact": True}
    recipe = dict(schema="v33_DPVO_encoder_sparse_context_rewrite", status="statically_checked_not_runtime_validated",
        candidate_accepted=False, script_sha256=C.EDITOR_SHA, parent={"sha256": C.MODELS[method]["parent_sha"]},
        output={"sha256": C.MODELS[method]["graph_sha"]}, inputs=C.INPUTS, outputs=C.OUTPUTS,
        source_pins={"x": "pinned"}, source_hashes_after={"x": "pinned"}, sources_and_parent_unchanged=True, preservation=proof)
    reference = dict(schema="v33_DPVO_sparse_context_board_CPU_ORT",
        status="completed_suffix_bitwise_gate_parent_rejection_retained_not_VO", suffix_bitwise_gate_passed=True,
        suffix_strict_gate_passed=True, source_pins={"x": "pinned"}, source_hashes_after={"x": "pinned"},
        sources_and_inputs_unchanged=True, cleanup_errors=[], receipt_sha256={method+"_sparse": C.MODELS[method]["recipe_sha"]},
        graph_preservation={method: proof}, gate={"rtol": 1e-4, "atol": 1e-5},
        original_CPU_strict_gate_each_method={"original": True, "spatial": False},
        historical_spatial_rejection={"exact_FP32_gate_passed": False, "retained": True},
        calls=[dict(method=method,frame_index=i,versus_current_parent_ORT={"fmap":dict(bitwise_equal=True,gate_passed=True)}) for i in range(3)])
    return recipe, reference


def test_original_accepts_and_spatial_remains_explicitly_rejected():
    a,b=lineage("original")
    assert C.validate_lineage(a,b,"original",False)["parent_CPU_strict_gate_passed"]
    a,b=lineage("spatial")
    with pytest.raises(ValueError, match="diagnostic"):
        C.validate_lineage(a,b,"spatial",False)
    assert not C.validate_lineage(a,b,"spatial",True)["parent_CPU_strict_gate_passed"]


@pytest.mark.parametrize("mutation", ["float_indices", "incomplete", "promote_spatial"])
def test_wrong_contract_incomplete_suffix_or_promoted_parent_rejected(mutation):
    a,b=copy.deepcopy(lineage("original"))
    if mutation=="float_indices":a["inputs"][1]["dtype"]="float32"
    elif mutation=="incomplete":b["calls"].pop()
    else:b["original_CPU_strict_gate_each_method"]["spatial"]=True
    with pytest.raises(ValueError):C.validate_lineage(a,b,"original",False)


@pytest.mark.parametrize("fail", [None, "build", "build_release"])
def test_exact_compiler_operations_release_and_primary_error(tmp_path,fail):
    events=[]
    class Fake:
        def config(self,**kw):
            assert kw==dict(target_platform="rk3588",float_dtype="float16",optimization_level=3)
            events.append("config");return 0
        def load_onnx(self,**kw):events.append("load");return 0
        def build(self,**kw):
            assert kw==dict(do_quantization=False)
            events.append("build");return -1 if fail else 0
        def export_rknn(self,path):events.append("export");Path(path).write_bytes(b"tiny fake");return 0
        def release(self):
            events.append("release")
            if fail=="build_release":raise RuntimeError("secondary release")
    report=dict(operations=[],cleanup_errors=[],released=False)
    if fail:
        with pytest.raises(ValueError,match="build"):
            C.perform_compile(lambda **kw:Fake(),Path("fake.onnx"),tmp_path,report)
        assert events==["config","load","build","release"]
        assert bool(report["cleanup_errors"]) is (fail=="build_release")
    else:
        C.perform_compile(lambda **kw:Fake(),Path("fake.onnx"),tmp_path,report)
        assert events==["config","load","build","export","release"]
        assert report["released"] and report["output"]["size"]==9
