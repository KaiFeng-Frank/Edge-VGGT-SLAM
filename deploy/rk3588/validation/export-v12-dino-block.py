#!/usr/bin/env python3
"""Export one captured, fixed-shape DINO block without executing any model.

This is an experimental ONNX artifact generator, not a production backend.
It reads CPU FP32 block state and prepared tokens captured on the ARM board.
ONNX checker/shape inference do not establish numerical or RKNN compatibility.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


OPSET = 17
LAYER_NORM_EPSILON = 1e-6
STATE_KEYS = (
    "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
    "attn.proj.weight", "attn.proj.bias", "ls1.gamma", "norm2.weight",
    "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight",
    "mlp.fc2.bias", "ls2.gamma",
)
SUPPORT_REFERENCE = (
    "https://github.com/airockchip/rknn-toolkit2/blob/master/doc/"
    "RKNNToolKit2_OP_Support-2.3.2.md"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_state(state: dict[str, np.ndarray], dim: int) -> None:
    if set(state) != set(STATE_KEYS):
        raise ValueError(
            f"Exact block state required; missing={sorted(set(STATE_KEYS) - set(state))}, "
            f"extra={sorted(set(state) - set(STATE_KEYS))}"
        )
    expected = {
        **{key: (dim,) for key in STATE_KEYS if key.endswith("gamma") or key.startswith("norm")},
        "attn.qkv.weight": (3 * dim, dim), "attn.qkv.bias": (3 * dim,),
        "attn.proj.weight": (dim, dim), "attn.proj.bias": (dim,),
        "mlp.fc1.weight": (4 * dim, dim), "mlp.fc1.bias": (4 * dim,),
        "mlp.fc2.weight": (dim, 4 * dim), "mlp.fc2.bias": (dim,),
    }
    for key in STATE_KEYS:
        value = state[key]
        if value.dtype != np.dtype("float32") or value.shape != expected[key]:
            raise ValueError(f"{key}: expected FP32 {expected[key]}, got {value.dtype} {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key}: nonfinite weights")


def build_block_onnx(
    state: dict[str, np.ndarray],
    input_shape: tuple[int, int, int],
    *,
    heads: int = 16,
    layer_norm: str = "standard",
) -> onnx.ModelProto:
    """Build a single block. Smaller dimensions are for structural self-tests only."""
    if layer_norm not in ("standard", "decomposed"):
        raise ValueError("layer_norm must be standard or decomposed")
    if len(input_shape) != 3 or any(type(n) is not int or n <= 0 for n in input_shape):
        raise ValueError("input_shape must contain three positive integers")
    batch, tokens, dim = input_shape
    if type(heads) is not int or heads <= 0 or dim % heads:
        raise ValueError("heads must divide the channel dimension")
    validate_state(state, dim)
    head_dim = dim // heads
    nodes = []
    initializers = []

    def constant(name: str, value: np.ndarray) -> str:
        # np.ascontiguousarray promotes a scalar to shape [1], changing Gather rank.
        initializers.append(numpy_helper.from_array(np.array(value, copy=True, order="C"), name=name))
        return name

    def node(op: str, inputs: list[str], output: str, **attributes) -> str:
        nodes.append(helper.make_node(op, inputs, [output], name=output, **attributes))
        return output

    def norm(name: str, x: str) -> str:
        weight = constant(f"{name}.weight", state[f"{name}.weight"])
        bias = constant(f"{name}.bias", state[f"{name}.bias"])
        if layer_norm == "standard":
            return node("LayerNormalization", [x, weight, bias], name, axis=-1,
                        epsilon=LAYER_NORM_EPSILON, stash_type=TensorProto.FLOAT)
        mean = node("ReduceMean", [x], f"{name}.mean", axes=[-1], keepdims=1)
        centered = node("Sub", [x, mean], f"{name}.centered")
        square = node("Mul", [centered, centered], f"{name}.square")
        variance = node("ReduceMean", [square], f"{name}.variance", axes=[-1], keepdims=1)
        epsilon = constant(f"{name}.epsilon", np.array(LAYER_NORM_EPSILON, dtype=np.float32))
        denominator = node("Sqrt", [node("Add", [variance, epsilon], f"{name}.variance_eps")],
                           f"{name}.denominator")
        normalized = node("Div", [centered, denominator], f"{name}.normalized")
        return node("Add", [node("Mul", [normalized, weight], f"{name}.scaled"), bias], name)

    def linear(name: str, x: str) -> str:
        # PyTorch stores [out, in]; ONNX MatMul takes [in, out].
        weight = constant(f"{name}.weight_transposed", state[f"{name}.weight"].T)
        bias = constant(f"{name}.bias", state[f"{name}.bias"])
        return node("Add", [node("MatMul", [x, weight], f"{name}.matmul"), bias], name)

    x1 = norm("norm1", "input")
    qkv = linear("attn.qkv", x1)
    qkv_shape = constant("qkv_shape", np.array([batch, tokens, 3, heads, head_dim], dtype=np.int64))
    qkv = node("Transpose", [node("Reshape", [qkv, qkv_shape], "qkv_reshape")],
               "qkv_transpose", perm=[2, 0, 3, 1, 4])
    components = []
    for index, name in enumerate(("query", "key", "value")):
        # Scalar Gather removes the Q/K/V dimension, giving [B,H,N,Dh].
        indices = constant(f"{name}_index", np.array(index, dtype=np.int64))
        components.append(node("Gather", [qkv, indices], name, axis=0))
    query, key, value = components
    key_t = node("Transpose", [key], "key_transpose", perm=[0, 1, 3, 2])
    scores = node("MatMul", [query, key_t], "attention_scores")
    scale = constant("attention_scale", np.array(head_dim ** -0.5, dtype=np.float32))
    scores = node("Mul", [scores, scale], "attention_scaled_scores")
    probability = node("Softmax", [scores], "attention_probabilities", axis=-1)
    attended = node("MatMul", [probability, value], "attention_values")
    attended = node("Transpose", [attended], "attention_heads_transpose", perm=[0, 2, 1, 3])
    output_shape = constant("output_shape", np.array(input_shape, dtype=np.int64))
    attended = node("Reshape", [attended, output_shape], "attention_heads_flatten")
    projected = linear("attn.proj", attended)
    gamma1 = constant("ls1.gamma", state["ls1.gamma"])
    residual1 = node("Add", ["input", node("Mul", [projected, gamma1], "ls1")], "residual1")
    hidden = linear("mlp.fc1", norm("norm2", residual1))
    inv_sqrt_two = constant("gelu_inv_sqrt_two", np.array(1.0 / math.sqrt(2.0), dtype=np.float32))
    one = constant("gelu_one", np.array(1.0, dtype=np.float32))
    half = constant("gelu_half", np.array(0.5, dtype=np.float32))
    erf = node("Erf", [node("Mul", [hidden, inv_sqrt_two], "gelu_scaled_input")], "gelu_erf")
    gelu = node("Mul", [node("Mul", [hidden, half], "gelu_half_input"),
                        node("Add", [erf, one], "gelu_erf_plus_one")], "gelu")
    mlp = linear("mlp.fc2", gelu)
    gamma2 = constant("ls2.gamma", state["ls2.gamma"])
    node("Add", [residual1, node("Mul", [mlp, gamma2], "ls2")], "output")
    graph = helper.make_graph(
        nodes, "captured_vggt_spark_dino_block0",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(input_shape))],
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="vggt-slam2-rk3588s/manual-block-export",
                              opset_imports=[helper.make_opsetid("", OPSET)])
    # IR8 is sufficient for this graph and avoids a host ONNX version raising the IR.
    model.ir_version = 8
    model.doc_string = (
        "Inference-only captured DINO block0. No RoPE/QK normalization/dropout. "
        "Explicit SDPA mathematics may differ numerically from PyTorch CPU Flash Attention. "
        "ONNX validation is not RKNN execution or numerical validation."
    )
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    checker.check_model(model, full_check=True)
    return model


def structural_self_test() -> dict:
    dim = 8
    shape = (1, 7, dim)
    state = {
        key: np.ones(value_shape, dtype=np.float32)
        for key, value_shape in {
            **{key: (dim,) for key in STATE_KEYS if key.endswith("gamma") or key.startswith("norm")},
            "attn.qkv.weight": (3 * dim, dim), "attn.qkv.bias": (3 * dim,),
            "attn.proj.weight": (dim, dim), "attn.proj.bias": (dim,),
            "mlp.fc1.weight": (4 * dim, dim), "mlp.fc1.bias": (4 * dim,),
            "mlp.fc2.weight": (dim, 4 * dim), "mlp.fc2.bias": (dim,),
        }.items()
    }
    state["mlp.fc1.weight"] = np.arange(4 * dim * dim, dtype=np.float32).reshape(4 * dim, dim)
    state["ls1.gamma"] = np.arange(dim, dtype=np.float32) / 10.0 + 0.25
    checked = []
    for mode in ("standard", "decomposed"):
        model = build_block_onnx(state, shape, heads=2, layer_norm=mode)
        ops = Counter(n.op_type for n in model.graph.node)
        assert ops["MatMul"] == 6 and ops["Softmax"] == 1 and ops["Erf"] == 1
        assert ops["Gather"] == 3 and ops["LayerNormalization"] == (2 if mode == "standard" else 0)
        assert [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim] == list(shape)
        value_shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                        for v in model.graph.value_info}
        assert value_shapes["query"] == [1, 2, 7, 4]
        assert value_shapes["attention_scores"] == [1, 2, 7, 7]
        initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        assert initializers["attention_scale"].dtype == np.float32
        assert initializers["attention_scale"].item() == 0.5
        assert initializers["query_index"].shape == ()
        assert np.array_equal(initializers["mlp.fc1.weight_transposed"], state["mlp.fc1.weight"].T)
        assert np.array_equal(initializers["ls1.gamma"], state["ls1.gamma"])
        by_name = {n.name: n for n in model.graph.node}
        assert list(by_name["residual1"].input) == ["input", "ls1"]
        assert list(by_name["output"].input) == ["residual1", "ls2"]
        if mode == "standard":
            for name in ("norm1", "norm2"):
                attrs = {a.name: helper.get_attribute_value(a) for a in by_name[name].attribute}
                assert attrs["axis"] == -1 and attrs["stash_type"] == TensorProto.FLOAT
                assert abs(attrs["epsilon"] - LAYER_NORM_EPSILON) < 1e-12
        checked.append({"mode": mode, "nodes": len(model.graph.node), "op_counts": dict(ops)})
    for bad in (dict(state, **{"ls1.gamma": np.ones((1,), np.float32)}),
                {key: value for key, value in state.items() if key != "norm1.bias"},
                dict(state, **{"norm1.bias": np.full((dim,), np.nan, np.float32)}),
                dict(state, **{"attn.qkv.weight": state["attn.qkv.weight"].astype(np.float16)})):
        try:
            build_block_onnx(bad, shape, heads=2)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid state was accepted")
    return {"passed": True, "graph_variants": checked, "invalid_state_cases": 4,
            "synthetic_only": True, "inference_executed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Captured prepared block input.npy")
    parser.add_argument("--state", type=Path, help="Captured block-state.npz; exact original dotted state keys")
    parser.add_argument("--capture-json", type=Path, help="Board capture provenance/architecture JSON")
    parser.add_argument("--output", type=Path, help="New output ONNX file, preferably in /dev/shm")
    parser.add_argument("--report", type=Path, help="New export report JSON")
    parser.add_argument("--layer-norm", choices=("standard", "decomposed"), default="standard")
    parser.add_argument("--self-test", action="store_true", help="Check small synthetic graph structure; no inference")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(structural_self_test(), sort_keys=True))
        return 0
    if any(getattr(args, key) is None for key in ("input", "state", "capture_json", "output", "report")):
        parser.error("--input, --state, --capture-json, --output and --report are required")
    paths = {"input": args.input.resolve(), "state": args.state.resolve(),
             "capture": args.capture_json.resolve(), "exporter": Path(__file__).resolve()}
    if len(set(paths.values()) | {args.output.resolve(), args.report.resolve()}) != 6:
        parser.error("Inputs and outputs must have distinct paths")
    if args.output.exists() or args.report.exists():
        parser.error("Output/report already exists; choose new paths")
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    capture = json.loads(paths["capture"].read_text())
    if capture.get("status") != "completed" or capture.get("kind") != "official_dino_block0_CPU_capture":
        raise ValueError("A completed original DINO block0 board capture is required")
    for key in ("source_unchanged", "image_unchanged", "checkpoint_stat_unchanged"):
        if capture.get(key) is not True:
            raise ValueError(f"Capture provenance {key} must be true")
    if capture.get("source_hashes") != capture.get("source_hashes_after"):
        raise ValueError("Captured upstream sources changed")
    cpu_reference = capture.get("cpu_reference", {})
    for key in ("all_elementwise_equal", "finite", "no_grad", "eval"):
        if cpu_reference.get(key) is not True:
            raise ValueError(f"Captured CPU reference {key} must be true")
    if cpu_reference.get("device") != "cpu" or cpu_reference.get("dtype") != "float32":
        raise ValueError("Original reference must be CPU FP32")
    artifacts = capture["artifacts"]
    for name, filename in (("input", "input.npy"), ("state", "block-state.npz")):
        info = artifacts[filename]
        if info.get("sha256") != hashes_before[name] or info.get("bytes") != paths[name].stat().st_size:
            raise ValueError(f"{filename}: file does not match its board capture receipt")
    architecture = capture["architecture"]
    expected_architecture = {
        "embed_dim": 1024, "num_heads": 16, "head_dim": 64,
        "patch_size": 14, "mlp_hidden_dim": 4096,
        "norm1_eps": LAYER_NORM_EPSILON, "norm2_eps": LAYER_NORM_EPSILON,
        "gelu_approximate": "none", "qk_norm": False, "rope": False,
        "dropout": 0, "drop_path": 0, "eval": True,
        "num_register_tokens": 4, "cls_tokens": 1, "patch_tokens": 777,
        "sequence_length": 782, "position_included_in_input": True,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"Captured architecture {key} must be {expected!r}")
    image_input = np.load(paths["input"], allow_pickle=False, mmap_mode="r")
    if image_input.dtype != np.float32 or image_input.shape != (1, 782, 1024):
        raise ValueError(f"Expected captured FP32 [1,782,1024], got {image_input.dtype} {image_input.shape}")
    if not np.isfinite(image_input).all():
        raise ValueError("Captured input is not finite")
    input_info = artifacts["input.npy"]
    if (input_info.get("shape") != list(image_input.shape) or input_info.get("dtype") != "float32"
            or input_info.get("finite") is not True
            or input_info.get("tensor_sha256") != hashlib.sha256(memoryview(image_input).cast("B")).hexdigest()):
        raise ValueError("Captured input tensor metadata/content mismatch")
    with np.load(paths["state"], allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    tensor_info = artifacts["block-state.npz"]["tensors"]
    if set(tensor_info) != set(state):
        raise ValueError("Captured state tensor keys do not match receipt")
    for key, value in state.items():
        info = tensor_info[key]
        contiguous = np.ascontiguousarray(value)
        if (info.get("shape") != list(value.shape) or info.get("dtype") != "float32"
                or info.get("finite") is not True or info.get("bytes") != value.nbytes
                or info.get("sha256") != hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()):
            raise ValueError(f"{key}: captured parameter tensor metadata/content mismatch")
    model = build_block_onnx(state, tuple(image_input.shape), heads=16, layer_norm=args.layer_norm)
    helper.set_model_props(model, {"captured_input_sha256": hashes_before["input"],
                                   "captured_block_state_sha256": hashes_before["state"],
                                   "capture_json_sha256": hashes_before["capture"],
                                   "exporter_sha256": hashes_before["exporter"],
                                   "layer_norm_mode": args.layer_norm})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(model.SerializeToString())
    checker.check_model(str(args.output), full_check=True)
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("Export inputs/source changed during export")
    weights_bytes = sum(value.nbytes for value in state.values())
    report = {
        "status": "exported_and_structurally_checked", "inference_executed": False,
        "numerical_equivalence_validated": False, "rknn_compilation_validated": False,
        "npu_execution_validated": False, "real_time_validated": False,
        "input_shape": list(image_input.shape), "input_dtype": "float32", "output_dtype": "float32",
        "onnx_version": onnx.__version__, "opset": OPSET, "ir_version": model.ir_version,
        "layer_norm": {"mode": args.layer_norm, "axis": -1, "epsilon": LAYER_NORM_EPSILON,
                       "statistics_dtype": "float32", "variance": "population"},
        "attention": {"num_heads": 16, "head_dim": 64, "scale": 0.125,
                      "causal": False, "mask": None, "qk_norm": False, "rope": False,
                      "implementation": "explicit MatMul/Softmax/MatMul"},
        "gelu": {"approximate": "none", "implementation": "0.5*x*(1+erf(x/sqrt(2)))"},
        "tokens": {"cls": 1, "registers": 4, "patches": 777,
                   "absolute_positions_already_in_input": True},
        "state_keys": sorted(state), "captured_parameter_bytes": weights_bytes,
        "node_count": len(model.graph.node), "operator_counts": dict(Counter(n.op_type for n in model.graph.node)),
        "initializer_count": len(model.graph.initializer),
        "initializer_bytes": sum(len(i.raw_data) for i in model.graph.initializer),
        "tensor_shapes": {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                          for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]},
        "captured_architecture": architecture, "source_sha256_before": hashes_before,
        "source_sha256_after": hashes_after, "source_unchanged": True,
        "captured_upstream_source_sha256": capture["source_hashes"],
        "captured_checkpoint": capture.get("checkpoint"), "captured_image": capture.get("image"),
        "captured_cpu_reference": cpu_reference,
        "output": {"path": str(args.output.resolve()), "bytes": args.output.stat().st_size,
                   "sha256": sha256(args.output)}, "support_reference": SUPPORT_REFERENCE,
        "limitations": ["FP32 mathematical graph; runtime precision/placement must be separately measured.",
                        "Explicit attention and normalization may round differently from PyTorch CPU Flash Attention.",
                        "Original weight values are exported; learned LayerScale gamma is not replaced by its initialization.",
                        "Only this captured fixed shape is represented; no padding tokens or dynamic axes."],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "node_count": report["node_count"],
                      "output_sha256": report["output"]["sha256"], "report": str(args.report.resolve()),
                      "inference_executed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
