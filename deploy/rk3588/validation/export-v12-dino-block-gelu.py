#!/usr/bin/env python3
"""Rewrite the frozen block's exact GELU into the Torch-export fusion pattern.

Only ONNX construction/checking is performed. No model inference runs here.
The mathematical function remains exact Erf GELU; FP32 operation ordering and
reciprocal rounding differ, so board numerical validation is still required.
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
from onnx import checker, helper, numpy_helper, shape_inference


STANDARD_SHA256 = "9baf8be0955376c6381dd71c9ea25db508342f192b48a26c2f1f6c642989f5c3"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_gelu(model):
    expected = {
        "gelu_scaled_input": ("Mul", ["mlp.fc1", "gelu_inv_sqrt_two"]),
        "gelu_erf": ("Erf", ["gelu_scaled_input"]),
        "gelu_erf_plus_one": ("Add", ["gelu_erf", "gelu_one"]),
        "gelu_half_input": ("Mul", ["mlp.fc1", "gelu_half"]),
        "gelu": ("Mul", ["gelu_half_input", "gelu_erf_plus_one"]),
    }
    by_name = {node.name: node for node in model.graph.node}
    for name, (kind, inputs) in expected.items():
        node = by_name.get(name)
        if node is None or node.op_type != kind or list(node.input) != inputs or list(node.output) != [name]:
            raise ValueError(f"Frozen exact GELU pattern mismatch at {name}")
    positions = [i for i, node in enumerate(model.graph.node) if node.name in expected]
    if positions != list(range(positions[0], positions[0] + 5)):
        raise ValueError("Frozen GELU nodes must form one contiguous five-node region")
    consumers = {node.name for node in model.graph.node
                 if any(value in (set(expected) - {"gelu"}) for value in node.input)}
    if consumers != {"gelu_erf", "gelu_erf_plus_one", "gelu"}:
        raise ValueError("GELU intermediate has an unexpected external consumer")
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    for name, value in (("gelu_inv_sqrt_two", 1.0 / math.sqrt(2.0)), ("gelu_one", 1.0), ("gelu_half", 0.5)):
        array = initializers[name]
        if array.dtype != np.float32 or array.shape != () or array.item() != np.float32(value).item():
            raise ValueError(f"Frozen GELU scalar mismatch: {name}")
    graph_before = {node.name: node.SerializeToString() for node in model.graph.node if node.name not in expected}
    weights_before = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer
                      if tensor.name != "gelu_inv_sqrt_two"}
    replacement = [
        helper.make_node("Div", ["mlp.fc1", "gelu_sqrt_two"], ["gelu_scaled_input"], name="gelu_scaled_input"),
        helper.make_node("Erf", ["gelu_scaled_input"], ["gelu_erf"], name="gelu_erf"),
        helper.make_node("Add", ["gelu_erf", "gelu_one"], ["gelu_erf_plus_one"], name="gelu_erf_plus_one"),
        helper.make_node("Mul", ["mlp.fc1", "gelu_erf_plus_one"], ["gelu_before_half"], name="gelu_before_half"),
        helper.make_node("Mul", ["gelu_before_half", "gelu_half"], ["gelu"], name="gelu"),
    ]
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes[:positions[0]] + replacement + nodes[positions[-1] + 1:])
    tensors = [tensor for tensor in model.graph.initializer if tensor.name != "gelu_inv_sqrt_two"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(tensors)
    model.graph.initializer.append(numpy_helper.from_array(np.array(math.sqrt(2.0), dtype=np.float32),
                                                         name="gelu_sqrt_two"))
    # Recompute all intermediate metadata; the removed half-input is not live.
    del model.graph.value_info[:]
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True, check_type=True, data_prop=True)
    checker.check_model(model, full_check=True)
    assert graph_before == {node.name: node.SerializeToString() for node in model.graph.node
                            if node.name not in {n.name for n in replacement}}
    assert weights_before == {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer
                              if tensor.name != "gelu_sqrt_two"}
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.input, args.output, args.report)]
    if len(set(paths)) != 3 or args.output.exists() or args.report.exists():
        parser.error("Use distinct input and new output/report paths")
    source_hash = sha256(args.input)
    if source_hash != STANDARD_SHA256:
        parser.error("Input must be the exact frozen standard block ONNX artifact")
    script_hash = sha256(__file__)
    model = rewrite_gelu(onnx.load(args.input))
    properties = {item.key: item.value for item in model.metadata_props}
    properties.update(gelu_pattern="Div -> Erf -> Add(1) -> Mul(hidden) -> Mul(0.5)",
                      original_standard_onnx_sha256=source_hash, gelu_rewriter_sha256=script_hash)
    helper.set_model_props(model, properties)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(model.SerializeToString())
    checker.check_model(str(args.output), full_check=True)
    if sha256(args.input) != source_hash or sha256(__file__) != script_hash:
        raise RuntimeError("Frozen source or rewriter changed during export")
    report = dict(status="rewritten_and_structurally_checked", script_sha256=script_hash,
                  original_standard_sha256=source_hash, original_standard_unchanged=True,
                  output=dict(path=str(args.output.resolve()), sha256=sha256(args.output),
                              bytes=args.output.stat().st_size),
                  node_count=len(model.graph.node), op_counts=dict(Counter(n.op_type for n in model.graph.node)),
                  onnx_version=onnx.__version__, opset=17, ir_version=model.ir_version,
                  mathematical_function="exact Erf GELU; no tanh approximation",
                  pattern="Div(hidden,sqrt(2))->Erf->Add(1)->Mul(hidden)->Mul(0.5)",
                  unchanged_weights_and_non_gelu_nodes=True,
                  fp32_operation_order_changed=True, fp32_numerical_equivalence_validated=False,
                  model_inference_executed=False, rknn_compilation_validated=False,
                  npu_execution_validated=False, realtime_validated=False,
                  limitations=["Only graph algebra is established; FP32 ordering/constant rounding differ.",
                               "Board ORT and compiled RKNN must be checked independently."])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
