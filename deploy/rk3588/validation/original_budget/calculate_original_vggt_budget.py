#!/usr/bin/env python3
"""Original VGGT static arithmetic and saved-metadata audit; stdlib only.

This script does not import or execute VGGT, Torch, NumPy, ORT or RKNN.
One MAC is counted as two arithmetic FLOPs. No hardware peak is assumed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from math import prod
from pathlib import Path


SOURCE_PINS = {
    "models/vggt.py": "64ce0625effa8569cac5da527532defa22bb3d3f1e4facfaf54fbd329740bde6",
    "models/aggregator.py": "623a943847a3259292bfd5c8675d8b80b09abc845eefadba7622e6a901cb9cfd",
    "layers/vision_transformer.py": "f781186de8e9a52acc4dfc07820cde01bb182a3eb6835f630b863eaf2cde6bde",
    "layers/block.py": "d8b8cb1bfe5f084d33810b9cc96626fa494a4beea2c3c175da197841070bbf17",
    "layers/attention.py": "9d80ff6c6f562efc29d806125fe47bd9b504a6356ce2d9b5d61e8ee4ea770a00",
    "layers/mlp.py": "255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa",
    "layers/patch_embed.py": "c2543ec293951d081db662b2125027625f6e632971630facaa7f911c2377d335",
    "heads/camera_head.py": "78b870018b20821d27587122003e68afd4f6b977bc13e0952fe507c31d6d3da2",
    "heads/dpt_head.py": "72e9ed26196113eb71e26b641b4262606b5aacc797accafc6c975f5db0d2ace2",
}
CHECKPOINT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def function(tree, name, cls=None):
    body = tree.body
    if cls is not None:
        matches = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
        require(len(matches) == 1, f"Expected class {cls}")
        body = matches[0].body
    matches = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(matches) == 1, f"Expected function {cls}.{name}")
    return matches[0]


def defaults(node):
    names = [arg.arg for arg in node.args.args]
    result = {}
    for name, value in zip(names[-len(node.args.defaults):], node.args.defaults):
        try:
            result[name] = ast.literal_eval(value)
        except (ValueError, TypeError):
            pass
    return result


def call_keywords(node, callee):
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == callee]
    require(len(calls) == 1, f"Expected one {callee} call")
    values = {}
    for keyword in calls[0].keywords:
        if keyword.arg is not None:
            try:
                values[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                pass
    return values


def audited_configuration(source):
    trees, proof = {}, {}
    for name, expected in SOURCE_PINS.items():
        path = source / name
        require(sha(path) == expected, f"Original source SHA mismatch: {name}")
        text = path.read_text()
        trees[name] = ast.parse(text)
        proof[name] = {"sha256": expected, "bytes": path.stat().st_size}
    agg = defaults(function(trees["models/aggregator.py"], "__init__", "Aggregator"))
    dino = call_keywords(function(trees["layers/vision_transformer.py"], "vit_large"),
                         "DinoVisionTransformer")
    camera = defaults(function(trees["heads/camera_head.py"], "__init__", "CameraHead"))
    iterations = defaults(function(trees["heads/camera_head.py"], "forward", "CameraHead"))
    dense = defaults(function(trees["heads/dpt_head.py"], "__init__", "DPTHead"))
    require({k: agg[k] for k in ("embed_dim", "depth", "num_heads", "mlp_ratio", "patch_size",
            "num_register_tokens", "aa_order", "aa_block_size", "patch_embed")} == {
            "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0,
            "patch_size": 14, "num_register_tokens": 4, "aa_order": ["frame", "global"],
            "aa_block_size": 1, "patch_embed": "dinov2_vitl14_reg"},
            "Unexpected original aggregator configuration")
    require({k: dino[k] for k in ("embed_dim", "depth", "num_heads", "mlp_ratio")} == {
            "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4},
            "Unexpected original internal DINO configuration")
    require({k: camera[k] for k in ("dim_in", "trunk_depth", "num_heads", "mlp_ratio")} == {
            "dim_in": 2048, "trunk_depth": 4, "num_heads": 16, "mlp_ratio": 4}
            and iterations["num_iterations"] == 4, "Unexpected original camera configuration")
    require({k: dense[k] for k in ("features", "out_channels", "intermediate_layer_idx", "patch_size")} == {
            "features": 256, "out_channels": [256, 512, 1024, 1024],
            "intermediate_layer_idx": [4, 11, 17, 23], "patch_size": 14},
            "Unexpected original depth configuration")
    for name, cls, fun in (("models/aggregator.py", "Aggregator", "forward"),
                           ("models/aggregator.py", "Aggregator", "_process_global_attention"),
                           ("layers/attention.py", "Attention", "forward"),
                           ("heads/camera_head.py", "CameraHead", "trunk_fn"),
                           ("heads/dpt_head.py", "DPTHead", "scratch_forward")):
        node = function(trees[name], fun, cls)
        proof[name].setdefault("audited_functions", []).append({
            "class": cls, "name": fun, "line": node.lineno,
            "ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
    return {"embedding": agg["embed_dim"], "depth_per_group": agg["depth"],
            "DINO_depth": dino["depth"], "heads": agg["num_heads"],
            "MLP_expansion": int(agg["mlp_ratio"]), "patch_size": agg["patch_size"],
            "DINO_special_tokens": 1 + agg["num_register_tokens"],
            "aggregator_special_tokens": 1 + agg["num_register_tokens"],
            "camera_embedding": camera["dim_in"], "camera_depth": camera["trunk_depth"],
            "camera_iterations": iterations["num_iterations"], "DPT_features": dense["features"],
            "DPT_channels": dense["out_channels"], "DPT_layers": dense["intermediate_layer_idx"]}, proof


def parameter_counts(c):
    d, depth, cd = c["embedding"], c["depth_per_group"], c["camera_embedding"]
    base = 12 * d * d + 15 * d
    # Affine QK normalization is per head-dimension, not per full embedding.
    normalized = base + 4 * (d // c["heads"])
    f, channels = c["DPT_features"], c["DPT_channels"]
    residual_unit = 2 * (9 * f * f + f)
    fusion_no_skip = residual_unit + f * f + f
    fusion_with_skip = 2 * residual_unit + f * f + f
    dense = (2 * cd + sum(cd * oc + oc for oc in channels)
             + 16 * channels[0] ** 2 + channels[0]
             + 4 * channels[1] ** 2 + channels[1]
             + 9 * channels[3] ** 2 + channels[3]
             + sum(9 * oc * f for oc in channels)
             + fusion_no_skip + 3 * fusion_with_skip
             + 9 * f * (f // 2) + f // 2
             + 9 * (f // 2) * 32 + 32 + 32 * 2 + 2)
    counts = {
        "original_DINO_encoder": depth * base + 3 * 14 ** 2 * d + d + 2 * d
             + 4 * d + ((518 // 14) ** 2 + 1) * d + 2 * d,
        "original_24_frame_24_global_blocks": 2 * depth * normalized,
        "original_aggregator_camera_register_tokens": 10 * d,
        "original_camera_head": 4 * (12 * cd ** 2 + 15 * cd) + 4 * cd + 9
             + (9 + 1) * cd + (cd + 1) * 3 * cd
             + (cd + 1) * (cd // 2) + (cd // 2 + 1) * 9,
        "original_depth_head": dense,
    }
    require(sum(counts.values()) == 1157941492, "Original active parameter sum mismatch")
    return counts


def depth_mac(h, w, c):
    ph, pw = h // c["patch_size"], w // c["patch_size"]
    q, q4, f = ph * pw, ((ph + 1) // 2) * ((pw + 1) // 2), c["DPT_features"]
    channels, cd = c["DPT_channels"], c["camera_embedding"]
    return {
        "projects_1x1": q * cd * sum(channels),
        "resize_convolutions": q * channels[0] ** 2 * 16 + q * channels[1] ** 2 * 4
            + q4 * channels[3] ** 2 * 9,
        "scratch_3x3": 9 * f * (16 * q * channels[0] + 4 * q * channels[1]
            + q * channels[2] + q4 * channels[3]),
        "fusion_spatial_3x3": 9 * f * f * (2 * q4 + 4 * q + 16 * q + 64 * q),
        "fusion_upsampled_1x1": f * f * (q + 4 * q + 16 * q + 64 * q),
        "output_3x3_1x1": 64 * q * f * (f // 2) * 9
            + h * w * ((f // 2) * 32 * 9 + 32 * 2),
    }


def case(h, w, s, c, evidence):
    d, depth, cd = c["embedding"], c["depth_per_group"], c["camera_embedding"]
    require(h % c["patch_size"] == w % c["patch_size"] == 0 and s >= 2,
            "Budget case requires complete patch-grid images and at least one overlap image")
    q = (h // c["patch_size"]) * (w // c["patch_size"])
    p = q + c["aggregator_special_tokens"]
    linear_per_group = depth * s * p * 12 * d * d
    frame_attention = depth * s * 2 * p * p * d
    global_attention = depth * 2 * (s * p) ** 2 * d
    patch = s * q * 3 * c["patch_size"] ** 2 * d
    camera_linear = c["camera_depth"] * c["camera_iterations"] * 12 * s * cd * cd
    camera_attention = c["camera_depth"] * c["camera_iterations"] * 2 * s * s * cd
    branch_per_iteration_weights = 9 * cd + 3 * cd * cd + cd * (cd // 2) + (cd // 2) * 9
    camera_branches = c["camera_iterations"] * s * branch_per_iteration_weights
    dense = depth_mac(h, w, c)
    backbone = 3 * linear_per_group + 2 * frame_attention + global_attention + patch
    camera = camera_linear + camera_attention + camera_branches
    full = backbone + camera + s * sum(dense.values())
    cached_old_DINO = patch // s + (linear_per_group + frame_attention) // s
    cached_full = full - cached_old_DINO
    cached_prefix = backbone + camera - cached_old_DINO
    fresh = s - 1
    updates = 20 / fresh
    paired_mac = 2 * (s * p * 12 * d * d) + 2 * s * p * p * d + 2 * (s * p) ** 2 * d
    return {
        "input_shape_SCHW": [s, 3, h, w], "input_evidence": evidence,
        "patch_grid": [h // 14, w // 14], "patch_tokens": q,
        "tokens_per_image": p, "global_tokens": s * p,
        "MAC_by_component": {
            "DINO_patch_convolution": patch,
            "DINO_24_blocks_linear": linear_per_group,
            "DINO_24_blocks_attention_QK_AV": frame_attention,
            "frame_24_blocks_linear": linear_per_group,
            "frame_24_blocks_attention_QK_AV": frame_attention,
            "global_24_blocks_linear": linear_per_group,
            "global_24_blocks_attention_QK_AV": global_attention,
            "camera_16_trunk_calls_linear": camera_linear,
            "camera_16_trunk_calls_attention_QK_AV": camera_attention,
            "camera_four_iterations_branch_linear": camera_branches,
            "depth_head_convolutions": s * sum(dense.values())},
        "DPT_conventional_dense_MAC_per_image": dense,
        "counted_full_forward_MAC": full, "counted_full_forward_FLOP": 2 * full,
        "one_old_DINO_exact_reuse_MAC_saved": cached_old_DINO,
        "one_old_DINO_exact_reuse_fraction_of_full_MAC": cached_old_DINO / full,
        "counted_full_forward_with_one_old_DINO_reused_FLOP": 2 * cached_full,
        "backbone_MLP_fraction_of_backbone_block_MAC":
            3 * depth * s * p * 8 * d * d / (backbone - patch),
        "attention_QK_AV_fraction_of_backbone_block_MAC":
            (2 * frame_attention + global_attention) / (backbone - patch),
        "full_original_backbone_camera_prefix_MAC": backbone + camera,
        "full_original_backbone_camera_prefix_with_old_DINO_reused_MAC": cached_prefix,
        "prefix_is_usable_SLAM_pose_or_registered_map_proof": False,
        "fresh_observations_per_overlap_batch": fresh,
        "required_batches_per_second_for_20_fresh_poses": updates,
        "conditional_batch_wait_at_exactly_20Hz_evenly_spaced_fresh_arrivals_seconds": (fresh - 1) / 20,
        "conditional_100ms_budget_after_batch_wait_before_transfers_queues_seconds":
            0.1 - (fresh - 1) / 20,
        "required_average_effective_TFLOP_per_second_by_counting_convention": {
            "current_full_forward_each_batch": updates * 2 * full / 1e12,
            "full_forward_with_one_old_DINO_reused": updates * 2 * cached_full / 1e12,
            "backbone_camera_prefix_with_one_old_DINO_reused": updates * 2 * cached_prefix / 1e12,
            "backbone_linear_only_with_one_old_DINO_reused":
                updates * 2 * (3 * linear_per_group - linear_per_group // s) / 1e12},
        "service_time_budget_seconds_per_batch_to_avoid_unbounded_serial_queue": 1 / updates,
        "image_arrival_to_pose_P95_target_seconds": 0.1,
        "dense_head_at_2_to_5_seconds_instead_of_each_batch": {
            "hypothetical_dense_forward_rate_Hz": [0.2, 0.5],
            "average_total_required_TFLOP_per_second_with_cached_prefix": [
                updates * 2 * cached_prefix / 1e12 + rate * 2 * s * sum(dense.values()) / 1e12
                for rate in (0.2, 0.5)],
            "scheduling_implemented_or_SL4_association_correctness_validated": False},
        "one_complete_frame_global_pair_MAC": paired_mac,
        "one_complete_frame_global_pair_FLOP": 2 * paired_mac,
        "pair_FP32_cross_boundary_feature_payload_bytes": {
            "two_separate_graphs_input_output_calls": 4 * s * p * d * 4,
            "one_combined_graph_input_output_call": 2 * s * p * d * 4,
            "potential_intermediate_payload_saved": 2 * s * p * d * 4,
            "includes_positions_or_internal_layout_copies": False,
            "payload_is_measured_transfer_or_time_saving": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Original vggt package directory")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a fresh path")
    config, source_proof = audited_configuration(args.source_root)
    result_dir = args.public_root / "results"
    paths = {name: result_dir / name for name in (
        "profile_336_v17.json", "original_only_v47.json", "frame_block_v17.json",
        "global_block_v17.json", "camera_head_v15.json")}
    paths["camera_capture112.json"] = args.public_root / "deploy/rk3588/validation/vggt-v15-camera-capture112.json"
    receipts = {name: json.loads(path.read_text()) for name, path in paths.items()}
    profile, clean = receipts["profile_336_v17.json"], receipts["original_only_v47.json"]
    require(profile["checkpoint"]["sha256"] == clean["original_checkpoint"]["sha256"] == CHECKPOINT_SHA,
            "Original checkpoint metadata mismatch")
    for name in ("models/vggt.py", "models/aggregator.py", "heads/camera_head.py", "heads/dpt_head.py",
                 "layers/vision_transformer.py", "layers/block.py", "layers/attention.py", "layers/mlp.py"):
        require(profile["source_hashes"]["third_party/vggt/vggt/" + name] == SOURCE_PINS[name],
                f"Captured source disagrees with audited source: {name}")
    for group in ("dino_blocks", "frame_blocks", "global_blocks"):
        require(profile["stage_summaries"][group]["calls"] == 24, "Full block call count mismatch")
    require(profile["stage_summaries"]["camera_trunk"]["calls"] == 16,
            "Original four-iteration camera call count mismatch")
    d = config["embedding"]
    for name in ("dino0", "frame0", "global0"):
        capture = profile["blocks"][name]
        require(capture["num_heads"] == 16 and capture["head_dim"] == 64,
                "Original attention shape mismatch")
        for key, shape in {"attn.qkv.weight": [3*d, d], "attn.proj.weight": [d, d],
                           "mlp.fc1.weight": [4*d, d], "mlp.fc2.weight": [d, 4*d]}.items():
            require(capture["weights"][key]["shape"] == shape and
                    capture["weights"][key]["dtype"] == "float32", "Original matrix shape mismatch")
        require(sum(prod(v["shape"]) for k, v in capture["weights"].items()
                    if k in ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"))
                == 12 * d * d, "Captured block matrix size disagrees with closed form")
        require(all(v["bytes"] == 4 * prod(v["shape"]) for v in capture["weights"].values()),
                "Captured original block FP32 byte convention mismatch")
    camera_capture = receipts["camera_capture112.json"]
    camera_tensors = camera_capture["artifacts"]["camera-state.npz"]["tensors"]
    require(len(camera_tensors) == 69 and sum(prod(x["shape"]) for x in camera_tensors.values()) == 216174610,
            "Camera parameter metadata mismatch")
    require(camera_capture["camera_iterations"] == 4 and camera_capture["CPU_trunk_call_count"] == 48,
            "Saved camera iteration evidence mismatch")
    counts = parameter_counts(config)
    require(sum(counts.values()) == clean["original518_cache_diagnostic"]["parameter_count"],
            "Source-derived parameter count disagrees with complete captured model")
    inputs = [
        (profile["input_tensor"]["shape"], "profile_336_v17.json:input_tensor.shape"),
        (clean["new_runtime_board_regression"]["actual_tensor_shape"],
         "original_only_v47.json:new_runtime_board_regression.actual_tensor_shape"),
        (clean["original518_cache_diagnostic"]["actual_input_shape"],
         "original_only_v47.json:original518_cache_diagnostic.actual_input_shape"),
    ]
    workloads = [case(shape[2], shape[3], shape[0], config, evidence) for shape, evidence in inputs]
    # Explicit analytical batching case, not a saved execution of this geometry.
    workloads.append(case(252, 336, 3, config, "analytical S=3 batching scenario; no execution asserted"))
    cd = config["camera_embedding"]
    block_matrix_elements = 12 * d * d
    camera_matrix_elements = 4 * 12 * cd * cd + 9 * cd + 3 * cd * cd + cd * (cd // 2) + (cd // 2) * 9
    require(sum(prod(v["shape"]) for k, v in camera_tensors.items()
                if k.endswith(".weight") and len(v["shape"]) == 2) == camera_matrix_elements,
            "Captured camera matrix sizes disagree with source closed form")
    streamed_linear_bytes = 2 * (72 * block_matrix_elements + 4 * camera_matrix_elements)
    report = {
        "schema": 1, "kind": "original_VGGT_source_derived_compute_budget_v48",
        "execution": {"stdlib_only": True, "actual_model_OR_ORT_OR_RKNN_OR_compile_run": False,
                      "board_access": False, "production_source_modified": False},
        "counting": {"FLOP_per_MAC": 2,
            "scope": "Dense matrices, conventional padded dense convolution, and QK/AV multiplication only",
            "excluded": ["bias", "residual", "normalization", "softmax", "GELU", "SiLU",
                         "RoPE", "position embedding/interpolation", "depth activation", "memory/layout",
                         "queue", "image loading", "pose decoding", "SL4 graph", "map alignment"],
            "mathematical_budget_is_measured_instruction_count": False,
            "hardware_peak_or_precision_throughput_assumption_used": False},
        "formulas_MAC": {
            "N": "(H/14)*(W/14)+5, per image; B=1 in these cases",
            "block_linear": "3*S*N*D^2 + S*N*D^2 + 2*S*N*D*(4*D) = 12*S*N*D^2",
            "per_image_attention_one_group": "2*S*N^2*D, QK and AV; heads*(D/heads)=D",
            "global_attention_one_group": "2*(S*N)^2*D",
            "backbone": "24*(36*S*N*D^2 + 4*S*N^2*D + 2*(S*N)^2*D) + S*(N-5)*3*14^2*D",
            "camera_trunk": "16*(12*S*2048^2 + 2*S^2*2048)",
            "camera_branches": "4*S*(9*2048 + 3*2048^2 + 2048*1024 + 1024*9)",
            "cached_DINO_old_image": "(patch conv + complete 24-block DINO work)/S, one unchanged overlap image",
            "DPT": "Per-convolution components calculated from audited DPT spatial scales; see each workload"},
        "configuration_from_fixed_source_AST": config,
        "source_SHA_verified": source_proof,
        "saved_receipts": {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                           for name, path in paths.items()},
        "checkpoint_provenance_from_receipts": {"sha256": CHECKPOINT_SHA,
                                                "checkpoint_rehashed_by_this_script": False},
        "source_derived_active_parameters_by_component": counts,
        "active_parameter_count": sum(counts.values()),
        "workloads": workloads,
        "FP16_weight_payload_and_conditional_streaming": {
            "logical_unique_active_parameter_FP16_bytes": 2 * sum(counts.values()),
            "logical_unique_active_parameter_FP16_GiB": 2 * sum(counts.values()) / 2**30,
            "entire_original_FP16_model_execution_or_quality_validated": False,
            "backbone_72_blocks_matrix_FP16_bytes": 2 * 72 * block_matrix_elements,
            "camera_unique_matrix_FP16_bytes": 2 * camera_matrix_elements,
            "camera_four_iteration_logical_matrix_reads_FP16_bytes_if_no_inter_iteration_cache":
                2 * 4 * camera_matrix_elements,
            "backbone_camera_logical_matrix_reads_FP16_bytes_per_forward_if_each_use_is_streamed_once":
                streamed_linear_bytes,
            "conditional_streamed_matrix_bytes_per_second_at_20_S2_updates": 20 * streamed_linear_bytes,
            "conditional_streamed_matrix_GB_per_second_at_20_S2_updates": 20 * streamed_linear_bytes / 1e9,
            "scope": "Matrix weights only, no biases/norms/patch convolution/depth head/packing duplication",
            "measured_DRAM_bandwidth_or_minimum_IO_requirement": False,
            "note": "Device caches, weight packing, persistent contexts and repeated reads change actual traffic. DINO reuse removes old-image MACs, but one fresh image still touches each DINO layer's weights. Unique parameter payload, SDK allocation and host-to-device upload are different quantities.",
            "existing_complete_camera_RKNN_DMA_bytes": receipts["camera_head_v15.json"]["memory"]["SDK_reported_bytes"]["total_dma_allocated_size"],
            "SDK_DMA_not_whole_model_or_whole_board_memory": True},
        "existing_measured_evidence": {
            "original_336_S2_CPU_unhooked_full_forward_seconds": profile["unhooked_forward_seconds"],
            "separate_hooked_full_forward_seconds": profile["hooked_forward_seconds"],
            "hooked_stage_seconds_must_not_be_added_to_unhooked_measurement": True,
            "captured_complete_DINO_frame_global_call_counts": [24, 24, 24],
            "captured_camera_trunk_calls": 16,
            "frame0_NPU_warm_IO_mean_ms": 1000 * receipts["frame_block_v17.json"]["RKNN_normal"]["warm_IO_mean_seconds"],
            "global0_NPU_warm_IO_mean_ms": 1000 * receipts["global_block_v17.json"]["timing"]["normal"]["warm"]["mean"],
            "frame_global_measured_as_one_chained_NPU_pair": False,
            "frame0_strict_FP32_gate_passed": receipts["frame_block_v17.json"]["RKNN_normal"]["numerical"]["satisfies_FP32_export_gate"],
            "global0_strict_FP32_gate_passed": receipts["global_block_v17.json"]["numerical_policy"]["NPU_strict_FP32_equivalence_passed"],
            "complete_camera_NPU_last_three_mean_ms": receipts["camera_head_v15.json"]["normal_timing"]["last_three_mean_ms"],
            "complete_camera_NPU_strict_FP32_elementwise_gate_passed": receipts["camera_head_v15.json"]["error_against_original_FP32_CameraHead"]["strict_elementwise_rtol1e4_atol1e5"],
            "camera_measurement_is_repeated_saved_input_not_fresh_stream": True,
            "camera_saving_ratio_or_full_model_uniform_latency_extrapolated": False},
        "source_and_saved_metadata_selfchecks": {
            "nine_source_SHA_pins_and_literal_configuration": True,
            "captured_block_matrix_shapes_and_FP32_byte_counts_match_closed_form": True,
            "complete_24_24_24_and_16_call_counts": True,
            "camera_69_parameter_shapes_sum_to_source_parameter_formula": True,
            "camera_matrix_metadata_sum_matches_conditional_streaming_formula": True,
            "active_parameter_source_sum_matches_complete_model_receipt": True},
        "reuse_scope": {
            "proven_unchanged_reuse": "DINO normalized-pixel encoder tail for one identical old image and unchanged evaluation state",
            "frame_and_global_context_reused_by_this_budget": False,
            "why_global_tail_is_not_generic_cache": "Original global attention is bidirectional over S*N tokens. A fresh image changes attention keys, values and softmax denominators for old queries, then changes inputs to later frame/global layers. An autoregressive old KV cache would not compute the same original outputs.",
            "retained_layers": [4, 11, 17, 23],
            "retention_omits_transformer_compute": False},
        "limits": [
            "No hardware impossibility proof, sustained FP16 peak assumption or promise of 20 Hz.",
            "Current component implementations and full CPU measurements do not meet the goal; another implementation needs actual complete-model numerical and fresh-stream evidence.",
            "S=2 overlap batches contribute one new observation, not two fresh poses. S=3 batching can contribute two new observations but has batch-wait latency and more global-attention work.",
            "Inverse average throughput cannot establish end-to-end P95 latency or 30-minute bounded queues.",
            "Lower input geometry is a workload count; original weights alone do not establish <=10-20% ATE/RPE loss at that geometry.",
            "Computing camera outputs before the dense head does not prove poses are aligned to the existing SL4 graph; the current graph uses dense predictions and overlap geometry.",
            "Asynchronous dense updates are conditional arithmetic scenarios, not a tested scheduling design or acceptance result.",
            "FP16 logical parameter bytes do not establish 8 GB operation with SDK contexts, activations, OS and maps."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "primary_FLOP": workloads[0]["counted_full_forward_FLOP"],
                      "primary_cached_20Hz_TFLOP_per_second": workloads[0][
                          "required_average_effective_TFLOP_per_second_by_counting_convention"][
                              "full_forward_with_one_old_DINO_reused"]}, allow_nan=False))


if __name__ == "__main__":
    main()
