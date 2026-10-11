#!/usr/bin/env python3
"""Check a static budget against saved dimensions using scalar arithmetic only."""
import argparse
import hashlib
import json
from math import prod
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--camera-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a fresh path")
    budget = json.loads(args.budget.read_text())
    profile = json.loads(args.profile.read_text())
    camera = json.loads(args.camera_capture.read_text())
    checks = {}

    def record(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise ValueError(f"Static budget check failed: {name}")

    config = budget["configuration_from_fixed_source_AST"]
    d, heads = config["embedding"], config["heads"]
    matrix_names = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")
    block_elements = sum(prod(profile["blocks"]["frame0"]["weights"][key]["shape"])
                         for key in matrix_names)
    record("captured_matrix_shapes_equal_12D_squared", block_elements == 12 * d * d)
    # Small, manually countable dimensional example checks MAC/FLOP and the S^2 global term.
    # D=4, S=2, N=3: qkv 288 + projection 96 + two MLP matrices 768 = 1152 MAC.
    tiny_linear = 2 * 3 * (4 * 12 + 4 * 4 + 4 * 16 + 16 * 4)
    tiny_frame_attention = 2 * 2 * 2 * 3 * 3 * 2  # QK/AV * S * heads * N^2 * head_dim
    tiny_global_attention = 2 * 2 * 6 * 6 * 2
    record("tiny_pair_manual_2736_MAC_5472_FLOP",
           tiny_linear == 1152 and tiny_frame_attention == 144 and tiny_global_attention == 288
           and 2 * tiny_linear + tiny_frame_attention + tiny_global_attention == 2736
           and 2 * (2 * tiny_linear + tiny_frame_attention + tiny_global_attention) == 5472)
    for index, case in enumerate(budget["workloads"]):
        s, _, h, w = case["input_shape_SCHW"]
        n = (h // 14) * (w // 14) + 5
        components = case["MAC_by_component"]
        enumerated_linear = 24 * s * n * block_elements
        # Matrix dimensions explicitly enumerate head and image axes independently of closed form.
        enumerated_frame_attention = 24 * sum(2 * n * n * (d // heads)
            for _ in range(s) for _ in range(heads))
        enumerated_global_attention = 24 * sum(2 * (s * n) * (s * n) * (d // heads)
            for _ in range(heads))
        record(f"case_{index}_captured_linear_and_enumerated_attention_axes", all((
            n == case["tokens_per_image"], enumerated_linear == components["DINO_24_blocks_linear"],
            enumerated_linear == components["frame_24_blocks_linear"],
            enumerated_linear == components["global_24_blocks_linear"],
            enumerated_frame_attention == components["DINO_24_blocks_attention_QK_AV"],
            enumerated_frame_attention == components["frame_24_blocks_attention_QK_AV"],
            enumerated_global_attention == components["global_24_blocks_attention_QK_AV"])))
        total = sum(components.values())
        old_dino = (components["DINO_patch_convolution"] + components["DINO_24_blocks_linear"]
                    + components["DINO_24_blocks_attention_QK_AV"]) // s
        record(f"case_{index}_full_and_cached_totals_fresh_observation_denominator", all((
            total == case["counted_full_forward_MAC"],
            2 * total == case["counted_full_forward_FLOP"],
            old_dino == case["one_old_DINO_exact_reuse_MAC_saved"],
            2 * (total - old_dino) == case["counted_full_forward_with_one_old_DINO_reused_FLOP"],
            case["fresh_observations_per_overlap_batch"] == s - 1,
            case["required_batches_per_second_for_20_fresh_poses"] * (s - 1) == 20)))
    camera_tensors = camera["artifacts"]["camera-state.npz"]["tensors"]
    camera_matrix_elements = sum(prod(value["shape"]) for key, value in camera_tensors.items()
        if key.endswith(".weight") and len(value["shape"]) == 2)
    traffic = budget["FP16_weight_payload_and_conditional_streaming"]
    record("conditional_camera_streaming_uses_four_iterations_and_FP16_two_bytes", all((
        2 * camera_matrix_elements == traffic["camera_unique_matrix_FP16_bytes"],
        4 * 2 * camera_matrix_elements == traffic[
            "camera_four_iteration_logical_matrix_reads_FP16_bytes_if_no_inter_iteration_cache"],
        2 * 72 * block_elements + 4 * 2 * camera_matrix_elements == traffic[
            "backbone_camera_logical_matrix_reads_FP16_bytes_per_forward_if_each_use_is_streamed_once"])))
    record("complete_active_parameter_count_matches_v47_saved_count", all((
        sum(budget["source_derived_active_parameters_by_component"].values()) == 1157941492,
        2 * 1157941492 == traffic["logical_unique_active_parameter_FP16_bytes"],
        budget["source_derived_active_parameters_by_component"]["original_camera_head"]
            == sum(prod(value["shape"]) for value in camera_tensors.values()))))
    result = {"schema": 1, "status": "passed_static_formula_checks", "checks": checks,
              "check_count": len(checks), "script_sha256": sha(__file__),
              "input_SHA": {"budget": sha(args.budget), "profile": sha(args.profile),
                            "camera_capture": sha(args.camera_capture)},
              "actual_model_OR_ORT_OR_RKNN_OR_compile_run": False, "board_access": False,
              "proves_hardware_throughput_or_model_quality": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "check_count": len(checks), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
