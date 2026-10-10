#!/usr/bin/env python3
"""Static original VGGT-SLAM2 capacity arithmetic; no Torch/NumPy/model imports.

MAC = one multiply-accumulate; arithmetic FLOP convention = 2 * MAC.
Counts dense linear/convolution and QK^T/AV MACs, not softmax, normalization,
GELU, RoPE, interpolation, memory traffic, graph optimization or retrieval.
"""
import argparse
import hashlib
import json
from pathlib import Path

D = 1024
GIB = 2 ** 30
MIB = 2 ** 20
OFFICIAL_URL = "https://www.rock-chips.com/a/cn/news/rockchip/2022/0303/1544.html"
SOURCE_FILES = [
    "third_party/vggt/vggt/models/vggt.py",
    "third_party/vggt/vggt/models/aggregator.py",
    "third_party/vggt/vggt/layers/vision_transformer.py",
    "third_party/vggt/vggt/layers/block.py",
    "third_party/vggt/vggt/layers/attention.py",
    "third_party/vggt/vggt/layers/mlp.py",
    "third_party/vggt/vggt/heads/camera_head.py",
    "third_party/vggt/vggt/heads/dpt_head.py",
    "third_party/vggt/vggt/utils/geometry.py",
    "vggt_slam/runtime.py", "vggt_slam/solver.py",
    "vggt_slam/submap.py", "vggt_slam/map.py", "vggt_slam/rknn_linear.py",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters():
    block_without_qknorm = 12 * D * D + 15 * D
    block_with_qknorm = block_without_qknorm + 4 * (D // 16)
    camera_d = 2048
    result = {
        "DINO_encoder": (24 * block_without_qknorm + 3 * 14 * 14 * D + D
                         + 2 * D + 4 * D + (37 * 37 + 1) * D + 2 * D),
        "frame_and_global_48_blocks": 48 * block_with_qknorm,
        "aggregator_camera_and_register_tokens": 10 * D,
        "camera_head": (4 * (12 * camera_d ** 2 + 15 * camera_d)
                        + 4 * camera_d + 9 + (9 + 1) * camera_d
                        + (camera_d + 1) * 3 * camera_d
                        + (camera_d + 1) * (camera_d // 2)
                        + (camera_d // 2 + 1) * 9),
        "depth_head": (4096 + 5769984 + 1048832 + 1049088 + 9438208
                       + 6488064 + 1245952 + 3 * 2426112 + 295040 + 36962),
    }
    assert sum(result.values()) == 1157941492
    return result


def dpt_mac(h, w):
    """Actual DPTHead convolution sizes, including odd-grid stride-2 rounding.

    ConvTranspose(kernel=stride) uses input-area*Cin*Cout*kernel^2. Conv2d
    uses output-area*Cin*Cout*kernel^2, the conventional padded dense count.
    Refinenet4 has two spatial convolutions; refinenet3/2/1 each have four.
    """
    ph, pw = h // 14, w // 14
    q, q4, f = ph * pw, ((ph + 1) // 2) * ((pw + 1) // 2), 256
    return {
        "four_projects": q * 2048 * (256 + 512 + 1024 + 1024),
        "three_resize_convolutions": (q * 256 ** 2 * 16 + q * 512 ** 2 * 4
                                      + q4 * 1024 ** 2 * 9),
        "four_scratch_convolutions": 9 * f * (16 * q * 256 + 4 * q * 512
                                              + q * 1024 + q4 * 1024),
        "four_feature_fusion_blocks": (9 * f * f * (2 * q4 + 4 * q + 16 * q + 64 * q)
                                       + f * f * (q + 4 * q + 16 * q + 64 * q)),
        "output_convolutions": 64 * q * 256 * 128 * 9 + h * w * (128 * 32 * 9 + 32 * 2),
    }


def workload(h, w, s):
    assert h % 14 == w % 14 == 0 and s >= 2
    q = (h // 14) * (w // 14)
    p = q + 5
    frame_att = 24 * s * 2 * p * p * D
    global_att = 24 * 2 * (s * p) ** 2 * D
    dino_lin = frame_lin = global_lin = 24 * s * 12 * p * D * D
    backbone = 3 * dino_lin + 2 * frame_att + global_att
    cd = 2048
    camera_trunk = 16 * (12 * s * cd * cd + 2 * s * s * cd)
    camera_branches = 4 * s * (9 * cd + 3 * cd * cd + cd * (cd // 2) + (cd // 2) * 9)
    dense_components = dpt_mac(h, w)
    patch_conv = s * q * 3 * 14 * 14 * D
    stages = {
        "DINO_patch_convolution": patch_conv,
        "DINO_24_blocks": dino_lin + frame_att,
        "frame_24_blocks": frame_lin + frame_att,
        "global_24_blocks": global_lin + global_att,
        "camera_16_trunk_calls_and_4_branches": camera_trunk + camera_branches,
        "depth_DPT_head": s * sum(dense_components.values()),
    }
    total = sum(stages.values())
    att_core = 2 * frame_att + global_att
    # Hypothetical NPU-only operation budget, NOT the documented FP16 peak.
    assumed_ops = 6e12
    return {
        "image_hw": [h, w], "sequence_images": s, "patches_per_image": q,
        "tokens_per_image": p, "global_tokens": s * p,
        "MAC_by_stage": stages,
        "DPT_MAC_components_per_image": dense_components,
        "backbone_MAC": backbone, "counted_model_MAC": total,
        "counted_model_arithmetic_FLOP": 2 * total,
        "amortized_MAC_per_input_image": total / s,
        "backbone_attention_core_MAC": att_core,
        "backbone_attention_core_fraction": att_core / backbone,
        "backbone_MLP_MAC": 72 * s * 8 * p * D * D,
        "backbone_MLP_fraction": 72 * s * 8 * p * D * D / backbone,
        "backbone_compute_only_free_attention_speedup_ceiling": backbone / (backbone - att_core),
        "counted_model_compute_only_free_attention_speedup_ceiling": total / (total - att_core),
        "fresh_images_per_overlap_update": s - 1,
        "20_batch_updates_per_second_required_arithmetic_TFLOP_per_second": 20 * 2 * total / 1e12,
        "20_new_poses_per_second_minimum_batch_rate": 20 / (s - 1),
        "20_new_poses_per_second_required_arithmetic_TFLOP_per_second": 20 / (s - 1) * 2 * total / 1e12,
        "hypothetical_skipping_only_dense_head_counted_MAC": total - stages["depth_DPT_head"],
        "hypothetical_skipping_only_dense_head_MAC_reduction_fraction": stages["depth_DPT_head"] / total,
        "optimistic_assumed_6e12_arithmetic_ops_per_second": {
            "assumed_seconds_per_forward": 2 * total / assumed_ops,
            "assumed_max_pair_updates_per_second": assumed_ops / (2 * total),
            "rigorous_hardware_lower_bound": False,
        },
        "current_retained_4_FP32_aggregator_outputs_bytes": 4 * s * p * 2048 * 4,
        "one_FP32_MLP_hidden_activation_bytes": s * p * 4096 * 4,
        "materialized_global_FP32_attention_probabilities_bytes": 16 * (s * p) ** 2 * 4,
        "attention_probability_materialization_is_unavoidable": False,
        "exact_overlap_DINO_one_image_cache_counted_MAC_saved": 24 * 12 * p * D * D + 24 * 2 * p * p * D + q * 3 * 14 * 14 * D,
        "online_freshness_note": "S=2 overlap+fresh contributes one new capture per update; dividing cost by two is amortized image cost, not two new online poses. S>2 can batch more fresh images but incurs batch-wait latency.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="Optional local checkout SHA verification; omitted uses recorded source pins")
    parser.add_argument("--evidence", type=Path, default=Path(__file__).with_name("evidence.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("capacity.json"))
    a = parser.parse_args()
    sidecar = json.loads(a.evidence.read_text())
    if sidecar.get("kind") != "sanitized_static_capacity_evidence":
        raise ValueError("Incorrect evidence sidecar kind")
    recorded_evidence = sidecar["evidence"]
    source_pins = sidecar["source_pins"]
    assert recorded_evidence["recorded_model_structure_receipt"]["parameters"] == 1157941492
    assert recorded_evidence["original_518_S3_mixed_linear_profile"]["FP16_host_weight_copy_bytes"] == 72 * 12 * D * D * 2
    if a.repo is not None:
        repo = a.repo.resolve(strict=True)
        actual_pins = {p:sha(repo/p) for p in SOURCE_FILES}
        if actual_pins != source_pins:
            raise ValueError("Local sources do not match the recorded evidence pins")
    par = parameters()
    fp32 = sum(par.values()) * 4
    copies = 72 * 12 * D * D * 2
    cases = [workload(*hw_s) for hw_s in [(252,336,2),(84,112,2),(168,224,2),
                                         (294,518,2),(294,518,3),(392,518,2)]]
    # Independent algebra checks, not implementation-mirroring model inference.
    for c in cases:
        s, p = c["sequence_images"], c["tokens_per_image"]
        assert c["backbone_MAC"] == 864 * s * p * D * D + 48 * s * (s + 2) * p * p * D
        assert c["counted_model_MAC"] > c["backbone_MAC"]
        assert c["counted_model_arithmetic_FLOP"] == 2 * c["counted_model_MAC"]
    assert cases[0]["backbone_MAC"] == 866909552640
    assert sum(dpt_mac(252,336).values()) == 47023828992
    result = {
        "kind": "static_original_VGGT_capacity_audit",
        "script_sha256": sha(__file__), "scope": "Original weights and original 24+24+24 topology, camera+depth outputs; no ORB, no alternative frontend.",
        "execution": {"stdlib_scalar_arithmetic_and_saved_receipts_only": True, "model_run": False, "host_image_decode": False, "ORT_run": False, "board_operation": False, "production_modified": False},
        "counting": {"one_MAC_arithmetic_FLOP": 2, "omitted": ["bias/addition", "layernorm/QKnorm", "softmax", "GELU", "RoPE", "interpolation", "memory movement", "graph optimization", "loop retrieval and additional loop model forwards"], "counts_are_architecture_MAC_budget_not_measured_execution_FLOPs": True, "padded_conv2d_dense_MAC_convention": True},
        "topology": {"DINO_blocks":24,"frame_blocks":24,"global_blocks":24,"embedding":1024,"heads":16,"MLP_expansion":4,"patch_size":14,"special_tokens_per_image":5,"camera_embedding":2048,"camera_blocks":4,"camera_iterations":4,"retained_aggregator_layers":[4,11,17,23],"retention_does_not_remove_any_transformer_call":True},
        "parameters_by_component": par, "parameter_count":sum(par.values()),
        "primary_case_index":0, "workloads":cases,
        "memory": {
            "FP32_pruned_model_parameter_bytes":fp32,"FP32_pruned_model_parameter_GiB":fp32/GIB,
            "hypothetical_all_FP16_parameter_bytes":fp32//2,"hypothetical_all_FP16_parameters_supported_or_validated":False,
            "default_CPU_has_no_optional_Linear_FP16_copies":True,
            "observed_optional_RKNN_Linear_FP16_copy_bytes":copies,"observed_optional_RKNN_Linear_FP16_copy_GiB":copies/GIB,
            "observed_optional_RKNN_Linear_parameter_payload_plus_FP16_copy_bytes":fp32+copies,
            "observed_optional_RKNN_Linear_parameter_payload_plus_FP16_copy_GiB":(fp32+copies)/GIB,
            "optional_RKNN_Linear_generous_8GiB_budget_remaining_before_activations_SDK_OS_maps_bytes":8*GIB-fp32-copies,
            "payload_not_a_universal_minimum_resident_memory": "Mmap pages can be evicted. The 6.001 GiB payload describes the observed optional RKNN per-Linear backend experiment, not default CPU main.py. Avoiding FP32 residency trades page faults/reloads; a new validated block-storage layout could remove duplication.",
            "not_included":["allocator workspace","SDK internal buffers","ONNX/RKNN model contexts","temporary activations","dense retained maps","retrieval model","operating system"],
            "graph_retention_source": "GraphMap retains every Submap; no eviction in add_submap. Submap keeps FP32 RGB tensors, FP32 XYZ, RGB8 colors and one aliased FP32 confidence array: at least 31 payload bytes per retained pixel. Export adds separate FP64 transform temporaries.",
            "minimum_map_payload_30min_at_2_to_5_seconds_per_S2_update": [
                {"image_hw":[h,w],"update_interval_seconds":interval,"updates":1800//interval,
                 "retained_image_records_including_overlap":2*(1800//interval),
                 "minimum_bytes":31*h*w*2*(1800//interval),
                 "minimum_GiB":31*h*w*2*(1800//interval)/GIB}
                for h,w in [(252,336),(392,518)] for interval in [2,5]],
            "map_payload_budget_excludes_confidence_export_filter_savings": "Confidence filters exported points, not stored dense Submap arrays. This estimate assumes distinct storage per update and does not count optional shared overlap frame storage twice; removing each overlap FP32 RGB duplicate would save 12 bytes for that record, not its new batch-specific XYZ/conf/colors.",
        },
        "hardware_planning": {"official_url":OFFICIAL_URL,"official_claim":"6 TOPS NPU; INT4/INT8/INT16/FP16 mixed support", "6TOPS_precision_unspecified":True,"assumed_NPU_arithmetic_ops_per_second":6e12,"conversion_to_FP16_TFLOPs_is_not_verified":True,"time_division_is_optimistic_planning_not_rigorous_lower_bound":True,"CPU_GPU_heterogeneous_throughput_not_estimated":True,"absolute_impossibility_claim":False},
        "source_pins":source_pins,
        "local_source_SHA_check_performed":a.repo is not None,
        "portable_evidence": {"file":a.evidence.name,"sha256":sha(a.evidence),"private_raw_receipts_required":False},
        "self_checks":{"parameter_sum_equals_recorded_structure":True,"MAC_decomposition_equals_closed_form_all_six_cases":True,"336_DPT_dense_convolution_count":True},
        "priority_next_steps":[
            "Validate complete-block/device-resident execution across successive original layers and real tokens; eliminate per-Linear weight uploads, synchronization and layout copies. MLP is the largest MAC share. Enforce original pose/K/depth numerical and quality gates across the entire model, not one block.",
            "Measure CPU attention-only offload integration end-to-end on exactly the same pair with actual transfers. V19 saved standalone operands and overhead do not represent live block integration, and strict FP32 conformance failed.",
            "Remove FP32/FP16 duplicate weight residency only after functional and error validation; bound dense map retention/compaction and retrieval growth before 30 minutes. Bounded capture queues do not bound maps or model workspaces.",
            "Exact overlap DINO reuse can save only the per-image independent encoder for unchanged input/model state. Later bidirectional global-attention token caches are not generally valid after fresh images arrive. Any more aggressive layer/token/window reduction changes original computation and requires explicit quality evaluation.",
        ],
        "evidence":recorded_evidence,
        "limits":["No real-time or quality claim from static MAC counts.","A 6 TOPS marketing figure is not an established FP16 sustained roofline.","20Hz means new input poses; repeating overlap or old poses does not meet it.","P95 latency needs an actual streaming measurement, not inverse throughput.","Original SL4 map association is projective and does not automatically imply metric SE3 accuracy.","The static count includes dense prediction every forward as current VGGT.forward does; asynchronous/sparse dense updates require a separate faithful output-frequency design."],
    }
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":a.output.name,"sha256":sha(a.output),"bytes":a.output.stat().st_size,"primary_GMAC":cases[0]["counted_model_MAC"]/1e9,"primary_20Hz_TFLOP_per_second":cases[0]["20_new_poses_per_second_required_arithmetic_TFLOP_per_second"],"all_static_checks_passed":True}))


if __name__ == "__main__":
    main()
