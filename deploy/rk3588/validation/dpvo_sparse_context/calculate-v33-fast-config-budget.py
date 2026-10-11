#!/usr/bin/env python3
"""Stdlib-only static budget for the pinned DPVO default/FAST configurations.

No neural evaluation, ONNX execution, geometry, SDK invocation or board job.
Reference source: DPVO MIT, Copyright (c) 2022 Princeton Vision & Learning Lab.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
MANIFEST_SHA = "ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375"
ENCODER_REVIEW_SHA = "0b642f4fd6db963cc81643ad6cb9a8bb182c34ce7b9f8f9f7604e70d5401ae3b"
PIN_PATHS = ["dpvo/config.py", "config/default.yaml", "config/fast.yaml", "dpvo/dpvo.py",
             "dpvo/net.py", "dpvo/blocks.py", "dpvo/extractor.py", "LICENSE"]
DIM = 384
CORR_DIM = 882
EDGE_LINEAR_MACS = CORR_DIM * DIM + 16 * DIM * DIM + 2 * DIM * 2
GROUP_LINEAR_MACS = DIM * DIM


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_base_config(text):
    """Parse only literal _C.NAME assignments; never import yacs or torch."""
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "_C":
                result[target.attr] = ast.literal_eval(node.value)
    return result


def parse_yaml_overlay(text):
    """The pinned YAML uses only flat, Python-literal scalar values."""
    result = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split(":", 1)
        if not key.strip().isidentifier() or key.strip() in result:
            raise ValueError("Unexpected flat config key")
        result[key.strip()] = ast.literal_eval(value.strip())
    return result


def update_linear_macs(E, Gkk, Gij):
    """21 original Linear layers; bias/activation/reduction/BA are excluded."""
    return E * EDGE_LINEAR_MACS + (Gkk + Gij) * GROUP_LINEAR_MACS


def pair_summary(pairs, M):
    return {"frame_pairs_Gij": len(pairs), "patch_groups_Gkk": len({i for i, _ in pairs}) * M,
            "edges_E": len(pairs) * M,
            "Update_linear_MACs": update_linear_macs(len(pairs) * M, len({i for i, _ in pairs}) * M, len(pairs))}


def simulate_no_keyframe_deletion(config, *, admitted=64):
    """Original forward/back append and source-age pruning, at frame-pair level.

    All M patches share the same frame-pair topology. This explicitly assumes every
    observation is admitted and no keyframe deletion: no simulated poses or flow.
    Original flow-based deletion depends on real geometry and is not predicted.
    """
    M, L, R = (config[k] for k in ("PATCHES_PER_FRAME", "PATCH_LIFETIME", "REMOVAL_WINDOW"))
    if min(M, L, R) <= 0 or admitted < 8 or config["LOOP_CLOSURE"] or config["CLASSIC_LOOP_CLOSURE"]:
        raise ValueError("Positive loop-disabled original lifecycle configuration required")
    pairs, rows = [], []
    for n in range(1, admitted + 1):
        # Exact source order is preserved here, although only counts are reported.
        forward = [(i, n - 1) for i in range(max(n - L, 0), n - 1)]
        backward = [(n - 1, j) for j in range(max(n - L, 0), n)]
        pairs.extend(forward)
        pairs.extend(backward)
        if len(set(pairs)) != len(pairs):
            raise ValueError("Unexpected duplicate frame pair")
        before = pair_summary(pairs, M)
        updates = 12 if n == 8 else int(n > 8)
        # Initial 8-frame block performs 12 Updates, without calling keyframe().
        if n > 8:
            pairs = [(i, j) for i, j in pairs if i >= n - R]
        after = pair_summary(pairs, M)
        rows.append({"admitted_frames": n, "new_forward_edges": len(forward) * M,
                     "new_backward_edges_including_self": len(backward) * M,
                     "before_Update": before, "after_source_age_pruning": after,
                     "graph_Update_calls": updates, "BA_iterations": 2 * updates})
    tail = rows[-1]
    stable_post_pairs = R * (2 * L - 1) - L * (L - 1) // 2
    stable_pre_pairs = stable_post_pairs + 2 * L - 1
    if admitted >= R + L and (tail["after_source_age_pruning"]["frame_pairs_Gij"] != stable_post_pairs or
                              tail["before_Update"]["frame_pairs_Gij"] != stable_pre_pairs):
        raise ValueError("Closed form disagrees with the original lifecycle simulation")
    pre = tail["before_Update"]
    E = pre["edges_E"]
    return {"assumptions": {"admitted_frames": admitted, "all_observations_admitted": True,
        "keyframe_deletion": False, "loop_closure": False,
        "motion_probe_or_flow_not_evaluated": True,
        "actual_graph_can_be_smaller_and_have_different_groups": True},
        "initialization": {**rows[7]["before_Update"], "admitted_frames": 8, "graph_Update_calls": 12,
                           "BA_iterations": 24, "total_Update_linear_MACs": rows[7]["before_Update"]["Update_linear_MACs"] * 12},
        "steady_no_deletion_envelope": {"before_Update": pre,
            "after_source_age_pruning": tail["after_source_age_pruning"],
            "closed_form_post_pairs": "R*(2*L-1)-L*(L-1)/2", "closed_form_pre_pairs": "post_pairs+(2*L-1)",
            "per_admitted_frame_graph_Update_calls": 1, "BA_iterations": 2,
            "FP32_net_ctx_corr_input_bytes": E * (384 + 384 + 882) * 4,
            "input_bytes_are_not_total_RSS_or_padded_topology_memory": True,
            "fresh20Hz_Update_linear_MACs_per_second": pre["Update_linear_MACs"] * 20,
            "fresh20Hz_Update_linear_FP_ops_per_second_using2perMAC": pre["Update_linear_MACs"] * 40},
        "termination": {"separate_offline_Update_calls": 12, "BA_iterations": 24, "fresh_pose_count": 0},
        "simulation_rows": rows}


def encoder_conv_macs(height=464, width=608):
    """Both original BasicEncoder4 branches; no norm/activation/sampling counts."""
    if height % 4 or width % 4:
        raise ValueError("Only this exact divisible-by-four shape contract is used")
    h2, w2, h4, w4 = height // 2, width // 2, height // 4, width // 4
    def conv(h, w, ci, co, k):
        return h * w * ci * co * k * k
    common = {
        "stem7x7_3to32": conv(h2, w2, 3, 32, 7),
        "layer1_four3x3_32to32": 4 * conv(h2, w2, 32, 32, 3),
        "layer2_stride2_3x3_32to64": conv(h4, w4, 32, 64, 3),
        "layer2_three3x3_64to64": 3 * conv(h4, w4, 64, 64, 3),
        "layer2_stride2_shortcut1x1_32to64": conv(h4, w4, 32, 64, 1),
    }
    prefix = sum(common.values())
    fnet = prefix + conv(h4, w4, 64, 128, 1)
    inet = prefix + conv(h4, w4, 64, 384, 1)
    return {"input_hw": [height, width], "each_branch_prefix": common,
            "fnet_conv_MACs": fnet, "inet_conv_MACs": inet, "total_conv_MACs": fnet + inet,
            "unchanged_by_M_or_FAST_or_sparse_context_gather": True,
            "excluded": "InstanceNorm/ReLU/bias/residual/normalization/pyramid/copy/sampling/correlation/BA"}


def build_report(source_dir, manifest_path, encoder_review_path):
    source_dir, manifest_path, encoder_review_path = map(Path, (source_dir, manifest_path, encoder_review_path))
    if sha256_file(manifest_path) != MANIFEST_SHA or sha256_file(encoder_review_path) != ENCODER_REVIEW_SHA:
        raise ValueError("Pinned source manifest or measured encoder review SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["commit"] != COMMIT:
        raise ValueError("Official source commit mismatch")
    records = {r["path"]: r for r in manifest["retrieved_files"] if "sha256" in r}
    before = {p: sha256_file(source_dir / p) for p in PIN_PATHS}
    if any(before[p] != records[p]["sha256"] for p in PIN_PATHS):
        raise ValueError("Official source byte identity mismatch")
    base = parse_base_config((source_dir / "dpvo/config.py").read_text())
    configs = {name: {**base, **parse_yaml_overlay((source_dir / ("config/" + name + ".yaml")).read_text())}
               for name in ("default", "fast")}
    expected = {"default": [96, 13, 22, 10], "fast": [48, 11, 16, 7]}
    for name, cfg in configs.items():
        actual = [cfg[k] for k in ("PATCHES_PER_FRAME", "PATCH_LIFETIME", "REMOVAL_WINDOW", "OPTIMIZATION_WINDOW")]
        if actual != expected[name] or cfg["KEYFRAME_INDEX"] != 4 or cfg["BUFFER_SIZE"] != 4096:
            raise ValueError("Pinned effective configuration mismatch")
    budgets = {name: {"effective_config": cfg, **simulate_no_keyframe_deletion(cfg)} for name, cfg in configs.items()}
    review = json.loads(encoder_review_path.read_text())
    if review["status"] != "passed" or review["failed_check_count"] != 0 or review["candidate_accepted"] is not False:
        raise ValueError("Measured encoder evidence state mismatch")
    timing = review["execution"]["normal"]["timing"]
    measured = {method: {"warm_samples_seconds": row["warm_seconds"],
        "warm_mean_ms": row["warm_mean_seconds"] * 1000,
        "warm_mean_divided_by_50ms": row["warm_mean_seconds"] / 0.05}
        for method, row in timing.items()}
    after = {p: sha256_file(source_dir / p) for p in PIN_PATHS}
    if before != after or sha256_file(encoder_review_path) != ENCODER_REVIEW_SHA:
        raise ValueError("Read-only evidence changed")
    d = budgets["default"]["steady_no_deletion_envelope"]["before_Update"]
    f = budgets["fast"]["steady_no_deletion_envelope"]["before_Update"]
    enc = encoder_conv_macs()
    return {"schema": "v33_DPVO_original_FAST_static_edge_and_compute_budget", "status": "static_budget_not_runtime_or_accuracy_validation",
        "calculator_sha256": sha256_file(__file__), "official_commit": COMMIT,
        "source_manifest_sha256": MANIFEST_SHA, "source_pins": before, "source_hashes_after": after,
        "source_configs_are_alternatives_not_default_plus_fast_overlay": True,
        "lifecycle_semantics": {"motion_probe_quantile_threshold": 2.0, "KEYFRAME_THRESH": 15.0,
            "first7_admitted_without_graph_Update": True, "eighth_admitted": "12 Updates, each 2 BA iterations",
            "later_admitted": "1 Update before geometry-dependent keyframe deletion and source-age pruning",
            "edge_order": "old patches→new target then new patches→window targets including self",
            "source_pruning_condition": "patch_source_slot < retained_n - REMOVAL_WINDOW",
            "motion_skips_predictions_and_EOF_reconstruction_are_not_fresh_poses": True},
        "MAC_convention": {"MAC": "one multiplication plus accumulation; 2 FP arithmetic ops for a conventional FLOP count",
            "linear_formula": "E*2699520+(Gkk+Gij)*147456", "original_Linear_layers": 21,
            "not_total_FLOPs_or_hardware_latency_bound": True,
            "excluded": "bias/LayerNorm/softmax/group gather/reduction/activations/correlation/BA/IO/topology"},
        "configurations": budgets, "original_encoder": enc,
        "static_FAST_over_default_ratios": {"steady_Update_edges": f["edges_E"] / d["edges_E"],
            "steady_Update_linear_MACs": f["Update_linear_MACs"] / d["Update_linear_MACs"],
            "encoder_conv_MACs": 1.0},
        "known_arithmetic_cross_checks": {"E96_Gkk96_Gij1_MACs": update_linear_macs(96, 96, 1),
            "E864_Gkk288_Gij9_MACs": update_linear_macs(864, 288, 9)},
        "20Hz_planning": {"fresh_pose_period_ms": 50, "separate_end_to_end_P95_target_ms": 100,
            "encoder_measured_evidence_sha256": ENCODER_REVIEW_SHA,
            "encoder_normal_receipt_sha256": review["receipts"]["normal"]["sha256"],
            "measured_rejected_encoder_components": measured,
            "warm_samples_each": 2, "thermal_controlled": False, "strict_FP32_encoder_gate_passed": False,
            "FAST_configuration_has_not_been_board_executed_here": True,
            "FAST48_sampling_does_not_reduce_dense_encoder_work": True,
            "encoder_sparse_context_changes_transfer_only_before_actual_compiler_optimization": True,
            "dense_fmap_remains_required_for_correlation": True,
            "serial_current_evidence_supports_stable20Hz": False,
            "claim": "FAST materially reduces graph work but does not establish 20Hz. The existing rejected encoder diagnostic alone averages207ms (spatial) /420ms (original). New sampling/placement/IO timings and complete fresh-frame Update+corr+BA/quality remain unmeasured; this is not a proof of hardware impossibility.",
            "original_encoder_plus_FAST_Update_20Hz_conv_and_linear_MACs_per_second":
                (enc["total_conv_MACs"] + f["Update_linear_MACs"]) * 20,
            "no6TOPS_to_FP16_TFLOPS_conversion": True,
            "no_original_VGGT_realtime_claim": True},
        "source_license": "MIT; neural weight license independent", "default_or_model_modified": False,
        "model_forward": False, "actual_ORT": False, "board_job": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--encoder-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Budget receipt must be new")
    result = build_report(args.official_source_dir, args.source_manifest, args.encoder_review)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "steady_Update_edges": {
        name: row["steady_no_deletion_envelope"]["before_Update"]["edges_E"]
        for name, row in result["configurations"].items()}}), flush=True)
    return result


if __name__ == "__main__":
    main()
