"""Re-audit frozen native518 budgets using AST/JSON only; no framework imports."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

ARCHIVE = Path('/path/to/research-artifacts/v94-native-capacity-and-critical-path-review-r1')
D100 = Path('/path/to/research-artifacts/v100-native518-integration-r1')
R109 = Path('/dev/shm/edge-vggt-v109-native-slam')
SOURCE_PINS = {
    'models_aggregator.py': '623a943847a3259292bfd5c8675d8b80b09abc845eefadba7622e6a901cb9cfd',
    'layers_vision_transformer.py': 'f781186de8e9a52acc4dfc07820cde01bb182a3eb6835f630b863eaf2cde6bde',
}
PINS = {}


def read(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert sha == expected, str(path)
    PINS[str(path)] = {'bytes': len(raw), 'sha256': sha}
    return raw


def load(path, expected=None):
    return json.loads(read(path, expected))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = ap.parse_args()
    trees = {name: ast.parse(read(ARCHIVE / 'evidence/upstream-source' / name, sha))
             for name, sha in SOURCE_PINS.items()}
    agg = next(n for n in trees['models_aggregator.py'].body if isinstance(n, ast.ClassDef) and n.name == 'Aggregator')
    init = next(n for n in agg.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    defaults = {a.arg: ast.literal_eval(v) for a, v in zip(init.args.args[-len(init.args.defaults):], init.args.defaults)
                if isinstance(v, (ast.Constant, ast.List))}
    assert [defaults[k] for k in ('img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio', 'num_register_tokens', 'aa_block_size')] == [518, 14, 1024, 24, 16, 4.0, 4, 1]
    assert defaults['patch_embed'] == 'dinov2_vitl14_reg' and defaults['aa_order'] == ['frame', 'global']
    vit = next(n for n in trees['layers_vision_transformer.py'].body if isinstance(n, ast.FunctionDef) and n.name == 'vit_large')
    call = next(n for n in ast.walk(vit) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'DinoVisionTransformer')
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords if isinstance(k.value, ast.Constant)}
    assert [kw[k] for k in ('embed_dim', 'depth', 'num_heads', 'mlp_ratio')] == [1024, 24, 16, 4]
    old = load(ARCHIVE / 'fresh-encoder-budget.json', '6bbfdcd95468639538d871ed896c8f387a989cc35bbeaf049e7f4f689200f18d')
    prior = load(ARCHIVE / 'evidence/native-target-feasibility-review.json', 'f02e8417561cda9f9ff47ac95640718a8034599e19fba17024c4b7add6082925')['full_path_after_one_old_DINO_cache_hit']
    native = load(ARCHIVE / 'evidence/capture.json', '3cdfa55556896b0f5356f710c65069b8b5e53f37a0f8cb6f550439f6dfb06730')
    assert native['full_model_forward_completed'] and native['image_shape'] == [2, 3, 392, 518]
    shape_capture = load('/dev/shm/edge-vggt-v99-native518/dino-reference.json', '01931ff710097ce06782de9f99431a20cb5cfdce472ded9e45f398df9c8bc8bb')
    weights = load('/dev/shm/edge-vggt-v53/capture.json', '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67')
    d, hidden, layers, frames, patch = 1024, 4096, 24, 2, 14
    t = (392 // patch) * (518 // patch) + 1 + 4
    matrix_shapes = {'attn.qkv.weight': [3*d, d], 'attn.proj.weight': [d, d], 'mlp.fc1.weight': [hidden, d], 'mlp.fc2.weight': [d, hidden]}
    catalog = {(r['role'], r['index']): r for r in weights['blocks']}
    assert set(catalog) == {(role, i) for role in ('frame', 'global') for i in range(layers)}
    for row in [shape_capture, *catalog.values()]:
        assert all(row['parameters'][k]['shape'] == shape for k, shape in matrix_shapes.items())
    assert shape_capture['input_shape'] == [1, t, d] and shape_capture['architecture']['depth_loaded'] == layers
    seen = []
    for i in range(4):
        receipt = load(D100 / f'exports/agg-group{i}/export.json')
        graph = receipt['blocks'][0]
        assert receipt['group_index'] == i and receipt['image_shape'] == [2, 3, 392, 518]
        assert graph['input_shape'] == [frames, t, d]
        assert [o['shape'] for o in graph['output_bindings'][:2]] == [[frames, t, d], [1, frames*t, d]]
        assert graph['all_original_parameter_values_preserved_by_parent_composition']
        assert graph['all_original_frame_and_global_computation_tensors_retained']
        assert receipt['original_parameter_value_count'] == 36*len(receipt['indices'])
        for parent in graph['parents']:
            seen.append(parent['index'])
            for role in ('frame', 'global'):
                assert parent['parameter_raw_sha256'][role] == {k: v['raw_sha256'] for k, v in catalog[role, parent['index']]['parameters'].items()}
    assert seen == list(range(layers))

    def block_budget(batch, tokens):
        products = {k: layers*batch*tokens*math.prod(shape) for k, shape in matrix_shapes.items()}
        products['QK_transpose_and_probability_V'] = layers*2*batch*tokens*tokens*d
        return dict(input_shape=[batch, tokens, d], complete_original_blocks=layers,
                    MAC_by_product=products, MAC=sum(products.values()))

    dino = block_budget(1, t)
    patch_mac = (392//patch)*(518//patch)*3*patch*patch*d
    fresh = dino['MAC'] + patch_mac
    frame, glob = block_budget(frames, t), block_budget(1, frames*t)
    agg_mac = frame['MAC'] + glob['MAC']
    cold, reuse = frames*fresh + agg_mac, fresh + agg_mac
    assert fresh == old['MAC_per_fresh_image']['full_counted_encoder']
    assert agg_mac == prior['warm_MAC_components']['two_frames_24_frame_and_24_global_blocks']
    head_mac = sum(prior['warm_MAC_components'][k] for k in ('camera_four_refinements_and_branches', 'two_frames_original_DPT_convolutions'))
    assert cold + head_mac == prior['cold_full_MAC_per_S2_sweep']
    assert reuse + head_mac == prior['warm_full_MAC_per_new_observation']
    # Current candidate repeats CPU token preparation for both images even on a hit.
    read(Path(__file__).parent / 'v110_native518_executor.py', 'e153a5f71ef0b10119207b2e43589b628a2e85ab1abeaeb72d2df2a7dd6fb638')
    read(Path(__file__).parent / 'run_native518_tail_reuse_chain.py', 'd645622ad58e4a7f291347b902ee5d55e16e12c0252e20f99c772cb32ba85f72')
    implemented_reuse = cold - dino['MAC']
    capacity = load(ARCHIVE / 'capacity-primary-evidence.json', '8587c41b1d0f5970cb8c720dbf024adf29d1c4405356698d3fa3b16de0f9e11c')
    gpu = load('/dev/shm/edge-vggt-v95/gpu-int8-route-review.json', '40fb5ced33877517f6e236c1c308daabec8b2955f8f4139aa40c073d5a1037d6')['conditional_arithmetic_inference']
    nominal = capacity['NPU_documented_nominal']['family_total_INT8_TOPS']
    cpu = capacity['arithmetic_derived_peak_rates']['CPU_INT8_dot_INT32_TOPS']
    modeled = gpu['modeled_GPU_INT8_TOPS']
    cap = nominal + cpu + modeled
    assert abs(cap - 8.880512) < 1e-12
    audit = load(R109 / 'saved-pair-chain-audit.json', '2fe622d51282e0ca36192a352c47e267230e6b02d4e763488a9ded74028f74a5')
    actual = load(R109 / 'pair-chain-r1/result.json', '966900291273b6f61af151307f2713ede5524e132847bd1b61c372e9ccdb523c')
    assert audit['status'].startswith('passed_') and audit['first_pair_all5_raw_byte_equal_v100']
    assert len(actual['pairs']) == 2 and actual['prepared_image_shape'] == [2, 3, 392, 518]
    assert all(p['executor_call']['complete_expected_execution'] and not p['executor_call']['image_or_feature_cache_used'] for p in actual['pairs'])
    measured = [{k: p[k] for k in ('index', 'source_indices', 'prediction_seconds', 'points_scale_factors_LM_seconds', 'image_read_to_original_pose_publication_seconds', 'original_model_timings')} for p in actual['pairs']]
    summary = {
        'schema': 'v110_independent_native518_compute_budget_audit', 'status': 'passed_static_and_saved_evidence_audit',
        'execution': 'stdlib AST/JSON and integer arithmetic only; no model/ORT/SDK/GPU/board calls',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_shape': [2, 3, 392, 518], 'tokens_per_image': t, 'channel_dim': d, 'heads': 16, 'MLP_hidden_dim': hidden,
        'counting_contract': 'One conventional scalar MAC is two counted arithmetic operations (FLOPs for floating-point products). Dense products only, not physical instructions or a universal exact-algorithm lower bound.',
        'exact_dense_components': {'DINO24_one_image': dino, 'patch_convolution_MAC_one_image': patch_mac, 'fresh_DINO_total_MAC': fresh, 'frame24_two_images': frame, 'global24_two_images': glob, 'aggregator48_MAC': agg_mac},
        'dense_product_parameter_scalars': {'DINO24': layers*sum(math.prod(s) for s in matrix_shapes.values()), 'aggregator48': 2*layers*sum(math.prod(s) for s in matrix_shapes.values()), 'original_encoder_linear_uses_per_weight_scalar': t, 'original_aggregator_linear_uses_per_weight_scalar': frames*t, 'note': 'Each encoder weight participates once per token; aggregator weights once per 2082 token rows. These are arithmetic uses, not bytes loaded or actual DDR traffic.'},
        'per_observation': {'DINO24_plus_aggregator48_cold_two_images_MAC': cold, 'DINO24_plus_aggregator48_with_exact_old_image_DINO_reuse_MAC': reuse, 'exact_old_DINO_MAC_saved': fresh, 'fraction_saved_from_counted_transformer_cold': fresh/cold, 'prior_camera_plus_two_DPT_MAC': head_mac, 'prior_complete_cold_MAC': cold+head_mac, 'prior_complete_with_old_DINO_reuse_MAC': reuse+head_mac, 'all_prior_budget_values_reconciled': True},
        'current_v110_source_cache_cut': {'source_review_only_not_execution': True, 'old_image_24_DINO_block_MAC_saved': dino['MAC'], 'CPU_patch_convolutions_per_pair_even_on_hit': 2, 'ideal_safe_patch_convolutions_per_pair_on_hit': 1, 'extra_old_patch_MAC_above_best_safe_reuse': patch_mac, 'DINO_plus_aggregator_MAC_on_hit': implemented_reuse, 'all_counted_full_pair_MAC_on_hit': implemented_reuse+head_mac, 'DINO_plus_aggregator_counted_TOPS_at20Hz_on_hit': 2*implemented_reuse*20/1e12},
        'requirements': {'fresh_DINO_counted_TOPS_at20Hz': 2*fresh*20/1e12, 'reused_DINO_plus_aggregator_counted_TOPS_at20Hz': 2*reuse*20/1e12, 'complete_prior_counted_TOPS_at20Hz': 2*(reuse+head_mac)*20/1e12, 'one_reused_transformer_pair_counted_tera_operations': 2*reuse/1e12, 'complete_pair_counted_TOPS_if_one_map_every2_seconds': (reuse+head_mac)/1e12, 'complete_pair_counted_TOPS_if_one_map_every5_seconds': 2*(reuse+head_mac)/5e12, 'map_budget_scope': 'One complete pair per map update only. This lower-frequency map count does not include an additional 20Hz pose stream, nor prove original registered-pose/head separation.'},
        'reuse_boundary': ['Only identical per-image DINO computations with identical prepared pixels, masks, geometry, original state and numerical backend are credited.', 'Original aggregator special tokens change first/other roles. Global bidirectional attention mixes both images; no old frame/global outputs or causal KV cache are subtracted.', 'Camera/DPT head omission or altered cadence is not credited as the same original registered SL4 service. Current scale estimation consumes depth.'],
        'qualified_capacity_screen': {'nominal_family_NPU_INT8_TOPS': nominal, 'maximum_issue_CPU_INT8_TOPS_from_saved_primary_sources': cpu, 'Mesa_cost_model_GPU_INT8_TOPS_not_measured_peak': modeled, 'conditional_sum_TOPS': cap, 'conditional_zero_overhead_transformer_pair_seconds': 2*reuse/(cap*1e12), 'conditional_zero_overhead_complete_pair_seconds': 2*(reuse+head_mac)/(cap*1e12), '20Hz_transformer_requirement_over_conditional_sum': 2*reuse*20/(cap*1e12), 'assumptions': 'Accepted precision for every counted product, compatible counting/engine assignment, independent workloads, simultaneous maximum sustained clocks/issue rates, zero transfer/nonlinear/thermal/queue/backend cost. This is not a certified whole-SoC maximum.', 'universal_hardware_impossibility_proven': False, 'DDR_bandwidth_lower_bound_proven': False},
        'what_v109_changes': {'original_native518_predictions_to_original_SL4_and_map_now_demonstrated': True, 'fresh_overlap_pairs': [[32,40],[40,48]], 'baseline_used_DINO_reuse': False, 'measured_pairs': measured, 'SDK_totals': audit['SDK_totals'], 'graph_nodes': audit['graph_nodes'], 'unique_timestamps': len(audit['unique_timestamps']), 'PLY_vertices': audit['PLY_vertices'], 'NPU_restored_hz': audit['NPU_restored_hz'], '20Hz_P95_map_cadence_trajectory_long_duration_memory_accepted': False},
        'actionable_synthesis': ['Validate bounded exact old-DINO reuse on the real two-pair executor, comparing every final prediction and original SL4/map output with v109; measure lifecycle savings instead of assuming half of the DINO wall time.', 'Compare supported core1/core7 on identical compiled artifacts and fresh inputs under the existing controller. Gate numerical/registered output parity and actual complete-boundary timing; combined core mask support alone does not prove multi-core utilization.', 'Treat 2-5s maps separately from 20Hz poses: their average dense budget is much smaller, and this capacity screen does not disqualify mapping. Actual native complete-chain time still requires substantial measured improvement.'],
        'limits': ['No universal dense-work or hardware latency lower bound; alternative exact algorithms need their own evidence.', 'FP16/INT8 routes remain allowed subject to original-model numerical and trajectory quality gates; bitwise FP32 is not added as a global acceptance requirement.', 'The previous complete budget includes heads; this audit independently recomputes DINO and aggregator, and reconciles the archived head budget without rerunning DPT derivation.', 'Bias/residual/split-recombination additions, Norm/GELU/softmax/RoPE/interpolation/layout/copies/queues/SL4/map work are excluded from MAC counts. Loaded SDK model bytes cannot determine per-request DDR traffic.', 'Two real sequential pairs do not establish stream rate/P95, general trajectory quality, 30-minute or 8GB acceptance.'],
        'source_locators': {'aggregator_configuration': [348, 414], 'original_image_encoder_and_role_tokens': [504, 521], 'alternating_frame_global_dependency': [548, 601], 'frame_BS_P_view': [605, 627], 'global_B_SP_view': [629, 657], 'original_DINO_token_preparation_and_24_layer_loop': [208, 270], 'original_vit_large_configuration': [370, 384]},
        'evidence': PINS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'transformer_reuse_MAC': reuse, 'transformer_20Hz_counted_TOPS': 2*reuse*20/1e12, 'complete_prior_20Hz_counted_TOPS': 2*(reuse+head_mac)*20/1e12}))


if __name__ == '__main__':
    main()
