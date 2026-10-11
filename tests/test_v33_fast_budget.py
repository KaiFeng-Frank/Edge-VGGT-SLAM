"""Pure arithmetic/config parsing; no neural or real-geometry execution."""
import importlib.util
from pathlib import Path

import pytest

ENTRY = Path(__file__).with_name("calculate-v33-fast-config-budget.py")
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_sparse_context/calculate-v33-fast-config-budget.py"
SPEC = importlib.util.spec_from_file_location("v33_fast_budget", ENTRY)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def config(M, L, R):
    return {"PATCHES_PER_FRAME": M, "PATCH_LIFETIME": L, "REMOVAL_WINDOW": R,
            "LOOP_CLOSURE": False, "CLASSIC_LOOP_CLOSURE": False}


def test_original21_linear_arithmetic_matches_fixed_contract():
    assert B.EDGE_LINEAR_MACS == 2699520
    assert B.update_linear_macs(96, 96, 1) == 273457152
    assert B.update_linear_macs(864, 288, 9) == 2376179712


@pytest.mark.parametrize("M,L,R,init,steady", [(96, 13, 22, 6144, 47712), (48, 11, 16, 3072, 14496)])
def test_lifecycle_append_prune_initialization_and_EOF_are_separate(M, L, R, init, steady):
    result = B.simulate_no_keyframe_deletion(config(M, L, R))
    assert result["initialization"]["edges_E"] == init
    assert result["initialization"]["graph_Update_calls"] == 12
    assert result["steady_no_deletion_envelope"]["before_Update"]["edges_E"] == steady
    assert all(row["graph_Update_calls"] == 0 for row in result["simulation_rows"][:7])
    assert all(row["graph_Update_calls"] == 1 for row in result["simulation_rows"][8:])
    assert result["termination"]["fresh_pose_count"] == 0
    # Eight full all-pairs views include 8 self groups, not merely 8*7.
    assert result["initialization"]["frame_pairs_Gij"] == 64


def test_FAST_config_merges_base_directly_not_default_first():
    base = B.parse_base_config("_C.X = 2\n_C.LOOP_CLOSURE = False\n")
    fast = B.parse_yaml_overlay("# comment\nX: 1\nMODE: 'RANDOM'\n")
    assert {**base, **fast} == {"X": 1, "LOOP_CLOSURE": False, "MODE": "RANDOM"}
    with pytest.raises((ValueError, SyntaxError)):
        B.parse_yaml_overlay("X: dangerous()")


def test_encoder_both_branches_count_all22_convolutions_and_do_not_depend_on_M():
    row = B.encoder_conv_macs()
    assert row["inet_conv_MACs"] - row["fnet_conv_MACs"] == 116 * 152 * 64 * (384 - 128)
    assert row["unchanged_by_M_or_FAST_or_sparse_context_gather"]
    assert B.encoder_conv_macs(232, 304)["total_conv_MACs"] * 4 == row["total_conv_MACs"]


def test_loop_enabled_and_incomplete_init_reject_static_envelope():
    cfg = config(48, 11, 16)
    cfg["LOOP_CLOSURE"] = True
    with pytest.raises(ValueError):
        B.simulate_no_keyframe_deletion(cfg)
    with pytest.raises(ValueError):
        B.simulate_no_keyframe_deletion(config(48, 11, 16), admitted=7)
