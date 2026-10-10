"""Synthetic guards for the isolated board-only MLP approximation probe."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load('_test_v22_mlp_probe', ROOT/'deploy/rk3588/validation/probe-v22-come-mlp-board.py')
merger = load('_test_v22_mlp_probe_merge', ROOT/'deploy/rk3588/validation/v22_token_merge.py')


def capture_fixture():
    return dict(kind='original_VGGT_CoMe_MLP_stage_capture',
        status='captured_original_FP32_stages_bitwise_bound',script_sha256=probe.CAPTURE_SCRIPT_SHA,
        original_weights=True,architecture_changed=False,ORB_used=False,npu_used=False,
        hooks_removed=True,cleanup_errors=[],full_model_forward_count=1,
        architecture=dict(views=2,grid_hw=[18,24],special_tokens=5,channels=1024,
            gelu_approximate='none',dropout=0.,dino_extra_norm=False,
            dino_hook='aggregator.patch_embed.blocks[14] forward output'),
        global0_vs_profile={key:True for key in ('input','output','pos')},
        full_predictions_vs_original={key:True for key in ('depth','depth_conf','intrinsic','extrinsic','pose_enc')},
        sources_and_inputs_unchanged=True)


@pytest.mark.parametrize('key,value',[
    ('script_sha256','wrong'),('original_weights',False),('npu_used',True),
    ('ORB_used',True),('hooks_removed',False),('full_model_forward_count',2),
    ('global0_vs_profile',{}),('full_predictions_vs_original',{}),
    ('sources_and_inputs_unchanged',False),('cleanup_errors',['failed']),
])
def test_capture_provenance_rejects_mutations(key,value):
    row = capture_fixture()
    probe.validate_capture(row)
    row[key] = value
    with pytest.raises(ValueError):
        probe.validate_capture(row)


def test_rejects_final_DINO_norm_and_wrong_stage():
    row = capture_fixture()
    for key,value in [('dino_extra_norm',True),('dino_hook','blocks[15]'),('grid_hw',[24,18])]:
        bad = copy.deepcopy(row)
        bad['architecture'][key] = value
        with pytest.raises(ValueError):
            probe.validate_capture(bad)


@pytest.mark.parametrize('count',[0,1,48,96])
def test_actual_mask_counts_and_specials_are_bounded(count):
    mask = np.zeros((2,6,8),np.bool_)
    mask.flat[:count] = True
    plan = merger.build_plan(mask)
    result = probe.validate_plan(plan,mask)
    assert result['actual_compact_rows']==874-count*8
    assert result['actual_merged_groups']==count
    assert result['special_rows_preserved']==10


@pytest.mark.parametrize('mutation',['special','view','group'])
def test_reverse_mapping_cannot_merge_specials_or_wrong_regions(mutation):
    mask = np.zeros((2,6,8),np.bool_)
    original = merger.build_plan(mask)
    plan = SimpleNamespace(views=original.views,tokens_per_view=original.tokens_per_view,
        compact_rows=original.compact_rows,reverse_indices=original.reverse_indices.copy(),
        view_offsets=original.view_offsets.copy())
    if mutation=='special':
        plan.reverse_indices[0,1] = plan.reverse_indices[0,0]
    elif mutation=='view':
        plan.reverse_indices[1,5] = 0
    else:
        plan.reverse_indices[0,6] = plan.reverse_indices[0,5]
    with pytest.raises(ValueError):
        probe.validate_plan(plan,mask)


def test_error_gate_distinguishes_signed_zero_and_rejects_nonfinite():
    reference = np.array([0.,1.],np.float32)
    actual = np.array([-0.,1.],np.float32)
    row = probe.error_stats(actual,reference)
    assert row['strict_FP32_allclose'] is True and row['bitwise_equal'] is False
    with pytest.raises(ValueError):
        probe.error_stats(np.array([0.,np.nan],np.float32),reference)
    with pytest.raises(ValueError):
        probe.error_stats(actual.astype(np.float64),reference)


def test_all_region_errors_include_all_specials_and_patch_channels():
    reference = np.ones((1,874,1024),np.float32)
    actual = reference.copy()
    actual.reshape(2,437,1024)[0,5,0] += 1.
    mask = np.zeros((2,6,8),np.bool_)
    mask[0,0,0] = True
    rows = probe.region_errors(actual,reference,mask)
    assert len(rows['regions'])==96
    assert rows['specials']['elements']==10*1024
    assert rows['merged_patches']['elements']==9*1024
    assert rows['unmerged_patches']['elements']==(864-9)*1024
    assert sum(row['outside_strict_FP32'] for row in rows['regions'])==1


def test_ABBA_timing_excludes_observer_hash_and_file_work():
    clock_value = [0.]
    seen = []
    def clock():
        return clock_value[0]
    def original():
        clock_value[0] += 2.
        return 'original'
    def merged():
        clock_value[0] += 1.
        return 'merged'
    def observe(index,name,value):
        assert name==value
        seen.append((index,name))
        clock_value[0] += 100.
        return dict(file_work_excluded=True)
    rows = probe.timed_calls(original,merged,observe,clock)
    assert [row['mode'] for row in rows]==list(probe.MEASURED_ORDER)
    assert [row['end_to_end_seconds'] for row in rows]==[2.,1.,1.,2.,2.,1.,1.,2.]
    assert len(seen)==8


def test_BF16_raw_bits_preserve_original_values_and_ties():
    torch = pytest.importorskip('torch')
    value = torch.tensor([0.,-0.,1.,1.0078125,2.],dtype=torch.bfloat16)
    bits = probe.bf16_bits(value)
    assert bits.dtype==np.uint16
    restored = (bits.astype(np.uint32)<<16).view(np.float32)
    assert restored.tobytes()==value.float().numpy().tobytes()
    assert bits[0]!=bits[1]
    with pytest.raises(ValueError):
        probe.bf16_bits(value.float())


def test_nonboard_execution_guard_runs_before_any_model_loading(tmp_path):
    with pytest.raises(ValueError,match='board-only'):
        probe.board_guard('x86_64',tmp_path/'new')
    with pytest.raises(ValueError,match='BASE/experiments'):
        probe.board_guard('aarch64',tmp_path/'new')
