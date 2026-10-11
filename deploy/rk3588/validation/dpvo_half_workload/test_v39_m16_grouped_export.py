"""Tiny metadata/topology tests only; no images, weights or neural forwards."""
import copy
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HELPER = Path(os.environ.get('V39_M16_EXPORT_HELPER', HERE/'export-v39-m16-grouped-update-board.py'))
GROUP = Path(os.environ.get('V39_GROUPED_SOURCE_DIR',
    '/run/user/1000/edge-vggt-v38/public-snapshot/deploy/rk3588/validation/dpvo_grouped_update'))
spec = importlib.util.spec_from_file_location('v39_M16_export_test', HELPER)
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)


def run_metadata():
    weights = {str(index): {'synthetic': True} for index in range(94)}
    calls = [dict(index=0, edges=16, frame_index=1, stage='stream'),
             dict(index=1, edges=64, frame_index=5, stage='stream'),
             dict(index=2, edges=64, frame_index=5, stage='stream'),
             dict(index=3, edges=64, frame_index=127, stage='stream'),
             dict(index=4, edges=64, frame_index=191, stage='stream'),
             dict(index=5, edges=64, frame_index=191, stage='termination_offline')]
    captures = [dict(frame_index=calls[index]['frame_index'], update_index=index,
                     edges=64, entire_active_graph=True, same_call_original_Torch=True)
                for index in (1, 3, 4)]
    return dict(schema='v39_DPVO_half_original_Torch_CPU_research_replay',
        status='completed_unpaced_half_Torch_CPU_research_not_realtime',
        script_sha256=export.RUNNER_SHA256, policy_sha256=export.POLICY_SHA256,
        frames_completed=192, seed=20261011, diagnostic_capture_enabled=True,
        cleanup_errors=[], affinity_restored=True, original_weights_unchanged=True,
        original_weights_before=weights, original_weights_after=copy.deepcopy(weights),
        sources_and_inputs_unchanged=True, source_pins={'toy': '0'*64}, source_hashes_after={'toy': '0'*64},
        config=dict(patches_per_frame=16, removal_window=8, optimization_window=4,
                    patch_lifetime=4, keyframe_index=2, capacity=64,
                    loop_closure=False, classic_loop_closure=False),
        update_calls=calls, update_captures=captures)


def test_actual_initial_observation_is_derived_not_hardcoded_13_or_34():
    run = run_metadata()
    assert [record['frame_index'] for record in export.validate_run(run)] == [5, 127, 191]


@pytest.mark.parametrize('mutation', [
    lambda r: r['update_captures'].pop(),
    lambda r: r['update_captures'][0].update(update_index=2),
    lambda r: r['update_captures'][2].update(update_index=5),
    lambda r: r['update_captures'][0].update(entire_active_graph=False),
    lambda r: r['update_captures'][0].update(same_call_original_Torch=False),
    lambda r: r['config'].update(patches_per_frame=48),
    lambda r: r['config'].update(keyframe_index=4),
    lambda r: r['original_weights_after'].pop('0'),
    lambda r: r['source_hashes_after'].update(toy='1'*64),
    lambda r: r.update(cleanup_errors=['destroy failed']),
    lambda r: r.update(affinity_restored=False),
])
def test_original_capture_metadata_rejections(mutation):
    run = run_metadata()
    mutation(run)
    with pytest.raises(ValueError):
        export.validate_run(run)


def toy_inputs():
    # Deliberately shuffled full-edge order: both complete sources retain M16.
    pairs = [(source, target, patch) for source in (0, 2)
             for target in (0, 2) for patch in range(16)]
    rng = np.random.default_rng(23)
    rng.shuffle(pairs)
    ii = np.array([row[0] for row in pairs], dtype=np.int64)
    jj = np.array([row[1] for row in pairs], dtype=np.int64)
    kk = np.array([row[0]*16+row[2] for row in pairs], dtype=np.int64)
    net = np.arange(64*384, dtype=np.float32).reshape(1, 64, 384)/1000
    arrays = dict(net=net, inp=net.copy(), corr=np.zeros((1, 64, 882), np.float32), ii=ii, jj=jj, kk=kk)
    gold = dict(net=net.copy(), delta=np.zeros((1, 64, 2), np.float32), weight=np.full((1, 64, 2), .5, np.float32))
    return arrays, gold


def test_complete_M16_plan_true_batch_and_full_edge_restore():
    parent = export.utilities(GROUP)
    adapter = parent.module(GROUP/'source_frame_batches.py', '_v39_M16_toy_adapter')
    arrays, gold = toy_inputs()
    export.validate_arrays(arrays, gold, 64)
    plan = export.build_plan(adapter, arrays,
        dict(source_slots=[0, 2], source_degrees=[2, 2], source_patch_counts=[32, 32]))
    assert len(plan.batches) == 1 and plan.batches[0].shape(384) == (2, 384, 2, 16)
    assert parent.model_key(plan.batches[0]) == 'B2_D2_M16'
    indices = np.sort(plan.batches[0].canonical_edge_indices.ravel())
    np.testing.assert_array_equal(indices, np.arange(64))
    original = {key: value.copy() for key, value in arrays.items()}
    def callback(batch, feed):
        return dict(net=feed['net'].copy(), delta=np.zeros(batch.shape(2), np.float32),
                    weight=np.full(batch.shape(2), .5, np.float32))
    identity = adapter.SnapshotIdentity(1, 1, 1, 1)
    restored, calls, seconds = parent.grouped_run(adapter, plan, arrays, callback, identity)
    assert len(calls) == 1 and seconds >= 0
    for name in gold:
        np.testing.assert_array_equal(restored[name], gold[name])
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], original[name])


@pytest.mark.parametrize('mutation', [
    lambda a, g: a.update(net=a['net'].astype(np.float64)),
    lambda a, g: a['corr'].__setitem__((0, 0, 0), np.nan),
    lambda a, g: a['ii'].__setitem__(0, 63),
    lambda a, g: a['kk'].__setitem__(0, 1024),
    lambda a, g: g['weight'].__setitem__((0, 0, 0), np.float32(-1e-7)),
    lambda a, g: g.update(delta=g['delta'][:, :-1]),
])
def test_input_finite_index_dtype_and_gold_domain_rejections(mutation):
    arrays, gold = toy_inputs()
    mutation(arrays, gold)
    with pytest.raises(ValueError):
        export.validate_arrays(arrays, gold, 64)


def test_partial_source_is_rejected_without_padding_or_dropping():
    parent = export.utilities(GROUP)
    adapter = parent.module(GROUP/'source_frame_batches.py', '_v39_M16_partial_test')
    arrays, _ = toy_inputs()
    with pytest.raises(ValueError):
        adapter.build_degree_plan(*(arrays[key][:-1] for key in ('ii', 'jj', 'kk')), patches_per_frame=16)


def test_saved_artifact_raw_hash_and_path_guards(tmp_path):
    path = tmp_path/'toy.npz'
    values = {'net': np.array([1, 2, 3], dtype=np.float32)}
    np.savez(path, **values)
    item = dict(file=path.name, bytes=path.stat().st_size, sha256=export.sha(path),
                arrays={key: export.meta(value) for key, value in values.items()})
    np.testing.assert_array_equal(export.read_artifact(tmp_path, item)['net'], values['net'])
    corrupted = copy.deepcopy(item)
    corrupted['arrays']['net']['tensor_sha256'] = '0'*64
    with pytest.raises(ValueError, match='raw SHA'):
        export.read_artifact(tmp_path, corrupted)
    with pytest.raises(ValueError, match='basename'):
        export.read_artifact(tmp_path, dict(item, file='../toy.npz'))


def test_fixed_threshold_does_not_accept_small_rejected_values():
    parent = export.utilities(GROUP)
    reference = np.zeros((1, 1, 2), np.float32)
    actual = reference.copy()
    actual[0, 0, 0] = np.float32(1.01e-5)
    assert parent.compare(actual, reference)['failed_elements'] == 1
    assert parent.compare(actual, reference)['gate_passed'] is False


def test_host_main_guard_is_before_output_or_model_import(monkeypatch, tmp_path):
    output = tmp_path/'must-not-exist'
    def reject():
        raise ValueError('synthetic board-only refusal')
    monkeypatch.setattr(export, 'board_guard', reject)
    with pytest.raises(ValueError, match='board-only'):
        export.main(['--run-dir', str(tmp_path), '--update-source-dir', str(GROUP),
                     '--v26-source-dir', str(tmp_path), '--checkpoint', str(tmp_path/'checkpoint'),
                     '--output-dir', str(output)])
    assert not output.exists()
