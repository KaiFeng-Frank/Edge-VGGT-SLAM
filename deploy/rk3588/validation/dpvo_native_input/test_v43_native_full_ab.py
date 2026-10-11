"""Fake cache and generated values only; no actual model, SDK or saved feed."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import uuid

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
HERE = Path(os.environ.get('V43_TRANSPORT_HELPERS', TEST_DIR))
if not (HERE/'run-v43-native-full-ab-board.py').is_file():
    HERE = TEST_DIR.parent/'deploy/rk3588/validation/dpvo_native_transport'


def load(path):
    name = '_v43_full_test_'+uuid.uuid4().hex
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m


D = load(HERE/'run-v43-native-full-ab-board.py')


def prerequisite():
    m = dict(closed=True, context_destroyed=True, cleanup_errors=[], sdk_run_api_calls=4,
             completed_calls=4, SDK_outputs_release_attempts=4, SDK_outputs_releases=4,
             current_DMA_bytes=0, current_model_host_bytes=0)
    return dict(schema='v43_native_single_B2D6_saved_state_ABBA',
        status='completed_single_B2D6_AB_bitwise_equal_diagnostic_not_adopted',
        model_key='B2_D6_M16', candidate_accepted=False, A_B_all_outputs_bitwise_equal=True,
        SDK_native_transport_equivalence_for_this_feed_only=True, cleanup_errors=[], sources_unchanged=True,
        calls=[dict(method=v, bitwise_equal_first_A=True) for v in ('A', 'B', 'B', 'A')*2],
        source_pins={'/FAKE/libv43_native_fp16_pack.so':'a'*64}, closed_contexts={'A':copy.deepcopy(m), 'B':copy.deepcopy(m)})


def test_actual_prerequisite_contract_does_not_claim_quality_adoption():
    row = prerequisite(); D.validate_single_prerequisite(row, 'a'*64)
    assert row['candidate_accepted'] is False


@pytest.mark.parametrize('failure', ['equal', 'release', 'library'])
def test_prerequisite_failure_rejected(failure):
    row = prerequisite()
    if failure == 'equal': row['calls'][1]['bitwise_equal_first_A'] = False
    if failure == 'release': row['closed_contexts']['B']['SDK_outputs_releases'] = 3
    with pytest.raises(ValueError): D.validate_single_prerequisite(row, 'b'*64 if failure == 'library' else 'a'*64)


class FakeCache:
    def __init__(self, specs, factory=None, **kw):
        self.specs, self.factory, self.kw = copy.deepcopy(specs), factory, kw
        self.state = dict(created=0, resident_contexts=0, current_DMA_bytes=0,
                          current_model_host_bytes=0, cleanup_errors=[], closed=False)
    def metrics(self): return copy.deepcopy(self.state)
    def close(self): self.state['closed'] = True


def fake_owner():
    specs = {key:dict(model_path='FAKE_MODEL', model_sha256='b'*64, inputs=[], outputs=[],
                     export_strict_FP32_gate_passed=True) for key in D.MODEL_KEYS}
    return NS(cache=FakeCache(specs), runtime=object(), route='full', source_pins={'FAKE_SOURCE':'a'*64},
              catalogue_provenance={'metadata_only_reporting':True})


def test_empty_cache_replacement_preserves_callback_specs_and_adds_native_factory():
    owner = fake_owner(); old = owner.cache
    seen = []
    class Session:
        @staticmethod
        def open_board(*args, **kw): seen.append((args, kw)); return 'SYNTHETIC_SESSION'
    runtime = NS(GroupedSession=Session, BoundedContextCache=FakeCache, source_pins={'FAKE_PACK':'c'*64},
                 input_transport='SYNTHETIC_NATIVE')
    cfg = {k:'FAKE' for k in ('abi_path', 'header_path', 'library_path', 'log_helper_path')}
    cfg.update(max_contexts=9, max_model_bytes=256 << 20, max_dma_bytes=512 << 20)
    D.replace_empty_cache(owner, runtime, cfg, {'FAKE_MANIFEST':'d'*64}, Path('FAKE_DIRECTORY'))
    assert old.state['closed'] and owner.cache is not old and owner.cache.specs == old.specs
    assert owner.runtime is runtime and owner.catalogue_provenance['metadata_only_reporting']
    assert owner.catalogue_provenance['native_input_transport_added']
    owner.cache.factory(owner.cache.specs['B2_D6_M16'])
    assert seen[0][1]['diagnostic_allow_native_fp16'] and seen[0][1]['diagnostic_allow_fp32_rejected']
    assert seen[0][1]['collect_perf_detail'] is False
    assert seen[0][1]['pins'] == owner.source_pins


def test_live_cache_cannot_be_replaced_or_leaked():
    owner = fake_owner(); owner.cache.state['created'] = 1
    with pytest.raises(ValueError, match='unused'): D.replace_empty_cache(owner, object(), {}, {}, Path('FAKE'))
    assert not owner.cache.state['closed']


def test_reused_schedule_and_bitwise_guard_with_generated_toy_outputs():
    candidates = [Path(os.environ['V43_PARENT_AB_HARNESS'])] if 'V43_PARENT_AB_HARNESS' in os.environ else [
        HERE.parent/'public-snapshot/deploy/rk3588/validation/dpvo_update_performance/run-v42-callback-ab-board.py',
        HERE.parent/'dpvo_update_performance/run-v42-callback-ab-board.py']
    path = next(p for p in candidates if p.is_file())
    H = D.load_parent(path, D.PARENT_HARNESS_SHA256)
    records, owners = {'A':[], 'B':[]}, {}
    states = []
    for observation in H.OBSERVATIONS:
        values = {n:np.zeros((1, 2, c), np.float32) for n,c in zip(H.INPUT_NAMES[:3], (384,384,882))}
        values.update({n:np.zeros(2, np.int64) for n in H.INPUT_NAMES[3:]})
        values = H.owned_inputs(values, 2)
        states.append(dict(observation_id=observation, edges=2, values=values,
                           before={n:H.metadata(v) for n,v in values.items()}))
    def owner(method):
        def call(*values):
            timing = {n:0.0 for n in ('forward_seconds', 'pack_seconds', 'input_set_seconds', 'run_api_seconds',
                                     'outputs_get_seconds', 'owned_copy_seconds', 'outputs_release_seconds')}
            timing['synchronous'] = True
            group = dict(model_key='B2_D6_M16', execution_status='completed', mode='normal',
                outputs_finite=True, weight_domain_valid=True, all_outputs_zero=False, outputs_released=True,
                timing=timing, native_log=dict(truncated=False, RKNN_error_detected=False))
            records[method].append(dict(status='completed_diagnostic_not_adopted', route='full', NPU_used=True,
                Torch_fallback=False, fullgraph_returned=True, candidate_accepted=False, strict_FP32_gate_passed=None,
                callback_seconds=0.0, model_keys=['B2_D6_M16'], SDK_execution=dict(full_graph_returned=True,
                padding=False, dropped_edges=False, sequential_groups=1, groups=[group], full_graph_seconds=0.0)))
            return {n:np.full((1,2,c),.5,np.float32) for n,c in zip(H.OUTPUT_NAMES,(384,2,2))}
        return call
    owners = {m:owner(m) for m in ('A','B')}; report = {}; saved = []
    def save(name, values): saved.append(name); return {'file':name}
    H.run_calls(owners, records, states, save, report)
    assert len(saved) == len(report['calls']) == 24 and report['A_B_all_saved_outputs_bitwise_equal']
    assert all(r['methods'][m]['warm_calls'] == 3 for r in report['warm_statistics'] for m in ('A','B'))
    assert all(c['candidate_accepted'] is False and c['CPU_reference_supplied'] is False for c in report['calls'])
