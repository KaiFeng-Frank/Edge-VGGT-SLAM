"""Generated-array/fake-callback protocol tests, never factory/SDK/model execution."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HELPER = Path(os.environ.get('V42_AB_HELPER',HERE/'run-v42-callback-ab-board.py'))
if not HELPER.is_file():
    HELPER = HERE.parent/'deploy/rk3588/validation/dpvo_online_replay/run-v42-callback-ab-board.py'
spec = importlib.util.spec_from_file_location('_v42_AB_generated_tests',HELPER)
h = importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def generated():
    e = 64
    return dict(net=np.ones((1,e,384),np.float32),inp=np.ones((1,e,384),np.float32),
        corr=np.ones((1,e,882),np.float32),ii=np.zeros(e,np.int64),
        jj=np.repeat(np.arange(4,dtype=np.int64),16),kk=np.tile(np.arange(16,dtype=np.int64),4))


def test_declared_24_call_interleaving_and_three_warm_per_method_state():
    calls = h.schedule()
    assert len(calls) == 24
    for obs in h.OBSERVATIONS:
        selected = [r for r in calls if r['observation_id'] == obs]
        assert [r['method'] for r in selected] == list('ABBAABBA')
        for method in ('A','B'):
            group = [r for r in selected if r['method'] == method]
            assert [r['state_method_repeat_index'] for r in group] == [0,1,2,3]
            assert [r['excluded_first_state_method_call'] for r in group] == [True,False,False,False]


def test_owned_readonly_inputs_and_explicit_extra_field_exclusion():
    raw = generated(); raw['extra_metadata'] = np.asarray([123],np.int64)
    owned = h.owned_inputs(raw,64)
    assert set(owned) == set(h.INPUT_NAMES)
    for n,v in owned.items():
        assert v.flags.owndata and v.flags.c_contiguous and not v.flags.writeable
        assert not np.shares_memory(v,raw[n])
    raw['net'].fill(9)
    assert np.all(owned['net'] == 1)


@pytest.mark.parametrize('change', ['index_width','nonfinite','patch_relation'])
def test_input_guards(change):
    raw = generated()
    if change == 'index_width':raw['ii'] = raw['ii'].astype(np.int32)
    elif change == 'nonfinite':raw['corr'][0,0,0] = np.nan
    else:raw['kk'][0] = 16
    with pytest.raises(ValueError):h.owned_inputs(raw,64)


class FakeCallback:
    # Flags below represent a fake protocol record; no real SDK is loaded.
    def __init__(self,records,delta=0):self.records,self.delta = records,delta
    def __call__(self,net,inp,corr,ii,jj,kk):
        e = ii.size
        output = dict(net=net.copy(),delta=np.full((1,e,2),self.delta,np.float32),
                      weight=np.full((1,e,2),.5,np.float32))
        group = dict(model_key='B1_D4_M16',execution_status='completed',mode='normal',outputs_finite=True,
            weight_domain_valid=True,all_outputs_zero=False,outputs_released=True,
            native_log=dict(truncated=False,RKNN_error_detected=False),
            timing=dict(synchronous=True,**{n:0. for n in ('forward_seconds','pack_seconds','input_set_seconds',
                'run_api_seconds','outputs_get_seconds','owned_copy_seconds','outputs_release_seconds')}))
        self.records.append(dict(status='completed_diagnostic_not_adopted',route='full',NPU_used=True,
            Torch_fallback=False,fullgraph_returned=True,candidate_accepted=False,strict_FP32_gate_passed=None,
            callback_seconds=0.,model_keys=['B1_D4_M16'],SDK_execution=dict(full_graph_returned=True,
            padding=False,dropped_edges=False,sequential_groups=1,groups=[group],full_graph_seconds=0.)))
        return output


def execute_fake(delta=0):
    records = dict(A=[],B=[])
    owners = {k:FakeCallback(v,delta if k == 'B' else 0) for k,v in records.items()}
    states = []
    for obs in h.OBSERVATIONS:
        values = h.owned_inputs(generated(),64)
        states.append(dict(observation_id=obs,edges=64,values=values,before={n:h.metadata(v) for n,v in values.items()}))
    saved = []; report = {}
    def save(name,values):saved.append(name);return dict(file=name,arrays={n:h.metadata(v) for n,v in values.items()})
    return owners,records,states,save,report,saved


def test_fake_schedule_preserves_fixed_inputs_and_keeps_nested_timing_separate():
    owners,records,states,save,report,saved = execute_fake()
    h.run_calls(owners,records,states,save,report)
    assert len(saved) == len(report['calls']) == 24 and len(records['A']) == len(records['B']) == 12
    assert report['A_B_all_saved_outputs_bitwise_equal'] is True
    assert all(v['methods'][m]['warm_calls'] == 3 for v in report['warm_statistics'] for m in ('A','B'))
    assert all(c['timing']['actual_callback_wall_seconds'] >= 0 and c['timing']['full_graph_seconds'] == 0 for c in report['calls'])


def test_fake_difference_is_saved_before_stopping_without_relaxation():
    owners,records,states,save,report,saved = execute_fake(delta=1e-8)
    with pytest.raises(ValueError,match='bytewise'):h.run_calls(owners,records,states,save,report)
    assert len(saved) == len(report['calls']) == 2
    assert report['calls'][-1]['bitwise_equal_first_A_for_state'] is False


def test_invalid_output_saved_before_protocol_rejection():
    owners,records,states,save,report,saved = execute_fake()
    real = owners['A']
    def invalid(*args):
        values = real(*args);values['weight'][0,0,0] = -.001;return values
    owners['A'] = invalid
    with pytest.raises(ValueError,match='weight domain'):h.run_calls(owners,records,states,save,report)
    assert len(saved) == 1 and report['calls'] == []
