"""Tiny saved-array audit arithmetic tests, no ORT/SDK/model execution."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

here=Path(__file__).resolve().parent
path=here/'audit-v29-update-ort.py'
if not path.is_file():path=here.parent/'deploy/rk3588/validation/audit-v29-update-ort.py'
spec=importlib.util.spec_from_file_location('_synthetic_v29_ort_audit',path)
audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit)


def test_exact_element_gate_and_nonfinite_JSON():
    ref=np.array([0.,1.,2.],np.float32);actual=ref.copy();actual[0]=np.float32(2e-5)
    row=audit.residual(actual,ref)
    assert row['failed_elements']==1 and not row['gate_passed']
    actual[0]=np.nan;bad=audit.residual(actual,ref)
    assert not bad['gate_passed'] and bad['relative_l2'] is None
    json.dumps(dict(residual=bad,finite=audit.finite(actual)),allow_nan=False)


@pytest.mark.parametrize('field,value,expected',[('relative_l2',1e-6*(1+1e-14),True),
    ('relative_l2',1e-6*(1+1e-6),False),('failed_elements',1,False),('gate_passed',False,False),('max_abs',1e-10,False)])
def test_report_rounding_does_not_relax_any_acceptance_field(field,value,expected):
    row=dict(relative_l2=1e-6,failed_elements=0,gate_passed=True,max_abs=0.,bitwise_equal=True)
    recorded=dict(row);recorded[field]=value
    assert audit.recorded_residual_matches(row,recorded)==expected


def test_artifact_hash_and_raw_shape_tampering_are_recorded(tmp_path):
    values={'net_out':np.arange(6,dtype=np.float32).reshape(1,3,2)}
    path=tmp_path/'data.npz';np.savez(path,**values)
    row=dict(file=path.name,sha256=audit.digest(path),bytes=path.stat().st_size,arrays={k:audit.metadata(v) for k,v in values.items()})
    checker=audit.Audit();actual=checker.artifact(tmp_path,row)
    assert checker.failures==[];np.testing.assert_array_equal(actual['net_out'],values['net_out'])
    row['arrays']['net_out']['shape']=[1,2,3]
    checker.artifact(tmp_path,row)
    assert 'raw_metadata:data.npz' in checker.failures


def test_all_source_states_have_unfiltered_identity_and_shifted_steps():
    capture=dict(scenarios=[]);export=dict(scenarios=[])
    for key in ('E96','E864'):
        for suffix in ('identity','shifted'):
            name=key+'_'+suffix;rows=[dict(step=i,inputs={},outputs={}) for i in range(4)]
            capture['scenarios'].append(dict(name=name,states=rows))
            export['scenarios'].append(dict(name=name,model_key=key,states=rows,topology={}))
    assert len(audit.expected_states(capture,export))==16
    capture['scenarios'][-1]['states'].pop()
    with pytest.raises(ValueError):audit.expected_states(capture,export)


def test_timing_stats_keep_models_separate():
    calls=[dict(model_key=key,timing={'forward_seconds':value}) for key,value in [('E96',.1),('E96',.2),('E864',1.)]]
    stats=audit.timing_summary(calls)
    assert stats['E96']['count']==2 and stats['E96']['mean_seconds']==pytest.approx(.15)
    assert stats['E864']['count']==1 and stats['E864']['mean_seconds']==1.
