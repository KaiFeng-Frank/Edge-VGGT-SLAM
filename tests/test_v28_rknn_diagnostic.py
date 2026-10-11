"""Synthetic diagnostic entrypoint checks; no board, model or SDK library."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj
    spec.loader.exec_module(obj);return obj


def helper_path(name):
    here=Path(__file__).resolve().parent
    p=here/name
    return p if p.is_file() else here.parent/'deploy/rk3588/validation'/name


baseline=load(Path(__file__).with_name('test_v28_rknn_inputs.py'),'_v28_diagnostic_test_fixture')


@pytest.fixture
def scenario(monkeypatch,tmp_path):
    diagnostic=load(helper_path('diagnose-v28-dpvo-update-rknn-board.py'),'_v28_diagnostic_synthetic')
    runner_path=helper_path('run-v28-dpvo-update-rknn-board.py')
    runner=load(runner_path,'_v28_diagnostic_real_execute_synthetic_main')
    assert hashlib.sha256(runner_path.read_bytes()).hexdigest()==diagnostic.FROZEN_SHA
    abi_path=Path(os.environ.get('EDGE_VGGT_ABI_HELPER',helper_path('run-v12-rknn-block-board.py')))
    assert hashlib.sha256(abi_path.read_bytes()).hexdigest()==baseline.ABI_SHA
    abi=load(abi_path,'_v28_diagnostic_abi_only')
    out=tmp_path/'new-diagnostic-output'
    state=dict(fault=None,source_ok=True,destroy_code=0,entry_changed=False,entered=False)
    arrays=baseline.feed();golden=dict(net=np.arange(12,dtype=np.float32).reshape(1,3,4),
                  delta=np.full((1,3,2),.25,np.float32),weight=np.full((1,3,2),.5,np.float32))
    binds=baseline.packing.validate_input_attrs(baseline.contracts(),baseline.attrs(baseline.contracts()))

    def fake_main(argv):
        # Only the frozen runner's main orchestration is replaced. Its actual
        # execute/compare/require code and typed packer operate on FakeSDK.
        state['entered']=True;assert '--collect-perf-detail' in argv
        out.mkdir(exist_ok=False)
        sdk=baseline.SyntheticSDK(abi,arrays,state['fault']);state['sdk']=sdk
        report=dict(status='failed',calls=[],cleanup_errors=[],destroy_code=0,
                    sources_and_inputs_unchanged=True,source_pins={'frozen':diagnostic.FROZEN_SHA},
                    source_hashes_after={'frozen':diagnostic.FROZEN_SHA})
        try:
            for index in range(3):
                values,timing=runner.execute(sdk,abi.CONTEXT(1),abi,baseline.packing,arrays,binds,
                                            baseline.output_contracts(),report['cleanup_errors'])
                path=out/f'call-{index:02d}.npz';np.savez(path,**values)
                report['calls'].append(dict(index=index,timing=timing,
                       outputs={'file':path.name},versus_original={key:runner.compare(value,golden[key])
                                                                 for key,value in values.items()}))
            report['status']='completed_mixed_input_Update_RKNN_probe_not_VO'
        except Exception as error:report.update(status='failed',error=repr(error))
        finally:
            state['synthetic_context_destroys']=state.get('synthetic_context_destroys',0)+1
            report['destroy_code']=state['destroy_code']
            if state['destroy_code']:report['cleanup_errors'].append('rknn_destroy '+str(state['destroy_code']))
            report['sources_and_inputs_unchanged']=state['source_ok']
            (out/'run.json').write_text(json.dumps(report,allow_nan=False))
        return 1 if report['status']=='failed' else 0

    runner.main=fake_main
    # Fake only the board predicate and the entrypoint's dynamic module loader.
    # No source files are altered and no real SDK calls are made.
    monkeypatch.setattr(diagnostic.platform,'machine',lambda:'aarch64')
    old_read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:b'rockchip,rk3588' if str(p)=='/proc/device-tree/compatible' else old_read(p))
    monkeypatch.setattr(diagnostic.importlib.util,'spec_from_file_location',lambda *a,**k:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda obj:None)))
    monkeypatch.setattr(diagnostic.importlib.util,'module_from_spec',lambda spec:runner)
    old_sha=diagnostic.sha;own_reads=0
    def monitored_sha(path):
        nonlocal own_reads
        if Path(path)==Path(diagnostic.__file__):
            own_reads+=1
            if state['entry_changed'] and own_reads>1:return 'b'*64
        return old_sha(path)
    monkeypatch.setattr(diagnostic,'sha',monitored_sha)
    args=['--frozen-runner',str(runner_path),'--output-dir',str(out)]
    return diagnostic,state,out,args


@pytest.mark.parametrize('fault,expected_guard',[('nonfinite','Nonfinite SDK output'),
                                                ('micro_negative','Invalid sigmoid weight range')])
def test_invalid_raw_output_is_saved_once_and_never_accepted(scenario,fault,expected_guard):
    diag,state,out,args=scenario;state['fault']=fault
    assert diag.main(args)==0
    report=json.loads((out/'run.json').read_text())
    assert report['status']=='single_call_invalid_output_diagnostic_saved_not_accepted'
    assert report['diagnostic_SDK_calls']==1 and len(report['calls'])==1
    assert report['diagnostic_post_copy_checks_intercepted']==[expected_guard]
    assert report['original_runner_terminal_status']=='failed'
    assert 'intentionally stops' in report['error']
    assert not report['candidate_runtime_accepted'] and not report['outputs_clipped_or_modified']
    assert not report['FP32_thresholds_relaxed'] and report['diagnostic_sources_unchanged']
    assert state['sdk'].run_calls==state['sdk'].releases==state['synthetic_context_destroys']==1
    with np.load(out/'call-00.npz',allow_pickle=False) as data:
        if fault=='nonfinite':
            assert np.isnan(data['delta'].flat[0])
            result=report['calls'][0]['versus_original']['delta']
            assert result['gate_passed'] is False and result['nan_elements']==1
            assert result['failed_elements'] is None
        else:assert data['weight'].flat[0]==np.float32(-1.19e-7)
        assert not any((data[key]==-9999).any() for key in data.files)


@pytest.mark.parametrize('fault,releases',[('input',0),('run',0),('partial',1),('frame',1),('bytes',1),('release',1)])
def test_protocol_or_API_failure_is_not_intercepted_as_saved(scenario,fault,releases):
    diag,state,out,args=scenario;state['fault']=fault
    assert diag.main(args)==1
    report=json.loads((out/'run.json').read_text())
    assert report['status']=='failed' and report['calls']==[]
    assert not report['candidate_runtime_accepted']
    assert state['sdk'].releases==releases and state['synthetic_context_destroys']==1


@pytest.mark.parametrize('failure',['source','entry_source','destroy'])
def test_saved_invalid_output_requires_clean_lifecycle_and_unchanged_sources(scenario,failure):
    diag,state,out,args=scenario;state['fault']='micro_negative'
    if failure=='source':state['source_ok']=False
    elif failure=='entry_source':state['entry_changed']=True
    else:state['destroy_code']=-2
    assert diag.main(args)==1
    report=json.loads((out/'run.json').read_text())
    assert len(report['calls'])==1 and report['status']=='failed'
    assert not report['candidate_runtime_accepted']
    if failure=='entry_source':assert not report['diagnostic_sources_unchanged']


def test_finite_valid_output_is_not_mislabeled_invalid_or_accepted(scenario):
    diag,state,out,args=scenario
    assert diag.main(args)==1
    report=json.loads((out/'run.json').read_text())
    assert report['diagnostic_post_copy_checks_intercepted']==[]
    assert not report['candidate_runtime_accepted'] and report['status']=='failed'
    assert state['sdk'].run_calls==state['sdk'].releases==1


@pytest.mark.parametrize('which',['board','source'])
def test_host_or_modified_frozen_source_is_rejected_before_output_or_SDK(scenario,monkeypatch,which):
    diag,state,out,args=scenario
    if which=='board':monkeypatch.setattr(diag.platform,'machine',lambda:'x86_64')
    else:monkeypatch.setattr(diag,'FROZEN_SHA','0'*64)
    with pytest.raises(ValueError):diag.main(args)
    assert not state['entered'] and not out.exists()
