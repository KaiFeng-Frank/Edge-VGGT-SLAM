"""Synthetic driver scheduling/failure receipts; no hardware or model execution."""
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "deploy" / "rk3588" / "validation" / "dpvo_source_frame"
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
io=load(HERE/'v33_rknn_source_frame.py','v33_driver_test_real_metadata_only')
driver_path=HERE/'run-v33-source-frame-rknn-board.py'
driver=load(driver_path,'v33_synthetic_root_driver')


@pytest.fixture
def setup(tmp_path,monkeypatch):
    export=tmp_path/'export';export.mkdir();compile_dir=tmp_path/'compile';compile_dir.mkdir()
    ins=[dict(name=n,dtype='float32',shape=[1,1,1,882 if n=='corr' else 384]) for n in io.INPUT_NAMES]
    outs=[dict(name=n,dtype='float32',shape=[1,1,1,384 if n=='net_out' else 2]) for n in io.OUTPUT_NAMES]
    feed={x['name']:np.full(x['shape'],.25,dtype=np.float32) for x in ins}
    reference={x['name']:np.full(x['shape'],.5 if x['name']=='weight' else .125,dtype=np.float32) for x in outs}
    source=tmp_path/'synthetic-source.py';source.write_text('# synthetic only\n')
    model=export/'model.onnx';model.write_bytes(b'not-a-real-ONNX-model')
    def artifact(name,values):
        p=export/name;np.savez(p,**values)
        return dict(file=name,bytes=p.stat().st_size,sha256=io.sha256(p),arrays={k:driver.meta(v) for k,v in values.items()})
    cases=[]
    for i in range(2):
        cases.append(dict(source_plan=dict(synthetic=True),inputs_artifact=artifact('input%d.npz'%i,feed),
                          reference_artifact=artifact('reference%d.npz'%i,reference),
                          ort_artifact=artifact('ort%d.npz'%i,reference)))
    pins={str(source):io.sha256(source)}
    manifest=dict(schema=io.MANIFEST_SCHEMA,status=io.MANIFEST_STATUS,original_parameters_unchanged=True,
        sources_and_inputs_unchanged=True,cleanup_errors=[],candidate_accepted=False,strict_FP32_gate_passed=True,
        outputs_finite=True,weight_domain_valid=True,inputs=ins,outputs=outs,cases=cases,
        source_pins=pins,source_hashes_after=pins,
        model=dict(file=model.name,bytes=model.stat().st_size,sha256=io.sha256(model)))
    mp=export/'manifest.json';mp.write_text(json.dumps(manifest))
    binary=compile_dir/'block.rknn';binary.write_bytes(b'not-a-real-RKNN-model')
    helper=HERE/'v33_rknn_source_frame.py';helper_sha=io.sha256(helper)
    compiled=dict(status='compiled_source_frame_diagnostic_not_runtime_validated',released=True,
        sources_and_inputs_unchanged=True,original_parameters_unchanged=True,candidate_accepted=False,
        configuration=driver.CONFIG,manifest_sha256=io.sha256(mp),io_helper_sha256=helper_sha,
        input=dict(sha256=io.sha256(model)),inputs=ins,outputs=outs,
        output=dict(file=binary.name,bytes=binary.stat().st_size,sha256=io.sha256(binary)))
    cp=compile_dir/'compile.json';cp.write_text(json.dumps(compiled))
    affinity={4,5,6,7};monkeypatch.setattr(driver.os,'sched_getaffinity',lambda _:set(affinity))
    monkeypatch.setattr(driver.os,'sched_setaffinity',lambda _,value:(affinity.clear(),affinity.update(value)))
    monkeypatch.setattr(driver.platform,'system',lambda:'Linux');monkeypatch.setattr(driver.platform,'machine',lambda:'aarch64')
    old_read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda self:b'rockchip,rk3588\0' if str(self)=='/proc/device-tree/compatible' else old_read(self))
    monkeypatch.setattr(driver,'BASE',tmp_path)
    state=SimpleNamespace(fault=None,calls=0,closed=False)
    class FakeSession:
        def execute(self,feed,ref):
            state.calls+=1;values={k:v.copy() for k,v in ref.items()}
            if state.fault=='nan':values['net_out'].flat[0]=np.nan
            if state.fault=='negative_weight':values['weight'].flat[0]=-1e-7
            if state.fault=='source_changed':source.write_text('# synthetic mutated source\n')
            comp={k:io.compare(v,ref[k]) for k,v in values.items()}
            row=dict(execution_status='completed',candidate_accepted=False,
                outputs_finite=bool(all(np.isfinite(v).all() for v in values.values())),
                weight_domain_valid=bool(((values['weight']>=0)&(values['weight']<=1)).all()),all_outputs_zero=False,
                strict_FP32_gate_passed=all(x['gate_passed'] for x in comp.values()),
                timing=dict(forward_seconds=.001,outputs_release_seconds=.00001),
                versus_reference=comp,outputs_released=True)
            return values,row
        def close(self):state.closed=True
        def metrics(self):return dict(synthetic=True,npu_used=False,closed=state.closed,
            context_destroyed=state.closed,cleanup_errors=['synthetic destroy failure'] if state.fault=='cleanup' else [])
    fake_io=SimpleNamespace(validate_manifest=io.validate_manifest,compare=io.compare,
                           SourceFrameSession=SimpleNamespace(open_board=lambda *a,**kw:FakeSession()))
    monkeypatch.setattr(driver,'module',lambda *a:fake_io)
    def run(flags=()):
        output=tmp_path/'output'
        args=['--export-dir',str(export),'--compile-dir',str(compile_dir),
              '--io-helper',str(helper),'--abi-helper',str(tmp_path/'fake-ABI'),
              '--header',str(tmp_path/'fake-header'),'--library',str(tmp_path/'fake-library'),
              '--output-dir',str(output),'--export-sha256',io.sha256(mp),
              '--compile-sha256',io.sha256(cp),'--io-helper-sha256',helper_sha,*flags]
        code=driver.main(args);return code,json.loads((output/'run.json').read_text()),output
    return state,run


@pytest.mark.parametrize('flags,count',[
    ([],6),(['--collect-perf-detail'],2),
    (['--collect-perf-detail','--diagnostic-observe-one-invalid-output'],1)])
def test_fixed_schedule_candidate_never_adopted_and_cleanup(setup,flags,count):
    state,run=setup;code,r,_=run(flags)
    assert code==0 and state.calls==count and len(r['calls'])==count
    assert r['status']=='completed_source_frame_RKNN_diagnostic_not_adopted' and not r['candidate_accepted']
    assert state.closed and r['session']['synthetic'] and not r['NPU_used']
    assert r['sources_and_inputs_unchanged'] and r['affinity_restored'] and r['cleanup_errors']==[]
    assert all(c['candidate_accepted'] is False for c in r['calls'])


@pytest.mark.parametrize('fault',['nan','negative_weight'])
def test_invalid_raw_saved_before_fatal_numerical_gate(setup,fault):
    state,run=setup;state.fault=fault;code,r,out=run()
    assert code==1 and r['status']=='failed' and state.calls==1 and state.closed
    assert r['calls'][0]['artifact']['file']=='call-0-output.npz'
    with np.load(out/'call-0-output.npz',allow_pickle=False) as raw:
        assert (np.isnan(raw['net_out']).any() if fault=='nan' else (raw['weight']<0).any())
    assert not r['candidate_accepted'] and r['sources_and_inputs_unchanged'] and r['affinity_restored']
    json.dumps(r,allow_nan=False)


def test_explicit_invalid_profile_preserves_rejection_and_single_call(setup):
    state,run=setup;state.fault='nan'
    code,r,_=run(['--collect-perf-detail','--diagnostic-observe-one-invalid-output'])
    assert code==0 and state.calls==1 and not r['all_outputs_finite']
    assert not r['actual_strict_FP32_gate_passed'] and not r['candidate_accepted'] and state.closed


@pytest.mark.parametrize('fault',['source_changed','cleanup'])
def test_final_source_or_destroy_failure_overrides_completed_status(setup,fault):
    state,run=setup;state.fault=fault;code,r,_=run()
    assert code==1 and r['status']=='failed' and r['cleanup_errors'] and state.closed and r['affinity_restored']
    if fault=='source_changed':assert not r['sources_and_inputs_unchanged']

