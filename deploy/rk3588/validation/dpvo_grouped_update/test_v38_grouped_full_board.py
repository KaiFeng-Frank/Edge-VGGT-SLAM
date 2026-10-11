"""Tiny generated integer topology/FP32 buffers and fake cache; no real forward."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pytest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module;spec.loader.exec_module(module)
    return module


TESTS = Path(__file__).resolve().parent
HERE = TESTS if (TESTS/'run-v38-grouped-full-board.py').exists() else TESTS.parent/'deploy/rk3588/validation/dpvo_grouped_update'
driver = load(HERE/'run-v38-grouped-full-board.py','_v38_full_driver_test')
SDK = Path(os.environ.get('V38_GROUPED_SDK', str(HERE if (HERE/'v37_grouped_rknn.py').exists() else
    Path('/run/user/1000/edge-vggt-v37/grouped_sdk'))))
runtime = load(SDK/'v37_grouped_rknn.py','_v38_fake_test_runtime')
C = runtime.C


@pytest.mark.parametrize('profile,full_calls,sdk_calls',[(False,9,48),(True,3,16)])
def test_fixed_schedule_covers_all_three_full_graphs(profile,full_calls,sdk_calls):
    rows = driver.schedule(profile)
    assert len(rows) == full_calls
    assert sum({13:1,127:5,191:10}[observation] for observation,_,_ in rows) == sdk_calls
    assert [row[0] for row in rows] == sorted(row[0] for row in rows)
    assert len([row for row in rows if row[2]=='warm']) == (0 if profile else 6)


def test_default_rejection_precedes_source_load_and_output_writes(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):
        raise AssertionError('No source/SDK load permitted without diagnostic flag')
    monkeypatch.setattr(driver,'load',forbidden)
    args=[]
    for name in ('manifest','compile-index','capture-dir','sdk-dir','adapter','abi-helper','header','log-helper','output-dir'):
        args.extend(['--'+name,str(tmp_path/name)])
    args.extend(['--compile-index-sha256','a'*64])
    with pytest.raises(ValueError,match='before side effects'):
        driver.main(args)
    assert not (tmp_path/'output-dir').exists()


@pytest.fixture
def index_fixture(tmp_path):
    keys=('B8_D8_M48','B1_D8_M48','B1_D9_M48','B4_D11_M48','B1_D12_M48','B6_D13_M48',
          'B1_D10_M48','B5_D11_M48','B2_D12_M48','B2_D13_M48','B1_D14_M48','B1_D15_M48','B1_D16_M48','B2_D17_M48')
    manifest=dict(models={});models={}
    for key in keys:
        b,d=map(int,re.fullmatch(r'B(\d+)_D(\d+)_M48',key).groups())
        manifest['models'][key]=dict(
            inputs=[dict(name=n,dtype='float32',shape=[b,c,d,48]) for n,c in zip(C.INPUT_NAMES,(384,384,882))],
            outputs=[dict(name=n,dtype='float32',shape=[b,c,d,48]) for n,c in zip(C.OUTPUT_NAMES,(384,2,2))])
        models[key]=dict(compile_dir=str(tmp_path/key),compile_sha256='a'*64,
            model=dict(file='block.rknn',bytes=20,sha256='b'*64))
    index=dict(schema='v38_grouped_Update_compile_index',status='staged_all_actual_shapes',
        manifest_sha256=driver.MANIFEST_SHA,compiler_sha256=driver.COMPILER_SHA,contracts_sha256=driver.CONTRACTS_SHA,
        candidate_accepted=False,host_model_inference_executed=False,host_calibration_executed=False,models=models)
    return tmp_path/'index.json',manifest,index


@pytest.mark.parametrize('bad',[None,'missing_shape','model_budget'])
def test_index_complete_actual_shapes_without_host_board_file_assumptions(index_fixture,bad):
    path,manifest,index=index_fixture
    if bad=='missing_shape':index['models'].pop('B1_D17_M48') if 'B1_D17_M48' in index['models'] else index['models'].pop('B2_D17_M48')
    if bad=='model_budget':index['models']['B8_D8_M48']['model']['bytes']=256<<20
    path.write_text(json.dumps(index))
    if bad:
        with pytest.raises(ValueError):
            driver.validate_compile_index(C,path,C.sha256(path),manifest,SDK,verify_artifacts=False)
    else:
        _,specs,compiled,_=driver.validate_compile_index(C,path,C.sha256(path),manifest,SDK,verify_artifacts=False)
        assert len(specs)==14 and compiled=={} and sum(v['model_bytes'] for v in specs.values())==280
        assert specs['B8_D8_M48']['inputs'][0]['shape']==[8,384,8,48]


def toy_graph():
    path=Path(os.environ.get('V38_GROUPED_ADAPTER',str(HERE/'source_frame_batches.py' if (HERE/'source_frame_batches.py').exists() else
        Path('/run/user/1000/edge-vggt-v36/update/source_frame_batches.py'))))
    assert C.sha256(path)==C.ADAPTER_SHA256
    adapter=load(path,'_v38_fake_group_adapter')
    rows=np.array([(s,t,s*48+p) for s,targets in ((0,(0,)),(1,(0,1))) for t in targets for p in range(48)],dtype=np.int64)
    rows=rows[np.random.default_rng(9).permutation(len(rows))]
    plan=adapter.build_degree_plan(*(np.array(rows[:,i],copy=True) for i in range(3)))
    net=np.broadcast_to(np.arange(len(rows),dtype=np.float32)[None,:,None],(1,len(rows),384)).copy()
    values=dict(net=net,inp=np.full_like(net,.25),corr=np.zeros((1,len(rows),882),dtype=np.float32))
    for value in values.values():value.flags.writeable=False
    identity=adapter.SnapshotIdentity(1,2,3,4)
    return adapter,plan,values,identity


@pytest.mark.parametrize('late_failure',[False,True])
def test_timed_full_call_retains_owned_groups_without_hashing_or_partial_restore(monkeypatch,late_failure):
    adapter,plan,inputs,identity=toy_graph()
    before={name:value.copy() for name,value in inputs.items()}
    seen=[]
    class FakeCache:
        def execute(self,key,feed,reference=None,*,timeout_ms):
            assert reference is None
            seen.append(key)
            output=dict(net_out=feed['net']+feed['inp'],delta=feed['net'][:,:2].copy(),weight=np.full_like(feed['net'][:,:2],.5))
            return output,dict(execution_status='failed' if late_failure and len(seen)==2 else 'completed',
                outputs_finite=True,weight_domain_valid=True,all_outputs_zero=False)
    monkeypatch.setattr(driver,'raw_metadata',lambda *args:(_ for _ in ()).throw(AssertionError('Hash/statistics inside full call')))
    if late_failure:
        with pytest.raises(runtime.GroupExecutionError) as caught:
            driver.full_call(runtime,adapter,plan,inputs,FakeCache(),identity)
        assert len(caught.value.completed_group_records)==2 and caught.value.raw_outputs
        assert caught.value.full_wall_seconds>=0
    else:
        outputs,row,groups=driver.full_call(runtime,adapter,plan,inputs,FakeCache(),identity)
        assert np.array_equal(outputs['net'],inputs['net']+inputs['inp'])
        assert np.array_equal(outputs['delta'],inputs['net'][:,:,:2]) and (outputs['weight']==.5).all()
        assert len(groups)==2 and row['sequential_groups']==2
        assert row['candidate_accepted'] is False and row['VO_state_committed'] is False
        assert row['full_wall_seconds']>=row['full_graph_seconds']>=0
    assert all(np.array_equal(inputs[name],value) for name,value in before.items())


def test_raw_nonfinite_metadata_and_owned_artifact_are_JSON_safe(tmp_path):
    value=np.array([np.nan,np.inf,-1,.5],dtype=np.float32)
    report=driver.artifact(tmp_path/'invalid-raw.npz',dict(weight=value))
    assert report['arrays']['weight']['nonfinite_elements']==2
    json.dumps(report,allow_nan=False)
    with np.load(tmp_path/'invalid-raw.npz',allow_pickle=False) as saved:
        assert saved['weight'].tobytes()==value.tobytes()
