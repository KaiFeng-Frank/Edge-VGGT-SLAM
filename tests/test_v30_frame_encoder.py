"""Synthetic only: original integer Patchifier/RNG and fake ORT lifecycle."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
HELPER=HERE/'v30_frame_encoder.py'
if not HELPER.is_file():HELPER=HERE.parent/'deploy/rk3588/validation/dpvo_stream/v30_frame_encoder.py'
spec=importlib.util.spec_from_file_location('v30_encoder_test',HELPER)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.fixture(scope='module')
def oracle():
    path=Path(os.environ.get('DPVO_CPU_OPS_REFERENCE',str(HERE.parent/'deploy/rk3588/validation/dpvo_cpu_ops.py')))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==module.SOURCE_PINS['dpvo_cpu_ops.py']
    spec=importlib.util.spec_from_file_location('v30_frozen_patchify_oracle',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def tiny_features():
    rng=np.random.default_rng(8)
    return (rng.normal(size=(3,5,7)).astype(np.float32),rng.normal(size=(4,5,7)).astype(np.float32),
            np.array([[1,1],[5,3],[2,2],[2,2]],dtype=np.int64))


def test_nonsquare_interior_and_repeated_integer_patches_match_original_cpuops(oracle):
    f,c,xy=tiny_features();result=module.features_to_patches(f,c,xy)
    coords=torch.from_numpy(xy[None].astype(np.float32))
    expected_g=oracle.patchify(torch.from_numpy(f[None]),coords,1).numpy()[0]
    expected_c=oracle.patchify(torch.from_numpy(c[None]),coords,0).numpy()[0,:,:,0,0]
    assert np.array_equal(result['gmap'],expected_g)
    assert np.array_equal(result['context'],expected_c)
    assert result['gmap'].shape==(4,3,3,3) and result['context'].shape==(4,4)
    for i,(x,y) in enumerate(xy):
        for dy in range(3):
            for dx in range(3):assert np.array_equal(result['patches'][i,:,dy,dx],[x+dx-1,y+dy-1,1])


def test_five_and_four_dimensional_singleton_features_do_not_rescale():
    f,c,xy=tiny_features();r=module.features_to_patches(f[None,None],c[None],xy)
    assert np.array_equal(r['fmap'],f) and np.array_equal(r['context'][0],c[:,1,1])


def test_returned_arrays_are_owned_and_inputs_are_unchanged():
    f,c,xy=tiny_features();before=[v.copy() for v in (f,c,xy)];out=module.features_to_patches(f,c,xy)
    for actual,expected in zip((f,c,xy),before):assert np.array_equal(actual,expected)
    f.fill(9);c.fill(10);xy.fill(1)
    assert not np.all(out['fmap']==9) and not np.all(out['context']==10)
    assert out['centers'][1,0]==5
    assert all(v.flags.c_contiguous and v.flags.owndata for v in out.values())


def test_all_uint8_values_use_original_FP32_normalization_order():
    image=np.stack([np.arange(256,dtype=np.uint8)]*3,axis=-1)[None]
    expected=(2*(torch.from_numpy(image.transpose(2,0,1).copy())[None]/255.0)-.5).numpy()
    actual=module.normalize_bgr(image)
    assert np.array_equal(actual,expected) and actual.dtype==np.float32
    assert actual[0,0,0,0]==-.5 and actual[0,0,0,255]==1.5


def test_colors_follow_original_online_normalized_BGR_to_RGB_truncation(oracle):
    f,c,xy=tiny_features();image=(np.arange(20*28*3).reshape(20,28,3)%256).astype(np.uint8)
    normalized=module.normalize_bgr(image)[0]
    result=module.features_to_patches(f,c,xy,normalized_bgr=normalized)
    coords=torch.from_numpy((4*xy+2)[None].astype(np.float32))
    clr=oracle.patchify(torch.from_numpy(normalized[None]),coords,0)[0,:,:,0,0]
    expected=((clr[:,[2,1,0]]+.5)*(255./2)).to(torch.uint8).numpy()
    assert np.array_equal(result['colors'],expected) and result['colors'].dtype==np.uint8


def test_shared_TorchCPU_rng_draws_x_then_y_and_no_depth_values():
    actual=torch.Generator(device='cpu').manual_seed(123)
    expected=torch.Generator(device='cpu').manual_seed(123)
    x=torch.randint(1,151,(1,96),generator=expected);y=torch.randint(1,115,(1,96),generator=expected)
    centers=module.random_centers(152,116,96,actual)
    assert np.array_equal(centers,torch.stack((x,y),dim=-1)[0].numpy())
    assert torch.equal(torch.rand((96,),generator=actual),torch.rand((96,),generator=expected))


@pytest.mark.parametrize('bad',[
    np.array([[0,2]],dtype=np.int64),np.array([[6,2]],dtype=np.int64),
    np.array([[2,4]],dtype=np.int64),np.array([[2,2]],dtype=np.float32),
    np.empty((0,2),dtype=np.int64)])
def test_invalid_centers_are_rejected(bad):
    f,c,_=tiny_features()
    with pytest.raises(ValueError):module.features_to_patches(f,c,bad)


def test_caller_supplies_depth_but_no_encoder_depth_rng():
    f,c,xy=tiny_features();d=np.array([0.,.2,.8,1.],dtype=np.float32)
    result=module.features_to_patches(f,c,xy,inverse_depths=d)
    assert np.array_equal(result['patches'][:,2],np.broadcast_to(d[:,None,None],(4,3,3)))
    with pytest.raises(ValueError):module.features_to_patches(f,c,xy,inverse_depths=np.array([np.nan]*4,dtype=np.float32))
    with pytest.raises(ValueError):module.features_to_patches(f,c,xy,inverse_depths=-np.ones(4,dtype=np.float32))


def test_host_guard_precedes_file_loading_and_session_creation(monkeypatch):
    monkeypatch.setattr(module.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(module,'_create_session',lambda _:pytest.fail('Session factory called on host'))
    with pytest.raises(ValueError,match='board-only'):module.BoardORTEncoder('missing.onnx','missing-export.json','missing-ort.json')


class FakeSession:
    """No graph or model computation; fixed metadata and broadcast constants."""
    def __init__(self):self.calls=0
    def get_providers(self):return ['CPUExecutionProvider']
    def get_inputs(self):return [SimpleNamespace(name='normalized_bgr',shape=list(module.INPUT_SHAPE),type='tensor(float)')]
    def get_outputs(self):return [SimpleNamespace(name=n,shape=list(s),type='tensor(float)') for n,s in zip(('fmap','imap'),module.OUTPUT_SHAPES)]
    def run(self,names,feed):
        assert names==['fmap','imap'] and tuple(feed['normalized_bgr'].shape)==module.INPUT_SHAPE
        self.calls+=1
        return [np.broadcast_to(np.array(v,dtype=np.float32),s) for v,s in zip((.75,1.25),module.OUTPUT_SHAPES)]


@pytest.fixture
def fake_contract(tmp_path,monkeypatch):
    model=tmp_path/'encoders.onnx';model.write_bytes(b'SYNTHETIC_NOT_AN_ONNX_MODEL')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(module,'MODEL_SHA',digest(model))
    pins={name:sha for name,sha in module.SOURCE_PINS.items()}
    common=dict(script_sha256=module.PROBE_SHA,capture_sha256=module.CAPTURE_SHA,checkpoint_sha256=module.CHECKPOINT_SHA,
                state_unchanged=True,source_pins=pins,source_hashes_after=pins.copy(),sources_and_inputs_unchanged=True,cleanup_errors=[])
    exported=dict(common,status='exported_not_runtime_validated',onnx=dict(file=model.name,sha256=module.MODEL_SHA,bytes=model.stat().st_size),input_shape=list(module.INPUT_SHAPE),output_shapes=[list(s) for s in module.OUTPUT_SHAPES])
    export=tmp_path/'export.json';export.write_text(json.dumps(exported));monkeypatch.setattr(module,'EXPORT_SHA',digest(export))
    verified=dict(common,status='completed_CPU_ORT_gate',export_sha256=module.EXPORT_SHA,onnx=exported['onnx'],exact_reference_gate_passed=True,providers=['CPUExecutionProvider'],graph_optimization='ORT_DISABLE_ALL')
    ort=tmp_path/'ort.json';ort.write_text(json.dumps(verified));monkeypatch.setattr(module,'ORT_SHA',digest(ort))
    monkeypatch.setattr(module,'_board_guard',lambda:None)
    session=FakeSession();monkeypatch.setattr(module,'_create_session',lambda _:session)
    kwargs=dict(model_sha256=module.MODEL_SHA,export_sha256=module.EXPORT_SHA,ort_sha256=module.ORT_SHA)
    return (model,export,ort),kwargs,session


def test_fake_ORT_owned_outputs_timing_and_idempotent_close(fake_contract):
    files,kwargs,session=fake_contract
    enc=module.BoardORTEncoder(*files,**kwargs)
    out=enc.encode(np.zeros((464,608,3),dtype=np.uint8),centers=np.array([[1,1],[150,114]],dtype=np.int64))
    assert out['fmap'].shape==(128,116,152) and np.all(out['gmap']==.75) and np.all(out['context']==1.25)
    assert np.all(out['patches'][:,2]==1) and session.calls==1
    assert set(out['timing'])=={'normalization_seconds','ORT_seconds','sampling_seconds','total_encode_seconds'}
    assert np.isclose(out['timing']['total_encode_seconds'],sum(v for k,v in out['timing'].items() if k!='total_encode_seconds'))
    metrics=enc.metrics();metrics['completed_frames']=999
    assert enc.metrics()['completed_frames']==1
    enc.check_sources();enc.close();enc.close()
    assert enc._session is None and enc.metrics()['closed'] and enc.metrics()['completed_frames']==1
    with pytest.raises(ValueError,match='closed'):enc.encode(np.zeros((464,608,3),dtype=np.uint8),centers=out['centers'])


def test_wrong_actual_session_names_fail_and_drop_session(fake_contract,monkeypatch):
    files,kwargs,session=fake_contract
    monkeypatch.setattr(session,'get_inputs',lambda:[SimpleNamespace(name='RGB',shape=list(module.INPUT_SHAPE),type='tensor(float)')])
    observed=[];original=module.BoardORTEncoder.close
    def close(self):original(self);observed.append((self._session,self._closed))
    monkeypatch.setattr(module.BoardORTEncoder,'close',close)
    with pytest.raises(ValueError,match='binding'):module.BoardORTEncoder(*files,**kwargs)
    assert observed==[(None,True)]


def test_changed_artifact_cannot_create_a_session(fake_contract,monkeypatch):
    files,kwargs,_=fake_contract;files[0].write_bytes(b'CHANGED')
    monkeypatch.setattr(module,'_create_session',lambda _:pytest.fail('Wrong bytes must not load'))
    with pytest.raises(ValueError,match='SHA differs'):module.BoardORTEncoder(*files,**kwargs)


def test_optional_input_hashes_are_pure_array_metadata_outside_encode():
    image=np.arange(5*7*3,dtype=np.uint8).reshape(5,7,3)
    normalized=module.normalize_bgr(image);report=module.frame_input_metadata(image)
    assert report['cropped_BGR_uint8']['tensor_sha256']==hashlib.sha256(image.tobytes()).hexdigest()
    assert report['normalized_BGR_FP32']['tensor_sha256']==hashlib.sha256(normalized.tobytes()).hexdigest()
    assert report['normalized_BGR_FP32']['shape']==[1,3,5,7]
