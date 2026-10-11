"""Generated pixels/features and fake SDK only; no actual models or board work."""
import ast
from contextlib import contextmanager
import copy
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO/'deploy/rk3588/validation'
HERE = PUBLIC/'dpvo_fast_rknn'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


H = load(HERE/'v35_rknn_encoder.py', '_v35_encoder_tests')
D = load(HERE/'run-v35-dpvo-fast-rknn-board.py', '_v35_driver_tests')
P = load(PUBLIC/'dpvo_stream/v30_frame_encoder.py', '_v35_original_patchifier_tests')


@pytest.fixture
def encoder(monkeypatch, tmp_path):
    """Bypass hardware construction explicitly; exercise real encode/error paths."""
    shapes = {'fmap': (1, 2, 4, 5), 'imap': (1, 4, 4, 5)}
    monkeypatch.setattr(H, 'OUTPUT_SHAPES', shapes)
    value = H.BoardRKNNEncoder.__new__(H.BoardRKNNEncoder)
    value._lock = threading.RLock()
    value._closed = False
    value._cleanup = []
    value._activity = {'sdk_run_api_calls': 0}
    value._context_record = {}
    value._directory = tmp_path
    source = tmp_path/'synthetic-source.py'; source.write_text('synthetic')
    value._pins = {str(source): H.sha(source)}
    value._metrics = dict(completed_frames=0, successful_sdk_frames=0,
        normalization_seconds=0., SDK_outer_seconds=0., sampling_seconds=0.,
        total_encode_seconds=0., calls=[], candidate_accepted=False)
    value._encoder = P
    class Context:
        closed = False
        invalid = None
        execute_error = None
        def execute(self, normalized, indices, activity):
            assert normalized.shape == (1,3,464,608)
            assert normalized.dtype == np.float32 and normalized.flags.c_contiguous
            assert indices is None
            activity['sdk_run_api_calls'] += 1
            if self.execute_error:
                raise self.execute_error
            actual = {k:np.arange(np.prod(s),dtype=np.float32).reshape(s)+1 for k,s in shapes.items()}
            if self.invalid == 'nonfinite':actual['fmap'][0,0,0,0] = np.nan
            if self.invalid == 'zero':actual['imap'].fill(0)
            return actual, dict(output_buffers_released=True,synchronous=True,sdk_io_seconds=.001)
        def close(self):
            self.closed = True
            value._context_record['context_destroyed'] = True
    value._context = Context()
    @contextmanager
    def native(directory):
        row = dict(truncated=False,RKNN_error_detected=False,raw_text='',raw_sha256='0'*64,bytes=0)
        yield row
        if getattr(value,'native_error',False) and not value._context.closed:
            row.update(RKNN_error_detected=True,raw_text='E RKNN synthetic normalize failure')
    value._log_helper = SimpleNamespace(_native_log=native)
    return value


def synthetic_image():return np.full((464,608,3), 17, np.uint8)
def synthetic_centers(count=48):return np.tile(np.array([[2,2]],np.int64),(count,1))


def test_diagnostic_flag_is_rejected_before_board_or_file_sideeffects(monkeypatch):
    monkeypatch.setattr(H,'board_guard',lambda:pytest.fail('guard should not execute'))
    with pytest.raises(ValueError,match='Explicit'):
        H.BoardRKNNEncoder(*['missing']*4,**{k:'missing' for k in ('compiler_source','sdk_helper',
            'parent_runtime','abi_helper','header','library','encoder_module','native_log_helper','log_directory')})


def test_no_cpu_fallback_one_call_and_stable_snapshots(encoder):
    a=encoder.encode(synthetic_image(),centers=synthetic_centers())
    b=encoder.encode(synthetic_image(),centers=synthetic_centers())
    assert a['fmap'].shape==(2,4,5) and a['gmap'].shape==(48,2,3,3)
    assert a['context'].shape==(48,4) and a['patches'].shape==(48,3,3,3)
    assert not np.shares_memory(a['fmap'],b['fmap'])
    assert encoder.metrics()['sdk_run_api_calls']==2
    assert encoder.metrics()['completed_frames']==2
    assert encoder.metrics()['candidate_accepted'] is False
    encoder.close();encoder.close()
    assert encoder.metrics()['NPU_used'] and encoder.metrics()['context']['context_destroyed']
    with pytest.raises(ValueError,match='closed'):encoder.encode(synthetic_image(),centers=synthetic_centers())


@pytest.mark.parametrize('invalid',['zero','nonfinite','native_error','api_error'])
def test_actual_invalid_outputs_or_native_log_are_fatal_and_saved(encoder, invalid):
    if invalid=='native_error':encoder.native_error=True
    elif invalid=='api_error':encoder._context.execute_error=RuntimeError('SDK synthetic failure')
    else:encoder._context.invalid=invalid
    with pytest.raises((ValueError,RuntimeError)):
        encoder.encode(synthetic_image(),centers=synthetic_centers())
    assert encoder._closed and encoder._context.closed
    report=json.loads((encoder._directory/'encoder-failure.json').read_text())
    assert report['metrics']['sdk_run_api_calls']==1
    assert report['metrics']['completed_frames']==0
    assert report['metrics']['calls'][0]['completed'] is False
    assert report['metrics']['candidate_accepted'] is False
    if invalid!='api_error':
        assert (encoder._directory/'encoder-failure-outputs.npz').exists()
        if invalid=='nonfinite':assert report['raw_output_artifact']['arrays']['fmap']['nonfinite_elements']==1


def test_sources_and_metrics_are_not_mutable_aliases(encoder):
    assert encoder.check_sources()==encoder._pins
    metrics=encoder.metrics();metrics['calls'].append('tamper')
    assert encoder.metrics()['calls']==[]
    Path(next(iter(encoder._pins))).write_text('changed')
    with pytest.raises(ValueError,match='changed'):encoder.check_sources()


def test_rng_same_x_y_then_depth_order_as_frozen_encoder():
    import torch
    a=torch.Generator(device='cpu').manual_seed(20261011)
    b=torch.Generator(device='cpu').manual_seed(20261011)
    centers=P.random_centers(152,116,48,a)
    x=torch.randint(1,151,(1,48),generator=b)
    y=torch.randint(1,115,(1,48),generator=b)
    assert np.array_equal(centers,torch.stack((x,y),-1)[0].numpy())
    depth=D.TorchDepthRNG(torch,a).random(48)
    assert np.array_equal(depth,torch.rand((48,),generator=b,dtype=torch.float32).numpy())
    assert np.array_equal(a.get_state().numpy(),b.get_state().numpy())


@pytest.mark.parametrize('bad_return',['size','frame_id',None])
def test_frozen_dense_transport_owned_copy_release_and_native_NHWC(monkeypatch,bad_return):
    base=load(PUBLIC/'run-v26-dpvo-rknn-board.py','_v35_dense_transport_test')
    abi=load(PUBLIC/'run-v12-rknn-block-board.py','_v35_abi_synthetic_test')
    monkeypatch.setattr(base,'INPUT_SHAPE',(1,3,2,4))
    monkeypatch.setattr(base,'OUTPUTS',( ('fmap',(1,2,2,4)), ('imap',(1,3,2,4)) ))
    image=np.arange(24,dtype=np.float32).reshape(1,3,2,4)
    allocations=[];events=[]
    def inputs(ctx,count,requests):
        assert count==1 and requests[0].type==0 and requests[0].fmt==1 and requests[0].pass_through==0
        packed=np.ctypeslib.as_array(ct.cast(requests[0].buf,ct.POINTER(ct.c_float)),shape=(24,))
        assert np.array_equal(packed,image.transpose(0,2,3,1).ravel())
        events.append('set');return 0
    def run(ctx,extend):extend._obj.frame_id=7;events.append('run');return 0
    def get(ctx,count,outputs,received):
        received._obj.frame_id=8 if bad_return=='frame_id' else 7
        for out,(name,shape) in zip(outputs,base.OUTPUTS):
            assert out.want_float==1 and out.is_prealloc==0
            array=np.full(shape,1+len(allocations),np.float32);allocations.append(array)
            out.buf=array.ctypes.data;out.size=array.nbytes+(4 if bad_return=='size' else 0)
        events.append('get');return 0
    def release(*args):
        events.append('release')
        for array in allocations:array.fill(-999)
        return 0
    api=SimpleNamespace(rknn_inputs_set=inputs,rknn_run=run,rknn_outputs_get=get,rknn_outputs_release=release)
    cleanup=[];activity={}
    if bad_return:
        with pytest.raises(ValueError):base.execute_call(api,abi,abi.CONTEXT(9),image,1,1000,cleanup,activity)
    else:
        values,timing=base.execute_call(api,abi,abi.CONTEXT(9),image,1,1000,cleanup,activity)
        assert np.all(values['fmap']==1) and np.all(values['imap']==2)
        assert timing['output_buffers_released'] and timing['sdk_io_seconds']>0
    assert events==['set','run','get','release'] and cleanup==[] and activity['sdk_run_api_calls']==1


def test_BA_timer_passes_identical_objects_and_preserves_failure():
    called=[];result=object();a=np.zeros(1,np.float32)
    def BA(*args,**kwargs):called.append((args,kwargs));args[0][0]=7;return result
    original=SimpleNamespace(BA=BA,reference_marker=object())
    timer=D.TimedGeometry(original)
    assert timer.BA(a,iterations=2) is result and a[0]==7
    assert called[0][0][0] is a and called[0][1]=={'iterations':2}
    assert timer.reference_marker is original.reference_marker and len(timer.calls)==1
    original.BA=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('original failure'))
    with pytest.raises(RuntimeError,match='original failure'):timer.BA(a)
    assert len(timer.calls)==2


def test_original_FAST_RNG_cache_and_Update_code_retained():
    parent=ast.parse((PUBLIC/'dpvo_fast/run-v34-dpvo-fast-board.py').read_text())
    current=ast.parse((HERE/'run-v35-dpvo-fast-rknn-board.py').read_text())
    for name in ('CompactCorrelation','TorchDepthRNG','parse_fast_config','load_fast_config'):
        a=next(n for n in parent.body if getattr(n,'name',None)==name)
        b=next(n for n in current.body if getattr(n,'name',None)==name)
        assert ast.dump(a)==ast.dump(b)
    for tree in (parent,current):
        function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='update')
        if tree is parent:expected=ast.dump(function)
        else:assert ast.dump(function)==expected
    assert 'BoardORTEncoder(' not in (HERE/'run-v35-dpvo-fast-rknn-board.py').read_text()


def test_reference_rejection_preserves_original_threshold_no_clipping():
    reference=np.ones((2,3),np.float32);actual=reference.copy();actual[0,0]+=.01
    gate=D.reference_checks({'fmap':actual},{'fmap':reference})['fmap']
    assert gate['failed_elements']==1 and not gate['gate_passed']
    assert gate['rtol']==1e-4 and gate['atol']==1e-5 and actual[0,0]>1


def test_real_lineage_compile_metadata_can_be_validated_without_inference():
    compiler=load(PUBLIC/'dpvo_encoder_opt/compile-v32-dpvo-encoder-diagnostic.py','_v35_lineage_metadata')
    # Published metadata redacts host paths; this checks the recorded ancestor
    # identity and the actual validation contract, not reconstructed raw bytes.
    evidence=json.loads((REPO/'results/dpvo_encoder_layout_v32.json').read_text())['evidence']['host_compile']
    value=evidence['data']
    assert evidence['raw_sha256']==H.COMPILE_SHA and value['output']['sha256']==H.MODEL_SHA
    compiler.validate_compilation(value,value['ancestor'],expected_script_sha256=H.COMPILER_SHA)
    changed=copy.deepcopy(value);changed['historical_FP32_gate']['passed']=True
    with pytest.raises(ValueError):compiler.validate_compilation(changed,changed['ancestor'],expected_script_sha256=H.COMPILER_SHA)
