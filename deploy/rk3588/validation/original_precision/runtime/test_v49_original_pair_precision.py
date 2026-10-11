"""Tiny generated data/fake SDK only; no actual model, ORT or SDK library."""
from __future__ import annotations

import copy
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def read_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = read_source(HERE / 'v49_quantized_pair_runtime_contracts.py', '_v49_tiny_contracts')
D = read_source(HERE / 'run-v49-original-pair-precision-board.py', '_v49_tiny_driver')
# The same test source works in private research and in original_precision/runtime.
PARENT_CANDIDATES = (
    HERE.parent / 'public-snapshot/deploy/rk3588/validation/run-v12-rknn-block-board.py',
    HERE.parents[1] / 'run-v12-rknn-block-board.py',
)
parent_path = next((p for p in PARENT_CANDIDATES if p.is_file()), None)
assert parent_path is not None and C.sha(parent_path) == C.PARENT_RUNTIME_SHA
R = read_source(parent_path, '_v49_tiny_typed_ABI')


def descriptor(direction='input', dtype=2, scale=0.03125, zp=-7, qnt=2):
    return {'index': 0, 'name': direction,
            'shape': list(C.INPUT_SHAPE if direction == 'input' else C.OUTPUT_SHAPE),
            'n_elems': C.ELEMENTS, 'size': C.ELEMENTS * ({0:4,1:2}.get(dtype,1)),
            'format': 3, 'type': dtype, 'quantization_type': qnt, 'zero_point': zp,
            'pass_through': 0, 'scale': scale, 'fractional_length': 0,
            'w_stride': 0, 'h_stride': 0, 'size_with_stride': C.ELEMENTS * ({0:4,1:2}.get(dtype,1))}


def fake_R():
    fields = {name: getattr(R, name) for name in ('Input', 'Output', 'RunExtend', 'OutputExtend', 'CONTEXT',
              'SDKVersion', 'TensorAttr', 'InputOutputNum', 'MemSize', 'check_call', 'query', 'attr_record')}
    fields['require_board'] = lambda: 'tiny_fake_RK3588_guard'
    return types.SimpleNamespace(**fields)


class FakeSDK:
    def __init__(self, *, bytes_override=None, get_code=0, frame_mismatch=False,
                 release_code=0, null_output=False, input_code=0, run_code=0):
        self.buffer = (ct.c_float * 24)(*[float(i) / 16 for i in range(24)])
        self.bytes_override = bytes_override
        self.get_code, self.frame_mismatch, self.release_code = get_code, frame_mismatch, release_code
        self.null_output, self.input_code, self.run_code = null_output, input_code, run_code
        self.events = []
        self.input_rows = []
        self.release_count = 0

    def rknn_inputs_set(self, context, count, inputs):
        self.events.append('inputs_set')
        assert count == 1
        row = inputs[0]
        self.input_rows.append({'pointer': int(row.buf), 'bytes': int(row.size),
                                'type': int(row.type), 'pass_through': int(row.pass_through), 'fmt': int(row.fmt)})
        return self.input_code

    def rknn_run(self, context, extend):
        self.events.append('run')
        assert extend._obj.non_block == 0 and extend._obj.fence_fd == -1
        extend._obj.frame_id = 71
        return self.run_code

    def rknn_outputs_get(self, context, count, outputs, extend):
        self.events.append('outputs_get')
        assert count == 1 and outputs[0].want_float == 1 and outputs[0].is_prealloc == 0
        outputs[0].buf = None if self.null_output else ct.cast(self.buffer, ct.c_void_p).value
        outputs[0].size = ct.sizeof(self.buffer) if self.bytes_override is None else self.bytes_override
        extend._obj.frame_id = 72 if self.frame_mismatch else 71
        return self.get_code

    def rknn_outputs_release(self, context, count, outputs):
        self.events.append('release')
        self.release_count += 1
        for i in range(24):
            self.buffer[i] = float('nan')  # Any unowned view would now be corrupted.
        return self.release_code


def execute(api, cleanup=None):
    inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    inputs.setflags(write=False)
    ledger, errors = [], [] if cleanup is None else cleanup
    result = D.execute_sdk_call(fake_R(), api, R.CONTEXT(1), inputs, 3, (1, 6, 4),
                                30000, errors, ledger, D.ROLES[1])
    return result, inputs, ledger, errors


@pytest.mark.parametrize('direction', ['input', 'output'])
@pytest.mark.parametrize('dtype,zp', [(2,-128), (2,127), (3,0), (3,255)])
def test_affine_integer_descriptors_use_SDK_float_conversion(direction, dtype, zp):
    policy = C.validate_sdk_attr(descriptor(direction, dtype, 0.0131, zp), direction)
    assert policy['logical_element_bytes'] == 1
    assert policy['API_input_pass_through'] == 0 and policy['API_output_want_float'] == 1
    assert policy['API_FP32_bytes'] == 3579904 and not policy['manual_quantization_or_dequantization']


@pytest.mark.parametrize('dtype,qnt,scale,zp', [(0,0,0,0), (1,0,0,0), (1,2,1,0)])
def test_known_floating_descriptors_are_explicit(dtype, qnt, scale, zp):
    assert C.validate_sdk_attr(descriptor(dtype=dtype, qnt=qnt, scale=scale, zp=zp), 'input')['descriptor_kind'] == 'floating'


@pytest.mark.parametrize('change', [
    {'scale':0}, {'scale':-1}, {'scale':float('nan')}, {'scale':float('inf')}, {'scale':True},
    {'zero_point':128}, {'zero_point':-129}, {'quantization_type':0}, {'quantization_type':1},
    {'format':2}, {'format':1}, {'type':4}, {'type':7}, {'pass_through':1},
    {'shape':[1,874,1024]}, {'name':'other'}, {'index':1}, {'size':3579904}, {'n_elems':True},
])
def test_unknown_integer_metadata_rejects_before_any_API_call(change):
    row = descriptor()
    row.update(change)
    with pytest.raises(ValueError):
        C.validate_sdk_attr(row, 'input')


@pytest.mark.parametrize('scale,zp,qnt', [(0.5,0,2), (1,1,2), (1,0,1)])
def test_nonidentity_floating_affine_is_not_silently_accepted(scale, zp, qnt):
    with pytest.raises(ValueError):
        C.validate_sdk_attr(descriptor(dtype=1, scale=scale, zp=zp, qnt=qnt), 'input')


def test_FP32_request_and_owned_copy_survive_release():
    api = FakeSDK()
    (owned,row), inputs, ledger, errors = execute(api)
    assert api.events == ['inputs_set','run','outputs_get','release'] and api.release_count == 1
    assert api.input_rows == [{'pointer': inputs.ctypes.data, 'bytes':96, 'type':0, 'pass_through':0, 'fmt':3}]
    assert np.array_equal(owned.reshape(-1), np.arange(24, dtype=np.float32)/16)
    assert np.isnan(np.ctypeslib.as_array(api.buffer)).all() and np.isfinite(owned).all()
    assert row['output_owned_copy_before_release'] and row['output_buffer_released'] and not errors
    assert row['SDK_output_pointer'] != row['owned_output_pointer'] and row['runtime_frame_id'] == row['output_frame_id']
    assert row['status'] == 'completed' and len(ledger) == 1
    phases = [row[k] for k in ('input_set_seconds','run_api_seconds','outputs_get_seconds',
                              'output_validation_and_copy_seconds','pre_release_bookkeeping_seconds','outputs_release_seconds')]
    assert all(t >= 0 for t in phases) and abs(sum(phases) - row['io_total_seconds']) < 1e-12


@pytest.mark.parametrize('options,release_count', [
    ({'bytes_override':24},1), ({'bytes_override':0},1), ({'frame_mismatch':True},1),
    ({'get_code':-4},1), ({'null_output':True},1), ({'get_code':-4,'null_output':True},0),
    ({'input_code':-2},0), ({'run_code':-3},0),
])
def test_wrong_bytes_frame_and_partial_get_release_without_unsafe_copy(options, release_count):
    api = FakeSDK(**options)
    ledger, errors = [], []
    with pytest.raises((ValueError, RuntimeError)):
        D.execute_sdk_call(fake_R(), api, R.CONTEXT(1), np.ones((2,3,4),np.float32), 3,
                           (1,6,4), 100, errors, ledger, D.ROLES[1])
    assert api.release_count == release_count and ledger[0]['status'] == 'failed'
    assert 'owned_output_pointer' not in ledger[0] and not errors


def test_release_failure_is_not_completed_or_hidden():
    api, ledger, errors = FakeSDK(release_code=-6), [], []
    with pytest.raises(RuntimeError):
        D.execute_sdk_call(fake_R(), api, R.CONTEXT(1), np.ones((2,3,4),np.float32), 3,
                           (1,6,4), 100, errors, ledger, D.ROLES[1])
    assert api.release_count == 1 and ledger[0]['output_buffer_released'] is False
    assert ledger[0]['status'] == 'failed' and len(errors) == 1


class FakeContextSDK:
    def __init__(self,init_code=0,destroy_code=0):
        self.init_code,self.destroy_code=init_code,destroy_code
        self.destroyed=[]
        self.init_flags=[]

    def rknn_init(self,context,filename,size,flags,extended):
        context._obj.value=91
        self.init_flags.append(flags)
        return self.init_code

    def rknn_set_core_mask(self,context,mask):
        assert mask==7
        return 0

    def rknn_query(self,context,command,value,size):
        obj=value._obj
        if command==5: obj.api_version=b'2.3.2 (fake)'; obj.drv_version=b'0.7.2'
        elif command==0: obj.n_input=1; obj.n_output=1
        elif command in (1,2):
            row=descriptor('input' if command==1 else 'output')
            obj.index=0; obj.name=row['name'].encode(); obj.n_dims=3
            for i,v in enumerate(row['shape']): obj.dims[i]=v
            obj.n_elems=row['n_elems']; obj.size=row['size']; obj.fmt=3; obj.type=2
            obj.qnt_type=2; obj.zp=row['zero_point']; obj.scale=row['scale']; obj.pass_through=0
        elif command==6: obj.total_weight_size=1024
        else: raise AssertionError('Unexpected fake query')
        return 0

    def rknn_destroy(self,context):
        self.destroyed.append(context.value)
        return self.destroy_code


def test_failed_partial_init_is_registered_and_destroyed():
    api,contexts,records,errors=FakeContextSDK(init_code=-5),{},{},[]
    with pytest.raises(RuntimeError):
        D.prepare_context(fake_R(),C,api,D.ROLES[1],Path('/fake/pair.rknn'),contexts,records,False)
    assert contexts[D.ROLES[1]].value==91 and records[D.ROLES[1]]['init_code']==-5
    D.destroy_contexts(fake_R(),api,contexts,records,errors)
    assert api.destroyed==[91] and records[D.ROLES[1]]['context_destroyed'] is True and not errors


def test_destroy_failure_does_not_skip_the_other_context():
    api,errors=FakeContextSDK(destroy_code=-7),[]
    contexts={D.ROLES[0]:R.CONTEXT(21),D.ROLES[1]:R.CONTEXT(22)}
    records={role:{} for role in D.ROLES}
    D.destroy_contexts(fake_R(),api,contexts,records,errors)
    assert api.destroyed==[22,21] and len(errors)==2
    assert all(row['context_destroyed'] is False for row in records.values())


def test_profile_context_flags_and_actual_nonidentity_integer_query_are_recorded():
    api,contexts,records,errors=FakeContextSDK(),{},{},[]
    D.prepare_context(fake_R(),C,api,D.ROLES[1],Path('/fake/pair.rknn'),contexts,records,True)
    row=records[D.ROLES[1]]
    assert api.init_flags==[8] and row['tensor_attributes_validated']
    assert row['integer_logical_IO_observed'] and row['input_attr']['scale']==0.03125
    assert not row['all_internal_operators_INT8_verified'] and not row['all_internal_operators_on_NPU_verified']
    D.destroy_contexts(fake_R(),api,contexts,records,errors)
    assert api.destroyed==[91] and not errors


def test_host_guard_precedes_even_low_level_fake_submission():
    runtime = fake_R()
    runtime.require_board = lambda: (_ for _ in ()).throw(ValueError('host'))
    api = FakeSDK()
    with pytest.raises(ValueError, match='host'):
        D.execute_sdk_call(runtime, api, R.CONTEXT(1), np.ones((2,3,4),np.float32), 3,
                           (1,6,4), 100, [], [], D.ROLES[1])
    assert api.events == []


def test_original_FP32_gate_is_fixed_and_failure_retained():
    reference = np.array([0.0,1.0,-2.0],np.float32)
    actual = reference + np.float32(0.003)
    comparison = C.compare(actual,reference)
    assert comparison['rtol'] == 1e-4 and comparison['atol'] == 1e-5
    assert not comparison['strict_FP32_gate_passed'] and comparison['outside_tolerance_elements'] == 3
    assert C.compare(reference.copy(),reference)['strict_FP32_gate_passed']
    assert D.implementation_difference(actual,reference)['reference_for_original_FP32_gate'] is False


def mock_compile():
    compiler_sha,capture_sha,model_sha = 'a'*64,'b'*64,'c'*64
    samples=[]
    for i in range(10):
        split='calibration' if i < 8 else 'evaluation'
        sample_id=f'{split}_{i if i<8 else i-8:02}'
        item={'file':sample_id+'/input.npy','sha256':f'{i+1:064x}',
              'raw_sha256':f'{i+21:064x}','shape':list(C.INPUT_SHAPE),'dtype':'float32',
              'tensor_bytes':C.FP32_IO_BYTES,'bytes':C.FP32_IO_BYTES+128}
        samples.append({'sample_id':sample_id,'split':split,'artifacts':{'input.npy':item}})
    source_hashes={f'/fake/source{i}':digest for i,digest in enumerate((compiler_sha,capture_sha,C.PAIR_EXPORT_SHA,C.PAIR_GRAPH_SHA,C.PAIR_ORT_SHA))}
    board={'machine':'aarch64','device_tree_compatible':'rockchip,rk3588s,','python':'3.11.9 (tiny fixture)'}
    row={'schema':'v49_original_VGGT_pair_board_PTQ_compile','status':'compiled_not_runtime_validated',
         'script_sha256':compiler_sha,'contracts_sha256':'d'*64,'capture_contracts_sha256':'e'*64,
         'capture_script_sha256':'f'*64,'pair_contracts_sha256':C.PAIR_CONTRACTS_SHA,
         'pair_export_sha256':C.PAIR_EXPORT_SHA,'pair_graph_sha256':C.PAIR_GRAPH_SHA,
         'passed_board_ORT_receipt_sha256':C.PAIR_ORT_SHA,'capture_manifest_sha256':capture_sha,
         'calibration_executed_on_board_only':True,'calibration_executed':True,
         'model_inference_on_host':False,'candidate_accepted':False,'released':True,'release_code':None,
         'runtime_inference_API_called':False,'production_python_excluded':True,
         'calibration_build_api_invoked':True,'calibration_may_have_executed':True,
         'source_and_inputs_unchanged':True,'cwd_restored':True,'environment_restored':True,
         'configuration':copy.deepcopy(C.PTQ_CONFIGURATION),'input_transform':copy.deepcopy(C.PTQ_INPUT_TRANSFORM),
         'build_options':copy.deepcopy(C.PTQ_BUILD_OPTIONS),'cleanup_errors':[], 'board':board,**board,
         'versions':{'rknn-toolkit2':'2.3.2','torch':'2.2.0'},
         'output':{'file':'pair-int8.rknn','size':1024,'sha256':model_sha},
         'operations':[{'name':name,'code':0,'status':'completed','seconds':0.01}
                       for name in ('config','load_onnx','build','export_rknn')],
         'dataset':{'file':'dataset.txt','sha256':'1'*64,'bytes':1040,'input_count':1,'line_count':8,
                    'rows':[{'sample_id':s['sample_id'],'path':str(C.BASE/'experiments/tiny-capture'/s['artifacts']['input.npy']['file']),
                             'input_sha256':s['artifacts']['input.npy']['sha256'],
                             'input_raw_sha256':s['artifacts']['input.npy']['raw_sha256'],'input_shape':list(C.INPUT_SHAPE)} for s in samples[:8]],
                    'heldout_samples_excluded':[s['sample_id'] for s in samples[8:]]},
         'heldout_sample_ids':[s['sample_id'] for s in samples[8:]],'capture_samples':samples,
         'source_and_input_hashes':source_hashes,'source_and_input_hashes_after':source_hashes.copy(),
         'original_checkpoint':{'sha256':C.CHECKPOINT_SHA,'bytes':5026874952}}
    captured={'samples':samples,'script_sha256':'f'*64,'checkpoint':{'sha256':C.CHECKPOINT_SHA,'bytes':5026874952}}
    return row,captured,(compiler_sha,capture_sha,model_sha)


def test_new_completed_board_quant_receipt_binds_exact_heldout():
    row,capture,identity=mock_compile()
    C.check_quantized_compile(row,*identity,'d'*64,'e'*64)
    evals=C.bind_compilation_to_capture(row,capture,capture['samples'])
    assert [s['sample_id'] for s in evals] == ['evaluation_00','evaluation_01']


@pytest.mark.parametrize('change', [
    {'status':'failed'}, {'schema':'v47_original_VGGT_pair_RKNN_compilation'}, {'candidate_accepted':True},
    {'calibration_executed':False}, {'calibration_executed':1}, {'model_inference_on_host':True},
    {'released':False}, {'release_code':2}, {'release_code':False}, {'cleanup_errors':['release failed']},
    {'source_and_inputs_unchanged':False}, {'cwd_restored':False}, {'environment_restored':False},
    {'build_options':{'do_quantization':True,'auto_hybrid':True}},
    {'input_transform':{'mean_values':[0], 'std_values':None,'quant_img_RGB2BGR':False,'transpose':False,'extra_batch_dimension':False}},
    {'versions':{'rknn-toolkit2':'2.3.2','torch':'2.3.1'}}, {'source_and_input_hashes_after':{}},
])
def test_incomplete_or_changed_quant_receipt_cannot_authorize_runtime(change):
    row,_,identity=mock_compile()
    row.update(change)
    with pytest.raises(ValueError):
        C.check_quantized_compile(row,*identity)


@pytest.mark.parametrize('field,value', [('status','failed'),('code',1),('code',False),('seconds',-1),('seconds',float('nan'))])
def test_four_compiler_operations_need_actual_completed_status_and_success_code(field,value):
    row,_,identity=mock_compile()
    row['operations'][2][field]=value
    with pytest.raises(ValueError):
        C.check_quantized_compile(row,*identity)


@pytest.mark.parametrize('change', ['heldout_leak','row_raw_sha','row_path','sample_copy','wrong_split','wrong_checkpoint'])
def test_calibration_and_evaluation_identity_cannot_drift(change):
    row,captured,_=mock_compile()
    if change=='heldout_leak': row['heldout_sample_ids'][0]='calibration_00'
    elif change=='row_raw_sha': row['dataset']['rows'][0]['input_raw_sha256']='9'*64
    elif change=='row_path': row['dataset']['rows'][0]['path']='/tmp/other.npy'
    elif change=='sample_copy': row['capture_samples']=copy.deepcopy(captured['samples'][::-1])
    elif change=='wrong_split': captured['samples']=copy.deepcopy(captured['samples']); captured['samples'][0]['split']='evaluation'
    else: row['original_checkpoint']['sha256']='9'*64
    with pytest.raises(ValueError):
        C.bind_compilation_to_capture(row,captured,captured['samples'])


def fake_calls():
    calls=[]
    counts={role:0 for role in D.ROLES}
    for sample_index in (0,1):
        for round_index,order in enumerate(D.schedule_for_sample(sample_index)):
            for role in order:
                calls.append({'sample_id':f'evaluation_{sample_index:02}','role':role,
                              'sample_call_index':round_index,'context_call_index':counts[role],
                              'first_in_context':counts[role]==0,'sdk_call_index':len(calls),
                              'byte_equal_to_first_for_sample':True,'io_total_seconds':10 if round_index==0 else (0.4 if role==D.ROLES[0] else 0.3),
                              'original_FP32_comparison':{'strict_FP32_gate_passed':role==D.ROLES[0]}})
                counts[role]+=1
    return calls


def test_schedule_denominator_and_true_cold_context_identity():
    calls=fake_calls()
    assert len(calls)==16 and sum(r['first_in_context'] for r in calls)==2
    result=D.summarize(calls,['evaluation_00','evaluation_01'],True)
    assert all(v['mean_denominator']==6 for v in result['implementations'].values())
    assert abs(result['PTQ_IO_difference']['nonquant_minus_PTQ_percent_of_nonquant']-25)<1e-10
    assert result['samples']['evaluation_00'][D.ROLES[0]]['first_for_sample_is_first_in_context']
    assert not result['samples']['evaluation_01'][D.ROLES[0]]['first_for_sample_is_first_in_context']
    assert not result['implementations'][D.ROLES[1]]['strict_original_FP32_gate_passed']
    profile=D.summarize(calls,['evaluation_00','evaluation_01'],False)
    assert 'PTQ_IO_difference' not in profile and profile['profile_IO_is_not_a_normal_speed_comparison']


def test_missing_or_duplicate_measured_calls_cannot_produce_summary():
    calls=fake_calls()
    with pytest.raises(ValueError):
        D.summarize(calls[:-1],['evaluation_00','evaluation_01'],True)
    calls[-1]['sample_call_index']=2
    with pytest.raises(ValueError):
        D.summarize(calls,['evaluation_00','evaluation_01'],True)


def test_frozen_nonquant_compile_receipt_is_comparator_only():
    # Reading a saved JSON is not actual inference. Public copies omit the raw receipt.
    path=HERE.parent.parent/'edge-vggt-v48/compile-r1/compile.json'
    if path.is_file():
        assert C.sha(path)==C.NONQUANT_COMPILE_SHA
        C.check_nonquantized_compile(json.loads(path.read_text()))
    else:
        assert C.NONQUANT_COMPILE_SHA=='5084e00143c4ab5fce94ceae81247c11939ddb7783d80d1b937bffcbdc0bcfb9'


def test_pure_sibling_capture_and_quant_interfaces_have_same_shapes_and_lineage():
    capture=read_source(HERE.parent/'capture/v49_capture_contracts.py','_v49_tiny_capture_contracts')
    quant=read_source(HERE.parent/'quant/v49_quant_contracts.py','_v49_tiny_quant_contracts')
    assert tuple(capture.INPUT_SHAPE)==C.INPUT_SHAPE==quant.INPUT_SHAPE
    assert tuple(capture.OUTPUT_SHAPE)==C.OUTPUT_SHAPE==quant.OUTPUT_SHAPE
    assert capture.PAIR_ONNX_SHA==C.PAIR_GRAPH_SHA==quant.PAIR_GRAPH_SHA
    assert C.PTQ_CONFIGURATION==quant.CONFIGURATION and capture.CHECKPOINT_SHA==C.CHECKPOINT_SHA
    assert callable(capture.load_capture) and callable(capture.load_array)
