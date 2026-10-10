"""No model or real images: fixed-order, reference and tensor-shape guards."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'deploy/rk3588/validation/run-v22-come-fullmodel-board.py'
spec = importlib.util.spec_from_file_location('_v22_test_fullmodel_driver',PATH)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


def reference_fixture():
    row = dict(kind='fixed_rectified_original_VGGT_resolution_pairs',
        status='completed_fixed_input_diagnostic',image_size=336,precision='float32',
        device='cpu',original_weights=True,ORB_used=False,npu_used=False,
        checkpoint=dict(sha256=driver.CHECKPOINT_SHA,bytes=driver.CHECKPOINT_BYTES),pairs=[],selected_inputs=[])
    for index,ids in enumerate(([1.,2.],[2.,3.],[3.,4.])):
        row['pairs'].append(dict(revision=index+1,frame_ids=ids,
            input_tensor=dict(shape=[2,3,252,336],dtype='float32'),
            arrays={key:dict(shape=list(shape),dtype='float32',finite=True)
                for key,shape in driver.PREDICTION_SHAPES.items()}))
        row['selected_inputs'].append(dict(revision=index+1,frame_ids=ids,
            images=[dict(frame_id=fid,timestamp=fid,samefile_validated=True,
                schedule_token=dict(frame_id=fid,timestamp=fid)) for fid in ids]))
    return row


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(image_size=112),lambda r:r.update(precision='float16'),
    lambda r:r.update(ORB_used=True),lambda r:r.update(npu_used=True),
    lambda r:r['checkpoint'].update(sha256='wrong'),
    lambda r:r['pairs'].pop(),
    lambda r:r['selected_inputs'][1]['images'][1].update(timestamp=3.1),
    lambda r:r['selected_inputs'][0]['images'][0].update(samefile_validated=False),
    lambda r:r['pairs'][0]['arrays']['depth'].update(shape=[2,252,336,1]),
    lambda r:r['pairs'][0]['input_tensor'].update(shape=[2,3,336,336]),
])
def test_reference_negative_mutations_start_from_passing_fixture(mutate):
    row = reference_fixture()
    driver.validate_reference(row)
    mutate(row)
    with pytest.raises(ValueError):
        driver.validate_reference(row)


def test_overlapping_frame_ids_cannot_be_relabelled():
    row = reference_fixture()
    ids = [2.5,3.]
    row['pairs'][1]['frame_ids'] = ids
    row['selected_inputs'][1]['frame_ids'] = ids
    row['selected_inputs'][1]['images'][0].update(frame_id=2.5,timestamp=2.5,
        schedule_token=dict(frame_id=2.5,timestamp=2.5))
    with pytest.raises(ValueError,match='overlap'):
        driver.validate_reference(row)


@pytest.mark.parametrize('mutation',['dtype','shape','nan','missing'])
def test_all_five_canonical_arrays_preserve_B1_dtype_and_finite(mutation):
    arrays = {key:np.zeros(shape,np.float32) for key,shape in driver.PREDICTION_SHAPES.items()}
    driver.validate_predictions(arrays)
    if mutation=='dtype':
        arrays['pose_enc'] = arrays['pose_enc'].astype(np.float64)
    elif mutation=='shape':
        arrays['intrinsic'] = arrays['intrinsic'][0]
    elif mutation=='nan':
        arrays['depth_conf'].flat[0] = np.nan
    else:
        del arrays['extrinsic']
    with pytest.raises(ValueError):
        driver.validate_predictions(arrays)


def test_signed_zero_reference_is_not_byte_equal():
    assert np.array_equal(np.array([0.],np.float32),np.array([-0.],np.float32))
    assert not driver.bitwise_equal(np.array([0.],np.float32),np.array([-0.],np.float32))


def test_ABBA_AB_BA_is_fixed_and_timed_forward_includes_inner_work():
    assert driver.CALL_ORDER==((0,'original'),(0,'merged'),(0,'merged'),(0,'original'),
        (1,'original'),(1,'merged'),(2,'merged'),(2,'original'))
    assert sum(method=='merged' for _,method in driver.CALL_ORDER)==4
    now = [0.]
    def forward():
        now[0] += 10.  # whole network
        now[0] += 1.   # predictor and plan included
        return {'raw':'synthetic'}
    value,seconds = driver.timed_forward(forward,lambda:now[0])
    assert value=={'raw':'synthetic'} and seconds==11.


def test_forward_and_hook_lifecycle_check_detects_leftover_patch():
    def module():
        return SimpleNamespace(_forward_pre_hooks={},_forward_hooks={})
    dino = [SimpleNamespace(mlp=module()) for _ in range(24)]
    dino[14]._forward_pre_hooks = {}
    dino[14]._forward_hooks = {}
    aggregator = module()
    aggregator.patch_embed = SimpleNamespace(blocks=dino)
    aggregator.frame_blocks = [SimpleNamespace(mlp=module()) for _ in range(24)]
    aggregator.global_blocks = [SimpleNamespace(mlp=module()) for _ in range(24)]
    model = SimpleNamespace(aggregator=aggregator)
    original = driver.model_forward_state(model)
    dino[15].mlp.forward = lambda:None
    assert driver.model_forward_state(model)!=original
    del dino[15].mlp.forward
    assert driver.model_forward_state(model)==original
    dino[14]._forward_hooks[99] = object()
    assert driver.model_forward_state(model)!=original


def test_wrong_module_SHA_is_rejected_before_import(tmp_path):
    path = tmp_path/'module.py'
    path.write_text('raise AssertionError("must never import")\n')
    with pytest.raises(ValueError,match='SHA mismatch'):
        driver.load_pinned_module(path,'0'*64,'_should_not_load')


def test_host_and_output_guard_prevents_real_inference(tmp_path):
    with pytest.raises(ValueError,match='board-only'):
        driver.board_guard('x86_64',tmp_path/'new')
    with pytest.raises(ValueError,match='BASE/experiments'):
        driver.board_guard('aarch64',tmp_path/'new')
