"""Generated pixels and map arithmetic only; no real image/model/board work."""
import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
single=load(HERE/'single_remap_cache.py','_v40_single_test')
parent,_=single.load_parent()
K=np.array([[517.3,0,318.6],[0,516.5,255.3],[0,0,1]],np.float64)
D=np.array([.2624,-.9531,-.0054,.0026,1.1633],np.float64)


@pytest.mark.parametrize('pattern',('random','gradient','border','zero'))
def test_concatenated_maps_and_single_call_equal_original_synthetic_pixels(pattern,monkeypatch):
    stripe=parent.FixedStripeUndistorter(K,D,expected_opencv_version=cv2.__version__)
    cached=single.SingleRemapUndistorter(K,D,expected_opencv_version=cv2.__version__)
    assert cached._map1.tobytes()==b''.join(s[2].tobytes() for s in stripe._stripes)
    assert cached._map2.tobytes()==b''.join(s[3].tobytes() for s in stripe._stripes)
    y,x=np.indices((480,640))
    if pattern=='random':image=np.random.default_rng(14).integers(0,256,(480,640,3),dtype=np.uint8)
    elif pattern=='gradient':image=np.stack((x%256,y%256,(7*x+3*y)%256),-1).astype(np.uint8)
    elif pattern=='border':image=np.repeat((((x<8)|(x>=632)|(y<8)|(y>=472))*255)[...,None],3,-1).astype(np.uint8)
    else:image=np.zeros((480,640,3),np.uint8)
    before=image.tobytes();reference=cv2.undistort(image,K,D);old=stripe.undistort_bgr(image)
    calls=[];remap=cv2.remap
    def counted(*args,**kwargs):calls.append(args[1].shape);return remap(*args,**kwargs)
    monkeypatch.setattr(cv2,'remap',counted)
    result=cached.undistort_bgr(image)
    assert calls==[(480,640,2)] and result.tobytes()==reference.tobytes()==old.tobytes()
    assert image.tobytes()==before and result.flags.owndata and not np.shares_memory(result,image)
    assert cached.metadata()['map_bytes']==1843200 and cached.metadata()['construction_peak_map_array_bytes']==3686400
    assert not cached._map1.flags.writeable and not cached._map2.flags.writeable
    stripe.close();cached.close();cached.close()
    assert cached.metadata()['current_map_bytes']==0


def test_partial_last_stripe_zero_D_repeat_ownership_and_close():
    k=np.array([[260,0,154.2],[0,260,13.5],[0,0,1]],np.float64);d=np.zeros(5,np.float64)
    handle=single.SingleRemapUndistorter(k,d,(31,321),expected_opencv_version=cv2.__version__)
    image=np.random.default_rng(2).integers(0,256,(31,321,3),dtype=np.uint8)
    first,second=handle.undistort_bgr(image),handle.undistort_bgr(image)
    assert first.tobytes()==second.tobytes()==cv2.undistort(image,k,d).tobytes()
    assert not np.shares_memory(first,second) and handle.metadata()['stripe_count']==3
    handle.close()
    with pytest.raises(RuntimeError,match='closed'):handle.undistort_bgr(image)


def test_parent_byte_mutation_and_budget_rejected(tmp_path):
    fake=tmp_path/'changed.py';fake.write_bytes(b'not_the_frozen_parent')
    with pytest.raises(ValueError,match='SHA'):single.load_parent(fake)
    with pytest.raises(ValueError,match='budget'):
        single.SingleRemapUndistorter(K,D,expected_opencv_version=cv2.__version__,max_map_bytes=1843199)


def test_no_new_fullgrid_map_computation(monkeypatch):
    sizes=[];build=cv2.initUndistortRectifyMap
    def counted(*args,**kwargs):sizes.append(args[4]);return build(*args,**kwargs)
    monkeypatch.setattr(cv2,'initUndistortRectifyMap',counted)
    handle=single.SingleRemapUndistorter(K,D,expected_opencv_version=cv2.__version__)
    assert sizes==[(640,6)]*80
    handle.close()
