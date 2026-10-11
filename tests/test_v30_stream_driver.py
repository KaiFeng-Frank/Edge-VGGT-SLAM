"""Tiny fake-native remapping tests; never load a model or native library."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
ENTRY=HERE/'run-v30-dpvo-stream-board.py'
if not ENTRY.is_file():ENTRY=HERE.parent/'deploy/rk3588/validation/dpvo_stream/run-v30-dpvo-stream-board.py'
spec=importlib.util.spec_from_file_location('v30_driver',ENTRY);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class FakeCache:
    made=[]
    def __init__(self,library,gmap,fmap,**kw):
        self.g=gmap.copy();self.f=fmap.copy();self.kw=kw;self.closed=False;self.calls=0
        self.preparation={'feature_inputs_copied':True,'context_bytes':self.g.nbytes+self.f.nbytes,'native':{'neon':1}}
        self.made.append(self)
    def correlate(self,coords,kk,jj,*,cache_key):
        assert cache_key==self.kw['cache_key'] and not self.closed;self.calls+=1
        v=self.g[0,kk,0,1,1]*self.f[0,jj,0,0,0]
        return np.broadcast_to(v[None,:,None],(1,len(kk),882)).copy(),{'total_seconds':0.}
    def close(self):self.closed=True
    def metrics(self):return {'closed':self.closed,'current_context_bytes':0 if self.closed else self.g.nbytes+self.f.nbytes,'completed_calls':self.calls}
def frames(M=2):
    out=[]
    for i in range(5):
        f=np.full((128,4,4),10+i,np.float32);g=np.empty((M,128,3,3),np.float32)
        for j in range(M):g[j].fill(1+M*i+j)
        f.flags.writeable=False;g.flags.writeable=False;out.append({'frame_id':str(i),'fmap':f,'gmap':g})
    return out
def query(c,records,ii,jj,kk):
    return c(np.zeros((1,len(ii),2,3,3),np.float32),ii=np.array(ii,np.int64),jj=np.array(jj,np.int64),kk=np.array(kk,np.int64),frames=records)
def test_noncontiguous_sources_targets_duplicates_and_cache_reuse():
    FakeCache.made=[];c=d.CompactCorrelation(FakeCache,'unused',2);f=frames()
    ii=[4,1,4,1];jj=[3,0,0,3];kk=[9,2,8,3]
    first=query(c,f,ii,jj,kk);expected=np.array([f[i]['gmap'][k%2,0,1,1]*f[j]['fmap'][0,0,0] for i,j,k in zip(ii,jj,kk)])
    np.testing.assert_array_equal(first[0,:,0],expected);assert FakeCache.made[0].g.shape[1]==4 and FakeCache.made[0].f.shape[1]==2
    second=query(c,f,ii[::-1],jj[::-1],kk[::-1]);assert len(FakeCache.made)==1
    np.testing.assert_array_equal(second[0,:,0],expected[::-1]);np.testing.assert_array_equal(first[0,:,0],expected)
    c.close();assert c.metrics()['completed_calls']==2 and not c.metrics()['current_context_open'] and FakeCache.made[0].closed

def test_delete_remap_preserves_feature_generation_identity_and_values():
    FakeCache.made=[];c=d.CompactCorrelation(FakeCache,'unused',2);f=frames()
    before=query(c,f,[4,1],[3,0],[9,2]);retained=[f[0],f[1],f[3],f[4]]
    after=query(c,retained,[3,1],[2,0],[7,2]);np.testing.assert_array_equal(after,before)
    assert len(FakeCache.made)==1 # exactly the same ordered feature generations
    query(c,retained,[3],[1],[7]);assert len(FakeCache.made)==2 and FakeCache.made[0].closed
    c.close()
@pytest.mark.parametrize('fault',['source_owner','target_range','mutable'])
def test_invalid_bindings_reject_before_native(fault):
    FakeCache.made=[];c=d.CompactCorrelation(FakeCache,'unused',2);f=frames();ii,jj,kk=[1],[0],[2]
    if fault=='source_owner':ii=[2]
    if fault=='target_range':jj=[5]
    if fault=='mutable':f[1]['gmap'].flags.writeable=True
    with pytest.raises(ValueError):query(c,f,ii,jj,kk)
    assert not FakeCache.made
