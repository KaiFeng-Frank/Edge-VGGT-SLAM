"""Generated cache/lifecycle checks only; no model, Torch, SDK or board calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import types

import numpy as np

ROOT=Path(__file__).parent
SOURCE=ROOT/'v110_native518_executor.py'
SOURCE_SHA='e153a5f71ef0b10119207b2e43589b628a2e85ab1abeaeb72d2df2a7dd6fb638'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
spec=importlib.util.spec_from_file_location('_v110_generated_cache_checks',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ArrayValue:
    def __init__(self,values):self.values=np.asarray(values,np.float32)
    def numpy(self):return self.values


def owner():
    x=module.Native518Executor.__new__(module.Native518Executor)
    x.np=np;x.torch=object();x.reuse_previous_tail=True;x.core_mask=1
    x._tail_pixels=x._tail_encoded=x._tail_key=x._tail_output_sha256=None
    x._cache_stats=dict(lookups=0,hits=0,misses=0,commits=0,drops=0)
    x.initialization=dict(config_sha256=module.CONFIG_SHA,checkpoint=dict(sha256='original-checkpoint'),
        frozen_helper_source_hashes={'native_original_prefix.py':'original-preparation-source'})
    x.dino_rows=[dict(file='original-group'+str(i),sha256=str(i)*64) for i in range(4)]
    x.prefix=types.SimpleNamespace(mean=ArrayValue([.485,.456,.406]),std=ArrayValue([.229,.224,.225]),
        verify_unchanged=lambda:None)
    x.owner=threading.get_ident();x.closed=x.failed=x.active=False
    x.pins={};x._held=[];x._totals=dict.fromkeys(module.COUNTERS,0)
    x.calls_attempted=x.calls_completed=0;x.cleanup_errors=[];x.last_call=x.last_timings={}
    x.expected_npu_hz=1000000000;x.repo=Path('/synthetic-source');x.models=[]
    x.protocol=types.SimpleNamespace(validate_images=lambda value:value)
    return x


pixels=np.zeros(module.SHAPE[1:],np.float32)
encoded=np.arange(1041*1024,dtype=np.float32).reshape(1,1041,1024)
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
cases=[]
x=owner();key=x._feature_key();pixel_sha=digest(pixels);original_encoded=encoded.copy()
x._commit_previous_tail(pixels,encoded,key,pixel_sha)
pixels[0,0,0]=.5;encoded[0,0,0]=-1
assert x._tail_pixels!=pixels.tobytes() and np.array_equal(x._tail_encoded,original_encoded)
assert x._tail_encoded.flags.owndata and not x._tail_encoded.flags.writeable
restored_pixels=np.zeros_like(pixels)
hit=x._lookup_previous_tail(restored_pixels,key,pixel_sha)
assert hit.flags.owndata and hit.flags.writeable and not np.shares_memory(hit,x._tail_encoded)
hit[0,0,0]=-2
assert x._tail_encoded[0,0,0]==original_encoded[0,0,0]
cases.append(dict(name='Owned input/output copies remain stable after external mutation',passed=True))

# Deliberately present the old hash with changed bytes. Hash identity alone
# would accept this wrong frame; the production helper must still reject it.
assert x._lookup_previous_tail(pixels,key,pixel_sha) is None
cases.append(dict(name='Forced same hash with changed pixel bytes misses',passed=True))

for name,change in (
    ('core mask',lambda y:setattr(y,'core_mask',7)),
    ('model payload SHA',lambda y:y.dino_rows[0].update(sha256='f'*64)),
    ('config SHA',lambda y:y.initialization.update(config_sha256='e'*64))):
    y=owner();old=y._feature_key();y._commit_previous_tail(restored_pixels,original_encoded,old,pixel_sha)
    change(y);new=y._feature_key()
    assert old!=new and y._lookup_previous_tail(restored_pixels,new,pixel_sha) is None
    cases.append(dict(name='Changed '+name+' invalidates key',passed=True))


class CountingContext:
    def __init__(self):self.calls=[]
    def run(self,values):
        self.calls.append(values[0].copy())
        return [values[0]+np.float32(1)]


for cache_hit,expected in ((True,4),(False,8)):
    context=CountingContext();values=[np.array([7],np.float32),np.array([7],np.float32)]
    for _ in range(4):values=module.Native518Executor._advance_DINO_group(context,values,cache_hit)
    assert len(context.calls)==expected and values[1][0]==11
    assert values[0][0]==(7 if cache_hit else 11)
    cases.append(dict(name='New tail executes all four groups on '+('hit' if cache_hit else 'miss'),passed=True))

y=owner();key=y._feature_key();y._commit_previous_tail(restored_pixels,original_encoded,key,pixel_sha)
y._commit_previous_tail(pixels,original_encoded,key,digest(pixels))
assert y._cache_metrics()['cache_entries']==1 and y._lookup_previous_tail(restored_pixels,key,pixel_sha) is None
assert y._cache_metrics()['cache_bytes']==6700608
cases.append(dict(name='Only immediately preceding tail remains; bounded one entry',passed=True))
y.close()
assert y.closed and y.prefix is None and y._cache_metrics()['cache_entries']==0
cases.append(dict(name='Close releases cached input/output storage',passed=True))

y=owner();key=y._feature_key();y._commit_previous_tail(restored_pixels,original_encoded,key,pixel_sha)
def fail_before_SDK(*args,**kwargs):raise RuntimeError('Generated pre-SDK owner creation failure')
y.NativeSequentialGroups=fail_before_SDK
try:
    y.predict(np.stack([restored_pixels,restored_pixels]))
    raise AssertionError('Expected generated failure')
except RuntimeError as exc:
    assert 'Generated pre-SDK' in str(exc) and exc.native_executor is y
assert y.failed and not y.active and y._cache_metrics()['cache_entries']==0 and y._held==[]
assert y._totals==dict.fromkeys(module.COUNTERS,0)
cases.append(dict(name='Failed predict drops cache and retains cleanup ownership without SDK calls',passed=True))

assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
receipt=dict(schema='v110_original_native518_tail_cache_generated_check',status='passed_generated_cache_checks',
    source_sha256=SOURCE_SHA,test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    cases=cases,generated_data_only=True,NN_Torch_model_SDK_GPU_or_board_execution=False,
    feature_cache_does_not_establish_model_parity_or_performance=True)
path=ROOT/'cache-synthetic-check.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],cases=len(cases),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
