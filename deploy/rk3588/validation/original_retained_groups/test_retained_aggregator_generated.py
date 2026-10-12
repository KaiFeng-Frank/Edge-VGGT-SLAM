"""Small generated Torch modules and fake typed SDK/NEON; no original model."""
import ctypes as ct
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from contextlib import redirect_stdout
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from vggt_slam import rknn_aggregator_blocks as original
from vggt_slam.patch_cache import PatchCacheHandle

ROOT = Path(original.__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'rknn_retained_aggregator.py'
if not RUNTIME.exists():
    RUNTIME = ROOT / 'vggt_slam/rknn_retained_aggregator.py'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

rt = load('_generated_retained_runtime', RUNTIME)
ABI = load('_generated_retained_ABI', ROOT / 'deploy/rk3588/validation/run-v12-rknn-block-board.py')
viz_attention = False

def slice_expand_and_flatten(token, B, S):
    query = token[:, 0:1].expand(B, 1, *token.shape[2:])
    other = token[:, 1:2].expand(B, S - 1, *token.shape[2:])
    return torch.cat([query, other], dim=1).reshape(B * S, *token.shape[2:])

class Rope(nn.Module):
    base_frequency, scaling_factor = 100, 1.

class Block(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(1), nn.LayerNorm(1)
        self.attn = nn.Module()
        self.attn.num_heads, self.attn.scale, self.attn.rope = 16, .125, rope
        self.attn.q_norm, self.attn.k_norm = nn.LayerNorm(1), nn.LayerNorm(1)
        self.attn.qkv, self.attn.proj = nn.Linear(1,1), nn.Linear(1,1)
        self.mlp = nn.Module()
        self.mlp.act = nn.GELU(approximate='none')
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(1,1), nn.Linear(1,1)
        self.ls1, self.ls2 = nn.Module(), nn.Module()
        self.ls1.gamma, self.ls2.gamma = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
    def forward(self, *args, **kwargs):
        raise AssertionError('Grouped execution must never call a bypassed block')

class Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, images):
        self.calls += 1
        return {'x_norm_patchtokens': torch.zeros(len(images),48,1024)}

class Positions:
    def __call__(self, batch, height, width, device):
        grid = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij'), dim=-1)
        return grid.reshape(1,height*width,2).expand(batch,-1,-1).to(device)

class Aggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth, self.aa_block_size, self.aa_block_num, self.target_layer = 24,1,24,21
        self.aa_order, self.patch_size, self.patch_start_idx, self.use_reentrant = ['frame','global'],14,5,False
        self.retained_layer_indices = {4,11,17,23}
        self.rope, self.position_getter, self.patch_embed = Rope(), Positions(), Patch()
        self.frame_blocks = nn.ModuleList([Block(self.rope) for _ in range(24)])
        self.global_blocks = nn.ModuleList([Block(self.rope) for _ in range(24)])
        self.camera_token = nn.Parameter(torch.zeros(1,2,1,1024))
        self.register_token = nn.Parameter(torch.zeros(1,2,4,1024))
        self.register_buffer('_resnet_mean',torch.zeros(1,1,3,1,1))
        self.register_buffer('_resnet_std',torch.ones(1,1,3,1,1))
    def forward(self, images, compute_similarity=False):
        raise AssertionError('This generated CPU placeholder is not a model')
    def _process_frame_attention(self): pass
    def _process_global_attention(self): pass

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = Aggregator()
        self.inference_image_size = 112
        self.point_head = self.track_head = None
        self.eval().requires_grad_(False)

class NativeFunction:
    def __init__(self, fn): self.fn = fn
    def __call__(self,*args): return self.fn(*args)

class Neon:
    def __init__(self):
        def encode(source,destination,count):
            a=np.ctypeslib.as_array(ct.cast(source,ct.POINTER(ct.c_float)),shape=(count,))
            b=np.ctypeslib.as_array(ct.cast(destination,ct.POINTER(ct.c_uint16)),shape=(count,)).view(np.float16)
            np.copyto(b,a,casting='unsafe')
        def decode(source,destination,count):
            a=np.ctypeslib.as_array(ct.cast(source,ct.POINTER(ct.c_uint16)),shape=(count,)).view(np.float16)
            b=np.ctypeslib.as_array(ct.cast(destination,ct.POINTER(ct.c_float)),shape=(count,))
            np.copyto(b,a)
        self.f32_to_f16,self.f16_to_f32=NativeFunction(encode),NativeFunction(decode)

class API:
    def __init__(self):
        self.created,self.destroyed,self.get_counts,self.release_counts=0,[],[],[]
        self.init_failure = self.release_failure = self.destroy_failure = self.partial_failure = None
        self.ordinals,self.feeds,self.storage={},[],[]
        # Stable callable identities, like cached CDLL function attributes.
        for name in ('init','set_core_mask','inputs_set','run','outputs_get','outputs_release','destroy'):
            setattr(self,'rknn_'+name,NativeFunction(getattr(self,name)))
    def init(self,context,*args):
        self.created+=1
        context._obj.value=self.created
        self.ordinals[self.created]=self.created-1
        return -7 if self.created==self.init_failure else 0
    def set_core_mask(self,context,mask): return 0 if mask==7 else -1
    def query(self,context,command,value):
        ordinal=self.ordinals[context.value]
        public=rt.bindings(ordinal)
        if command==5:return SimpleNamespace(api_version=b'2.3.2-fake')
        if command==0:return SimpleNamespace(n_input=1,n_output=len(public))
        if command==6:return SimpleNamespace(total_dma_allocated_size=0)
        b=dict(index=0,name='input',shape=list(rt.FRAME)) if command==1 else public[value.index]
        count=int(np.prod(b['shape']))
        return dict(index=b['index'],name=b['name'],shape=b['shape'],n_elems=count,
            type=1,format=3,pass_through=0,size=count*2,quantization_type=0,zero_point=0,scale=1.)
    def inputs_set(self,context,count,inputs):
        assert count==1 and (inputs[0].type,inputs[0].pass_through,inputs[0].fmt)==(1,0,3)
        data=np.ctypeslib.as_array(ct.cast(inputs[0].buf,ct.POINTER(ct.c_uint16)),shape=(2*53*1024,)).view(np.float16).reshape(rt.FRAME).copy()
        self.feeds.append((context.value,data.copy()))
        ordinal=self.ordinals[context.value]
        self.storage=[data+np.float16(ordinal+1),(data+np.float16(ordinal+2)).reshape(rt.GLOBAL)]
        if ordinal==3:
            target=(data+np.float16(8)).reshape(rt.GLOBAL)
            self.last_target=target.astype(np.float32)
            self.storage.append(target)
        return 0
    def run(self,context,run):run._obj.frame_id=17;return 0
    def outputs_get(self,context,count,outputs,received):
        self.get_counts.append(count)
        received._obj.frame_id=17
        for i in range(count):
            if self.partial_failure==len(self.get_counts) and i>0:continue
            outputs[i].buf,outputs[i].size=self.storage[i].ctypes.data,self.storage[i].nbytes
        return -9 if self.partial_failure==len(self.get_counts) else 0
    def outputs_release(self,context,count,outputs):
        self.release_counts.append(count)
        if self.release_failure==len(self.release_counts):return -8
        for value in self.storage:value.fill(np.nan)
        return 0
    def destroy(self,context):
        if context.value==self.destroy_failure:return -13
        self.destroyed.append(context.value)
        return 0


def fixture(open_all=True):
    h=rt.RKNNRetainedOriginalAggregator.__new__(rt.RKNNRetainedOriginalAggregator)
    h.lock=threading.RLock()
    h.closed=h.failed=h.active=h.installed=False
    h.model=Model();h.aggregator=h.model.aggregator
    h.contexts,h.groups,h.guard=[],[],None
    h.api,h.neon=API(),Neon()
    h.base=SimpleNamespace(**{name:getattr(ABI,name) for name in ('CONTEXT','Input','Output','RunExtend',
        'OutputExtend','TensorAttr','SDKVersion','InputOutputNum','MemSize','check_call')})
    h.base.query=lambda api,ctx,command,value:api.query(ctx,command,value)
    h.base.attr_record=lambda value:value
    h.np,h.torch,h.timeout,h.core_mask=np,torch,30000,7
    h.stats=dict(context_init_attempts=0,contexts_created=0,contexts_destroyed=0,
        sdk_run_attempts=0,sdk_runs_succeeded=0,outputs_acquired=0,outputs_released=0,
        sdk_io_seconds=0.,completed_groups=0,completed_pairs=0,completed_blocks=0,
        completed_aggregator_calls=0,group_calls={},aggregator_forward_seconds=0.,guard_seconds=0.,
        cleanup_errors=[],model_memory=[])
    h.stats['half_io']=dict(calls_attempted=0,calls_completed=0,input_conversion_calls=0,output_conversion_calls=0,
        **{name+'_seconds':0. for name in ('input_conversion','output_conversion','owned_half_copy','sdk_api',
        'route','rknn_inputs_set','rknn_run','rknn_outputs_get','rknn_outputs_release')})
    agg=h.aggregator
    h.original_forward,h.original_had_forward,h.original_forward_value=agg.forward,False,None
    h.aggregator_globals=h.original_forward.__func__.__globals__
    h.slice_expand=slice_expand_and_flatten
    h.original_patch_embed,h.original_position_getter=agg.patch_embed,agg.position_getter
    h.forward_callable=None
    pos=agg.position_getter(2,6,8,device='cpu')+1
    pos=torch.cat([torch.zeros(2,5,2,dtype=torch.int64),pos],dim=1).numpy().copy()
    h.positions={'frame':pos,'global':pos.reshape(1,106,2).copy()}
    for value in h.positions.values():value.setflags(write=False)
    # Exercise all864 startup raw identities with tiny generated scalar weights.
    parameter_count=0
    for role in ('frame','global'):
        for block in getattr(agg,role+'_blocks'):
            state=dict(block.named_parameters());parameter_count+=len(state)
            parent=dict(parameter_raw_sha256={name:hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
                for name,value in state.items()},epsilons={name:h._norm_eps(block,name) for name in original.EPS},
                gelu_approximate='none',rope_frequency=100,rope_scaling_factor=1.,
                all_18_caller_parameter_values_preserved_in_graph=True)
            h._validate_block(block,parent,role)
    assert parameter_count==864
    if open_all:
        for ordinal,(start,end) in enumerate(rt.RANGES):
            h._open_group(ordinal,list(range(start,end+1)),rt.bindings(ordinal),Path('synthetic-not-a-model.rknn'))
        h.module_state,h.tensor_state=h._module_state(),h._tensor_state()
        h.config_state,h.backend_state=h._config_state(),h._backend_state()
        h.forward_callable=h._forward
        agg.forward=h.forward_callable;agg._vggt_rknn_blocks_owner=h;h.installed=True
        h.guard=agg.register_forward_pre_hook(h._guard,with_kwargs=True)
    return h


class Tests(unittest.TestCase):
    def test_prefix_retained_list_target20_and_cache_owner(self):
        h=fixture();cache=None
        try:
            with torch.no_grad(),redirect_stdout(io.StringIO()):
                values=h.aggregator(torch.zeros(1,2,3,84,112))
                self.assertEqual(len(values[0]),24)
                self.assertEqual([i for i,v in enumerate(values[0]) if v is not None],[4,11,17,23])
                self.assertTrue(all(tuple(values[0][i].shape)==(1,2,53,2048) for i in rt.RETAINED))
                expected=torch.from_numpy(h.api.last_target).view(2,53,1024).clone()
                expected=expected/(expected.norm(dim=-1,keepdim=True)+1e-8)
                self.assertTrue(torch.equal(values[2],expected))
                self.assertEqual(values[1],5);self.assertIsNone(values[3])
                self.assertEqual(h.api.get_counts,[2,2,2,3])
                cache=PatchCacheHandle(h.model,8<<20)
                second=h.aggregator(torch.zeros(1,2,3,84,112))
                self.assertTrue(torch.equal(second[2],values[2]))
                with self.assertRaisesRegex(RuntimeError,'Close original DINO patch_cache'):
                    h.close()
                self.assertFalse(h.closed)
                cache.close();cache=None
        finally:
            if cache is not None:cache.close()
            h.close()
        self.assertTrue(h.metrics()['original_forward_restored'])
        self.assertEqual(h.metrics()['live_contexts'],0)
        self.assertEqual(h.stats['contexts_created'],4)
        self.assertEqual(h.stats['contexts_destroyed'],4)
        self.assertEqual(h.stats['completed_blocks'],96)

    def test_first_and_last_parameters_config_and_hooks_detected(self):
        for kind in ('first_parameter','last_parameter','epsilon','hook'):
            h=fixture()
            try:
                with torch.no_grad():
                    if kind=='first_parameter':h.aggregator.frame_blocks[0].norm1.weight.add_(1)
                    elif kind=='last_parameter':h.aggregator.global_blocks[23].mlp.fc2.weight.add_(1)
                    elif kind=='epsilon':h.aggregator.global_blocks[23].norm2.eps*=2
                    else:h.aggregator.frame_blocks[11].attn.proj.register_forward_hook(lambda *args:None)
                    with self.assertRaises(RuntimeError):h._check_lifecycle()
            finally:h.close()

    def test_partial_initialization_destroys_nonzero_failed_handle(self):
        h=fixture(open_all=False);h.api.init_failure=3
        before=rt.method_identity(h.aggregator.forward)
        with self.assertRaises(RuntimeError):
            for ordinal,(start,end) in enumerate(rt.RANGES):
                h._open_group(ordinal,list(range(start,end+1)),rt.bindings(ordinal),Path('synthetic.rknn'))
        h.close()
        self.assertEqual(h.stats['contexts_created'],3)
        self.assertEqual(h.stats['contexts_destroyed'],3)
        self.assertEqual(h.metrics()['live_contexts'],0)
        self.assertEqual(rt.method_identity(h.aggregator.forward),before)

    def test_partial_outputs_get_failure_releases_and_restores(self):
        h=fixture();h.api.partial_failure=3
        with torch.no_grad(),self.assertRaises(RuntimeError):
            h.aggregator(torch.zeros(1,2,3,84,112))
        h.close()
        self.assertEqual(h.api.release_counts,[2,2,2])
        self.assertEqual(h.stats['outputs_acquired'],5)
        self.assertEqual(h.stats['outputs_released'],5)
        self.assertTrue(h.metrics()['original_forward_restored'])
        self.assertEqual(h.metrics()['live_contexts'],0)

    def test_release_destroy_failure_not_reported_as_clean(self):
        h=fixture();h.api.release_failure=1;h.api.destroy_failure=2
        with torch.no_grad(),self.assertRaises(RuntimeError):
            h.aggregator(torch.zeros(1,2,3,84,112))
        with self.assertRaises(RuntimeError):h.close()
        self.assertEqual(h.stats['outputs_acquired'],2)
        self.assertEqual(h.stats['outputs_released'],0)
        self.assertEqual(h.stats['contexts_destroyed'],3)
        self.assertEqual(h.metrics()['live_contexts'],1)
        self.assertTrue(h.metrics()['original_forward_restored'])
        self.assertTrue(any('outputs_release' in v for v in h.stats['cleanup_errors']))
        self.assertTrue(any('destroy' in v for v in h.stats['cleanup_errors']))

    def test_backend_group_or_position_changes_detected(self):
        for kind in ('groups','positions'):
            h=fixture()
            try:
                if kind=='groups':h.groups.pop(0)
                else:
                    h.positions['frame'].setflags(write=True)
                    h.positions['frame'][0,0,0]=999
                with torch.no_grad(),self.assertRaises(RuntimeError):h._check_lifecycle()
            finally:h.close()


if __name__=='__main__':
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    record=dict(scope='Generated tiny864 scalar parameters + fake typed API/NEON; no original model/SDK',
        actual_model_or_SDK_or_compiler_executed=False,
        runtime_sha256=hashlib.sha256(RUNTIME.read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),
        errors=len(result.errors),seconds=time.perf_counter()-started,python=__import__('sys').version,
        torch_version=torch.__version__,numpy_version=np.__version__)
    (HERE/'runtime-generated-tests.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
