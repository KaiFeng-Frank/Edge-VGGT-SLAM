"""Generated lifecycle checks only: no Torch, model, SDK, or device execution."""
import ctypes as ct
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

ROOT = Path('/dev/shm/edge-vggt-v50/public-snapshot')
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
runtime = load(HERE / 'rknn_calibrated_camera.py', '_generated_camera')
io = load(Path('/dev/shm/edge-vggt-v80/rknn_calibrated_retained.py'), '_generated_io')
parent = runtime.parent
old_init, old_load, old_contract = parent.RKNNOriginalCameraHead.__init__, parent.load_source, runtime.load_compilation
original_forward = parent.RKNNOriginalCameraHead._make_forward
outcomes = []

class API:
    def __init__(self, failure): self.failure, self.live, self.destroyed = failure, {1}, []
    def rknn_init(self, pointer, *_):
        ct.cast(pointer, ct.POINTER(ct.c_uint64))[0] = 2; self.live.add(2)
        return -1 if self.failure == 'init_nonzero_context' else 0
    def rknn_set_core_mask(self, *_): return -1 if self.failure in ('mask', 'pending_destroy') else 0
    def rknn_destroy(self, context):
        if self.failure == 'pending_destroy' and context.value == 2: return -1
        if self.failure == 'old_destroy_once' and context.value == 1 and not hasattr(self, 'old_failed'):
            self.old_failed = True; return -1
        self.live.remove(context.value); self.destroyed.append(context.value); return 0

class Base:
    CONTEXT = ct.c_uint64
    TensorAttr = staticmethod(lambda **kw: NS(**kw))
    SDKVersion = InputOutputNum = MemSize = staticmethod(lambda: NS())
    def __init__(self, api): self.api = api
    @staticmethod
    def check_call(code, name):
        if code != 0: raise RuntimeError(name + ':' + str(code))
    @staticmethod
    def attr_record(attr): return attr.row
    def query(self, api, context, command, attr):
        if command == 5: return NS(api_version=b'2.3.2', drv_version=b'generated')
        if command == 0: return NS(n_input=2, n_output=1)
        if command == 6: return NS(total_dma_allocated_size=10, total_internal_size=4, total_weight_size=6)
        shape = ([1, 2, 2048] if attr.index == 0 else [1, 2, 9]) if command == 1 else [1, 2, 9]
        name = ('camera_tokens' if attr.index == 0 else 'conditioning_pose') if command == 1 else 'pose_delta'
        count = math.prod(shape)
        row = dict(index=attr.index, name=name, shape=shape, n_elems=count, type=2, format=3,
            pass_through=0, size=count, quantization_type=2, zero_point=0, scale=.125)
        if self.api.failure == 'invalid_affine': row['scale'] = 0.
        return NS(name=name.encode(), row=row)

try:
    parent.load_source = lambda *a: io
    runtime.load_compilation = lambda *a: (Path('/generated/result.json'), {'rknn': {'sha256': '5'*64, 'bytes': 1}}, Path('/generated/model.rknn'))
    for failure in ('success', 'early_parent', 'init_nonzero_context', 'mask', 'invalid_affine', 'old_destroy_once', 'pending_destroy'):
        api = API(failure); holders = []
        def fake_init(self, model, **options):
            holders.append(self); self.lock = threading.RLock(); self.closed = self.failed = False
            self.base = self.api = self.head = self.patch = None; self.context = None
            self.stats = dict(contexts_created=0, contexts_destroyed=0, cleanup_errors=[], core_mask=7,
                model_sha256='0'*64, sdk_memory={'weight_bytes': 1}, candidate_accepted=False)
            if failure == 'early_parent': api.live.clear(); raise ValueError('before API binding')
            self.api = api; self.base = Base(api); self.context = ct.c_uint64(1)
            self.stats['contexts_created'] = 1; self.head = NS(_vggt_rknn_camera_owner=self)
        parent.RKNNOriginalCameraHead.__init__ = fake_init
        try:
            handle = runtime.RKNNCalibratedCameraHead(NS(), '/generated/result.json', 'a'*64, source_root=ROOT)
            assert failure == 'success'
            assert handle.metrics()['contexts_created'] == 2 and handle.metrics()['contexts_destroyed'] == 1
            assert handle.metrics()['live_contexts'] == 1 and handle.context.value == 2
            handle.close(); handle.close(); metrics = handle.metrics()
        except BaseException as error:
            assert failure != 'success'; metrics = error.camera_rknn_metrics
        assert metrics['closed']
        if failure == 'pending_destroy':
            assert metrics['live_contexts'] == 1 and metrics['cleanup_errors'] and api.live == {2}
            assert metrics['contexts_created'] == 2 and metrics['contexts_destroyed'] == 1
        else:
            assert metrics['live_contexts'] == 0 and not api.live
            assert metrics['contexts_created'] == metrics['contexts_destroyed']
        assert parent.RKNNOriginalCameraHead._make_forward is original_forward
        outcomes.append(dict(case=failure, passed=True, live_contexts=metrics['live_contexts'],
            contexts_created=metrics['contexts_created'], contexts_destroyed=metrics['contexts_destroyed']))
finally:
    parent.RKNNOriginalCameraHead.__init__, parent.load_source, runtime.load_compilation = old_init, old_load, old_contract
assert 'torch' not in sys.modules
report = dict(schema='v82_calibrated_camera_generated_checks', passed=len(outcomes),
    actual_model_or_SDK_or_device_execution=False, cases=outcomes,
    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (HERE/'rknn_calibrated_camera.py', HERE/'run_calibrated_camera_complete.py', Path(__file__))})
(HERE/'calibrated-camera-generated-result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
