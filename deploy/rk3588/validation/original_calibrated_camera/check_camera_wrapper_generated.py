"""Run the real wrapper with generated installers; no Torch, SDK, or board calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

HERE = Path(__file__).parent
source = HERE / 'run_calibrated_camera_complete.py'
spec = importlib.util.spec_from_file_location('_camera_wrapper_generated', source)
wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
original_read = Path.read_bytes
outcomes = []

class Handle:
    def __init__(self, close_failure): self.close_failure, self.closed, self.close_calls = close_failure, False, 0
    def close(self):
        self.close_calls += 1; self.closed = True
        if self.close_failure: raise RuntimeError('generated destroy failure')
    def metrics(self):
        return dict(closed=self.closed, live_contexts=int(self.close_failure), contexts_created=2,
            contexts_destroyed=1 if self.close_failure else 2,
            cleanup_errors=['generated destroy failure'] if self.close_failure else [])

for case in ('success', 'comparator_before_cleanup_failure', 'camera_close_failure'):
    with tempfile.TemporaryDirectory(prefix='wrapper-', dir=HERE) as name:
        root = Path(name); receipt = root/'wrapper.json'; output = root/'output'; handles = []
        package = ModuleType('vggt_slam'); package.__path__ = []
        modules = {'vggt_slam': package}
        for short, installer in [('rknn_retained_aggregator','install_retained_aggregator'), ('patch_cache','install_patch_cache'), ('rknn_camera_head','install_rknn_camera_head')]:
            module = ModuleType('vggt_slam.'+short); setattr(module, installer, lambda *a, **kw: None)
            setattr(package, short, module); modules[module.__name__] = module
        camera_module = modules['vggt_slam.rknn_camera_head']
        original_camera = camera_module.install_rknn_camera_head
        def install_camera(*args, **options):
            handle = Handle(case == 'camera_close_failure'); handles.append(handle); return handle
        class Loader:
            def __init__(self, name): self.name = name
            def exec_module(self, module):
                if self.name == '_v82_calibrated_camera': module.install_calibrated_camera = install_camera
                else: module.install_calibrated_retained = lambda *a, **kw: None
        def fake_spec(name, path): return NS(name=name, loader=Loader(name))
        def fake_module(spec): return ModuleType(spec.name)
        def fake_run(*args, **kw):
            camera_module.install_rknn_camera_head(NS(), split_fc1=True)
            if case == 'comparator_before_cleanup_failure': raise ValueError('generated early comparator failure')
        def fake_sha(path):
            path = str(path)
            if path.endswith('/patch_cache.py'): return wrapper.CACHE_SHA
            if path.endswith('/rknn_camera_head.py'): return 'e20518d08c47746e1bf58e07bd6d0270bc7c7f6daa4d0858593f478ce6dfb09e'
            if path.endswith(wrapper.ENTRY): return wrapper.ENTRY_SHA
            if path == '/generated/retained.py': return 'a'*64
            if path == '/generated/mapping.json': return 'b'*64
            if path == '/generated/camera.py': return 'c'*64
            if path == '/generated/compiled.json': return 'd'*64
            return hashlib.sha256(original_read(Path(path))).hexdigest()
        argv = [str(source), '--adapter-source','/generated/retained.py','--adapter-sha256','a'*64,
            '--quantized-manifest','/generated/mapping.json','--quantized-manifest-sha256','b'*64,
            '--camera-adapter-source','/generated/camera.py','--camera-adapter-sha256','c'*64,
            '--camera-compiled','/generated/compiled.json','--camera-compiled-sha256','d'*64,
            '--receipt',str(receipt),'--','--retained-aggregator','--paired-aggregator','--half-io-library',
            '--dino-manifest','--grouped-dino','--depth-manifest','--reference-run','--camera-split4','--output',str(output)]
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv), patch.object(wrapper, 'sha', fake_sha), \
             patch.object(Path, 'read_bytes', lambda p: b'rockchip,rk3588s\0' if str(p) == '/proc/device-tree/compatible' else original_read(p)), \
             patch.object(wrapper.importlib.util, 'spec_from_file_location', fake_spec), \
             patch.object(wrapper.importlib.util, 'module_from_spec', fake_module), patch.object(wrapper.runpy, 'run_path', fake_run):
            try:
                code = wrapper.main(); assert case != 'comparator_before_cleanup_failure'
                assert code == (1 if case == 'camera_close_failure' else 0)
            except ValueError:
                assert case == 'comparator_before_cleanup_failure'
            saved = json.loads(receipt.read_text())
            assert len(handles) == 1 and handles[0].close_calls == 1
            assert camera_module.install_rknn_camera_head is original_camera and saved['installers_restored']
            assert saved['camera_resources_closed'] == (case != 'camera_close_failure')
            assert saved['status'].startswith('completed_') == (case == 'success')
            outcomes.append(dict(case=case, passed=True, status=saved['status'],
                camera_resources_closed=saved['camera_resources_closed']))
assert 'torch' not in sys.modules
report = dict(schema='v82_calibrated_camera_wrapper_generated_checks', passed=len(outcomes),
    actual_model_or_SDK_or_device_execution=False, cases=outcomes,
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,Path(__file__))})
(HERE/'camera-wrapper-generated-result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
