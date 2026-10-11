#!/usr/bin/env python3
"""Import-only ABI preflight; no model load, tensor forward, ORT session or RKNN object."""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert platform.machine() == 'aarch64'
assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
prefix = Path('/mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-venv-r1')
assert Path(sys.prefix).resolve() == prefix
assert '/mnt/vggt-data/vggt-slam2/src' not in sys.path
report = dict(schema='v49_toolchain_import_only_ABI_preflight',status='failed',
              script_sha256=sha(__file__),prefix=str(prefix),machine=platform.machine(),
              model_loaded=False,model_forward_executed=False,RKNN_object_created=False,
              ORT_session_created=False,modules={})
try:
    for name in ('torch','numpy','onnx','onnxoptimizer','onnxruntime','cv2','scipy','fast_histogram','rknn.api'):
        t=time.monotonic()
        m=importlib.import_module(name)
        p=Path(inspect.getfile(m)).resolve()
        assert p.is_relative_to(prefix), (name,p)
        report['modules'][name]=dict(path=str(p),sha256=sha(p),version=str(getattr(m,'__version__',None)),seconds=time.monotonic()-t)
    report['versions']={name:importlib.metadata.version(name) for name in ('torch','numpy','onnx','onnxoptimizer','onnxruntime','rknn-toolkit2','protobuf')}
    report['status']='imports_passed_no_model_execution'
except Exception as exc:
    report['error']=f'{type(exc).__name__}: {exc}'
out=Path('/mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-install-r1/imports.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
sys.exit(0 if report['status']=='imports_passed_no_model_execution' else 2)
