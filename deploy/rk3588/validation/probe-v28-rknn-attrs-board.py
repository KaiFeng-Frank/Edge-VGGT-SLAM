#!/usr/bin/env python3
"""Inspect compiled Update SDK contracts on RK3588, with no inference call."""
import argparse,ctypes as ct,hashlib,importlib.util,json,platform,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2')
ABI=BASE/'src/deploy/rk3588/validation/run-v12-rknn-block-board.py'
HEADER=BASE/'src/deploy/rk3588/reference/rknn_api_v2.3.2.h'
LIB=Path('/usr/lib/librknnrt.so')
PINS={str(ABI):'37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608',str(HEADER):'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'}
PINS[str(LIB)]='d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board only')
    require(not a.output.exists() and a.output.resolve().is_relative_to(BASE/'experiments'),'New isolated output required')
    pins=dict(PINS);pins[str(a.model)]=a.model_sha256;pins[str(Path(__file__))]=sha(__file__)
    r=dict(status='failed',script_sha256=sha(__file__),model_inference_executed=False,NPU_execution_measured=False,source_pins=pins,cleanup_errors=[]);context=None;api=None
    try:
        for path,digest in pins.items():require(sha(path)==digest,'Pinned SDK/model/source differs')
        spec=importlib.util.spec_from_file_location('_v28_attrs_abi',ABI);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        api=m.bind_api(ct.CDLL(str(LIB)));context=m.CONTEXT();data=a.model.read_bytes();buffer=ct.create_string_buffer(data)
        m.check_call(api.rknn_init(ct.byref(context),ct.cast(buffer,ct.c_void_p),len(data),0,None),'rknn_init');m.check_call(api.rknn_set_core_mask(context,7),'rknn_set_core_mask')
        counts=m.query(api,context,0,m.InputOutputNum());v=m.query(api,context,5,m.SDKVersion());memory=m.query(api,context,6,m.MemSize())
        r.update(status='queried_contract_no_inference',inputs=[m.attr_record(m.query(api,context,1,m.TensorAttr(index=i))) for i in range(counts.n_input)],outputs=[m.attr_record(m.query(api,context,2,m.TensorAttr(index=i))) for i in range(counts.n_output)],versions=dict(api=bytes(v.api_version).decode(),driver=bytes(v.drv_version).decode()),memory=dict(total_dma_allocated_size=int(memory.total_dma_allocated_size),total_weight_size=int(memory.total_weight_size),total_internal_size=int(memory.total_internal_size)),abi=m.abi_record())
    except Exception as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        if context is not None and context.value:
            code=int(api.rknn_destroy(context));r['destroy_code']=code
            if code:r['status']='failed';r['cleanup_errors'].append('rknn_destroy '+str(code))
        r['source_hashes_after']={path:sha(path) for path in pins};r['sources_unchanged']=r['source_hashes_after']==pins
        if not r['sources_unchanged']:r['status']='failed'
        a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r),flush=True);return 0 if r['status']=='queried_contract_no_inference' else 1
if __name__=='__main__':raise SystemExit(main())
