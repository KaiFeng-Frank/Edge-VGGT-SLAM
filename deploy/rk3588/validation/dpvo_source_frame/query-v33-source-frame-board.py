#!/usr/bin/env python3
"""Query exact logical/native tensor attributes; no rknn_run invocation."""
import ctypes as ct,hashlib,importlib.util,json,platform,sys,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2');B=BASE/'experiments'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def main():
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'Board-only query')
    out=B/'vggt-v33-source-frame-attrs-r1';out.mkdir(exist_ok=False)
    paths={BASE/'src/deploy/rk3588/validation/run-v12-rknn-block-board.py':'37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608',
           BASE/'src/deploy/rk3588/reference/rknn_api_v2.3.2.h':'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a',
           Path('/usr/lib/librknnrt.so'):'d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8',
           B/'vggt-v33-source-frame-compile-r1/block.rknn':'4cd92fedc1b88540bcefdcbe50164f8b7d7201cc07682540e25047d3cc64b975'}
    pins={str(p):v for p,v in paths.items()};pins[str(Path(__file__).resolve())]=sha(__file__)
    r=dict(schema='v33_board_SDK_attribute_query',status='failed',inference_executed=False,SDK_run_calls=0,cleanup_errors=[],source_pins=pins)
    api=abi=ctx=None
    try:
        require({p:sha(p) for p in pins}==pins,'Query source/model mismatch')
        source=next(iter(paths));spec=importlib.util.spec_from_file_location('_v33_query_abi',source);abi=importlib.util.module_from_spec(spec);spec.loader.exec_module(abi)
        api=abi.bind_api(ct.CDLL('/usr/lib/librknnrt.so'));ctx=abi.CONTEXT();model=ct.create_string_buffer(str(B/'vggt-v33-source-frame-compile-r1/block.rknn').encode())
        abi.check_call(api.rknn_init(ct.byref(ctx),ct.cast(model,ct.c_void_p),0,0,None),'init')
        abi.check_call(api.rknn_set_core_mask(ctx,7),'core_mask')
        version=abi.query(api,ctx,5,abi.SDKVersion());r['SDK_version']=bytes(version.api_version).decode();r['driver_version']=bytes(version.drv_version).decode()
        n=abi.query(api,ctx,0,abi.InputOutputNum());require(n.n_input==n.n_output==3,'Counts changed')
        r['inputs']=[abi.attr_record(abi.query(api,ctx,1,abi.TensorAttr(index=i))) for i in range(3)]
        r['outputs']=[abi.attr_record(abi.query(api,ctx,2,abi.TensorAttr(index=i))) for i in range(3)]
        r['status']='queried_attributes_no_inference'
    except Exception as e:r.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        if ctx is not None and ctx.value:
            try:abi.check_call(api.rknn_destroy(ctx),'destroy');r['context_destroyed']=True
            except Exception as e:r['cleanup_errors'].append(repr(e));r['status']='failed'
        r['source_hashes_after']={p:sha(p) for p in pins};r['sources_and_inputs_unchanged']=r['source_hashes_after']==pins
        if not r['sources_and_inputs_unchanged']:r['status']='failed'
        (out/'query.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r),flush=True)
    return 0 if r['status']=='queried_attributes_no_inference' else 1
if __name__=='__main__':raise SystemExit(main())
