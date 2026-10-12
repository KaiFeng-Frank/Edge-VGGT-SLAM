#!/usr/bin/env python3
"""Original CameraHead FC1/GELU output partition diagnostic on RK3588S only."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
RUNTIME_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
BASE = Path('/mnt/vggt-data/vggt-slam2')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--capture-sha256', required=True)
    p.add_argument('--compiled', nargs=2, action='append', required=True, metavar=('DIRECTORY', 'RECEIPT_SHA'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    assert platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
    assert int(Path('/sys/class/devfreq/fdab0000.npu/cur_freq').read_text()) == 300000000
    affinity = os.sched_getaffinity(0)
    a.output.mkdir(parents=True, exist_ok=False)
    r = dict(status='running', script_sha256=sha(__file__), original_model_only=True,
        scope='original camera trunk.0 FC1 plus exact GELU; four original refinement inputs',
        core_mask=7, init_flags=8 if a.profile else 0, profile=a.profile,
        normal_timing=not a.profile, clock_writes=False, full_model_or_realtime_validated=False,
        capture_sha256=a.capture_sha256, graphs=[], calls=[], cleanup_errors=[],
        contexts_created=0, contexts_destroyed=0, outputs_acquired=0, outputs_released=0,
        sdk_run_attempts=0, sdk_runs_succeeded=0)
    contexts = []
    api = base = None
    def write(): (a.output/'result.json').write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')
    write()
    try:
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import numpy as np
        def compare(actual, expected):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            d = actual.astype(np.float64)-expected.astype(np.float64)
            den = float(np.linalg.norm(expected.astype(np.float64)))
            outside = int(np.count_nonzero(np.abs(d) > 1e-5+1e-4*np.abs(expected.astype(np.float64))))
            return dict(bitwise_equal=actual.tobytes()==expected.tobytes(), elements=actual.size,
                max_abs=float(np.max(np.abs(d))), relative_l2=float(np.linalg.norm(d)/den) if den else None,
                outside=outside, strict_pass=outside==0, rtol=1e-4, atol=1e-5)
        def save(name, value):
            np.save(a.output/name, value, allow_pickle=False)
            return dict(file=name, sha256=sha(a.output/name), shape=list(value.shape), dtype=str(value.dtype))
        capture_path = a.capture/'capture.json'
        assert sha(capture_path) == a.capture_sha256
        capture = json.loads(capture_path.read_text())
        assert capture['status'] == 'completed' and capture['cleanup_errors'] == []
        assert capture['all_original_poses_bitwise_equal']
        inputs, refs = [], []
        for i in range(4):
            for name, shape, dest in [(f'input-{i}.npy', (1,2,2048), inputs), (f'output-{i}.npy', (1,2,8192), refs)]:
                row = capture['artifacts'][name]
                assert sha(a.capture/name) == row['sha256']
                value = np.load(a.capture/name, allow_pickle=False)
                assert value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
                dest.append(np.ascontiguousarray(value))
        for directory, digest in a.compiled:
            directory = Path(directory)
            assert sha(directory/'result.json') == digest
            graph = json.loads((directory/'result.json').read_text())
            assert graph['status'].startswith('compiled') and graph['cleanup_errors'] == [] and graph['released'] is True
            for key in ('onnx', 'rknn'): assert sha(directory/graph[key]['file']) == graph[key]['sha256']
            r['graphs'].append(dict(chunks=graph['chunks'], directory=str(directory), receipt_sha256=digest,
                onnx=graph['onnx'], rknn=graph['rknn']))
        assert sorted(g['chunks'] for g in r['graphs']) == [1,2,4]
        if not a.profile:
            import onnxruntime as ort
            r['ORT_version'] = ort.__version__
            for g in r['graphs']:
                options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = ort.InferenceSession(str(Path(g['directory'])/g['onnx']['file']), sess_options=options,
                    providers=['CPUExecutionProvider'])
                g['onnx_checks'] = []
                for i, x in enumerate(inputs):
                    y, = session.run(['output'], {'input':x})
                    row = dict(iteration=i, comparison=compare(y, refs[i]), artifact=save(f'ort-{g["chunks"]}-{i}.npy',y))
                    g['onnx_checks'].append(row)
                    assert row['comparison']['strict_pass'], 'FP32 ONNX original-operator validation failed'
                del session
        abi_path = BASE/'src/deploy/rk3588/validation/run-v12-rknn-block-board.py'
        assert sha(abi_path) == ABI_SHA and sha('/usr/lib/librknnrt.so') == RUNTIME_SHA
        spec = importlib.util.spec_from_file_location('_v69_abi',abi_path)
        base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
        api = base.bind_api(ct.CDLL('/usr/lib/librknnrt.so'))
        r.update(abi_sha256=ABI_SHA, runtime_sha256=RUNTIME_SHA)
        for g in r['graphs']:
            context = base.CONTEXT(); contexts.append(context)
            path = ct.create_string_buffer(os.fsencode(Path(g['directory'])/g['rknn']['file']))
            code = api.rknn_init(ct.byref(context),ct.cast(path,ct.c_void_p),0,r['init_flags'],None)
            r['contexts_created'] += int(bool(context.value)); base.check_call(code,'init')
            base.check_call(api.rknn_set_core_mask(context,7),'core_mask')
            version = base.query(api,context,5,base.SDKVersion())
            g['sdk_version'] = bytes(version.api_version).decode()
            g['driver_version'] = bytes(version.drv_version).decode()
            num = base.query(api, context,0,base.InputOutputNum()); assert num.n_input == num.n_output == 1
            for key, query, shape in [('input',1,[1,2,2048]),('output',2,[1,2,8192])]:
                desc = base.attr_record(base.query(api,context,query,base.TensorAttr(index=0)))
                assert desc['shape'] == shape and desc['type'] == 1 and desc['format'] in (0,1,3)
                assert desc['size'] == int(np.prod(shape))*2 and desc['pass_through'] == 0
                assert desc['quantization_type'] == 0 or (desc['quantization_type']==2 and desc['zero_point']==0 and desc['scale']==1.)
                g[key] = desc
            mem = base.query(api,context,6,base.MemSize())
            g['memory'] = dict(weight_bytes=int(mem.total_weight_size), internal_bytes=int(mem.total_internal_size), dma_bytes=int(mem.total_dma_allocated_size))
        baselines = {}
        # All calls, including first use, are measured; three balanced rounds.
        for cycle in range(1 if a.profile else 3):
            for iteration, x in enumerate(inputs):
                order = list(range(3)); shift = (cycle+iteration)%3; order = order[shift:]+order[:shift]
                for j in order:
                    g, context = r['graphs'][j], contexts[j]
                    request = (base.Input*1)(base.Input(0,x.ctypes.data,x.nbytes,0,0,g['input']['format']))
                    outputs = (base.Output*1)(); outputs[0].want_float = 1
                    run, received = base.RunExtend(0,0,30000,-1), base.OutputExtend()
                    row = dict(cycle=cycle, iteration=iteration, chunks=g['chunks'], first_use=cycle==0 and iteration==0)
                    r['calls'].append(row); got = False; started = time.perf_counter()
                    try:
                        base.check_call(api.rknn_inputs_set(context,1,request),'inputs_set'); t1=time.perf_counter()
                        r['sdk_run_attempts']+=1; code=api.rknn_run(context,ct.byref(run)); t2=time.perf_counter()
                        r['sdk_runs_succeeded']+=int(code==0); base.check_call(code,'run')
                        code=api.rknn_outputs_get(context,1,outputs,ct.byref(received)); t3=time.perf_counter()
                        r['outputs_acquired']+=int(bool(outputs[0].buf)); base.check_call(code,'outputs_get'); got=True
                        assert run.frame_id==received.frame_id and outputs[0].buf and outputs[0].size==65536
                        y=np.ctypeslib.as_array(ct.cast(outputs[0].buf,ct.POINTER(ct.c_float)),shape=(16384,)).copy().reshape(1,2,8192)
                        t4=time.perf_counter()
                        row.update(input_set_seconds=t1-started,run_seconds=t2-t1,output_get_seconds=t3-t2,copy_seconds=t4-t3)
                    finally:
                        if got or outputs[0].buf:
                            tick=time.perf_counter(); code=api.rknn_outputs_release(context,1,outputs)
                            row['release_seconds']=time.perf_counter()-tick
                            r['outputs_released']+=int(code==0)
                            if code: r['cleanup_errors'].append('outputs_release:'+str(code))
                            base.check_call(code,'outputs_release')
                        row['complete_io_seconds']=time.perf_counter()-started
                    row['original_FP32']=compare(y,refs[iteration])
                    row['artifact']=save(f'call-{len(r["calls"])-1:02d}.npy',y)
                    key=(g['chunks'],iteration)
                    if key in baselines:
                        row['same_graph_repeat']=compare(y,baselines[key]); assert row['same_graph_repeat']['bitwise_equal']
                    else: baselines[key]=y
                    if a.profile: row['perf_detail']=base.perf_detail(api,context)
        for row in r['calls']:
            y=np.load(a.output/row['artifact']['file'],allow_pickle=False)
            row['unsplit_FP16']=compare(y,baselines[(1,row['iteration'])])
        r['timing']={str(g['chunks']):dict(warm_mean_seconds=float(np.mean([c['complete_io_seconds'] for c in r['calls'] if c['chunks']==g['chunks'] and not c['first_use']])),
            warm_count=sum(c['chunks']==g['chunks'] and not c['first_use'] for c in r['calls'])) for g in r['graphs']}
        r['status']='completed_component_diagnostic_not_quality_accepted'
    except BaseException as ex:
        r.update(status='failed',error=f'{type(ex).__name__}: {ex}'); raise
    finally:
        for context in reversed(contexts):
            if context.value:
                try:
                    code=api.rknn_destroy(context)
                    if code:r['cleanup_errors'].append('destroy:'+str(code))
                    else:r['contexts_destroyed']+=1; context.value=0
                except BaseException as ex:r['cleanup_errors'].append('destroy exception:'+str(ex))
        r['live_contexts']=sum(bool(x.value) for x in contexts)
        try:
            os.sched_setaffinity(0,affinity)
            r['affinity_restored']=os.sched_getaffinity(0)==affinity
            if not r['affinity_restored']:r['cleanup_errors'].append('affinity restore mismatch')
        except BaseException as ex:r['cleanup_errors'].append('affinity restore:'+str(ex))
        if r['cleanup_errors'] or r['live_contexts']:r['status']='failed_cleanup'
        write(); print(json.dumps({k:r.get(k) for k in ('status','error','timing','contexts_created','contexts_destroyed','sdk_run_attempts','outputs_acquired','outputs_released','cleanup_errors')}),flush=True)
    assert r['status']=='completed_component_diagnostic_not_quality_accepted'

if __name__=='__main__':main()
