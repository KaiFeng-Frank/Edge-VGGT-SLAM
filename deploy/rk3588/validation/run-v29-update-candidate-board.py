#!/usr/bin/env python3
"""Saved-state Update candidate diagnostics, CPU ORT or synchronous RKNN.

No Torch/checkpoint loading, BA, live trajectory or production replacement.
Raw outputs are saved before numerical judgment, including NaN/Inf taps.
Exit 0: saved-state numerical gates passed; 2: execution completed but gates
failed; 1: execution/provenance/cleanup failure. Read run.json, not transport exit.
"""
from __future__ import annotations
import argparse
import contextlib
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import traceback
import numpy as np

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
CAPTURE_SHA = '87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
EXPORT_SHA = '56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac'
PACKING_SHA = '040f55b89accb7a03a523ef2ab8c4bf12755f15da34813dc6ee607b6deb90c88'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIB_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
CORE_NAMES = ('net_out', 'delta', 'weight')
REF_NAMES = ('net', 'delta', 'weight')
RTOL, ATOL = 1e-4, 1e-5


def require(ok, reason):
    if not ok: raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj
    spec.loader.exec_module(obj); return obj


def meta(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes,
                tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def finite_counts(a):
    mask = np.isfinite(a); values = a[mask]
    return dict(elements=a.size, finite=int(mask.sum()), nan=int(np.isnan(a).sum()),
                positive_inf=int(np.isposinf(a).sum()), negative_inf=int(np.isneginf(a).sum()),
                all_finite=bool(mask.all()), all_zero=bool((a == 0).all()),
                finite_min=float(values.min()) if values.size else None,
                finite_max=float(values.max()) if values.size else None)


def compare(a, b):
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape, 'Comparison FP32 shape/dtype mismatch')
    finite = np.isfinite(a) & np.isfinite(b); bad = int((~finite).sum())
    if bad:
        return dict(rtol=RTOL, atol=ATOL, gate_passed=False, nonfinite_pairs=bad,
                    failed_elements=None, max_abs=None, relative_l2=None, comparison_undefined=True)
    d = a.astype(np.float64)-b.astype(np.float64)
    count = int(np.count_nonzero(np.abs(d) > ATOL+RTOL*np.abs(b.astype(np.float64))))
    return dict(rtol=RTOL, atol=ATOL, gate_passed=count == 0, failed_elements=count,
                max_abs=float(np.abs(d).max()), relative_l2=float(np.linalg.norm(d.ravel()) /
                max(np.linalg.norm(b.astype(np.float64).ravel()), 1e-30)), bitwise_equal=a.tobytes() == b.tobytes())


class Inputs:
    def __init__(self): self.pins = {}
    def pin(self, path, expected):
        path = Path(path).resolve(); require(sha(path) == expected, 'Pinned file differs: '+str(path))
        self.pins[str(path)] = expected; return path
    def child(self, folder, name):
        require(isinstance(name, str) and Path(name).name == name and name not in ('', '.', '..'), 'Artifact basename required')
        return Path(folder)/name
    def artifact(self, folder, item):
        path = self.pin(self.child(folder, item['file']), item['sha256'])
        require(path.stat().st_size == item['bytes'], 'Artifact bytes differ')
        with np.load(path, allow_pickle=False) as z: values = {k:z[k].copy() for k in z.files}
        require({k:meta(v) for k,v in values.items()} == item['arrays'], 'Artifact raw metadata differs')
        return values
    def unchanged(self): return {p:sha(p) for p in self.pins} == self.pins


def load_chain(a, pins):
    capture = json.loads(pins.pin(a.capture_dir/'capture.json', CAPTURE_SHA).read_text())
    exported = json.loads(pins.pin(a.export_dir/'export.json', EXPORT_SHA).read_text())
    manifest = json.loads(pins.pin(a.manifest, a.manifest_sha256).read_text())
    require(capture['status'] == 'captured_original_Update_states_not_VO' and capture['original_weights_unchanged'], 'Original capture required')
    require(exported['status'] == 'exported_Update_board_ORT_pending' and exported['capture_sha256'] == CAPTURE_SHA
            and exported['exact_wrapper_FP32_gate_passed'] and exported['original_weights_unchanged'], 'Original validated wrapper export required')
    for receipt in (capture, exported):
        require(receipt['sources_and_inputs_unchanged'] and not receipt['cleanup_errors']
                and receipt['source_pins'] == receipt['source_hashes_after'], 'Ancestor source/cleanup check failed')
    require(manifest['schema'] == 'v29_DPVO_Update_candidate_manifest'
            and manifest['source_export_sha256'] == EXPORT_SHA and manifest['source_capture_sha256'] == CAPTURE_SHA,
            'Candidate parent identity differs')
    require(0 < len(manifest['models']) <= 2 and set(manifest['models']) <= {'E96','E864'}, 'Only original E96/E864 candidate shapes supported')
    for key, row in manifest['models'].items():
        original = exported['models'][key]
        require(row['source_onnx_sha256'] == original['sha256'] and row['inputs'] == original['inputs'], 'Original parent graph/input contract differs')
        require(row['outputs'][:3] == original['outputs'] and tuple(x['name'] for x in row['outputs'][:3]) == CORE_NAMES,
                'Original three outputs must remain first and unchanged')
        require(3 <= len(row['outputs']) <= 32 and len({x['name'] for x in row['outputs']}) == len(row['outputs']), 'Duplicate/unbounded output names')
        for output in row['outputs']:
            shape = output['shape']
            require(output['dtype'] == 1 and 1 <= len(shape) <= 4 and all(type(x) is int and x > 0 for x in shape), 'Only fixed rank1..4 FP32 logical outputs supported')
        require(sum(math.prod(x['shape'])*4 for x in row['outputs']) <= 128 << 20, 'Output payload budget exceeded')
        for folder, graph in ((a.export_dir, original), (a.manifest.parent, row)):
            path = pins.pin(pins.child(folder, graph['file']), graph['sha256'])
            require(path.stat().st_size == graph['bytes'], 'ONNX bytes differ')
    return capture, exported, manifest


def records(capture, exported, manifest, key=None, first_only=False):
    selected = {key} if key else set(manifest['models'])
    require(selected <= set(manifest['models']), 'Requested model missing in candidate manifest')
    rows = []
    for case in capture['scenarios']:
        rewritten = next(x for x in exported['scenarios'] if x['name'] == case['name'])
        if rewritten['model_key'] not in selected: continue
        require([x['step'] for x in case['states']] == list(range(4)), 'Four captured states per case required')
        for state in case['states']:
            wrapper = next(x for x in rewritten['states'] if x['step'] == state['step'])
            rows.append(dict(case=case['name'], step=state['step'], model_key=rewritten['model_key'],
                             topology=rewritten['topology'], inputs=state['inputs'], original=state['outputs'], wrapper=wrapper['outputs']))
    require(len(rows) == 8*len(selected), 'Expected all eight states for each selected model')
    return rows[:1] if first_only else rows


def load_feed(row, a, pins, packing, model, abi):
    inputs = pins.artifact(a.capture_dir, row['inputs'])
    topology = pins.artifact(a.export_dir, row['topology'])
    original = pins.artifact(a.capture_dir, row['original']); wrapped = pins.artifact(a.export_dir, row['wrapper'])
    feed = dict(net=inputs['net'], inp=inputs['ctx'], corr=inputs['corr'], **topology)
    require(set(feed) == set(packing.INPUT_NAMES), 'Exactly original thirteen runtime inputs required')
    # Reuse the frozen domain/dtype validator without claiming SDK execution.
    attrs = []
    for i, item in enumerate(model['inputs']):
        integer = item['name'] in packing.INDEX_NAMES; count = math.prod(item['shape'])
        attrs.append(dict(index=i, name=item['name'], shape=item['shape'], n_elems=count,
                          size=count*(8 if integer else 4), format=3, type=8 if integer else 0,
                          quantization_type=0, scale=1., zero_point=0, fractional_length=0))
    bound = packing.validate_input_attrs(model['inputs'], attrs)
    # Pure ctypes packing executes the pinned value checks before ORT as well
    # as SDK. This preparatory validation/copy is outside forward timing.
    packing.pack_inputs(abi, feed, bound)
    for name in ('kk', 'ij'):
        mask = feed[name+'_valid']
        require(bool((np.diff(mask,axis=1) <= 0).all()), 'Group valid mask must be a prefix')
        nonempty = mask.sum(axis=1) > 0
        require(bool(nonempty[feed[name+'_inverse']].all()), 'Inverse must not address an empty group')
    return feed, original, wrapped, bound


def output_bindings(packing, contracts, attrs):
    require(len(attrs) == len(contracts), 'Output attr count differs')
    return [packing._attr(raw, dict(name=item['name'], shape=tuple(item['shape']), logical_dtype='float32'), i)
            for i, (item,raw) in enumerate(zip(contracts,attrs))]


def execute_sdk(api, context, abi, packing, feed, bindings, outputs_bound, cleanup):
    begin = time.monotonic(); packed = packing.pack_inputs(abi, feed, bindings); pack_end = time.monotonic()
    outputs = (abi.Output*len(outputs_bound))()
    for i in range(len(outputs)): outputs[i] = abi.Output(1,0,i,None,0)
    run = abi.RunExtend(0,0,30000,-1); received = abi.OutputExtend(); got = False; owned = {}; timing = {}
    try:
        abi.check_call(api.rknn_inputs_set(context,13,packed.requests), 'rknn_inputs_set'); set_end = time.monotonic()
        abi.check_call(api.rknn_run(context,ct.byref(run)), 'rknn_run'); run_end = time.monotonic()
        abi.check_call(api.rknn_outputs_get(context,len(outputs),outputs,ct.byref(received)), 'rknn_outputs_get'); got = True; get_end = time.monotonic()
        require(run.frame_id == received.frame_id, 'Synchronous output frame mismatch')
        for i, row in enumerate(outputs_bound):
            count = math.prod(row['shape'])
            require(outputs[i].buf and outputs[i].size == count*4, 'Returned FP32 bytes differ')
            owned[row['name']] = np.ctypeslib.as_array(ct.cast(outputs[i].buf,ct.POINTER(ct.c_float)),shape=(count,)).copy().reshape(row['shape'])
        copy_end = time.monotonic()
        timing.update(pack_seconds=pack_end-begin,input_set_seconds=set_end-pack_end,run_api_seconds=run_end-set_end,
                      get_seconds=get_end-run_end,copy_seconds=copy_end-get_end,frame_id=int(run.frame_id),input_records=packed.records)
    finally:
        if got or any(x.buf for x in outputs):
            t = time.monotonic()
            try:
                code = int(api.rknn_outputs_release(context,len(outputs),outputs))
                if code: cleanup.append('rknn_outputs_release '+str(code))
                abi.check_call(code,'rknn_outputs_release')
                timing.update(release_seconds=time.monotonic()-t,forward_seconds=time.monotonic()-begin,outputs_released=True)
            except Exception as error:
                if not cleanup: cleanup.append(repr(error))
                raise
    return owned, timing


def save_call(out, index, state, values, model, original, wrapped, timing):
    require(tuple(values) == tuple(x['name'] for x in model['outputs']), 'Output name/order mismatch')
    for item in model['outputs']:
        a = values[item['name']]
        require(a.dtype == np.float32 and a.shape == tuple(item['shape']), 'Output shape/dtype mismatch')
    path = out/f'call-{index:02d}.npz'; np.savez(path,**values)
    finite = {key:finite_counts(a) for key,a in values.items()}
    checks = {key:compare(values[name],original[key]) for name,key in zip(CORE_NAMES,REF_NAMES)}
    wrapper_checks = {key:compare(values[name],wrapped[key]) for name,key in zip(CORE_NAMES,REF_NAMES)}
    w = values['weight']; domain = bool(np.isfinite(w).all() and ((w >= 0)&(w <= 1)).all())
    return dict(index=index,case=state['case'],step=state['step'],model_key=state['model_key'],timing=timing,
                outputs=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in values.items()}),
                finite_counts=finite,all_outputs_finite=all(v['all_finite'] for v in finite.values()),weight_domain_passed=domain,
                versus_original=checks,versus_wrapper=wrapper_checks,
                core_strict_FP32_gate_passed=all(x['gate_passed'] for x in (*checks.values(),*wrapper_checks.values())))


@contextlib.contextmanager
def native_log(path):
    # One process, no overlapping native workloads. Capture fd1+fd2 including
    # destroy; SDK errors can coexist with a zero C API return.
    sys.stdout.flush(); sys.stderr.flush(); saved = []; stream = None
    try:
        stream = Path(path).open('xb',buffering=0)
        for fd in (1,2):
            saved.append((fd,os.dup(fd))); os.dup2(stream.fileno(),fd)
        yield
    finally:
        with contextlib.suppress(Exception): ct.CDLL(None).fflush(None)
        for fd, old in reversed(saved): os.dup2(old,fd); os.close(old)
        if stream is not None: stream.close()


def native_errors(path):
    text = Path(path).read_text(errors='replace')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
    return [line for line in text.splitlines() if re.search(r'\bE\s+RKNN\b',line)]


def run_ort(a, pins, manifest, selected, exported, packing, abi, out, report):
    import onnxruntime as ort
    sessions = {}
    try:
        for key in dict.fromkeys(x['model_key'] for x in selected):
            model = manifest['models'][key]; opts = ort.SessionOptions()
            opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(a.manifest.parent/model['file']),sess_options=opts,providers=['CPUExecutionProvider'])
            require(session.get_providers() == ['CPUExecutionProvider'], 'CPU ORT only')
            for actual,wanted in ((session.get_inputs(),model['inputs']),(session.get_outputs(),model['outputs'])):
                require([x.name for x in actual] == [x['name'] for x in wanted], 'ORT I/O names/order differ')
                for x,row in zip(actual,wanted):
                    require(x.shape == row['shape'] and x.type == ('tensor(int64)' if row['dtype']==7 else 'tensor(float)'), 'ORT shape/type differs')
            sessions[key]=session
        report.update(ORT_version=ort.__version__,ORT_optimization='disabled',providers=['CPUExecutionProvider'])
        for row in selected:
            model=manifest['models'][row['model_key']];feed,original,wrapped,bound=load_feed(row,a,pins,packing,model,abi)
            begin=time.monotonic(); prepared={k:np.array(v,copy=True,order='C') for k,v in feed.items()}; pack_end=time.monotonic()
            # CPU ORT never consumes SDK packed/cast buffers.
            results=sessions[row['model_key']].run([x['name'] for x in model['outputs']],prepared); run_end=time.monotonic()
            require(len(results)==len(model['outputs']), 'ORT returned output count differs')
            values={x['name']:np.array(value,copy=True,order='C') for x,value in zip(model['outputs'],results)}; copy_end=time.monotonic()
            timing=dict(pack_seconds=pack_end-begin,run_seconds=run_end-pack_end,copy_seconds=copy_end-run_end,forward_seconds=copy_end-begin)
            report['calls'].append(save_call(out,len(report['calls']),row,values,model,original,wrapped,timing))
    finally: sessions.clear()


def run_sdk(a, pins, manifest, selected, packing, abi, out, report):
    require(a.model_key and a.compile_dir and a.compile_sha256 and a.compiler_helper and a.compiler_sha256
            and a.ort_dir and a.ort_sha256,'SDK requires explicit model/compile/compiler and candidate ORT evidence')
    model=manifest['models'][a.model_key]
    ort=json.loads(pins.pin(a.ort_dir/'run.json',a.ort_sha256).read_text())
    require(ort['schema']=='v29_Update_candidate_board_diagnostic' and ort['mode']=='ort'
            and ort['manifest_sha256']==a.manifest_sha256 and ort['capture_sha256']==CAPTURE_SHA and ort['export_sha256']==EXPORT_SHA
            and ort['execution_status']=='completed' and ort['sources_and_inputs_unchanged']
            and ort['source_pins']==ort['source_hashes_after'] and not ort['cleanup_errors'], 'Completed same-candidate CPU ORT evidence required')
    ort_states={(x['model_key'],x['case'],x['step']):x for x in ort['calls']}
    require(len(ort_states)==len(ort['calls']), 'Duplicate candidate ORT saved state')
    planned=selected[:1] if a.mode=='sdk-profile' else selected
    require({(x['model_key'],x['case'],x['step']) for x in planned} <= set(ort_states),'ORT did not exercise every requested saved state')
    require(ort['candidate_accepted'] or a.allow_unvalidated_ort_diagnostic,'Unaccepted ORT requires explicit diagnostic permission flag')
    report['ORT_candidate_accepted']=ort['candidate_accepted'];report['explicit_unvalidated_ORT_diagnostic']=a.allow_unvalidated_ort_diagnostic
    compiler=json.loads(pins.pin(a.compile_dir/'compile.json',a.compile_sha256).read_text())
    pins.pin(a.compiler_helper,a.compiler_sha256)
    require(compiler['status']=='compiled_not_runtime_validated' and compiler['script_sha256']==a.compiler_sha256
            and compiler['input']['sha256']==model['sha256'] and compiler['released'] and not compiler.get('release_error'),'Compile provenance differs')
    config=compiler['configuration']
    require(all(config.get(k)==v for k,v in dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False).items()),'Expected original nonquantized FP16 compiler configuration')
    require([x['name'] for x in compiler['operations']]==['config','load_onnx','build','export_rknn']
            and all(x['code'] in (0,None) for x in compiler['operations']),'Compiler stage failure')
    path=pins.pin(a.compile_dir/Path(compiler['output']['path']).name,compiler['output']['sha256'])
    require(path.stat().st_size==compiler['output']['size'],'RKNN bytes differ')
    for file,digest in ((a.header,HEADER_SHA),(a.library,LIB_SHA)):pins.pin(file,digest)
    api=None; context=abi.CONTEXT(); log=out/'sdk-native.log'
    try:
        with native_log(log):
            try:
                api=abi.bind_api(ct.CDLL(str(a.library)));data=path.read_bytes();buffer=ct.create_string_buffer(data)
                abi.check_call(api.rknn_init(ct.byref(context),ct.cast(buffer,ct.c_void_p),len(data),8 if a.mode=='sdk-profile' else 0,None),'rknn_init')
                abi.check_call(api.rknn_set_core_mask(context,7),'rknn_set_core_mask')
                version=abi.query(api,context,5,abi.SDKVersion()); count=abi.query(api,context,0,abi.InputOutputNum())
                require(count.n_input==13 and count.n_output==len(model['outputs']),'SDK I/O count differs')
                raw_in=[abi.attr_record(abi.query(api,context,1,abi.TensorAttr(index=i))) for i in range(13)]
                raw_out=[abi.attr_record(abi.query(api,context,2,abi.TensorAttr(index=i))) for i in range(count.n_output)]
                report.update(input_attributes=raw_in,output_attributes=raw_out,versions=dict(api=bytes(version.api_version).decode(),driver=bytes(version.drv_version).decode()),abi=abi.abi_record(),core_mask=7,init_flags=8 if a.mode=='sdk-profile' else 0)
                require(report['versions']['api'].startswith('2.3.2'),'Pinned SDK 2.3.2 required')
                bindings=packing.validate_input_attrs(model['inputs'],raw_in);outputs_bound=output_bindings(packing,model['outputs'],raw_out)
                memory=abi.query(api,context,6,abi.MemSize());report['memory']={k:int(getattr(memory,k)) for k in ('total_dma_allocated_size','total_weight_size','total_internal_size')}
                schedule=planned if a.mode=='sdk-profile' or a.first_only else [planned[0]]*4+planned[1:]
                seen={}
                for row in schedule:
                    feed,original,wrapped,_=load_feed(row,a,pins,packing,model,abi)
                    candidate_ort=pins.artifact(a.ort_dir,ort_states[(row['model_key'],row['case'],row['step'])]['outputs'])
                    require(tuple(candidate_ort)==tuple(x['name'] for x in model['outputs']), 'Candidate ORT saved outputs differ')
                    values,timing=execute_sdk(api,context,abi,packing,feed,bindings,outputs_bound,report['cleanup_errors'])
                    call=save_call(out,len(report['calls']),row,values,model,original,wrapped,timing)
                    call['versus_candidate_ORT']={name:compare(value,candidate_ort[name]) for name,value in values.items()}
                    call['candidate_ORT_strict_FP32_gate_passed']=all(x['gate_passed'] for x in call['versus_candidate_ORT'].values())
                    key=(row['case'],row['step']);metadata=call['outputs']['arrays'];call['repeat_bitwise_equal']=key not in seen or seen[key]==metadata;seen[key]=metadata
                    report['calls'].append(call)
                    if a.mode=='sdk-profile':call['profile']=abi.perf_detail(api,context)
            finally:
                if context.value:
                    try:
                        code=int(api.rknn_destroy(context));report['destroy_code']=code
                        if code:report['cleanup_errors'].append('rknn_destroy '+str(code))
                    except Exception as error:report['cleanup_errors'].append(repr(error))
    finally:
        if log.exists():report['native_log']=dict(file=log.name,sha256=sha(log),bytes=log.stat().st_size,errors=native_errors(log))
    require(not report.get('native_log',{}).get('errors'),'Native E RKNN log reported an internal failure')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('ort','sdk-normal','sdk-profile'),required=True)
    for name in ('capture-dir','export-dir','manifest','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--model-key',choices=('E96','E864'))
    p.add_argument('--first-only',action='store_true');p.add_argument('--allow-unvalidated-ort-diagnostic',action='store_true')
    for name in ('compile-dir','compiler-helper','ort-dir'):p.add_argument('--'+name,type=Path)
    for name in ('compile','compiler','ort'):p.add_argument('--'+name+'-sha256')
    p.add_argument('--packing-helper',type=Path,default=Path(__file__).with_name('v28_rknn_inputs.py'))
    p.add_argument('--abi-helper',type=Path,default=BASE/'src/deploy/rk3588/validation/run-v12-rknn-block-board.py')
    p.add_argument('--header',type=Path,default=BASE/'src/deploy/rk3588/reference/rknn_api_v2.3.2.h')
    p.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'));a=p.parse_args(argv)
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board only')
    out=a.output_dir.resolve();require(out.is_relative_to(BASE/'experiments'),'New isolated board output required');out.mkdir(exist_ok=False)
    pins=Inputs();prior=os.sched_getaffinity(0)
    report=dict(schema='v29_Update_candidate_board_diagnostic',status='execution_failed',execution_status='failed',numerical_status='not_evaluated',
                mode=a.mode,script_sha256=sha(__file__),manifest_sha256=a.manifest_sha256,capture_sha256=CAPTURE_SHA,export_sha256=EXPORT_SHA,
                candidate_accepted=False,actual_board_execution=True,CPU_model_fallback=False,production_changed=False,BA_executed=False,ORB_used=False,
                valid_pose_output=False,realtime_validated=False,FP32_thresholds_relaxed=False,outputs_clipped_or_modified=False,calls=[],cleanup_errors=[])
    try:
        pins.pin(__file__,report['script_sha256']);packing=module(pins.pin(a.packing_helper,PACKING_SHA),'_v29_frozen_packing')
        abi=module(pins.pin(a.abi_helper,ABI_SHA),'_v29_frozen_abi')
        capture,exported,manifest=load_chain(a,pins);selected=records(capture,exported,manifest,a.model_key,a.first_only)
        report.update(selected_saved_states=len(selected),first_only=a.first_only,model_keys=sorted({x['model_key'] for x in selected}),manifest_model_roles={k:v['role'] for k,v in manifest['models'].items()})
        require({4,5,6,7} <= prior, 'Required CPU4..7 unavailable in inherited affinity')
        os.sched_setaffinity(0,{4,5,6,7});report['cpu_affinity']=sorted(os.sched_getaffinity(0))
        require(report['cpu_affinity']==[4,5,6,7], 'CPU affinity setting failed')
        if a.mode=='ort':run_ort(a,pins,manifest,selected,exported,packing,abi,out,report)
        else:run_sdk(a,pins,manifest,selected,packing,abi,out,report)
        require(report['calls'],'No saved output calls')
        accepted=all(c['all_outputs_finite'] and c['weight_domain_passed'] and c['core_strict_FP32_gate_passed'] and c.get('repeat_bitwise_equal',True)
                     and c.get('candidate_ORT_strict_FP32_gate_passed',True) for c in report['calls'])
        if a.mode!='ort':accepted=accepted and report['ORT_candidate_accepted']
        report.update(execution_status='completed',candidate_accepted=accepted,numerical_status='accepted' if accepted else 'rejected',status='completed_numerical_accepted' if accepted else 'completed_numerical_rejected')
    except Exception as error:report.update(status='execution_failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        try:
            report['source_hashes_after']={path:sha(path) for path in pins.pins};report['sources_and_inputs_unchanged']=report['source_hashes_after']==pins.pins
            require(report['sources_and_inputs_unchanged'],'Source or input changed')
        except Exception as error:report['cleanup_errors'].append(repr(error))
        try:
            os.sched_setaffinity(0,prior);report['affinity_restored']=set(os.sched_getaffinity(0))==set(prior)
            require(report['affinity_restored'],'CPU affinity restoration failed')
        except Exception as error:report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']:
            report.update(status='execution_failed',execution_status='failed',candidate_accepted=False)
        report.update(source_pins=pins.pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      timing_scope='Owned input preparation, synchronous execution/get, owned output copy and SDK release. Hashing, finite/gate statistics, artifact I/O, graph initialization, profile querying and topology loading excluded. Profile is a separate process/context.')
        (out/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','execution_status','numerical_status','candidate_accepted','error','cleanup_errors')}),flush=True)
    return 1 if report['execution_status']=='failed' else 0 if report['candidate_accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
