#!/usr/bin/env python3
"""Board-only representative native518 global0 inputs and same-input CPU refs.

The unchanged original24-layer NPU DINO and original standalone frame0 B1
precede every captured input. This is a component data capture, not a full
model or performance result. The standalone frame0 compiler boundary is not
certified byte-identical to the internal deployed retained-group boundary.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

BASE=Path('/mnt/vggt-data/vggt-slam2'); E=BASE/'experiments'; SRC=BASE/'src'
S100=E/'vggt-v100-source-r1'
CHECKPOINT=dict(bytes=5026874952,sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
CONFIG_SHA='7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
PREFIX_SHA='cbb80f6559ed8856f05382899b7e1025b2637f08108ae6c35abb9d6954a78b07'
OWNED_SHA='0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
CAL_SHA='a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
QUALITY_SHA='37a7856a6037fe16b29dc601dcf488c12a513d457dcdca8bb2e747ed43f1f77c'
NATIVE_SHA='4ebbc4adaaed5e3c3cf8558b4947ec3fc090d56e82f8c61c9509951eb34dd35d'
WEIGHTS_SHA='4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
FRAME_SHA='4686e3b30225f6d47c556551e06ce1e30a62b27bbfcde42dbd122a467cb067ab'
HELPER_SHA='56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
SELECTED=(0,1,2,3,5,6); FRAME=(2,1041,1024); GLOBAL=(1,2082,1024)
LIMIT=70000000; SUCCESS='completed_original_native518_global0_input_capture'
COUNTERS=('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()


def atomic(path,row):
    t=path.with_suffix(path.suffix+'.tmp'); t.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n'); t.replace(path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=SRC)
    p.add_argument('--frozen-source-dir',type=Path,default=S100)
    p.add_argument('--config',type=Path,default=S100/'full-native-sequential-config-r1.json')
    p.add_argument('--calibration-metadata',type=Path,default=E/'vggt-v56-calibration-r1/capture.json')
    p.add_argument('--quality-receipt',type=Path,required=True)
    p.add_argument('--native-reference',type=Path,default=E/'vggt-v98-reference-r1/capture.json')
    p.add_argument('--weights-capture',type=Path,default=E/'vggt-v53-capture-r1/capture.json')
    p.add_argument('--frame-compile',type=Path,default=E/'vggt-v98-frame-b1-compile-r1/result.json')
    p.add_argument('--runtime-helper',type=Path,default=E/'vggt-v99-source-r1/run_native_block_r1.py')
    p.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=float,default=170.)
    p.add_argument('--skip-reference-control',action='store_true')
    a=p.parse_args(); sys.dont_write_bytecode=True
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'NN/SDK execution is RK3588/S only')
    require(0<a.timeout_seconds<=170.,'Use external180s/75C/restoring1GHz controller')
    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False); result=out/'result.json'
    start=time.monotonic(); deadline=start+a.timeout_seconds
    oldenv,oldaff,oldpath=dict(os.environ),os.sched_getaffinity(0),sys.path[:]
    pins={}; owners=[]; handlers={}; prefix=block=None; torch=np=None; prefix_seal=block_seal=None
    r=dict(schema='v115_original_native518_global0_representative_capture',status='running',script_sha256=sha(__file__),
        checkpoint=CHECKPOINT,original_model_only=True,actual_execution_on_board=True,component_capture_only=True,
        full_model_forward_executed=False,performance_measurement=False,candidate_accepted=False,
        original_image_size=518,image_shape=[2,3,392,518],logical_shape=list(GLOBAL),
        upstream_route='Original FP16 NPU DINO24 -> original finalnorm/special-role tokens -> standalone original frame0 B1 -> global reshape',
        standalone_frame0_compiled_boundary=True,deployed_internal_frame0_byte_identity_established=False,
        original_CPU_global0_on_exact_NPU_derived_input=True,CPU_DINO_blocks_or_CPU_frame0_executed=False,
        expected_NPU_hz=1000000000,core_mask=7,init_flags=0,maximum_live_weighted_NPU_contexts=1,
        expected_SDK_totals=dict(contexts_created=5,contexts_destroyed=5,runs_completed=54,output_buffers_acquired=54,output_buffers_released=54),
        original_CPU_global0_forward_count=0,reference_control_requested=not a.skip_reference_control,
        original_CPU_reference_control_count=0,calibration_ids=['calib00','calib01','calib02','calib03'],
        evaluation_ids=['heldout00','heldout01'],records=[],platform_observations=[],cleanup_errors=[],
        storage_limit_bytes=LIMIT,CPU_calibration_reference_payloads_saved=False,fullmodel_quality_or_realtime_accepted=False)
    atomic(result,r)

    def pin(path,digest,size=None):
        path=Path(path).resolve(strict=True)
        require((size is None or path.stat().st_size==size) and sha(path)==digest,'Pinned source/artifact differs: '+str(path))
        pins[str(path)]=digest; return path

    def observe(stage):
        require(time.monotonic()<deadline,'Capture deadline reached')
        z=H.sample(); require(z['npu_hz']==1000000000 and z['temperatures_c'] and max(z['temperatures_c'].values())<75.,'Clock/thermal bound failed')
        r['platform_observations'].append(dict(stage=stage,**z))

    def metadata(value):
        require(value.dtype==np.float32 and value.flags.c_contiguous and np.isfinite(value).all(),'Finite contiguous FP32 tensor required')
        return dict(shape=list(value.shape),dtype='float32',tensor_bytes=value.nbytes,finite=True,
            raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(),minimum=float(value.min()),maximum=float(value.max()),
            l2_norm=float(np.linalg.norm(value.astype(np.float64))))

    def save(name,value):
        row=metadata(value); path=out/(name+'.npy'); temporary=path.with_suffix('.npy.tmp')
        require(sum(f.stat().st_size for f in out.iterdir() if f.is_file())+value.nbytes+4096<LIMIT,'New capture storage limit reached')
        with temporary.open('xb') as f: np.save(f,value,allow_pickle=False)
        temporary.replace(path); row.update(file=path.name,bytes=path.stat().st_size,sha256=sha(path)); pins[str(path)]=row['sha256']; return row

    def load(folder,row,shape,dtype):
        path=pin(folder/row['file'],row['sha256'],row['bytes']); value=np.load(path,allow_pickle=False)
        require(value.shape==shape and value.dtype==dtype and value.flags.c_contiguous and np.isfinite(value).all()
            and hashlib.sha256(value.tobytes()).hexdigest()==row['raw_sha256'],'Pinned captured array differs')
        return value

    def state(model):
        return [(name,id(v),v.data_ptr(),v._version,str(v.dtype),str(v.device)) for name,v in model.named_parameters()]

    def context(row,shape):
        owner=None
        try:
            owner=O.OwnedContext(row['file'],row['sha256'],input_shapes=[shape],output_shapes=[shape],
                input_names=['input'],output_names=['output'],core_mask=7,flags=0,source_root=a.repo,
                library=a.library,runtime_helper=a.runtime_helper,timeout_ms=30000)
        except BaseException as exc:
            owner=getattr(exc,'owned_context',None)
            if owner is not None: owners.append(owner)
            raise
        owners.append(owner); return owner

    try:
        def interrupted(number,frame): raise InterruptedError('External capture controller stopped worker')
        for number in (signal.SIGTERM,signal.SIGINT): handlers[number]=signal.signal(number,interrupted)
        os.sched_setaffinity(0,{4,5,6,7}); os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        pin(__file__,r['script_sha256']); config=json.loads(pin(a.config,CONFIG_SHA).read_text())
        P=module(pin(a.frozen_source_dir/'native_original_prefix.py',PREFIX_SHA),'_v115_original_prefix')
        O=module(pin(a.frozen_source_dir/'native_heads_runtime.py',OWNED_SHA),'_v115_frozen_owned_context')
        H=module(pin(a.runtime_helper,HELPER_SHA),'_v115_frozen_platform_utils')
        pin(a.library,O.SDK_SHA); pin(a.repo/'deploy/rk3588/validation/run-v12-rknn-block-board.py',O.ABI_SHA)
        cp=pin(a.calibration_metadata,CAL_SHA); cal=json.loads(cp.read_text())
        qp=pin(a.quality_receipt,QUALITY_SHA); quality=json.loads(qp.read_text())
        rp=pin(a.native_reference,NATIVE_SHA); native=json.loads(rp.read_text())
        wp=pin(a.weights_capture,WEIGHTS_SHA); weights_capture=json.loads(wp.read_text())
        fp=pin(a.frame_compile,FRAME_SHA); frame=json.loads(fp.read_text())
        require(cal['checkpoint']==native['checkpoint']==quality['checkpoint']==weights_capture['checkpoint']==CHECKPOINT,
            'Original checkpoint identities differ')
        require(cal['status']=='completed_original_FP32_calibration_capture' and not cal['cleanup_errors']
            and quality['status']=='completed_original_native518_host_FP32_prediction_capture'
            and native['status']=='completed_original_native518_block_references','Completed source provenance required')
        selected=[cal['samples'][i] for i in SELECTED]
        expected=[[0,48],[96,144],[192,240],[288,336],[480,528],[576,624]]
        require([s['indices'] for s in selected]==expected,'Exact six predeclared source pairs required')
        excluded={v['sha256'] for v in quality['source_images']}|{v['sha256'] for v in native['source_images']}
        calpng={v['sha256'] for s in selected[:4] for v in s['images']}; evalpng={v['sha256'] for s in selected[4:] for v in s['images']}
        require(len(calpng)==8 and len(evalpng)==4 and not calpng&evalpng and not (calpng|evalpng)&excluded,'Source PNG partitions overlap')
        r.update(calibration_evaluation_disjoint=True,quality_and_fixed_PNG_disjoint=True,heldout_not_in_calibration=True,
            source_hashes=native['source_hashes'],source_PNG_metadata_sha256=CAL_SHA,quality_reference_sha256=QUALITY_SHA,
            native_reference_sha256=NATIVE_SHA,weights_capture_sha256=WEIGHTS_SHA,original_positions=native['artifacts']['global_positions'])
        for name,digest in native['source_hashes'].items(): pin(a.repo/name,digest)
        dino=[]
        for i,row in enumerate(config['dino']):
            manifest=pin(row['file'],row['sha256']); d=json.loads(manifest.read_text())
            require(d['schema']=='original_native518_single_block_FP16_compile' and d['status']=='compiled_not_quality_accepted'
                and d['role']=='dino' and d['logical_shape']==[1,1041,1024] and not d['configuration']['model_pruning']
                and not d['build_options']['do_quantization'] and d['sdk_released'] and not d['cleanup_errors'],'Original DINO compilation required')
            ep=Path(d['onnx']['file']).parent/'export.json'; export=json.loads(ep.read_text())
            require(export['group_index']==i and export['indices']==list(range(6*i,6*i+6)) and export['graph']['sha256']==d['onnx']['sha256']
                and export['proof']['original_parameter_count']==84 and export['proof']['namespace_inverse_protobuf_checks_passed']
                and export['source_and_inputs_unchanged'],'All original DINO layers in order required')
            pin(ep,sha(ep)); model=pin(manifest.parent/d['rknn']['file'],d['rknn']['sha256'],d['rknn']['bytes'])
            dino.append(dict(file=str(model),sha256=d['rknn']['sha256'],bytes=d['rknn']['bytes'],indices=export['indices'],compile_sha256=row['sha256']))
        require(len(dino)==4 and frame['role']=='frame' and frame['logical_shape']==list(FRAME)
            and frame['onnx']['sha256']=='683b4ec52a6d4692fa777b2180a143de347b4765edded313717946c5a0a450f9'
            and not frame['build_options']['do_quantization'] and frame['sdk_released'] and not frame['cleanup_errors'],'Frozen complete original upstream models required')
        fm=pin(fp.parent/frame['rknn']['file'],frame['rknn']['sha256'],frame['rknn']['bytes'])
        frame_row=dict(file=str(fm),sha256=frame['rknn']['sha256'],bytes=frame['rknn']['bytes'])
        r['upstream_models']=dict(DINO=dino,frame0=dict(frame_row,compile_sha256=FRAME_SHA,original_parameters=18))
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(torch.__version__==native['versions']['torch'],'Same original reference Torch implementation required')
        r['versions']=dict(torch=torch.__version__,numpy=np.__version__,threads=1,interop_threads=1)
        observe('prefix_initialization'); prefix=P.OriginalPrefix(repo=a.repo); prefix_seal=state(prefix.dino)
        pins.update(prefix.pins); r['checkpoint_verified_before_deserialization']=True
        prepared=[]
        for i,sample in enumerate(selected):
            paths=[pin(v['file'],v['sha256'],v['bytes']) for v in sample['images']]; observe('prepare_'+str(i))
            with torch.inference_mode():
                pixels=prefix.preprocess([str(path) for path in paths],image_size=518,mode='crop')
                require(tuple(pixels.shape)==(2,3,392,518),'Reject changed native image shape')
                owned_pixels=np.array(pixels.numpy(),copy=True,order='C'); pixel_meta=metadata(owned_pixels)
                normalized=((pixels.unsqueeze(0)-prefix.mean)/prefix.std).view(2,3,392,518)
                values=[prefix.dino.prepare_tokens_with_masks(normalized[j:j+1].contiguous(),masks=None) for j in range(2)]
                require(all(tuple(v.shape)==(1,1041,1024) for v in values),'All native DINO tokens required')
                prepared.append([np.array(v.numpy(),copy=True,order='C') for v in values])
            r['records'].append(dict(id=r['calibration_ids'][i] if i<4 else r['evaluation_ids'][i-4],
                role='calibration' if i<4 else 'heldout',source_pair_indices=sample['indices'],source_images=sample['images'],
                preprocessed_pixels=pixel_meta,prepared_frame_raw_sha256=[hashlib.sha256(owned_pixels[j].tobytes()).hexdigest() for j in range(2)]))
            del pixels,normalized,values,owned_pixels
        encoded=prepared; prepared=None
        for group,row in enumerate(dino):
            observe('DINO_group_'+str(group)); owner=context(row,(1,1041,1024))
            try:
                for i,frames in enumerate(encoded):
                    for j in range(2):
                        observe(f'DINO{group}_pair{i}_frame{j}'); frames[j]=owner.run([frames[j]])[0]
                        require(frames[j].shape==(1,1041,1024) and np.isfinite(frames[j]).all(),'Actual complete DINO group output required')
            finally: owner.close()
            atomic(result,r)
        observe('frame0_initialization'); owner=context(frame_row,FRAME)
        try:
            for i,frames in enumerate(encoded):
                observe('frame0_pair_'+str(i)); original_tokens=prefix.finish_dino(frames)
                actual=owner.run([original_tokens])[0]; require(actual.shape==FRAME,'Complete two-frame original frame0 output required')
                value=np.array(actual.reshape(GLOBAL),copy=True,order='C'); r['records'][i]['input']=save(r['records'][i]['id']+'-input',value)
                r['records'][i]['frame0_output_raw_sha256']=hashlib.sha256(actual.tobytes()).hexdigest()
                require(r['records'][i]['frame0_output_raw_sha256']==r['records'][i]['input']['raw_sha256'],'Only original global reshape allowed')
                del original_tokens,actual,value
        finally: owner.close()
        encoded=None
        require(all(v.metrics()['closed'] for v in owners),'All NPU contexts must close before CPU reference work')
        prefix.verify_unchanged(); require(state(prefix.dino)==prefix_seal,'Original DINO state changed')
        from vggt.layers.block import Block
        from vggt.layers.rope import RotaryPositionEmbedding2D
        row=next(v for v in weights_capture['blocks'] if v['role']=='global' and v['index']==0)
        wf=pin(wp.parent/row['weights']['file'],row['weights']['sha256'],row['weights']['bytes'])
        with np.load(wf,allow_pickle=False) as archive: weights={name:np.array(archive[name],copy=True,order='C') for name in archive.files}
        require(len(weights)==18 and set(weights)==set(row['parameters']),'All original global0 parameter keys required')
        for name,value in weights.items():
            meta=row['parameters'][name]; require(value.dtype==np.float32 and list(value.shape)==meta['shape'] and np.isfinite(value).all()
                and hashlib.sha256(value.tobytes()).hexdigest()==meta['raw_sha256'],'Original global0 parameter differs: '+name)
        rope=RotaryPositionEmbedding2D(frequency=row['rope_frequency'],scaling_factor=row['rope_scaling_factor'])
        with torch.device('meta'): block=Block(dim=1024,num_heads=16,mlp_ratio=4,qkv_bias=True,proj_bias=True,ffn_bias=True,init_values=.01,qk_norm=True,rope=rope)
        block.load_state_dict({name:torch.from_numpy(value) for name,value in weights.items()},strict=True,assign=True); block.eval().requires_grad_(False)
        require({name:m.eps for name,m in block.named_modules() if isinstance(m,torch.nn.LayerNorm)}==row['epsilons']
            and block.mlp.act.approximate==row['gelu_approximate']=='none' and block.attn.fused_attn
            and block.attn.num_heads==16 and block.attn.head_dim==64 and block.attn.scale==.125 and block.attn.attn_drop.p==0,
            'Original global0 architecture and math required')
        require(sha(inspect.getfile(Block))==native['source_hashes']['third_party/vggt/vggt/layers/block.py'],'Original Block source differs')
        block_seal=state(block); r['original_parameters']=copy.deepcopy(row['parameters']); r['original_global0_weights']=dict(absolute_file=str(wf),**row['weights'])
        pos=load(rp.parent,native['artifacts']['global_positions'],(1,2082,2),np.int64); tp=torch.from_numpy(pos)
        if not a.skip_reference_control:
            observe('original_fixed_control'); x=load(rp.parent,native['artifacts']['global0_input'],GLOBAL,np.float32)
            expected=load(rp.parent,native['artifacts']['global0_output'],GLOBAL,np.float32)
            with torch.inference_mode(): value=np.array(block(torch.from_numpy(x),pos=tp).numpy(),copy=True,order='C')
            require(value.tobytes()==expected.tobytes(),'Original global0 CPU control must remain byte-identical')
            r['original_CPU_reference_control_count']=1; r['original_CPU_reference_control_byte_equal']=True
            del x,expected,value
        for record in r['records']:
            observe('original_CPU_'+record['id']); x=load(out,record['input'],GLOBAL,np.float32); before=record['input']['raw_sha256']
            with torch.inference_mode(): value=np.array(block(torch.from_numpy(x),pos=tp).numpy(),copy=True,order='C')
            require(value.shape==GLOBAL and hashlib.sha256(x.tobytes()).hexdigest()==before,'Same-input full original CPU reference required')
            record['reference']=save(record['id']+'-original-output',value) if record['role']=='heldout' else metadata(value)
            record['reference']['saved']=record['role']=='heldout'; r['original_CPU_global0_forward_count']+=1
            del x,value; atomic(result,r)
        require(state(block)==block_seal and state(prefix.dino)==prefix_seal,'Original parameters changed')
        for name,value in block.named_parameters(): require(hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()==row['parameters'][name]['raw_sha256'],'Original learned bytes changed')
        cos,sin=rope.frequency_cache[(32,38,torch.device('cpu'),torch.float32)]
        require(np.array_equal(cos.numpy(),load(rp.parent,native['artifacts']['rope_cos'],(38,32),np.float32))
            and np.array_equal(sin.numpy(),load(rp.parent,native['artifacts']['rope_sin'],(38,32),np.float32)),'Original native RoPE tables differ')
        inputs={record['input']['raw_sha256'] for record in r['records'][:4]}; held={record['input']['raw_sha256'] for record in r['records'][4:]}
        require(len(inputs)==4 and len(held)==2 and not inputs&held,'Captured activation partitions overlap')
        dataset=out/'dataset.txt'; dataset.write_text(''.join(str(out/record['input']['file'])+'\n' for record in r['records'][:4]))
        r['dataset']=dict(file=dataset.name,bytes=dataset.stat().st_size,sha256=sha(dataset),line_count=4,input_count=1,paths_are_absolute=True)
        pins[str(dataset)]=r['dataset']['sha256']; r.update(status=SUCCESS,model_state_unchanged=True,
            hooks_removed=not any(m._forward_hooks or m._forward_pre_hooks for m in list(prefix.dino.modules())+list(block.modules())),
            original_positions_and_RoPE_tables_verified=True,all_capture_inputs_unique=True,
            new_array_files=8,new_array_bytes=8*8528000,reference_inputs_unchanged=True)
    except BaseException as exc:
        r.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        for owner in reversed(owners):
            try: owner.close()
            except BaseException as exc: r['cleanup_errors'].append('SDK close: '+repr(exc))
        r['resources']=[owner.metrics() for owner in owners]
        r['SDK_totals']={key:sum(v[key] for v in r['resources']) for key in COUNTERS}
        r['all_NPU_contexts_closed']=all(v['closed'] and not v['live_contexts'] and not v['live_output_buffers'] for v in r['resources'])
        r['resource_counts_balanced']=all(v['resource_counts_balanced'] and not v['cleanup_errors'] for v in r['resources'])
        r['complete_expected_execution']=r['SDK_totals']==r['expected_SDK_totals'] and r['original_CPU_global0_forward_count']==6
        try:
            if prefix is not None: prefix.verify_unchanged(); r['checkpoint_stat_unchanged']=True
        except BaseException as exc: r['cleanup_errors'].append('Original prefix verification: '+repr(exc))
        changed=[]
        for path,digest in pins.items():
            if prefix is not None and Path(path)==prefix.checkpoint: continue
            try: require(sha(path)==digest,'Pinned source/input changed: '+path)
            except BaseException as exc: changed.append(str(exc))
        r['pinned_files']=pins; r['source_and_inputs_unchanged']=not changed; r['changed_sources_or_inputs']=changed
        try: os.sched_setaffinity(0,oldaff); r['affinity_restored']=os.sched_getaffinity(0)==oldaff
        except BaseException as exc: r['cleanup_errors'].append('Affinity restore: '+repr(exc))
        os.environ.clear(); os.environ.update(oldenv); sys.path[:]=oldpath
        r.update(environment_restored=dict(os.environ)==oldenv,sys_path_restored=sys.path==oldpath,
            elapsed_seconds=time.monotonic()-start,peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        for number,handler in handlers.items(): signal.signal(number,handler)
        if (changed or r['cleanup_errors'] or not r['all_NPU_contexts_closed'] or not r['resource_counts_balanced']
            or not r.get('affinity_restored') or not r['environment_restored'] or not r['sys_path_restored']
            or (r['status']==SUCCESS and not (r['complete_expected_execution'] and r.get('hooks_removed')
                and r.get('model_state_unchanged') and r.get('checkpoint_stat_unchanged')))): r['status']='failed_cleanup'
        for _ in range(4):
            atomic(result,r); size=sum(f.stat().st_size for f in out.iterdir() if f.is_file())
            if size>=LIMIT: r['status']='failed_storage_limit'
            if r.get('new_capture_bytes')==size: break
            r['new_capture_bytes']=size
        atomic(result,r); print(json.dumps({key:r.get(key) for key in ('status','SDK_totals','original_CPU_global0_forward_count','new_capture_bytes','cleanup_errors')}),flush=True)
    return 0 if r['status']==SUCCESS else 1


if __name__=='__main__': raise SystemExit(main())
