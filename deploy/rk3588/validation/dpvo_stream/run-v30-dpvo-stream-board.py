#!/usr/bin/env python3
"""RK3588-only unpaced fresh-observation DPVO CPU research replay.

Original neural weights, CPU ORT encoders, original Torch Update, native CPU
correlation and the v29 vectorized online BA equations. No ORB/LK/PnP, GT input,
NPU, interpolation-based fresh poses or real-time acceptance. DPVO-derived
lifecycle/math retains Princeton Vision & Learning Lab MIT attribution.
"""
import argparse,gc,hashlib,importlib.util,json,os,platform,resource,sys,time,traceback
from pathlib import Path
import numpy as np
BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
PINS={
 'dpvo_probe_common.py':'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9',
 'dpvo_cpu_ops.py':'15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32',
 'manifest.json':'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375'}
CHECKPOINT='30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
GEOMETRY='c893f12d188baced438504cfc592a8900e18e40ef8223af0a7e944e4484b0c3b'
NATIVE='a6c37bf4b840b492736d5c5916be6e2c0e8680693d01d0fe3a5fe62702ecb779'
LIBRARY='829d1b42fa24d28b1e280b882c0a9e8556b0e7cdbe02f7489eeef15ba3581bca'
BUILD='34fbf049852b351e7df1097cc806fbf83e1e1035e18b01ca4e9e8ba39499cd18'
RGB='532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3'
CAPTURE='e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088'
def require(ok,why):
    if not ok:raise ValueError(why)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()
def meta(a):return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def require_board():
    require(platform.machine()=='aarch64' and platform.system()=='Linux','Actual model execution restricted to board')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 device tree required')
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

class CompactCorrelation:
    """Owned native cache of only referenced immutable source/target features."""
    def __init__(self,factory,library,patch_count=96):
        self.factory=factory;self.library=library;self.M=patch_count;self.cache=None;self.key=None
        self.calls=[];self.generations=[];self.closed=[];self.attempted=0
    def __call__(self,coords,*,ii,jj,kk,frames):
        begin=time.monotonic();self.attempted+=1
        require(all(isinstance(a,np.ndarray) and a.dtype==np.int64 and a.ndim==1 for a in (ii,jj,kk)),'Int64 edge IDs required')
        require(ii.shape==jj.shape==kk.shape and len(kk)>0,'Nonempty equal edge arrays required')
        require(np.array_equal(ii,kk//self.M) and min(ii.min(),jj.min())>=0 and max(ii.max(),jj.max())<len(frames),'Original source ownership/ranges differ')
        sources=np.unique(ii);targets=np.unique(jj)
        key=json.dumps({'source':[frames[i]['frame_id'] for i in sources],'target':[frames[j]['frame_id'] for j in targets],'patches_per_frame':self.M},separators=(',',':'))
        if key!=self.key:
            self.close_current()
            source_arrays=[frames[i]['gmap'] for i in sources];target_arrays=[frames[j]['fmap'] for j in targets]
            require(all(a.dtype==np.float32 and not a.flags.writeable for a in source_arrays+target_arrays),'Features must be immutable FP32 generation-owned arrays')
            gmap=np.ascontiguousarray(np.concatenate(source_arrays,axis=0)[None]);fmap=np.ascontiguousarray(np.stack(target_arrays,axis=0)[None])
            require(gmap.shape[1:]==(len(sources)*self.M,128,3,3) and fmap.shape[2]==128,'Feature contract differs')
            self.cache=self.factory(self.library,gmap,fmap,threads=4,cache_key=key,library_sha256=LIBRARY);self.key=key
            self.generations.append(dict(index=len(self.generations),key=key,source_frame_ids=[frames[i]['frame_id'] for i in sources],target_frame_ids=[frames[j]['frame_id'] for j in targets],preparation=self.cache.preparation))
        local_kk=np.ascontiguousarray(np.searchsorted(sources,ii)*self.M+kk%self.M,dtype=np.int64)
        local_jj=np.ascontiguousarray(np.searchsorted(targets,jj),dtype=np.int64)
        out,timing=self.cache.correlate(coords,local_kk,local_jj,cache_key=key)
        require(out.shape==(1,len(kk),882) and out.dtype==np.float32 and np.isfinite(out).all(),'Actual native output invalid')
        self.calls.append(dict(index=len(self.calls),edges=len(kk),generation=len(self.generations)-1,total_seconds=time.monotonic()-begin,native_query=timing))
        return out
    def close_current(self):
        if self.cache is not None:
            cache=self.cache;self.cache=None;self.key=None;cache.close();m=cache.metrics()
            require(m['closed'] and m['current_context_bytes']==0,'Native cache not released')
            self.closed.append(m)
    def close(self):self.close_current()
    def metrics(self):return dict(attempted_calls=self.attempted,completed_calls=len(self.calls),generations=self.generations,calls=self.calls,closed_contexts=self.closed,current_context_open=self.cache is not None)

class TorchDepthRNG:
    def __init__(self,torch,generator):self.torch=torch;self.generator=generator;self.calls=0
    def random(self,size,dtype=np.float32):
        require(np.dtype(dtype)==np.float32,'Original random depth is FP32')
        shape=(size,) if isinstance(size,int) else tuple(size);self.calls+=1
        return self.torch.rand(shape,dtype=self.torch.float32,generator=self.generator).numpy()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','checkpoint','geometry','native-helper','library','build-receipt','encoder-model','encoder-export','encoder-ort','encoder-capture-dir','dataset-dir','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stream-module',type=Path,default=Path(__file__).with_name('dpvo_stream.py'))
    p.add_argument('--encoder-module',type=Path,default=Path(__file__).with_name('v30_frame_encoder.py'))
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--frames',type=int,default=192);p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--run-time-limit',type=float,default=1200)
    a=p.parse_args(argv);require_board();out=a.output_dir.resolve();require(out.is_relative_to(BASE),'Isolated board output required');out.mkdir(exist_ok=False)
    require(8<=a.frames<=798 and a.run_time_limit>0,'Bounded explicit sequence required')
    begin=time.monotonic();affinity=os.sched_getaffinity(0);pins={};encoder=correlator=neural=stream=None;weights_before=None
    report=dict(schema='v30_DPVO_fresh_frame_CPU_replay',status='failed',script_sha256=sha(__file__),actual_inference_on_RK3588S=True,original_VGGT_default_unchanged=True,different_model_from_VGGT=True,ORB_used=False,added_LK_PnP_frontend=False,NPU_used=False,GT_or_sensor_depth_input=False,unpaced_file_replay=True,physical_camera=False,realtime_validated=False,accuracy_vs_VGGT_validated=False,frames_requested=a.frames,frames_completed=0,frames=[],update_calls=[],cleanup_errors=[],seed=a.seed)
    def pin(path,digest):
        path=Path(path).resolve();require(sha(path)==digest,'Source/artifact SHA differs: '+str(path));pins[str(path)]=digest;return path
    def save(name,arrays):
        path=out/name;np.savez(path,**arrays);return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in arrays.items()})
    def save_snapshot(name):
        value=stream.snapshot();arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
        return dict(metadata=plain({k:v for k,v in value.items() if k not in arrays}),artifact=save(name,arrays))
    def artifact(folder,item,metadata=None):
        name=item['file'];require(Path(name).name==name,'Unsafe artifact basename');path=pin(folder/name,item['sha256'])
        require(path.stat().st_size==item['bytes'],'Saved capture artifact bytes differ')
        with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        require({k:meta(v) for k,v in values.items()}==(item['arrays'] if metadata is None else metadata),'Saved capture raw metadata differs')
        return values
    try:
        pin(__file__,report['script_sha256'])
        source_manifest=json.loads(a.source_manifest.read_text());require(source_manifest['schema']=='v30_stream_source_freeze','Frozen v30 source manifest required')
        for path in (a.stream_module,a.encoder_module):pin(path,source_manifest['sources'][path.name])
        pin(a.source_manifest,sha(a.source_manifest));pin(a.geometry,GEOMETRY);pin(a.native_helper,NATIVE);pin(a.library,LIBRARY);pin(a.build_receipt,BUILD)
        build=json.loads(a.build_receipt.read_text());require(build['status']=='compiled_and_synthetic_smoke_passed' and build['output']['sha256']==LIBRARY,'Native build chain differs')
        for name,digest in PINS.items():pin(a.source_dir/name,digest)
        source=json.loads((a.source_dir/'manifest.json').read_text())
        for row in source['retrieved_files']:
            if row.get('available_in_pinned_tree') is False:
                require('sha256' not in row and not (a.source_dir/'official'/row['path']).exists(),'Absent upstream source entry unexpectedly present')
                continue
            pin(a.source_dir/'official'/row['path'],row['sha256'])
        pin(a.checkpoint,CHECKPOINT);pin(a.dataset_dir/'rgb.txt',RGB);pin(a.encoder_capture_dir/'capture.json',CAPTURE)
        for path in (a.encoder_model,a.encoder_export,a.encoder_ort):pin(path,sha(path))
        selected=[]
        for line in (a.dataset_dir/'rgb.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'):continue
            t,name=line.split();path=(a.dataset_dir/name).resolve();require(path.is_relative_to(a.dataset_dir.resolve()) and name.startswith('rgb/'),'RGB path escaped dataset')
            digest=sha(path);pins[str(path)]=digest
            selected.append(dict(frame_id=t,timestamp=float(t),relative_path=name,bytes=path.stat().st_size,sha256=digest))
            if len(selected)==a.frames:break
        require(len(selected)==a.frames and all(np.isfinite(r['timestamp']) for r in selected) and all(b['timestamp']>x['timestamp'] for x,b in zip(selected,selected[1:])),'Full ordered requested source sequence required')
        manifest=dict(schema='v30_first_contiguous_TUM_RGB_frames',dataset='TUM RGB-D freiburg1_xyz',rgb_index_sha256=RGB,frames=selected,frame_selection='first N consecutive RGB index entries, stride one',GT_read=False)
        (out/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');report['input_manifest_sha256']=sha(out/'input-manifest.json')
        require({4,5,6,7}<=affinity,'Four A76 CPUs required');os.sched_setaffinity(0,{4,5,6,7})
        import torch,cv2
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        common=module(a.source_dir/'dpvo_probe_common.py','_v30_original_neural_loader');geo=module(a.geometry,'_v30_vector_geometry');native=module(a.native_helper,'_v30_native_corr')
        enc=module(a.encoder_module,'_v30_encoder');core=module(a.stream_module,'_v30_stream')
        report['online_lifecycle_source_sha256']=core.verify_sources(a.source_dir/'official')
        for name,digest in core.SOURCE_PINS.items():pin(a.source_dir/'official'/name,digest)
        start=time.monotonic();neural,loader=common.load_neural(a.source_dir/'official',a.source_dir/'manifest.json',a.checkpoint,CHECKPOINT,a.source_dir/'dpvo_cpu_ops.py',cpu_ops_sha256=PINS['dpvo_cpu_ops.py'])
        weights_before={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()};require(len(weights_before)==94,'Original complete weights required')
        captured=json.loads((a.encoder_capture_dir/'capture.json').read_text());require(weights_before==captured['state_before'],'Original parameters differ')
        report.update(neural_load_seconds=time.monotonic()-start,loader=loader,original_weights_before=weights_before)
        encoder=enc.BoardORTEncoder(a.encoder_model,a.encoder_export,a.encoder_ort)
        generator=torch.Generator(device='cpu');generator.manual_seed(a.seed);rng=TorchDepthRNG(torch,generator)
        report['random_generator_before']=meta(generator.get_state().numpy())
        correlator=CompactCorrelation(native.PreparedCorrelation,a.library)
        def update(net,ctx,corr,ii,jj,kk):
            start=time.monotonic();inputs=(net,ctx,corr,ii,jj,kk)
            require(all(isinstance(x,np.ndarray) and np.isfinite(x).all() for x in inputs),'Finite owned neural input required')
            with torch.inference_mode():
                n,(d,w,unused)=neural.update(torch.from_numpy(net),torch.from_numpy(ctx),torch.from_numpy(corr),None,torch.from_numpy(ii),torch.from_numpy(jj),torch.from_numpy(kk))
                require(unused is None,'Original Update auxiliary output differs')
                values={key:v.detach().numpy().copy() for key,v in [('net',n),('delta',d),('weight',w)]}
            require(all(np.isfinite(v).all() for v in values.values()) and np.all((values['weight']>=0)&(values['weight']<=1)),'Original Update finite/domain failed')
            report['update_calls'].append(dict(index=len(report['update_calls']),edges=len(kk),seconds=time.monotonic()-start,weight_min=float(values['weight'].min()),weight_max=float(values['weight'].max())))
            return values
        config=core.StreamConfig();stream=core.DPVOStream(config,geometry=geo,update=update,correlate=correlator,rng=rng)
        report['config']=plain(vars(config));report.update(cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,numpy_version=np.__version__,torch_version=torch.__version__,OpenCV_version=cv2.__version__,loop_closure=False,precision='CPU FP32, not original CUDA mixed-precision arithmetic',random_scope='Shared Torch CPU Generator: original x draw then y draw then per-patch depth. Reproducible CPU RNG, not CUDA RNG bit equivalence.')
        K=np.array([[517.3,0,318.6],[0,516.5,255.3],[0,0,1]],np.float64);D=np.array([.2624,-.9531,-.0054,.0026,1.1633],np.float64)
        intrinsics=np.array([517.3,516.5,302.6,247.3],np.float32)
        replay_start=time.monotonic();report['replay_begin_monotonic']=replay_start
        for index,item in enumerate(selected):
            require(time.monotonic()-replay_start<a.run_time_limit,'Predeclared replay time limit exceeded')
            row=dict(index=index,frame_id=item['frame_id'],timestamp=item['timestamp'],arrival_monotonic=time.monotonic())
            report['active_frame']=index;report['active_stage']='decode';bgr=cv2.imread(str(a.dataset_dir/item['relative_path']),cv2.IMREAD_COLOR)
            require(bgr is not None and bgr.dtype==np.uint8 and bgr.shape==(480,640,3),'Original BGR input shape differs')
            image=np.ascontiguousarray(cv2.undistort(bgr,K,D)[8:-8,16:-16]);decoded=time.monotonic()
            report['active_stage']='encoder';record=encoder.encode(image,generator=generator,patch_count=96)
            if index<3:
                original_input=captured['inputs'][index]
                require(original_input['frame_id']==item['frame_id'] and original_input['source']=={k:item[k] for k in ('frame_id','relative_path','bytes','sha256')},'First-three source image identity differs')
                require(meta(image.transpose(2,0,1))==original_input['cropped_BGR_uint8'],'First-three original rectification/crop bytes differ')
                normalized=enc.normalize_bgr(image)
                require(meta(normalized)['tensor_sha256']==original_input['array']['tensor_sha256'],'First-three original normalization bytes differ')
                row['original_input_raw_bytes_match']=True
                del normalized
                reference_row=next(r for r in captured['calls'] if r['frame_index']==index)
                refs=artifact(a.encoder_capture_dir,reference_row['artifact'],reference_row['arrays'])
                expected=enc.features_to_patches(refs['fmap'],refs['imap'],record['centers'])
                checks={}
                for key in ('fmap','gmap','context','patches'):
                    x,y=record[key],expected[key];require(x.shape==y.shape and x.dtype==y.dtype==np.float32,'First-three original reference shape differs')
                    diff=np.abs(x.astype(np.float64)-y.astype(np.float64));failures=int(np.count_nonzero(diff>1e-5+1e-4*np.abs(y.astype(np.float64))))
                    require(np.isfinite(x).all() and np.isfinite(y).all() and failures==0,'First-three encoder/patch sampler original FP32 gate failed')
                    checks[key]=dict(rtol=1e-4,atol=1e-5,failed_elements=failures,max_abs=float(diff.max()))
                row['first_three_original_capture_gate']=checks
                del refs,expected,diff
            encoded=time.monotonic();record.update(frame_id=item['frame_id'],timestamp=item['timestamp'],captured_monotonic=row['arrival_monotonic'],intrinsics=intrinsics.copy())
            report['active_stage']='stream';result=stream.process_encoded(record);processed=time.monotonic()
            row.update(decode_undistort_crop_seconds=decoded-row['arrival_monotonic'],encode_seconds=encoded-decoded,stream_seconds=processed-encoded,processing_seconds=processed-row['arrival_monotonic'],pose_available_monotonic=processed,result=plain(result))
            report['frames'].append(row);report['frames_completed']=index+1
            print(json.dumps(dict(event='frame_complete',index=index,frame_id=item['frame_id'],seconds=row['processing_seconds'],result=plain(result))),flush=True)
            (out/'progress.json').write_text(json.dumps(dict(frames_completed=index+1,current_frame=item['frame_id'],seconds=processed-replay_start),indent=2)+'\n')
        report.update(replay_elapsed_seconds=time.monotonic()-replay_start,metrics_before_termination=plain(stream.metrics()),snapshot_before_termination=save_snapshot('before-termination.npz'))
        report['active_stage']='termination_offline';start=time.monotonic();terminated=stream.terminate();report.update(termination_seconds=time.monotonic()-start,termination=plain(terminated),stream_metrics=plain(stream.metrics()),final_snapshot=save_snapshot('final-state.npz'),random_generator_after=meta(generator.get_state().numpy()),random_depth_draw_calls=rng.calls)
        report.pop('active_frame',None);report.pop('active_stage',None)
        report.update(status='completed_unpaced_fresh_frame_CPU_VO_research_not_realtime',frame_processing_seconds=[r['processing_seconds'] for r in report['frames']],timing_scope='Serial unpaced file replay. Per-frame time includes decoding, original undistort/crop, encoder input conversion/execution/patch extraction and learned stream processing/cache rebuild/BA. First three frames additionally include saved original encoder/patch-array validation. Model loading/source hashes and report I/O are outside per-frame time but inside driver elapsed; termination is separate offline work. Arrival timestamps are software file-read admission, not physical capture or paced queue latency. No P95 camera/20Hz claim.')
        if not report['snapshot_before_termination']['metadata']['initialized']:
            report['status']='completed_unpaced_initialization_incomplete_no_fresh_pose'
    except Exception as e:report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        if stream is not None:
            try:report['stream_metrics_at_exit']=plain(stream.metrics())
            except Exception as e:report['cleanup_errors'].append(repr(e))
        if correlator is not None:
            try:correlator.close()
            except Exception as e:report['cleanup_errors'].append(repr(e))
            finally:report['correlation']=correlator.metrics()
        if encoder is not None:
            try:report['encoder_sources_after']=encoder.check_sources()
            except Exception as e:report['cleanup_errors'].append(repr(e))
            finally:
                try:encoder.close();report['encoder_metrics']=plain(encoder.metrics())
                except Exception as e:report['cleanup_errors'].append(repr(e))
        if neural is not None:
            after={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()};report.update(original_weights_after=after,original_weights_unchanged=after==weights_before)
            if after!=weights_before:report['cleanup_errors'].append('Original neural weights changed')
        try:
            after={p:sha(p) for p in pins};report.update(source_hashes_after=after,sources_and_inputs_unchanged=after==pins)
            if after!=pins:report['cleanup_errors'].append('Pinned source/input changed')
        except Exception as e:report['cleanup_errors'].append(repr(e))
        try:
            os.sched_setaffinity(0,affinity);require(os.sched_getaffinity(0)==affinity,'Affinity restore failed');report['affinity_restored']=True
        except Exception as e:report['cleanup_errors'].append(repr(e))
        if report['cleanup_errors']:report['status']='failed'
        report.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,actual_driver_elapsed_seconds=time.monotonic()-begin)
        (out/'run.json').write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','frames_completed','cleanup_errors','peak_rss_mib','actual_driver_elapsed_seconds')}),flush=True)
    return 0 if report['status'].startswith('completed_') else 1
if __name__=='__main__':raise SystemExit(main())
