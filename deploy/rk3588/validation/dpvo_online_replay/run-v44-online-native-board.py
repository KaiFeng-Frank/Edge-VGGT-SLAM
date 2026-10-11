#!/usr/bin/env python3
"""Board-only v44 replay: frozen v43 flow with native FP16 input transport.

Minimal derivative of the frozen v43 RKNN replay; no trajectory logic change. Original Torch
is used only for explicit startup motion probes; default encoder stays CPU,
with an optional separately pinned half INT8 RKNN encoder. DPVO lifecycle,
RNG, geometry, correlation, weights and preprocessing remain frozen.
Historical FP32 rejection is retained. Returned RKNN values feed research VO
state synchronously; no candidate adoption or VGGT default change is implied.
DPVO: Princeton Vision & Learning Lab (2022), MIT, adjacent LICENSE.
"""
import argparse, hashlib, importlib.util, json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np

BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
POLICY='cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
PARENT='0f62c1df8ea68f88ec13b71b54f1f7b0b070736b7058ebd85fac013c2694b526'
CACHE='9743962325ba1c4ada689769eaf800a617ba1f46f4602062a6dab3b23b5bdf48'
LIBRARY='23ffa5b4b3e9251b8a41e149da2610c9db5818fe5fe148ab18f8b45b41486582'
BUILD='79a3d203ac505349d9154749059750a312217c70dc0bccec9ff42fdc9aac7194'
CACHED_CPU_DRIVER='18ad8f7bc3cc48790d2329b35372fd77473942c698cd3e7e291298194e498c2d'
TORCH_ENCODER='517ed0746c82b8236cf591a1bef02cf12dfde56fcd1a16c9870f17921697c580'
ONLINE_FACTORY='ac2b29b7eabc17d1dd47ec7a89734d6c1512dcbf37845ef5a32e9667b9b6be79'
ONLINE_CALLBACK='4f4a5457dbdec5aebe1ddd6204a37c9073896ac99ed90abd3fbfe3d1f78db566'
ONLINE_CONTRACTS='30e68837815bc6f135f6d244b9ae4d166cf3ed43991249ca97bc07aa40a6ebb0'
V41_ONLINE_DRIVER='73f5a4c011d957f95d8d8e54687a167709f22cc6104611a7d62bf7b4f68ea6d9'
V42_ONLINE_DRIVER='17d8e26018b1499ae576b1474946ee393b5ce701c562987d8f9060f7b181960d'
V43_ONLINE_DRIVER='176d3cb37abcf38818ac37a38e9ba1924b84caf4b3071784850058755ca96d65'
NPU_ENCODER='ebe62f848421ffb97262f1c70b5e1af95e7e6f0db0ca6b66850797dd9f89a52f'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def require(ok,msg):
    if not ok: raise ValueError(msg)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def online_stream_class(core,online):
    """Only the frozen core's explicit probe argument selects the owner route."""
    class OnlineDPVOStream(core.DPVOStream):
        def _neural(self,*args,probe,**kwargs):
            previous=online.route
            online.route='motion_probe' if probe else 'full'
            try:
                return super()._neural(*args,probe=probe,**kwargs)
            finally:
                online.route=previous
    return OnlineDPVOStream


def read_board_health(sysfs_root):
    """Read temperatures and cpufreq only; unavailable files stay diagnostic."""
    began=time.monotonic();root=Path(sysfs_root)
    value=dict(monotonic=began,read_only=True,thermal=[],cpufreq=[],read_errors=[])
    def read(path,integer=False):
        try:
            raw=path.read_text().strip()
            return int(raw) if integer else raw
        except (OSError,ValueError) as error:
            value['read_errors'].append(dict(path=str(path),error=repr(error)))
            return None
    for zone in sorted((root/'class/thermal').glob('thermal_zone*')):
        temperature=read(zone/'temp',True)
        value['thermal'].append(dict(zone=zone.name,type=read(zone/'type'),
            temperature_millidegrees_C=temperature,
            temperature_C=None if temperature is None else temperature/1000.))
    for policy in sorted((root/'devices/system/cpu/cpufreq').glob('policy*')):
        value['cpufreq'].append(dict(policy=policy.name,
            scaling_cur_freq_kHz=read(policy/'scaling_cur_freq',True),
            scaling_governor=read(policy/'scaling_governor')))
    value['read_seconds']=time.monotonic()-began
    return value


def npu_activity(online_metrics,encoder_metrics):
    """Actual SDK run counters, including evicted context histories."""
    cache=online_metrics.get('cache',{})
    rows=[row['metrics'] for row in cache.get('histories',[])]
    rows.extend(cache.get('sessions',{}).values())
    update_calls=sum(int(row.get('sdk_run_api_calls',0)) for row in rows)
    update_used=any(row.get('npu_used') is True and row.get('sdk_run_api_calls',0)>0 for row in rows)
    encoder_calls=int(encoder_metrics.get('sdk_run_api_calls',0))
    encoder_used=encoder_metrics.get('NPU_used') is True and encoder_calls>0
    return dict(Update_sdk_run_api_calls=update_calls,encoder_sdk_run_api_calls=encoder_calls,
        Update_NPU_used=update_used,encoder_NPU_used=encoder_used,NPU_used=update_used or encoder_used,
        internal_all_NPU_verified=False,INT8_operator_coverage_verified=False)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-driver','policy','torch-encoder','source-dir','checkpoint','geometry','stream-module','encoder-module','frame-cache-helper','library','build-receipt','encoder-capture-dir','dataset-dir','output-dir','cached-cpu-driver','online-factory','online-config','v41-online-driver','v42-online-driver','v43-online-driver'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--online-factory-sha256',required=True)
    p.add_argument('--online-config-sha256',required=True)
    p.add_argument('--online-failure-dir',type=Path)
    p.add_argument('--diagnostic-allow-fp32-rejected',action='store_true')
    p.add_argument('--diagnostic-allow-native-fp16',action='store_true')
    p.add_argument('--npu-encoder',type=Path)
    p.add_argument('--npu-encoder-sha256')
    p.add_argument('--encoder-config',type=Path)
    p.add_argument('--encoder-config-sha256')
    p.add_argument('--health-sysfs-root',type=Path,default=Path('/sys'))
    p.add_argument('--capture-update-states',action='store_true')
    p.add_argument('--undistort-helper',type=Path,required=True)
    return p.parse_args(argv)


def main(argv=None):
    a=parse_args(argv)
    require(a.diagnostic_allow_fp32_rejected is True and a.diagnostic_allow_native_fp16 is True,'Explicit FP32-rejected and native-FP16 research diagnostic flags required')
    require(sha(a.cached_cpu_driver)==CACHED_CPU_DRIVER,'Frozen cached CPU replay source differs')
    require(sha(a.v41_online_driver)==V41_ONLINE_DRIVER,'Frozen v41 online replay source differs')
    require(sha(a.v42_online_driver)==V42_ONLINE_DRIVER,'Frozen v42 online replay source differs')
    require(sha(a.v43_online_driver)==V43_ONLINE_DRIVER,'Frozen v43 online replay source differs')
    require(a.online_factory_sha256==ONLINE_FACTORY and sha(a.online_factory)==ONLINE_FACTORY,'Frozen online factory differs')
    if a.npu_encoder is None:
        require(all(v is None for v in (a.npu_encoder_sha256,a.encoder_config,a.encoder_config_sha256)),'NPU encoder options require --npu-encoder')
    else:
        require(a.npu_encoder_sha256==NPU_ENCODER and a.encoder_config is not None and a.encoder_config_sha256 is not None,'Complete pinned NPU encoder configuration required')
    require(sha(a.parent_driver)==PARENT,'Frozen v34 helper source differs')
    parent=module(a.parent_driver,'_v39_parent');parent.require_board()
    out=a.output_dir.resolve();require(out.is_relative_to(BASE),'Isolated board output required');out.mkdir(exist_ok=False)
    begin=time.monotonic();affinity=os.sched_getaffinity(0);pins={};encoder=correlator=stream=neural=undistorter=online=None;weights_before=None
    report=dict(schema='v44_cached_undistort_native_FP16_online_RKNN_research_replay',status='failed',script_sha256=sha(__file__),candidate_accepted=False,realtime_validated=False,original_VGGT_default_unchanged=True,different_model_from_VGGT=True,actual_inference_on_RK3588S=True,ORB_used=False,added_LK_PnP_frontend=False,NPU_used=False,GT_or_sensor_depth_input=False,unpaced_file_replay=True,physical_camera=False,frames_requested=192,frames_completed=0,seed=20261011,frames=[],update_calls=[],BA_calls=[],update_captures=[],cleanup_errors=[],diagnostic_capture_enabled=a.capture_update_states)
    report.update(diagnostic_only=True,diagnostic_allow_fp32_rejected=True,adopted=False,
        historical_SDK_strict_FP32_rejected=True,SDK_FP32_equivalence_validated=False,
        FP32_thresholds_relaxed=False,output_clipping=False,complete_Update_Torch_fallback=False,
        VO_state_committed=False,VO_state_commit_observations=0,NPU_Update_research_feedback_committed=False,
        VO_state_commit_scope='Synchronous research replay owner state; no adoption',
        encoder_requested='half_INT8_RKNN_diagnostic' if a.npu_encoder else 'original_Torch_CPU_FP32',
        original_cached_CPU_driver_sha256=CACHED_CPU_DRIVER,original_v41_online_driver_sha256=V41_ONLINE_DRIVER,original_v42_online_driver_sha256=V42_ONLINE_DRIVER,
        metadata_only_reporting=True,native_input_transport_added=True,online_factory_sha256=ONLINE_FACTORY,
        original_v43_online_driver_sha256=V43_ONLINE_DRIVER,diagnostic_allow_native_fp16=True,
        native_full_saved_AB_prerequisite_sha256='b7e28619870af18f2a95c6d411281bbfe249fb1a69f97ab0c4df5cbe6adc0968',
        native_update_input_transport='FP16_NC1HWC2_C2_8_pass_through_1',closed_flag_optimization_added=False,
        online_callback_sha256=ONLINE_CALLBACK,online_update_records=[],motion_probe_Torch_calls=[],
        board_health_samples=[],board_health_policy='Read-only start/end and after each16 publications. Periodic/end reads count in replay denominator, after publication outside that frame latency; start read before replay clock. No sysfs/settings writes.')
    def pin(path,digest):
        path=Path(path).resolve();require(sha(path)==digest,'Source/artifact SHA differs: '+str(path));pins[str(path)]=digest;return path
    def save(name,arrays):
        path=out/name;np.savez(path,**arrays)
        return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:parent.meta(v) for k,v in arrays.items()})
    def snapshot(name):
        value=stream.snapshot();arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
        return dict(metadata=parent.plain({k:v for k,v in value.items() if k not in arrays}),artifact=save(name,arrays))
    try:
        for path,digest in ((Path(__file__),report['script_sha256']),(a.parent_driver,PARENT),(a.policy,POLICY),(a.geometry,parent.GEOMETRY),(a.stream_module,parent.STREAM_MODULE_SHA256),(a.encoder_module,parent.ENCODER_MODULE_SHA256),(a.frame_cache_helper,CACHE),(a.library,LIBRARY),(a.build_receipt,BUILD)):
            pin(path,digest)
        pin(a.torch_encoder,TORCH_ENCODER)
        pin(a.cached_cpu_driver,CACHED_CPU_DRIVER);pin(a.online_factory,ONLINE_FACTORY);pin(a.v41_online_driver,V41_ONLINE_DRIVER)
        pin(a.v42_online_driver,V42_ONLINE_DRIVER);pin(a.v43_online_driver,V43_ONLINE_DRIVER)
        online_settings=json.loads(pin(a.online_config,a.online_config_sha256).read_text())
        require(isinstance(online_settings,dict),'Online config must be a JSON object')
        require(not any(key in online_settings for key in ('motion_probe','on_record','identity_reader','diagnostic_allow_fp32_rejected','diagnostic_allow_native_fp16','failure_dir')),'Online runtime owner arguments are assigned by driver')
        require(online_settings.get('callback_sha256')==ONLINE_CALLBACK,'Frozen online callback SHA differs')
        pin(online_settings['callback_path'],ONLINE_CALLBACK)
        require(online_settings.get('contracts_sha256')==ONLINE_CONTRACTS,'Frozen v42 online contracts SHA differs')
        pin(online_settings['contracts'],ONLINE_CONTRACTS)
        report['online_config_sha256']=a.online_config_sha256;report['online_config']=dict(online_settings)
        encoder_settings=None
        if a.npu_encoder is not None:
            pin(a.npu_encoder,NPU_ENCODER)
            encoder_settings=json.loads(pin(a.encoder_config,a.encoder_config_sha256).read_text())
            require(isinstance(encoder_settings,dict) and encoder_settings.get('diagnostic_allow_fp32_rejected') is True,'Explicit rejected half encoder config required')
            report['encoder_config_sha256']=a.encoder_config_sha256;report['encoder_config']=dict(encoder_settings)
        pin(a.undistort_helper,'6ae53bd252de2333bf105e0f5f98c842dd2cc79244f8f1acf9572809bd627c63')
        pin(a.undistort_helper.with_name('fixed_undistort_cache.py'),'14c7331fcb9bc84e13fa8d9b38f0a0575b529729b08dd539915f3714a632972c')
        report['original_replay_source_sha256']='562008c8650404d78c4974eed0bcfab347f7db6ff0671ac8b4e7f05fa6c92ba2'
        policy=json.loads(a.policy.read_text());report['policy_sha256']=POLICY;report['policy']=policy
        build=json.loads(a.build_receipt.read_text());require(build['status']=='compiled_and_synthetic_smoke_passed' and build['output']['sha256']==LIBRARY,'Native build differs')
        for name,digest in parent.PINS.items(): pin(a.source_dir/name,digest)
        manifest=json.loads((a.source_dir/'manifest.json').read_text())
        for row in manifest['retrieved_files']:
            path=a.source_dir/'official'/row['path']
            if row.get('available_in_pinned_tree') is False:
                require('sha256' not in row and not path.exists(),'Absent original file became present');continue
            pin(path,row['sha256'])
        pin(a.checkpoint,parent.CHECKPOINT);pin(a.dataset_dir/'rgb.txt',parent.RGB);pin(a.encoder_capture_dir/'capture.json',parent.CAPTURE)
        selected=[]
        for line in (a.dataset_dir/'rgb.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'):continue
            stamp,name=line.split();path=(a.dataset_dir/name).resolve()
            require(path.is_relative_to(a.dataset_dir.resolve()) and name.startswith('rgb/'),'RGB path escape')
            pin(path,sha(path));selected.append(dict(frame_id=stamp,timestamp=float(stamp),relative_path=name,bytes=path.stat().st_size,sha256=pins[str(path)]))
            if len(selected)==192:break
        require(len(selected)==192 and all(np.isfinite(x['timestamp']) for x in selected) and all(b['timestamp']>x['timestamp'] for x,b in zip(selected,selected[1:])),'192 consecutive ordered frames required')
        input_manifest=dict(schema='v39_first_contiguous_TUM_RGB_frames',dataset='TUM RGB-D freiburg1_xyz',rgb_index_sha256=parent.RGB,frames=selected,frame_selection='first192 consecutive RGB entries, stride1',GT_read=False)
        (out/'input-manifest.json').write_text(json.dumps(input_manifest,indent=2)+'\n');report['input_manifest_sha256']=sha(out/'input-manifest.json')
        require({4,5,6,7}<=affinity,'Four A76 CPUs required');os.sched_setaffinity(0,{4,5,6,7})
        import torch,cv2
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        require(cv2.__version__=='4.10.0' and cv2.getNumThreads()==4,'Original post-affinity OpenCV contract differs')
        undistort_module=module(a.undistort_helper,'_v41_exact_undistort')
        report['OpenCV_threads']=cv2.getNumThreads()
        common=module(a.source_dir/'dpvo_probe_common.py','_v39_neural');geo=module(a.geometry,'_v39_geo');core=module(a.stream_module,'_v39_core');sampling=module(a.encoder_module,'_v39_sampling');cache=module(a.frame_cache_helper,'_v39_cache');enc=module(a.torch_encoder,'_v39_torch_encoder')
        report['online_lifecycle_source_sha256']=core.verify_sources(a.source_dir/'official')
        _,report['original_FAST_provenance']=parent.load_fast_config(a.source_dir/'official')
        config=core.StreamConfig(**policy['stream_config']);report['config']=vars(config)
        start=time.monotonic();neural,loader=common.load_neural(a.source_dir/'official',a.source_dir/'manifest.json',a.checkpoint,parent.CHECKPOINT,a.source_dir/'dpvo_cpu_ops.py',cpu_ops_sha256=parent.PINS['dpvo_cpu_ops.py'])
        weights_before={k:parent.meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
        captured=json.loads((a.encoder_capture_dir/'capture.json').read_text());require(len(weights_before)==94 and weights_before==captured['state_before'],'Original94 parameters differ')
        report.update(neural_load_seconds=time.monotonic()-start,loader=loader,original_weights_before=weights_before)
        if encoder_settings is None:
            encoder=enc.BoardTorchEncoder(neural,common,sampling,torch,source_files=[a.torch_encoder,a.encoder_module,a.source_dir/'dpvo_probe_common.py'])
        else:
            encoder_logs=Path(encoder_settings['log_directory']).resolve()
            require(encoder_logs.is_relative_to(out),'Encoder diagnostic logs must be inside this run output')
            encoder_logs.mkdir(parents=True,exist_ok=True)
            npu_enc=module(a.npu_encoder,'_v41_half_npu_encoder')
            encoder=npu_enc.BoardHalfRKNNEncoder(**encoder_settings)
        generator=torch.Generator(device='cpu').manual_seed(report['seed']);rng=parent.TorchDepthRNG(torch,generator);report['random_generator_before']=parent.meta(generator.get_state().numpy())
        correlator=cache.PerFrameCorrelation(a.library,library_sha256=LIBRARY,patch_count=16,threads=4)
        original_ba=geo.BA
        def measured_ba(*args,**kwargs):
            start=time.monotonic();result=original_ba(*args,**kwargs)
            report['BA_calls'].append(dict(index=len(report['BA_calls']),stage=report.get('active_stage'),frame_index=report.get('active_frame'),seconds=time.monotonic()-start));return result
        geo.BA=measured_ba
        captured_frames=set()
        def original_torch_motion_probe(net,ctx,corr,ii,jj,kk):
            start=time.monotonic()
            with torch.inference_mode():
                n,(d,w,unused)=neural.update(torch.from_numpy(net),torch.from_numpy(ctx),torch.from_numpy(corr),None,torch.from_numpy(ii),torch.from_numpy(jj),torch.from_numpy(kk))
                require(unused is None,'Original Update auxiliary output differs')
                values={key:v.detach().numpy().copy() for key,v in [('net',n),('delta',d),('weight',w)]}
            require(all(np.isfinite(v).all() for v in values.values()) and np.all((values['weight']>=0)&(values['weight']<=1)),'Original startup Update finite/domain failed')
            report['motion_probe_Torch_calls'].append(dict(index=len(report['motion_probe_Torch_calls']),
                edges=len(kk),frame_index=report.get('active_frame'),stage=report.get('active_stage'),
                seconds=time.monotonic()-start,explicit_startup_route=True))
            return values
        def on_online_record(record):
            record.update(frame_index=report.get('active_frame'),stage=report.get('active_stage'))
            report['online_update_records'].append(record)
        online_failure_dir=(a.online_failure_dir or out/'online-sdk').resolve()
        require(online_failure_dir.is_relative_to(out),'Online SDK diagnostic logs must be inside this run output')
        online_failure_dir.mkdir(parents=True,exist_ok=True)
        factory=module(a.online_factory,'_v41_online_factory')
        online=factory.open_board_online_from_index(**online_settings,
            failure_dir=online_failure_dir,motion_probe=original_torch_motion_probe,
            diagnostic_allow_fp32_rejected=True,diagnostic_allow_native_fp16=True,on_record=on_online_record)
        for path,digest in online.source_pins.items():pin(path,digest)
        report['online_catalogue_provenance']=online.catalogue_provenance
        def update(net,ctx,corr,ii,jj,kk):
            start=time.monotonic();inputs=(net,ctx,corr,ii,jj,kk)
            require(all(isinstance(x,np.ndarray) and np.isfinite(x).all() for x in inputs),'Finite neural input required')
            values=online(*inputs)
            require(all(np.isfinite(v).all() for v in values.values()) and np.all((values['weight']>=0)&(values['weight']<=1)),'Update finite/domain failed')
            forward_end=time.monotonic();sources,counts=np.unique(ii,return_counts=True)
            sdk_record=report['online_update_records'][-1]
            row=dict(index=len(report['update_calls']),edges=len(kk),frame_index=report.get('active_frame'),stage=report.get('active_stage'),seconds=forward_end-start,source_slots=sources.tolist(),source_degrees=(counts//16).tolist(),source_patch_counts=counts.tolist(),route=online.route,execution_backend=sdk_record['execution_backend'],NPU_used=sdk_record['NPU_used'],SDK_record_index=sdk_record['index'])
            frame=report.get('active_frame')
            if a.capture_update_states and online.route=='full' and report.get('active_stage')=='stream' and len(kk)>16 and (not captured_frames or frame in (127,191)) and frame not in captured_frames:
                before=save('update-%03d-input.npz'%frame,dict(zip(('net','inp','corr','ii','jj','kk'),inputs)))
                after=save('update-%03d-output.npz'%frame,values)
                report['update_captures'].append(dict(frame_index=frame,update_index=row['index'],edges=len(kk),input=before,output=after,output_backend='RKNN_complete_source_groups',entire_active_graph=True,same_call_original_Torch=False,reference_supplied=False,candidate_accepted=False))
                captured_frames.add(frame)
            row['capture_and_metadata_seconds']=time.monotonic()-forward_end;report['update_calls'].append(row);return values
        stream=online_stream_class(core,online)(config,geometry=geo,update=update,correlate=correlator,rng=rng)
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,numpy_version=np.__version__,torch_version=torch.__version__,OpenCV_version=cv2.__version__,loop_closure=False,precision='Diagnostic RKNN complete Update; original Torch startup probe; '+('half INT8 RKNN encoder' if a.npu_encoder else 'original Torch CPU FP32 encoder'),pool_level4_shape_hw=[14,19])
        K=np.array([[517.3,0,318.6],[0,516.5,255.3],[0,0,1]],np.float64);D=np.array([.2624,-.9531,-.0054,.0026,1.1633],np.float64);intrinsics=np.array(policy['intrinsics'],np.float32)
        report['board_health_start']=read_board_health(a.health_sysfs_root)
        replay_start=time.monotonic();report['replay_begin_monotonic']=replay_start
        with (out/'events.ndjson').open('w') as events:
            for index,item in enumerate(selected):
                require(time.monotonic()-replay_start<1200,'Bounded1200s replay exceeded')
                row=dict(index=index,frame_id=item['frame_id'],timestamp=item['timestamp'],arrival_monotonic=time.monotonic())
                report['active_frame']=index;report['active_stage']='decode'
                bgr=cv2.imread(str(a.dataset_dir/item['relative_path']),cv2.IMREAD_COLOR)
                require(bgr is not None and bgr.dtype==np.uint8 and bgr.shape==(480,640,3),'BGR shape differs')
                if undistorter is None:
                    construction=time.monotonic()
                    undistorter=undistort_module.SingleRemapUndistorter(K,D,(480,640),expected_opencv_version='4.10.0')
                    report['undistort_cold_construction_seconds']=time.monotonic()-construction
                crop=np.ascontiguousarray(undistorter.undistort_bgr(bgr)[8:-8,16:-16])
                image=np.ascontiguousarray(cv2.resize(crop,(304,232),interpolation=cv2.INTER_AREA));decoded=time.monotonic()
                if index<3:
                    original=captured['inputs'][index]
                    require(original['source']=={k:item[k] for k in ('frame_id','relative_path','bytes','sha256')} and parent.meta(crop.transpose(2,0,1))==original['cropped_BGR_uint8'],'First3 full crop/source identity differs')
                    row.update(original_full_crop_bytes_match=True,resized_BGR=parent.meta(image),normalized_half_input=parent.meta(sampling.normalize_bgr(image)))
                report['active_stage']='encoder';record=encoder.encode(image,generator=generator,patch_count=16);encoded=time.monotonic()
                record.update(frame_id=item['frame_id'],timestamp=item['timestamp'],captured_monotonic=row['arrival_monotonic'],intrinsics=intrinsics.copy())
                report['active_stage']='stream';updates_before=len(report['update_calls']);result=stream.process_encoded(record);processed=time.monotonic()
                report['VO_state_committed']=True;report['VO_state_commit_observations']+=1
                if any(call.get('route')=='full' and call.get('NPU_used') is True for call in report['update_calls'][updates_before:]):
                    report['NPU_Update_research_feedback_committed']=True
                row.update(decode_undistort_crop_resize_seconds=decoded-row['arrival_monotonic'],encode_seconds=encoded-decoded,stream_seconds=processed-encoded,processing_seconds=processed-row['arrival_monotonic'],result=parent.plain(result),processed_monotonic=processed)
                events.write(json.dumps(dict(index=index,frame_id=item['frame_id'],result=parent.plain(result)),allow_nan=False)+'\n');events.flush()
                published=time.monotonic();row.update(pose_available_monotonic=published,admission_to_publication_seconds=published-row['arrival_monotonic'],publication_seconds=published-processed)
                report['frames'].append(row);report['frames_completed']=index+1
                if index%16==0 or index==191: print(json.dumps(dict(event='frame_complete',index=index,seconds=row['admission_to_publication_seconds'],fresh=bool(result['fresh_pose_valid']))),flush=True)
                (out/'progress.json').write_text(json.dumps(dict(frames_completed=index+1,seconds=published-replay_start))+'\n')
                if (index+1)%16==0:
                    report['board_health_samples'].append(dict(frame_index=index,publication_monotonic=published,health=read_board_health(a.health_sysfs_root)))
        report['board_health_end']=read_board_health(a.health_sysfs_root)
        report.update(replay_elapsed_seconds=time.monotonic()-replay_start,metrics_before_termination=parent.plain(stream.metrics()),snapshot_before_termination=snapshot('before-termination.npz'))
        report['active_stage']='termination_offline';start=time.monotonic();terminated=stream.terminate()
        report.update(termination_seconds=time.monotonic()-start,termination=parent.plain(terminated),stream_metrics=parent.plain(stream.metrics()),final_snapshot=snapshot('final-state.npz'),random_generator_after=parent.meta(generator.get_state().numpy()),random_depth_draw_calls=rng.calls)
        report.pop('active_frame',None);report.pop('active_stage',None)
        report.update(status='completed_unpaced_half_online_RKNN_research_not_realtime',timing_scope='Serial unpaced file replay. Admission before imread; publication after JSON event serialization and flush. Includes first-frame map construction, cached undistort/crop/resize, encoder, patch sampling, full graph correlation/Update/BA, first3 crop identity checks and optional full Update captures. Model/source loading outside replay; progress I/O and periodic/end read-only board temperature/cpufreq sampling inside replay denominator; health reads happen after publication outside individual frame latency. EOF separate offline. No physical-capture or paced queue claim.')
        if not report['snapshot_before_termination']['metadata']['initialized']: report['status']='completed_unpaced_initialization_incomplete_no_fresh_pose'
    except Exception as e: report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        report['board_health_exit']=read_board_health(a.health_sysfs_root)
        update_metrics_before={}
        if online is not None:
            try:
                update_metrics_before=online.metrics();report['online_metrics_before_close']=update_metrics_before
            except Exception as e:report['cleanup_errors'].append('online metrics: '+repr(e))
            finally:
                try:online.close()
                except Exception as e:report['cleanup_errors'].append('online close: '+repr(e))
                try:report['online_metrics']=online.metrics()
                except Exception as e:report['cleanup_errors'].append('closed online metrics: '+repr(e))
        if stream is not None:
            try: report['stream_metrics_at_exit']=parent.plain(stream.metrics())
            except Exception as e: report['cleanup_errors'].append(repr(e))
        if undistorter is not None:
            try:
                undistorter.close();report['undistort_cache']=undistorter.metadata()
                require(report['undistort_cache']['closed'] and report['undistort_cache']['current_map_bytes']==0,'Undistort maps not released')
            except Exception as e: report['cleanup_errors'].append(repr(e))
        if correlator is not None:
            try: correlator.close();report['correlation']=parent.plain(correlator.metrics())
            except Exception as e: report['cleanup_errors'].append(repr(e))
        if encoder is not None:
            try: report['encoder_sources_after']=encoder.check_sources()
            except Exception as e: report['cleanup_errors'].append(repr(e))
            finally:
                try: encoder.close();report['encoder_metrics']=encoder.metrics()
                except Exception as e: report['cleanup_errors'].append(repr(e))
        activity=npu_activity(update_metrics_before,report.get('encoder_metrics',{}))
        report['SDK_activity']=activity;report['NPU_used']=activity['NPU_used']
        if neural is not None:
            after={k:parent.meta(v.detach().numpy()) for k,v in neural.state_dict().items()};report.update(original_weights_after=after,original_weights_unchanged=after==weights_before)
            if after!=weights_before: report['cleanup_errors'].append('Neural weights changed')
        try:
            after={path:sha(path) for path in pins};report.update(source_hashes_after=after,sources_and_inputs_unchanged=after==pins)
            if after!=pins: report['cleanup_errors'].append('Pinned source/input changed')
        except Exception as e: report['cleanup_errors'].append(repr(e))
        try: os.sched_setaffinity(0,affinity);require(os.sched_getaffinity(0)==affinity,'Affinity restore failed');report['affinity_restored']=True
        except Exception as e: report['cleanup_errors'].append(repr(e))
        if report['cleanup_errors']:report['status']='failed'
        report.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,actual_driver_elapsed_seconds=time.monotonic()-begin)
        (out/'run.json').write_text(json.dumps(parent.plain(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','frames_completed','peak_rss_mib','cleanup_errors','actual_driver_elapsed_seconds')}),flush=True)
    return 0 if report['status'].startswith('completed_') else 1

if __name__=='__main__':raise SystemExit(main())
