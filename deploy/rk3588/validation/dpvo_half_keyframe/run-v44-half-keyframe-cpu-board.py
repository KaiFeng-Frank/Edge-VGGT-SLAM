#!/usr/bin/env python3
"""Board-only v44 CPU half replay: single-variable keyframe15->7.5.

Exact cached undistortion, original FP32 DPVO weights and frozen v30 lifecycle.
Derived from SHA-pinned v41 cached CPU driver; only runtime policy parameter
keyframe_threshold changes. Motion probe2, M16/L4/R8/O4, math and EOF stay fixed.
DPVO attribution: Princeton Vision & Learning Lab, MIT. Research, not adopted.
"""
import argparse, hashlib, importlib.util, json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np

BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
POLICY='5c5797880d1ffa92318819eb595b819c31fbbe016bbc88afbc8dcb704534cbdd'
BASELINE_POLICY='cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
BASELINE_DRIVER='18ad8f7bc3cc48790d2329b35372fd77473942c698cd3e7e291298194e498c2d'
PARENT='0f62c1df8ea68f88ec13b71b54f1f7b0b070736b7058ebd85fac013c2694b526'
CACHE='9743962325ba1c4ada689769eaf800a617ba1f46f4602062a6dab3b23b5bdf48'
LIBRARY='23ffa5b4b3e9251b8a41e149da2610c9db5818fe5fe148ab18f8b45b41486582'
BUILD='79a3d203ac505349d9154749059750a312217c70dc0bccec9ff42fdc9aac7194'

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


POLICY_METADATA_OVERRIDES = {('schema',): 'v44_DPVO_half_keyframe7p5_single_variable_predeclared_policy', ('preset',): 'half_M16_L4_R8_O4_KF2_threshold7p5', ('lifecycle', 'keyframe_flow_pair_slots'): 'n-3 <-> n-1 (KEYFRAME_INDEX2), remove slot n-2 iff original bidirectional mean feature-grid flow<7.5', ('mathematical_scope', 'feature_pixel_thresholds_unscaled'): False, ('mathematical_scope', 'threshold_scale_effect'): 'Only keyframe threshold is rescaled15->7.5 for image resize0.5. Motion probe2 and original fastBA feature-pixel gates remain unchanged, so their equivalent full-resolution pixel range remains doubled. The fixed3x3 patch and radius3 correlation support also widen by2. This is a changed research policy, not original FAST numerical or trajectory equivalence.', ('mathematical_scope', 'additional_preset_runs_planned'): True}
TRIAL_DECLARATION = {'schema': 'v44_single_variable_keyframe_threshold_trial', 'parent_policy_sha256': 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a', 'parent_cached_CPU_driver_sha256': '18ad8f7bc3cc48790d2329b35372fd77473942c698cd3e7e291298194e498c2d', 'parent_cached_CPU_run_sha256': 'c1f0a4636601fb1cd702ca57018af5fd937ecdbb9e722b0293428810bf66d05d', 'runtime_parameter_changes': [{'path': 'stream_config.keyframe_threshold', 'before': 15.0, 'after': 7.5, 'unit': 'CNN feature-grid pixels (image pixels divided by original RES4 once)'}], 'motion_probe_threshold_feature_pixels_unchanged': 2.0, 'threshold_selected_from_projection_units_not_GT': True, 'expected_first_keyframe_decision_divergence_at_baseline_state': {'observation_id': 54, 'saved_mean_bidirectional_flow': 8.138959407806396}, 'common_fresh_and_all_192_source_coverage_required': True, 'quality_10_percent_and_20_percent_limits_unchanged': True, 'candidate_accepted': False, 'realtime_validated': False, 'trajectory_accuracy_validated': False}

def validate_single_variable_policy(candidate, baseline):
    """Require the pinned base policy plus the one declared runtime change."""
    require(baseline['stream_config']['keyframe_threshold'] == 15.0,
            'Baseline keyframe threshold must be15')
    expected = json.loads(json.dumps(baseline))
    expected['stream_config']['keyframe_threshold'] = 7.5
    for parts, value in POLICY_METADATA_OVERRIDES.items():
        target = expected
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    expected['research_trial'] = TRIAL_DECLARATION
    require(candidate == expected, 'Candidate differs beyond declared single-variable policy')
    return json.loads(json.dumps(TRIAL_DECLARATION))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-driver','policy','torch-encoder','source-dir','checkpoint','geometry','stream-module','encoder-module','frame-cache-helper','library','build-receipt','encoder-capture-dir','dataset-dir','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--capture-update-states',action='store_true')
    p.add_argument('--undistort-helper',type=Path,required=True)
    p.add_argument('--baseline-driver',type=Path,required=True)
    p.add_argument('--baseline-policy',type=Path,required=True)
    a=p.parse_args()
    require(sha(a.parent_driver)==PARENT,'Frozen v34 helper source differs')
    parent=module(a.parent_driver,'_v39_parent');parent.require_board()
    out=a.output_dir.resolve();require(out.is_relative_to(BASE),'Isolated board output required');out.mkdir(exist_ok=False)
    begin=time.monotonic();affinity=os.sched_getaffinity(0);pins={};encoder=correlator=stream=neural=undistorter=None;weights_before=None
    report=dict(schema='v44_half_keyframe7p5_original_Torch_CPU_research_replay',status='failed',script_sha256=sha(__file__),candidate_accepted=False,realtime_validated=False,original_VGGT_default_unchanged=True,different_model_from_VGGT=True,actual_inference_on_RK3588S=True,ORB_used=False,added_LK_PnP_frontend=False,NPU_used=False,GT_or_sensor_depth_input=False,unpaced_file_replay=True,physical_camera=False,frames_requested=192,frames_completed=0,seed=20261011,frames=[],update_calls=[],BA_calls=[],update_captures=[],cleanup_errors=[],diagnostic_capture_enabled=a.capture_update_states)
    def pin(path,digest):
        path=Path(path).resolve();require(sha(path)==digest,'Source/artifact SHA differs: '+str(path));pins[str(path)]=digest;return path
    def save(name,arrays):
        path=out/name;np.savez(path,**arrays)
        return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:parent.meta(v) for k,v in arrays.items()})
    def snapshot(name):
        value=stream.snapshot();arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
        return dict(metadata=parent.plain({k:v for k,v in value.items() if k not in arrays}),artifact=save(name,arrays))
    try:
        for path,digest in ((Path(__file__),report['script_sha256']),(a.parent_driver,PARENT),(a.baseline_driver,BASELINE_DRIVER),(a.baseline_policy,BASELINE_POLICY),(a.policy,POLICY),(a.geometry,parent.GEOMETRY),(a.stream_module,parent.STREAM_MODULE_SHA256),(a.encoder_module,parent.ENCODER_MODULE_SHA256),(a.frame_cache_helper,CACHE),(a.library,LIBRARY),(a.build_receipt,BUILD)):
            pin(path,digest)
        pin(a.torch_encoder,sha(a.torch_encoder))
        pin(a.undistort_helper,'6ae53bd252de2333bf105e0f5f98c842dd2cc79244f8f1acf9572809bd627c63')
        pin(a.undistort_helper.with_name('fixed_undistort_cache.py'),'14c7331fcb9bc84e13fa8d9b38f0a0575b529729b08dd539915f3714a632972c')
        report['original_replay_source_sha256']='562008c8650404d78c4974eed0bcfab347f7db6ff0671ac8b4e7f05fa6c92ba2'
        policy=json.loads(a.policy.read_text());report['policy_sha256']=POLICY;report['policy']=policy
        report['research_trial']=validate_single_variable_policy(policy,json.loads(a.baseline_policy.read_text()))
        report['cached_CPU_parent_source_sha256']=BASELINE_DRIVER
        report['baseline_policy_sha256']=BASELINE_POLICY
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
        encoder=enc.BoardTorchEncoder(neural,common,sampling,torch,source_files=[a.torch_encoder,a.encoder_module,a.source_dir/'dpvo_probe_common.py'])
        generator=torch.Generator(device='cpu').manual_seed(report['seed']);rng=parent.TorchDepthRNG(torch,generator);report['random_generator_before']=parent.meta(generator.get_state().numpy())
        correlator=cache.PerFrameCorrelation(a.library,library_sha256=LIBRARY,patch_count=16,threads=4)
        original_ba=geo.BA
        def measured_ba(*args,**kwargs):
            start=time.monotonic();result=original_ba(*args,**kwargs)
            report['BA_calls'].append(dict(index=len(report['BA_calls']),stage=report.get('active_stage'),frame_index=report.get('active_frame'),seconds=time.monotonic()-start));return result
        geo.BA=measured_ba
        captured_frames=set()
        def update(net,ctx,corr,ii,jj,kk):
            start=time.monotonic();inputs=(net,ctx,corr,ii,jj,kk)
            require(all(isinstance(x,np.ndarray) and np.isfinite(x).all() for x in inputs),'Finite neural input required')
            with torch.inference_mode():
                n,(d,w,unused)=neural.update(torch.from_numpy(net),torch.from_numpy(ctx),torch.from_numpy(corr),None,torch.from_numpy(ii),torch.from_numpy(jj),torch.from_numpy(kk))
                require(unused is None,'Original Update auxiliary output differs')
                values={key:v.detach().numpy().copy() for key,v in [('net',n),('delta',d),('weight',w)]}
            require(all(np.isfinite(v).all() for v in values.values()) and np.all((values['weight']>=0)&(values['weight']<=1)),'Update finite/domain failed')
            forward_end=time.monotonic();sources,counts=np.unique(ii,return_counts=True)
            row=dict(index=len(report['update_calls']),edges=len(kk),frame_index=report.get('active_frame'),stage=report.get('active_stage'),seconds=forward_end-start,source_slots=sources.tolist(),source_degrees=(counts//16).tolist(),source_patch_counts=counts.tolist())
            frame=report.get('active_frame')
            if a.capture_update_states and report.get('active_stage')=='stream' and len(kk)>16 and (not captured_frames or frame in (127,191)) and frame not in captured_frames:
                before=save('update-%03d-input.npz'%frame,dict(zip(('net','inp','corr','ii','jj','kk'),inputs)))
                after=save('update-%03d-gold.npz'%frame,values)
                report['update_captures'].append(dict(frame_index=frame,update_index=row['index'],edges=len(kk),input=before,gold=after,entire_active_graph=True,same_call_original_Torch=True))
                captured_frames.add(frame)
            row['capture_and_metadata_seconds']=time.monotonic()-forward_end;report['update_calls'].append(row);return values
        stream=core.DPVOStream(config,geometry=geo,update=update,correlate=correlator,rng=rng)
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,numpy_version=np.__version__,torch_version=torch.__version__,OpenCV_version=cv2.__version__,loop_closure=False,precision='original Torch CPU FP32',pool_level4_shape_hw=[14,19])
        K=np.array([[517.3,0,318.6],[0,516.5,255.3],[0,0,1]],np.float64);D=np.array([.2624,-.9531,-.0054,.0026,1.1633],np.float64);intrinsics=np.array(policy['intrinsics'],np.float32)
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
                report['active_stage']='stream';result=stream.process_encoded(record);processed=time.monotonic()
                row.update(decode_undistort_crop_resize_seconds=decoded-row['arrival_monotonic'],encode_seconds=encoded-decoded,stream_seconds=processed-encoded,processing_seconds=processed-row['arrival_monotonic'],result=parent.plain(result),processed_monotonic=processed)
                events.write(json.dumps(dict(index=index,frame_id=item['frame_id'],result=parent.plain(result)),allow_nan=False)+'\n');events.flush()
                published=time.monotonic();row.update(pose_available_monotonic=published,admission_to_publication_seconds=published-row['arrival_monotonic'],publication_seconds=published-processed)
                report['frames'].append(row);report['frames_completed']=index+1
                if index%16==0 or index==191: print(json.dumps(dict(event='frame_complete',index=index,seconds=row['admission_to_publication_seconds'],fresh=bool(result['fresh_pose_valid']))),flush=True)
                (out/'progress.json').write_text(json.dumps(dict(frames_completed=index+1,seconds=published-replay_start))+'\n')
        report.update(replay_elapsed_seconds=time.monotonic()-replay_start,metrics_before_termination=parent.plain(stream.metrics()),snapshot_before_termination=snapshot('before-termination.npz'))
        report['active_stage']='termination_offline';start=time.monotonic();terminated=stream.terminate()
        report.update(termination_seconds=time.monotonic()-start,termination=parent.plain(terminated),stream_metrics=parent.plain(stream.metrics()),final_snapshot=snapshot('final-state.npz'),random_generator_after=parent.meta(generator.get_state().numpy()),random_depth_draw_calls=rng.calls)
        report.pop('active_frame',None);report.pop('active_stage',None)
        report.update(status='completed_unpaced_half_keyframe7p5_Torch_CPU_research_not_realtime',timing_scope='Serial unpaced file replay. Admission before imread; publication after JSON event serialization and flush. Includes first-frame map construction, cached undistort/crop/resize, encoder, patch sampling, full graph correlation/Update/BA, first3 crop identity checks and optional full Update captures. Model/source loading outside replay; progress I/O inside replay denominator. EOF separate offline. No physical-capture or paced queue claim.')
        if not report['snapshot_before_termination']['metadata']['initialized']: report['status']='completed_unpaced_initialization_incomplete_no_fresh_pose'
    except Exception as e: report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
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
