#!/usr/bin/env python3
"""Saved v44 native online records audit. No model/ORT/SDK/BA execution.

This checks recorded transport, source identity, counters and cleanup. It does
not re-convert native payloads or establish original Torch/trajectory quality.
DPVO, Princeton Vision & Learning Lab (2022), MIT; adjacent LICENSE.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import uuid

PARENT_AUDITOR_SHA256 = '9a453f57d5ff7cce4d24f5ef6ac117318416b6f9ef1e340694c8fef1918d0b53'
DRIVER_SHA256 = '68d44f3722ea37310084d903bcaf72520ca3ef72cd794e6f972abd8fd45c93ab'
FACTORY_SHA256 = 'ac2b29b7eabc17d1dd47ec7a89734d6c1512dcbf37845ef5a32e9667b9b6be79'
INDEX_SHA256 = 'eeb9241c9eab52039aa64fcf6546ce4624d1388756a9e919f42c22de4083a85c'
MODEL_KEYS = {f'B{b}_D{d}_M16' for b in (1,2) for d in range(4,8)} | {'B1_D3_M16'}
REQUIRED_SOURCES = {
    'run-v44-online-native-board.py':DRIVER_SHA256, 'v44_online_factory.py':FACTORY_SHA256,
    'run-v43-online-npu-board.py':'176d3cb37abcf38818ac37a38e9ba1924b84caf4b3071784850058755ca96d65',
    'v43_online_factory.py':'e15ff495ac626e7a1a7271a918b39854b4cc45989b77e334db56d8234ea5f300',
    'v43_online_update.py':'4f4a5457dbdec5aebe1ddd6204a37c9073896ac99ed90abd3fbfe3d1f78db566',
    'v43_native_fp16_grouped.py':'222c254361a50d9180725ea780116a2302a29be8396136b15db91e7d11e912ca',
    'v43_native_fp16_pack.py':'b3d2503d1bb27ec0352cfe54d759efdcee070764ad266841847005fa9b13f8ce',
    'v43_native_fp16_pack.c':'78b46cb0abb2a32b2a8da2d4a15beccab725cc5b897297304699ec3bb535443f',
    'v43_native_fp16_pack.h':'e2fe575b74908d73869fc4c6d2dd00294a0f176f89b5a28cb7af2fcdfc9011dc',
    'libv43_native_fp16_pack.so':'5de153a44839915a76d68e7cb3141e31c46ae6a3dc83e8b32273bc9ecd3d0c6a',
    'query.json':'89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642',
    'conversion.json':'94c4cbd4adc458e34109d9cfa003112890069509616790fe5d7b8a0909883ee8',
    'run-v43-native-full-ab-board.py':'4c056f07828db7b5fa74914148e5a196840af2275988eb4f4104a459488fa9c2'}
SDK_SOURCES = {
    'run-v12-rknn-block-board.py':'37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608',
    'rknn_api_v2.3.2.h':'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a',
    'librknnrt.so':'d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'}
FULL_AB_SHA256 = 'b7e28619870af18f2a95c6d411281bbfe249fb1a69f97ab0c4df5cbe6adc0968'
BUILD_SHA256 = '7f907ca8c477ddb80d6d18802d160bde6d2c628619406a2504c7b5f6684f28cb'
TIMINGS = ('pack_seconds','input_set_seconds','run_api_seconds','outputs_get_seconds',
    'owned_copy_seconds','outputs_release_seconds','forward_seconds')


def require(ok, reason):
    if not ok: raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read_json(path):
    def invalid(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def load_parent(path):
    require(sha(path)==PARENT_AUDITOR_SHA256,'Frozen native auditor source SHA differs')
    name='_v44_native_audit_parent_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def validate_log(log):
    raw=log['raw_text'].encode()
    require(log['truncated'] is False and log['RKNN_error_detected'] is False
        and log['bytes']==len(raw) and log['raw_sha256']==hashlib.sha256(raw).hexdigest()
        and re.search(rb'(?:^|\n)\s*E\s+RKNN\b',raw) is None,
        'Native error/log text, bytes, SHA or truncation differs')


def validate_source_maps(before, after):
    require(isinstance(before,dict) and before and before==after,'Recorded source hashes changed or absent')
    require(all(isinstance(p,str) and isinstance(d,str) and re.fullmatch('[0-9a-f]{64}',d)
        for p,d in before.items()),'Malformed source pin map')


def validate_required_pins(pins, required):
    pairs={(Path(p).name,d) for p,d in pins.items()}
    require(set(required.items())<=pairs,'Required frozen source prerequisite pins absent from owning scope')


def validate_model_binding(metrics, model_row):
    artifact=model_row['model'];path=str(Path(model_row['compile_dir'])/artifact['file'])
    # The frozen parent uses ctypes.create_string_buffer(payload), with one NUL
    # byte of owned host storage; rknn_init receives len(payload), not len(buffer).
    require(metrics['source_pins'].get(path)==artifact['sha256']
        and metrics['model_host_buffer_bytes']==artifact['bytes']+1,
        'Context model SHA or exact NUL-terminated host buffer size differs')
    return artifact['bytes']+1


def validate_native_context(parent, metrics, key):
    count=parent.validate_native_queries(metrics,key)
    for row in metrics['native_input_queries']:
        a=row['attribute']
        require(a['fl']==0 and a['pass_through']==0,'Native query neutral metadata/pass-through changed')
    validate_log(metrics['native_input_query_log'])
    require(metrics['synthetic'] is False and metrics['collect_perf_detail'] is False
        and metrics['core_mask']==7 and metrics['SDK_version'].startswith('2.3.2 ')
        and metrics['driver_version']=='0.7.2','Actual fixed SDK/core/mode identity differs')
    return count


def validate_group(parent, group):
    packed=parent.validate_pack_records(group,'B')
    require(group['execution_status']=='completed' and group['mode']=='normal'
        and group['outputs_released'] is True and group['outputs_finite'] is True
        and group['weight_domain_valid'] is True and group['all_outputs_zero'] is False
        and group['timing']['synchronous'] is True
        and group['input_transport']==parent.NATIVE
        and group['diagnostic_allow_native_fp16'] is True
        and group['transport_equivalence_validated'] is False,
        'SDK native completion/finite/domain/release or diagnostic labels differ')
    validate_log(group['native_log'])
    for field in TIMINGS:
        value=group['timing'][field]
        require(isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value) and value>=0,'Nonfinite/negative SDK interval')
    frame_id=group['timing']['frame_id']
    require(isinstance(frame_id,int) and not isinstance(frame_id,bool) and 0<=frame_id<2**64,
        'Malformed synchronous SDK frame ID')
    require(sum(group['timing'][f] for f in TIMINGS[:-1])<=group['timing']['forward_seconds']+1e-6,
        'Subintervals exceed containing SDK forward clock')
    return packed


def validate_descriptors(row):
    e=row['edges'];require(isinstance(e,int) and not isinstance(e,bool) and e>0,'Positive edge count required')
    inputs={n:dict(shape=[1,e,c],dtype='float32',bytes=e*c*4) for n,c in [('net',384),('inp',384),('corr',882)]}
    inputs.update({n:dict(shape=[e],dtype='int64',bytes=e*8) for n in ('ii','jj','kk')})
    outputs={n:dict(shape=[1,e,c],dtype='float32',bytes=e*c*4) for n,c in [('net',384),('delta',2),('weight',2)]}
    require(row['inputs']==inputs and row['outputs']==outputs,'Full input/output shape/dtype/bytes descriptor differs')


def validate_records(parent, records, owner_calls):
    require(len(records)==len(owner_calls),'Saved owner/callback record count differs')
    shapes=Counter();pack_count=0;full=0;probe=0;intervals=Counter()
    for index,(row,owned) in enumerate(zip(records,owner_calls)):
        require(row['index']==index and {k:v for k,v in row.items() if k not in ('frame_index','stage')}==owned,
            'Callback index or copied owner record differs')
        require(row['status']=='completed_diagnostic_not_adopted' and row['candidate_accepted'] is False
            and row['SDK_FP32_equivalence_validated'] is False and row['historical_SDK_strict_FP32_rejected'] is True
            and row['Torch_fallback'] is False and row['outputs_committed_by_callback'] is False,
            'Callback numerical rejection/fallback/state ownership differs')
        validate_descriptors(row)
        if row['route']=='motion_probe':
            require(row['edges']==16 and row['explicit_startup_route'] is True
                and row['execution_backend']=='original_Torch_startup_probe' and row['NPU_used'] is False
                and row['fullgraph_returned'] is False and 'SDK_execution' not in row,
                'Only explicit original Torch D1 startup is allowed')
            probe+=1;continue
        require(row['route']=='full' and row['execution_backend']=='RKNN_complete_source_groups'
            and row['NPU_used'] is True and row['fullgraph_returned'] is True
            and row['strict_FP32_gate_passed'] is None,'Full Update must use diagnostic RKNN without reference')
        details=row['SDK_execution'];groups=details['groups'];plan=row['degree_plan']
        require(details['candidate_accepted'] is False and details['VO_state_committed'] is False
            and details['full_graph_returned'] is True and details['padding'] is False
            and details['dropped_edges'] is False and details['versus_reference'] is None
            and details['strict_FP32_gate_passed'] is None and details['sequential_groups']==len(groups)
            and row['model_keys']==[g['model_key'] for g in groups]
            and plan['all_edges_exactly_once'] is True and plan['original_edge_order_restored'] is True
            and plan['padding'] is False and plan['edges']==row['edges'],
            'Complete immutable graph/scatter/no reference declaration differs')
        for group in groups:
            key=group['model_key'];b,d,m=parent.dimensions(key)
            pack_count+=validate_group(parent,group);shapes[key]+=1
            for field in TIMINGS:intervals[field]+=group['timing'][field]
        require(sum(b*d*m for b,d,m in map(parent.dimensions,row['model_keys']))==row['edges'],
            'Groups do not cover the full declared original edge count')
        require(0<=sum(g['timing']['forward_seconds'] for g in groups)<=details['full_graph_seconds']+1e-6
            and details['full_graph_seconds']<=row['callback_seconds']+1e-6,
            'Nested SDK/fullgraph/callback clocks inconsistent')
        full+=1
    return dict(full_callbacks=full,motion_probe_callbacks=probe,SDK_calls=sum(shapes.values()),
        native_pack_records=pack_count,shape_histogram=dict(shapes),nested_SDK_interval_sums_seconds=dict(intervals))


def audit(run, index, parent):
    require(run['schema']=='v44_cached_undistort_native_FP16_online_RKNN_research_replay'
        and run['status']=='completed_unpaced_half_online_RKNN_research_not_realtime'
        and run['script_sha256']==DRIVER_SHA256 and run['online_factory_sha256']==FACTORY_SHA256,
        'Exact completed v44 driver/factory receipt required')
    require(run['frames_requested']==run['frames_completed']==len(run['frames'])==192
        and run['seed']==20261011 and run['cleanup_errors']==[] and run['affinity_restored'] is True,
        'Complete replay/source seed/cleanup required')
    require(run['candidate_accepted'] is False and run['adopted'] is False and run['realtime_validated'] is False
        and run['historical_SDK_strict_FP32_rejected'] is True and run['SDK_FP32_equivalence_validated'] is False
        and run['FP32_thresholds_relaxed'] is False and run['output_clipping'] is False
        and run['complete_Update_Torch_fallback'] is False and run['original_VGGT_default_unchanged'] is True
        and run['diagnostic_allow_native_fp16'] is True and run['native_input_transport_added'] is True
        and run['closed_flag_optimization_added'] is False,'Native transport cannot clear Torch/quality rejection')
    require(run['original_weights_unchanged'] is True and len(run['original_weights_before'])==94
        and run['original_weights_before']==run['original_weights_after'],'Original94 parameter metadata differs')
    require(run['sources_and_inputs_unchanged'] is True,'Recorded source/input verification failed')
    validate_source_maps(run['source_pins'],run['source_hashes_after'])
    validate_required_pins(run['source_pins'],REQUIRED_SOURCES)
    source_pairs={(Path(p).name,d) for p,d in run['source_pins'].items()}
    require(('run.json',FULL_AB_SHA256) in source_pairs
        and ('build.json',BUILD_SHA256) in source_pairs,'Required frozen native/SDK/source prerequisite pins absent')
    provenance=run['online_catalogue_provenance']
    require(provenance['compile_index_sha256']==INDEX_SHA256
        and provenance['native_full_saved_AB_prerequisite_sha256']==FULL_AB_SHA256
        and provenance['native_transport_sha256']==REQUIRED_SOURCES['v43_native_fp16_grouped.py']
        and provenance['native_online_input_transport'] is True and provenance['closed_flag_optimization_added'] is False
        and provenance['candidate_accepted'] is False and provenance['historical_SDK_strict_FP32_rejected'] is True,
        'Exact nine catalogue/native prerequisite provenance differs')
    owner=run['online_metrics'];cache=owner['cache'];before=run['online_metrics_before_close']
    summary=validate_records(parent,run['online_update_records'],owner['calls'])
    require(owner['requests']==len(run['online_update_records'])==215
        and owner['full_attempts']==owner['full_completed']==summary['full_callbacks']==181
        and owner['motion_probe_attempts']==owner['motion_probe_completed']==summary['motion_probe_callbacks']==34
        and owner['failed_calls']==0 and owner['closed'] is True and owner['failed'] is False
        and len(run['motion_probe_Torch_calls'])==34 and owner['shape_histogram']==summary['shape_histogram'],
        'Owner callback/probe/full/shapes totals disagree')
    require(summary['SDK_calls']==965 and summary['native_pack_records']==2895
        and set(owner['model_keys'])==set(index['models'])==set(summary['shape_histogram'])==MODEL_KEYS,
        'Expected complete nine-model native online workload differs')
    require(cache['created']==len(cache['histories'])==9 and cache['hits']==956 and cache['evicted']==0
        and cache['closed'] is True and cache['cleanup_errors']==[] and cache['resident_contexts']==0
        and cache['sessions']=={} and cache['current_DMA_bytes']==cache['current_model_host_bytes']==0
        and before['calls']==owner['calls'] and before['cache']['resident_contexts']==9,
        'Context cache reuse/close/declarations disagree')
    query_count=0;released=0;model_bytes=0;dma_bytes=0;models={};seen=set()
    for history in cache['histories']:
        key=history['model_key'];metrics=history['metrics'];require(key not in seen,'Duplicate context history')
        seen.add(key);query_count+=validate_native_context(parent,metrics,key)
        validate_log(metrics['initialization_native_log'])
        validate_source_maps(metrics['source_pins'],metrics['source_hashes_after'])
        validate_required_pins(metrics['source_pins'],SDK_SOURCES)
        require(metrics['closed'] is True and metrics['context_destroyed'] is True and metrics['cleanup_errors']==[]
            and metrics['current_DMA_bytes']==metrics['current_model_host_bytes']==0
            and metrics['sources_and_inputs_unchanged'] is True and metrics['npu_used'] is True
            and metrics['sdk_run_api_calls']==metrics['completed_calls']==metrics['SDK_outputs_release_attempts']
                ==metrics['SDK_outputs_releases']==summary['shape_histogram'][key],
            'Context native call/release/destroy/source totals disagree')
        artifact=index['models'][key]['model']
        model_bytes+=validate_model_binding(metrics,index['models'][key])
        released+=metrics['SDK_outputs_releases']
        dma_bytes+=metrics['memory']['total_dma_allocated_size'];models[key]=artifact['sha256']
    require(seen==MODEL_KEYS and query_count==54 and released==965
        and model_bytes==cache['peak_resident_model_bytes'] and dma_bytes==cache['peak_resident_DMA_bytes']
        and model_bytes<=cache['max_model_bytes'] and dma_bytes<=cache['max_dma_bytes']
        and cache['max_contexts']==cache['peak_resident_contexts']==9,'Native query/model/DMA bounds disagree')
    activity=run['SDK_activity'];encoder=run['encoder_metrics']
    require(activity['Update_sdk_run_api_calls']==released and activity['encoder_sdk_run_api_calls']==192
        and activity['Update_NPU_used'] is True and activity['encoder_NPU_used'] is True
        and activity['internal_all_NPU_verified'] is False,'Actual SDK counters or placement claim differs')
    require(encoder['sdk_run_api_calls']==encoder['successful_sdk_frames']==encoder['completed_frames']==192
        and encoder['closed'] is True and encoder['cleanup_errors']==[]
        and encoder['context']['closed'] is True and encoder['context']['context_destroyed'] is True
        and encoder['context']['completed_output_copy_release_calls']==192,
        'Encoder calls/copy/release/destroy declaration differs')
    for row in encoder['calls']:
        validate_log(row['native_log']);require(row['completed'] is True
            and row['SDK_timing']['synchronous'] is True and row['SDK_timing']['output_buffers_released'] is True,
            'Encoder synchronous release/log guard differs')
    require(run['correlation']['closed'] is True and run['correlation']['current_context_bytes']==0
        and run['undistort_cache']['closed'] is True and run['undistort_cache']['current_map_bytes']==0,
        'Native correlation/undistort storage not released')
    return dict(schema='v44_native_online_saved_transport_records_independent_review',
        status='passed_saved_native_records_not_numerical_acceptance',candidate_accepted=False,
        historical_SDK_strict_FP32_rejected=True,original_Torch_quality_validated_by_this_audit=False,
        numerical_arrays_audited_by_this_audit=False,frames=192,**summary,native_contexts=9,
        native_attribute_queries=query_count,model_sha256_by_key=models,
        lifecycle=dict(Update_outputs_releases=released,Update_contexts_destroyed=9,encoder_SDK_calls=192,
            encoder_copied_released_calls=192,encoder_context_destroyed=True,current_DMA_bytes=0,current_model_host_bytes=0,
            peak_model_bytes=model_bytes,compiled_model_payload_bytes=model_bytes-9,
            host_model_buffer_NUL_bytes=9,peak_DMA_bytes=dma_bytes,cleanup_errors=[]),
        frozen_source_sha256=REQUIRED_SOURCES,frozen_per_context_SDK_source_sha256=SDK_SOURCES,
        compile_index_sha256=INDEX_SHA256,
        limitations=['Only terminal saved JSON and frozen source/index bytes; no model/ORT/SDK/BA execution.',
            'All 54 queries and 2895 input descriptors checked; per-call native payload bytes were not saved and are not re-converted.',
            'Recorded API/log/release/destroy/source declarations are cross-checked across callbacks and all nine context histories. Board source/model binaries are not independently rehashed on the host.',
            'The frozen runtime checks run/get synchronous frame-ID equality; the saved single frame_id is not a monotonic API-call counter.',
            'Original Torch thresholds/quality and saved-array/trajectory parity are separate audits. Historical rejection remains.',
            'Timing intervals are nested diagnostics, not additive to complete replay or a 20Hz/controlled-thermal claim.'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--run-sha256',required=True)
    p.add_argument('--compile-index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    default=Path(__file__).resolve().parent.parent/'dpvo_native_input/audit-v43-native-full-ab.py'
    if not default.exists():default=Path(__file__).resolve().parent.parent/'public-snapshot/deploy/rk3588/validation/dpvo_native_input/audit-v43-native-full-ab.py'
    p.add_argument('--native-auditor',type=Path,default=default)
    p.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent)
    a=p.parse_args(argv);require(not a.output.exists(),'New audit output required')
    try:
        path=a.run_dir/'run.json';require(sha(path)==a.run_sha256,'Actual saved run SHA differs')
        require(sha(a.compile_index)==INDEX_SHA256,'Frozen nine-model index SHA differs')
        for name,want in [('run-v44-online-native-board.py',DRIVER_SHA256),('v44_online_factory.py',FACTORY_SHA256)]:
            require(sha(a.source_dir/name)==want,'Frozen online execution source bytes differ')
        result=audit(read_json(path),read_json(a.compile_index),load_parent(a.native_auditor))
        result.update(auditor_sha256=sha(__file__),parent_auditor_sha256=PARENT_AUDITOR_SHA256,
            actual_run_sha256=a.run_sha256,independent_board_binary_rehash=False)
    except Exception as error:
        result=dict(status='failed_saved_transport_record_audit',error=repr(error),auditor_sha256=sha(__file__),candidate_accepted=False)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','error','SDK_calls','native_pack_records','native_attribute_queries')}))
    return int(result['status']=='failed_saved_transport_record_audit')


if __name__=='__main__':raise SystemExit(main())
