#!/usr/bin/env python3
"""Saved v45 closed-query/native transport audit; no SDK/model/ORT execution.

The frozen v44 auditor retains every original transport, source, lifecycle,
counter and rejection check. The new schema/cache declarations are first
checked independently, then only those declarations are mapped for the private
parent audit. Ledger/source/input/output records are never altered.
DPVO: Copyright (c) 2022 Princeton University; adjacent MIT LICENSE.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

V44_AUDITOR_SHA256 = '0dec30e487637a31d3ab86c9d5959c76427ad8c9e65d24f2761e115ccf1ce355'
V44_DRIVER_SHA256 = '68d44f3722ea37310084d903bcaf72520ca3ef72cd794e6f972abd8fd45c93ab'
V44_FACTORY_SHA256 = 'ac2b29b7eabc17d1dd47ec7a89734d6c1512dcbf37845ef5a32e9667b9b6be79'
DRIVER_SHA256 = 'd51b96d4dc3da24e4f0d284d539593020dd6e14b5ad3f974b81256e460568a2c'
FACTORY_SHA256 = '0e573fd8d917b59bcdd843d66db0e8136cef8c076b6452595bec61251329af56'
CLOSED_FLAG_SHA256 = '014685c3b7736070bf3b1a1e3a333a928468e013cf454e4acac4ee53da1354b2'
CLOSED_AB_SHA256 = '1a2132b70849fe823e3d85617d588f2d1dd5626326adfa4c14c0e718d2f7faf4'
PARENT_RUNTIME_SHA256 = '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'
SCHEMA = 'v45_cached_undistort_native_FP16_closed_flag_online_RKNN_research_replay'
EXTRA_SOURCES = {'run-v45-online-native-closed-board.py':DRIVER_SHA256,
    'v45_online_factory.py':FACTORY_SHA256, 'v44_closed_flag_cache.py':CLOSED_FLAG_SHA256}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def load_v44(path):
    require(sha(path)==V44_AUDITOR_SHA256,'Frozen v44 auditor source SHA differs')
    name='_v45_saved_audit_parent_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value)
    # Only this private module's experiment identity is rebound. Its function
    # implementations, SDK/source guards and frozen file bytes remain intact.
    value.DRIVER_SHA256=DRIVER_SHA256;value.FACTORY_SHA256=FACTORY_SHA256
    value.REQUIRED_SOURCES=dict(value.REQUIRED_SOURCES,**EXTRA_SOURCES)
    return value


def validate_closed_integration(run):
    require(run['schema']==SCHEMA and run['script_sha256']==DRIVER_SHA256
        and run['online_factory_sha256']==FACTORY_SHA256
        and run['original_v44_online_driver_sha256']==V44_DRIVER_SHA256
        and run['diagnostic_allow_closed_flag'] is True
        and run['closed_flag_optimization_added'] is True
        and run['closed_flag_helper_sha256']==CLOSED_FLAG_SHA256
        and run['closed_full_saved_AB_prerequisite_sha256']==CLOSED_AB_SHA256,
        'Exact v45 closed-query schema/source/diagnostic prerequisite required')
    p=run['online_catalogue_provenance']
    require(p['closed_flag_optimization_added'] is True and p['cache_closed_flag_only'] is True
        and p['closed_flag_helper_sha256']==CLOSED_FLAG_SHA256
        and p['closed_full_saved_AB_prerequisite_sha256']==CLOSED_AB_SHA256
        and p['original_v44_factory_sha256']==V44_FACTORY_SHA256
        and p['live_closed_query_trajectory_validated'] is False
        and p['candidate_accepted'] is False and p['numerical_execution_changed'] is False,
        'Closed-query runtime binding may not change neural/transport acceptance')
    proof=p['closed_flag_source_proof']
    require(proof['parent_runtime_sha256']==PARENT_RUNTIME_SHA256
        and proof['changed_method']=='BoundedContextCache._get'
        and proof['changed_expression_count']==1 and proof['all_other_get_AST_preserved'] is True
        and proof['cold_and_budget_full_metrics_preserved'] is True
        and proof['lifecycle_full_metrics_preserved'] is True
        and proof['numerical_execution_changed'] is False
        and proof['native_pack_or_SDK_changed'] is False and proof['candidate_accepted'] is False,
        'Frozen cache-only source proof differs')
    pairs={(Path(path).name,d) for path,d in run['source_pins'].items()}
    require(set(EXTRA_SOURCES.items())<=pairs and ('run.json',CLOSED_AB_SHA256) in pairs,
        'Closed integration source/actual AB prerequisite pins missing')
    return proof


def parent_view(run):
    """Compatibility view after explicit new-schema guards; no ledger copies."""
    validate_closed_integration(run)
    result=dict(run,schema='v44_cached_undistort_native_FP16_online_RKNN_research_replay',
        closed_flag_optimization_added=False)
    result['online_catalogue_provenance']=dict(run['online_catalogue_provenance'],
        closed_flag_optimization_added=False)
    return result


def audit(run,index,v44,native_parent):
    proof=validate_closed_integration(run)
    result=v44.audit(parent_view(run),index,native_parent)
    result.update(schema='v45_closed_native_online_saved_transport_records_independent_review',
        status='passed_saved_closed_native_records_not_numerical_acceptance',
        source_run_schema=SCHEMA,closed_flag_optimization_checked=True,
        closed_flag_helper_sha256=CLOSED_FLAG_SHA256,closed_AB_prerequisite_sha256=CLOSED_AB_SHA256,
        cache_source_proof=proof,parent_v44_auditor_sha256=V44_AUDITOR_SHA256,
        parent_compatibility_view_fields=['schema','closed_flag_optimization_added',
            'online_catalogue_provenance.closed_flag_optimization_added'],
        ledger_source_input_output_records_modified=False)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--run-sha256',required=True)
    p.add_argument('--compile-index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    here=Path(__file__).resolve().parent
    default=here/'audit-v44-native-online-records.py'
    if not default.exists():default=here.parent/'public-snapshot/deploy/rk3588/validation/dpvo_online_replay/audit-v44-native-online-records.py'
    p.add_argument('--v44-auditor',type=Path,default=default)
    native=here.parent/'dpvo_native_input/audit-v43-native-full-ab.py'
    if not native.exists():native=here.parent/'public-snapshot/deploy/rk3588/validation/dpvo_native_input/audit-v43-native-full-ab.py'
    p.add_argument('--native-auditor',type=Path,default=native)
    p.add_argument('--source-dir',type=Path,default=here)
    a=p.parse_args(argv);require(not a.output.exists(),'New audit output required')
    try:
        v44=load_v44(a.v44_auditor);run=a.run_dir/'run.json'
        require(sha(run)==a.run_sha256,'Actual saved run SHA differs')
        require(sha(a.compile_index)==v44.INDEX_SHA256,'Frozen nine-model index SHA differs')
        for name,want in [('run-v45-online-native-closed-board.py',DRIVER_SHA256),('v45_online_factory.py',FACTORY_SHA256)]:
            require(sha(a.source_dir/name)==want,'Frozen online execution source bytes differ')
        result=audit(v44.read_json(run),v44.read_json(a.compile_index),v44,v44.load_parent(a.native_auditor))
        result.update(auditor_sha256=sha(__file__),actual_run_sha256=a.run_sha256,
            parent_native_auditor_sha256=v44.PARENT_AUDITOR_SHA256,independent_board_binary_rehash=False)
    except Exception as error:
        result=dict(status='failed_saved_closed_transport_record_audit',error=repr(error),
            auditor_sha256=sha(__file__),candidate_accepted=False)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','error','SDK_calls','native_pack_records','native_attribute_queries')}))
    return int(result['status']=='failed_saved_closed_transport_record_audit')


if __name__=='__main__':raise SystemExit(main())
