"""Generated compatibility/header/hash rejection tests; no model/SDK/data."""
import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest

TEST=Path(__file__).resolve().parent
HERE=Path(os.environ.get('V45_ONLINE_SOURCE',TEST))
if not (HERE/'audit-v45-closed-native-online-records.py').exists():
    HERE=TEST.parent/'deploy/rk3588/validation/dpvo_online_replay'
PARENT=HERE/'audit-v44-native-online-records.py'
if not PARENT.exists():PARENT=HERE.parent/'public-snapshot/deploy/rk3588/validation/dpvo_online_replay/audit-v44-native-online-records.py'
spec=importlib.util.spec_from_file_location('_v45_records',HERE/'audit-v45-closed-native-online-records.py')
auditor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=auditor;spec.loader.exec_module(auditor)


def generated():
    proof=dict(parent_runtime_sha256=auditor.PARENT_RUNTIME_SHA256,
        changed_method='BoundedContextCache._get',changed_expression_count=1,
        all_other_get_AST_preserved=True,cold_and_budget_full_metrics_preserved=True,
        lifecycle_full_metrics_preserved=True,numerical_execution_changed=False,
        native_pack_or_SDK_changed=False,candidate_accepted=False)
    p=dict(closed_flag_optimization_added=True,cache_closed_flag_only=True,
        closed_flag_helper_sha256=auditor.CLOSED_FLAG_SHA256,
        closed_full_saved_AB_prerequisite_sha256=auditor.CLOSED_AB_SHA256,
        original_v44_factory_sha256=auditor.V44_FACTORY_SHA256,
        live_closed_query_trajectory_validated=False,candidate_accepted=False,
        numerical_execution_changed=False,closed_flag_source_proof=proof)
    return dict(schema=auditor.SCHEMA,script_sha256=auditor.DRIVER_SHA256,
        online_factory_sha256=auditor.FACTORY_SHA256,
        original_v44_online_driver_sha256=auditor.V44_DRIVER_SHA256,
        diagnostic_allow_closed_flag=True,closed_flag_optimization_added=True,
        closed_flag_helper_sha256=auditor.CLOSED_FLAG_SHA256,
        closed_full_saved_AB_prerequisite_sha256=auditor.CLOSED_AB_SHA256,
        online_catalogue_provenance=p,
        source_pins=dict({f'/generated/{n}':h for n,h in auditor.EXTRA_SOURCES.items()},
            **{'/generated/closedAB/run.json':auditor.CLOSED_AB_SHA256}),
        online_update_records=[dict(generated=True)],online_metrics=dict(generated=True),
        source_hashes_after=dict(generated=True))


def test_only_validated_schema_cache_declarations_mapped_no_ledger_or_source_changes():
    run=generated();before=copy.deepcopy(run);view=auditor.parent_view(run)
    assert run==before
    assert view['online_update_records'] is run['online_update_records']
    assert view['online_metrics'] is run['online_metrics']
    assert view['source_pins'] is run['source_pins']
    assert view['source_hashes_after'] is run['source_hashes_after']
    assert view['online_catalogue_provenance']['closed_flag_source_proof'] is run['online_catalogue_provenance']['closed_flag_source_proof']
    changed={k for k in view if view[k]!=run[k]}
    assert changed=={'schema','closed_flag_optimization_added','online_catalogue_provenance'}


@pytest.mark.parametrize('change',('flag','prerequisite','proof','native_math','accepted'))
def test_parent_compatibility_view_cannot_hide_changed_new_contract(change):
    run=generated();p=run['online_catalogue_provenance']
    if change=='flag':run['closed_flag_optimization_added']=False
    if change=='prerequisite':run['closed_full_saved_AB_prerequisite_sha256']='0'*64
    if change=='proof':p['closed_flag_source_proof']['changed_expression_count']=2
    if change=='native_math':p['numerical_execution_changed']=True
    if change=='accepted':p['candidate_accepted']=True
    with pytest.raises(ValueError):auditor.parent_view(run)


def test_frozen_parent_is_private_all_original_guard_functions_and_SDK_pins_retained():
    a=auditor.load_v44(PARENT);b=auditor.load_v44(PARENT)
    assert a is not b and a.audit.__code__.co_code==b.audit.__code__.co_code
    assert a.DRIVER_SHA256==auditor.DRIVER_SHA256 and a.FACTORY_SHA256==auditor.FACTORY_SHA256
    assert a.REQUIRED_SOURCES['run-v44-online-native-board.py']==auditor.V44_DRIVER_SHA256
    assert a.REQUIRED_SOURCES['v44_online_factory.py']==auditor.V44_FACTORY_SHA256
    assert a.REQUIRED_SOURCES['v44_closed_flag_cache.py']==auditor.CLOSED_FLAG_SHA256
    assert a.SDK_SOURCES==b.SDK_SOURCES and a.INDEX_SHA256==b.INDEX_SHA256
    assert auditor.sha(PARENT)==auditor.V44_AUDITOR_SHA256


def test_changed_parent_bytes_rejected_before_import(tmp_path):
    path=tmp_path/'fake.py';path.write_text('raise RuntimeError("must not import")')
    with pytest.raises(ValueError,match='SHA differs'):auditor.load_v44(path)
