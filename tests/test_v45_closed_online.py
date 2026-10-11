"""Generated prerequisite/fake cache/AST integration tests; no model or SDK."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

TEST = Path(__file__).resolve().parent
HERE = Path(os.environ.get('V45_ONLINE_SOURCE', TEST))
if not (HERE/'v45_online_factory.py').exists():
    HERE = TEST.parent/'deploy/rk3588/validation/dpvo_online_replay'
VALIDATION = Path(os.environ.get('V45_VALIDATION', HERE.parent))
if not (VALIDATION/'dpvo_native_input/run-v43-native-full-ab-board.py').exists():
    VALIDATION = HERE.parent/'public-snapshot/deploy/rk3588/validation'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


factory = load(HERE/'v45_online_factory.py', '_v45_fake_factory')
driver = load(HERE/'run-v45-online-native-closed-board.py', '_v45_fake_driver')
harness = load(VALIDATION/'dpvo_native_input/run-v43-native-full-ab-board.py', '_v45_frozen_cache_swap')


def prerequisite():
    life = dict(sdk_run_api_calls=56, outputs_released=56, contexts_destroyed=6,
        closed=True, current_DMA_bytes=0, current_model_host_bytes=0)
    proof = dict(parent_runtime_sha256=factory.PARENT_RUNTIME_SHA256,
        changed_method='BoundedContextCache._get', changed_expression_count=1,
        all_other_get_AST_preserved=True, cold_and_budget_full_metrics_preserved=True,
        lifecycle_full_metrics_preserved=True, numerical_execution_changed=False,
        native_pack_or_SDK_changed=False, candidate_accepted=False)
    return dict(schema='v44_native_cache_closed_flag_three_saved_state_ABBA_diagnostic',
        status='completed_AB_bitwise_equal_diagnostic_not_adopted',
        script_sha256=factory.CLOSED_AB_DRIVER_SHA256, closed_flag_helper_sha256=factory.CLOSED_FLAG_SHA256,
        parent_runtime_sha256=factory.PARENT_RUNTIME_SHA256, native_harness_sha256=factory.NATIVE_HARNESS_SHA256,
        compile_index_sha256=factory.INDEX_SHA256,
        callback_sha256=dict(A=factory.CALLBACK_SHA256, B=factory.CALLBACK_SHA256),
        same_callback_for_both_methods=True, same_native_input_transport=True,
        only_cache_live_hit_closed_query_changed=True, same_session_source_pins_for_both_methods=True,
        A_B_all_saved_outputs_bitwise_equal=True, closed_flag_bitwise_for_three_saved_states_only=True,
        historical_SDK_strict_FP32_rejected=True, strict_FP32_equivalence_validated=False,
        candidate_accepted=False, CPU_teacher_used=False, recurrent_feedback=False, BA_executed=False,
        sources_and_inputs_unchanged=True, affinity_restored=True, cleanup_errors=[],
        source_pins={'generated':'1'*64}, source_hashes_after={'generated':'1'*64},
        calls=[dict(observation_id=o, method=m, bitwise_equal_first_A_for_state=True)
            for o in (34,127,191) for m in ('A','B','B','A')*2],
        lifecycle={m:copy.deepcopy(life) for m in ('A','B')}, closed_flag_source_proof=proof)


def test_generated_executed_prerequisite_contract():
    factory.validate_prerequisite(prerequisite())


@pytest.mark.parametrize('change', ('bitwise','schedule','release','cleanup','source','native','proof','accepted'))
def test_partial_changed_or_numerically_accepted_prerequisite_rejected(change):
    row = prerequisite()
    if change=='bitwise': row['calls'][0]['bitwise_equal_first_A_for_state']=False
    if change=='schedule': row['calls'].pop()
    if change=='release': row['lifecycle']['B']['outputs_released']=55
    if change=='cleanup': row['cleanup_errors']=['generated cleanup failure']
    if change=='source': row['source_hashes_after']['generated']='2'*64
    if change=='native': row['same_native_input_transport']=False
    if change=='proof': row['closed_flag_source_proof']['changed_expression_count']=2
    if change=='accepted': row['candidate_accepted']=True
    with pytest.raises(ValueError): factory.validate_prerequisite(row)


class EmptyCache:
    def __init__(self):
        self.specs={k:dict(model_path=k,model_sha256='generated',inputs=[],outputs=[],
            export_strict_FP32_gate_passed=True) for k in harness.MODEL_KEYS}
        self.state=dict(created=0,resident_contexts=0,current_DMA_bytes=0,
            current_model_host_bytes=0,cleanup_errors=[],closed=False)
        self.close_calls=0
    def metrics(self): return copy.deepcopy(self.state)
    def close(self): self.close_calls+=1;self.state['closed']=True


class ClosedCache(EmptyCache):
    def __init__(self,specs,open_session,**budgets):
        super().__init__();self.specs=specs;self.open_session=open_session;self.budgets=budgets


@pytest.fixture
def stack(tmp_path,monkeypatch):
    receipt=tmp_path/'closed.json';receipt.write_text(json.dumps(prerequisite()))
    kwargs=dict(v44_factory=tmp_path/'v44.py',v44_factory_sha256=factory.PARENT_FACTORY_SHA256,
        closed_flag_helper=tmp_path/'closed.py',closed_flag_helper_sha256=factory.CLOSED_FLAG_SHA256,
        closed_ab_receipt=receipt,closed_ab_receipt_sha256=factory.CLOSED_AB_SHA256,
        diagnostic_allow_closed_flag=True,diagnostic_allow_native_fp16=True,
        diagnostic_allow_fp32_rejected=True,native_ab_harness=tmp_path/'harness.py',
        native_ab_harness_sha256=factory.NATIVE_HARNESS_SHA256,
        callback_sha256=factory.CALLBACK_SHA256,compile_index_sha256=factory.INDEX_SHA256,
        max_contexts=9,max_model_bytes=256<<20,max_dma_bytes=512<<20,
        parent_runtime=tmp_path/'original-runtime.py',failure_dir=tmp_path,
        abi_path=tmp_path/'ABI.py',header_path=tmp_path/'header.h',library_path=tmp_path/'SDK.so',
        log_helper_path=tmp_path/'log.py',motion_probe=object(),on_record=object(),identity_reader=object())
    old=EmptyCache();seen=dict(parent=[],bind=[],session=[],checks=[])
    def open_session(*a,**k): seen['session'].append((a,k));return object()
    native=SimpleNamespace(BoundedContextCache=EmptyCache,
        GroupedSession=SimpleNamespace(open_board=open_session),source_pins={},
        execute_full_graph=object(),pack_inputs=object(),input_transport=factory.NATIVE_TRANSPORT)
    candidate=SimpleNamespace(**vars(native));candidate.BoundedContextCache=ClosedCache
    candidate.closed_flag_source_proof=prerequisite()['closed_flag_source_proof']
    class Owner:
        cache=old;runtime=native;route='full';source_pins={};close_calls=0
        catalogue_provenance=dict(candidate_accepted=False,native_online_input_transport=True,
            metadata_only_reporting=True,closed_flag_optimization_added=False)
        def close(self): self.close_calls+=1;self.cache.close()
    owner=Owner()
    def parent_open(**k):seen['parent'].append(k);return owner
    def bind(runtime,**k):seen['bind'].append((runtime,k));return candidate
    parent=SimpleNamespace(open_board_online_from_index=parent_open)
    closed=SimpleNamespace(bind_runtime=bind)
    def pin(path,expected,pins):
        path=Path(path).resolve();pins[str(path)]=expected;return path
    monkeypatch.setattr(factory,'_pin',pin)
    monkeypatch.setattr(factory,'_recheck',lambda pins:seen['checks'].append(dict(pins)))
    monkeypatch.setattr(factory,'_load',lambda p,e,label:
        dict(v44_factory=parent,closed_cache=closed,native_AB_harness=harness)[label])
    return kwargs,owner,old,native,candidate,seen


@pytest.mark.parametrize('flag', ('diagnostic_allow_closed_flag','diagnostic_allow_native_fp16','diagnostic_allow_fp32_rejected'))
def test_all_flags_required_before_source_IO_in_factory_and_driver(stack,monkeypatch,flag):
    kwargs,*_=stack;kwargs[flag]=False
    monkeypatch.setattr(factory,'_pin',lambda *a:pytest.fail('No read before explicit flags'))
    with pytest.raises(ValueError,match='All three'):factory.open_board_online_from_index(**kwargs)
    flags={name:True for name in ('diagnostic_allow_closed_flag','diagnostic_allow_native_fp16','diagnostic_allow_fp32_rejected')};flags[flag]=False
    monkeypatch.setattr(driver,'parse_args',lambda _:SimpleNamespace(**flags))
    with pytest.raises(ValueError,match='All three'):driver.main([])


def test_only_empty_native_cache_replaced_before_context_creation(stack):
    kwargs,owner,old,native,candidate,seen=stack
    assert factory.open_board_online_from_index(**kwargs) is owner
    assert owner.runtime is candidate and owner.cache is not old and old.close_calls==1
    assert set(owner.cache.specs)==harness.MODEL_KEYS and seen['session']==[]
    passed=seen['parent'][0]
    assert 'diagnostic_allow_closed_flag' not in passed and 'closed_ab_receipt' not in passed
    for key in ('motion_probe','on_record','identity_reader','failure_dir'):
        assert passed[key] is kwargs[key]
    assert seen['bind']==[(native,dict(parent_runtime_path=kwargs['parent_runtime'],diagnostic_allow_closed_flag=True))]
    owner.cache.open_session(owner.cache.specs['B2_D6_M16'])
    session_kwargs=seen['session'][0][1]
    assert session_kwargs['diagnostic_allow_native_fp16'] and session_kwargs['diagnostic_allow_fp32_rejected']
    assert session_kwargs['collect_perf_detail'] is False
    assert str(kwargs['closed_ab_receipt'].resolve()) in session_kwargs['pins']
    assert owner.catalogue_provenance['closed_flag_optimization_added'] is True
    assert owner.catalogue_provenance['candidate_accepted'] is False
    assert candidate.GroupedSession is native.GroupedSession
    assert candidate.pack_inputs is native.pack_inputs
    assert candidate.execute_full_graph is native.execute_full_graph
    owner.close();assert owner.close_calls==1 and owner.cache.metrics()['closed']


@pytest.mark.parametrize('live', ('created','resident_contexts'))
def test_no_live_cache_migration_and_owner_closed_on_rejection(stack,live):
    kwargs,owner,old,*_=stack;old.state[live]=1
    with pytest.raises(ValueError,match='unused validated'):factory.open_board_online_from_index(**kwargs)
    assert owner.close_calls==1 and owner.cache is old and old.metrics()['closed']


def test_native_session_change_or_source_failure_is_fatal_without_fallback(stack,monkeypatch):
    kwargs,owner,old,native,candidate,seen=stack
    candidate.GroupedSession=object()
    with pytest.raises(ValueError,match='Only the bounded cache class'):factory.open_board_online_from_index(**kwargs)
    assert owner.close_calls==1 and old.metrics()['closed']


def test_source_change_after_replacement_closes_new_cache(stack,monkeypatch):
    kwargs,owner,old,*_=stack;checks=[]
    def recheck(pins):
        checks.append(dict(pins))
        if len(checks)==2:raise ValueError('generated pinned source changed')
    monkeypatch.setattr(factory,'_recheck',recheck)
    with pytest.raises(ValueError,match='pinned source changed'):factory.open_board_online_from_index(**kwargs)
    assert owner.cache is not old and owner.close_calls==1 and owner.cache.metrics()['closed']


def test_exact_parent_hash_and_prerequisite_hash_required_before_import(tmp_path):
    path=tmp_path/'fake.py';path.write_text('raise RuntimeError("must not import")')
    with pytest.raises(ValueError,match='SHA differs'):factory._load(path,'0'*64,'generated')
    with pytest.raises(ValueError,match='artifact differs'):factory._pin(path,'0'*64,{})


def test_original_frame_loop_update_probe_RNG_EOF_and_finally_AST_unchanged():
    parent=VALIDATION/'dpvo_online_replay/run-v44-online-native-board.py'
    assert factory.sha256(parent)==driver.V44_ONLINE_DRIVER
    old=ast.parse(parent.read_text());new=ast.parse((HERE/'run-v45-online-native-closed-board.py').read_text())
    def functions(t):return {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
    a,b=functions(old),functions(new)
    for name in set(a)-{'main','parse_args'}:
        assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
    def nested(main):return {n.name:n for n in main.body if isinstance(n,ast.FunctionDef)}
    for name,node in nested(a['main']).items():
        assert ast.dump(node,include_attributes=False)==ast.dump(nested(b['main'])[name],include_attributes=False),name
    def loop(main):return next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts]==['index','item'])
    assert ast.dump(loop(a['main']),include_attributes=False)==ast.dump(loop(b['main']),include_attributes=False)
    oldtry=next(n for n in a['main'].body if isinstance(n,ast.Try));newtry=next(n for n in b['main'].body if isinstance(n,ast.Try))
    assert ast.dump(ast.Module(body=oldtry.finalbody,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=newtry.finalbody,type_ignores=[]),include_attributes=False)
    def rngnodes(main):return [n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('rng','generator','terminated') for t in n.targets)]
    assert [ast.dump(n,include_attributes=False) for n in rngnodes(a['main'])]==[ast.dump(n,include_attributes=False) for n in rngnodes(b['main'])]
    assert driver.POLICY=='cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
    assert driver.ONLINE_FACTORY==factory.sha256(HERE/'v45_online_factory.py')
