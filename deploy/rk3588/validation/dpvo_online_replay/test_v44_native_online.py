"""Generated metadata/fake cache and AST tests only; no neural/SDK execution."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

TEST = Path(__file__).resolve().parent
HERE = Path(os.environ.get('V44_ONLINE_SOURCE', TEST))
if not (HERE/'v44_online_factory.py').is_file():
    HERE = TEST.parent/'deploy/rk3588/validation/dpvo_online_replay'
VALIDATION = Path(os.environ.get('V44_VALIDATION', HERE.parent))
if not (VALIDATION/'dpvo_online_replay/v43_online_factory.py').is_file():
    VALIDATION = HERE.parent/'public-snapshot/deploy/rk3588/validation'
ANCESTOR = VALIDATION/'dpvo_online_replay'
NATIVE = VALIDATION/'dpvo_native_input'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


factory = load(HERE/'v44_online_factory.py', '_v44_fake_factory')
driver = load(HERE/'run-v44-online-native-board.py', '_v44_fake_driver')
harness = load(NATIVE/'run-v43-native-full-ab-board.py', '_v44_frozen_AB_helper')


def prerequisites():
    lifecycle = dict(sdk_run_api_calls=56, outputs_released=56, contexts_destroyed=6,
        closed=True, current_DMA_bytes=0, current_model_host_bytes=0)
    full = dict(schema='v43_native_three_complete_saved_state_ABBA_diagnostic',
        status='completed_AB_bitwise_equal_diagnostic_not_adopted', candidate_accepted=False,
        historical_SDK_strict_FP32_rejected=True, same_callback_for_both_methods=True,
        only_input_transport_changed=True, A_B_all_saved_outputs_bitwise_equal=True,
        native_transport_bitwise_for_three_saved_states_only=True,
        sources_and_inputs_unchanged=True, affinity_restored=True, cleanup_errors=[],
        script_sha256=factory.AB_HARNESS_SHA256, factory_sha256=factory.PARENT_FACTORY_SHA256,
        callback_sha256=dict(A=factory.CALLBACK_SHA256, B=factory.CALLBACK_SHA256),
        compile_index_sha256=factory.INDEX_SHA256,
        calls=[dict(observation_id=obs, method=method, bitwise_equal_first_A_for_state=True)
            for obs in (34,127,191) for method in ('A','B','B','A')*2],
        lifecycle={method:copy.deepcopy(lifecycle) for method in ('A','B')})
    build = dict(status='compiled_ABI_loaded_conversion_not_yet_verified',
        output=dict(sha256=factory.PACKING_LIBRARY_SHA256), sources_unchanged=True, cleanup_errors=[])
    conversion = dict(status='passed_conversion_reference_and_ownership_only',
        library_sha256=factory.PACKING_LIBRARY_SHA256, query_sha256=factory.QUERY_SHA256,
        packer_sha256=factory.PACKER_SHA256, cleanup_errors=[])
    return full, build, conversion


def test_generated_complete_prerequisite_metadata_passes():
    factory.validate_prerequisites(*prerequisites())


@pytest.mark.parametrize('failure', ('bitwise','calls','schedule','release','DMA','cleanup','callback','library'))
def test_prerequisite_rejects_partial_or_different_execution(failure):
    full, build, conversion = prerequisites()
    if failure == 'bitwise': full['calls'][0]['bitwise_equal_first_A_for_state'] = False
    if failure == 'calls': full['calls'].pop()
    if failure == 'schedule': full['calls'][1]['method'] = 'A'
    if failure == 'release': full['lifecycle']['B']['outputs_released'] = 55
    if failure == 'DMA': full['lifecycle']['A']['current_DMA_bytes'] = 1
    if failure == 'cleanup': full['cleanup_errors'] = ['Generated release failure']
    if failure == 'callback': full['callback_sha256']['B'] = 'different'
    if failure == 'library': conversion['library_sha256'] = 'different'
    with pytest.raises(ValueError): factory.validate_prerequisites(full, build, conversion)


def arguments(tmp_path):
    import json
    full, build, conversion = prerequisites()
    for name, row in [('full.json',full),('build.json',build),('conversion.json',conversion)]:
        (tmp_path/name).write_text(json.dumps(row))
    paths = dict(v43_factory=tmp_path/'v43.py', native_ab_harness=tmp_path/'harness.py',
        transport=tmp_path/'transport.py', packer=tmp_path/'packer.py',
        packing_library=tmp_path/'native.so', query=tmp_path/'query.json',
        packing_build_receipt=tmp_path/'build.json', conversion_receipt=tmp_path/'conversion.json',
        full_ab_receipt=tmp_path/'full.json')
    digests = dict(v43_factory=factory.PARENT_FACTORY_SHA256, native_ab_harness=factory.AB_HARNESS_SHA256,
        transport=factory.TRANSPORT_SHA256, packer=factory.PACKER_SHA256,
        packing_library=factory.PACKING_LIBRARY_SHA256, query=factory.QUERY_SHA256,
        packing_build_receipt=factory.BUILD_SHA256, conversion_receipt=factory.CONVERSION_SHA256,
        full_ab_receipt=factory.FULL_AB_SHA256)
    return dict(paths, **{k+'_sha256':v for k,v in digests.items()},
        diagnostic_allow_native_fp16=True, diagnostic_allow_fp32_rejected=True,
        callback_sha256=factory.CALLBACK_SHA256, compile_index_sha256=factory.INDEX_SHA256,
        max_contexts=9, max_model_bytes=256<<20, max_dma_bytes=512<<20,
        parent_runtime=tmp_path/'original-runtime.py', contracts=tmp_path/'contracts.py',
        contracts_sha256='fixed-contract', parent_contracts=tmp_path/'parent.py',
        runtime_bridge=tmp_path/'bridge.py', abi_path=tmp_path/'ABI.py',
        header_path=tmp_path/'header.h', library_path=tmp_path/'SDK.so',
        log_helper_path=tmp_path/'log.py', failure_dir=tmp_path,
        motion_probe=object(), on_record=object(), identity_reader=object())


class EmptyCache:
    def __init__(self):
        self.specs = {k:dict(model_path=k, model_sha256='generated', inputs=[], outputs=[],
            export_strict_FP32_gate_passed=True) for k in factory.MODEL_KEYS}
        self.state = dict(created=0, resident_contexts=0, current_DMA_bytes=0,
            current_model_host_bytes=0, cleanup_errors=[], closed=False)
        self.close_calls = 0
    def metrics(self): return copy.deepcopy(self.state)
    def close(self): self.close_calls += 1; self.state['closed'] = True


class NewCache(EmptyCache):
    def __init__(self, specs, open_session, **budgets):
        super().__init__(); self.specs, self.open_session, self.budgets = specs, open_session, budgets


@pytest.fixture
def fake_stack(tmp_path, monkeypatch):
    seen = dict(parent=[], native=[], session=[], checks=[])
    old = EmptyCache()
    class Owner:
        runtime = 'original'; route = 'full'; source_pins = {}
        catalogue_provenance = dict(candidate_accepted=False, metadata_only_reporting=True)
        cache = old; close_calls = 0
        def close(self): self.close_calls += 1; self.cache.close()
    owner = Owner()
    def parent_open(**kwargs): seen['parent'].append(kwargs); return owner
    parent = SimpleNamespace(open_board_online_from_index=parent_open)
    def open_session(*args, **kwargs): seen['session'].append((args,kwargs)); return object()
    runtime = SimpleNamespace(source_pins={}, input_transport='generated_native_transport',
        BoundedContextCache=NewCache, GroupedSession=SimpleNamespace(open_board=open_session))
    def native_load(*args, **kwargs): seen['native'].append((args,kwargs)); return runtime
    native = SimpleNamespace(load_runtime=native_load)
    def pin(path, digest, pins):
        path = Path(path).resolve(); pins[str(path)] = digest; return path
    monkeypatch.setattr(factory, '_pin', pin)
    monkeypatch.setattr(factory, '_recheck', lambda pins: seen['checks'].append(dict(pins)))
    monkeypatch.setattr(factory, '_load', lambda p, e, label:
        dict(v43_factory=parent, native_AB_harness=harness, native_transport=native)[label])
    return arguments(tmp_path), owner, old, runtime, seen


@pytest.mark.parametrize('flag', ('diagnostic_allow_native_fp16','diagnostic_allow_fp32_rejected'))
def test_both_flags_required_before_reading_or_loading_sources(tmp_path, monkeypatch, flag):
    args = arguments(tmp_path); args[flag] = False
    monkeypatch.setattr(factory, '_pin', lambda *a: pytest.fail('No source read before explicit flags'))
    with pytest.raises(ValueError, match='Explicit native-FP16'): factory.open_board_online_from_index(**args)


def test_delegates_same_callback_then_reuses_frozen_empty_cache_replacement(fake_stack):
    args, owner, old, runtime, seen = fake_stack
    result = factory.open_board_online_from_index(**args)
    assert result is owner and owner.cache is not old and owner.runtime is runtime
    assert old.close_calls == 1 and old.metrics()['closed']
    actual = seen['parent'][0]
    for key in ('motion_probe','on_record','identity_reader','failure_dir'):
        assert actual[key] is args[key]
    assert actual['diagnostic_allow_fp32_rejected'] is True
    assert 'diagnostic_allow_native_fp16' not in actual and 'transport' not in actual
    assert set(owner.cache.specs) == factory.MODEL_KEYS and len(owner.cache.specs) == 9
    owner.cache.specs['B1_D3_M16']['model_sha256'] = 'new-cache-owned'
    assert old.specs['B1_D3_M16']['model_sha256'] == 'generated'
    owner.cache.open_session(owner.cache.specs['B2_D6_M16'])
    kwargs = seen['session'][0][1]
    assert kwargs['diagnostic_allow_native_fp16'] is True and kwargs['diagnostic_allow_fp32_rejected'] is True
    assert kwargs['collect_perf_detail'] is False and kwargs['abi_path'] == args['abi_path']
    assert str(args['full_ab_receipt'].resolve()) in kwargs['pins']
    assert str(args['packing_build_receipt'].resolve()) in owner.source_pins
    assert owner.catalogue_provenance['closed_flag_optimization_added'] is False
    assert owner.catalogue_provenance['candidate_accepted'] is False
    owner.close(); assert owner.close_calls == 1 and owner.cache.metrics()['closed']


@pytest.mark.parametrize('live', ('created','resident_contexts'))
def test_cannot_replace_used_cache_and_cleanup_parent(fake_stack, live):
    args, owner, old, runtime, seen = fake_stack; old.state[live] = 1
    with pytest.raises(ValueError, match='unused validated'): factory.open_board_online_from_index(**args)
    assert owner.close_calls == 1 and owner.cache is old


def test_native_runtime_construction_failure_closes_original_owner(fake_stack, monkeypatch):
    args, owner, old, runtime, seen = fake_stack
    def load(path, expected, label):
        if label == 'native_transport':
            return SimpleNamespace(load_runtime=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('Generated pin failure')))
        return SimpleNamespace(open_board_online_from_index=lambda **kw: owner) if label == 'v43_factory' else harness
    monkeypatch.setattr(factory,'_load',load)
    with pytest.raises(RuntimeError, match='Generated pin'): factory.open_board_online_from_index(**args)
    assert owner.close_calls == 1 and old.metrics()['closed']


def test_changed_source_after_replacement_closes_new_cache(fake_stack, monkeypatch):
    args, owner, old, runtime, seen = fake_stack; count=[]
    def check(pins):
        count.append(1)
        if len(count)==2: raise ValueError('Generated changed source')
    monkeypatch.setattr(factory,'_recheck',check)
    with pytest.raises(ValueError, match='Generated changed source'): factory.open_board_online_from_index(**args)
    assert owner.close_calls == 1 and owner.cache is not old and owner.cache.metrics()['closed']


def test_exact_source_digest_rejects_tampered_generated_bytes(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'original');pins={}
    factory._pin(p,hashlib.sha256(b'original').hexdigest(),pins)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):factory._recheck(pins)


def test_original_trajectory_update_probe_RNG_EOF_finally_AST_and_source_pins():
    parent=ast.parse((ANCESTOR/'run-v43-online-npu-board.py').read_text())
    child=ast.parse((HERE/'run-v44-online-native-board.py').read_text())
    functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body
        if isinstance(n,ast.FunctionDef) and n.name not in ('main','parse_args')}
    assert functions(parent)==functions(child)
    main=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    frame=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
        and [v.id for v in n.target.elts]==['index','item'])
    assert ast.dump(frame(main(parent)),include_attributes=False)==ast.dump(frame(main(child)),include_attributes=False)
    for name in ('update','original_torch_motion_probe','measured_ba','snapshot','save','on_online_record'):
        function=lambda t:next(n for n in ast.walk(main(t)) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(function(parent),include_attributes=False)==ast.dump(function(child),include_attributes=False)
    cleanup=lambda t:[ast.dump(n,include_attributes=False) for n in next(n for n in t.body if isinstance(n,ast.Try)).finalbody]
    assert cleanup(main(parent))==cleanup(main(child))
    calls=lambda t,attr:[ast.dump(n,include_attributes=False) for n in ast.walk(t)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==attr]
    for attr in ('terminate','manual_seed','process_encoded'):
        assert calls(parent,attr)==calls(child,attr) and len(calls(child,attr))==1
    assert driver.V43_ONLINE_DRIVER==factory.sha256(ANCESTOR/'run-v43-online-npu-board.py')
    assert driver.ONLINE_FACTORY==factory.sha256(HERE/'v44_online_factory.py')
    assert driver.ONLINE_CALLBACK==factory.sha256(ANCESTOR/'v43_online_update.py')
    assert factory.AB_HARNESS_SHA256==factory.sha256(NATIVE/'run-v43-native-full-ab-board.py')
    assert factory.TRANSPORT_SHA256==factory.sha256(NATIVE/'v43_native_fp16_grouped.py')


def test_driver_native_flag_is_explicit_and_v43_parent_is_required():
    keys=('parent-driver','policy','torch-encoder','source-dir','checkpoint','geometry',
        'stream-module','encoder-module','frame-cache-helper','library','build-receipt',
        'encoder-capture-dir','dataset-dir','output-dir','cached-cpu-driver','online-factory',
        'online-config','v41-online-driver','v42-online-driver','v43-online-driver','undistort-helper')
    argv=[v for key in keys for v in ('--'+key,'generated')]+[
        '--online-factory-sha256','x','--online-config-sha256','y','--diagnostic-allow-fp32-rejected']
    a=driver.parse_args(argv);assert a.diagnostic_allow_native_fp16 is False
    a=driver.parse_args(argv+['--diagnostic-allow-native-fp16'])
    assert a.diagnostic_allow_native_fp16 is True and a.v43_online_driver==Path('generated')
    with pytest.raises(SystemExit):driver.parse_args([v for i,v in enumerate(argv) if i not in (36,37)])
