"""Research-only closed-query cache integration over frozen v44 native inputs.

The original callable/metadata callback, native sessions, nine models, complete
source groups and explicit Torch D1 startup route are retained. Only an empty
cache is replaced before its first context is created. No SDK, model, packing
library or board work occurs on import. Numerical/quality rejection remains.
DPVO: Copyright (c) 2022 Princeton University; adjacent MIT LICENSE.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

PARENT_FACTORY_SHA256 = 'ac2b29b7eabc17d1dd47ec7a89734d6c1512dcbf37845ef5a32e9667b9b6be79'
CLOSED_FLAG_SHA256 = '014685c3b7736070bf3b1a1e3a333a928468e013cf454e4acac4ee53da1354b2'
CLOSED_AB_SHA256 = '1a2132b70849fe823e3d85617d588f2d1dd5626326adfa4c14c0e718d2f7faf4'
CLOSED_AB_DRIVER_SHA256 = 'c809ac7e5adccb1dcab3655fb807e461257e0d63499c84113aac9ec8330a1388'
PARENT_RUNTIME_SHA256 = '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'
NATIVE_HARNESS_SHA256 = '4c056f07828db7b5fa74914148e5a196840af2275988eb4f4104a459488fa9c2'
CALLBACK_SHA256 = '4f4a5457dbdec5aebe1ddd6204a37c9073896ac99ed90abd3fbfe3d1f78db566'
INDEX_SHA256 = 'eeb9241c9eab52039aa64fcf6546ce4624d1388756a9e919f42c22de4083a85c'
NATIVE_TRANSPORT = 'native_FP16_NC1HWC2_C2_8_pass_through_1'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def _pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(sha256(path) == expected, 'Frozen closed-online source/artifact differs: ' + str(path))
    pins[str(path)] = expected
    return path


def _recheck(pins):
    for path, expected in pins.items():
        require(sha256(path) == expected, 'Frozen closed-online source/artifact changed: ' + path)


def _load(path, expected, label):
    require(sha256(path) == expected, 'Exact frozen ' + label + ' SHA differs')
    name = '_v45_closed_online_' + label + '_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def validate_prerequisite(row):
    """Exact-pinned executed AB metadata; not a new numerical array check."""
    require(row['schema'] == 'v44_native_cache_closed_flag_three_saved_state_ABBA_diagnostic'
        and row['status'] == 'completed_AB_bitwise_equal_diagnostic_not_adopted'
        and row['script_sha256'] == CLOSED_AB_DRIVER_SHA256
        and row['closed_flag_helper_sha256'] == CLOSED_FLAG_SHA256
        and row['parent_runtime_sha256'] == PARENT_RUNTIME_SHA256
        and row['native_harness_sha256'] == NATIVE_HARNESS_SHA256
        and row['compile_index_sha256'] == INDEX_SHA256
        and row['callback_sha256'] == dict(A=CALLBACK_SHA256, B=CALLBACK_SHA256)
        and row['same_callback_for_both_methods'] is True
        and row['same_native_input_transport'] is True
        and row['only_cache_live_hit_closed_query_changed'] is True
        and row['same_session_source_pins_for_both_methods'] is True
        and row['A_B_all_saved_outputs_bitwise_equal'] is True
        and row['closed_flag_bitwise_for_three_saved_states_only'] is True
        and row['historical_SDK_strict_FP32_rejected'] is True
        and row['strict_FP32_equivalence_validated'] is False
        and row['candidate_accepted'] is False and row['CPU_teacher_used'] is False
        and row['recurrent_feedback'] is False and row['BA_executed'] is False
        and row['sources_and_inputs_unchanged'] is True and row['affinity_restored'] is True
        and row['cleanup_errors'] == []
        and row['source_pins'] and row['source_pins'] == row['source_hashes_after'],
        'Completed original/native closed-query AB prerequisite required')
    expected = [(obs, method) for obs in (34,127,191) for method in ('A','B','B','A')*2]
    require([(v['observation_id'],v['method']) for v in row['calls']] == expected
        and all(v['bitwise_equal_first_A_for_state'] is True for v in row['calls']),
        'All24 executed closed-query ABBA outputs must be bitwise equal')
    for method in ('A','B'):
        life = row['lifecycle'][method]
        require(life['sdk_run_api_calls'] == life['outputs_released'] == 56
            and life['contexts_destroyed'] == 6 and life['closed'] is True
            and life['current_DMA_bytes'] == life['current_model_host_bytes'] == 0,
            'Closed-query prerequisite SDK release/context cleanup differs')
    proof = row['closed_flag_source_proof']
    require(proof['parent_runtime_sha256'] == PARENT_RUNTIME_SHA256
        and proof['changed_method'] == 'BoundedContextCache._get'
        and proof['changed_expression_count'] == 1 and proof['all_other_get_AST_preserved'] is True
        and proof['cold_and_budget_full_metrics_preserved'] is True
        and proof['lifecycle_full_metrics_preserved'] is True
        and proof['numerical_execution_changed'] is False and proof['native_pack_or_SDK_changed'] is False
        and proof['candidate_accepted'] is False, 'Only the locked cache-hit closed query may differ')


def open_board_online_from_index(*, v44_factory, v44_factory_sha256,
        closed_flag_helper, closed_flag_helper_sha256, closed_ab_receipt, closed_ab_receipt_sha256,
        diagnostic_allow_closed_flag=False, **kwargs):
    """Same v44 configuration, with explicit cache sources and an executed AB.

    The frozen v44 owner validates all nine models and constructs its unused
    native cache. Reuse its already-loaded native runtime/library, bind the
    cache-only subclass, and call the frozen replace_empty_cache helper.
    Owner calls/close/source checks and state ownership remain original.
    """
    require(diagnostic_allow_closed_flag is True
        and kwargs.get('diagnostic_allow_native_fp16') is True
        and kwargs.get('diagnostic_allow_fp32_rejected') is True,
        'All three explicit closed-query/native-FP16/FP32-rejected diagnostic flags required')
    pins = {}
    for path, actual, expected in (
            (v44_factory, v44_factory_sha256, PARENT_FACTORY_SHA256),
            (closed_flag_helper, closed_flag_helper_sha256, CLOSED_FLAG_SHA256),
            (closed_ab_receipt, closed_ab_receipt_sha256, CLOSED_AB_SHA256),
            (kwargs['native_ab_harness'], kwargs['native_ab_harness_sha256'], NATIVE_HARNESS_SHA256)):
        require(actual == expected, 'Exact frozen closed-online prerequisite SHA differs')
        _pin(path, expected, pins)
    _pin(__file__, sha256(__file__), pins)
    validate_prerequisite(json.loads(Path(closed_ab_receipt).read_text()))
    require(kwargs.get('callback_sha256') == CALLBACK_SHA256
        and kwargs.get('compile_index_sha256') == INDEX_SHA256 and kwargs.get('max_contexts') == 9,
        'Same frozen native nine-model metadata callback required')
    parent = _load(v44_factory, PARENT_FACTORY_SHA256, 'v44_factory')
    closed = _load(closed_flag_helper, CLOSED_FLAG_SHA256, 'closed_cache')
    harness = _load(kwargs['native_ab_harness'], NATIVE_HARNESS_SHA256, 'native_AB_harness')
    _recheck(pins)
    owner = parent.open_board_online_from_index(**kwargs)
    try:
        native = owner.runtime
        require(native.input_transport == NATIVE_TRANSPORT, 'Frozen native session transport required')
        candidate = closed.bind_runtime(native, parent_runtime_path=kwargs['parent_runtime'],
            diagnostic_allow_closed_flag=True)
        require(candidate.GroupedSession is native.GroupedSession
            and candidate.execute_full_graph is native.execute_full_graph
            and candidate.pack_inputs is native.pack_inputs
            and candidate.BoundedContextCache is not native.BoundedContextCache,
            'Only the bounded cache class may change; native/SDK/math remain frozen')
        harness.replace_empty_cache(owner, candidate, kwargs, pins, kwargs['failure_dir'])
        owner.catalogue_provenance = dict(owner.catalogue_provenance,
            closed_flag_optimization_added=True, cache_closed_flag_only=True,
            closed_full_saved_AB_prerequisite_sha256=CLOSED_AB_SHA256,
            closed_flag_helper_sha256=CLOSED_FLAG_SHA256,
            closed_flag_source_proof=candidate.closed_flag_source_proof,
            original_v44_factory_sha256=PARENT_FACTORY_SHA256,
            live_closed_query_trajectory_validated=False, candidate_accepted=False,
            numerical_execution_changed=False, historical_SDK_strict_FP32_rejected=True)
        _recheck(owner.source_pins)
    except BaseException:
        owner.close()
        raise
    return owner
