"""Board-only native input wrapper around the frozen v43 nine-shape factory.

DPVO, Princeton Vision & Learning Lab (2022), MIT; adjacent LICENSE.
Only an empty, validated context cache is replaced. The frozen callback,
complete-source plan, D1 startup, outputs and research VO rules are unchanged.
Historical FP32 and workload rejection remain; no candidate adoption or default
VGGT change. Import performs no SDK, model or packing-library execution.
"""
import importlib.util
import json
from pathlib import Path
import sys
import uuid

PARENT_FACTORY_SHA256 = 'e15ff495ac626e7a1a7271a918b39854b4cc45989b77e334db56d8234ea5f300'
AB_HARNESS_SHA256 = '4c056f07828db7b5fa74914148e5a196840af2275988eb4f4104a459488fa9c2'
FULL_AB_SHA256 = 'b7e28619870af18f2a95c6d411281bbfe249fb1a69f97ab0c4df5cbe6adc0968'
TRANSPORT_SHA256 = '222c254361a50d9180725ea780116a2302a29be8396136b15db91e7d11e912ca'
PACKER_SHA256 = 'b3d2503d1bb27ec0352cfe54d759efdcee070764ad266841847005fa9b13f8ce'
PACKING_LIBRARY_SHA256 = '5de153a44839915a76d68e7cb3141e31c46ae6a3dc83e8b32273bc9ecd3d0c6a'
QUERY_SHA256 = '89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642'
BUILD_SHA256 = '7f907ca8c477ddb80d6d18802d160bde6d2c628619406a2504c7b5f6684f28cb'
CONVERSION_SHA256 = '94c4cbd4adc458e34109d9cfa003112890069509616790fe5d7b8a0909883ee8'
CALLBACK_SHA256 = '4f4a5457dbdec5aebe1ddd6204a37c9073896ac99ed90abd3fbfe3d1f78db566'
INDEX_SHA256 = 'eeb9241c9eab52039aa64fcf6546ce4624d1388756a9e919f42c22de4083a85c'
MODEL_KEYS = frozenset([f'B{b}_D{d}_M16' for b in (1, 2) for d in range(4, 8)] + ['B1_D3_M16'])


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _pin(path, digest, pins):
    path = Path(path).resolve(strict=True)
    require(sha256(path) == digest, 'Frozen native-online source/artifact differs: ' + str(path))
    pins[str(path)] = digest
    return path


def _recheck(pins):
    for path, digest in pins.items():
        require(sha256(path) == digest, 'Frozen native-online source/artifact changed: ' + path)


def _load(path, expected, label):
    require(sha256(path) == expected, 'Exact frozen ' + label + ' SHA differs')
    name = '_v44_native_online_' + label + '_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def validate_prerequisites(full, build, conversion):
    """Pinned actual experiment metadata; not a new array/model validation."""
    require(full['schema'] == 'v43_native_three_complete_saved_state_ABBA_diagnostic'
        and full['status'] == 'completed_AB_bitwise_equal_diagnostic_not_adopted'
        and full['candidate_accepted'] is False
        and full['historical_SDK_strict_FP32_rejected'] is True
        and full['same_callback_for_both_methods'] is True
        and full['only_input_transport_changed'] is True
        and full['A_B_all_saved_outputs_bitwise_equal'] is True
        and full['native_transport_bitwise_for_three_saved_states_only'] is True
        and full['sources_and_inputs_unchanged'] is True
        and full['affinity_restored'] is True and full['cleanup_errors'] == []
        and full['script_sha256'] == AB_HARNESS_SHA256
        and full['factory_sha256'] == PARENT_FACTORY_SHA256
        and full['callback_sha256'] == dict(A=CALLBACK_SHA256, B=CALLBACK_SHA256)
        and full['compile_index_sha256'] == INDEX_SHA256,
        'Actual complete saved-state native AB prerequisite required')
    expected = [(obs, method) for obs in (34, 127, 191) for method in ('A', 'B', 'B', 'A')*2]
    require([(v['observation_id'], v['method']) for v in full['calls']] == expected
        and all(v['bitwise_equal_first_A_for_state'] is True for v in full['calls']),
        'Complete actual 24-call ABBA bitwise prerequisite required')
    for method in ('A', 'B'):
        value = full['lifecycle'][method]
        require(value['sdk_run_api_calls'] == value['outputs_released'] == 56
            and value['contexts_destroyed'] == 6 and value['closed'] is True
            and value['current_DMA_bytes'] == value['current_model_host_bytes'] == 0,
            'Actual complete AB context/release prerequisite failed')
    require(build['status'] == 'compiled_ABI_loaded_conversion_not_yet_verified'
        and build['output']['sha256'] == PACKING_LIBRARY_SHA256
        and build['sources_unchanged'] is True and build['cleanup_errors'] == []
        and conversion['status'] == 'passed_conversion_reference_and_ownership_only'
        and conversion['library_sha256'] == PACKING_LIBRARY_SHA256
        and conversion['query_sha256'] == QUERY_SHA256
        and conversion['packer_sha256'] == PACKER_SHA256
        and conversion['cleanup_errors'] == [], 'Actual packing build/conversion prerequisite required')


def open_board_online_from_index(*, v43_factory, v43_factory_sha256,
        native_ab_harness, native_ab_harness_sha256, transport, transport_sha256,
        packer, packer_sha256, packing_library, packing_library_sha256,
        query, query_sha256, packing_build_receipt, packing_build_receipt_sha256,
        conversion_receipt, conversion_receipt_sha256, full_ab_receipt, full_ab_receipt_sha256,
        diagnostic_allow_native_fp16=False, **kwargs):
    """Return the original callable/metrics/close with native input submission.

    Same v43 arguments plus explicit transport sources and executed prerequisites.
    Both diagnostic flags must be true. The original factory validates all nine
    models before this wrapper closes its unused cache. No live context or VO
    state is transferred. All new pins join the original owner's source map,
    which the unchanged replay driver rehashes before execution and after close.
    """
    require(diagnostic_allow_native_fp16 is True and kwargs.get('diagnostic_allow_fp32_rejected') is True,
        'Explicit native-FP16 and FP32-rejected online diagnostic flags required')
    pairs = (
        (v43_factory, v43_factory_sha256, PARENT_FACTORY_SHA256),
        (native_ab_harness, native_ab_harness_sha256, AB_HARNESS_SHA256),
        (transport, transport_sha256, TRANSPORT_SHA256),
        (packer, packer_sha256, PACKER_SHA256),
        (packing_library, packing_library_sha256, PACKING_LIBRARY_SHA256),
        (query, query_sha256, QUERY_SHA256),
        (packing_build_receipt, packing_build_receipt_sha256, BUILD_SHA256),
        (conversion_receipt, conversion_receipt_sha256, CONVERSION_SHA256),
        (full_ab_receipt, full_ab_receipt_sha256, FULL_AB_SHA256))
    pins = {}
    for path, actual, expected in pairs:
        require(actual == expected, 'Exact frozen native-online prerequisite SHA differs')
        _pin(path, expected, pins)
    _pin(__file__, sha256(__file__), pins)
    validate_prerequisites(json.loads(Path(full_ab_receipt).read_text()),
        json.loads(Path(packing_build_receipt).read_text()),
        json.loads(Path(conversion_receipt).read_text()))
    require(kwargs.get('callback_sha256') == CALLBACK_SHA256
        and kwargs.get('compile_index_sha256') == INDEX_SHA256
        and kwargs.get('max_contexts') == 9, 'Same frozen nine-model metadata callback required')
    parent = _load(v43_factory, PARENT_FACTORY_SHA256, 'v43_factory')
    harness = _load(native_ab_harness, AB_HARNESS_SHA256, 'native_AB_harness')
    native = _load(transport, TRANSPORT_SHA256, 'native_transport')
    _recheck(pins)
    owner = parent.open_board_online_from_index(**kwargs)
    try:
        runtime = native.load_runtime(kwargs['parent_runtime'], kwargs['contracts'], kwargs['contracts_sha256'],
            kwargs['parent_contracts'], runtime_bridge=kwargs['runtime_bridge'],
            packer_path=packer, packer_sha256=PACKER_SHA256,
            packing_library_path=packing_library, packing_library_sha256=PACKING_LIBRARY_SHA256,
            packing_source_pins=pins, native_query_receipt=query, native_query_sha256=QUERY_SHA256,
            diagnostic_allow_native_fp16=True)
        harness.replace_empty_cache(owner, runtime, kwargs, pins, kwargs['failure_dir'])
        owner.catalogue_provenance = dict(owner.catalogue_provenance,
            native_online_input_transport=True, native_full_saved_AB_prerequisite_sha256=FULL_AB_SHA256,
            original_v43_factory_sha256=PARENT_FACTORY_SHA256,
            original_v43_callback_sha256=CALLBACK_SHA256,
            native_transport_sha256=TRANSPORT_SHA256,
            historical_SDK_strict_FP32_rejected=True, candidate_accepted=False,
            live_native_transport_trajectory_validated=False, numerical_thresholds_relaxed=False,
            closed_flag_optimization_added=False)
        _recheck(owner.source_pins)
    except BaseException:
        owner.close()
        raise
    return owner
