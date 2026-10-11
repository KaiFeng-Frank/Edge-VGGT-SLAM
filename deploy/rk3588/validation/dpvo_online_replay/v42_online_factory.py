"""Board-only nine-model catalogue using frozen v41/v40/v37 ancestors.

DPVO, Princeton University (2022), MIT; adjacent LICENSE. Import performs no
SDK, Torch or ORT inference. D1 startup is explicit original Torch; all complete
Update groups use RKNN with historical FP32 rejection retained as diagnostic.
"""
import importlib.util
from pathlib import Path
import sys
import uuid

BRIDGE_SHA256 = 'd4cdd1f9f21b04e44b6369c8ad7aea1fe4fbd43855b13993c6fdf4550f7bf0d7'
ADAPTER_SHA256 = '0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1'


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path, expected, label):
    path = Path(path).resolve(strict=True)
    if sha256(path) != expected:
        raise ValueError('Exact ' + label + ' SHA differs')
    name = '_v42_online_' + label + '_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def open_board_online_from_index(*, compile_index, compile_index_sha256,
        parent_manifest, additions_manifest, additions_manifest_sha256,
        addition_exporter, addition_exporter_sha256,
        parent_compiler, addition_compiler, addition_compiler_sha256,
        contracts, contracts_sha256, m16_contracts, parent_contracts,
        parent_runtime, runtime_bridge, adapter_path, callback_path, callback_sha256,
        abi_path, header_path, library_path, log_helper_path,
        failure_dir, motion_probe, diagnostic_allow_fp32_rejected,
        old_online_compile_index, old_online_compile_index_sha256,
        v41_contracts, v41_contracts_sha256, v41_callback, v41_callback_sha256,
        eof_manifest, eof_manifest_sha256, eof_exporter, eof_exporter_sha256,
        eof_compiler, eof_compiler_sha256,
        identity_reader=None, on_record=None, max_contexts=9,
        max_model_bytes=256 << 20, max_dma_bytes=512 << 20):
    """Return the same callable/metrics/close API as the frozen v41 factory.

    Preserve the old index bytes and entries; add only the independent v42 D3
    manifest/compiled artifact. Staging must retain frozen v41/v40/v37 contracts
    beside the new contract, because the private frozen SDK bridge reloads it.
    All artifacts are checked before native context creation. No CPU fallback.
    """
    if diagnostic_allow_fp32_rejected is not True:
        raise ValueError('Explicit FP32-rejected online research diagnostic required')
    c = load(contracts, contracts_sha256, 'contracts')
    if (v41_contracts_sha256 != c.V41_CONTRACTS_SHA256
            or v41_callback_sha256 != c.V41_CALLBACK_SHA256
            or addition_compiler_sha256 != c.V41_COMPILER_SHA256):
        raise ValueError('Frozen original v41 source chain changed')
    # The bridge creates a separate module instance, so require its default
    # dependency files to be exactly the explicit pinned staging paths.
    for explicit, name in ((v41_contracts, 'v41_online_contracts.py'),
                           (m16_contracts, 'v40_m16_grouped_contracts.py'),
                           (parent_contracts, 'v37_grouped_contracts.py')):
        if Path(explicit).resolve(strict=True) != Path(contracts).resolve(strict=True).with_name(name):
            raise ValueError('Frozen contract dependencies must be adjacent: ' + name)
    old = c.load_v41_parent(v41_contracts)
    c.load_m16_parent(m16_contracts); c.load_parent(parent_contracts)
    parent, pins = old.validate_export_manifest(parent_manifest, c.EXPORT_MANIFEST_SHA256)
    added, extra = old.validate_export_manifest(additions_manifest, additions_manifest_sha256,
        parent_manifest=parent, exporter_sha256=addition_exporter_sha256,
        allow_fp32_rejected_diagnostic=True)
    pins.update(extra)
    eof, extra = c.validate_export_manifest(eof_manifest, eof_manifest_sha256,
        parent_manifest=parent, exporter_sha256=eof_exporter_sha256,
        allow_fp32_rejected_diagnostic=True)
    pins.update(extra)
    _, specs, _, extra = c.validate_compile_index(compile_index, compile_index_sha256,
        parent, added, eof, old_online_compile_index=old_online_compile_index,
        old_online_compile_index_sha256=old_online_compile_index_sha256,
        additions_manifest_sha256=additions_manifest_sha256, eof_manifest_sha256=eof_manifest_sha256,
        parent_compiler=parent_compiler, addition_compiler=addition_compiler,
        addition_compiler_sha256=addition_compiler_sha256, eof_compiler=eof_compiler,
        eof_compiler_sha256=eof_compiler_sha256, m16_contracts=m16_contracts,
        v41_contracts=v41_contracts, online_contracts=contracts, online_contracts_sha256=contracts_sha256)
    pins.update(extra)
    for path, digest in ((contracts, contracts_sha256), (v41_contracts, c.V41_CONTRACTS_SHA256),
            (v41_callback, c.V41_CALLBACK_SHA256), (m16_contracts, c.M16_CONTRACTS_SHA256),
            (parent_contracts, c.PARENT_CONTRACTS_SHA256), (addition_exporter, addition_exporter_sha256),
            (parent_compiler, c.M16_COMPILER_SHA256), (addition_compiler, addition_compiler_sha256),
            (eof_exporter, eof_exporter_sha256), (eof_compiler, eof_compiler_sha256),
            (callback_path, callback_sha256), (runtime_bridge, BRIDGE_SHA256),
            (adapter_path, ADAPTER_SHA256), (__file__, sha256(__file__))):
        c.pin(path, digest, pins)
    c.recheck_pins(pins)
    bridge = load(runtime_bridge, BRIDGE_SHA256, 'bridge')
    runtime = bridge.load_runtime(parent_runtime, contracts, contracts_sha256, parent_contracts)
    adapter = load(adapter_path, ADAPTER_SHA256, 'adapter')
    callback_module = load(callback_path, callback_sha256, 'callback')
    abi_file = c.pin(abi_path, c.ABI_SHA256, pins)
    abi = load(abi_file, c.ABI_SHA256, 'ABI'); abi.require_board()
    failure_dir = Path(failure_dir).resolve(strict=True)
    if not failure_dir.is_dir():
        raise ValueError('Existing isolated diagnostics directory required')
    def factory(spec):
        return runtime.GroupedSession.open_board(spec['model_path'], spec['model_sha256'],
            spec['inputs'], spec['outputs'], abi_path=abi_path, header_path=header_path,
            library_path=library_path, log_helper_path=log_helper_path,
            export_strict_FP32_gate_passed=spec['export_strict_FP32_gate_passed'],
            diagnostic_allow_fp32_rejected=True, pins=pins,
            collect_perf_detail=False, log_directory=failure_dir)
    cache = runtime.BoundedContextCache(specs, factory, max_contexts=max_contexts,
        max_model_bytes=max_model_bytes, max_dma_bytes=max_dma_bytes)
    try:
        callback = callback_module.OnlineM16Update(runtime, adapter, cache,
            motion_probe=motion_probe, diagnostic_allow_fp32_rejected=True,
            failure_dir=failure_dir, identity_reader=identity_reader, on_record=on_record)
    except BaseException:
        cache.close()
        raise
    callback.source_pins = dict(pins)
    callback.catalogue_provenance = dict(compile_index_sha256=compile_index_sha256,
        old_online_compile_index_sha256=old_online_compile_index_sha256,
        parent_manifest_sha256=c.EXPORT_MANIFEST_SHA256,
        additions_manifest_sha256=additions_manifest_sha256, eof_manifest_sha256=eof_manifest_sha256,
        old_eight_index_entries_preserved=True, new_model_keys=list(c.NEW_KEYS),
        candidate_accepted=False, original_half_resolution_preset_quality_rejected=True,
        direct_startup_Torch_motion_probe=True, complete_Update_executor='RKNN_no_Torch_fallback',
        all_internal_operators_NPU=False)
    return callback
