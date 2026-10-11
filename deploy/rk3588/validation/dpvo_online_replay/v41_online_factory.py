"""Board-only catalogue assembly using frozen v40/v37 SDK implementations.

No SDK/Torch/ORT inference is performed on import. DPVO MIT, adjacent LICENSE.
All full Update contexts are RKNN; explicit D1 startup route is provided by the
owner as the original Torch callable. Historical FP32 rejection stays visible.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import uuid

BRIDGE_SHA256 = 'd4cdd1f9f21b04e44b6369c8ad7aea1fe4fbd43855b13993c6fdf4550f7bf0d7'
ADAPTER_SHA256 = '0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def load(path, expected, label):
    path = Path(path).resolve(strict=True)
    if sha256(path) != expected:
        raise ValueError('Exact '+label+' SHA differs')
    name = '_v41_online_'+label+'_'+uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def open_board_online_from_index(*, compile_index, compile_index_sha256,
        parent_manifest, additions_manifest, additions_manifest_sha256,
        addition_exporter, addition_exporter_sha256,
        parent_compiler, addition_compiler, addition_compiler_sha256,
        contracts, contracts_sha256, m16_contracts, parent_contracts,
        parent_runtime, runtime_bridge, adapter_path, callback_path, callback_sha256,
        abi_path, header_path, library_path, log_helper_path,
        failure_dir, motion_probe, diagnostic_allow_fp32_rejected,
        identity_reader=None, on_record=None, max_contexts=8,
        max_model_bytes=256 << 20, max_dma_bytes=512 << 20):
    """Return the callable owner closes before native/runtime source cleanup.

    Source paths are explicit and pinned. New online contracts keep byte-exact
    v40/v37 contract files adjacent. SDK ancestors retain their own directory.
    All eight compile/model artifacts are verified before any context init.
    """
    if diagnostic_allow_fp32_rejected is not True:
        raise ValueError('Explicit FP32-rejected online research diagnostic required')
    c = load(contracts, contracts_sha256, 'contracts')
    c.load_m16_parent(m16_contracts); c.load_parent(parent_contracts)
    old, pins = c.validate_export_manifest(parent_manifest, c.EXPORT_MANIFEST_SHA256)
    added, extra = c.validate_export_manifest(additions_manifest, additions_manifest_sha256,
        parent_manifest=old, exporter_sha256=addition_exporter_sha256,
        allow_fp32_rejected_diagnostic=True)
    pins.update(extra)
    _, specs, _, extra = c.validate_compile_index(compile_index, compile_index_sha256, old, added,
        additions_manifest_sha256=additions_manifest_sha256, parent_compiler=parent_compiler,
        addition_compiler=addition_compiler, addition_compiler_sha256=addition_compiler_sha256,
        m16_contracts=m16_contracts, online_contracts=contracts, online_contracts_sha256=contracts_sha256)
    pins.update(extra)
    for path, digest in ((contracts, contracts_sha256), (m16_contracts, c.M16_CONTRACTS_SHA256),
            (parent_contracts, c.PARENT_CONTRACTS_SHA256), (addition_exporter, addition_exporter_sha256),
            (parent_compiler, c.M16_COMPILER_SHA256), (addition_compiler, addition_compiler_sha256),
            (callback_path, callback_sha256), (runtime_bridge, BRIDGE_SHA256),
            (adapter_path, ADAPTER_SHA256), (__file__, sha256(__file__))):
        c.pin(path, digest, pins)
    c.recheck_pins(pins)
    bridge = load(runtime_bridge, BRIDGE_SHA256, 'bridge')
    runtime = bridge.load_runtime(parent_runtime, contracts, contracts_sha256, parent_contracts)
    adapter = load(adapter_path, ADAPTER_SHA256, 'adapter')
    callback_module = load(callback_path, callback_sha256, 'callback')
    # Before creating any native context, the original ABI checks actual board.
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
        parent_manifest_sha256=c.EXPORT_MANIFEST_SHA256,
        additions_manifest_sha256=additions_manifest_sha256,
        candidate_accepted=False, original_half_resolution_preset_quality_rejected=True,
        direct_startup_Torch_motion_probe=True,
        complete_Update_executor='RKNN_no_Torch_fallback',
        all_internal_operators_NPU=False)
    return callback
