#!/usr/bin/env python3
"""Profile the four v80 calibrated original VGGT retained groups; no model loading.

Eight diagnostic calls use original saved activations, flags8 and the exact
frozen v80 FP32 API conversion route. Internal W8A8/FP16 is unchanged. This is neither a normal benchmark nor a full-model
or accuracy test. The caller controls frequency; this script only reads it.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

BASE = Path('/mnt/vggt-data/vggt-slam2')
MANIFEST_SHA = '771a9b82007d2feefd65ebbf3f4b0fb27070b79ff7f6d0faab1a8c866da90e43'
RUNTIME_SHA = '04fbb5b2ad7db22bf26d52d4a6b5a3fadffa042557919575938b035f5dd06b5f'
HELPER_SHA = '4fea665334e9d12a3d55839c3150680e269bab9794074d698bed2ad0a6e41c93'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
ADAPTER_SHA = 'f3f106034c6621a1e3ed07bdb265107b8522103f7d186ba4bbe0d0cefda16f2f'
QUANTIZED_MANIFEST_SHA = '2d322f3afed5af4ab510a879960b941118c8a9ec585a7c4bd9bf06269fba1d97'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def sample():
    root = Path('/sys/class/devfreq/fdab0000.npu')
    return dict(npu_hz=int((root / 'cur_freq').read_text()),
        governor=(root / 'governor').read_text().strip(),
        temperatures_c={str((p / 'type').read_text()).strip(): int((p / 'temp').read_text()) / 1000
            for p in Path('/sys/class/thermal').glob('thermal_zone*')
            if (p / 'type').is_file() and (p / 'temp').is_file()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--manifest', type=Path, default=BASE / 'experiments/vggt-v67-retained-groups-r1/compiled.json')
    p.add_argument('--adapter-source', type=Path, default=BASE / 'experiments/vggt-v80-source-r1/rknn_calibrated_retained.py')
    p.add_argument('--quantized-manifest', type=Path, default=BASE / 'experiments/vggt-v80-retained-int8-r1/compiled.json')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--expected-npu-hz', type=int, choices=(300000000, 1000000000), default=300000000)
    a = p.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    pins, contexts, api, abi, adapter = {}, [], None, None, None
    quantized_paths, quantized_identities = None, None
    old_affinity = os.sched_getaffinity(0)
    r = dict(schema='v82_original_calibrated_retained_groups_SDK_profile', status='failed',
        script_sha256=sha(__file__), original_model_only=True, candidate_accepted=False,
        actual_model_or_ORT_or_compiler_execution=False, model_parameters_loaded=False,
        quantized_manifest_sha256=QUANTIZED_MANIFEST_SHA, adapter_source_sha256=ADAPTER_SHA,
        API_FP32_conversion_not_internal_FP32=True, original_ONNX_graphs_unchanged=True,
        requested_quantized_dtype='w8a8', all_internal_operators_INT8_verified=False,
        profile_init_flags=8, core_mask=7, timeout_ms=30000, hidden_warmup_calls=0,
        normal_benchmark_timing=False, full_model_or_trajectory_or_realtime_validated=False,
        expected_npu_hz=a.expected_npu_hz, frequency_written=False,
        frequency_scope='Before and after each SDK call only; external controller owns frequency',
        context_schedule='One original group context at a time; two original inputs per group',
        calls=[], queries=[], contexts_created=0, contexts_destroyed=0, cleanup_errors=[])
    stats = dict(sdk_run_attempts=0, sdk_runs_succeeded=0, outputs_acquired=0,
        outputs_released=0, sdk_io_seconds=0., cleanup_errors=r['cleanup_errors'],
        fp32_io=dict(calls_attempted=0, calls_completed=0, owned_output_copy_seconds=0.,
            finite_check_seconds=0., route_seconds=0., **{x + '_seconds': 0. for x in (
                'sdk_api', 'rknn_inputs_set', 'rknn_run', 'rknn_outputs_get', 'rknn_outputs_release')}))
    def pin(path, digest):
        path = Path(path).resolve(strict=True); require(sha(path) == digest, 'Artifact SHA differs: ' + str(path))
        pins[path] = digest; return path
    def artifact(directory, row):
        directory = Path(directory).resolve(strict=True); relative = Path(row['file'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
        path = (directory / relative).resolve(strict=True)
        require(path.is_relative_to(directory) and path.is_file() and path.stat().st_size == row['bytes'], 'Artifact path/size differs')
        return pin(path, row['sha256'])
    def load(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
    def destroy(context):
        if context.value:
            try:
                abi.check_call(api.rknn_destroy(context), 'rknn_destroy')
                context.value = 0; r['contexts_destroyed'] += 1
            except BaseException as exc: r['cleanup_errors'].append('Destroy: ' + str(exc))
    try:
        abi = load(a.repo / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA, '_v82_profile_ABI')
        r['board'] = abi.require_board()  # Must precede NumPy, runtime and shared-library loading.
        pin(a.repo / 'vggt_slam/__init__.py', hashlib.sha256(b'').hexdigest())
        pin(a.repo / 'vggt_slam/rknn_aggregator_blocks.py', HELPER_SHA)
        pin(a.repo / 'deploy/rk3588/reference/rknn_api_v2.3.2.h', HEADER_SHA)
        sys.path.insert(0, str(a.repo.resolve(strict=True)))
        runtime_path = pin(a.repo / 'vggt_slam/rknn_retained_aggregator.py', RUNTIME_SHA)
        adapter = load(a.adapter_source, ADAPTER_SHA, '_v82_calibrated_retained')
        rt = adapter.parent
        require(Path(rt.__file__).resolve() == runtime_path, 'Adapter parent import escaped pinned runtime')
        require(Path(rt.original.__file__).resolve() == (a.repo / 'vggt_slam/rknn_aggregator_blocks.py').resolve(), 'Runtime helper import escaped selected source')
        require('torch' not in sys.modules, 'This component profile must not import Torch')
        import numpy as np
        manifest_path = pin(a.manifest, MANIFEST_SHA); directory = manifest_path.parent
        m = json.loads(manifest_path.read_text())
        quantized_manifest = pin(a.quantized_manifest, QUANTIZED_MANIFEST_SHA)
        mapping, quantized_paths, quantized_identities = adapter.load_mapping(
            quantized_manifest, QUANTIZED_MANIFEST_SHA, manifest_path, MANIFEST_SHA)
        for model, (_, _, digest) in zip(quantized_paths, quantized_identities):
            pin(model, digest)
        r['quantized_identities'] = [dict(group=i, compile_receipt_sha256=c, model_sha256=d)
            for i, c, d in quantized_identities]
        require(m['schema'] == 'v67_original_VGGT_retained_groups_RKNN_manifest'
            and m['status'] == 'compiled_not_quality_accepted' and m['cleanup_errors'] == []
            and m['original_parent_manifest_sha256'] == rt.PARENT_SHA and len(m['blocks']) == 4
            and m['original_target_tokens_global_index'] == 20, 'Frozen complete original group manifest required')
        parent = json.loads(pin(Path(m['original_parent_manifest_directory']) / 'compiled.json', rt.PARENT_SHA).read_text())
        for relative, digest in m['source_hashes'].items():
            source = (a.repo / relative).resolve(strict=True)
            require(source.is_relative_to(a.repo.resolve()), 'Source path escaped repo'); pin(source, digest)
        require({4, 5, 6, 7} <= old_affinity, 'Four original CPU cores unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7}); r['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        api = abi.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so', LIBRARY_SHA))))
        route = SimpleNamespace(base=abi, api=api, np=np, stats=stats, timeout=30000)
        r.update(manifest_sha256=MANIFEST_SHA, runtime_sha256=RUNTIME_SHA,
                 ABI_sha256=ABI_SHA, library_sha256=LIBRARY_SHA)
        for ordinal, row in enumerate(m['blocks']):
            start, end = rt.RANGES[ordinal]; public = rt.bindings(ordinal)
            require(row['index'] == ordinal and row['indices'] == list(range(start, end + 1))
                and row['parents'] == parent['blocks'][start:end + 1]
                and row['input_shape'] == list(rt.FRAME) and row['output_bindings'] == public,
                'Original group lineage or public bindings differ')
            artifact(directory, row); model = artifact(directory, row['rknn'])
            checked = json.loads(artifact(directory, row['check_receipt']).read_text())
            artifact(directory, row['compile_receipt'])
            rt.RKNNRetainedOriginalAggregator._check_receipts(route, directory, row, ordinal, m['export_manifest_sha256'], model)
            require([c['tag'] for c in checked['cases']] == ['v53', 'v56-calib00'], 'Two original saved cases required')
            # The unchanged FP16 model above is checked only for parent lineage.
            # Execute the validated calibrated model selected by the v80 adapter.
            quantized_model = quantized_paths[ordinal]
            context = abi.CONTEXT(); contexts.append(context)
            try:
                filename = ct.create_string_buffer(os.fsencode(quantized_model))
                code = api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 8, None)
                r['contexts_created'] += int(bool(context.value)); abi.check_call(code, 'rknn_init')
                require(bool(context.value), 'Empty SDK context'); abi.check_call(api.rknn_set_core_mask(context, 7), 'rknn_set_core_mask')
                counts = abi.query(api, context, 0, abi.InputOutputNum())
                require((counts.n_input, counts.n_output) == (1, len(public)), 'Original 2/2/2/3 outputs required')
                version = abi.query(api, context, 5, abi.SDKVersion())
                require(re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()), 'SDK2.3.2 required')
                descriptors = []
                for command, bindings in ((1, [dict(index=0, name='input', shape=list(rt.FRAME))]), (2, public)):
                    for b in bindings:
                        attr = abi.query(api, context, command, abi.TensorAttr(index=b['index']))
                        descriptors.append(adapter.calibrated_io_attr(abi, attr, b['shape'], b['name'], b['index']))
                mem = abi.query(api, context, 6, abi.MemSize())
                r['queries'].append(dict(group=ordinal, indices=row['indices'], model_sha256=quantized_identities[ordinal][2],
                    original_FP16_model_sha256=row['rknn']['sha256'],
                    input=descriptors[0], outputs=descriptors[1:], SDK_version=bytes(version.api_version).decode(),
                    driver_version=bytes(version.drv_version).decode(), weight_bytes=int(mem.total_weight_size),
                    internal_bytes=int(mem.total_internal_size), dma_bytes=int(mem.total_dma_allocated_size)))
                for case in checked['cases']:
                    x = np.load(artifact(case['directory'], case['input']), allow_pickle=False)
                    require(x.shape == rt.FRAME and x.dtype == np.float32 and x.flags.c_contiguous
                        and np.isfinite(x).all() and hashlib.sha256(x.tobytes()).hexdigest() == case['input']['raw_sha256'], 'Original captured input differs')
                    before = sample(); require(before['npu_hz'] == a.expected_npu_hz, 'Unexpected NPU frequency before SDK call')
                    previous = dict(stats['fp32_io'])
                    y = adapter.RKNNCalibratedRetained._run_group(route, context, x, public)
                    after = sample(); require(after['npu_hz'] == a.expected_npu_hz, 'Unexpected NPU frequency after SDK call')
                    perf = abi.perf_detail(api, context)
                    require(len(perf['raw_text'].encode()) == perf['bytes']
                        and hashlib.sha256(perf['raw_text'].encode()).hexdigest() == perf['raw_sha256'], 'Profile text must preserve SDK bytes')
                    stem = f'group{ordinal}-{case["tag"]}'; text = a.output_dir / (stem + '.txt')
                    text.write_bytes(perf['raw_text'].encode()); npz = a.output_dir / (stem + '.npz')
                    with npz.open('xb') as stream: np.savez(stream, **{f'output{i}': v for i, v in enumerate(y)})
                    require(hashlib.sha256(x.tobytes()).hexdigest() == case['input']['raw_sha256'], 'Captured input mutated')
                    r['calls'].append(dict(group=ordinal, input=case['tag'], source_input=case['input'],
                        output_bindings=public, timing={k: v - previous[k] for k, v in stats['fp32_io'].items()},
                        platform_before=before, platform_after=after, profile=dict(file=text.name, bytes=text.stat().st_size, sha256=sha(text)),
                        artifact=dict(file=npz.name, bytes=npz.stat().st_size, sha256=sha(npz)),
                        outputs=[dict(index=i, shape=list(v.shape), dtype=str(v.dtype), tensor_bytes=v.nbytes,
                            raw_sha256=hashlib.sha256(v.tobytes()).hexdigest(), finite=True, owned=True) for i, v in enumerate(y)]))
            finally: destroy(context)
        require(len(r['calls']) == 8 and stats['sdk_runs_succeeded'] == stats['sdk_run_attempts'] == 8
            and stats['outputs_acquired'] == stats['outputs_released'] == 18, 'Eight profile calls and eighteen released outputs required')
        require(stats['fp32_io']['calls_attempted'] == stats['fp32_io']['calls_completed'] == 8,
            'All eight original FP32 API routes must complete')
        require('torch' not in sys.modules, 'Component profile must not import Torch')
        r['status'] = 'completed_original_group_profiles_not_quality_accepted'
    except BaseException as exc: r['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for context in reversed(contexts): destroy(context)
        r['live_contexts'] = sum(bool(c.value) for c in contexts); r['stats'] = stats
        try:
            require(adapter is not None and quantized_paths is not None, 'Calibrated mapping not loaded')
            _, final_paths, final_identities = adapter.load_mapping(
                a.quantized_manifest, QUANTIZED_MANIFEST_SHA, a.manifest, MANIFEST_SHA)
            r['quantized_lineage_rechecked'] = (final_paths == quantized_paths and final_identities == quantized_identities)
        except BaseException as exc:
            r['quantized_lineage_rechecked'] = False; r['cleanup_errors'].append('Mapping identity: ' + str(exc))
        try: r['source_and_inputs_unchanged'] = bool(pins) and all(sha(path) == digest for path, digest in pins.items())
        except BaseException as exc: r['source_and_inputs_unchanged'] = False; r['cleanup_errors'].append('Identity: ' + str(exc))
        try:
            os.sched_setaffinity(0, old_affinity); r['affinity_restored'] = os.sched_getaffinity(0) == old_affinity
        except BaseException as exc: r['affinity_restored'] = False; r['cleanup_errors'].append('Affinity: ' + str(exc))
        r['environment_changed'] = False
        if (r['cleanup_errors'] or r['live_contexts'] or not r['source_and_inputs_unchanged']
                or not r['affinity_restored'] or not r['quantized_lineage_rechecked']
                or r['contexts_created'] != r['contexts_destroyed']
                or (r['status'] == 'completed_original_group_profiles_not_quality_accepted'
                    and r['contexts_created'] != 4)):
            r['status'] = 'failed_cleanup_or_identity'
        tmp = a.output_dir / 'result.json.tmp'; tmp.write_text(json.dumps(r, indent=2, allow_nan=False) + '\n'); tmp.replace(a.output_dir / 'result.json')
        print(json.dumps({k: r.get(k) for k in ('status', 'error', 'contexts_created', 'contexts_destroyed', 'cleanup_errors')}), flush=True)
    return 0 if r['status'] == 'completed_original_group_profiles_not_quality_accepted' else 1


if __name__ == '__main__': sys.exit(main())
