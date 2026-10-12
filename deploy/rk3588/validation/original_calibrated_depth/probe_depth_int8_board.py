#!/usr/bin/env python3
"""Board-only original DPT FP16/W8A8 comparison on two frozen real frames.

No checkpoint loading or model/head construction. Reuse the pinned original
depth SDK route and original CPU activate_head. Full component timing includes
input views/finite, synchronous SDK conversions/copy/release, original activation
and chunk output views/finite. Hashes, saves and comparisons are outside timing.
The authorized external controller owns NPU clock and thermal bounds.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
PARENT_SHA = '7efacdb27be4aeb134fd2b64e2819136320fc45d25973517584522d314ba7894'
AGGREGATOR_SHA = '4fea665334e9d12a3d55839c3150680e269bab9794074d698bed2ad0a6e41c93'
MANIFEST_SHA = '9a6e5435df543b8484a4fc50da4b065995f32153233548829234dd149fbc7135'
CAPTURE_SHA = 'fa33dc6a05c2e286ca9c9a17c0f62df21ee2b19da0f3ea693fcff3718bc99033'
CHECK_SHA = '0e32292723b9b97dd5c177e08ec0579ca59b312d23634f3215b8e574b27c143b'
CONTRACT_SHA = 'efdad006ed594636041e5fdeb315b81ec8154d89d74887fabcf26f77e12bb9ba'
HEAD_ACT_SHA = '428bdb087f457b809a66e22d7d01147c1cb4dab43bb0d98bc61c7f05d8383a1e'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
SCHEDULE = 'AB' + 'ABBA' * 3
INPUT_NAMES = ('tokens_4', 'tokens_11', 'tokens_17', 'tokens_23')
INPUT_SHAPE = (1, 48, 2048)
OUTPUT_SHAPE = (1, 2, 84, 112)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sample():
    root = Path('/sys/class/devfreq/fdab0000.npu')
    return dict(npu_hz=int((root / 'cur_freq').read_text()),
        governor=(root / 'governor').read_text().strip(),
        temperatures_c={(p / 'type').read_text().strip(): int((p / 'temp').read_text()) / 1000
            for p in Path('/sys/class/thermal').glob('thermal_zone*')
            if (p / 'type').is_file() and (p / 'temp').is_file()})


def compare(value, reference, np):
    require(value.dtype == reference.dtype == np.float32 and value.shape == reference.shape
        and np.isfinite(value).all() and np.isfinite(reference).all(), 'Comparison tensors differ')
    difference = value.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    tolerance = 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))
    outside = int(np.count_nonzero(absolute > tolerance))
    return dict(elements=int(value.size), max_abs=float(absolute.max()),
        rmse=float(np.sqrt(np.mean(difference * difference))),
        relative_l2=float(np.linalg.norm(difference) / max(float(np.linalg.norm(reference.astype(np.float64))), 1e-30)),
        outside_fixed_gate=outside, maximum_tolerance_ratio=float(np.max(absolute / tolerance)),
        fixed_gate_pass=outside == 0, bitwise_equal=value.tobytes() == reference.tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=BASE / 'src')
    parser.add_argument('--depth-manifest', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--original-check', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--adapter-source', type=Path, required=True)
    parser.add_argument('--adapter-sha256', required=True)
    parser.add_argument('--compiler-source', type=Path, required=True)
    parser.add_argument('--compiler-sha256', required=True)
    parser.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout-ms', type=int, default=30000)
    parser.add_argument('--expected-npu-hz', type=int, choices=(1000000000,), default=1000000000)
    args = parser.parse_args(argv)
    require(0 < args.timeout_ms <= 30000, 'Timeout must be 1..30000 ms')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    affinity, old_path = os.sched_getaffinity(0), sys.path[:]
    env_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {key: os.environ.get(key) for key in env_names}
    contexts, pins, base, api = [], {}, None, None
    report = dict(schema='v85_original_DPT_FP16_INT8_component_probe', status='failed',
        script_sha256=sha(__file__), original_model_only=True,
        checkpoint_or_trained_model_constructed_or_loaded=False,
        Torch_scope='Pinned original CPU FP32 activate_head only; no learned module construction',
        actual_execution_on_board_only=True, host_GPU_used=False, GPU_used=False,
        candidate_accepted=False, quality_accepted=False,
        full_model_or_trajectory_or_realtime_validated=False,
        requested_quantized_dtype='w8a8', API_FP32_conversion_not_internal_FP32=True,
        all_internal_operators_INT8_verified=False, init_flags=0, core_mask=7,
        normal_benchmark_timing=True, schedule_per_frame=SCHEDULE, captured_frames=2,
        hidden_SDK_warmup_calls=0, first_per_route_per_frame=1, warm_per_route_per_frame=6,
        clock_writes=False, expected_npu_hz=args.expected_npu_hz,
        frequency_scope='Before/after each complete component; external controller owns clock/thermal bounds',
        input_scope='Frozen v54 original CPU tokens for two real frames; independent component inputs, not live replay',
        activation=dict(source_sha256=HEAD_ACT_SHA, depth='exp', confidence='expp1', original_chunk_views=True),
        capture_sha256=CAPTURE_SHA, original_manifest_sha256=MANIFEST_SHA,
        original_FP32_check_sha256=CHECK_SHA, candidate_receipt_sha256=args.candidate_sha256,
        adapter_source_sha256=args.adapter_sha256, compiler_source_sha256=args.compiler_sha256,
        strict_rtol=1e-4, strict_atol=1e-5, calls=[], graphs=[], cleanup_errors=[],
        contexts_created=0, contexts_destroyed=0, sdk_run_attempts=0, sdk_runs_succeeded=0,
        outputs_acquired=0, outputs_released=0,
        timing_scope='Original patch/frame input views and finite checks; full synchronous FP32 API conversions/run/get/owned copy/release; original CPU activation/final chunk views and finite checks. No saves/hash/comparison/profile queries.')

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(re.fullmatch('[0-9a-f]{64}', digest) and (size is None or path.stat().st_size == size)
            and sha(path) == digest, 'Frozen file identity differs: ' + str(path))
        pins[path] = digest
        return path

    def artifact(root, row):
        root, relative = Path(root).resolve(strict=True), Path(row['file'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), 'Artifact escaped original directory')
        return pin(path, row['sha256'], row['bytes'])

    def load(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    setup_begin = time.perf_counter()
    try:
        validation = args.repo / 'deploy/rk3588/validation'
        base = load(validation / 'run-v12-rknn-block-board.py', ABI_SHA, '_v85_depth_ABI')
        report['board'] = base.require_board()  # Before NumPy/Torch/SDK libraries.
        require({4, 5, 6, 7} <= affinity, 'Original CPUs 4..7 unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update({key: '1' for key in env_names})
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report['NumPy_version'], report['Torch_version'] = np.__version__, torch.__version__
        C = load(validation / 'original_depth/depth_contract.py', CONTRACT_SHA, '_v85_depth_contract')
        require(C.INPUT_NAMES == INPUT_NAMES and C.INPUT_SHAPE == INPUT_SHAPE and C.OUTPUT_SHAPE == OUTPUT_SHAPE,
            'Original depth contract differs')
        for name, digest in C.SOURCE_HASHES.items():
            pin(args.repo / name, digest)
        head_act = load(args.repo / 'third_party/vggt/vggt/heads/head_act.py', HEAD_ACT_SHA, '_v85_depth_head_act')
        pin(args.repo / 'vggt_slam/__init__.py', hashlib.sha256(b'').hexdigest())
        pin(args.repo / 'vggt_slam/rknn_depth_head.py', PARENT_SHA)
        pin(args.repo / 'vggt_slam/rknn_aggregator_blocks.py', AGGREGATOR_SHA)
        pin(args.repo / 'deploy/rk3588/reference/rknn_api_v2.3.2.h', HEADER_SHA)
        sys.path.insert(0, str(args.repo.resolve(strict=True)))
        adapter = load(args.adapter_source, args.adapter_sha256, '_v85_depth_calibrated')
        require(Path(adapter.parent.__file__).resolve() == (args.repo / 'vggt_slam/rknn_depth_head.py').resolve(),
            'Original depth runtime import escaped source root')
        receipt_path, candidate, model_B = adapter.load_compilation(args.candidate_dir / 'result.json',
            args.candidate_sha256, args.repo, compiler_source=args.compiler_source,
            compiler_sha256=args.compiler_sha256)
        pin(receipt_path, args.candidate_sha256)
        pin(args.compiler_source, args.compiler_sha256)
        model_B = pin(model_B, candidate['rknn']['sha256'], candidate['rknn']['bytes'])
        manifest_path = pin(args.depth_manifest, MANIFEST_SHA)
        manifest = json.loads(manifest_path.read_text())
        require(manifest['schema'] == 'v54_original_VGGT_depth_logits_RKNN_manifest'
            and manifest['checkpoint'] == C.CHECKPOINT
            and manifest['contract_source_sha256'] == CONTRACT_SHA
            and len(manifest['parameters']) == 62 and len(manifest['positions']) == 5
            and manifest['all_62_original_parameter_values_preserved_in_graph'] is True
            and manifest['original_inplace_relu_residual_skip_preserved'] is True
            and manifest['all_five_original_align_corners_resizes_preserved'] is True,
            'Unchanged original FP16 depth manifest required')
        artifact(manifest_path.parent, manifest['graph'])
        model_A = artifact(manifest_path.parent, manifest['rknn'])
        capture_path = pin(args.capture_dir / 'capture.json', CAPTURE_SHA)
        capture = json.loads(capture_path.read_text())
        require(capture['schema'] == 'v54_original_VGGT_depth_capture'
            and capture['status'] == 'completed_original_FP32_depth_capture'
            and capture['checkpoint'] == C.CHECKPOINT and capture['architecture'] == manifest['architecture']
            and capture['original_model_only'] is True and capture['actual_inference_on_board_only'] is True
            and capture['model_parameter_identity_unchanged'] is True and capture['hooks_removed'] is True
            and capture['cleanup_errors'] == [] and capture['original_raw_logits_calls'] == 2,
            'Original two-frame CPU capture proof required')
        fields = ('shape', 'dtype', 'tensor_bytes', 'raw_sha256')
        require(set(capture['artifacts']['state']['tensors']) == set(C.PARAMETER_SHAPES)
            and all({k: row[k] for k in fields} == capture['artifacts']['state']['tensors'][name]
                for name, row in manifest['parameters'].items()), 'Original 62 learned tensor metadata differs')
        # Do not deserialize state/weights or construct any trained module.
        check = json.loads(pin(args.original_check, CHECK_SHA).read_text())
        require(check['schema'] == 'v54_original_depth_FP32_ORT_check'
            and check['status'] == 'passed_original_FP32_logit_gate'
            and check['capture_sha256'] == CAPTURE_SHA
            and check['export_sha256'] == manifest['export_manifest_sha256']
            and len(check['calls']) == 2 and all(x['fixed_gate_pass'] for x in check['calls']),
            'Frozen original FP32 graph check required')

        def owns_numpy_storage(value):
            # The original route returns copy().reshape(), whose view owns its
            # storage through a NumPy base; flags.owndata alone would reject it.
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            return bool(root.flags.owndata)

        def info(value):
            require(value.dtype == np.float32 and value.flags.c_contiguous and np.isfinite(value).all(),
                'Contiguous finite FP32 tensor required')
            return dict(shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes,
                raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(), finite=True, owned=owns_numpy_storage(value))

        def captured_arrays(key, shapes):
            row = capture['artifacts'][key]
            with np.load(artifact(capture_path.parent, row), allow_pickle=False) as archive:
                require(set(archive.files) == set(shapes) == set(row['tensors']), 'Original archive keys differ')
                values = {name: np.array(archive[name], copy=True, order='C') for name in shapes}
            for name, value in values.items():
                actual = info(value)
                require(tuple(value.shape) == shapes[name]
                    and {k: actual[k] for k in fields} == row['tensors'][name], 'Captured original array differs: ' + name)
            return values

        tokens = captured_arrays('tokens', {name: (1, 2, 53, 2048) for name in INPUT_NAMES})
        references = captured_arrays('references', {'raw_logits': (2, 2, 84, 112),
            'depth': (1, 2, 84, 112, 1), 'depth_conf': (1, 2, 84, 112)})
        report['original_input_and_reference_artifacts'] = {key: capture['artifacts'][key] for key in ('tokens', 'references')}
        calibration_row = candidate['calibration']['capture']
        calibration_path = pin(calibration_row['file'], calibration_row['sha256'], calibration_row['bytes'])
        calibration = json.loads(calibration_path.read_text())
        calibration_images = [item for row in calibration['samples'] for item in row['images']]
        require(len(calibration_images) == 16 and len(capture['source_images']) == 2,
            'Sixteen calibration and two original source image identities required')
        for image in calibration_images + capture['source_images']:
            pin(image['file'], image['sha256'], image['bytes'])
        calibration_shas = {item['sha256'] for item in calibration_images}
        overlap = sorted(calibration_shas & {item['sha256'] for item in capture['source_images']})
        report['source_image_split'] = dict(original_images=capture['source_images'],
            calibration_image_count=16, calibration_image_sha256=sorted(calibration_shas), overlap_sha256=overlap,
            held_out_by_source_image_SHA=not overlap,
            scope='Source RGB image identity only; two frames from the same dataset are not a trajectory or dataset-generalization acceptance.')
        per_frame = []
        for frame in range(2):
            values = [np.ascontiguousarray(tokens[name][:, frame, 5:, :]) for name in INPUT_NAMES]
            require(all(value.shape == INPUT_SHAPE for value in values)
                and {name: info(value)['raw_sha256'] for name, value in zip(INPUT_NAMES, values)}
                    == check['calls'][frame]['original_input_raw_sha256'], 'Original patch/frame selection differs')
            expected = dict(logits=np.ascontiguousarray(references['raw_logits'][frame:frame + 1]),
                depth=np.ascontiguousarray(references['depth'][:, frame:frame + 1]),
                depth_conf=np.ascontiguousarray(references['depth_conf'][:, frame:frame + 1]))
            with torch.inference_mode():
                depth, conf = head_act.activate_head(torch.from_numpy(expected['logits']), activation='exp', conf_activation='expp1')
                activated = dict(depth=depth.view(1, 1, *depth.shape[1:]).contiguous().numpy().copy(),
                    depth_conf=conf.view(1, 1, *conf.shape[1:]).contiguous().numpy().copy())
            require(all(compare(activated[name], expected[name], np)['bitwise_equal'] for name in activated),
                'Pinned original CPU activation does not reproduce captured reference')
            per_frame.append(expected)
        report['original_activation_reference_byte_equal'] = True
        library = base.bind_api(ct.CDLL(str(pin(args.library, LIBRARY_SHA))))

        class Counted:
            def __getattr__(self, name):
                function = getattr(library, name)
                def call(*values):
                    acquired = sum(bool(values[2][i].buf) for i in range(values[1])) if name == 'rknn_outputs_release' else 0
                    if name == 'rknn_run': report['sdk_run_attempts'] += 1
                    code = function(*values)
                    if name == 'rknn_run': report['sdk_runs_succeeded'] += int(code == 0)
                    if name == 'rknn_outputs_get': report['outputs_acquired'] += sum(bool(values[2][i].buf) for i in range(values[1]))
                    if name == 'rknn_outputs_release': report['outputs_released'] += acquired * int(code == 0)
                    return code
                return call

        api, owners = Counted(), []
        for route, path in (('A', model_A), ('B', model_B)):
            context = base.CONTEXT(); contexts.append(context)
            filename = ct.create_string_buffer(os.fsencode(path))
            code = api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
            report['contexts_created'] += int(bool(context.value)); base.check_call(code, 'rknn_init DPT ' + route)
            require(bool(context.value), 'Empty DPT context')
            base.check_call(api.rknn_set_core_mask(context, 7), 'rknn_set_core_mask DPT ' + route)
            version = base.query(api, context, 5, base.SDKVersion())
            require(re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()), 'Typed SDK2.3.2 required')
            count = base.query(api, context, 0, base.InputOutputNum())
            require((count.n_input, count.n_output) == (4, 1), 'Original four-input/one-logit ABI required')
            attrs = [base.query(api, context, 1, base.TensorAttr(index=i)) for i in range(4)]
            output = base.query(api, context, 2, base.TensorAttr(index=0))
            if route == 'A':
                descriptors = [adapter.parent.floating_binding(base, attr, i, name, INPUT_SHAPE, (3,))
                    for i, (attr, name) in enumerate(zip(attrs, INPUT_NAMES))]
                out_desc = adapter.parent.floating_binding(base, output, 0, 'logits', OUTPUT_SHAPE, (0, 3))
            else:
                descriptors = [adapter.calibrated_depth_io_attr(base, attr, INPUT_SHAPE, name, i)
                    for i, (attr, name) in enumerate(zip(attrs, INPUT_NAMES))]
                out_desc = adapter.calibrated_depth_io_attr(base, output, OUTPUT_SHAPE, 'logits', 0, formats=(0, 3))
            memory = base.query(api, context, 6, base.MemSize())
            report['graphs'].append(dict(route=route, model_sha256=pins[path], model_bytes=path.stat().st_size,
                graph_sha256=manifest['graph']['sha256'], inputs=descriptors, output=out_desc,
                SDK_version=bytes(version.api_version).decode(), driver_version=bytes(version.drv_version).decode(),
                memory=dict(weight_bytes=int(memory.total_weight_size), internal_bytes=int(memory.total_internal_size),
                    dma_bytes=int(memory.total_dma_allocated_size)),
                internal_precision='Original FP16' if route == 'A' else 'Requested W8A8/channel/normal PTQ'))
            owners.append(SimpleNamespace(base=base, api=api, contexts=[context], np=np, inputs=descriptors,
                timeout=args.timeout_ms, contract=C, stats=dict(sdk_run_attempts=0, sdk_runs_succeeded=0,
                    sdk_io_seconds=0., cleanup_errors=report['cleanup_errors'])))

        report['setup_seconds_excludes_calls'] = time.perf_counter() - setup_begin
        for frame in range(2):
            first, previous = {}, {}
            for index, route in enumerate(SCHEDULE):
                owner = owners[route == 'B']
                before = sample(); require(before['npu_hz'] == args.expected_npu_hz, 'Unexpected NPU frequency before component')
                sdk_before = owner.stats['sdk_io_seconds']
                begin = time.perf_counter()
                arrays = [np.ascontiguousarray(tokens[name][:, frame, 5:, :]) for name in INPUT_NAMES]
                require(all(value.dtype == np.float32 and value.shape == INPUT_SHAPE and np.isfinite(value).all()
                    for value in arrays), 'Original four finite FP32 patch views required')
                prepared = time.perf_counter()
                raw = adapter.parent.RKNNOriginalDepthHead._run_depth(owner, arrays)
                require(raw.shape == OUTPUT_SHAPE and raw.dtype == np.float32 and owns_numpy_storage(raw)
                    and np.isfinite(raw).all(), 'Owned dense finite FP32 DPT logits required')
                activation_begin = time.perf_counter()
                with torch.inference_mode():
                    depth, conf = head_act.activate_head(torch.from_numpy(raw), activation='exp', conf_activation='expp1')
                    depth = depth.view(1, 1, *depth.shape[1:])
                    conf = conf.view(1, 1, *conf.shape[1:])
                    values = dict(logits=raw, depth=depth.contiguous().numpy().copy(), depth_conf=conf.contiguous().numpy().copy())
                    require(all(value.dtype == np.float32 and value.flags.c_contiguous and np.isfinite(value).all()
                        for value in values.values()), 'Original activated depth/conf must be finite FP32')
                end = time.perf_counter()
                after = sample(); require(after['npu_hz'] == args.expected_npu_hz, 'Unexpected NPU frequency after component')
                first_call = route not in first
                if first_call: first[route] = values
                target = out / f'frame{frame}-call{index:02d}-{route}.npz'
                with target.open('xb') as stream: np.savez(stream, **values)
                record = dict(frame=frame, index=index, route=route, first_for_frame_route=first_call,
                    complete_component_seconds=end - begin, input_preparation_seconds=prepared - begin,
                    sdk_io_seconds=owner.stats['sdk_io_seconds'] - sdk_before,
                    activation_and_owned_finite_seconds=end - activation_begin,
                    platform_before=before, platform_after=after,
                    vs_original_CPU={name: compare(value, per_frame[frame][name], np) for name, value in values.items()},
                    vs_first_same_frame_route={name: compare(value, first[route][name], np) for name, value in values.items()},
                    artifact=dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target),
                        tensors={name: info(value) for name, value in values.items()}))
                if route in previous:
                    record['bitwise_equal_previous_same_frame_route'] = all(values[name].tobytes() == previous[route][name].tobytes() for name in values)
                if route == 'B':
                    record['vs_first_A'] = {name: compare(value, first['A'][name], np) for name, value in values.items()}
                previous[route] = values
                report['calls'].append(record)
        report['summary_seconds'] = {route: dict(calls=14, excluded_first_per_frame=2, warm_count=12,
            first_by_frame=[next(c['complete_component_seconds'] for c in report['calls'] if c['route'] == route and c['frame'] == frame) for frame in range(2)],
            warm_mean=float(np.mean([c['complete_component_seconds'] for c in report['calls'] if c['route'] == route and not c['first_for_frame_route']])),
            warm_sdk_io_mean=float(np.mean([c['sdk_io_seconds'] for c in report['calls'] if c['route'] == route and not c['first_for_frame_route']]))) for route in 'AB'}
        report['repeated_outputs_byte_stable'] = {route: all(c['bitwise_equal_previous_same_frame_route'] for c in report['calls']
            if c['route'] == route and not c['first_for_frame_route']) for route in 'AB'}
        require(len(report['calls']) == 28 and all(sum(c['frame'] == frame and c['route'] == route
            for c in report['calls']) == 7 for frame in range(2) for route in 'AB'), 'Per-frame AB+ABBA*3 schedule incomplete')
        report['status'] = 'completed_original_DPT_component_AB_not_quality_accepted'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for context in reversed(contexts):
            if context.value:
                try:
                    base.check_call(api.rknn_destroy(context), 'rknn_destroy DPT')
                    context.value = 0; report['contexts_destroyed'] += 1
                except BaseException as exc:
                    report['cleanup_errors'].append('Context cleanup: ' + str(exc))
        report['live_contexts'] = sum(bool(context.value) for context in contexts)
        if report['status'].startswith('completed'):
            report['lifecycle_verified'] = (report['contexts_created'] == report['contexts_destroyed'] == 2
                and report['sdk_run_attempts'] == report['sdk_runs_succeeded'] == 28
                and report['outputs_acquired'] == report['outputs_released'] == 28)
            if not report['lifecycle_verified']: report['cleanup_errors'].append('2 contexts / 28 runs / 28 outputs invariant failed')
        try:
            report['source_and_inputs_unchanged'] = bool(pins) and all(sha(path) == digest for path, digest in pins.items())
        except BaseException as exc:
            report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity cleanup: ' + str(exc))
        try:
            os.sched_setaffinity(0, affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            for key, value in old_env.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
            sys.path[:] = old_path
            report['environment_restored'] = all(os.environ.get(key) == value for key, value in old_env.items()) and sys.path == old_path
        except BaseException as exc:
            report['affinity_restored'] = report['environment_restored'] = False
            report['cleanup_errors'].append('Process environment/affinity cleanup: ' + str(exc))
        if report['cleanup_errors'] or report['live_contexts'] or not report.get('source_and_inputs_unchanged') or not report.get('affinity_restored') or not report.get('environment_restored'):
            report['status'] = 'failed_cleanup_or_identity'
        temporary = out / 'result.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(out / 'result.json')
        print(json.dumps(dict(status=report['status'], error=report.get('error'), receipt_sha256=sha(out / 'result.json'))), flush=True)
    return 0 if report['status'].startswith('completed') else 1


if __name__ == '__main__':
    sys.exit(main())
