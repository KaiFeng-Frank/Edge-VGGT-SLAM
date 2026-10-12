#!/usr/bin/env python3
"""Bounded original CameraHead split4 FP16/PTQ comparison on RK3588S only.

One frozen v15 real token pair, original learned empty pose, four sequential
unactivated FP32 refinements and pinned original head_act. No full model is
constructed. File hashes, saves and comparisons are outside head timing.
The external authorized controller owns the NPU frequency and thermal limit.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
STEP_SHA = 'dbe4fc933891581099084a569e2e847e15c582082680bf31bcf5dacc86210109'
HELPER_SHA = 'b26efe9c6cd62a2b7977e6e9c096cf0bd9bbf63076a063b9a332ef59565af258'
CAMERA_ADAPTER_SHA = 'e20518d08c47746e1bf58e07bd6d0270bc7c7f6daa4d0858593f478ce6dfb09e'
CALIBRATED_ADAPTER_SHA = 'f3f106034c6621a1e3ed07bdb265107b8522103f7d186ba4bbe0d0cefda16f2f'
RETAINED_SHA = '04fbb5b2ad7db22bf26d52d4a6b5a3fadffa042557919575938b035f5dd06b5f'
RETAINED_HELPER_SHA = '4fea665334e9d12a3d55839c3150680e269bab9794074d698bed2ad0a6e41c93'
HEAD_ACT_SHA = '428bdb087f457b809a66e22d7d01147c1cb4dab43bb0d98bc61c7f05d8383a1e'
CAPTURE_SHA = '1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1'
EXPORT_SHA = '8ccf7ff593346962766b4eae6da7eb95ce43eeaa3183a8fc0e82bf134c172503'
ORIGINAL_COMPILE_SHA = 'c8771d743d9840bcd32b22c6894f791ed239b4ad3795db703e7f7122329fbc7a'
SPLIT_COMPILE_SHA = 'a2cdad4a6d2af60c6b3244dfefb3d1a1fadead72c6b2374ef2c089912d318f74'
SPLIT_CHECK_SHA = '9c58744a713a8ad686e66e811163c260e82f6522f0a936dbcc92cb5fdfdc4c2a'
SPLIT_GRAPH_SHA = 'f1b9eee30144ec643aa351db724c2f740a79d82a8f9df5468f4338e377600a7f'
SPLIT_GRAPH_BYTES = 864771004
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
TAIL_CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
INPUTS = [('camera_tokens', [1, 2, 2048]), ('conditioning_pose', [1, 2, 9])]
SCHEDULE = 'AB' + 'ABBA' * 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def sample():
    root = Path('/sys/class/devfreq/fdab0000.npu')
    return dict(npu_hz=int((root / 'cur_freq').read_text()),
        governor=(root / 'governor').read_text().strip(),
        temperatures_c={(p / 'type').read_text().strip(): int((p / 'temp').read_text()) / 1000
            for p in Path('/sys/class/thermal').glob('thermal_zone*')
            if (p / 'type').is_file() and (p / 'temp').is_file()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--capture-dir', type=Path, default=BASE / 'experiments/vggt-v15-camera/capture112-r1')
    p.add_argument('--export-dir', type=Path, default=BASE / 'experiments/vggt-v16-camera-step/export112-r1')
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--candidate-sha256', required=True, help='Completed PTQ compiler result.json SHA256')
    p.add_argument('--compiler-source', type=Path, required=True)
    p.add_argument('--compiler-sha256', required=True)
    p.add_argument('--calibrated-adapter-source', type=Path,
        default=BASE / 'experiments/vggt-v80-source-r1/rknn_calibrated_retained.py')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--expected-npu-hz', type=int, choices=(1000000000,), default=1000000000)
    a = p.parse_args(argv)
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0)
    env_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {key: os.environ.get(key) for key in env_names}
    contexts, pins, base, api = [], {}, None, None
    r = dict(schema='v82_original_CameraHead_split4_FP16_INT8_probe', status='failed',
        script_sha256=sha(__file__), original_model_only=True, original_Torch_model_constructed=False,
        Torch_scope='Original CPU FP32 head_act only; no checkpoint loading or trained-model construction',
        candidate_accepted=False, quality_accepted=False, full_model_or_trajectory_or_realtime_validated=False,
        requested_quantized_dtype='w8a8', API_FP32_conversion_not_internal_FP32=True,
        all_internal_operators_INT8_verified=False, actual_execution_on_board_only=True,
        init_flags=0, core_mask=7, normal_benchmark_timing=True, schedule=SCHEDULE,
        hidden_SDK_warmup_calls=0, first_per_route=1, warm_per_route=6,
        required_outer_iterations=4, trunk_blocks_per_step=4,
        clock_writes=False, expected_npu_hz=a.expected_npu_hz,
        frequency_scope='Before and after each complete head; external controller owns clocks/thermal bounds',
        input_scope='Frozen v15 real original camera tokens; component comparison, not a fresh-observation replay',
        conditioning_semantics='First original learned empty pose; later previous unactivated FP32 prediction',
        activation=dict(source_sha256=HEAD_ACT_SHA, trans_act='linear', quat_act='linear', fl_act='relu'),
        capture_sha256=CAPTURE_SHA, export_sha256=EXPORT_SHA,
        candidate_receipt_sha256=a.candidate_sha256, compiler_source_sha256=a.compiler_sha256,
        strict_rtol=1e-4, strict_atol=1e-5, calls=[], graphs=[], cleanup_errors=[],
        contexts_created=0, contexts_destroyed=0, outputs_acquired=0, outputs_released=0,
        sdk_run_attempts=0, sdk_runs_succeeded=0, npu_run_api_called=False, npu_run_api_call_count=0,
        timing_scope='All four sequential SDK routes, original unactivated FP32 accumulation, four pinned original head_act calls and finite checks. No hashes/saves/comparisons/profile queries in head timing.')

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require((size is None or path.stat().st_size == size) and sha(path) == digest,
            'Frozen file identity differs: ' + str(path))
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

    try:
        validation = a.repo / 'deploy/rk3588/validation'
        base = load(validation / 'run-v12-rknn-block-board.py', ABI_SHA, '_v82_camera_ABI')
        r['board'] = base.require_board()  # Before NumPy/Torch/runtime/shared libraries.
        require({4, 5, 6, 7} <= affinity, 'Original CPUs 4..7 unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update({key: '1' for key in env_names})
        r['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        r['NumPy_version'], r['Torch_version'] = np.__version__, torch.__version__
        step = load(validation / 'run-v16-camera-step-board.py', STEP_SHA, '_v82_camera_step')
        helper = load(validation / 'export-v16-camera-step-board.py', HELPER_SHA, '_v82_camera_reference')
        camera = load(a.repo / 'vggt_slam/rknn_camera_head.py', CAMERA_ADAPTER_SHA, '_v82_camera_lineage')
        head_act = load(a.repo / 'third_party/vggt/vggt/heads/head_act.py', HEAD_ACT_SHA, '_v82_original_head_act')
        pin(a.repo / 'vggt_slam/__init__.py', hashlib.sha256(b'').hexdigest())
        pin(a.repo / 'vggt_slam/rknn_retained_aggregator.py', RETAINED_SHA)
        pin(a.repo / 'vggt_slam/rknn_aggregator_blocks.py', RETAINED_HELPER_SHA)
        sys.path.insert(0, str(a.repo.resolve(strict=True)))
        calibrated = load(a.calibrated_adapter_source, CALIBRATED_ADAPTER_SHA, '_v82_affine_IO')
        require(Path(calibrated.parent.__file__).resolve() == (a.repo / 'vggt_slam/rknn_retained_aggregator.py').resolve(), 'Affine helper parent import escaped selected repo')
        pin(validation.parent / 'reference/rknn_api_v2.3.2.h', HEADER_SHA)

        def raw_sha(value):
            return hashlib.sha256(memoryview(value).cast('B')).hexdigest()

        def info(value):
            require(value.dtype == np.float32 and value.flags.c_contiguous and np.isfinite(value).all(), 'Finite owned contiguous FP32 required')
            return dict(shape=list(value.shape), dtype='float32', finite=True, bytes=value.nbytes, sha256=raw_sha(value))

        def compare(value, reference):
            require(value.shape == reference.shape and value.dtype == reference.dtype == np.float32
                and np.isfinite(value).all() and np.isfinite(reference).all(), 'Invalid original FP32 comparison')
            ref = reference.astype(np.float64)
            difference = np.abs(value.astype(np.float64) - ref)
            outside = int((difference > 1e-5 + 1e-4 * np.abs(ref)).sum())
            return dict(bitwise_equal=value.tobytes() == reference.tobytes(), strict_pass=outside == 0,
                outside=outside, elements=value.size, max_abs=float(difference.max()),
                relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(ref), np.finfo(np.float64).tiny)))

        def activate(prediction):
            helper.pose_array(prediction)
            value = head_act.activate_pose(torch.from_numpy(prediction),
                trans_act='linear', quat_act='linear', fl_act='relu')
            require(value.device.type == 'cpu' and value.dtype == torch.float32 and not value.requires_grad,
                'Original CPU FP32 head_act required')
            result = value.detach().contiguous().numpy().copy()
            helper.pose_array(result)
            return result

        cap = json.loads(pin(a.capture_dir / 'capture.json', CAPTURE_SHA).read_text())
        export = json.loads(pin(a.export_dir / 'export.json', EXPORT_SHA).read_text())
        original_compile = json.loads(pin(validation / 'vggt-v16-camera-step-compile112-r1.json', ORIGINAL_COMPILE_SHA).read_text())
        require(cap['status'] == 'completed' and cap['camera_iterations'] == 4 and not cap['cleanup_errors']
            and all(cap['checkpoint'][key] == value for key, value in CHECKPOINT.items())
            and cap['source_and_inputs_unchanged'] is True
            and export['status'] == 'exported_and_structurally_checked' and not export['cleanup_errors']
            and export['capture']['sha256'] == CAPTURE_SHA and export['source_and_inputs_unchanged'] is True
            and export['weight_sharing']['original_parameters_verified'] == 68
            and export['outer_iteration_contract']['all_four_original_iterations_array_equal'] is True
            and original_compile['status'] == 'compiled_not_runtime_validated' and original_compile['released']
            and original_compile['export_receipt_sha256'] == EXPORT_SHA, 'Original v15/v16 lineage differs')
        weights = cap['artifacts']['camera-state.npz']['tensors']
        require(len(weights) == 69, 'All original 69 camera parameters including empty pose required')
        expected_parameters = {x['parameter']: dict(shape=x['shape'], dtype='float32',
            tensor_bytes=int(np.prod(x['shape'])) * 4, finite=True, raw_sha256=x['raw_sha256'])
            for x in export['weight_sharing']['parameters']}
        empty_info = export['outer_iteration_contract']['empty_weight']
        expected_parameters['empty_pose_tokens'] = dict(shape=empty_info['shape'], dtype='float32',
            tensor_bytes=36, finite=True, raw_sha256=empty_info['sha256'])
        require(len(expected_parameters) == 69 and set(expected_parameters) == set(weights)
            and all(v['shape'] == weights[name]['shape'] and v['raw_sha256'] == weights[name]['sha256']
                for name, v in expected_parameters.items()), 'Original graph/empty parameter metadata differs')
        raw = np.load(artifact(a.capture_dir, dict(file='input.npy', **cap['artifacts']['input.npy'])), allow_pickle=False)
        require(raw.shape == (1, 2, 2048) and info(raw)['sha256'] == cap['artifacts']['input.npy']['tensor_sha256'], 'Original real camera tokens differ')
        expected_final = np.load(artifact(a.capture_dir, dict(file='output.npy', **cap['artifacts']['output.npy'])), allow_pickle=False)
        helper.pose_array(expected_final)
        require(info(expected_final)['sha256'] == cap['artifacts']['output.npy']['tensor_sha256'], 'Original final camera pose differs')
        seed = np.load(artifact(a.export_dir, dict(file='conditioning-seed.npy', **export['artifacts']['conditioning-seed.npy'])), allow_pickle=False)
        empty = np.load(artifact(a.export_dir, dict(file='empty-pose-token.npy', **export['artifacts']['empty-pose-token.npy'])), allow_pickle=False)
        helper.pose_array(seed)
        require(info(empty) == empty_info and seed.tobytes() == np.broadcast_to(empty, (1, 2, 9)).tobytes(), 'Original first learned conditioning differs')
        item = export['artifacts']['step-iterations.npz']
        with np.load(artifact(a.export_dir, dict(file='step-iterations.npz', **item)), allow_pickle=False) as saved:
            references = {key: np.ascontiguousarray(saved[key]) for key in saved.files}
        require(set(references) == {f'{key}_{i}' for key in ('conditioning', 'delta', 'preactivation', 'activation') for i in range(4)}, 'All four original CPU refinements required')
        reference_prediction = None
        with torch.inference_mode():
            for i in range(4):
                for key in ('conditioning', 'delta', 'preactivation', 'activation'):
                    name = f'{key}_{i}'; helper.pose_array(references[name]); require(info(references[name]) == item['tensors'][name], 'Original reference tensor identity differs')
                require(references[f'conditioning_{i}'].tobytes() == (seed if reference_prediction is None else reference_prediction).tobytes(), 'Original unactivated CPU recurrence differs')
                reference_prediction = helper.accumulate_delta(reference_prediction, references[f'delta_{i}'])
                require(reference_prediction.tobytes() == references[f'preactivation_{i}'].tobytes()
                    and activate(reference_prediction).tobytes() == references[f'activation_{i}'].tobytes(), 'Pinned original activation/accumulation differs from v16 reference')
        require(references['activation_3'].tobytes() == expected_final.tobytes(), 'Original final reference differs')
        r['original_reference_activation_contract_checks'] = 4
        r['original_camera_parameters_verified_by_frozen_lineage'] = 69
        baseline_dir = BASE / 'experiments/vggt-v69-camera-step-split4-r1'
        model_A = camera._split4_artifact(a.repo, weights)
        prior_compile = pin(baseline_dir / 'result.json', SPLIT_COMPILE_SHA)
        pin(BASE / 'experiments/vggt-v69-head-ort-r1/result.json', SPLIT_CHECK_SHA)
        for name, digest in [('compile_camera_step_split_board.py', camera.SPLIT_BUILDER_SHA),
                ('compile_camera_fc1_splits_board.py', camera.SPLIT_HELPER_SHA),
                ('probe_camera_step_split_board.py', camera.SPLIT_PROBE_SHA)]:
            pin(validation / 'original_camera_partition' / name, digest)
        pin(baseline_dir / 'camera-step-split4.onnx', SPLIT_GRAPH_SHA, SPLIT_GRAPH_BYTES)
        prior = json.loads(prior_compile.read_text())
        model_A = pin(model_A, prior['rknn']['sha256'], prior['rknn']['bytes'])
        candidate_path = pin(a.candidate_dir / 'result.json', a.candidate_sha256)
        candidate = json.loads(candidate_path.read_text())
        pin(a.compiler_source, a.compiler_sha256)
        calibration = candidate['calibration']
        require(candidate['schema'] == 'v82_original_CameraHead_step_split4_INT8_compile'
            and candidate['status'] == 'compiled_not_quality_accepted'
            and candidate['script_sha256'] == a.compiler_sha256 and candidate['checkpoint'] == CHECKPOINT
            and candidate['graph_sha256'] == candidate['graph']['sha256'] == SPLIT_GRAPH_SHA
            and candidate['graph']['bytes'] == SPLIT_GRAPH_BYTES and candidate['original_ONNX_modified'] is False
            and candidate['prior_FP16_compile_receipt_sha256'] == SPLIT_COMPILE_SHA
            and candidate['prior_FP32_check_receipt_sha256'] == SPLIT_CHECK_SHA
            and candidate['original_export_sha256'] == EXPORT_SHA
            and candidate['required_outer_iterations'] == candidate['trunk_blocks_per_step'] == 4
            and candidate['original_empty_and_FP32_outer_accumulation_unchanged'] is True
            and candidate['original_parameters'] == expected_parameters
            and candidate['configuration'] == CONFIG
            and candidate['build_options'] == dict(do_quantization=True, dataset=str(a.candidate_dir.resolve() / 'dataset.txt'), auto_hybrid=False)
            and candidate['original_model_only'] is True and candidate['candidate_accepted'] is False
            and candidate['quality_accepted'] is False and candidate['calibration_executed'] is True
            and candidate['released'] is True and candidate['release_code'] in (None, 0)
            and candidate['cleanup_errors'] == [] and candidate['source_and_inputs_unchanged'] is True
            and candidate['environment_restored'] is True and candidate['affinity_restored'] is True
            and calibration['source_calibration_sha256'] == CALIBRATION_SHA
            and calibration['original_tail_capture_sha256'] == TAIL_CAPTURE_SHA
            and calibration['pair_count'] == len(calibration['samples']) == 8
            and calibration['iterations_per_pair'] == 4
            and calibration['dataset']['input_count'] == 2 and calibration['dataset']['line_count'] == 32,
            'Completed original 69-parameter/32-input PTQ lineage required')
        require([x['name'] for x in candidate['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
            and all(x['code'] in (None, 0) and x['status'] == 'completed' for x in candidate['operations']), 'Candidate SDK compilation incomplete')
        pin(candidate['graph']['file'], SPLIT_GRAPH_SHA, SPLIT_GRAPH_BYTES)
        capture = calibration['capture']; capture_path = pin(capture['file'], capture['sha256'], capture['bytes'])
        captured = json.loads(capture_path.read_text())
        pin(candidate['capture_source']['file'], candidate['capture_source']['sha256'])
        require(candidate['capture_sha256'] == capture['sha256'] and captured['schema'] == 'v82_original_VGGT_camera_PTQ_calibration_capture'
            and captured['status'] == 'completed_original_FP32_camera_calibration_capture'
            and captured['script_sha256'] == candidate['capture_source']['sha256']
            and captured['checkpoint'] == CHECKPOINT and captured['parameters'] == expected_parameters
            and captured['samples'] == calibration['samples'] and captured['full_head_forward_count'] == 8
            and captured['NPU_or_cache_installed'] is False and captured['model_state_unchanged'] is True
            and captured['all_final_poses_bitwise_equal_to_original_capture'] is True
            and captured['source_and_inputs_unchanged'] is True and captured['affinity_restored'] is True
            and captured['cleanup_errors'] == [], 'Actual original camera calibration capture differs')
        dataset = artifact(candidate_path.parent, calibration['dataset'])
        rows = calibration['dataset']['rows']; require(len(rows) == 32, '32 ordered PTQ rows required')
        expected_lines = []
        for index, row in enumerate(rows):
            s = calibration['samples'][index // 4]; iteration = index % 4
            require(s['sample_id'] == row['sample_id'] == f'calib{index // 4:02d}'
                and s['source_pair_indices'] == row['source_pair_indices'] == [index // 4 * 96, index // 4 * 96 + 48]
                and row['iteration'] == iteration and row['inputs'] == [s['camera_tokens'], s['iterations'][iteration]['conditioning_pose']], 'Ordered original two-input calibration rows differ')
            paths = [str(artifact(capture_path.parent, x)) for x in row['inputs']]
            require(paths == row['paths'], 'Actual calibration input paths differ')
            expected_lines.append(' '.join(paths))
        require(dataset.read_text().splitlines() == expected_lines, 'Pinned two-input PTQ dataset differs')
        for relative, digest in dict(candidate['source_hashes'], **candidate['runtime_source_hashes']).items():
            relative = Path(relative)
            require(not relative.is_absolute() and '..' not in relative.parts, 'Relative original source path required')
            pin(a.repo / relative, digest)
        model_B = artifact(candidate_path.parent, candidate['rknn'])
        graphs = [dict(route='A', model=model_A), dict(route='B', model=model_B)]
        r['graphs'] = [dict(route=x['route'], graph_sha256=SPLIT_GRAPH_SHA,
            model_sha256=pins[x['model']], model_bytes=x['model'].stat().st_size,
            internal_precision='Original split4 FP16' if x['route'] == 'A' else 'Requested W8A8/channel/normal PTQ') for x in graphs]
        library = base.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so', LIBRARY_SHA))))

        class Counted:
            def __getattr__(self, name):
                function = getattr(library, name)
                def call(*args):
                    acquired = sum(bool(args[2][i].buf) for i in range(args[1])) if name == 'rknn_outputs_release' else 0
                    if name == 'rknn_run': r['sdk_run_attempts'] += 1
                    code = function(*args)
                    if name == 'rknn_run': r['sdk_runs_succeeded'] += int(code == 0)
                    if name == 'rknn_outputs_get': r['outputs_acquired'] += sum(bool(args[2][i].buf) for i in range(args[1]))
                    if name == 'rknn_outputs_release': r['outputs_released'] += acquired * int(code == 0)
                    return code
                return call

        api = Counted()
        for graph in graphs:
            context = base.CONTEXT(); contexts.append(context)
            filename = ct.create_string_buffer(os.fsencode(graph['model']))
            code = api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
            r['contexts_created'] += int(bool(context.value)); base.check_call(code, 'rknn_init')
            require(bool(context.value), 'Empty camera context')
            base.check_call(api.rknn_set_core_mask(context, 7), 'rknn_set_core_mask')
            version = base.query(api, context, 5, base.SDKVersion())
            require(bytes(version.api_version).decode().startswith('2.3.2'), 'Frozen SDK2.3.2 required')
            count = base.query(api, context, 0, base.InputOutputNum())
            require((count.n_input, count.n_output) == (2, 1), 'Original two-input/one-delta ABI required')
            attrs = [base.query(api, context, 1, base.TensorAttr(index=i)) for i in range(2)]
            output = base.query(api, context, 2, base.TensorAttr(index=0))
            if graph['route'] == 'A':
                graph['bindings'] = step.validate_bindings(base, attrs, output)
            else:
                bindings = {}
                for i, (attr, (name, shape)) in enumerate(zip(attrs, INPUTS)):
                    desc = calibrated.calibrated_io_attr(base, attr, shape, name, i)
                    bindings[name] = dict(index=desc['index'], format=desc['format'])
                calibrated.calibrated_io_attr(base, output, [1, 2, 9], 'pose_delta', 0)
                graph['bindings'] = bindings
            memory = base.query(api, context, 6, base.MemSize())
            r['graphs'][len(contexts) - 1].update(inputs=[base.attr_record(x) for x in attrs], output=base.attr_record(output),
                SDK_version=bytes(version.api_version).decode(), driver_version=bytes(version.drv_version).decode(),
                memory=dict(weight_bytes=int(memory.total_weight_size), internal_bytes=int(memory.total_internal_size), dma_bytes=int(memory.total_dma_allocated_size)))
        seen, first = {}, {}
        for index, route in enumerate(SCHEDULE):
            graph, context = graphs[route == 'B'], contexts[route == 'B']
            before = sample(); require(before['npu_hz'] == a.expected_npu_hz, 'Unexpected NPU frequency before head')
            captured_values, steps, prediction = {}, [], None
            begin = time.perf_counter()
            with torch.inference_mode():
                require(np.isfinite(raw).all(), 'Original camera tokens must be finite')
                for i in range(4):
                    conditioning = seed if prediction is None else prediction
                    helper.pose_array(conditioning)
                    delta, timing = step.execute_step(base, api, context, raw, conditioning,
                        graph['bindings'], 30000, r['cleanup_errors'], r)
                    prediction = helper.accumulate_delta(prediction, delta)
                    helper.pose_array(prediction)
                    activation = activate(prediction)
                    captured_values.update({f'conditioning_{i}': conditioning, f'delta_{i}': delta,
                        f'preactivation_{i}': prediction, f'activation_{i}': activation})
                    steps.append(dict(iteration=i, **timing))
            seconds = time.perf_counter() - begin
            after = sample(); require(after['npu_hz'] == a.expected_npu_hz, 'Unexpected NPU frequency after head')
            captured_values['final_pose'] = captured_values['activation_3']
            first_for_route = route not in first
            if first_for_route: first[route] = captured_values
            artifact_path = out / f'head-{index:02d}-{route}.npz'
            with artifact_path.open('xb') as stream: np.savez(stream, **captured_values)
            record = dict(index=index, route=route, first_for_route=first_for_route,
                complete_head_seconds=seconds, sdk_io_seconds=sum(x['io_total_seconds'] for x in steps),
                steps=steps, platform_before=before, platform_after=after,
                vs_original_CPU={key: compare(value, expected_final if key == 'final_pose' else references[key]) for key, value in captured_values.items()},
                vs_first_same_route={key: compare(value, first[route][key]) for key, value in captured_values.items()},
                artifact=dict(file=artifact_path.name, bytes=artifact_path.stat().st_size, sha256=sha(artifact_path),
                    tensors={key: info(value) for key, value in captured_values.items()}))
            if route in seen:
                record['bitwise_equal_previous_same_route'] = all(captured_values[k].tobytes() == seen[route][k].tobytes() for k in captured_values)
            if route == 'B':
                record['vs_first_A'] = {key: compare(value, first['A'][key]) for key, value in captured_values.items()}
            seen[route] = captured_values
            r['calls'].append(record)
        r['first_B_vs_first_A'] = {key: compare(value, first['A'][key]) for key, value in first['B'].items()}
        r['summary_seconds'] = {route: dict(calls=7, warm_count=6,
            first=next(c['complete_head_seconds'] for c in r['calls'] if c['route'] == route),
            warm_mean=float(np.mean([c['complete_head_seconds'] for c in r['calls'] if c['route'] == route and not c['first_for_route']])),
            warm_sdk_io_mean=float(np.mean([c['sdk_io_seconds'] for c in r['calls'] if c['route'] == route and not c['first_for_route']]))) for route in 'AB'}
        require(len(r['calls']) == 14 and all(sum(c['route'] == route for c in r['calls']) == 7 for route in 'AB'), 'Original AB + ABBA*3 schedule incomplete')
        r['status'] = 'completed_original_full_CameraHead_component_AB_not_quality_accepted'
    except BaseException as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for context in reversed(contexts):
            if context.value:
                try:
                    base.check_call(api.rknn_destroy(context), 'rknn_destroy')
                    context.value = 0; r['contexts_destroyed'] += 1
                except BaseException as exc:
                    r['cleanup_errors'].append('Context cleanup: ' + str(exc))
        r['live_contexts'] = sum(bool(context.value) for context in contexts)
        if r['status'].startswith('completed'):
            r['lifecycle_verified'] = (r['contexts_created'] == r['contexts_destroyed'] == 2
                and r['outputs_acquired'] == r['outputs_released'] == 56
                and r['sdk_run_attempts'] == r['sdk_runs_succeeded'] == r['npu_run_api_call_count'] == 56)
            if not r['lifecycle_verified']: r['cleanup_errors'].append('2 contexts / 56 steps / 56 outputs invariant failed')
        try:
            r['source_and_inputs_unchanged'] = bool(pins) and all(sha(path) == digest for path, digest in pins.items())
        except BaseException as exc:
            r['source_and_inputs_unchanged'] = False; r['cleanup_errors'].append('Identity cleanup: ' + str(exc))
        try:
            os.sched_setaffinity(0, affinity); r['affinity_restored'] = os.sched_getaffinity(0) == affinity
            for key, value in old_env.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
            r['environment_restored'] = all(os.environ.get(key) == value for key, value in old_env.items())
        except BaseException as exc:
            r['affinity_restored'] = r['environment_restored'] = False
            r['cleanup_errors'].append('Process environment/affinity cleanup: ' + str(exc))
        if r['cleanup_errors'] or r['live_contexts'] or not r.get('source_and_inputs_unchanged') or not r.get('affinity_restored') or not r.get('environment_restored'):
            r['status'] = 'failed_cleanup_or_identity'
        temp = out / 'result.json.tmp'
        temp.write_text(json.dumps(r, indent=2, allow_nan=False) + '\n')
        temp.replace(out / 'result.json')
        print(json.dumps(dict(status=r['status'], error=r.get('error'), receipt_sha256=sha(out / 'result.json'))), flush=True)
    return 0 if r['status'].startswith('completed') else 1


if __name__ == '__main__':
    sys.exit(main())
