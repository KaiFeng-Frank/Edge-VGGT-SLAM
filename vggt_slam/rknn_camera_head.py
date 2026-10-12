"""Reuse the original v16 CameraHead step graph with its four-iteration IO.

Install on the unchanged original eval CPU FP32 model. The original last
aggregator tensor supplies its camera token; no captured token is substituted.
Four synchronous calls share one existing RKNN step context. The original
learned empty pose, FP32 delta accumulation and activate_pose remain outside
the graph, and forward still returns all four activated pose tensors.
"""
from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import threading
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
MODEL_SHA = '0821683fe1f9526c52777b0ef1ba45ba4c744375f9fd731f0f3d2068e4355af9'
MODEL_BYTES = 434452453
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
STEP_SHA = 'dbe4fc933891581099084a569e2e847e15c582082680bf31bcf5dacc86210109'
CAPTURE_HELPER_SHA = 'd50c217354b2f48f50626ff1ea2c98671d5611743ae0e97bed1c04b2991341a0'
CAPTURE_SHA = '1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1'
EXPORT_SHA = '8ccf7ff593346962766b4eae6da7eb95ce43eeaa3183a8fc0e82bf134c172503'
COMPILE_SHA = 'c8771d743d9840bcd32b22c6894f791ed239b4ad3795db703e7f7122329fbc7a'
HEAD_ACT_SHA = '428bdb087f457b809a66e22d7d01147c1cb4dab43bb0d98bc61c7f05d8383a1e'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
SPLIT_COMPILE_SHA = 'a2cdad4a6d2af60c6b3244dfefb3d1a1fadead72c6b2374ef2c089912d318f74'
SPLIT_CHECK_SHA = '9c58744a713a8ad686e66e811163c260e82f6522f0a936dbcc92cb5fdfdc4c2a'
SPLIT_MODEL_SHA = '0c216630b608ddd72ae288385c886d20d5e002b85fad3f741ed1308360bd7f51'
SPLIT_MODEL_BYTES = 434656165
SPLIT_GRAPH_SHA = 'f1b9eee30144ec643aa351db724c2f740a79d82a8f9df5468f4338e377600a7f'
ORIGINAL_GRAPH_SHA = 'f89fb0e66f43a3d3b94a76ec0124d4f521fa1ce9a6c93efab5e362d3a614eb9f'
SPLIT_BUILDER_SHA = '9af02ea5042285ef8846b3a656162bf8fb35df1b0487eb0f2812e879313e2aee'
SPLIT_HELPER_SHA = 'cb107c093cee8359724095a5298d0cdf2407499503a1fa0358da49d2055c2894'
SPLIT_PROBE_SHA = 'e7279111524148e8fc681cc11d873bc8ba23d85d1f027a900c57b250e8828dbc'


def _split4_artifact(source_root, weights):
    """Select only the already measured original split4 graph; no acceptance claim."""
    directory = BASE / 'experiments/vggt-v69-camera-step-split4-r1'
    compile_path = pin(directory / 'result.json', SPLIT_COMPILE_SHA)
    check_path = pin(BASE / 'experiments/vggt-v69-head-ort-r1/result.json', SPLIT_CHECK_SHA)
    compiled, checked = json.loads(compile_path.read_text()), json.loads(check_path.read_text())
    config = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
                  mean_values=None, std_values=None, quant_img_RGB2BGR=False,
                  model_pruning=False, remove_reshape=False, quantize_weight=False,
                  enable_flash_attention=False)
    if (compiled.get('schema') != 'v69_original_CameraHead_step_FC1_split_compile'
            or compiled.get('status') != 'compiled_not_quality_accepted'
            or [compiled.get(k) for k in ('chunks', 'trunk_blocks_per_step', 'required_outer_iterations')] != [4, 4, 4]
            or compiled.get('baseline_onnx_sha256') != ORIGINAL_GRAPH_SHA
            or compiled.get('script_sha256') != SPLIT_BUILDER_SHA or compiled.get('helper_sha256') != SPLIT_HELPER_SHA
            or compiled.get('configuration') != config
            or compiled.get('build_options') != dict(do_quantization=False, dataset=None, auto_hybrid=False)
            or compiled.get('released') is not True or compiled.get('release_code') not in (None, 0)
            or compiled.get('cleanup_errors') != [] or compiled.get('source_and_inputs_unchanged') is not True
            or compiled.get('environment_restored') is not True or compiled.get('candidate_accepted') is not False):
        raise ValueError('Frozen original split4 compilation provenance differs')
    operations = compiled.get('operations', [])
    if ([x.get('name') for x in operations] != ['config', 'load_onnx', 'build', 'export_rknn']
            or any(x.get('status') != 'completed' or x.get('code') not in (None, 0) for x in operations)):
        raise ValueError('Original split4 compilation did not complete cleanly')
    graph = compiled.get('onnx', {})
    artifact = compiled.get('rknn', {})
    if (graph != dict(file='camera-step-split4.onnx', bytes=864771004, sha256=SPLIT_GRAPH_SHA)
            or artifact != dict(file='camera-step-split4.rknn', bytes=SPLIT_MODEL_BYTES, sha256=SPLIT_MODEL_SHA)):
        raise ValueError('Frozen split4 artifact identity differs')
    proof = compiled.get('graph_proof', {})
    flags = ('all_other_original_nodes_protobuf_unchanged', 'all_other_original_initializers_protobuf_unchanged',
             'all_original_FC1_parameter_values_preserved', 'FC1_weights_not_duplicated',
             'original_GELU_nodes_protobuf_unchanged', 'original_IO_protobuf_unchanged')
    replacements = proof.get('replacements', [])
    if (any(proof.get(k) is not True for k in flags) or len(replacements) != 4
            or [r.get('index') for r in replacements] != list(range(4))):
        raise ValueError('Original four-block FC1 preservation proof differs')
    for index, row in enumerate(replacements):
        weight, bias = f'trunk.{index}.mlp.fc1.weight', f'trunk.{index}.mlp.fc1.bias'
        if (row.get('chunks') != 4 or row.get('original_weight') != 'camera_head.' + weight
                or row.get('original_bias') != 'camera_head.' + bias
                or row.get('weight_raw_sha256') != weights[weight]['sha256']
                or row.get('bias_raw_sha256') != weights[bias]['sha256']
                or row.get('output_channel_ranges') != [[i * 2048, (i + 1) * 2048] for i in range(4)]):
            raise ValueError('Original FC1 parameter or channel identity differs')
    checks = checked.get('onnx_checks', [])
    graphs = [dict(route='A', onnx_sha256=ORIGINAL_GRAPH_SHA, rknn_sha256=MODEL_SHA),
              dict(route='B', onnx_sha256=SPLIT_GRAPH_SHA, rknn_sha256=SPLIT_MODEL_SHA)]
    if (checked.get('schema') != 'v69_original_CameraHead_step_split_probe'
            or checked.get('mode') != 'check_onnx' or checked.get('status') != 'passed_original_FP32_fixed_conditioning_ONNX_gate'
            or checked.get('script_sha256') != SPLIT_PROBE_SHA or checked.get('candidate_receipt_sha256') != SPLIT_COMPILE_SHA
            or checked.get('capture_sha256') != CAPTURE_SHA or checked.get('export_sha256') != EXPORT_SHA
            or checked.get('graphs') != graphs or len(checks) != 8
            or [(r.get('route'), r.get('iteration')) for r in checks] != [(route, i) for route in 'AB' for i in range(4)]
            or any(r.get('comparison', {}).get('strict_pass') is not True for r in checks)
            or checked.get('contexts_created') != 0 or checked.get('npu_run_api_called') is not False
            or checked.get('cleanup_errors') != [] or checked.get('source_and_inputs_unchanged') is not True
            or checked.get('affinity_restored') is not True):
        raise ValueError('Passed same-graph original FP32 fixed-conditioning check required')
    validation = source_root / 'deploy/rk3588/validation/original_camera_partition'
    for name, digest in [('compile_camera_step_split_board.py', SPLIT_BUILDER_SHA),
                         ('compile_camera_fc1_splits_board.py', SPLIT_HELPER_SHA),
                         ('probe_camera_step_split_board.py', SPLIT_PROBE_SHA)]:
        pin(validation / name, digest)
    return directory / artifact['file']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(path, expected):
    path = Path(path).resolve(strict=True)
    if sha(path) != expected:
        raise ValueError('Existing original CameraHead artifact changed: ' + str(path))
    return path


def load_source(path, expected, name):
    spec = importlib.util.spec_from_file_location(name, pin(path, expected))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RKNNOriginalCameraHead:
    def __init__(self, model, *, source_root=BASE / 'src',
                 model_path=None, split_fc1=False,
                 library_path='/usr/lib/librknnrt.so', core_mask=7, timeout_ms=30000):
        self.lock = threading.RLock()
        self.closed = self.failed = False
        self.base = self.api = self.head = self.patch = None
        self.context = None
        self.stats = dict(npu_run_api_called=False, npu_run_api_call_count=0,
                          completed_head_calls=0, completed_refinements=0,
                          sdk_io_seconds=0., head_forward_seconds=0., cleanup_errors=[],
                          contexts_created=0, contexts_destroyed=0, last_steps=[],
                          candidate_accepted=False, full_model_quality_validated=False,
                          real_time_validated=False, all_operators_on_npu_verified=False)
        try:
            if type(split_fc1) is not bool:
                raise ValueError('split_fc1 must be bool')
            if type(core_mask) is not int or core_mask not in (1, 7) or type(timeout_ms) is not int or not 0 < timeout_ms <= 600000:
                raise ValueError('core_mask 1/7 and timeout_ms 1..600000 required')
            source_root = Path(source_root).resolve(strict=True)
            validation = source_root / 'deploy/rk3588/validation'
            self.base = load_source(validation / 'run-v12-rknn-block-board.py', ABI_SHA, '_v53_camera_ABI')
            self.stats['board'] = self.base.require_board()
            import numpy as np
            import torch
            self.np, self.torch, self.timeout = np, torch, timeout_ms
            self.step = load_source(validation / 'run-v16-camera-step-board.py', STEP_SHA, '_v53_camera_step')
            capture_helper = load_source(validation / 'capture-v15-camera-board.py', CAPTURE_HELPER_SHA, '_v53_camera_capture')
            capture = json.loads(pin(validation / 'vggt-v15-camera-capture112.json', CAPTURE_SHA).read_text())
            exported = json.loads(pin(validation / 'vggt-v16-camera-step-export112-r1.json', EXPORT_SHA).read_text())
            compiled = json.loads(pin(validation / 'vggt-v16-camera-step-compile112-r1.json', COMPILE_SHA).read_text())
            if (capture['status'] != 'completed' or capture['camera_iterations'] != 4
                    or capture['checkpoint']['sha256'] != 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
                    or exported['status'] != 'exported_and_structurally_checked'
                    or exported['capture']['sha256'] != CAPTURE_SHA
                    or exported['weight_sharing']['original_parameters_verified'] != 68
                    or exported['outer_iteration_contract']['all_four_original_iterations_array_equal'] is not True
                    or compiled['status'] != 'compiled_not_runtime_validated'
                    or compiled['export_receipt_sha256'] != EXPORT_SHA
                    or compiled['input']['sha256'] != exported['onnx']['sha256']
                    or compiled['output']['sha256'] != MODEL_SHA or compiled['output']['size'] != MODEL_BYTES):
                raise ValueError('Existing original capture/export/compile provenance mismatch')
            head = model.camera_head
            self.head = head
            if (head is None or any(m.training for m in head.modules())
                    or getattr(head, '_vggt_rknn_camera_owner', None) is not None
                    or capture_helper.camera_contract(head) != capture['architecture']):
                raise ValueError('Install once on the unchanged original eval CameraHead')
            weights = capture['artifacts']['camera-state.npz']['tensors']
            state = dict(head.named_parameters())
            if len(weights) != 69 or set(weights) != set(state):
                raise ValueError('All 69 original head parameters, including learned empty pose, required')
            for name, tensor in state.items():
                info = weights[name]
                if tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or not tensor.is_contiguous():
                    raise ValueError('Original head weights must remain contiguous CPU FP32')
                value = tensor.detach().numpy()
                if (list(value.shape) != info['shape'] or value.nbytes != info['bytes']
                        or hashlib.sha256(memoryview(value).cast('B')).hexdigest() != info['sha256']):
                    raise ValueError('Original CameraHead parameter changed: ' + name)
            self.seed = np.broadcast_to(head.empty_pose_tokens.detach().numpy(), (1, 2, 9)).copy()
            pin(source_root / 'third_party/vggt/vggt/heads/head_act.py', HEAD_ACT_SHA)
            from vggt.heads import head_act
            pin(head_act.__file__, HEAD_ACT_SHA)
            self.activate_pose = head_act.activate_pose
            pin(source_root / 'deploy/rk3588/reference/rknn_api_v2.3.2.h', HEADER_SHA)
            expected_sha, expected_bytes = MODEL_SHA, MODEL_BYTES
            default_path = BASE / 'experiments/vggt-v16-camera-step/camera-step-fp16.rknn'
            if split_fc1:
                default_path = _split4_artifact(source_root, weights)
                expected_sha, expected_bytes = SPLIT_MODEL_SHA, SPLIT_MODEL_BYTES
                self.stats.update(camera_graph_variant='original_v69_FC1_split4',
                                  compile_receipt_sha256=SPLIT_COMPILE_SHA, check_receipt_sha256=SPLIT_CHECK_SHA,
                                  baseline_model_sha256=MODEL_SHA, original_GELU_nodes_protobuf_unchanged=True,
                                  compiled_rounding_equivalence_validated=False, compiled_graph_changed=True)
            path = pin(default_path if model_path is None else model_path, expected_sha)
            if path.stat().st_size != expected_bytes:
                raise ValueError('Existing shared-step RKNN byte size mismatch')
            self.api = self.base.bind_api(ct.CDLL(str(pin(library_path, LIBRARY_SHA))))
            self.context = self.base.CONTEXT()
            filename = ct.create_string_buffer(str(path).encode())
            code = self.api.rknn_init(ct.byref(self.context), ct.cast(filename, ct.c_void_p), 0, 0, None)
            self.stats['contexts_created'] = int(bool(self.context.value))
            self.base.check_call(code, 'rknn_init')
            if not self.context.value:
                raise RuntimeError('Empty CameraHead RKNN context')
            self.base.check_call(self.api.rknn_set_core_mask(self.context, core_mask), 'rknn_set_core_mask')
            version = self.base.query(self.api, self.context, 5, self.base.SDKVersion())
            if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()):
                raise ValueError('Frozen typed ABI requires RKNN runtime 2.3.2')
            counts = self.base.query(self.api, self.context, 0, self.base.InputOutputNum())
            if (counts.n_input, counts.n_output) != (2, 1):
                raise ValueError('Existing raw-camera/conditioning step requires two inputs and one delta')
            attrs = [self.base.query(self.api, self.context, 1, self.base.TensorAttr(index=i)) for i in range(2)]
            output = self.base.query(self.api, self.context, 2, self.base.TensorAttr(index=0))
            self.bindings = self.step.validate_bindings(self.base, attrs, output)
            memory = self.base.query(self.api, self.context, 6, self.base.MemSize())
            self.stats.update(model_sha256=expected_sha, verified_original_parameters=69,
                              sdk_dma_bytes=int(memory.total_dma_allocated_size),
                              sdk_memory=dict(weight_bytes=int(memory.total_weight_size),
                                              internal_bytes=int(memory.total_internal_size),
                                              dma_bytes=int(memory.total_dma_allocated_size)),
                              input_attrs=[self.base.attr_record(a) for a in attrs],
                              output_attr=self.base.attr_record(output), core_mask=core_mask)
            forward = self._make_forward()
            self.patch = ('forward' in head.__dict__, head.__dict__.get('forward'), head.forward, forward)
            head.forward = forward
            head._vggt_rknn_camera_owner = self
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _make_forward(self):
        def forward(aggregated_tokens_list, num_iterations=4):
            with self.lock:
                torch, np = self.torch, self.np
                if (self.closed or self.failed or self.head.training or torch.is_grad_enabled()
                        or type(num_iterations) is not int or num_iterations != 4
                        or not isinstance(aggregated_tokens_list, (list, tuple)) or not aggregated_tokens_list):
                    raise RuntimeError('Original eval CameraHead requires four fixed-pair inference iterations')
                tokens = aggregated_tokens_list[-1]
                if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 4 or tokens.shape[:2] != (1, 2)
                        or tokens.shape[2] < 1 or tokens.shape[3] != 2048 or tokens.device.type != 'cpu'
                        or tokens.dtype != torch.float32 or tokens.requires_grad or tokens.layout != torch.strided):
                    raise ValueError('Original last-layer [1,2,P,2048] CPU FP32 aggregated tokens required')
                begin = time.perf_counter()
                raw = tokens[:, :, 0].detach().contiguous().numpy()
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite original camera tokens')
                prediction, poses, steps = None, [], []
                try:
                    for index in range(4):
                        conditioning = self.seed if prediction is None else prediction
                        delta, row = self.step.execute_step(self.base, self.api, self.context, raw,
                            conditioning, self.bindings, self.timeout, self.stats['cleanup_errors'], self.stats)
                        self.stats['completed_refinements'] += 1
                        self.stats['sdk_io_seconds'] += row['io_total_seconds']
                        prediction = delta if prediction is None else prediction + delta
                        if not np.isfinite(prediction).all():
                            raise ValueError('Nonfinite original unactivated FP32 pose accumulation')
                        poses.append(self.activate_pose(torch.from_numpy(prediction),
                            trans_act=self.head.trans_act, quat_act=self.head.quat_act, fl_act=self.head.fl_act))
                        steps.append(dict(row, iteration=index))
                    self.stats['completed_head_calls'] += 1
                    self.stats['last_steps'] = steps
                    return poses
                except BaseException:
                    self.failed = True
                    raise
                finally:
                    self.stats['head_forward_seconds'] += time.perf_counter() - begin
        return forward

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.patch is not None:
                had, value, original, forward = self.patch
                if self.head.forward is forward:
                    if had:
                        self.head.forward = value
                    else:
                        del self.head.forward
                elif self.head.forward != original:
                    self.stats['cleanup_errors'].append('CameraHead forward changed after installation')
            if self.head is not None and getattr(self.head, '_vggt_rknn_camera_owner', None) is self:
                del self.head._vggt_rknn_camera_owner
            if self.context is not None and self.context.value and self.api is not None:
                try:
                    self.base.check_call(self.api.rknn_destroy(self.context), 'rknn_destroy')
                    self.stats['contexts_destroyed'] += 1
                    self.context.value = 0
                except BaseException as exc:
                    self.stats['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
            if self.stats['cleanup_errors']:
                raise RuntimeError('CameraHead cleanup failed: ' + '; '.join(self.stats['cleanup_errors']))

    def metrics(self):
        with self.lock:
            return copy.deepcopy(dict(self.stats, closed=self.closed, failed=self.failed,
                                      live_contexts=int(self.context is not None and bool(self.context.value))))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def install_rknn_camera_head(model, **options):
    return RKNNOriginalCameraHead(model, **options)
