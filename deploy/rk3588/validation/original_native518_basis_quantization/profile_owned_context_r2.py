#!/usr/bin/env python3
"""Private native518 original camera/DPT consumers and reusable owned SDK IO.

No model is constructed. Original camera four-step FP32 recurrence/activation
and original DPT exp/expp1 remain on CPU. SDK execution is RK3588-board-only.
Constructors pin existing original artifacts; callers own external clocks,
thread settings, process deadlines, image preparation and quality assessment.
"""
from __future__ import annotations
import copy
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import threading
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
EXPERIMENT = BASE / 'experiments'
HELPER_SHA = '56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
SDK_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
CAMERA_SOURCE_SHA = 'e20518d08c47746e1bf58e07bd6d0270bc7c7f6daa4d0858593f478ce6dfb09e'
MATH_SHA = 'b26efe9c6cd62a2b7977e6e9c096cf0bd9bbf63076a063b9a332ef59565af258'
HEAD_ACT_SHA = '428bdb087f457b809a66e22d7d01147c1cb4dab43bb0d98bc61c7f05d8383a1e'
DEPTH_EXPORT_SHA = '8e3f465242924445348a589d5b589e4176fabde9e272ad0fe7d49950886f6033'
DEPTH_COMPILE_SHA = '838b608fb73e96d0f2faec68c64a4bb5c1222e0ecfc406f68151171b314ba020'
CAMERA_MODEL_SHA = '0c216630b608ddd72ae288385c886d20d5e002b85fad3f741ed1308360bd7f51'
DEPTH_MODEL_SHA = 'e7219a4e612fcff26f0a76ea7a24536188a0567cbd3941ba58774a32d42fb8c6'
EMPTY_RAW_SHA = 'a4f798a79001dce1fb81a8081beb6cefa8c37ecb8e19379f582367a40e41da83'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
LAYERS = (4, 11, 17, 23)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def pin(path, digest, size=None):
    path = Path(path).resolve(strict=True)
    require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Required SHA256 malformed')
    require((size is None or path.stat().st_size == size) and sha(path) == digest,
            'Pinned original artifact/source differs: ' + str(path))
    return path


def module(path, digest, name):
    spec = importlib.util.spec_from_file_location(name, pin(path, digest))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def original_native_head_rows(source_root=SOURCE):
    """Pure source/receipt/artifact verification; no NumPy/Torch/SDK execution.

    Defaults select the measured original FP16 camera split4 and native DPT.
    Precision/trajectory acceptance is deliberately not inferred from pins.
    """
    root = Path(source_root).resolve(strict=True); validation = root / 'deploy/rk3588/validation'
    camera = module(root / 'vggt_slam/rknn_camera_head.py', CAMERA_SOURCE_SHA, '_v100_original_camera_lineage')
    captured = json.loads(pin(validation / 'vggt-v15-camera-capture112.json', camera.CAPTURE_SHA).read_text())
    exported = json.loads(pin(validation / 'vggt-v16-camera-step-export112-r1.json', camera.EXPORT_SHA).read_text())
    weights = captured['artifacts']['camera-state.npz']['tensors']
    require(captured['status'] == 'completed' and
            {k: captured['checkpoint'][k] for k in CHECKPOINT} == CHECKPOINT and
            captured['camera_iterations'] == 4 and len(weights) == 69 and
            exported['status'] == 'exported_and_structurally_checked' and
            exported['capture']['sha256'] == camera.CAPTURE_SHA and
            exported['weight_sharing']['original_parameters_verified'] == 68 and
            exported['outer_iteration_contract']['iterations'] == 4 and
            exported['outer_iteration_contract']['all_four_original_iterations_array_equal'] is True and
            exported['outer_iteration_contract']['empty_weight'] == weights['empty_pose_tokens'],
            'Original 69-parameter/four-step camera provenance differs')
    model = camera._split4_artifact(root, weights)
    pin(model, CAMERA_MODEL_SHA, camera.SPLIT_MODEL_BYTES)
    empty = dict(exported['artifacts']['empty-pose-token.npy'])
    require(empty['tensor_sha256'] == EMPTY_RAW_SHA and empty['shape'] == [1, 1, 9], 'Original learned camera seed differs')
    empty['file'] = str(pin(EXPERIMENT / 'vggt-v16-camera-step/export112-r1/empty-pose-token.npy', empty['sha256'], empty['bytes']))
    camera_row = dict(file=str(model), bytes=camera.SPLIT_MODEL_BYTES, sha256=CAMERA_MODEL_SHA,
        input_names=['camera_tokens', 'conditioning_pose'], input_shapes=[[1, 2, 2048], [1, 2, 9]],
        output_names=['pose_delta'], output_shapes=[[1, 2, 9]], seed=empty,
        compile_receipt_sha256=camera.SPLIT_COMPILE_SHA, check_receipt_sha256=camera.SPLIT_CHECK_SHA,
        original_parameter_count=69, precision='original_v69_split4_FP16')
    ep = pin(EXPERIMENT / 'vggt-v99-depth-export-r1/export.json', DEPTH_EXPORT_SHA)
    cp = pin(EXPERIMENT / 'vggt-v99-depth-compile-r1/result.json', DEPTH_COMPILE_SHA)
    e, c = json.loads(ep.read_text()), json.loads(cp.read_text())
    flags = ('all_62_original_parameter_values_preserved_in_graph', 'all_five_original_position_constants_preserved_in_graph',
        'original_inplace_relu_residual_skip_preserved', 'all_five_original_align_corners_resizes_preserved',
        'odd_width_refinenet4_explicit_original_layer3_target_preserved', 'pinned_source_and_inputs_unchanged')
    require(e['schema'] == 'v99_original_native518_depth_logits_static_export' and
        e['status'] == 'static_export_only_not_runtime_validated' and e['checkpoint'] == CHECKPOINT and
        all(e[k] is True for k in flags) and e['cleanup_errors'] == [] and
        len(e['original_parameter_metadata']) == 62 and len(e['native_position_metadata']) == 5 and
        e['architecture']['per_frame_input_shape'] == [1, 1036, 2048] and
        e['architecture']['per_frame_raw_logits_shape'] == [1, 2, 392, 518] and
        e['architecture']['activation'] == 'exp' and e['architecture']['conf_activation'] == 'expp1',
        'Original complete native DPT export/geometry differs')
    require(c['schema'] == 'original_native518_complete_depth_FP16_compile' and c['status'] == 'compiled_not_quality_accepted' and
        c['onnx']['sha256'] == e['graph']['sha256'] and c['original_ONNX_modified'] is False and
        c['configuration'] == dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
            mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
            remove_reshape=False, quantize_weight=False, enable_flash_attention=False) and
        c['build_options'] == dict(do_quantization=False, dataset=None, auto_hybrid=False) and
        [x['operation'] for x in c['calls']] == ['config', 'load_onnx', 'build', 'export_rknn'] and
        all(x['status'] == 'completed' and x['returncode'] in (None, 0) for x in c['calls']) and
        c['sdk_released'] is True and c['sdk_release_code'] in (None, 0) and c['cleanup_errors'] == [] and
        all(c[k] is True for k in ('affinity_restored', 'environment_restored', 'cwd_restored', 'pins_unchanged')),
        'Original native DPT compilation did not complete cleanly')
    row = c['rknn']; require(row['sha256'] == DEPTH_MODEL_SHA and row['bytes'] == 133505463, 'Native DPT payload identity differs')
    depth_row = dict(row, file=str(pin(cp.parent / row['file'], DEPTH_MODEL_SHA, row['bytes'])),
        input_names=[f'tokens_{i}' for i in LAYERS], input_shapes=[[1, 1036, 2048]] * 4,
        output_names=['logits'], output_shapes=[[1, 2, 392, 518]],
        compile_receipt_sha256=DEPTH_COMPILE_SHA, export_receipt_sha256=DEPTH_EXPORT_SHA,
        original_parameter_count=62, precision='original_native518_FP16',
        raw_FP32_graph_gate_passed=False, activated_FP32_consumer_gate_passed=True)
    pin(root / 'third_party/vggt/vggt/heads/head_act.py', HEAD_ACT_SHA)
    return camera_row, depth_row


class OwnedContext:
    """One owned flags0 floating graph; .run(inputs)->owned FP32 list.

    SDK descriptors map declared names to physical indices. All output owners
    remain reachable on partial get/release failure; failed release forbids
    context destruction until a later successful retry. No precision fallback.
    """
    def __init__(self, model_path, model_sha256, *, input_shapes, output_shapes,
                 input_names=None, output_names=None, core_mask=1, flags=0,
                 source_root=SOURCE, abi_source=None, library='/usr/lib/librknnrt.so',
                 runtime_helper=EXPERIMENT / 'vggt-v99-source-r1/run_native_block_r1.py', timeout_ms=30000):
        self.owner = threading.get_ident(); self.context = ct.c_uint64(0)
        self.api = self.base = None; self.pending = []; self.closed = self.failed = self.active = False
        self.input_shapes = [tuple(s) for s in input_shapes]; self.output_shapes = [tuple(s) for s in output_shapes]
        self.timeout = timeout_ms; self.last_timings = {}; self.pins = {}
        self.stats = dict(init_attempts=0, contexts_created=0, destroy_attempts=0, contexts_destroyed=0,
            input_set_attempts=0, input_set_successes=0, run_attempts=0, runs_completed=0,
            output_get_attempts=0, output_get_successes=0, output_buffers_acquired=0,
            output_release_attempts=0, output_buffers_released=0, cleanup_errors=[],
            core_mask=core_mask, init_flags=flags, model_sha256=model_sha256)
        try:
            require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
                    'Actual SDK execution is restricted to RK3588/S')
            require(core_mask in (1, 2, 4, 7) and flags in (0, 8) and 0 < timeout_ms <= 600000,
                    'Ordinary flags0 and explicit bounded supported core mask required')
            H = module(runtime_helper, HELPER_SHA, '_v100_heads_frozen_IO_helpers')
            abi = H.pin(abi_source or Path(source_root) / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA, self.pins)
            self.base = module(abi, ABI_SHA, '_v100_heads_frozen_typed_ABI')
            model = H.pin(model_path, model_sha256, self.pins); lib = H.pin(library, SDK_SHA, self.pins)
            self.api = self.base.bind_api(ct.CDLL(str(lib)))
            import numpy as np
            self.np = np; self.filename_owner = ct.create_string_buffer(os.fsencode(model))
            self.stats['init_attempts'] += 1
            try: code = self.api.rknn_init(ct.byref(self.context), ct.cast(self.filename_owner, ct.c_void_p), 0, flags, None)
            finally: self.stats['contexts_created'] += int(bool(self.context.value))
            self.base.check_call(code, 'rknn_init'); require(self.context.value, 'Empty SDK context after successful init')
            self.base.check_call(self.api.rknn_set_core_mask(self.context, core_mask), 'rknn_set_core_mask; no fallback')
            version = self.base.query(self.api, self.context, 5, self.base.SDKVersion())
            self.stats['api_version'] = bytes(version.api_version).decode(errors='replace')
            require(self.stats['api_version'].startswith('2.3.2'), 'Frozen typed ABI requires runtime2.3.2')
            number = self.base.query(self.api, self.context, 0, self.base.InputOutputNum())
            require((number.n_input, number.n_output) == (len(self.input_shapes), len(self.output_shapes)), 'Graph IO counts differ')
            self.inputs = self._descriptors(1, self.input_shapes, input_names)
            self.outputs = self._descriptors(2, self.output_shapes, output_names)
            self.stats.update(input_descriptors=self.inputs, output_descriptors=self.outputs)
            mem = self.base.query(self.api, self.context, 6, self.base.MemSize())
            self.stats['SDK_memory'] = dict(weight_bytes=int(mem.total_weight_size), internal_bytes=int(mem.total_internal_size),
                dma_bytes=int(mem.total_dma_allocated_size), scope='SDK per-context fields, not physical unique-byte totals')
        except BaseException as exc:
            self.failed = True
            try: self.close()
            except BaseException as err: exc.add_note('Partial owned-context cleanup: ' + str(err))
            exc.owned_context = self
            raise

    def _descriptors(self, command, shapes, names):
        b = self.base
        rows = [b.attr_record(b.query(self.api, self.context, command, b.TensorAttr(index=i))) for i in range(len(shapes))]
        require([r['index'] for r in rows] == list(range(len(shapes))), 'SDK logical descriptor indices differ')
        if names is not None:
            require(len(names) == len(rows) and len(set(names)) == len(names) and {r['name'] for r in rows} == set(names), 'Logical IO names differ')
            rows = [next(r for r in rows if r['name'] == name) for name in names]
        for row, shape in zip(rows, shapes):
            identity = row['type'] == 1 and row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
            require(row['shape'] == list(shape) and row['n_elems'] == math.prod(shape) and row['type'] in (0, 1) and
                (row['quantization_type'] == 0 or identity) and row['format'] in (0, 3) and row['pass_through'] == 0 and
                row['size'] == math.prod(shape) * (4 if row['type'] == 0 else 2) and math.isfinite(row['scale']),
                'Unsupported logical floating descriptor: ' + str(row))
        return rows

    def _release(self, entry):
        outputs, acquired = entry; self.stats['output_release_attempts'] += 1
        code = self.api.rknn_outputs_release(self.context, len(outputs), outputs)
        if code == 0:
            self.stats['output_buffers_released'] += acquired
            self.pending[:] = [p for p in self.pending if p is not entry]
        else: self.stats['cleanup_errors'].append('outputs_release returned ' + str(code) + '; output/context owners retained')
        self.base.check_call(code, 'rknn_outputs_release')

    def run(self, inputs):
        require(threading.get_ident() == self.owner and not self.closed and not self.failed and not self.active,
                'Owned context requires idle live original owner thread')
        self.active = True; start = time.perf_counter(); times = {}; entry = None
        b, api, ctx, np = self.base, self.api, self.context, self.np
        outputs = (b.Output * len(self.outputs))(); ys = []
        try:
            require(len(inputs) == len(self.inputs), 'Complete graph input list required')
            tick = time.perf_counter(); values = []
            for x, shape in zip(inputs, self.input_shapes):
                require(isinstance(x, np.ndarray) and x.dtype == np.float32 and x.shape == shape and np.isfinite(x).all(), 'Finite logical original FP32 input required')
                values.append(np.ascontiguousarray(x))
            ins = (b.Input * len(values))()
            for i, (x, row) in enumerate(zip(values, self.inputs)):
                ins[i] = b.Input(index=row['index'], buf=int(x.ctypes.data), size=x.nbytes, pass_through=0, type=0, fmt=row['format'])
            for out, row in zip(outputs, self.outputs): out.index = row['index']; out.want_float = 1; out.is_prealloc = 0
            times['input_prepare_and_finite_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter(); self.stats['input_set_attempts'] += 1
            b.check_call(api.rknn_inputs_set(ctx, len(values), ins), 'rknn_inputs_set')
            self.stats['input_set_successes'] += 1; times['inputs_set_seconds'] = time.perf_counter() - tick
            run = b.RunExtend(non_block=0, timeout_ms=self.timeout, fence_fd=-1); received = b.OutputExtend()
            tick = time.perf_counter(); self.stats['run_attempts'] += 1
            b.check_call(api.rknn_run(ctx, ct.byref(run)), 'rknn_run')
            self.stats['runs_completed'] += 1; times['SDK_run_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter(); self.stats['output_get_attempts'] += 1
            try: code = api.rknn_outputs_get(ctx, len(outputs), outputs, ct.byref(received))
            finally:
                acquired = sum(bool(o.buf) for o in outputs)
                if acquired:
                    entry = (outputs, acquired); self.pending.append(entry); self.stats['output_buffers_acquired'] += acquired
                times['outputs_get_seconds'] = time.perf_counter() - tick
            b.check_call(code, 'rknn_outputs_get'); self.stats['output_get_successes'] += 1
            require(run.frame_id == received.frame_id, 'Synchronous SDK frame ID differs')
            tick = time.perf_counter()
            for out, row, shape in zip(outputs, self.outputs, self.output_shapes):
                require(out.index == row['index'] and out.buf and out.size == math.prod(shape) * 4, 'Expected complete logical FP32 output size/index differs')
                view = np.ctypeslib.as_array(ct.cast(out.buf, ct.POINTER(ct.c_float)), shape=(math.prod(shape),))
                ys.append(view.copy().reshape(shape)); del view
            times['all_owned_copies_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter()
            if entry is not None: self._release(entry); entry = None
            times['outputs_release_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter()
            owned_outputs_finite = all(np.isfinite(y).all() and y.flags.c_contiguous for y in ys)
            if not owned_outputs_finite: self.last_failed_outputs = ys
            require(owned_outputs_finite, 'Nonfinite owned SDK output')
            times['output_finite_seconds'] = time.perf_counter() - tick
            return ys
        except BaseException:
            self.failed = True
            raise
        finally:
            if entry is not None:
                try: self._release(entry)
                except BaseException:
                    self.failed = True
                    raise
                finally:
                    times['complete_boundary_seconds'] = time.perf_counter() - start
                    self.last_timings = times; self.active = False
            else:
                times['complete_boundary_seconds'] = time.perf_counter() - start
                self.last_timings = times; self.active = False

    def close(self):
        require(threading.get_ident() == self.owner and not self.active, 'Close requires idle original context owner')
        if self.closed: return
        errors = []
        for entry in list(self.pending):
            try: self._release(entry)
            except BaseException as exc: errors.append(str(exc))
        if self.context.value and not self.pending and self.api is not None:
            self.stats['destroy_attempts'] += 1
            try:
                self.base.check_call(self.api.rknn_destroy(self.context), 'rknn_destroy')
                self.context.value = 0; self.stats['contexts_destroyed'] += 1
            except BaseException as exc: errors.append(str(exc)); self.stats['cleanup_errors'].append(str(exc))
        self.closed = not self.context.value and not self.pending
        if not self.closed or errors: self.failed = True
        require(self.closed and not errors, 'Owned context cleanup incomplete: ' + '; '.join(errors))

    def metrics(self):
        s = dict(self.stats, closed=self.closed, failed=self.failed, live_contexts=int(bool(self.context.value)),
            live_output_buffers=sum(p[1] for p in self.pending), resource_counts_balanced=(self.stats['contexts_created'] == self.stats['contexts_destroyed'] and
                self.stats['output_buffers_acquired'] == self.stats['output_buffers_released']))
        return copy.deepcopy(s)


class NativeHeads:
    """.run(retained_outputs,capture=False) -> original batched head arrays.

    Four actual RKNN camera deltas refine the same unactivated FP32 pose.
    Two actual native DPT calls each use all four actual retained layers.
    .last_capture optionally retains owned deltas/logits, never teacher inputs.
    Pose encoding→extrinsics/intrinsics remains the original caller's codec.
    """
    def __init__(self, camera_row=None, depth_row=None, *, source_root=SOURCE, core_mask=1,
                 abi_source=None, library='/usr/lib/librknnrt.so',
                 runtime_helper=EXPERIMENT / 'vggt-v99-source-r1/run_native_block_r1.py', timeout_ms=30000):
        self.owner = threading.get_ident(); self.contexts = []; self.closed = self.failed = self.active = False
        self.last_timings = {}; self.last_capture = None; self.completed_calls = self.completed_refinements = self.completed_depth_frames = 0
        try:
            require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'Native head execution is board-only')
            require((camera_row is None) == (depth_row is None), 'Supply both verified head rows or use both defaults')
            if camera_row is None: camera_row, depth_row = original_native_head_rows(source_root)
            self.rows = (copy.deepcopy(camera_row), copy.deepcopy(depth_row))
            require(camera_row['sha256'] == CAMERA_MODEL_SHA and camera_row['bytes'] == 434656165 and
                depth_row['sha256'] == DEPTH_MODEL_SHA and depth_row['bytes'] == 133505463, 'Only pinned original head graph payloads are supported')
            import numpy as np
            import torch
            self.np, self.torch = np, torch
            root = Path(source_root)
            self.math = module(root / 'deploy/rk3588/validation/export-v16-camera-step-board.py', MATH_SHA, '_v100_original_camera_FP32_recurrence')
            self.activation = module(root / 'third_party/vggt/vggt/heads/head_act.py', HEAD_ACT_SHA, '_v100_original_head_activation')
            empty = camera_row['seed']; value = np.load(pin(empty['file'], empty['sha256'], empty['bytes']), allow_pickle=False)
            require(value.dtype == np.float32 and value.shape == (1, 1, 9) and value.flags.c_contiguous and
                np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == EMPTY_RAW_SHA, 'Original learned empty camera token differs')
            self.seed = np.broadcast_to(value, (1, 2, 9)).copy()
            for row in (camera_row, depth_row):
                try:
                    graph = OwnedContext(row['file'], row['sha256'], input_shapes=row['input_shapes'], output_shapes=row['output_shapes'],
                        input_names=row['input_names'], output_names=row['output_names'], core_mask=core_mask, source_root=root,
                        abi_source=abi_source, library=library, runtime_helper=runtime_helper, timeout_ms=timeout_ms)
                except BaseException as exc:
                    if hasattr(exc, 'owned_context'): self.contexts.append(exc.owned_context)
                    raise
                self.contexts.append(graph)
            self.camera, self.depth = self.contexts
        except BaseException as exc:
            self.failed = True
            try: self.close()
            except BaseException as err: exc.add_note('Partial native heads cleanup: ' + str(err))
            exc.native_heads = self
            raise

    def run(self, retained_outputs, *, capture=False):
        require(threading.get_ident() == self.owner and not self.closed and not self.failed and not self.active, 'Heads require idle live original owner')
        self.active = True; start = time.perf_counter(); self.last_timings = dict(camera_steps=[], depth_frames=[]); self.last_capture = None
        try:
            np, torch = self.np, self.torch
            require(set(retained_outputs) == set(LAYERS), 'All four original retained layers required')
            for value in retained_outputs.values():
                require(isinstance(value, np.ndarray) and value.shape == (1, 2, 1041, 2048) and value.dtype == np.float32 and
                    value.flags.c_contiguous and np.isfinite(value).all(), 'Actual native retained FP32 shape/finite differs')
            with torch.inference_mode():
                camera_start = time.perf_counter(); raw = np.ascontiguousarray(retained_outputs[23][:, :, 0]); prediction = None; poses = []; deltas = []
                for index in range(4):
                    step_start = time.perf_counter(); delta = self.camera.run([raw, self.seed if prediction is None else prediction])[0]
                    prediction = self.math.accumulate_delta(prediction, delta)
                    pose = self.activation.activate_pose(torch.from_numpy(prediction), trans_act='linear', quat_act='linear', fl_act='relu')
                    owned_pose = pose.contiguous().numpy().copy(); require(np.isfinite(owned_pose).all(), 'Nonfinite original activated camera pose')
                    poses.append(owned_pose); deltas.append(delta); self.completed_refinements += 1
                    self.last_timings['camera_steps'].append(dict(self.camera.last_timings, iteration=index, full_step_seconds=time.perf_counter() - step_start))
                self.last_timings['camera_complete_seconds'] = time.perf_counter() - camera_start
                depth_start = time.perf_counter(); depths = []; confidences = []; raw_logits = []
                for frame in range(2):
                    frame_start = time.perf_counter()
                    inputs = [np.ascontiguousarray(retained_outputs[layer][:, frame, 5:]) for layer in LAYERS]
                    raw_depth = self.depth.run(inputs)[0]
                    depth, conf = self.activation.activate_head(torch.from_numpy(raw_depth), activation='exp', conf_activation='expp1')
                    depths.append(depth.view(1, 1, *depth.shape[1:])); confidences.append(conf.view(1, 1, *conf.shape[1:])); raw_logits.append(raw_depth)
                    self.completed_depth_frames += 1
                    self.last_timings['depth_frames'].append(dict(self.depth.last_timings, frame_index=frame, full_frame_seconds=time.perf_counter() - frame_start))
                depth = torch.cat(depths, dim=1).contiguous().numpy().copy(); confidence = torch.cat(confidences, dim=1).contiguous().numpy().copy()
                require(depth.shape == (1, 2, 392, 518, 1) and confidence.shape == (1, 2, 392, 518) and
                    np.isfinite(depth).all() and np.isfinite(confidence).all(), 'Complete finite original native depth/conf outputs required')
                self.last_timings['depth_complete_seconds'] = time.perf_counter() - depth_start
            if capture: self.last_capture = dict(camera_tokens=raw, camera_deltas=deltas, depth_logits=raw_logits)
            self.completed_calls += 1
            return dict(pose_enc=poses[-1], pose_enc_list=poses, depth=depth, depth_conf=confidence)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.last_timings['complete_heads_seconds'] = time.perf_counter() - start; self.active = False

    def close(self):
        require(threading.get_ident() == self.owner and not self.active, 'Heads close requires idle original owner')
        if self.closed: return
        errors = []
        for context in reversed(self.contexts):
            try: context.close()
            except BaseException as exc: errors.append(str(exc))
        self.closed = all(c.closed for c in self.contexts)
        if not self.closed or errors: self.failed = True
        require(self.closed and not errors, 'Native heads cleanup incomplete: ' + '; '.join(errors))

    def metrics(self):
        rows = [c.metrics() for c in self.contexts]
        return copy.deepcopy(dict(closed=self.closed, failed=self.failed, contexts=rows, completed_head_calls=self.completed_calls,
            completed_refinements=self.completed_refinements, completed_depth_frames=self.completed_depth_frames,
            live_contexts=sum(r['live_contexts'] for r in rows), live_output_buffers=sum(r['live_output_buffers'] for r in rows),
            resources_balanced=all(r['resource_counts_balanced'] for r in rows), cleanup_errors=[e for r in rows for e in r['cleanup_errors']],
            original_layer_indices=list(LAYERS), original_camera_iterations=4, frame_chunks=2, patch_start_idx=5,
            original_parameters_verified=dict(camera=69, depth=62), original_math_sources=dict(camera_recurrence=MATH_SHA, activation=HEAD_ACT_SHA),
            precision='existing original FP16 graphs with logical FP32 IO/consumer arithmetic',
            precision_quality_or_realtime_accepted=False, full_model_constructed=False, system_or_frequency_writes=False))
