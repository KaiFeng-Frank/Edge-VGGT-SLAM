"""Experimental PTQ of the unchanged original split4 CameraHead step graph.

The original adapter validates all 69 parameters and source/graph lineage.
Only its SDK context and logical I/O descriptors change; inherited four-step
FP32 accumulation, original activation and owned FP32 output route are retained.
"""
from __future__ import annotations
import ctypes as ct
import json
from pathlib import Path
import re

from vggt_slam import rknn_camera_head as parent

PARENT_SOURCE_SHA = 'e20518d08c47746e1bf58e07bd6d0270bc7c7f6daa4d0858593f478ce6dfb09e'
COMPILER_SOURCE_SHA = '2f935b2937b7465a140a8d1a408cdb0a7205a6a89e8d54833271cf0e51e4f374'
CAPTURE_SOURCE_SHA = '0337c6fbdb2ea3d70073f6cd08e8e6f8f230c4284b198ecdf6dfa48f0eaf00e8'
CAPTURE_SHA = 'ebb8c7902b1ae6418b57f7b1737e0ce1f12bfe44cca0a3d89f05bb091755a509'
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
TAIL_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
IO_HELPER_SHA = 'f3f106034c6621a1e3ed07bdb265107b8522103f7d186ba4bbe0d0cefda16f2f'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
INPUTS = [('camera_tokens', [1, 2, 2048]), ('conditioning_pose', [1, 2, 9])]


def require(condition, message):
    if not condition: raise ValueError(message)


def artifact(directory, row):
    directory = Path(directory).resolve(strict=True); relative = Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative compiled artifact required')
    path = (directory / relative).resolve(strict=True)
    require(path.is_relative_to(directory) and path.stat().st_size == row['bytes'], 'Compiled artifact path/size differs')
    return parent.pin(path, row['sha256'])


def load_compilation(path, digest, source_root):
    """Pure reading: bind a clean PTQ leaf to the unchanged original graph."""
    parent.pin(source_root / 'vggt_slam/rknn_camera_head.py', PARENT_SOURCE_SHA)
    require(Path(parent.__file__).resolve() == (source_root / 'vggt_slam/rknn_camera_head.py').resolve(), 'Original adapter import escaped source root')
    path = parent.pin(path, digest); c = json.loads(path.read_text())
    require(c['schema'] == 'v82_original_CameraHead_step_split4_INT8_compile'
        and c['status'] == 'compiled_not_quality_accepted' and c['script_sha256'] == COMPILER_SOURCE_SHA
        and c['graph_sha256'] == parent.SPLIT_GRAPH_SHA and c['graph']['sha256'] == parent.SPLIT_GRAPH_SHA
        and c['graph']['bytes'] == 864771004 and c['original_ONNX_modified'] is False
        and c['original_export_sha256'] == parent.EXPORT_SHA
        and c['prior_FP16_compile_receipt_sha256'] == parent.SPLIT_COMPILE_SHA
        and c['prior_FP32_check_receipt_sha256'] == parent.SPLIT_CHECK_SHA
        and c['configuration'] == CONFIG and c['compile_cpus'] == [4, 6]
        and c['build_options'] == dict(do_quantization=True, dataset=str(path.parent / 'dataset.txt'), auto_hybrid=False)
        and c['candidate_accepted'] is False and c['quality_accepted'] is False
        and c['checkpoint'] == dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
        and c['original_model_only'] is True and c['model_inference_executed'] is False and c['ORT_rerun'] is False
        and c['released'] is True and c['release_code'] in (None, 0) and c['cleanup_errors'] == []
        and all(c[k] is True for k in ('source_and_inputs_unchanged', 'environment_restored', 'affinity_restored'))
        and c['required_outer_iterations'] == c['trunk_blocks_per_step'] == 4
        and c['original_empty_and_FP32_outer_accumulation_unchanged'] is True,
        'Clean original split4 PTQ compilation required')
    require([o['name'] for o in c['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
        and all(o['status'] == 'completed' and o['code'] in (None, 0) for o in c['operations']), 'Four successful compile operations required')
    calibration = c['calibration']; cap_row = calibration['capture']
    cap_path = parent.pin(cap_row['file'], CAPTURE_SHA)
    require(cap_path.stat().st_size == cap_row['bytes'] and cap_row['sha256'] == CAPTURE_SHA
        and c['capture_sha256'] == CAPTURE_SHA and calibration['source_calibration_sha256'] == CALIBRATION_SHA
        and calibration['original_tail_capture_sha256'] == TAIL_SHA
        and calibration['pair_count'] == 8 and calibration['iterations_per_pair'] == 4,
        'Original eight-pair/32-step calibration required')
    captured = json.loads(cap_path.read_text())
    require(captured['schema'] == 'v82_original_VGGT_camera_PTQ_calibration_capture'
        and captured['status'] == 'completed_original_FP32_camera_calibration_capture'
        and captured['script_sha256'] == CAPTURE_SOURCE_SHA
        and captured['checkpoint'] == c['checkpoint'] and captured['original_model_only'] is True
        and captured['conditioning_count'] == captured['delta_count'] == 32
        and captured['tail_blocks_forward_count'] == dict(frame23=8, global23=8)
        and captured['parameters'] == c['original_parameters'] and len(c['original_parameters']) == 69
        and calibration['samples'] == captured['samples'] and len(captured['samples']) == 8
        and captured['full_head_forward_count'] == 8 and captured['all_final_poses_bitwise_equal_to_original_capture'] is True
        and captured['model_state_unchanged'] is True and captured['source_and_inputs_unchanged'] is True
        and captured['NPU_or_cache_installed'] is False and captured['cleanup_errors'] == []
        and all(captured[k] is True for k in ('hooks_removed', 'affinity_restored', 'environment_restored')),
        'Verified original FP32 camera capture required')
    require(c['capture_source']['sha256'] == CAPTURE_SOURCE_SHA, 'Frozen calibration source required')
    parent.pin(c['capture_source']['file'], CAPTURE_SOURCE_SHA)
    require(c['source_hashes'] == captured['source_hashes'] and c['runtime_source_hashes'] == captured['runtime_source_hashes'], 'Original source lineage differs')
    for name, sha in dict(c['source_hashes'], **c['runtime_source_hashes']).items():
        target = (source_root / name).resolve(strict=True)
        require(target.is_relative_to(source_root), 'Original source path escaped root'); parent.pin(target, sha)
    dataset = calibration['dataset']; artifact(path.parent, dataset)
    rows = dataset['rows']
    require(dataset['file'] == 'dataset.txt' and dataset['input_count'] == 2 and dataset['line_count'] == len(rows) == 32
        and [(r['sample_id'], r['iteration']) for r in rows] == [(f'calib{i:02d}', step) for i in range(8) for step in range(4)],
        'Exactly 32 ordered original two-input calibration rows required')
    require(c['input_bindings'] == [dict(index=i, name=n, shape=s, dtype='float32') for i, (n, s) in enumerate(INPUTS)]
        and c['output_bindings'] == [dict(index=0, name='pose_delta', shape=[1, 2, 9], dtype='float32')], 'Original logical graph I/O required')
    return path, c, artifact(path.parent, c['rknn'])


class RKNNCalibratedCameraHead(parent.RKNNOriginalCameraHead):
    def __init__(self, model, compiled_path, compiled_sha256, *, source_root=parent.BASE / 'src',
                 io_helper_source=parent.BASE / 'experiments/vggt-v80-source-r1/rknn_calibrated_retained.py',
                 split_fc1=True, **options):
        require(split_fc1 is True, 'The unchanged original split4 CameraHead is required')
        source_root = Path(source_root).resolve(strict=True)
        receipt, compiled, model_path = load_compilation(compiled_path, compiled_sha256, source_root)
        io = parent.load_source(io_helper_source, IO_HELPER_SHA, '_v82_calibrated_camera_IO')
        self._extra_contexts = []
        try:
            super().__init__(model, source_root=source_root, split_fc1=True, **options)
            self.stats.update(context_init_attempts=1, original_FP16_model_sha256=self.stats['model_sha256'],
                              original_FP16_sdk_memory=dict(self.stats['sdk_memory']))
            candidate = self.base.CONTEXT(); self._extra_contexts.append(candidate)
            filename = ct.create_string_buffer(str(model_path).encode())
            self.stats['context_init_attempts'] += 1
            code = self.api.rknn_init(ct.byref(candidate), ct.cast(filename, ct.c_void_p), 0, 0, None)
            self.stats['contexts_created'] += int(bool(candidate.value)); self.base.check_call(code, 'rknn_init calibrated CameraHead')
            require(bool(candidate.value), 'Empty calibrated CameraHead context')
            self.base.check_call(self.api.rknn_set_core_mask(candidate, self.stats['core_mask']), 'rknn_set_core_mask calibrated CameraHead')
            version = self.base.query(self.api, candidate, 5, self.base.SDKVersion())
            require(re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()), 'Typed SDK2.3.2 required')
            counts = self.base.query(self.api, candidate, 0, self.base.InputOutputNum())
            require((counts.n_input, counts.n_output) == (2, 1), 'Two original inputs and one original delta required')
            attrs = [self.base.query(self.api, candidate, 1, self.base.TensorAttr(index=i)) for i in range(2)]
            bindings, descriptors, shapes = {}, [], dict(INPUTS)
            for i, attr in enumerate(attrs):
                name = bytes(attr.name).decode()
                require(name in shapes and name not in bindings, 'Original CameraHead input names required')
                row = io.calibrated_io_attr(self.base, attr, shapes[name], name, i)
                descriptors.append(row); bindings[name] = dict(index=i, format=row['format'])
            require(set(bindings) == set(shapes), 'Both original input bindings required')
            output = self.base.query(self.api, candidate, 2, self.base.TensorAttr(index=0))
            output_descriptor = io.calibrated_io_attr(self.base, output, [1, 2, 9], 'pose_delta', 0)
            memory = self.base.query(self.api, candidate, 6, self.base.MemSize())
            self.base.check_call(self.api.rknn_destroy(self.context), 'rknn_destroy original FP16 CameraHead')
            self.stats['contexts_destroyed'] += 1; self.context.value = 0
            self.context = candidate; self._extra_contexts.remove(candidate); self.bindings = bindings
            self.stats.update(model_sha256=compiled['rknn']['sha256'], model_bytes=compiled['rknn']['bytes'],
                compile_receipt_sha256=compiled_sha256, capture_sha256=CAPTURE_SHA, compiler_source_sha256=COMPILER_SOURCE_SHA,
                calibrated_camera=True, requested_quantized_dtype='w8a8', original_ONNX_graph_unchanged=True,
                original_four_iterations_FP32_accumulation_and_activation_unchanged=True, candidate_accepted=False,
                all_internal_operators_INT8_verified=False, API_FP32_conversion_not_internal_FP32=True,
                sdk_dma_bytes=int(memory.total_dma_allocated_size),
                sdk_memory=dict(weight_bytes=int(memory.total_weight_size), internal_bytes=int(memory.total_internal_size), dma_bytes=int(memory.total_dma_allocated_size)),
                input_attrs=descriptors, output_attr=output_descriptor,
                calibrated_receipt=str(receipt), calibrated_io_helper_sha256=IO_HELPER_SHA)
        except BaseException as exc:
            self.failed = True
            try: self.close()
            except BaseException as cleanup: exc.add_note('CameraHead cleanup: ' + str(cleanup))
            exc.camera_rknn_metrics = self.metrics()
            raise

    def close(self):
        with self.lock:
            for context in reversed(self._extra_contexts):
                if context.value and self.api is not None:
                    try:
                        self.base.check_call(self.api.rknn_destroy(context), 'rknn_destroy pending calibrated CameraHead')
                        self.stats['contexts_destroyed'] += 1; context.value = 0
                    except BaseException as exc: self.stats['cleanup_errors'].append('Pending calibrated context: ' + str(exc))
            super().close()

    def metrics(self):
        result = super().metrics()
        result['live_contexts'] += sum(bool(c.value) for c in self._extra_contexts)
        return result


def install_calibrated_camera(model, compiled_path, compiled_sha256, **options):
    return RKNNCalibratedCameraHead(model, compiled_path, compiled_sha256, **options)
