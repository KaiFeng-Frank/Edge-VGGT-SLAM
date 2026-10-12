"""Experimental W8A8 execution of the unchanged original VGGT DPT graph.

The original adapter verifies all original parameters, spatial constants,
chunking and CPU activation. This leaf swaps only the compiled RKNN context.
"""
from __future__ import annotations
import ctypes as ct
import json
import math
from pathlib import Path
import re
from vggt_slam import rknn_depth_head as parent

PARENT_SOURCE_SHA = '7efacdb27be4aeb134fd2b64e2819136320fc45d25973517584522d314ba7894'
PARENT_MANIFEST_SHA = '9a6e5435df543b8484a4fc50da4b065995f32153233548829234dd149fbc7135'
GRAPH_SHA = '5f885c944f6ceff2c3eb8f5030249b90a85f5040a9a5ba892f74b95803af2a5b'
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
TAIL_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
IO_HELPER_SHA = 'f3f106034c6621a1e3ed07bdb265107b8522103f7d186ba4bbe0d0cefda16f2f'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
INPUT_NAMES = ['tokens_4', 'tokens_11', 'tokens_17', 'tokens_23']
INPUT_SHAPE = [1, 48, 2048]
OUTPUT_SHAPE = [1, 2, 84, 112]


def require(condition, message):
    if not condition: raise ValueError(message)


def artifact(directory, row):
    root, relative = Path(directory).resolve(strict=True), Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.stat().st_size == row['bytes'], 'Artifact path or size differs')
    return parent.pin(path, row['sha256'])


def load_compilation(path, digest, source_root, *, compiler_source, compiler_sha256):
    """Read and verify a pinned compiler receipt without creating a model/context."""
    source_root = Path(source_root).resolve(strict=True)
    parent.pin(source_root / 'vggt_slam/rknn_depth_head.py', PARENT_SOURCE_SHA)
    require(Path(parent.__file__).resolve() == source_root / 'vggt_slam/rknn_depth_head.py', 'Original depth import escaped source root')
    parent.pin(compiler_source, compiler_sha256)
    path = parent.pin(path, digest); c = json.loads(path.read_text())
    require(c['schema'] == 'v85_original_DPT_depth_logits_INT8_compile'
        and c['status'] == 'compiled_not_quality_accepted' and c['script_sha256'] == compiler_sha256
        and c['graph_sha256'] == c['graph']['sha256'] == GRAPH_SHA and c['graph']['bytes'] == 135995381
        and c['original_ONNX_modified'] is False and c['original_depth_manifest_sha256'] == PARENT_MANIFEST_SHA
        and c['configuration'] == CONFIG and c['compile_cpus'] == [4, 6]
        and c['build_options'] == dict(do_quantization=True, dataset=str(path.parent / 'dataset.txt'), auto_hybrid=False)
        and c['candidate_accepted'] is False and c['quality_accepted'] is False
        and c['original_model_only'] is True and c['model_inference_executed'] is False and c['ORT_rerun'] is False
        and c['calibration_executed'] is True and c['toolkit_version'] == '2.3.2'
        and c['released'] is True and c['release_code'] in (None, 0) and c['cleanup_errors'] == []
        and all(c[k] is True for k in ('source_and_inputs_unchanged', 'environment_restored', 'affinity_restored')),
        'A clean original DPT W8A8 compile is required')
    require([o['name'] for o in c['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
        and all(o['status'] == 'completed' and o['code'] in (None, 0) for o in c['operations']), 'Four successful SDK compile operations required')
    graph = parent.pin(c['graph']['file'], GRAPH_SHA)
    original_path = parent.pin(graph.parent / 'compiled.json', PARENT_MANIFEST_SHA)
    original = json.loads(original_path.read_text())
    require(c['checkpoint'] == original['checkpoint'] and c['original_parameters'] == {name: dict(row, finite=True) for name, row in original['parameters'].items()}
        and len(c['original_parameters']) == 62 and c['original_positions'] == original['positions']
        and c['original_architecture'] == original['architecture']
        and c['original_export_sha256'] == original['export_manifest_sha256']
        and c['original_activate_head_CPU_unchanged'] is True, 'Original 62 parameters, positions and architecture required')
    require(c['prior_FP32_check_receipt_sha256'] == '0e32292723b9b97dd5c177e08ec0579ca59b312d23634f3215b8e574b27c143b'
        and c['prior_FP16_compiled_manifest_sha256'] == PARENT_MANIFEST_SHA, 'Original FP32 gate and FP16 ancestor required')
    calibration = c['calibration']; cap_row = calibration['capture']
    cap_path = parent.pin(cap_row['file'], cap_row['sha256']); captured = json.loads(cap_path.read_text())
    require(cap_path.stat().st_size == cap_row['bytes'] and c['capture_sha256'] == cap_row['sha256']
        and calibration['source_calibration_sha256'] == CALIBRATION_SHA
        and calibration['original_tail_capture_sha256'] == TAIL_SHA
        and calibration['pair_count'] == 8 and calibration['frames_per_pair'] == 2,
        'Original eight-pair sixteen-frame calibration required')
    require(captured['schema'] == 'v85_original_VGGT_depth_PTQ_calibration_capture'
        and captured['status'] == 'completed_original_FP32_depth_calibration_capture'
        and captured['script_sha256'] == c['capture_source']['sha256']
        and captured['checkpoint'] == c['checkpoint'] and captured['parameters'] == c['original_parameters']
        and calibration['samples'] == captured['samples'] and len(captured['samples']) == 8
        and captured['NPU_or_cache_installed'] is False and captured['cleanup_errors'] == []
        and all(captured[k] is True for k in ('original_model_only', 'model_state_unchanged',
            'source_and_inputs_unchanged', 'hooks_removed', 'affinity_restored', 'environment_restored',
            'all_depth_and_conf_bitwise_equal_to_original_capture')), 'Original FP32 capture proof required')
    parent.pin(c['capture_source']['file'], c['capture_source']['sha256'])
    require(c['source_hashes'] == captured['source_hashes'] and c['runtime_source_hashes'] == captured['runtime_source_hashes'], 'Source lineage differs')
    for name, sha in dict(c['source_hashes'], **c['runtime_source_hashes']).items():
        target = (source_root / name).resolve(strict=True)
        require(target.is_relative_to(source_root), 'Source path escaped root'); parent.pin(target, sha)
    dataset = calibration['dataset']; dataset_path = artifact(path.parent, dataset); rows = dataset['rows']
    require(dataset['file'] == 'dataset.txt' and dataset['input_count'] == 4 and dataset['line_count'] == len(rows) == 16
        and [(r['sample_id'], r['frame']) for r in rows] == [(f'calib{i:02d}', f) for i in range(8) for f in range(2)], 'Ordered sixteen-frame dataset required')
    require(dataset_path.read_text() == ''.join(' '.join(r['paths']) + '\n' for r in rows), 'Dataset content differs')
    require(c['input_bindings'] == [dict(index=i, name=n, shape=INPUT_SHAPE, dtype='float32') for i,n in enumerate(INPUT_NAMES)]
        and c['output_bindings'] == [dict(index=0, name='logits', shape=OUTPUT_SHAPE, dtype='float32')], 'Original logical I/O required')
    return path, c, artifact(path.parent, c['rknn'])


def calibrated_depth_io_attr(base, attr, shape, name, index=0, formats=(3,)):
    """Logical floating/affine INT8 I/O with original DPT output format support."""
    row = base.attr_record(attr)
    identity = row['type'] == 1 and row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
    integer, count = row['type'] in (2, 3), math.prod(shape)
    if (row['index'] != index or row['name'] != name or row['shape'] != list(shape)
            or row['n_elems'] != count or row['type'] not in (0, 1, 2, 3)
            or row['format'] not in formats or row['pass_through'] != 0
            or not math.isfinite(row['scale'])
            or row['size'] != count * {0: 4, 1: 2, 2: 1, 3: 1}.get(row['type'], 0)
            or (integer and (row['quantization_type'] != 2 or row['scale'] <= 0.
                or not ((-128 <= row['zero_point'] <= 127) if row['type'] == 2 else (0 <= row['zero_point'] <= 255))))
            or (not integer and row['quantization_type'] != 0 and not identity)):
        raise ValueError('Expected original logical floating or finite affine DPT I/O: ' + str(row))
    return row


class RKNNCalibratedDepthHead(parent.RKNNOriginalDepthHead):
    def __init__(self, model, manifest_path, manifest_sha256, *, compiled_path, compiled_sha256,
            compiler_source, compiler_sha256, source_root=parent.BASE / 'src',
            io_helper_source=parent.BASE / 'experiments/vggt-v80-source-r1/rknn_calibrated_retained.py', **options):
        source_root = Path(source_root).resolve(strict=True)
        require(manifest_sha256 == PARENT_MANIFEST_SHA, 'Frozen original DPT parent required')
        receipt, compiled, model_path = load_compilation(compiled_path, compiled_sha256, source_root,
            compiler_source=compiler_source, compiler_sha256=compiler_sha256)
        parent.pin(io_helper_source, IO_HELPER_SHA)
        self._extra_contexts = []
        try:
            super().__init__(model, manifest_path, manifest_sha256, source_root=source_root, **options)
            self.stats['original_FP16_model_memory'] = list(self.stats['model_memory'])
            candidate = self.base.CONTEXT(); self._extra_contexts.append(candidate)
            filename = ct.create_string_buffer(str(model_path).encode())
            self.stats['context_init_attempts'] += 1
            code = self.api.rknn_init(ct.byref(candidate), ct.cast(filename, ct.c_void_p), 0, 0, None)
            self.stats['contexts_created'] += int(bool(candidate.value)); self.base.check_call(code, 'rknn_init calibrated DPT')
            require(bool(candidate.value), 'Empty calibrated DPT context')
            self.base.check_call(self.api.rknn_set_core_mask(candidate, self.stats['core_mask']), 'rknn_set_core_mask calibrated DPT')
            version = self.base.query(self.api, candidate, 5, self.base.SDKVersion())
            require(re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()), 'Typed SDK 2.3.2 required')
            counts = self.base.query(self.api, candidate, 0, self.base.InputOutputNum())
            require((counts.n_input, counts.n_output) == (4, 1), 'Four original token inputs and one logit output required')
            descriptors = [calibrated_depth_io_attr(self.base,
                self.base.query(self.api, candidate, 1, self.base.TensorAttr(index=i)), INPUT_SHAPE, name, i)
                for i, name in enumerate(INPUT_NAMES)]
            output = calibrated_depth_io_attr(self.base,
                self.base.query(self.api, candidate, 2, self.base.TensorAttr(index=0)), OUTPUT_SHAPE, 'logits', 0, formats=(0, 3))
            memory = self.base.query(self.api, candidate, 6, self.base.MemSize())
            self.base.check_call(self.api.rknn_destroy(self.contexts[0]), 'rknn_destroy original FP16 DPT')
            self.stats['contexts_destroyed'] += 1; self.contexts[0].value = 0
            self.contexts[0] = candidate; self._extra_contexts.remove(candidate); self.inputs = descriptors
            self.stats.update(model_sha256=compiled['rknn']['sha256'], model_bytes=compiled['rknn']['bytes'],
                compile_receipt_sha256=compiled_sha256, compiler_source_sha256=compiler_sha256,
                capture_sha256=compiled['capture_sha256'], calibrated_depth=True,
                requested_quantized_dtype='w8a8', original_ONNX_graph_unchanged=True,
                original_chunking_and_CPU_activation_unchanged=True, candidate_accepted=False,
                all_internal_operators_INT8_verified=False, API_FP32_conversion_not_internal_FP32=True,
                model_memory=[dict(sdk_dma_bytes=int(memory.total_dma_allocated_size),
                    weight_bytes=int(memory.total_weight_size), internal_bytes=int(memory.total_internal_size),
                    inputs=descriptors, output=output)], calibrated_receipt=str(receipt), calibrated_io_helper_sha256=IO_HELPER_SHA)
        except BaseException as exc:
            self.failed = True
            try: self.close()
            except BaseException as cleanup: exc.add_note('DPT cleanup: ' + str(cleanup))
            exc.depth_rknn_metrics = self.metrics()
            raise

    def close(self):
        with self.lock:
            for context in reversed(self._extra_contexts):
                if context.value and self.api is not None:
                    try:
                        self.base.check_call(self.api.rknn_destroy(context), 'rknn_destroy pending calibrated DPT')
                        self.stats['contexts_destroyed'] += 1; context.value = 0
                    except BaseException as exc: self.stats['cleanup_errors'].append('Pending calibrated DPT: ' + str(exc))
            super().close()

    def metrics(self):
        result = super().metrics()
        result['live_contexts'] += sum(bool(c.value) for c in self._extra_contexts)
        return result


def install_calibrated_depth(model, manifest_path, manifest_sha256, **options):
    return RKNNCalibratedDepthHead(model, manifest_path, manifest_sha256, **options)
