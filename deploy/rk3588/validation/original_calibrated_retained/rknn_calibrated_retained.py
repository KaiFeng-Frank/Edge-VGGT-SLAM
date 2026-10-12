"""Private four-group W8A8 diagnostic of the complete original aggregator.

The unchanged FP16 parent's constructor checks the original graphs, sources,
all 864 parameter tensors, positions and retained/global20 consumers. Only the
four RKNN files and SDK I/O route change. FP32 API conversion does not mean
FP32 internal math or numerical acceptance. Install before cache/session.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import re
import time

from vggt_slam import rknn_retained_aggregator as parent

PARENT_RUNTIME_SHA = '04fbb5b2ad7db22bf26d52d4a6b5a3fadffa042557919575938b035f5dd06b5f'
PARENT_MANIFEST_SHA = '771a9b82007d2feefd65ebbf3f4b0fb27070b79ff7f6d0faab1a8c866da90e43'
CAPTURE_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
COMPILER_SOURCE_SHA = 'fae89ae5de8434081cc581b364b94ad31d5e74ebde3e836b4b3ae4032eb9e251'
COMPILER_SOURCE_SHAS = (COMPILER_SOURCE_SHA,
    '13061f83535dadc1b951604556df1f260c168cf86fe14868a8ec734fd507139e')
SCHEMA = 'v80_original_VGGT_calibrated_retained_RKNN_manifest'
COMPILE_SCHEMA = 'v80_original_VGGT_retained_INT8_compile'
CONFIG = dict(parent.CONFIG, quantized_dtype='w8a8',
              quantized_algorithm='normal', quantized_method='channel')
PAIRS = [[i, i + 48] for i in range(0, 673, 96)]


def calibrated_io_attr(base, attr, shape, name, index=0):
    """The v56 floating/affine INT8 logical ABI; API always converts FP32."""
    row = base.attr_record(attr)
    identity = (row['type'] == 1 and row['quantization_type'] == 2
                and row['zero_point'] == 0 and row['scale'] == 1.)
    integer, count = row['type'] in (2, 3), math.prod(shape)
    if (row['index'] != index or row['name'] != name or row['shape'] != list(shape)
            or row['n_elems'] != count or row['type'] not in (0, 1, 2, 3)
            or row['format'] != 3 or row['pass_through'] != 0
            or not math.isfinite(row['scale'])
            or row['size'] != count * {0: 4, 1: 2, 2: 1, 3: 1}.get(row['type'], 0)
            or (integer and (row['quantization_type'] != 2 or row['scale'] <= 0.
                or not ((-128 <= row['zero_point'] <= 127) if row['type'] == 2
                        else (0 <= row['zero_point'] <= 255))))
            or (not integer and row['quantization_type'] != 0 and not identity)):
        raise ValueError('Expected fixed floating or finite affine INT8/UINT8 group I/O: ' + str(row))
    return row


def load_mapping(manifest_path, digest, original_manifest_path, original_digest):
    """No model/SDK calls: pin the four compiler leaves to the original groups."""
    pin = parent.original.pin
    if original_digest != PARENT_MANIFEST_SHA:
        raise ValueError('Frozen original retained-group manifest required')
    original_path = pin(original_manifest_path, original_digest)
    original = json.loads(original_path.read_text())
    path = pin(manifest_path, digest)
    mapping = json.loads(path.read_text())
    compiler_digests = mapping.get('compiler_source_sha256_by_group',
                                  [mapping['compiler_source_sha256']] * 4)
    capture_dir = Path(mapping['calibration_directory']).resolve(strict=True)
    capture = json.loads(pin(capture_dir / 'capture.json', CAPTURE_SHA).read_text())
    if (mapping['schema'] != SCHEMA or mapping['status'] != 'compiled_not_quality_accepted'
            or mapping['candidate_accepted'] is not False
            or mapping['parent_manifest_sha256'] != PARENT_MANIFEST_SHA
            or mapping['capture_sha256'] != CAPTURE_SHA
            or Path(mapping['parent_manifest_directory']).resolve(strict=True) != original_path.parent
            or mapping['compiler_source_sha256'] not in COMPILER_SOURCE_SHAS
            or len(compiler_digests) != 4
            or any(value not in COMPILER_SOURCE_SHAS for value in compiler_digests)
            or mapping['compiled_count'] != 4
            or len(mapping['blocks']) != 4
            or capture['schema'] != 'v56_original_VGGT_calibration_capture'
            or capture['status'] != 'completed_original_FP32_calibration_capture'
            or capture['full_forward_completed'] != 8 or capture['cleanup_errors']
            or capture['checkpoint'] != original['checkpoint']
            or [s['indices'] for s in capture['samples']] != PAIRS):
        raise ValueError('Frozen original parent/eight calibration forwards/four groups differ')
    paths, identities = [], []
    for index, row in enumerate(mapping['blocks']):
        old = original['blocks'][index]
        start = parent.RANGES[index][0]
        leaf_path = parent.artifact(path.parent, row['compile_receipt'])
        leaf = json.loads(leaf_path.read_text())
        calibration = leaf['calibration']
        dataset_path = parent.artifact(leaf_path.parent, calibration['dataset'])
        samples = capture['groups']['aggregator'][start]['samples']
        expected = [str(parent.artifact(capture_dir, sample['artifact'])) for sample in samples]
        if (row['index'] != index or row['indices'] != old['indices']
                or leaf['schema'] != COMPILE_SCHEMA or leaf['status'] != 'compiled_not_quality_accepted'
                or leaf['group_index'] != index or leaf['indices'] != old['indices']
                or leaf['parent_manifest_sha256'] != PARENT_MANIFEST_SHA
                or Path(leaf['parent_manifest_directory']).resolve(strict=True) != original_path.parent
                or Path(leaf['calibration_directory']).resolve(strict=True) != capture_dir
                or leaf['parent_group'] != old or leaf['graph_sha256'] != old['sha256']
                or leaf['prior_check_receipt'] != old['check_receipt']
                or leaf['prior_FP16_compile_receipt'] != old['compile_receipt']
                or leaf['original_ONNX_modified'] is not False
                or pin(leaf['graph']['file'], leaf['graph']['sha256']) != parent.artifact(original_path.parent, old)
                or leaf['graph']['sha256'] != old['sha256'] or leaf['graph']['bytes'] != old['bytes']
                or leaf['capture_sha256'] != CAPTURE_SHA
                or calibration['capture']['sha256'] != CAPTURE_SHA
                or calibration['capture']['bytes'] != (capture_dir / 'capture.json').stat().st_size
                or calibration['group_start_index'] != start or calibration['samples'] != samples
                or len(samples) != 8 or calibration['dataset']['input_count'] != 1
                or calibration['dataset']['line_count'] != 8
                or dataset_path.read_text().splitlines() != expected
                or leaf['configuration'] != CONFIG
                or leaf['build_options'] != dict(do_quantization=True, dataset=str(dataset_path), auto_hybrid=False)
                or leaf['script_sha256'] != compiler_digests[index]
                or (leaf['script_sha256'] == COMPILER_SOURCE_SHAS[1]
                    and leaf.get('compile_cpus') != [4, 6])
                or [op['name'] for op in leaf['operations']] != ['config', 'load_onnx', 'build', 'export_rknn']
                or any(op['status'] != 'completed' or op['code'] not in (0, None) for op in leaf['operations'])
                or leaf['released'] is not True or leaf['release_code'] not in (0, None)
                or leaf['cleanup_errors'] or leaf['candidate_accepted'] is not False
                or any(leaf[key] is not True for key in ('source_and_inputs_unchanged',
                    'environment_restored', 'affinity_restored'))):
            raise ValueError('Pinned original INT8 compilation/calibration lineage differs for group ' + str(index))
        candidate = parent.artifact(leaf_path.parent, leaf['rknn'])
        paths.append(candidate)
        identities.append((index, row['compile_receipt']['sha256'], leaf['rknn']['sha256']))
    return mapping, tuple(paths), tuple(identities)


class RKNNCalibratedRetained(parent.RKNNRetainedOriginalAggregator):
    def __init__(self, model, manifest_path, manifest_sha256, *,
                 quantized_manifest, quantized_manifest_sha256, **options):
        parent.original.pin(parent.__file__, PARENT_RUNTIME_SHA)
        mapping, self.quantized_paths, self.quantized_identities = load_mapping(
            quantized_manifest, quantized_manifest_sha256, manifest_path, manifest_sha256)
        super().__init__(model, manifest_path, manifest_sha256, **options)
        self.stats['fp32_io'] = dict(calls_attempted=0, calls_completed=0,
            owned_output_copy_seconds=0., finite_check_seconds=0., route_seconds=0.,
            **{name + '_seconds': 0. for name in ('sdk_api', 'rknn_inputs_set',
                'rknn_run', 'rknn_outputs_get', 'rknn_outputs_release')})
        self.stats.update(quantized_manifest_sha256=quantized_manifest_sha256,
            original_retained_manifest_sha256=manifest_sha256,
            calibration_capture_sha256=CAPTURE_SHA,
            compiler_source_sha256=mapping['compiler_source_sha256'],
            compiler_source_sha256_by_group=mapping.get('compiler_source_sha256_by_group',
                [mapping['compiler_source_sha256']] * 4),
            compile_receipt_sha256=[item[1] for item in self.quantized_identities],
            quantized_rknn_sha256=[item[2] for item in self.quantized_identities],
            requested_quantized_dtype='w8a8', requested_quantized_algorithm='normal',
            requested_quantized_method='channel', internal_precision_changed=True,
            requested_float_dtype_for_unquantized_operators='float16',
            original_ONNX_graphs_unchanged=True, all_internal_operators_INT8_verified=False,
            half_io_aggregator_enabled=False, input_api_type=0, input_pass_through=0,
            output_want_float=1, API_FP32_conversion_not_internal_FP32=True)

    def _backend_state(self):
        return super()._backend_state(), self.quantized_identities, tuple(map(str, self.quantized_paths))

    def _open_group(self, ordinal, indices, public, original_path):
        # The parent has already checked its original graph/check/compile/RKNN.
        context = self.base.CONTEXT()
        self.contexts.append(context)
        filename = ct.create_string_buffer(str(self.quantized_paths[ordinal]).encode())
        self.stats['context_init_attempts'] += 1
        code = self.api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
        self.stats['contexts_created'] += int(bool(context.value))
        self.base.check_call(code, 'rknn_init')
        if not context.value:
            raise ValueError('Empty calibrated retained-group context')
        self.base.check_call(self.api.rknn_set_core_mask(context, self.core_mask), 'rknn_set_core_mask')
        version = self.base.query(self.api, context, 5, self.base.SDKVersion())
        if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()):
            raise ValueError('Frozen SDK2.3.2 required')
        count = self.base.query(self.api, context, 0, self.base.InputOutputNum())
        if (count.n_input, count.n_output) != (1, len(public)):
            raise ValueError('Original retained two/three-output ABI differs')
        inp = calibrated_io_attr(self.base, self.base.query(self.api, context, 1,
            self.base.TensorAttr(index=0)), parent.FRAME, 'input')
        outs = [calibrated_io_attr(self.base, self.base.query(self.api, context, 2,
            self.base.TensorAttr(index=b['index'])), b['shape'], b['name'], b['index']) for b in public]
        memory = self.base.query(self.api, context, 6, self.base.MemSize())
        self.stats['model_memory'].append(dict(group=ordinal,
            sdk_dma_bytes=int(memory.total_dma_allocated_size), input=inp, outputs=outs,
            sdk_version=bytes(version.api_version).decode(), driver_version=bytes(version.drv_version).decode()))
        self.groups.append((ordinal, indices, context, public))

    def _run_group(self, context, array, public):
        base, api, np = self.base, self.api, self.np
        timing = self.stats['fp32_io']
        timing['calls_attempted'] += 1
        started = time.perf_counter()
        def call(name, fn, *args):
            tick = time.perf_counter()
            try:
                return fn(*args)
            finally:
                elapsed = time.perf_counter() - tick
                timing[name + '_seconds'] += elapsed
                timing['sdk_api_seconds'] += elapsed
        try:
            if (array.shape != parent.FRAME or array.dtype != np.float32
                    or not array.flags.c_contiguous or not np.isfinite(array).all()):
                raise ValueError('Dense finite original FP32 frame input required')
            inputs = (base.Input * 1)(base.Input(0, array.ctypes.data, array.nbytes, 0, 0, 3))
            count, got, owned = len(public), False, []
            outputs = (base.Output * count)()
            for index in range(count):
                outputs[index].index, outputs[index].want_float = index, 1
            run, received = base.RunExtend(0, 0, self.timeout, -1), base.OutputExtend()
            begin = time.perf_counter()
            try:
                base.check_call(call('rknn_inputs_set', api.rknn_inputs_set, context, 1, inputs), 'rknn_inputs_set')
                self.stats['sdk_run_attempts'] += 1
                code = call('rknn_run', api.rknn_run, context, ct.byref(run))
                self.stats['sdk_runs_succeeded'] += int(code == 0)
                base.check_call(code, 'rknn_run')
                code = call('rknn_outputs_get', api.rknn_outputs_get, context, count, outputs, ct.byref(received))
                self.stats['outputs_acquired'] += sum(bool(output.buf) for output in outputs)
                base.check_call(code, 'rknn_outputs_get')
                got = True
                if run.frame_id != received.frame_id:
                    raise ValueError('Synchronous original group frame ID differs')
                tick = time.perf_counter()
                try:
                    for output, binding in zip(outputs, public):
                        shape, size = tuple(binding['shape']), math.prod(binding['shape'])
                        if not output.buf or output.size != size * 4:
                            raise ValueError('FP32-converted group output pointer/size differs')
                        view = np.ctypeslib.as_array(ct.cast(output.buf, ct.POINTER(ct.c_float)), shape=(size,))
                        owned.append(np.array(view, dtype=np.float32, copy=True).reshape(shape))
                finally:
                    timing['owned_output_copy_seconds'] += time.perf_counter() - tick
            finally:
                try:
                    if got or any(output.buf for output in outputs):
                        acquired = sum(bool(output.buf) for output in outputs)
                        code = call('rknn_outputs_release', api.rknn_outputs_release, context, count, outputs)
                        if code == 0:
                            self.stats['outputs_released'] += acquired
                        else:
                            self.stats['cleanup_errors'].append('rknn_outputs_release returned ' + str(code))
                        base.check_call(code, 'rknn_outputs_release')
                finally:
                    self.stats['sdk_io_seconds'] += time.perf_counter() - begin
            tick = time.perf_counter()
            try:
                if any(not value.flags.c_contiguous or not np.isfinite(value).all() for value in owned):
                    raise ValueError('Nonfinite FP32-converted original group output')
            finally:
                timing['finite_check_seconds'] += time.perf_counter() - tick
            timing['calls_completed'] += 1
            return owned
        finally:
            timing['route_seconds'] += time.perf_counter() - started


def install_calibrated_retained(model, manifest_path, manifest_sha256, **options):
    return RKNNCalibratedRetained(model, manifest_path, manifest_sha256, **options)
