"""Original VGGT DPT logits on RKNN; original CPU chunking/activation remain.

Only depth_head._forward_impl is overridden. The original depth_head.forward
still chooses its frame chunks and concatenates them. Four retained token
layers are sliced exactly as before; each frame supplies one static graph call.
The pinned original activate_head and final views run on owned CPU FP32 logits.
Install before the patch cache and close the cache before restoring this handle.
"""
from __future__ import annotations

from vggt_slam.shared_context_registry import initialize as initialize_shared_context, destroy as destroy_shared_context

import ctypes as ct
import importlib.util
import json
import math
from pathlib import Path
import re
import threading
import time

from .rknn_aggregator_blocks import (
    ABI_SHA, BASE, HEADER_SHA, LIBRARY_SHA, RKNNOriginalAggregatorBlocks, artifact, pin,
)

CONTRACT_SHA = 'efdad006ed594636041e5fdeb315b81ec8154d89d74887fabcf26f77e12bb9ba'


def load_source(path, expected, name):
    spec = importlib.util.spec_from_file_location(name, pin(path, expected))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def floating_binding(base, attr, index, name, shape, formats):
    row = base.attr_record(attr)
    identity = row['type'] == 1 and row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
    if (row['index'] != index or row['name'] != name or row['shape'] != list(shape)
            or row['n_elems'] != math.prod(shape) or row['type'] not in (0, 1)
            or (row['quantization_type'] != 0 and not identity) or row['format'] not in formats
            or row['pass_through'] != 0 or not math.isfinite(row['scale'])
            or row['size'] != math.prod(shape)*(4 if row['type'] == 0 else 2)):
        raise ValueError(f'Expected original logical floating DPT binding: {row}')
    return row


class RKNNOriginalDepthHead(RKNNOriginalAggregatorBlocks):
    """One original logits context, with the existing context metrics interface."""

    def __init__(self, model, manifest_path, manifest_sha256, *, source_root=BASE / 'src',
                 library_path='/usr/lib/librknnrt.so', core_mask=7, timeout_ms=30000):
        self.lock = threading.RLock()
        self.contexts, self.head, self.patch = [], None, None
        self.api = self.base = None
        self.closed = self.failed = False
        self.stats = dict(context_init_attempts=0, contexts_created=0, contexts_destroyed=0,
            sdk_run_attempts=0, sdk_runs_succeeded=0, sdk_io_seconds=0., completed_chunk_calls=0,
            completed_frames=0, cleanup_errors=[], model_memory=[], candidate_accepted=False,
            full_model_quality_validated=False, real_time_validated=False, all_operators_on_npu_verified=False)
        try:
            if type(core_mask) is not int or core_mask not in (1, 2, 4, 7) or type(timeout_ms) is not int or not 0 < timeout_ms <= 600000:
                raise ValueError('core_mask must be 1/2/4/7; timeout_ms must be 1..600000')
            source_root = Path(source_root).resolve(strict=True)
            self.base = load_source(source_root / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA, '_v54_depth_ABI')
            self.stats['board'] = self.base.require_board()
            pin(source_root / 'deploy/rk3588/reference/rknn_api_v2.3.2.h', HEADER_SHA)
            import numpy as np
            import torch
            self.np, self.torch, self.timeout = np, torch, timeout_ms
            self.contract = C = load_source(source_root / 'deploy/rk3588/validation/original_depth/depth_contract.py',
                                            CONTRACT_SHA, '_v54_depth_contract')
            manifest_path = pin(manifest_path, manifest_sha256)
            manifest = json.loads(manifest_path.read_text())
            if (manifest['schema'] != 'v54_original_VGGT_depth_logits_RKNN_manifest'
                    or manifest['checkpoint'] != C.CHECKPOINT or manifest['contract_source_sha256'] != CONTRACT_SHA
                    or manifest['all_62_original_parameter_values_preserved_in_graph'] is not True
                    or manifest['original_inplace_relu_residual_skip_preserved'] is not True
                    or manifest['all_five_original_align_corners_resizes_preserved'] is not True):
                raise ValueError('The unchanged original depth-logits graph provenance is required')
            for relative, expected in manifest['source_hashes'].items():
                source = (source_root / relative).resolve(strict=True)
                if not source.is_relative_to(source_root):
                    raise ValueError('Source path escapes original source root')
                pin(source, expected)
            if any(manifest['source_hashes'].get(name) != value for name, value in C.SOURCE_HASHES.items()):
                raise ValueError('All fixed original DPT and activation sources are required')
            from vggt.heads import dpt_head, head_act
            pin(dpt_head.__file__, C.SOURCE_HASHES['third_party/vggt/vggt/heads/dpt_head.py'])
            pin(head_act.__file__, C.SOURCE_HASHES['third_party/vggt/vggt/heads/head_act.py'])
            if dpt_head.activate_head is not head_act.activate_head:
                raise ValueError('Original DPT activate_head binding changed')
            self.activate_head = dpt_head.activate_head
            head = self.head = model.depth_head
            if (head is None or any(m.training for m in model.modules())
                    or getattr(head, '_vggt_rknn_depth_owner', None) is not None
                    or getattr(model.aggregator, '_vggt_patch_cache_owner', None) is not None
                    or model.inference_image_size != 112 or model.depth_frames_chunk_size != 1
                    or head.forward.__func__ is not dpt_head.DPTHead.forward
                    or head._forward_impl.__func__ is not dpt_head.DPTHead._forward_impl
                    or head.scratch.output_conv2[1].inplace is not True
                    or any(getattr(getattr(head.scratch, f'refinenet{i}'), f'resConfUnit{unit}').activation.inplace is not True
                           for i in range(1,5) for unit in ((1,2) if i < 4 else (2,)))
                    or C.head_contract(head) != manifest['architecture']):
                raise ValueError('Install once on original eval depth head with its one-frame chunk policy')
            parameters = manifest['parameters']
            state = dict(head.named_parameters())
            if set(state) != set(C.PARAMETER_SHAPES) or set(parameters) != set(C.PARAMETER_SHAPES):
                raise ValueError('Every original learned depth-head tensor must be present')
            for name, value in state.items():
                info, shape = parameters[name], C.PARAMETER_SHAPES[name]
                if (value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous()
                        or tuple(value.shape) != shape or info['shape'] != list(shape) or info['dtype'] != 'float32'
                        or info['tensor_bytes'] != math.prod(shape)*4):
                    raise ValueError('Original depth parameter shape/dtype changed: '+name)
                array = value.detach().numpy()
                if not np.isfinite(array).all() or C.raw_sha256(array) != info['raw_sha256']:
                    raise ValueError('Original depth parameter identity changed: '+name)
            positions = manifest['positions']
            if set(positions) != set(C.POSITION_SHAPES):
                raise ValueError('All five captured original positional additions are required')
            for name, shape in C.POSITION_SHAPES.items():
                info = positions[name]
                if (info['shape'] != list(shape) or info['dtype'] != 'float32' or info['tensor_bytes'] != math.prod(shape)*4
                        or not re.fullmatch('[0-9a-f]{64}', info['raw_sha256'])):
                    raise ValueError('Captured original positional constant metadata differs')
            graph = manifest['graph']
            if (graph['input_names'] != list(C.INPUT_NAMES)
                    or graph['input_shapes'] != {name: list(C.INPUT_SHAPE) for name in C.INPUT_NAMES}
                    or graph['output_name'] != 'logits' or graph['output_shape'] != list(C.OUTPUT_SHAPE)):
                raise ValueError('Original four token views and raw-logit output layout differ')
            artifact(manifest_path.parent, graph)
            path = artifact(manifest_path.parent, manifest['rknn'])
            self.api = self.base.bind_api(ct.CDLL(str(pin(library_path, LIBRARY_SHA))))
            context = self.base.CONTEXT()
            self.contexts.append(context)
            filename = ct.create_string_buffer(str(path).encode())
            self.stats['context_init_attempts'] += 1
            code = initialize_shared_context(self.api, ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
            self.stats['contexts_created'] += int(bool(context.value))
            self.base.check_call(code, 'rknn_init')
            if not context.value:
                raise RuntimeError('RKNN returned an empty original depth context')
            self.base.check_call(self.api.rknn_set_core_mask(context, core_mask), 'rknn_set_core_mask')
            version = self.base.query(self.api, context, 5, self.base.SDKVersion())
            if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()):
                raise ValueError('Frozen typed ABI requires RKNN runtime 2.3.2')
            counts = self.base.query(self.api, context, 0, self.base.InputOutputNum())
            if (counts.n_input, counts.n_output) != (4, 1):
                raise ValueError('Original depth graph requires four token inputs and one raw output')
            self.inputs = [floating_binding(self.base, self.base.query(self.api, context, 1, self.base.TensorAttr(index=i)),
                            i, name, C.INPUT_SHAPE, (3,)) for i, name in enumerate(C.INPUT_NAMES)]
            output = floating_binding(self.base, self.base.query(self.api, context, 2, self.base.TensorAttr(index=0)),
                                      0, 'logits', C.OUTPUT_SHAPE, (0, 3))
            memory = self.base.query(self.api, context, 6, self.base.MemSize())
            self.stats.update(manifest_sha256=manifest_sha256, core_mask=core_mask, timeout_ms=timeout_ms,
                verified_original_parameters=len(state), original_layers=list(C.LAYERS),
                original_chunking_preserved=True, original_activation_preserved=True,
                model_memory=[dict(sdk_dma_bytes=int(memory.total_dma_allocated_size), inputs=self.inputs, output=output)])
            forward_impl = self._make_forward_impl()
            self.patch = ('_forward_impl' in head.__dict__, head.__dict__.get('_forward_impl'), head._forward_impl, forward_impl)
            head._forward_impl = forward_impl
            head._vggt_rknn_depth_owner = self
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _make_forward_impl(self):
        def forward_impl(aggregated_tokens_list, images, patch_start_idx, frames_start_idx=None, frames_end_idx=None):
            with self.lock:
                torch, np, C = self.torch, self.np, self.contract
                if (self.closed or self.failed or self.head.training or torch.is_grad_enabled()
                        or not isinstance(images, torch.Tensor) or images.device.type != 'cpu'
                        or images.dtype != torch.float32 or images.requires_grad
                        or tuple(images.shape) not in ((1,1,3,84,112), (1,2,3,84,112))
                        or type(patch_start_idx) is not int or patch_start_idx != 5
                        or not isinstance(aggregated_tokens_list, (list,tuple)) or len(aggregated_tokens_list) != 24):
                    raise RuntimeError('Original depth RKNN requires retained CPU FP32 fixed-image tokens; no fallback')
                full_frames = images.shape[1]
                if frames_start_idx is None and frames_end_idx is None:
                    start, end = 0, full_frames
                elif (type(frames_start_idx) is int and type(frames_end_idx) is int
                      and 0 <= frames_start_idx < frames_end_idx <= full_frames):
                    start, end = frames_start_idx, frames_end_idx
                else:
                    raise ValueError('Original frame chunk bounds differ')
                selected = []
                for layer in C.LAYERS:
                    value = aggregated_tokens_list[layer]
                    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32
                            or value.requires_grad or value.layout != torch.strided
                            or tuple(value.shape) != (1,full_frames,53,2048)):
                        raise ValueError('Original retained DPT layer layout differs')
                    # Exact original patch/frame selection and B*S flattening, before its LayerNorm graph input.
                    array = value.detach().numpy()[:, start:end, patch_start_idx:]
                    selected.append(array.reshape(end-start,48,2048))
                try:
                    outputs = []
                    for frame in range(end-start):
                        arrays = [np.ascontiguousarray(value[frame:frame+1]) for value in selected]
                        if not all(np.isfinite(value).all() for value in arrays):
                            raise ValueError('Nonfinite original DPT token view')
                        raw = self._run_depth(arrays)
                        if not np.isfinite(raw).all():
                            raise ValueError('Nonfinite original RKNN depth logits')
                        outputs.append(raw)
                        self.stats['completed_frames'] += 1
                    out = torch.from_numpy(outputs[0] if len(outputs)==1 else np.concatenate(outputs,axis=0))
                    preds, conf = self.activate_head(out, activation=self.head.activation, conf_activation=self.head.conf_activation)
                    # These are the original _forward_impl's final view operations.
                    preds = preds.view(1,end-start,*preds.shape[1:])
                    conf = conf.view(1,end-start,*conf.shape[1:])
                    self.stats['completed_chunk_calls'] += 1
                    return preds, conf
                except BaseException:
                    self.failed = True
                    raise
        return forward_impl

    def _run_depth(self, arrays):
        base, api, context = self.base, self.api, self.contexts[0]
        inputs = (base.Input*4)(*(base.Input(i,array.ctypes.data,array.nbytes,0,0,self.inputs[i]['format'])
                                 for i,array in enumerate(arrays)))
        outputs = (base.Output*1)()
        outputs[0].want_float = 1
        run, received = base.RunExtend(0,0,self.timeout,-1), base.OutputExtend()
        got, begin = False, time.perf_counter()
        try:
            base.check_call(api.rknn_inputs_set(context,4,inputs),'rknn_inputs_set')
            self.stats['sdk_run_attempts'] += 1
            code = api.rknn_run(context,ct.byref(run))
            self.stats['sdk_runs_succeeded'] += int(code==0)
            base.check_call(code,'rknn_run')
            base.check_call(api.rknn_outputs_get(context,1,outputs,ct.byref(received)),'rknn_outputs_get')
            got = True
            shape = self.contract.OUTPUT_SHAPE
            if run.frame_id != received.frame_id or not outputs[0].buf or outputs[0].size != math.prod(shape)*4:
                raise RuntimeError('Original raw-logit synchronous frame/pointer/byte-size mismatch')
            view = self.np.ctypeslib.as_array(ct.cast(outputs[0].buf,ct.POINTER(ct.c_float)),shape=(math.prod(shape),))
            return view.copy().reshape(shape)
        finally:
            if got or outputs[0].buf:
                code = api.rknn_outputs_release(context,1,outputs)
                if code:
                    self.stats['cleanup_errors'].append(f'rknn_outputs_release returned {code}')
                base.check_call(code,'rknn_outputs_release')
            self.stats['sdk_io_seconds'] += time.perf_counter()-begin

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.patch is not None:
                had,value,original,forward = self.patch
                if self.head._forward_impl is forward:
                    if had:
                        self.head._forward_impl = value
                    else:
                        del self.head._forward_impl
                elif self.head._forward_impl != original:
                    self.stats['cleanup_errors'].append('Depth _forward_impl changed after installation; not overwritten')
            if self.head is not None and getattr(self.head,'_vggt_rknn_depth_owner',None) is self:
                del self.head._vggt_rknn_depth_owner
            for context in self.contexts:
                if context.value and self.api is not None:
                    try:
                        self.base.check_call(destroy_shared_context(self.api, context),'rknn_destroy')
                        self.stats['contexts_destroyed'] += 1
                    except BaseException as exc:
                        self.stats['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
                    finally:
                        context.value = 0
            if self.stats['cleanup_errors']:
                raise RuntimeError('Original depth cleanup failed: '+'; '.join(self.stats['cleanup_errors']))


def install_rknn_depth_head(model, manifest_path, manifest_sha256, **options):
    return RKNNOriginalDepthHead(model,manifest_path,manifest_sha256,**options)
