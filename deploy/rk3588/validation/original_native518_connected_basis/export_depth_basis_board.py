#!/usr/bin/env python3
"""Board-only original native DPT basis fold and same-input FP32 logits gate.

The original 79 node protos stay unchanged. The shared LayerNorm affine
parameters are folded into its four original 1x1 projections, together with
T=diag(R1024,R1024). Inputs to the folded graph are zT; projected coordinates
are canonical. Original exp/expp1 activation remains outside both graphs.
No checkpoint, RKNN, GPU or calibration is loaded by this worker.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time
import traceback

E = Path('/mnt/vggt-data/vggt-slam2/experiments')
PARENT_EXPORT_SHA = '8e3f465242924445348a589d5b589e4176fabde9e272ad0fe7d49950886f6033'
PARENT_GRAPH_SHA = '40cb912d096397b2a086bf536b66a3c18e57ed8e392d48c4d7e1079895d2b92e'
CAPTURE_SHA = '969ab6f5611efa62d81b5d62625a15815248668e1e079db17ac4d9ed9c96932b'
ALGEBRA_SOURCE_SHA = 'cdf7e3032aaa8153e4f3ea50f5a61f595d7376eed6bb1ae5140027cbb5fbdc2c'
ALGEBRA_RECEIPT_SHA = 'e31ff44070f9ae63a6dcb7a3630dde85ea4526a2979f0fd9b67ab903bbf13ddf'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
INPUT_NAMES = ['tokens_4', 'tokens_11', 'tokens_17', 'tokens_23']
INPUT_SHAPE, OUTPUT_SHAPE = [1, 1036, 2048], [1, 2, 392, 518]
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
CHANGED = ['norm.weight', 'norm.bias'] + [f'projects.{i}.{kind}' for i in range(4) for kind in ('weight', 'bias')]
EXPORT_SUCCESS = 'exported_original_DPT_basis_consumer_not_numerically_accepted'
CHECK_SUCCESS = 'checked_original_DPT_basis_consumer_fixed_FP32_gate_passed'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def relative_path(directory, file):
    require(isinstance(file, str) and not Path(file).is_absolute() and '..' not in Path(file).parts,
            'Receipt-relative artifact required')
    return directory / file


def validate_lineage(export, capture):
    require(export['schema'] == 'v99_original_native518_depth_logits_static_export' and
            export['status'] == 'static_export_only_not_runtime_validated' and
            export['checkpoint'] == CHECKPOINT and export['all_62_original_parameter_values_preserved_in_graph'] and
            export['pinned_source_and_inputs_unchanged'] and not export['cleanup_errors'], 'Pinned original DPT export required')
    require(capture['schema'] == 'v99_original_native518_head_reference_continuation' and
            capture['status'] == 'completed_original_native518_head_reference_continuation' and
            capture['checkpoint'] == CHECKPOINT and capture['original_model_only'] and
            capture['prefix_is_original_CPU_not_NPU'] and capture['partial_reference_continuation'] and
            capture['all_requested_final_outputs_bitwise_equal_v92'] and
            capture['depth_original_parameter_identity_unchanged'] and capture['retained_tokens_unchanged'] and
            capture['prepared_pixels_unchanged'] and capture['hooks_removed'] and
            capture['pinned_source_and_inputs_unchanged'] and capture['affinity_restored'] and
            not capture['cleanup_errors'], 'Untouched original CPU reference continuation required')
    require(export['capture_sha256'] == CAPTURE_SHA and
            export['capture_script_sha256'] == capture['script_sha256'] and
            export['original_parameter_metadata'] == capture['depth_original_parameter_metadata'] and
            export['native_reference_chunks'] == capture['depth_chunks'] and
            export['source_hashes'] == capture['source_hashes'], 'Original reference and 62-parameter lineage disagree')
    architecture = export['architecture']
    require(architecture['input_names'] == INPUT_NAMES and architecture['per_frame_input_shape'] == INPUT_SHAPE and
            architecture['per_frame_raw_logits_shape'] == OUTPUT_SHAPE and architecture['norm_epsilon'] == 1e-5 and
            architecture['activation'] == 'exp' and architecture['conf_activation'] == 'expp1' and
            architecture['frames_chunk_size'] == 1 and architecture['align_corners'] and
            architecture['original_inplace_ReLU_residual_alias_preserved'], 'Original native DPT architecture differs')
    require(len(export['original_parameter_metadata']) == 62 and len(export['native_position_metadata']) == 5 and
            export['graph']['sha256'] == PARENT_GRAPH_SHA and export['graph']['bytes'] == 246272013 and
            export['graph']['node_count'] == 79 and export['graph']['initializer_count'] == 76,
            'Complete original DPT graph identity differs')


def validate_algebra(row, source_sha):
    require(row['schema'] == 'generated_basis_consumer_algebra_v130_design_check' and
            row['status'] == 'generated_algebra_passed_original_graph_not_exported_or_accepted' and
            row['script_sha256'] == source_sha and row['orthogonality_symmetry_and_mean_preservation_exact'] and
            row['original_model_values_or_saved_captures_loaded'] is False and row['NN_or_SDK_executed'] is False and
            row['full_model_speed_quality_or_realtime_accepted'] is False and
            {v['epsilon'] for v in row['DPT_projection_checks']} == {1e-6, 1e-5}, 'Generated consumer proof differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['export', 'check'], required=True)
    p.add_argument('--parent-export', type=Path, default=E / 'vggt-v99-depth-export-r1/export.json')
    p.add_argument('--onnx', type=Path, default=E / 'vggt-v99-depth-export-r1/original-depth-logits-native518.onnx')
    p.add_argument('--head-reference', type=Path, default=E / 'vggt-v99-head-reference-r1/capture.json')
    p.add_argument('--algebra-source', type=Path, required=True)
    p.add_argument('--algebra-source-sha256', required=True)
    p.add_argument('--algebra-receipt', type=Path, required=True)
    p.add_argument('--algebra-receipt-sha256', required=True)
    p.add_argument('--export-receipt', type=Path)
    p.add_argument('--export-sha256')
    p.add_argument('--frame-index', type=int, choices=[0, 1])
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'Board-only worker; host original-model export or inference forbidden')
    require(a.mode == 'export' or (a.export_receipt is not None and a.export_sha256 is not None and a.frame_index is not None),
            'Check requires export receipt/digest and frame index')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    old_env, old_affinity = dict(os.environ), set(os.sched_getaffinity(0))
    watched, handlers, session = {}, {}, None
    started = time.monotonic()
    report = dict(schema='v130_original_native518_DPT_basis_consumer_' + a.mode, mode=a.mode, status='running',
        script_sha256=sha(__file__), original_model_only=True, actual_execution_on_board_only=True,
        checkpoint_deserialized=False, RKNN_GPU_calibration_or_other_model_executed=False, candidate_accepted=False,
        full_model_quality_performance_realtime_or_8GB_accepted=False, system_or_frequency_writes=False,
        checkpoint=CHECKPOINT, fixed_gate=GATE,
        fixed_gate_definition='relative_l2 <= 1e-5 and max_abs <= 1e-3; strict allclose is an independent diagnostic',
        strict_allclose_is_diagnostic=True, original_exp_expp1_activation_remains_outside_graph=True,
        reference_scope='Original two native DPT frame chunks from the bitwise-v92 partial original CPU continuation; raw logits only',
        maximum_simultaneous_ORT_sessions=1, sessions_created=0, sessions_dropped=0, calls_completed=0,
        caller_owns_CPU_thermal_and_timeout_controller=True, timings_are_not_NPU_or_full_model_performance=True,
        outputs=[], comparisons={}, cleanup_errors=[])

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        atomic(out / 'result.json', report)
        print(stage, flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(len(digest) == 64 and sha(path) == digest and (size is None or path.stat().st_size == size),
                'Pinned file differs: ' + str(path))
        require(path not in watched or watched[path] == digest, 'Conflicting pinned identity')
        watched[path] = digest
        return path

    def raw_meta(value):
        return dict(shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
                    raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(), finite=bool(np.isfinite(value).all()))

    def file_meta(path):
        return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))

    def tensor_equal(value, expected):
        found = raw_meta(value)
        require(all(found[key] == expected[key] for key in ('shape', 'dtype', 'tensor_bytes', 'raw_sha256')) and found['finite'],
                'Original tensor identity differs')
        return found

    def basis_right(value):
        # T=diag(R1024,R1024); no dense 2048x2048 matrix is materialized.
        require(value.shape[-1] == 2048, 'Original combined feature dimension required')
        shaped = np.array(value, dtype=np.float64, copy=True, order='C').reshape(value.shape[:-1] + (2, 4, 4, 4, 4, 4))
        for axis in range(shaped.ndim - 5, shaped.ndim):
            shaped = .5 * np.sum(shaped, axis=axis, keepdims=True) - shaped
        return np.ascontiguousarray(shaped.reshape(value.shape), dtype=np.float32)

    def compare(actual, expected):
        x, y = actual.astype(np.float64), expected.astype(np.float64)
        delta = x - y
        norm = float(np.linalg.norm(y.reshape(-1)))
        rel = float(np.linalg.norm(delta.reshape(-1)) / norm) if norm > 0 else None
        maximum = float(np.max(np.abs(delta)))
        outside = int(np.count_nonzero(np.abs(delta) > GATE['atol'] + GATE['rtol'] * np.abs(y)))
        passed = rel is not None and rel <= GATE['relative_l2_max'] and maximum <= GATE['max_abs_max']
        return dict(relative_l2=rel, reference_norm=norm, zero_reference_norm=norm == 0, max_abs=maximum,
                    outside_count=outside, strict_pass=outside == 0, fixed_numeric_gate_passed=passed,
                    fixed_gate_passed=passed, byte_equal=actual.tobytes() == expected.tobytes())

    def save_array(name, value):
        path = out / name
        np.save(path, value, allow_pickle=False)
        return dict(**file_meta(path), **raw_meta(value))

    def verify_ports(model):
        def binding(port):
            t = port.type.tensor_type
            return dict(name=port.name, dtype='float32' if t.elem_type == 1 else str(t.elem_type),
                        shape=[v.dim_value for v in t.shape.dim])
        require([binding(v) for v in model.graph.input] == [dict(name=n, dtype='float32', shape=INPUT_SHAPE) for n in INPUT_NAMES] and
                [binding(v) for v in model.graph.output] == [dict(name='logits', dtype='float32', shape=OUTPUT_SHAPE)], 'Original logical bindings differ')
        require(len(model.graph.node) == 79 and len(model.graph.initializer) == 76 and model.ir_version == 8 and
                [(v.domain, v.version) for v in model.opset_import] == [('', 17)], 'Original graph dimensions/opset differ')

    def verify_parent(model, export):
        verify_ports(model)
        tensors = {v.name: v for v in model.graph.initializer}
        require(len(tensors) == 76 and all(not v.external_data for v in tensors.values()), 'Self-contained unique original initializers required')
        for name, meta in export['original_parameter_metadata'].items():
            tensor_equal(numpy_helper.to_array(tensors[name]), meta)
        for name, meta in export['native_position_metadata'].items():
            tensor_equal(numpy_helper.to_array(tensors['position.' + name]), meta)
        nodes = {v.name: v for v in model.graph.node}
        for i, token in enumerate(INPUT_NAMES):
            ln = nodes[f'branch{i}.norm']
            attrs = {v.name: helper.get_attribute_value(v) for v in ln.attribute}
            require(ln.op_type == 'LayerNormalization' and list(ln.input) == [token, 'norm.weight', 'norm.bias'] and
                    attrs['axis'] == -1 and abs(attrs['epsilon'] - 1e-5) < 1e-12 and attrs['stash_type'] == 1,
                    'Original LayerNorm axis/epsilon/affine consumers differ')
            conv = nodes[f'projects.{i}']
            require(conv.op_type == 'Conv' and list(conv.input)[1:] == [f'projects.{i}.weight', f'projects.{i}.bias'],
                    'Original projection connections differ')
        return tensors

    def structural_proof(parent, candidate):
        verify_ports(candidate)
        require(len(parent.graph.node) == len(candidate.graph.node) and
                all(x.SerializeToString() == y.SerializeToString() for x, y in zip(parent.graph.node, candidate.graph.node)),
                'An original node proto changed')
        original = {v.name: v for v in parent.graph.initializer}
        require([v.name for v in parent.graph.initializer] == [v.name for v in candidate.graph.initializer], 'Initializer order or coverage changed')
        for v in candidate.graph.initializer:
            if v.name not in CHANGED:
                require(v.SerializeToString() == original[v.name].SerializeToString(), 'Untouched initializer proto changed: ' + v.name)
        inverse = copy.deepcopy(candidate)
        for v in inverse.graph.initializer:
            if v.name in CHANGED:
                v.CopyFrom(original[v.name])
        parent_bytes = parent.SerializeToString()
        require(inverse.SerializeToString() == parent_bytes, 'Whole-parent inverse protobuf proof failed')
        result = dict(all79_original_node_protos_byte_identical=True, all66_untouched_initializer_protos_byte_identical=True,
            all52_untouched_original_learned_tensors_byte_identical=True, all5_original_positions_byte_identical=True,
            all_original_resize_relu_alias_and_fusion_nodes_byte_identical=True, original_initializer_order_preserved=True,
            inverse_entire_model_proto_byte_equal=True, inverse_model_sha256=hashlib.sha256(parent_bytes).hexdigest(),
            changed_initializer_names=CHANGED, node_count=79, initializer_count=76)
        del inverse, parent_bytes
        return result

    progress('preflight')
    try:
        def interrupted(number, frame):
            raise InterruptedError('Controller termination signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.signal(number, interrupted)
        pin(__file__, report['script_sha256'])
        parent_receipt = pin(a.parent_export, PARENT_EXPORT_SHA)
        capture_file = pin(a.head_reference, CAPTURE_SHA)
        parent_file = pin(a.onnx, PARENT_GRAPH_SHA, 246272013)
        export, capture = json.loads(parent_receipt.read_text()), json.loads(capture_file.read_text())
        validate_lineage(export, capture)
        require(a.algebra_source_sha256 == ALGEBRA_SOURCE_SHA and a.algebra_receipt_sha256 == ALGEBRA_RECEIPT_SHA,
                'Reviewed generated consumer proof pins required')
        algebra_source = pin(a.algebra_source, a.algebra_source_sha256)
        algebra_file = pin(a.algebra_receipt, a.algebra_receipt_sha256)
        validate_algebra(json.loads(algebra_file.read_text()), a.algebra_source_sha256)
        report.update(parent_export=dict(file=str(parent_receipt), sha256=PARENT_EXPORT_SHA),
            parent_graph=dict(file=str(parent_file), bytes=246272013, sha256=PARENT_GRAPH_SHA),
            head_reference=dict(file=str(capture_file), sha256=CAPTURE_SHA),
            algebra_source=dict(file=str(algebra_source), sha256=a.algebra_source_sha256),
            algebra_receipt=dict(file=str(algebra_file), sha256=a.algebra_receipt_sha256),
            original_parameter_metadata=export['original_parameter_metadata'], source_hashes=export['source_hashes'],
            native_position_metadata=export['native_position_metadata'], architecture=export['architecture'],
            original62_parameter_and_checkpoint_source_lineage_verified=True,
            basis=dict(name='T=diag(R1024,R1024)', R='((J4-2I4)/2) tensor-power 5',
                input_convention='row vectors u=zT', projection_weight_fold='W_fold=(W_original*gamma)T',
                projection_bias_fold='b_fold=W_original beta+b_original', transform_dtype='float64 then float32',
                factor_order='five base-4 feature axes in increasing axis order', full_inverse_input_operator_added=False))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.sched_setaffinity(0, old_affinity & {4, 5, 6, 7})
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        report['versions'] = dict(numpy=np.__version__, onnx=onnx.__version__, python=platform.python_version())
        parent = onnx.load(str(parent_file), load_external_data=False)
        original = verify_parent(parent, export)
        report['all62_original_learned_values_verified'] = True
        progress('original_graph_and_parameters_verified')
        if a.mode == 'export':
            require(shutil.disk_usage(out).free >= 384 << 20, 'At least 384 MiB free required for one complete graph and receipt')
            candidate = copy.deepcopy(parent)
            gamma = numpy_helper.to_array(original['norm.weight']).astype(np.float64)
            beta = numpy_helper.to_array(original['norm.bias']).astype(np.float64)
            replacements = {'norm.weight': np.ones(2048, dtype=np.float32), 'norm.bias': np.zeros(2048, dtype=np.float32)}
            folds = []
            for i in range(4):
                wn, bn = f'projects.{i}.weight', f'projects.{i}.bias'
                weight = numpy_helper.to_array(original[wn]).astype(np.float64)
                bias = numpy_helper.to_array(original[bn]).astype(np.float64)
                matrix = weight[:, :, 0, 0]
                replacements[wn] = basis_right(matrix * gamma).reshape(weight.shape)
                replacements[bn] = np.ascontiguousarray(matrix @ beta + bias, dtype=np.float32)
                folds.append(dict(project=i, source_weight=export['original_parameter_metadata'][wn],
                    source_bias=export['original_parameter_metadata'][bn], derived_weight=raw_meta(replacements[wn]),
                    derived_bias=raw_meta(replacements[bn]), original_beta_times_original_weight=True))
                del weight, bias, matrix
            for value in candidate.graph.initializer:
                if value.name in replacements:
                    value.CopyFrom(numpy_helper.from_array(replacements[value.name], value.name))
            report['proof'] = structural_proof(parent, candidate)
            report['folds'] = folds
            report['derived_initializer_metadata'] = {name: raw_meta(value) for name, value in replacements.items()}
            onnx.checker.check_model(candidate, full_check=True)
            destination = out / 'original-depth-logits-native518-basis.onnx'
            onnx.save(candidate, str(destination))
            report['graph'] = dict(**file_meta(destination), input_bindings=[dict(index=i, name=n, shape=INPUT_SHAPE, dtype='float32')
                for i, n in enumerate(INPUT_NAMES)], output_bindings=[dict(index=0, name='logits', shape=OUTPUT_SHAPE, dtype='float32')],
                input_coordinates='basis T', output_coordinates='canonical raw logits', node_count=79, initializer_count=76)
            report['status'] = EXPORT_SUCCESS
            del parent, original, candidate, replacements, gamma, beta
        else:
            folded_receipt = pin(a.export_receipt, a.export_sha256)
            folded = json.loads(folded_receipt.read_text())
            require(folded['schema'] == 'v130_original_native518_DPT_basis_consumer_export' and folded['status'] == EXPORT_SUCCESS and
                    folded['script_sha256'] == report['script_sha256'] and folded['checkpoint'] == CHECKPOINT and
                    folded['all62_original_learned_values_verified'] and folded['original62_parameter_and_checkpoint_source_lineage_verified'] and
                    folded['source_hashes'] == export['source_hashes'] and folded['original_parameter_metadata'] == export['original_parameter_metadata'] and
                    folded['parent_export']['sha256'] == PARENT_EXPORT_SHA and folded['head_reference']['sha256'] == CAPTURE_SHA and
                    folded['algebra_source']['sha256'] == ALGEBRA_SOURCE_SHA and folded['algebra_receipt']['sha256'] == ALGEBRA_RECEIPT_SHA and
                    folded['source_and_inputs_unchanged'] and folded['environment_restored'] and folded['affinity_restored'] and
                    not folded['cleanup_errors'], 'Completed same-source folded export required')
            graph_row = folded['graph']
            folded_file = pin(relative_path(folded_receipt.parent, graph_row['file']), graph_row['sha256'], graph_row['bytes'])
            candidate = onnx.load(str(folded_file), load_external_data=False)
            proof = structural_proof(parent, candidate)
            require(proof == folded['proof'] and set(folded['derived_initializer_metadata']) == set(CHANGED), 'Export structural proof differs')
            for value in candidate.graph.initializer:
                if value.name in CHANGED:
                    tensor_equal(numpy_helper.to_array(value), folded['derived_initializer_metadata'][value.name])
            report.update(export_receipt=dict(file=str(folded_receipt), sha256=a.export_sha256), graph=graph_row, proof=proof, frame_index=a.frame_index)
            del parent, original, candidate
            gc.collect()
            chunks = [v for v in capture['depth_chunks'] if v['frame_index'] == a.frame_index]
            require(len(chunks) == 1 and set(chunks[0]['inputs']) == set(INPUT_NAMES), 'Original frame chunk required')
            chunk = chunks[0]
            canonical, input_records = {}, {}
            for name in INPUT_NAMES:
                meta = chunk['inputs'][name]
                path = pin(relative_path(capture_file.parent, meta['file']), meta['sha256'], meta['bytes'])
                value = np.load(path, allow_pickle=False)
                require(list(value.shape) == INPUT_SHAPE and value.dtype == np.float32 and value.flags.c_contiguous, 'Original native patch tokens differ')
                tensor_equal(value, meta)
                canonical[name], input_records[name] = value, meta
            reference_meta = chunk['raw_logits']
            reference_path = pin(relative_path(capture_file.parent, reference_meta['file']), reference_meta['sha256'], reference_meta['bytes'])
            reference = np.load(reference_path, allow_pickle=False)
            tensor_equal(reference, reference_meta)
            require(list(reference.shape) == OUTPUT_SHAPE and reference.dtype == np.float32, 'Original raw-logit reference differs')
            reference_copy = out / 'original-reference-logits.npy'
            shutil.copyfile(reference_path, reference_copy)
            require(sha(reference_copy) == reference_meta['sha256'], 'Original reference copy differs')
            report['outputs'].append(dict(route='original_reference', **file_meta(reference_copy), **raw_meta(reference)))
            report['input_artifacts'] = input_records
            rotated = {name: basis_right(value) for name, value in canonical.items()}
            report['rotated_input_metadata'] = {name: raw_meta(value) for name, value in rotated.items()}
            import onnxruntime as ort
            report['versions']['onnxruntime'] = ort.__version__
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            values = {}
            for route, path, inputs in [('baseline', parent_file, canonical), ('folded', folded_file, rotated)]:
                progress('starting_' + route + '_original_DPT_logits')
                before = time.monotonic()
                session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                report['sessions_created'] += 1
                require(session.get_providers() == ['CPUExecutionProvider'], 'CPU-only ONNX provider required')
                require([(v.name, v.shape, v.type) for v in session.get_inputs()] == [(n, INPUT_SHAPE, 'tensor(float)') for n in INPUT_NAMES] and
                        [(v.name, v.shape, v.type) for v in session.get_outputs()] == [('logits', OUTPUT_SHAPE, 'tensor(float)')], 'Runtime canonical bindings differ')
                value = np.array(session.run(['logits'], inputs)[0], dtype=np.float32, copy=True, order='C')
                report['calls_completed'] += 1
                record = save_array(route + '-logits.npy', value)
                report['outputs'].append(dict(route=route, CPU_session_forward_and_save_seconds=time.monotonic() - before, **record))
                values[route] = value
                session = None
                gc.collect()
                report['sessions_dropped'] += 1
                progress('saved_' + route + '_logits')
                require(record['shape'] == OUTPUT_SHAPE and record['finite'], 'Nonfinite or changed-shape original DPT logits; raw array preserved')
            report['comparisons'] = dict(folded_vs_baseline=compare(values['folded'], values['baseline']),
                baseline_vs_original_Torch_capture=compare(values['baseline'], reference),
                folded_vs_original_Torch_capture=compare(values['folded'], reference))
            report['FP32_gate_passed'] = report['comparisons']['folded_vs_baseline']['fixed_numeric_gate_passed']
            report['main_gate'] = 'folded raw logits vs baseline raw logits on the exact same original canonical input views'
            report['matched_Torch_comparisons_are_reported_separately'] = True
            progress('all_saved_raw_logits_compared')
            require(report['FP32_gate_passed'], 'Original fixed numeric DPT basis-consumer gate failed; all outputs preserved')
            require(report['sessions_created'] == report['sessions_dropped'] == report['calls_completed'] == 2, 'Sequential original consumer resource counts differ')
            report['status'] = CHECK_SUCCESS
    except BaseException as error:
        report.update(status='failed_preserved_available_evidence', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        raise
    finally:
        if session is not None:
            session = None
            gc.collect()
            report['sessions_dropped'] += 1
        report['ORT_sessions_dropped'] = report['sessions_created'] == report['sessions_dropped']
        for number, handler in handlers.items():
            signal.signal(number, handler)
        try:
            os.sched_setaffinity(0, old_affinity)
            report['affinity_restored'] = set(os.sched_getaffinity(0)) == old_affinity
        except BaseException as error:
            report['cleanup_errors'].append('affinity: ' + repr(error))
        os.environ.clear()
        os.environ.update(old_env)
        report['environment_restored'] = dict(os.environ) == old_env
        try:
            report['source_and_inputs_unchanged'] = all(sha(path) == digest for path, digest in watched.items())
            require(report['source_and_inputs_unchanged'], 'Pinned source/input/graph changed during stage')
        except BaseException as error:
            report['cleanup_errors'].append('pinned identity: ' + repr(error))
        if report['cleanup_errors']:
            report['status'] = 'failed_cleanup_or_pinned_identity'
        progress('terminal')
        if report['cleanup_errors']:
            raise RuntimeError('Cleanup or pinned identity failed: ' + repr(report['cleanup_errors']))


if __name__ == '__main__':
    main()
