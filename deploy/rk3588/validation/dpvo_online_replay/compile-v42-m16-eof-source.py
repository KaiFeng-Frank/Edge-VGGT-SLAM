#!/usr/bin/env python3
"""Non-quantized compilation of the actual missing EOF B1/D3/M16 graph.

No model forward. Exact v37 config/load/build(False)/export calls are reused.
The CPU teacher uses saved NPU-feedback inputs; earlier RKNN rejection remains.
Host verifies graph/manifest/ancestor bytes; board absolute source pins are
declared unchanged by the CPU/ORT exporter, not rehashed on this host.
DPVO, Princeton University (2022), MIT; SDK and weights licenses are separate.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import traceback

EXPORTER_SHA256 = 'ae036a3fdfb0f9dc1c5cf0dd2dae37bd59ab313974904c20611936d42f9eca52'
V41_CONTRACTS_SHA256 = '3533d791978085216c4a28cfae45d2c8ce9e2d62a8862efbf66db02d34be5e21'
PARENT_MANIFEST_SHA256 = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
M16_CONTRACTS_SHA256 = '7be6bf750dabb8a3d88464fd1502bde3bf7690596a39791710cf41a9f8cb62fe'
PARENT_CONTRACTS_SHA256 = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
PARENT_COMPILER_SHA256 = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'
NEW_KEYS = ('B1_D3_M16',)


def _module(path, expected, name):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen compiler/contract SHA differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load_contracts(path, expected, *, v41_contracts=None, m16_contracts=None, parent_contracts=None):
    value = _module(path, expected, '_v42_EOF_compiler_contracts')
    value.load_v41_parent(v41_contracts)
    value.load_m16_parent(m16_contracts)
    value.load_parent(parent_contracts)
    return value


def compile_operations(toolkit, model, output, operations, *, parent_compiler=None):
    """Exactly the frozen four SDK operations; fake-testable, no inference."""
    path = parent_compiler or Path(__file__).with_name('compile-v37-grouped-update.py')
    parent = _module(path, PARENT_COMPILER_SHA256, '_v42_compiler_original_operations')
    return parent.compile_operations(toolkit, model, output, operations)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--parent-manifest', type=Path, required=True)
    parser.add_argument('--model-key', choices=NEW_KEYS, required=True)
    parser.add_argument('--contracts', type=Path, default=Path(__file__).with_name('v42_online_contracts.py'))
    parser.add_argument('--contracts-sha256', required=True)
    parser.add_argument('--v41-contracts', type=Path, default=Path(__file__).with_name('v41_online_contracts.py'))
    parser.add_argument('--m16-contracts', type=Path, default=Path(__file__).with_name('v40_m16_grouped_contracts.py'))
    parser.add_argument('--parent-contracts', type=Path, default=Path(__file__).with_name('v37_grouped_contracts.py'))
    parser.add_argument('--parent-compiler', type=Path, default=Path(__file__).with_name('compile-v37-grouped-update.py'))
    parser.add_argument('--diagnostic-allow-fp32-rejected', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    c = load_contracts(args.contracts, args.contracts_sha256,
                       v41_contracts=args.v41_contracts, m16_contracts=args.m16_contracts, parent_contracts=args.parent_contracts)
    parent_manifest, pins = c.load_v41_parent().validate_export_manifest(args.parent_manifest, PARENT_MANIFEST_SHA256,
                                                       verify_source_files=False)
    manifest, more = c.validate_export_manifest(args.manifest, args.manifest_sha256,
        parent_manifest=parent_manifest, exporter_sha256=EXPORTER_SHA256,
        allow_fp32_rejected_diagnostic=args.diagnostic_allow_fp32_rejected, verify_source_files=False)
    pins.update(more)
    c.require(set(manifest['models']) == set(NEW_KEYS), 'Only the actual complete EOF B1D3 source graph may compile here')
    selected = manifest['models'][args.model_key]
    model = c.artifact(args.manifest.parent, selected['model'], pins)
    parent_compiler = c.pin(args.parent_compiler, PARENT_COMPILER_SHA256, pins)
    c.pin(args.m16_contracts, M16_CONTRACTS_SHA256, pins)
    c.pin(args.parent_contracts, PARENT_CONTRACTS_SHA256, pins)
    c.pin(args.contracts, args.contracts_sha256, pins)
    c.pin(args.v41_contracts, V41_CONTRACTS_SHA256, pins)
    script = Path(__file__).resolve();c.pin(script, c.sha256(script), pins)
    out = args.output_dir.resolve()
    c.require(not out.exists() and not any(out == Path(p) or out.is_relative_to(Path(p))
              or Path(p).is_relative_to(out) for p in pins), 'New non-overlapping output directory required')
    out.mkdir(parents=True, exist_ok=False)
    temporary = out/'tmp';temporary.mkdir()
    previous_cwd = Path.cwd();os.chdir(out)
    os.environ['TMPDIR'] = str(temporary);tempfile.tempdir = str(temporary)
    report = dict(schema=c.COMPILE_SCHEMA,status='failed',script_sha256=c.sha256(script),
        contracts_sha256=args.contracts_sha256,v41_contracts_sha256=V41_CONTRACTS_SHA256,
        parent_compiler_sha256=PARENT_COMPILER_SHA256,
        parent_contracts_sha256=PARENT_CONTRACTS_SHA256,m16_contracts_sha256=M16_CONTRACTS_SHA256,
        parent_manifest_sha256=PARENT_MANIFEST_SHA256,exporter_sha256=EXPORTER_SHA256,
        manifest_sha256=args.manifest_sha256,model_key=args.model_key,
        input=dict(file=selected['model']['file'],bytes=model.stat().st_size,sha256=c.sha256(model)),
        logical_inputs=selected['inputs'],logical_outputs=selected['outputs'],configuration=c.CONFIGURATION,
        export_strict_FP32_gate_passed=manifest['strict_FP32_gate_passed'],
        teacher_reference_kind=manifest['teacher_reference_kind'],
        original_CPU_trajectory_gold_used=False,historical_SDK_strict_FP32_rejected=True,
        source_run_sha256=manifest['source_run_sha256'],source_failure_sha256=manifest['source_failure_sha256'],
        source_failed_input_sha256=manifest['source_failed_input_sha256'],
        source_before_termination_sha256=manifest['source_before_termination_sha256'],
        diagnostic_allow_fp32_rejected=args.diagnostic_allow_fp32_rejected,candidate_accepted=False,
        model_inference_executed=False,NPU_execution_or_placement_validated=False,realtime_validated=False,
        source_provenance_boundary=__doc__.strip(),export_declared_source_pins=manifest['source_pins'],
        operations=[],released=False,machine=platform.machine(),python=sys.version)
    toolkit,error = None,None
    try:
        import torch
        import numpy as np
        import onnx
        from rknn.api import RKNN
        report['versions'] = {name:importlib.metadata.version(name) for name in
            ('rknn-toolkit2','torch','numpy','onnx','onnxruntime','protobuf')}
        c.require(sys.version_info[:2] == (3,11) and torch.__version__.startswith('2.4.0')
            and np.__version__ == '1.26.4' and report['versions']['rknn-toolkit2'] == '2.3.2',
            'Use isolated CPython3.11/Torch2.4CPU/NumPy1.26.4/RKNN2.3.2')
        graph = onnx.load(model,load_external_data=False);onnx.checker.check_model(graph,full_check=True)
        c.require(all(not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL
            for tensor in graph.graph.initializer), 'External initializer data prohibited')
        def graph_io(rows):
            return [dict(name=value.name,dtype='float32' if value.type.tensor_type.elem_type == 1 else 'unsupported',
                shape=[dim.dim_value for dim in value.type.tensor_type.shape.dim]) for value in rows]
        c.require(graph_io(graph.graph.input) == selected['inputs']
            and graph_io(graph.graph.output) == selected['outputs'], 'Actual ONNX IO differs from fixed complete-source export')
        toolkit = RKNN(verbose=True)
        compile_operations(toolkit,model,out/'block.rknn',report['operations'],parent_compiler=parent_compiler)
        result = out/'block.rknn';c.require(result.is_file() and result.stat().st_size > 0,'Compiled RKNN file missing')
        report['output'] = dict(file='block.rknn',bytes=result.stat().st_size,sha256=c.sha256(result))
        report['status'] = 'compiled_group_diagnostic_not_runtime_validated'
    except BaseException as exc:
        error = exc;report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if toolkit is not None:
            try:toolkit.release();report['released'] = True
            except Exception as exc:report['release_error'] = repr(exc);error = error or exc
        try:
            report['source_pins'] = pins;report['source_hashes_after'] = c.recheck_pins(pins)
            report['sources_and_inputs_unchanged'] = True
        except Exception as exc:
            report['sources_and_inputs_unchanged'] = False;report['source_error'] = repr(exc);error = error or exc
        if error:report['status'] = 'failed'
        report['peak_RSS_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (out/'compile.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        os.chdir(previous_cwd)
    print(json.dumps({key:report.get(key) for key in
        ('status','model_key','export_strict_FP32_gate_passed','error','output')}),flush=True)
    return int(error is not None)


if __name__ == '__main__':raise SystemExit(main())
