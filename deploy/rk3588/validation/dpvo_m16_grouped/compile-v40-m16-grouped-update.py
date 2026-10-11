#!/usr/bin/env python3
"""Compile one pinned actual M16 grouped Update graph, without any forward.

Uses the exact v37 RKNN config/load/build(False)/export API sequence. Compiler
and actual hardware jobs are root-owned. A compiled file never implies runtime
placement, FP16 quality, recurrent VO acceptance or a default model change.
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
import time
import traceback

CONTRACTS_SHA256 = '7be6bf750dabb8a3d88464fd1502bde3bf7690596a39791710cf41a9f8cb62fe'
PARENT_COMPILER_SHA256 = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'


def _module(path, expected, name):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen compiler/contract ancestor SHA differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_contracts(path, parent_path=None):
    module = _module(path, CONTRACTS_SHA256, '_v40_M16_compiler_contracts')
    module.load_parent(parent_path)
    return module


def compile_operations(toolkit, model, output, operations, *, parent_compiler=None):
    """Delegate unchanged four operations to the frozen parent; fake-testable."""
    path = parent_compiler or Path(__file__).with_name('compile-v37-grouped-update.py')
    parent = _module(path, PARENT_COMPILER_SHA256, '_v40_M16_compiler_ancestor')
    return parent.compile_operations(toolkit, model, output, operations)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--contracts', type=Path, default=Path(__file__).with_name('v40_m16_grouped_contracts.py'))
    parser.add_argument('--parent-contracts', type=Path, default=Path(__file__).with_name('v37_grouped_contracts.py'))
    parser.add_argument('--parent-compiler', type=Path, default=Path(__file__).with_name('compile-v37-grouped-update.py'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    c = load_contracts(args.contracts, args.parent_contracts)
    manifest, pins = c.validate_export_manifest(args.manifest, args.manifest_sha256, verify_source_files=False)
    c.require(args.model_key in manifest['models'], 'Unknown actual M16 shape key')
    selected = manifest['models'][args.model_key]
    model = c.artifact(args.manifest.parent, selected['model'], pins)
    script = Path(__file__).resolve()
    parent_compiler = c.pin(args.parent_compiler, PARENT_COMPILER_SHA256, pins)
    c.pin(c.parent_contracts_path(), c.PARENT_CONTRACTS_SHA256, pins)
    c.pin(args.contracts, CONTRACTS_SHA256, pins)
    c.pin(script, c.sha256(script), pins)
    out = args.output_dir.resolve()
    c.require(not out.exists() and not any(out == Path(p) or out.is_relative_to(Path(p))
              or Path(p).is_relative_to(out) for p in pins), 'New non-overlapping output directory required')
    out.mkdir(parents=True, exist_ok=False)
    temporary = out/'tmp'; temporary.mkdir()
    previous_cwd = Path.cwd()
    os.chdir(out)  # RKNN debug files stay beside this experiment, away from root cwd.
    os.environ['TMPDIR'] = str(temporary)
    tempfile.tempdir = str(temporary)
    report = dict(schema=c.COMPILE_SCHEMA, status='failed', script_sha256=c.sha256(script),
        contracts_sha256=CONTRACTS_SHA256, parent_compiler_sha256=PARENT_COMPILER_SHA256,
        parent_contracts_sha256=c.PARENT_CONTRACTS_SHA256,
        manifest_sha256=args.manifest_sha256, model_key=args.model_key,
        input=dict(file=selected['model']['file'], bytes=model.stat().st_size, sha256=c.sha256(model)),
        logical_inputs=selected['inputs'], logical_outputs=selected['outputs'], configuration=c.CONFIGURATION,
        export_strict_FP32_gate_passed=manifest['strict_FP32_gate_passed'],
        diagnostic_allow_fp32_rejected=False, candidate_accepted=False,
        model_inference_executed=False, NPU_execution_or_placement_validated=False, realtime_validated=False,
        source_provenance_boundary='Host verifies exact v39 manifest/selected ONNX/new compiler/contract and frozen compiler/contract ancestors. Board absolute source files are declared unchanged by the accepted CPU/ORT export, not rehashed on this host.',
        export_declared_source_pins=manifest['source_pins'], operations=[], released=False,
        machine=platform.machine(), python=sys.version)
    toolkit, error = None, None
    try:
        import torch
        import numpy as np
        import onnx
        from rknn.api import RKNN
        report['versions'] = {name: importlib.metadata.version(name) for name in
            ('rknn-toolkit2', 'torch', 'numpy', 'onnx', 'onnxruntime', 'protobuf')}
        c.require(sys.version_info[:2] == (3, 11) and torch.__version__.startswith('2.4.0')
                  and np.__version__ == '1.26.4' and report['versions']['rknn-toolkit2'] == '2.3.2',
                  'Use isolated CPython3.11/Torch2.4CPU/NumPy1.26.4/RKNN2.3.2')
        graph = onnx.load(model, load_external_data=False)
        onnx.checker.check_model(graph, full_check=True)
        c.require(all(not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL
                      for tensor in graph.graph.initializer), 'External initializer data prohibited')
        def graph_io(rows):
            return [dict(name=value.name, dtype='float32' if value.type.tensor_type.elem_type == 1 else 'unsupported',
                         shape=[dim.dim_value for dim in value.type.tensor_type.shape.dim]) for value in rows]
        c.require(graph_io(graph.graph.input) == selected['inputs'] and graph_io(graph.graph.output) == selected['outputs'],
                  'Actual ONNX IO differs from fixed M16 export contract')
        toolkit = RKNN(verbose=True)
        compile_operations(toolkit, model, out/'block.rknn', report['operations'], parent_compiler=parent_compiler)
        result = out/'block.rknn'
        c.require(result.is_file() and result.stat().st_size > 0, 'Compiled RKNN file missing')
        report['output'] = dict(file='block.rknn', bytes=result.stat().st_size, sha256=c.sha256(result))
        report['status'] = 'compiled_group_diagnostic_not_runtime_validated'
    except BaseException as exc:
        error = exc
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        if toolkit is not None:
            try:
                toolkit.release()
                report['released'] = True
            except Exception as exc:
                report['release_error'] = repr(exc); error = error or exc
        try:
            report['source_pins'] = pins
            report['source_hashes_after'] = c.recheck_pins(pins)
            report['sources_and_inputs_unchanged'] = True
        except Exception as exc:
            report['sources_and_inputs_unchanged'] = False
            report['source_error'] = repr(exc); error = error or exc
        if error:
            report['status'] = 'failed'
        report['peak_RSS_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (out/'compile.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        os.chdir(previous_cwd)
    print(json.dumps({key: report.get(key) for key in ('status', 'model_key', 'export_strict_FP32_gate_passed', 'error', 'output')}), flush=True)
    return int(error is not None)


if __name__ == '__main__':
    raise SystemExit(main())
