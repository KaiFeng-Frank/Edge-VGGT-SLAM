#!/usr/bin/env python3
"""Static RKNN compilation of one complete true-B M48 Update diagnostic graph.

Only root runs compilation. No model inference. Rejected CPU/ORT ancestry must
be explicitly authorized as a diagnostic; compilation never adopts a model.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
CONTRACTS_SHA256 = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'


def load_contracts(path):
    import hashlib
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != CONTRACTS_SHA256:
        raise ValueError('Frozen grouped contract SHA differs')
    spec = importlib.util.spec_from_file_location('_v37_compiler_contracts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_operations(toolkit, model, output, operations):
    """Fake-testable API sequence; never performs a forward."""
    calls = [('config', lambda: toolkit.config(target_platform='rk3588', float_dtype='float16', optimization_level=3)),
             ('load_onnx', lambda: toolkit.load_onnx(model=str(model))),
             ('build', lambda: toolkit.build(do_quantization=False)),
             ('export_rknn', lambda: toolkit.export_rknn(str(output)))]
    for name, call in calls:
        begin = time.monotonic()
        code = call()
        operations.append(dict(name=name, code=code, seconds=time.monotonic()-begin))
        if code not in (0, None):
            raise RuntimeError(name+' returned '+str(code))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--contracts', type=Path, default=Path(__file__).with_name('v37_grouped_contracts.py'))
    parser.add_argument('--diagnostic-allow-fp32-rejected', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    c = load_contracts(args.contracts)
    manifest, pins = c.validate_export_manifest(args.manifest, args.manifest_sha256,
        allow_fp32_rejected_diagnostic=args.diagnostic_allow_fp32_rejected, verify_source_files=False)
    c.require(args.model_key in manifest['models'], 'Unknown actual grouped shape key')
    selected = manifest['models'][args.model_key]
    model = c.artifact(args.manifest.parent, selected['model'], pins)
    c.pin(args.contracts, CONTRACTS_SHA256, pins)
    c.pin(__file__, c.sha256(__file__), pins)
    out = args.output_dir.resolve()
    c.require(not out.exists() and not any(out == Path(p) or out.is_relative_to(Path(p))
              or Path(p).is_relative_to(out) for p in pins), 'New non-overlapping output directory required')
    out.mkdir(parents=True, exist_ok=False)
    temporary = out/'tmp'; temporary.mkdir()
    previous_cwd = Path.cwd()
    os.chdir(out)  # RKNN debug intermediates must not land on the full root disk.
    os.environ['TMPDIR'] = str(temporary)
    tempfile.tempdir = str(temporary)
    report = dict(schema=c.COMPILE_SCHEMA, status='failed', script_sha256=c.sha256(__file__),
        contracts_sha256=CONTRACTS_SHA256, manifest_sha256=args.manifest_sha256, model_key=args.model_key,
        input=dict(file=selected['model']['file'], bytes=model.stat().st_size, sha256=c.sha256(model)),
        logical_inputs=selected['inputs'], logical_outputs=selected['outputs'], configuration=c.CONFIGURATION,
        export_strict_FP32_gate_passed=manifest['strict_FP32_gate_passed'],
        diagnostic_allow_fp32_rejected=args.diagnostic_allow_fp32_rejected, candidate_accepted=False,
        model_inference_executed=False, NPU_execution_or_placement_validated=False, realtime_validated=False,
        source_provenance_boundary='Host verifies exact manifest/selected ONNX/compiler/contract bytes. Board absolute source files are declared unchanged by the frozen export, not rehashed on this host.',
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
        c.require(all(not tensor.external_data for tensor in graph.graph.initializer), 'External initializer data prohibited')
        def graph_io(rows):
            return [dict(name=value.name, dtype='float32' if value.type.tensor_type.elem_type == 1 else 'unsupported',
                         shape=[dim.dim_value for dim in value.type.tensor_type.shape.dim]) for value in rows]
        c.require(graph_io(graph.graph.input) == selected['inputs'] and graph_io(graph.graph.output) == selected['outputs'],
                  'Actual ONNX IO differs from frozen export contract')
        toolkit = RKNN(verbose=True)
        compile_operations(toolkit, model, out/'block.rknn', report['operations'])
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
