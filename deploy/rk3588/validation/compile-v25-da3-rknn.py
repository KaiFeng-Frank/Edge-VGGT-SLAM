#!/usr/bin/env python3
"""Compile unchanged DA3 ONNX while retaining camera-token slicing before transpose.

The previous two Gather rule exclusions remain. v24's actual profile showed
swap_slice_transpose moving a camera-token slice after a CPU transpose of
all 433 tokens. This variant disables only that additional named rule.
No graph, weights, dtype or numerical threshold changes; no inference here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


DISABLE_RULES = ('gather_to_conv', 'fuse_channel_shuffle_split_concat_to_gather', 'swap_slice_transpose')
FROZEN_V24_R2_REFERENCE_SHA256 = '2ad1069c8d5563051120a2080bc3f79499df81d94911edb316d83b61b994f0d6'
FROZEN_V24_R1_REFERENCE_SHA256 = 'e9c66360e2de65f173682650ee8536e321acbc0698f895172d69a15cddb44a91'
FROZEN_V12_REFERENCE_SHA256 = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--onnx-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(argv)
    if sha(a.onnx) != a.onnx_sha256:
        p.error('ONNX SHA256 mismatch')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='starting', model_inference_executed=False,
                  production_changed=False, realtime_validated=False,
                  input=dict(path=str(a.onnx.resolve()), sha256=sha(a.onnx)),
                  script_sha256=sha(__file__), machine=platform.machine(),
                  python=sys.version, configuration=dict(target_platform='rk3588',
                  float_dtype='float16', optimization_level=3, do_quantization=False,
                  disable_rules=list(DISABLE_RULES)),
                  compiler_change='retain v24 two Gather exclusions and additionally disable observed swap_slice_transpose',
                  frozen_v24_r2_reference_sha256=FROZEN_V24_R2_REFERENCE_SHA256,
                  frozen_v24_r1_reference_sha256=FROZEN_V24_R1_REFERENCE_SHA256,
                  frozen_v12_reference_sha256=FROZEN_V12_REFERENCE_SHA256,
                  operations=[])
    toolkit = None
    error = None
    try:
        import torch
        import numpy as np
        import onnx
        from rknn.api import RKNN
        report['versions'] = {n: importlib.metadata.version(n) for n in
                              ('rknn-toolkit2', 'torch', 'numpy', 'onnx', 'onnxruntime', 'protobuf')}
        if not torch.__version__.startswith('2.4.0') or np.__version__ != '1.26.4':
            raise RuntimeError('Use isolated Torch 2.4.0 and NumPy 1.26.4 compiler environment')
        if report['versions']['rknn-toolkit2'] != '2.3.2':
            raise RuntimeError('Expected RKNN compiler 2.3.2')
        onnx.checker.check_model(str(a.onnx), full_check=True)
        toolkit = RKNN(verbose=True)
        calls = [
            ('config', lambda: toolkit.config(target_platform='rk3588', float_dtype='float16', optimization_level=3, disable_rules=list(DISABLE_RULES))),
            ('load_onnx', lambda: toolkit.load_onnx(model=str(a.onnx.resolve()))),
            ('build', lambda: toolkit.build(do_quantization=False)),
            ('export_rknn', lambda: toolkit.export_rknn(str(a.output_dir.resolve() / 'block.rknn'))),
        ]
        for name, call in calls:
            print('START', name, flush=True)
            t = time.monotonic()
            code = call()
            report['operations'].append(dict(name=name, code=code, seconds=time.monotonic()-t))
            if code not in (0, None):
                raise RuntimeError(f'{name} returned {code}')
        output = a.output_dir / 'block.rknn'
        report['output'] = dict(path=str(output.resolve()), size=output.stat().st_size, sha256=sha(output))
        report['status'] = 'compiled_not_runtime_validated'
    except BaseException as exc:
        error = exc
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        if toolkit is not None:
            try:
                toolkit.release()
                report['released'] = True
            except Exception as exc:
                report['release_error'] = repr(exc)
                error = error or exc
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (a.output_dir / 'compile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
