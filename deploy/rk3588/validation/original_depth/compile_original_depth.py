#!/usr/bin/env python3
"""Compile the original fixed DPT logits graph on RK3588 without quantization."""
import argparse
import json
import os
from pathlib import Path
import time

import depth_contract as C


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        parser.error('RK3588 board execution required')
    if C.sha256(args.manifest) != args.manifest_sha256:
        raise ValueError('Original depth export manifest changed')
    root = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text())
    if (manifest['schema'] != 'v54_original_VGGT_depth_logits_ONNX_manifest'
            or manifest['checkpoint'] != C.CHECKPOINT):
        raise ValueError('Complete original depth logits export required')
    row = manifest['graph']
    source = (root / row['file']).resolve(strict=True)
    if (not source.is_relative_to(root) or source.stat().st_size != row['bytes']
            or C.sha256(source) != row['sha256']):
        raise ValueError('Original ONNX identity mismatch')
    os.sched_setaffinity(0, {4, 5, 6, 7})
    folder = root / 'compiled'
    folder.mkdir(exist_ok=False)
    os.chdir(folder)
    os.environ['TMPDIR'] = str(folder)
    from rknn.api import RKNN
    config = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
                  mean_values=None, std_values=None, quant_img_RGB2BGR=False,
                  model_pruning=False, remove_reshape=False, quantize_weight=False,
                  enable_flash_attention=False)
    report = dict(status='running', configuration=config, operations=[],
                  export_manifest_sha256=args.manifest_sha256,
                  compile_source_sha256=C.sha256(__file__))
    api = RKNN(verbose=False, verbose_file=str(folder / 'sdk.log'))
    begin = time.monotonic()
    try:
        for name, fn in (
            ('config', lambda: api.config(**config)),
            ('load_onnx', lambda: api.load_onnx(model=str(source))),
            ('build', lambda: api.build(do_quantization=False, dataset=None, auto_hybrid=False)),
            ('export_rknn', lambda: api.export_rknn(str(folder / 'depth.rknn'))),
        ):
            start = time.monotonic()
            code = fn()
            report['operations'].append(dict(name=name, code=code, seconds=time.monotonic()-start))
            if code not in (None, 0):
                raise RuntimeError(f'{name} returned {code}')
        artifact = folder / 'depth.rknn'
        rknn = dict(file=str(artifact.relative_to(root)), bytes=artifact.stat().st_size,
                    sha256=C.sha256(artifact))
        manifest.update(schema='v54_original_VGGT_depth_logits_RKNN_manifest',
                        status='compiled_not_quality_accepted', rknn=rknn,
                        export_manifest_sha256=args.manifest_sha256,
                        compile_source_sha256=report['compile_source_sha256'], configuration=config)
        (root / 'compiled.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        report.update(status='compiled_not_quality_accepted', rknn=rknn,
                      compiled_manifest_sha256=C.sha256(root / 'compiled.json'))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        api.release()
        report['seconds'] = time.monotonic()-begin
        (folder / 'compile.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
