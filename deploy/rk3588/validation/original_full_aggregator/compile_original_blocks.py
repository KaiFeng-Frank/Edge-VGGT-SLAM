#!/usr/bin/env python3
"""Compile complete original VGGT block groups on RK3588, one SDK process each."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--worker', type=int)
    args = p.parse_args()
    if b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        p.error('RK3588 board execution required')
    if sha(args.manifest) != args.manifest_sha256:
        raise ValueError('Original export manifest hash mismatch')
    root = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text())
    is_dino = manifest.get('schema') == 'v54_original_VGGT_24_DINO_block_ONNX_manifest'
    expected = ([('dino', i) for i in range(24)] if is_dino else
                [(r, i) for i in range(24) for r in ('frame', 'global')])
    if [(b['role'], b['index']) for b in manifest['blocks']] != expected:
        raise ValueError('Exactly the complete ordered original block group is required')
    total = len(expected)
    os.sched_setaffinity(0, {4, 5, 6, 7})
    if args.worker is not None:
        b = manifest['blocks'][args.worker]
        source = root / b['file']
        if source.stat().st_size != b['bytes'] or sha(source) != b['sha256']:
            raise ValueError('Original ONNX identity mismatch')
        folder = root / 'compiled' / f"{b['role']}-{b['index']:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        os.chdir(folder)
        os.environ['TMPDIR'] = str(folder)
        from rknn.api import RKNN
        config = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
                      mean_values=None, std_values=None, quant_img_RGB2BGR=False,
                      model_pruning=False, remove_reshape=False, quantize_weight=False,
                      enable_flash_attention=False)
        report = dict(role=b['role'], index=b['index'], source_sha256=b['sha256'],
                      configuration=config, operations=[], status='running')
        api = RKNN(verbose=False, verbose_file=str(folder / 'sdk.log'))
        begin = time.monotonic()
        try:
            for name, fn in (
                ('config', lambda: api.config(**config)),
                ('load_onnx', lambda: api.load_onnx(model=str(source))),
                ('build', lambda: api.build(do_quantization=False, dataset=None, auto_hybrid=False)),
                ('export_rknn', lambda: api.export_rknn(str(folder / 'block.rknn'))),
            ):
                start = time.monotonic()
                code = fn()
                report['operations'].append(dict(name=name, code=code, seconds=time.monotonic()-start))
                if code not in (None, 0):
                    raise RuntimeError(f'{name} returned {code}')
            output = folder / 'block.rknn'
            report.update(status='compiled', rknn=dict(file=str(output.relative_to(root)),
                          bytes=output.stat().st_size, sha256=sha(output)))
        except BaseException as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            api.release()
            report['seconds'] = time.monotonic() - begin
            write(folder / 'compile.json', report)
        return
    begin = time.monotonic()
    for i, b in enumerate(manifest['blocks']):
        folder = root / 'compiled' / f"{b['role']}-{b['index']:02d}"
        receipt = folder / 'compile.json'
        if not receipt.is_file():
            subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                            '--manifest', str(args.manifest.resolve()),
                            '--manifest-sha256', args.manifest_sha256, '--worker', str(i)], check=True)
        result = json.loads(receipt.read_text())
        if result.get('status') != 'compiled' or result.get('source_sha256') != b['sha256']:
            raise ValueError('Compiled source or completion mismatch')
        artifact = root / result['rknn']['file']
        if artifact.stat().st_size != result['rknn']['bytes'] or sha(artifact) != result['rknn']['sha256']:
            raise ValueError('Compiled RKNN bytes changed')
        b['rknn'] = result['rknn']
        print(json.dumps(dict(completed=i+1, total=total, role=b['role'], index=b['index'],
                              block_seconds=result['seconds'], elapsed_seconds=time.monotonic()-begin)), flush=True)
        write(root / 'compile-progress.json', dict(completed=i+1, total=total, elapsed_seconds=time.monotonic()-begin))
    manifest.update(schema=('v54_original_VGGT_24_DINO_block_RKNN_manifest' if is_dino else
                            'v53_original_VGGT_48_block_RKNN_manifest'), status='compiled_not_quality_accepted',
                    export_manifest_sha256=args.manifest_sha256, compile_seconds=time.monotonic()-begin)
    write(root / 'compiled.json', manifest)
    print(json.dumps(dict(manifest=str(root/'compiled.json'), sha256=sha(root/'compiled.json'))), flush=True)


if __name__ == '__main__':
    main()
