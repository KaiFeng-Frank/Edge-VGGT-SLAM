#!/usr/bin/env python3
"""RK3588-only isolated build and tiny smoke of the v36 frame cache."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2/experiments')
CORE_FILES = ('dpvo_frame_corr.cpp', 'dpvo_frame_corr.h', 'dpvo_frame_corr.py')
PARENT_PINS = {
    'dpvo_corr.cpp': '3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990',
    'dpvo_corr.h': '1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def board_guard():
    require(platform.machine() == 'aarch64' and platform.system() == 'Linux', 'RK3588 board only')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588 device tree required')


def build_command(source_dir, parent_dir, output, compiler, toolchain_root):
    r = Path(toolchain_root)
    return [str(compiler), '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp',
            '-fno-fast-math', '-ffp-contract=off', '--sysroot=' + str(r),
            '-B' + str(r / 'usr/bin'), '-B' + str(r / 'usr/lib/gcc/aarch64-linux-gnu/11'),
            '-I', str(parent_dir), str(Path(source_dir) / 'dpvo_frame_corr.cpp'), '-o', str(output)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'parent-dir', 'toolchain-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-freeze-sha256', required=True)
    p.add_argument('--bootstrap-sha256', required=True)
    a = p.parse_args(argv)
    board_guard()  # Before output writes, toolchain loading or compilation.
    out = a.output_dir.resolve()
    require(out.is_relative_to(BASE), 'New isolated board experiments output required')
    out.mkdir(exist_ok=False)
    report = dict(schema='v36_immutable_frame_correlation_native_build', status='failed',
                  script_sha256=sha(__file__), cleanup_errors=[], real_features_loaded=False,
                  neural_or_BA_executed=False, candidate_accepted=False)
    pins = {}
    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, 'Source/toolchain SHA mismatch: ' + str(path))
        pins[str(path)] = expected
        return path
    try:
        pin(__file__, report['script_sha256'])
        frozen = json.loads(pin(a.source_dir / 'cache-source-freeze.json', a.source_freeze_sha256).read_text())
        require(frozen['schema'] == 'v36_immutable_frame_correlation_source_freeze'
                and frozen['parent_source_sha256'] == PARENT_PINS, 'Cache/parent source contract differs')
        for name in CORE_FILES:
            pin(a.source_dir / name, frozen['candidate_source_sha256'][name])
        for name, digest in PARENT_PINS.items():
            pin(a.parent_dir / name, digest)
        t = a.toolchain_dir.resolve()
        prior = json.loads(pin(t / 'bootstrap-r2.json', a.bootstrap_sha256).read_text())
        require(prior['status'] == 'isolated_compiler_and_synthetic_smoke_passed', 'Verified isolated compiler required')
        r = t / 'root'
        compiler = pin(r / 'usr/bin/aarch64-linux-gnu-g++-11', prior['compiler_sha256'])
        pin(t / 'manifest.json', sha(t / 'manifest.json'))
        target = out / 'libdpvo_frame_corr.so'
        command = build_command(a.source_dir.resolve(), a.parent_dir.resolve(), target, compiler, r)
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(r / 'usr/lib/aarch64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH', '')
        env['TMPDIR'] = str(out)
        result = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True)
        report.update(command=command, exit_code=result.returncode, output_log=result.stdout + result.stderr,
                      compiler_sha256=prior['compiler_sha256'],
                      toolchain_bootstrap_sha256=a.bootstrap_sha256,
                      toolchain_manifest_sha256=sha(t / 'manifest.json'),
                      candidate_source_sha256={n: frozen['candidate_source_sha256'][n] for n in CORE_FILES},
                      parent_source_sha256=PARENT_PINS, source_freeze_sha256=a.source_freeze_sha256)
        require(result.returncode == 0, 'Native build failed: ' + result.stderr)
        import numpy as np
        spec = importlib.util.spec_from_file_location('_v36_native_smoke', a.source_dir / 'dpvo_frame_corr.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        g = np.ones((1, 8, 1, 1), np.float32)
        f = np.ones((8, 8, 8), np.float32)
        g.flags.writeable = f.flags.writeable = False
        record = dict(frame_id='generated-build-smoke-only', gmap=g, fmap=f)
        xy, idx = np.full((1, 1, 2, 1, 1), 3.0, np.float32), np.zeros(1, np.int64)
        with module.PerFrameCorrelation(target, library_sha256=sha(target), patch_count=1,
                                       channels=8, patch_shape=(1, 1), threads=4) as cache:
            output = cache(xy, ii=idx, jj=idx, kk=idx, frames=(record,))
            center = output.reshape(1, 1, 7, 7, 1, 1, 2)[0, 0, 3, 3, 0, 0]
            require(np.array_equal(center, np.array([8, 8], np.float32)), 'Tiny smoke center differs')
            before = cache.metrics()
        after = cache.metrics()
        require(after['closed'] and after['current_context_bytes'] == 0
                and after['current_cached_frames'] == 0, 'Tiny smoke cache cleanup failed')
        report.update(status='compiled_and_synthetic_smoke_passed',
                      output=dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target)),
                      smoke=dict(center=center.tolist(), before_close=before, after_close=after,
                                 actual_neon=before['calls'][0]['native_query']['native']['neon']))
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            after = {path: sha(path) for path in pins}
            report.update(source_pins=pins, source_hashes_after=after, sources_and_inputs_unchanged=after == pins)
            require(after == pins, 'Pinned build source changed')
        except Exception as exc:
            report['cleanup_errors'].append(repr(exc))
            report['status'] = 'failed'
        (out / 'build.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'output', 'cleanup_errors')}), flush=True)
    return 0 if report['status'] == 'compiled_and_synthetic_smoke_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
