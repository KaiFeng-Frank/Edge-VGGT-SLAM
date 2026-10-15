#!/usr/bin/env python3
"""Sequential board export and three matched FP32 checks, after calibration ends."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

W = Path(__file__).resolve().parent
P = W.parent
ADB = ['adb', '-s', '0123456789ABCDEF']
B = '/mnt/vggt-data/vggt-slam2'
E = B + '/experiments'
S = '/dev/shm/edge-vggt-v129-joint-mlp-source-r1'
SDK = E + '/vggt-v49-toolchain-venv-r1/bin/python'
PROD = B + '/venv/bin/python'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def shell(args, **kwargs):
    return subprocess.run(ADB + ['shell', shlex.join(args)], check=True, **kwargs)

def main():
    live = shell(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout
    assert not any('prepare_hybrid_pair_board.py' in line or 'run_sampled_cpu_stage_r126.py' in line for line in live.splitlines()), 'Another owned board CPU stage is active'
    H = W / 'executed-source-r1'
    H.mkdir(exist_ok=False)
    inputs = [(W / 'export_joint_mlp_board.py', '82f35353bf57b662b95b89cf5472e18e50e832519e8545ab7237994e35e06c07'),
              (P / 'check_captured_pair_fp32_board.py', 'b0f4bd489d1e6ec8eb28869f609d7e48469aa9d19b2120d7906cd9284059bc4c'),
              (P / 'executed-FP32-source-r1/run_sampled_cpu_stage_r126.py', '700ec5b12b6f15cf9907cfacbff3a63258ef70a14cf5efe06ba5470991de1b89')]
    shell(['mkdir', S], capture_output=True)
    for path, digest in inputs:
        assert sha(path) == digest
        shutil.copyfile(path, H / path.name)
        subprocess.run(ADB + ['push', str(path), S + '/' + path.name], check=True, capture_output=True)
    actual = shell(['sha256sum', *[S + '/' + path.name for path, _ in inputs]], capture_output=True, text=True).stdout
    actual = dict(line.strip().split(None, 1)[::-1] for line in actual.splitlines())
    assert all(actual[S + '/' + path.name] == digest for path, digest in inputs)
    (W / 'board-source-pins.json').write_text(json.dumps(actual, indent=2) + '\n')
    def run_stage(label, args):
        output = '/dev/shm/edge-vggt-v129-joint-mlp-' + label + '-r1'
        command = ['env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', '-u', 'VIRTUAL_ENV',
                   'LD_LIBRARY_PATH=' + B + '/sysroot/usr/lib/aarch64-linux-gnu',
                   'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                   'NUMEXPR_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'taskset', '-c', '4-7',
                   SDK, '-B', '-u', S + '/export_joint_mlp_board.py', '--reference-helper',
                   S + '/check_captured_pair_fp32_board.py', '--output-dir', output, *args]
        launcher = H / ('run_' + label + '_r1.sh')
        launcher.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.join(command) + '\n')
        subprocess.run(ADB + ['push', str(launcher), S + '/' + launcher.name], check=True, capture_output=True)
        got = shell(['sha256sum', S + '/' + launcher.name], capture_output=True, text=True).stdout.split()[0]
        assert got == sha(launcher)
        stage = 'native518joint129' + label + 'r1'
        argv = ['env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', '-u', 'VIRTUAL_ENV',
                'LD_LIBRARY_PATH=' + B + '/sysroot/usr/lib/aarch64-linux-gnu', PROD, '-B', '-u',
                S + '/run_sampled_cpu_stage_r126.py', '--expected-npu-hz', '1000000000',
                '--stage', stage, '--timeout', '180', '--', '/bin/sh', S + '/' + launcher.name]
        (W / (label + '-launch.json')).write_text(json.dumps(dict(argv=argv, launcher_sha256=got), indent=2) + '\n')
        print('Launching board-only joint MLP ' + label, flush=True)
        with (W / (label + '-controller-stdout.log')).open('wb') as log:
            done = subprocess.run(ADB + ['shell', shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
        local = W / (label + '-r1')
        subprocess.run(ADB + ['pull', output + '/', str(local)], check=True)
        for leaf in (f'vggt-v83-{stage}-control-r1.json', f'vggt-v83-{stage}-r1.log'):
            subprocess.run(ADB + ['pull', E + '/' + leaf, str(W / leaf)], check=True)
        r = json.loads((local / 'result.json').read_text())
        ctl = json.loads((W / f'vggt-v83-{stage}-control-r1.json').read_text())
        artifacts = [v['onnx'] for v in r['models']] if label == 'export' else [v for row in r['checks'] for v in row['outputs']]
        for artifact in artifacts:
            path = local / artifact['file']
            assert path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256']
            with path.open('rb') as f:
                os.fsync(f.fileno())
        print(json.dumps(dict(label=label, status=r['status'], seconds=ctl['seconds'], sha256=sha(local / 'result.json'))), flush=True)
        assert done.returncode == 0 and ctl['status'] == 'completed' and ctl['returncode'] == 0 and not ctl['cleanup_errors']
        assert not r['cleanup_errors'] and r['source_and_inputs_unchanged'] and r['ORT_sessions_dropped']
        assert r['status'] == ('exported_original_joint_MLP_pair_not_FP32_or_NPU_accepted' if label == 'export' else 'checked_original_joint_MLP_pair_fixed_FP32_gate_passed_not_NPU_accepted')
        return local / 'result.json'
    exported = run_stage('export', ['--mode', 'export'])
    for batch in range(3):
        capture = P / f'capture-batch{batch}-r1/result.json'
        checked = P / f'fp32-batch{batch}-r1/result.json'
        run_stage('check' + str(batch), ['--mode', 'check', '--export-receipt', '/dev/shm/edge-vggt-v129-joint-mlp-export-r1/result.json',
                   '--export-sha256', sha(exported), '--capture-receipt', f'/dev/shm/edge-vggt-v128-capture-batch{batch}-r1/result.json',
                   '--capture-sha256', sha(capture), '--fp32-reference-receipt', f'/dev/shm/edge-vggt-v128-fp32-batch{batch}-r1/result.json',
                   '--fp32-reference-sha256', sha(checked)])
    print('Actual export and all six joint-MLP board FP32 calls completed; no NPU performance acceptance.', flush=True)

if __name__ == '__main__':
    main()
