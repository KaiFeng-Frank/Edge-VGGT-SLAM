#!/usr/bin/env python3
"""Transport the gated R129 FP16 compiler; execute only with explicit --execute.

The host performs no model/ORT/SDK operation. Root owns board scheduling.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

W = Path(__file__).resolve().parent
W128 = W.parent; W127 = W128.parent / 'v127-basis-fp16-work'
B = '/mnt/vggt-data/vggt-slam2'; E = B + '/experiments'
S = '/dev/shm/edge-vggt-v129-joint-mlp-compile-source-r1'
OUT = '/dev/shm/edge-vggt-v129-joint-mlp-compile-r1'
PROD = B + '/venv/bin/python'; SDK = E + '/vggt-v49-toolchain-venv-r1/bin/python'
PIN = dict(compile_joint_mlp_fp16_board_py='4edc92073654e88fa73862b0f50e23bc987c1ec54f70b1c756a0a4d5519e41c3',
    compile_basis_pair_fp16_py='2bbf180a9a70444ba592b7898fa23b96b11f36da3859f07cc5b435f0a25acb11',
    run_sampled_cpu_stage_r126_py='700ec5b12b6f15cf9907cfacbff3a63258ef70a14cf5efe06ba5470991de1b89')


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def sync_tree(folder):
    for path in folder.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream: os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--adb', type=Path, default=Path('adb'))
    p.add_argument('--serial', default='0123456789ABCDEF')
    p.add_argument('--expected-npu-hz', choices=(300000000, 1000000000), type=int, default=1000000000)
    p.add_argument('--timeout', type=int, default=600); a = p.parse_args()
    require(0 < a.timeout <= 900, 'Bounded CPU compilation timeout required')
    sources = [W / 'compile_joint_mlp_fp16_board.py', W127 / 'compile_basis_pair_fp16.py',
        W128 / 'executed-FP32-source-r1/run_sampled_cpu_stage_r126.py']
    for source in sources: require(sha(source) == PIN[source.name.replace('.', '_')], 'Frozen source differs: ' + source.name)
    spec = importlib.util.spec_from_file_location('r129_metadata_preflight', sources[0]); worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    export_path = W / 'export-r1/result.json'; checks = [W / ('check' + str(i) + '-r1/result.json') for i in range(3)]
    require(sha(export_path) == worker.EXPORT_SHA and [sha(path) for path in checks] == list(worker.CHECK_SHAS), 'Actual six-input gates differ')
    exported = json.loads(export_path.read_text()); worker.validate_metadata(exported, [json.loads(path.read_text()) for path in checks])
    env = ['env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', '-u', 'VIRTUAL_ENV',
        'LD_LIBRARY_PATH=' + B + '/sysroot/usr/lib/aarch64-linux-gnu', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=']
    compile_command = [*env, 'taskset', '-c', '4-7', SDK, '-B', '-u', S + '/compile_joint_mlp_fp16_board.py',
        '--onnx', '/dev/shm/edge-vggt-v129-joint-mlp-export-r1/pair-joint-mlp.onnx',
        '--export-receipt', '/dev/shm/edge-vggt-v129-joint-mlp-export-r1/result.json',
        '--check-receipts', *['/dev/shm/edge-vggt-v129-joint-mlp-check' + str(i) + '-r1/result.json' for i in range(3)],
        '--configuration-source', S + '/compile_basis_pair_fp16.py', '--output-dir', OUT]
    stage = 'native518joint129fp16r1'
    command = [*env, PROD, '-B', '-u', S + '/run_sampled_cpu_stage_r126.py', '--expected-npu-hz', str(a.expected_npu_hz),
        '--stage', stage, '--timeout', str(a.timeout), '--', '/bin/sh', S + '/run_joint_mlp_fp16_r1.sh']
    if not a.execute:
        print(json.dumps(dict(status='prepared_not_staged_or_executed', CPU_only_no_frequency_writes=True,
            compile_argv=compile_command, controller_argv=command), indent=2)); return
    adb = [str(a.adb), '-s', a.serial]
    def shell(argv, **kwargs): return subprocess.run([*adb, 'shell', shlex.join(argv)], check=True, **kwargs)
    processes = shell(['ps', '-eww', '-o', 'pid,args'], capture_output=True, text=True).stdout.splitlines()
    busy_names = ('prepare_hybrid_pair_board.py', 'compile_hybrid_pair_step2_board.py', 'run_hybrid_pair_board.py',
        'export_joint_mlp_board.py', 'compile_joint_mlp_fp16_board.py',
        'run_sampled_cpu_stage_r126.py', 'npu_frequency_test.py')
    require(not any(any(name in line for name in busy_names) and 'ps -eww' not in line for line in processes), 'Another scoped board worker is active; root must serialize jobs')
    frozen = W / 'executed-FP16-source-r1'; frozen.mkdir(exist_ok=False)
    for source in sources: shutil.copyfile(source, frozen / source.name)
    (frozen / 'run_joint_mlp_fp16_r1.sh').write_text('#!/bin/sh\nset -eu\nexec ' + shlex.join(compile_command) + '\n')
    (frozen / 'SOURCE_MANIFEST.json').write_text(json.dumps({path.name: sha(path) for path in frozen.iterdir() if path.is_file()}, indent=2) + '\n')
    sync_tree(frozen); shell(['mkdir', S], stdout=subprocess.PIPE)
    for path in frozen.iterdir(): subprocess.run([*adb, 'push', str(path), S + '/' + path.name], check=True, capture_output=True)
    output = shell(['sha256sum', *[S + '/' + path.name for path in frozen.iterdir()]], capture_output=True, text=True).stdout
    board_pins = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in output.splitlines()}
    require(all(board_pins[S + '/' + path.name] == sha(path) for path in frozen.iterdir()), 'Staged source SHA differs')
    (W / 'compile-board-source-pins.json').write_text(json.dumps(board_pins, indent=2) + '\n')
    (W / 'compile-launch-r1.json').write_text(json.dumps(dict(argv=command, CPU_only=True, frequency_written=False), indent=2) + '\n')
    with (W / 'compile-controller-stdout-r1.log').open('wb') as stream:
        done = subprocess.run([*adb, 'shell', shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
    local = W / 'compile-r1'; pulls = []
    for remote, destination in [(OUT + '/', local),
        (E + '/vggt-v83-' + stage + '-control-r1.json', W / ('vggt-v83-' + stage + '-control-r1.json')),
        (E + '/vggt-v83-' + stage + '-r1.log', W / ('vggt-v83-' + stage + '-r1.log'))]:
        result = subprocess.run([*adb, 'pull', remote, str(destination)], capture_output=True, text=True)
        pulls.append(dict(remote=remote, returncode=result.returncode, stderr=result.stderr))
    sync_tree(W)
    (W / 'compile-transport-r1.json').write_text(json.dumps(dict(adb_exit=done.returncode, pulls=pulls), indent=2) + '\n')
    require(all(row['returncode'] == 0 for row in pulls), 'Compile evidence pull incomplete; preserve available output')
    report = json.loads((local / 'result.json').read_text()); control = json.loads((W / ('vggt-v83-' + stage + '-control-r1.json')).read_text())
    for row in report.get('generated_files', []):
        path = local / row['file']; require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Pulled generated artifact differs')
    require(done.returncode == 0 and control['status'] == 'completed' and control['returncode'] == 0 and not control['cleanup_errors'] and
        report['status'] == worker.SUCCESS and report['sdk_released'] and not report['cleanup_errors'], 'Board compilation failed; saved evidence is retained')
    print(json.dumps(dict(status=report['status'], receipt_sha256=sha(local / 'result.json'), rknn=report['rknn'],
        compiler_schedule_and_precision_unverified=True), indent=2))


if __name__ == '__main__':
    main()
