#!/usr/bin/env python3
"""Prepare/run the bounded board R129 A/J/J/A component through ADB only.

No host NN/SDK/ORT execution. Without --execute, print the prepared commands.
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
W127 = W.parent.parent / 'v127-basis-fp16-work'
B = '/mnt/vggt-data/vggt-slam2'; E = B + '/experiments'; PROD = B + '/venv/bin/python'
S = '/dev/shm/edge-vggt-v129-joint-mlp-runtime-source-r1'
OUT = '/dev/shm/edge-vggt-v129-joint-mlp-runtime-r1'; CONTROL = '/dev/shm/edge-vggt-v129-joint-mlp-runtime-control-r1'
PINS = {'run_joint_mlp_fp16_board.py': '4b9f83cead499cc120cae590bdf48d538a9dae308273dbd65a88b5e73505540e',
    'compile_joint_mlp_fp16_board.py': '4edc92073654e88fa73862b0f50e23bc987c1ec54f70b1c756a0a4d5519e41c3',
    'npu_frequency_test.py': 'cb991cf03abc872e3590dd92f26fd0f5ac548d620c5ca791f9354537a9f793b9'}


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
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--execute', action='store_true')
    p.add_argument('--joint-compile', type=Path, default=W / 'compile-r1/result.json')
    p.add_argument('--board-joint-compile', default='/dev/shm/edge-vggt-v129-joint-mlp-compile-r1/result.json')
    p.add_argument('--adb', type=Path, default=Path('adb'))
    p.add_argument('--serial', default='0123456789ABCDEF'); a = p.parse_args()
    sources = [W / 'run_joint_mlp_fp16_board.py', W / 'compile_joint_mlp_fp16_board.py', W127 / 'runtime-source-r1/npu_frequency_test.py']
    require(all(sha(path) == PINS[path.name] for path in sources), 'Frozen worker/compiler/controller source differs')
    compiled = json.loads(a.joint_compile.read_text()); digest = sha(a.joint_compile)
    spec = importlib.util.spec_from_file_location('r129_runtime_metadata', sources[0]); worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    require(compiled['schema'] == 'v129_original_native518_joint_MLP_pair_FP16_compile' and
        compiled['status'] == 'compiled_original_joint_MLP_pair_not_runtime_or_quality_accepted' and
        compiled['script_sha256'] == PINS['compile_joint_mlp_fp16_board.py'] and compiled['sdk_released'] and
        compiled['all_operations_succeeded'] and not compiled['cleanup_errors'] and
        all(compiled[k] for k in ('source_and_inputs_unchanged', 'cwd_restored', 'affinity_restored', 'environment_restored')), 'Actual successful immutable joint compile required')
    model = a.joint_compile.parent / compiled['rknn']['file']
    require(model.stat().st_size == compiled['rknn']['bytes'] and sha(model) == compiled['rknn']['sha256'], 'Pulled actual RKNN differs')
    env = ['env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', '-u', 'VIRTUAL_ENV', 'LD_LIBRARY_PATH=' + B + '/sysroot/usr/lib/aarch64-linux-gnu',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=']
    run = [*env, 'taskset', '-c', '4,6', PROD, '-B', '-u', S + '/run_joint_mlp_fp16_board.py', '--joint-compile', a.board_joint_compile,
        '--joint-compile-sha256', digest, '--compiler-source', S + '/compile_joint_mlp_fp16_board.py', '--output-dir', OUT]
    command = [*env, PROD, '-B', '-u', S + '/npu_frequency_test.py', '--execute-frequency-test', '--frequency-hz', '1000000000',
        '--temperature-limit-c', '75', '--timeout-seconds', '180', '--launcher', S + '/run_joint_mlp_r1.sh', '--output', CONTROL]
    if not a.execute:
        print(json.dumps(dict(status='prepared_not_staged_or_executed', worker_argv=run, controller_argv=command), indent=2)); return
    adb = [str(a.adb), '-s', a.serial]
    def shell(argv, **kwargs): return subprocess.run([*adb, 'shell', shlex.join(argv)], check=True, **kwargs)
    processes = shell(['ps', '-eww', '-o', 'pid,args'], capture_output=True, text=True).stdout.splitlines()
    busy = ('prepare_hybrid_pair_board.py', 'compile_hybrid_pair_step2_board.py', 'run_hybrid_pair_board.py',
        'run_sampled_cpu_stage_r126.py', 'npu_frequency_test.py', 'export_joint_mlp_board.py', 'compile_joint_mlp_fp16_board.py', 'run_joint_mlp_fp16_board.py')
    require(not any(any(name in line for name in busy) and 'ps -eww' not in line for line in processes), 'Root must serialize active scoped board jobs')
    for path in (S, OUT, CONTROL): shell(['test', '!', '-e', path], capture_output=True)
    expected = shell(['sha256sum', a.board_joint_compile], capture_output=True, text=True).stdout.split()[0]
    require(expected == digest, 'Board compiled receipt differs')
    frozen = W / 'executed-runtime-source-r1'; frozen.mkdir(exist_ok=False)
    for source in sources: shutil.copyfile(source, frozen / source.name)
    (frozen / 'run_joint_mlp_r1.sh').write_text('#!/bin/sh\nset -eu\nexec ' + shlex.join(run) + '\n')
    (frozen / 'SOURCE_MANIFEST.json').write_text(json.dumps({path.name: sha(path) for path in frozen.iterdir()}, indent=2) + '\n')
    sync_tree(frozen); shell(['mkdir', S], stdout=subprocess.PIPE)
    for path in frozen.iterdir(): subprocess.run([*adb, 'push', str(path), S + '/' + path.name], check=True, capture_output=True)
    rows = shell(['sha256sum', *[S + '/' + path.name for path in frozen.iterdir()]], capture_output=True, text=True).stdout.splitlines()
    board_pins = {row.split(None, 1)[1].strip(): row.split(None, 1)[0] for row in rows}
    require(all(board_pins[S + '/' + path.name] == sha(path) for path in frozen.iterdir()), 'Staged runtime source differs')
    (W / 'runtime-board-source-pins-r1.json').write_text(json.dumps(board_pins, indent=2) + '\n')
    (W / 'runtime-launch-r1.json').write_text(json.dumps(dict(argv=command, core_mask=1, init_flags=0, calls=8), indent=2) + '\n')
    with (W / 'runtime-controller-stdout-r1.log').open('wb') as stream:
        done = subprocess.run([*adb, 'shell', shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
    local, control_dir = W / 'runtime-r1', W / 'runtime-control-r1'; pulls = []
    for remote, destination in ((OUT + '/', local), (CONTROL + '/', control_dir)):
        result = subprocess.run([*adb, 'pull', remote, str(destination)], capture_output=True, text=True)
        pulls.append(dict(remote=remote, returncode=result.returncode, stderr=result.stderr))
    if local.exists(): sync_tree(local)
    if control_dir.exists(): sync_tree(control_dir)
    (W / 'runtime-transport-r1.json').write_text(json.dumps(dict(adb_exit=done.returncode, pulls=pulls), indent=2) + '\n')
    require(all(row['returncode'] == 0 for row in pulls), 'Pull failed; obtainable evidence retained')
    report, control = json.loads((local / 'result.json').read_text()), json.loads((control_dir / 'control.json').read_text())
    for call in report.get('calls', []):
        for row in call.get('outputs', []):
            path = local / row['file']; require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Pulled saved output differs')
    require(done.returncode == 0 and control['status'] == 'worker_completed_pending_result_review' and control['child_returncode'] == 0 and
        control['frequency_governor_limits_restored'] and not control['restore_errors'] and report['status'] == worker.SUCCESS and
        report['resource_totals'] == report['expected_resource_counts'] and report['all_contexts_and_pending_outputs_closed'] and not report['cleanup_errors'],
        'Runtime/control failed; saved evidence retained')
    print(json.dumps(dict(status=report['status'], receipt_sha256=sha(local / 'result.json'), timing_summary=report['timing_summary'],
        full_model_or_trajectory_accepted=False), indent=2))


if __name__ == '__main__':
    main()
