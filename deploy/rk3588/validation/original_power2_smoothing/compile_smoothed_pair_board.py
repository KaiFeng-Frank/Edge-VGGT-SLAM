#!/usr/bin/env python3
"""Board-only W8A8 compilation after the original pair FP32 check has passed."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
CAPTURE_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def artifact(root, row):
    relative = Path(row['file'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = (root / relative).resolve(strict=True)
    assert path.is_relative_to(root.resolve()) and path.stat().st_size == row['bytes']
    assert sha(path) == row['sha256']
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checked', type=Path, required=True)
    p.add_argument('--checked-sha256', required=True)
    p.add_argument('--check-source', type=Path, required=True)
    p.add_argument('--check-source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
    assert sha(a.checked) == a.checked_sha256 and sha(a.check_source) == a.check_source_sha256
    checked = json.loads(a.checked.read_text())
    assert checked['status'] == 'checked_original_FP32_equivalence_not_quality_accepted'
    assert checked['script_sha256'] == a.check_source_sha256
    check = a.checked.parent / 'check.json'
    assert sha(check) == checked['check_receipt_sha256']
    receipt = json.loads(check.read_text())
    assert receipt['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] is True
    assert len(receipt['cases']) == 2
    assert all(len(case['checks']) == 2 and all(row['versus_baseline_FP32_ONNX']['strict_pass']
        for row in case['checks']) for case in receipt['cases'])
    graph = artifact(a.checked.parent, checked['onnx'])
    assert receipt['candidate_onnx_sha256'] == checked['onnx']['sha256'] == sha(graph)
    assert receipt['original_pair_onnx_sha256'] == 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
    calibration = BASE / 'experiments/vggt-v56-calibration-r1'
    capture_path = calibration / 'capture.json'
    assert sha(capture_path) == CAPTURE_SHA
    capture = json.loads(capture_path.read_text())
    assert capture['status'] == 'completed_original_FP32_calibration_capture'
    assert capture['full_forward_completed'] == 8 and not capture['cleanup_errors']
    assert capture['checkpoint'] == dict(bytes=5026874952,
        sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
    group = capture['groups']['aggregator'][0]
    assert group['index'] == 0 and len(group['samples']) == 8
    paths = [artifact(calibration, row['artifact']) for row in group['samples']]
    dataset = artifact(calibration, group['dataset'])
    assert dataset.read_text().splitlines() == [str(path) for path in paths]
    assert importlib.metadata.version('rknn-toolkit2') == '2.3.2'
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    from rknn.api import RKNN
    pins = {str(path): sha(path) for path in [Path(__file__).resolve(), a.check_source,
        a.checked, check, graph, capture_path, dataset, *paths]}
    report = dict(schema='v78_original_pair_power2_smooth_W8A8_compile', status='running',
        original_model_only=True, original_checkpoint_unchanged=True, candidate_accepted=False,
        script_sha256=sha(__file__), check_source_sha256=a.check_source_sha256,
        checked_sha256=a.checked_sha256, check_receipt_sha256=sha(check),
        configuration=CONFIG, build_options=dict(do_quantization=True, auto_hybrid=False),
        capture_sha256=CAPTURE_SHA, dataset_sha256=sha(dataset),
        original_graph_sha256='efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65',
        candidate_graph_sha256=sha(graph), operations=[], cleanup_errors=[], released=False)
    sdk = None
    cwd, temp, begin = Path.cwd(), os.environ.get('TMPDIR'), time.monotonic()
    try:
        os.chdir(a.output)
        os.environ['TMPDIR'] = str(a.output)
        sdk = RKNN(verbose=False, verbose_file=str(a.output / 'sdk.log'))
        for name, function in [('config', lambda: sdk.config(**CONFIG)),
                ('load_onnx', lambda: sdk.load_onnx(model=str(graph))),
                ('build', lambda: sdk.build(do_quantization=True, dataset=str(dataset), auto_hybrid=False)),
                ('export_rknn', lambda: sdk.export_rknn(str(a.output / 'pair.rknn')))]:
            started = time.monotonic()
            code = function()
            report['operations'].append(dict(name=name, code=code, seconds=time.monotonic()-started))
            assert code in (0, None), (name, code)
        path = a.output / 'pair.rknn'
        report.update(status='compiled_not_quality_accepted',
            rknn=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if sdk is not None:
            try:
                code = sdk.release()
                report.update(release_code=code, released=code in (0, None))
                if code not in (0, None): report['cleanup_errors'].append('SDK release returned ' + str(code))
            except BaseException as exc:
                report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
        try:
            os.chdir(cwd)
            if temp is None: os.environ.pop('TMPDIR', None)
            else: os.environ['TMPDIR'] = temp
            report['environment_restored'] = Path.cwd() == cwd and os.environ.get('TMPDIR') == temp
        except BaseException as exc:
            report['environment_restored'] = False
            report['cleanup_errors'].append(f'Environment restore: {type(exc).__name__}: {exc}')
        report['source_and_inputs_unchanged'] = True
        for path, value in pins.items():
            try:
                if sha(path) != value:
                    raise RuntimeError('Pinned source or input changed: ' + path)
            except BaseException as exc:
                report['source_and_inputs_unchanged'] = False
                report['cleanup_errors'].append(f'Identity check: {type(exc).__name__}: {exc}')
        report['seconds'] = time.monotonic()-begin
        if report['cleanup_errors'] or not report['source_and_inputs_unchanged'] or not report['environment_restored']:
            report['status'] = 'failed_cleanup_or_identity'
        (a.output / 'compile.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k: report.get(k) for k in ('status', 'seconds', 'rknn', 'cleanup_errors')}), flush=True)
    return 0 if report['status'] == 'compiled_not_quality_accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
