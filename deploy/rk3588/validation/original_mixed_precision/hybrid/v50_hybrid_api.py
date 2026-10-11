"""Original VGGT hybrid API calls only; imports no real SDK or model."""
from __future__ import annotations

from pathlib import Path
import time

STEP1_OPTIONS = {'rknn_batch_size': None, 'proposal': False,
                 'proposal_dataset_size': 1, 'custom_hybrid': None}


def call(report, name, operation):
    row = {'name': name, 'status': 'failed', 'code': None}
    report['operations'].append(row)
    started = time.monotonic_ns()
    try:
        code = operation()
        row['code'] = code
        if code is not None and (type(code) is not int or code != 0):
            raise RuntimeError(f'{name} returned {code}')
        row['status'] = 'completed'
        return code
    except BaseException as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        row['seconds'] = (time.monotonic_ns() - started) / 1e9


def step1(toolkit, graph, dataset, report, C):
    C.require_board()
    config = {key: value for key, value in C.CONFIGURATION.items() if key != 'do_quantization'}
    config.update(mean_values=None, std_values=None, quant_img_RGB2BGR=False)
    call(report, 'config', lambda: toolkit.config(**config))
    call(report, 'load_onnx', lambda: toolkit.load_onnx(model=str(Path(graph).resolve())))
    report['calibration_api_invoked'] = True
    call(report, 'hybrid_quantization_step1', lambda: toolkit.hybrid_quantization_step1(
        dataset=str(Path(dataset).resolve()), **STEP1_OPTIONS))
    report['calibration_completed_on_board'] = True


def step2(toolkit, model, data, approved_cfg, output, report, C):
    """Draft API wrapper; caller must separately approve exact generated policy."""
    C.require_board()
    if (report.get('precision_policy_approved') is not True
            or C.sha(approved_cfg) != report.get('approved_configuration_sha256')):
        raise ValueError('Exact generated precision configuration approval required')
    call(report, 'hybrid_quantization_step2', lambda: toolkit.hybrid_quantization_step2(
        model_input=str(Path(model).resolve()), data_input=str(Path(data).resolve()),
        model_quantization_cfg=str(Path(approved_cfg).resolve())))
    call(report, 'export_rknn', lambda: toolkit.export_rknn(str(Path(output).resolve())))


def release(toolkit, report):
    if toolkit is None:
        return
    try:
        code = toolkit.release()
        report['release_code'] = code
        if code is not None and (type(code) is not int or code != 0):
            raise RuntimeError(f'release returned {code}')
        report['released'] = True
    except BaseException as exc:
        report['released'] = False
        report['cleanup_errors'].append(f'SDK release: {type(exc).__name__}: {exc}')
