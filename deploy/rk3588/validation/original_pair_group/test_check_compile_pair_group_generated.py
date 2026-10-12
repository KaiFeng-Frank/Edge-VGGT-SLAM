"""Generated tensors and fake APIs only; no original model, ORT or RKNN execution."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).with_name('check_compile_pair_group.py')
spec = importlib.util.spec_from_file_location('v64_generated_checker', SOURCE)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def toy_pair(x, index):
    frame = np.add(x, np.float32(index + 1))
    return [frame, np.add(frame.reshape(1, 106, 1024), np.float32(.125))]


def prepared(tmp_path, monkeypatch, *, mismatch=False, original_mismatch=False):
    fake = ModuleType('onnxruntime')
    fake.__version__ = 'generated-only'
    fake.SessionOptions = SimpleNamespace
    fake.GraphOptimizationLevel = SimpleNamespace(ORT_DISABLE_ALL=0)
    calls = []

    class Session:
        def __init__(self, path, sess_options, providers):
            self.role = Path(path).name
            assert providers == ['CPUExecutionProvider'] and sess_options.graph_optimization_level == 0

        def get_inputs(self):
            return [SimpleNamespace(name='input')]

        def get_outputs(self):
            names = [b['name'] for b in C.BINDINGS] if self.role == 'group' else ['frame_output', 'global_output']
            return [SimpleNamespace(name=n) for n in names]

        def run(self, names, feeds):
            calls.append((self.role, list(names)))
            x = feeds['input']
            if self.role != 'group':
                return toy_pair(x, int(self.role[-1]))
            first = toy_pair(x, 0)
            result = first + toy_pair(first[1].reshape(C.INPUT_SHAPE), 1)
            if mismatch:
                result[-1] = result[-1].copy()
                result[-1].flat[0] += np.float32(.01)
            return result

    fake.InferenceSession = Session
    monkeypatch.setitem(sys.modules, 'onnxruntime', fake)
    cases = []
    for i in range(2):
        directory = tmp_path / f'capture{i}'
        directory.mkdir()
        x = np.full(C.INPUT_SHAPE, i / 8, dtype=np.float32)
        input_row = C.save_tensor(np, directory, 'input.npy', x, dict(name='input', shape=C.INPUT_SHAPE))
        refs = None
        if i == 0:
            first = toy_pair(x, 0)
            values = first + toy_pair(first[1].reshape(C.INPUT_SHAPE), 1)
            if original_mismatch:
                values[0] = values[0] + np.float32(.02)
            refs = [C.save_tensor(np, directory, f'ref-{j}.npy', v, C.BINDINGS[j]) for j, v in enumerate(values)]
        cases.append(dict(tag=f'case{i}', directory=str(directory), input=input_row, original_outputs=refs))
    output = tmp_path / 'check'
    output.mkdir()
    report = dict(cases=cases)
    return output, report, calls


def test_all_four_outputs_two_real_shape_generated_cases(monkeypatch, tmp_path):
    output, report, calls = prepared(tmp_path, monkeypatch)
    C.run_check(Path('group'), [Path('parent0'), Path('parent1')], report['cases'], output, report)
    assert len(calls) == 6 and report['all_separate_ONNX_checks_byte_equal']
    assert report['original_FP32_reference_case_count'] == 1
    assert report['original_FP32_gate_passed_where_reference_available']
    assert not report['original_FP32_references_cover_all_cases']
    assert all(len(c['group_outputs']) == len(c['chain_outputs']) == len(c['checks']) == 4 for c in report['cases'])
    assert report['sessions_dropped']


def test_changed_fourth_output_blocks_check_and_drops_sessions(monkeypatch, tmp_path):
    output, report, _ = prepared(tmp_path, monkeypatch, mismatch=True)
    with pytest.raises(ValueError, match='compilation blocked'):
        C.run_check(Path('group'), [Path('parent0'), Path('parent1')], report['cases'], output, report)
    assert not report['all_separate_ONNX_checks_byte_equal'] and report['sessions_dropped']
    assert not report['cases'][0]['checks'][3]['byte_equal_to_two_separate_FP32_ONNX']


def test_original_gate_failure_retained_without_rejecting_unchanged_chain(monkeypatch, tmp_path):
    output, report, _ = prepared(tmp_path, monkeypatch, original_mismatch=True)
    C.run_check(Path('group'), [Path('parent0'), Path('parent1')], report['cases'], output, report)
    assert report['all_separate_ONNX_checks_byte_equal']
    assert not report['original_FP32_gate_passed_where_reference_available']
    assert report['status'] == 'checked_separate_ONNX_equivalence_not_quality_accepted'


def test_failed_check_precedes_sdk_import_and_build_failure_releases(monkeypatch, tmp_path):
    output, report, _ = prepared(tmp_path, monkeypatch)
    C.run_check(Path('group'), [Path('parent0'), Path('parent1')], report['cases'], output, report)
    report.update(script_sha256=C.sha(SOURCE), manifest_sha256='manifest', graph_sha256='graph',
        cleanup_errors=[], source_and_inputs_unchanged=True)
    args = SimpleNamespace(check_report=output / 'check.json', sha256='manifest', output_dir=tmp_path)
    build_report = dict(script_sha256=C.sha(SOURCE), graph_sha256='graph', operations=[], cleanup_errors=[])
    fake = ModuleType('rknn.api')
    fake.__file__ = str(SOURCE)
    seen = []

    class SDK:
        __module__ = 'rknn.api'
        def __init__(self, **kwargs): seen.append('init')
        def config(self, **kwargs): return 0
        def load_onnx(self, **kwargs): return 0
        def build(self, **kwargs): seen.append('build'); return -1
        def release(self): seen.append('release'); return 0

    fake.RKNN = SDK
    monkeypatch.setitem(sys.modules, 'rknn.api', fake)
    monkeypatch.setattr(C.importlib.metadata, 'version', lambda name: '2.3.2')
    report['all_separate_ONNX_checks_byte_equal'] = False
    C.write(args.check_report, report)
    args.check_report_sha256 = C.sha(args.check_report)
    with pytest.raises(ValueError, match='before compilation'):
        C.run_compile(Path('group'), args, build_report)
    assert seen == []
    report['all_separate_ONNX_checks_byte_equal'] = True
    C.write(args.check_report, report)
    args.check_report_sha256 = C.sha(args.check_report)
    with pytest.raises(ValueError, match='build returned -1'):
        C.run_compile(Path('group'), args, build_report)
    assert seen == ['init', 'build', 'release'] and build_report['released']


def test_host_guard_precedes_optional_library_imports(tmp_path):
    if sys.platform != 'linux' or Path('/proc/device-tree/compatible').exists():
        pytest.skip('Expected host without RK3588 device tree')
    destination = tmp_path / 'must-not-exist'
    result = subprocess.run([sys.executable, '-B', str(SOURCE), '--manifest', str(tmp_path / 'none'),
        '--sha256', 'none', '--mode', 'check', '--parent-manifest', str(tmp_path / 'none'),
        '--capture-dir', str(tmp_path), '--output-dir', str(destination)], text=True, capture_output=True)
    assert result.returncode == 2 and 'restricted to RK3588' in result.stderr
    assert not destination.exists()
