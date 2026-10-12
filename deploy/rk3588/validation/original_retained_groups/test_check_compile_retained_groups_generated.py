"""Generated tensors and fake APIs only; no original model, ORT or RKNN execution."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).with_name('check_compile_retained_groups.py')
spec = importlib.util.spec_from_file_location('v67_generated_checker', SOURCE)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def toy_pair(x, index):
    frame = np.add(x, np.float32(index + 1))
    return [frame, np.add(frame.reshape(1, 106, 1024), np.float32(.125))]


def prepared(tmp_path, monkeypatch, *, group_index=0, mismatch=False, original_mismatch=False, reuse_output=False):
    indices, bindings, sources = C.group_spec(group_index)
    fake = ModuleType('onnxruntime')
    fake.__version__ = 'generated-only'
    fake.SessionOptions = SimpleNamespace
    fake.GraphOptimizationLevel = SimpleNamespace(ORT_DISABLE_ALL=0)
    calls = []
    shared_global = np.empty((1, 106, 1024), dtype=np.float32)

    class Session:
        def __init__(self, path, sess_options, providers):
            self.role = Path(path).name
            assert providers == ['CPUExecutionProvider'] and sess_options.graph_optimization_level == 0

        def get_inputs(self):
            return [SimpleNamespace(name='input')]

        def get_outputs(self):
            names = [b['name'] for b in bindings] if self.role == 'group' else ['frame_output', 'global_output']
            return [SimpleNamespace(name=n) for n in names]

        def run(self, names, feeds):
            calls.append((self.role, list(names)))
            x = feeds['input']
            if self.role != 'group':
                result = toy_pair(x, int(self.role[len('parent'):]))
                if reuse_output:
                    np.copyto(shared_global, result[1])
                    result[1] = shared_global
                return result
            target = None
            for index in indices:
                result = toy_pair(x, index)
                if index == 20 and group_index == 3:
                    target = result[1].copy()
                x = result[1].reshape(C.INPUT_SHAPE)
            if group_index == 3:
                result.append(target)
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
            bridge = x
            target = None
            for index in indices:
                values = toy_pair(bridge, index)
                if index == 20 and group_index == 3:
                    target = values[1].copy()
                bridge = values[1].reshape(C.INPUT_SHAPE)
            if group_index == 3:
                values.append(target)
            if original_mismatch:
                values[0] = values[0] + np.float32(.02)
            refs = [C.save_tensor(np, directory, f'ref-{j}.npy', v, bindings[j]) for j, v in enumerate(values)]
        cases.append(dict(tag=f'case{i}', directory=str(directory), input=input_row, original_outputs=refs))
    output = tmp_path / 'check'
    output.mkdir()
    report = dict(cases=cases, group_index=group_index, indices=indices, output_bindings=bindings,
        parent_graph_sha256=[str(i).zfill(2)*32 for i in indices])
    return output, report, calls


@pytest.mark.parametrize('group_index',range(4))
def test_complete_fake_chain_and_retained_outputs_per_group(monkeypatch, tmp_path, group_index):
    output, report, calls = prepared(tmp_path, monkeypatch, group_index=group_index)
    indices, bindings, _ = C.group_spec(group_index)
    C.run_check(Path('group'), [Path(f'parent{i}') for i in report['indices']], report['cases'], output, report)
    assert len(calls) == 2*(len(indices)+1) and report['all_separate_ONNX_checks_byte_equal']
    assert report['original_FP32_reference_case_count'] == 1
    assert report['original_FP32_gate_passed_where_reference_available']
    assert not report['original_FP32_references_cover_all_cases']
    assert all(len(c['group_outputs']) == len(c['chain_outputs']) == len(c['checks']) == len(bindings) for c in report['cases'])
    assert report['sessions_dropped']
    for case in report['cases']:
        assert case['actual_separate_pair_ORT_calls'] == len(indices)
        assert case['actual_original_global_to_frame_views'] == len(indices)-1
        assert case['no_capture_intermediate_substitution']
        steps = case['chain_steps']
        assert [s['index'] for s in steps] == indices
        assert steps[0]['input_raw_sha256'] == case['input']['raw_sha256']
        assert all(steps[i]['input_raw_sha256'] == steps[i-1]['global_output_raw_sha256'] for i in range(1,len(indices)))


def test_changed_global_retained_output_blocks_check_and_drops_sessions(monkeypatch, tmp_path):
    output, report, _ = prepared(tmp_path, monkeypatch, mismatch=True)
    with pytest.raises(ValueError, match='compilation blocked'):
        C.run_check(Path('group'), [Path(f'parent{i}') for i in report['indices']], report['cases'], output, report)
    assert not report['all_separate_ONNX_checks_byte_equal'] and report['sessions_dropped']
    assert not report['cases'][0]['checks'][1]['byte_equal_to_separate_FP32_ONNX_chain']



def test_target_global20_is_owned_distinct_and_wrong_target_blocks_compile(monkeypatch,tmp_path):
    output, report, _ = prepared(tmp_path,monkeypatch,group_index=3,mismatch=True,reuse_output=True)
    with pytest.raises(ValueError,match='compilation blocked'):
        C.run_check(Path('group'),[Path(f'parent{i}') for i in report['indices']],report['cases'],output,report)
    case=report['cases'][0]
    assert case['target_global_source_pair_index']==20
    assert not case['checks'][2]['byte_equal_to_separate_FP32_ONNX_chain']
    assert case['checks'][0]['byte_equal_to_separate_FP32_ONNX_chain']
    assert case['checks'][1]['byte_equal_to_separate_FP32_ONNX_chain']
    target=np.load(output/case['chain_outputs'][2]['file'],allow_pickle=False)
    final=np.load(output/case['chain_outputs'][1]['file'],allow_pickle=False)
    assert not np.array_equal(target,final)
    assert target.flat[0] == np.float32(60.375)
    assert report['sessions_dropped']

def test_original_gate_failure_retained_without_rejecting_unchanged_chain(monkeypatch, tmp_path):
    output, report, _ = prepared(tmp_path, monkeypatch, original_mismatch=True)
    C.run_check(Path('group'), [Path(f'parent{i}') for i in report['indices']], report['cases'], output, report)
    assert report['all_separate_ONNX_checks_byte_equal']
    assert not report['original_FP32_gate_passed_where_reference_available']
    assert report['status'] == 'checked_separate_ONNX_equivalence_not_quality_accepted'


def test_failed_check_precedes_sdk_import_and_build_failure_releases(monkeypatch, tmp_path):
    output, report, _ = prepared(tmp_path, monkeypatch)
    C.run_check(Path('group'), [Path(f'parent{i}') for i in report['indices']], report['cases'], output, report)
    report.update(schema='v67_original_VGGT_retained_group_check_compile', script_sha256=C.sha(SOURCE), manifest_sha256='manifest', graph_sha256='graph',
        cleanup_errors=[], source_and_inputs_unchanged=True)
    args = SimpleNamespace(check_report=output / 'check.json', sha256='manifest', output_dir=tmp_path, group_index=0)
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
    args.group_index = 1
    with pytest.raises(ValueError, match='before compilation'):
        C.run_compile(Path('group'), args, build_report)
    assert seen == []
    args.group_index = 0
    with pytest.raises(ValueError, match='build returned -1'):
        C.run_compile(Path('group'), args, build_report)
    assert seen == ['init', 'build', 'release'] and build_report['released']


def test_host_guard_precedes_optional_library_imports(tmp_path):
    if sys.platform != 'linux' or Path('/proc/device-tree/compatible').exists():
        pytest.skip('Expected host without RK3588 device tree')
    destination = tmp_path / 'must-not-exist'
    result = subprocess.run([sys.executable, '-B', str(SOURCE), '--manifest', str(tmp_path / 'none'),
        '--sha256', 'none', '--group-index', '0', '--mode', 'check', '--parent-manifest', str(tmp_path / 'none'),
        '--capture-dir', str(tmp_path), '--output-dir', str(destination)], text=True, capture_output=True)
    assert result.returncode == 2 and 'restricted to RK3588' in result.stderr
    assert not destination.exists()
