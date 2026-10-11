"""Complete-runner argument/dispatch doubles; no SDK, Torch or model execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    source = ROOT / 'deploy/rk3588/validation/original_full_aggregator/compare_complete_model.py'
    spec = importlib.util.spec_from_file_location('_generated_complete_runner', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('args,message', [
    (['--half-io-library', 'generated'], 'supplied together'),
    (['--half-io-library-sha256', 'd' * 64], 'supplied together'),
    (['--half-io-library', 'generated', '--half-io-library-sha256', 'd' * 64], 'requires --paired-aggregator'),
    (['--paired-aggregator', '--half-io-library', 'generated', '--half-io-library-sha256', 'BAD'], 'lowercase SHA256'),
])
def test_invalid_half_io_arguments_reject_before_board_or_loading(args, message):
    runner = load_runner()
    base = ['generated', '--manifest', 'generated', '--manifest-sha256', 'a' * 64, '--output', 'generated']
    error = io.StringIO()
    with patch.object(sys, 'argv', base + args), contextlib.redirect_stderr(error), \
            patch.object(Path, 'read_bytes', side_effect=AssertionError('must reject before board read')):
        with pytest.raises(SystemExit) as exc:
            runner.main()
    assert exc.value.code == 2 and message in error.getvalue()


@pytest.mark.parametrize('half_io,grouped,fail', [
    (False, False, 'aggregator'), (True, False, 'aggregator'),
    (True, True, 'grouped'), (True, True, 'predict'), (True, True, 'close_grouped'),
])
def test_backend_dispatch_and_failure_cleanup_remove_phase_hooks(tmp_path, half_io, grouped, fail):
    runner = load_runner()
    runner.BASE = tmp_path
    indices = list(range(32, 798, 48)) + [797]
    rows = [dict(timestamp=float(i), path=tmp_path / f'{i}.png') for i in range(798)]
    source_dir = tmp_path / 'src/vggt_slam'
    for name in ('patch_cache.py', 'rknn_aggregator_blocks.py', 'rknn_camera_head.py',
                 'rknn_dino_blocks.py', 'rknn_depth_head.py', 'rknn_aggregator_pairs.py',
                 'rknn_half_io_pairs.py', 'rknn_half_io_dino.py', 'native/fp16_neon.c',
                 'rknn_grouped_dino.py'):
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'# Generated grouped service double; never imported from disk.\n'
                           if name == 'rknn_grouped_dino.py' else (ROOT / 'vggt_slam' / name).read_bytes())
    reference = tmp_path / 'reference'
    (reference / 'cpu_fp32').mkdir(parents=True)
    (reference / 'result.json').write_text(json.dumps(dict(status='completed_not_quality_acceptance',
        input_indices=indices, input_timestamps=[float(i) for i in indices], image_shape=[2, 3, 84, 112],
        branches={'cpu_fp32': {'status': 'completed'}})))
    for i in range(1, 17):
        (reference / 'cpu_fp32' / f'pair-{i:02d}.npz').write_bytes(b'generated unopened reference placeholder')
    events = []
    def phase():
        return SimpleNamespace(register_forward_pre_hook=lambda _: SimpleNamespace(remove=lambda: events.append('remove')),
                               register_forward_hook=lambda _: SimpleNamespace(remove=lambda: events.append('remove')))
    model = SimpleNamespace(aggregator=phase(), camera_head=phase(), depth_head=phase())
    model.aggregator.patch_embed = phase()
    names = ('torch', 'vggt_slam.runtime', 'vggt_slam.solver', 'vggt_slam.patch_cache',
             'vggt_slam.timestamp_replay', 'vggt_slam.rknn_aggregator_pairs',
             'vggt_slam.rknn_half_io_pairs', 'vggt_slam.rknn_camera_head',
             'vggt_slam.rknn_grouped_dino', 'vggt_slam.rknn_dino_blocks',
             'vggt_slam.rknn_half_io_dino', 'vggt_slam.rknn_depth_head')
    modules = {name: ModuleType(name) for name in names}
    modules['torch'].set_num_threads = lambda n: events.append(('threads', n))
    modules['torch'].set_num_interop_threads = lambda n: None
    modules['torch'].inference_mode = contextlib.nullcontext
    modules['vggt_slam.runtime'].load_model = lambda *a, **k: model
    def fail_prediction(*args, **kwargs):
        events.append('predict')
        raise RuntimeError('generated prediction failure')
    modules['vggt_slam.solver'].Solver = lambda *a, **k: SimpleNamespace(run_predictions=fail_prediction)
    class Handle:
        def __init__(self, name):
            self.name = name
        def close(self):
            events.append('close_' + self.name)
            if fail == 'close_grouped' and self.name == 'grouped':
                raise RuntimeError('generated grouped close failure')
    def install_cache(*args):
        assert grouped and args == (model,)
        events.append('install_cache')
        return Handle('cache')
    modules['vggt_slam.patch_cache'].install_patch_cache = install_cache
    modules['vggt_slam.timestamp_replay'].read_rgb_index = lambda _: rows
    def install(kind):
        def function(*args, **kwargs):
            events.append((kind, args, kwargs))
            if (kind in ('ordinary', 'half') and fail == 'aggregator') or (kind == 'grouped' and fail == 'grouped'):
                raise RuntimeError('generated installer failure')
            return Handle('grouped' if kind == 'grouped' else 'aggregator')
        return function
    modules['vggt_slam.rknn_aggregator_pairs'].install_rknn_aggregator_pairs = install('ordinary')
    modules['vggt_slam.rknn_half_io_pairs'].install_half_io_pairs = install('half')
    def install_head(name):
        def function(*args, **kwargs):
            assert grouped and args[0] is model
            events.append('install_' + name)
            return Handle(name)
        return function
    modules['vggt_slam.rknn_camera_head'].install_rknn_camera_head = install_head('camera')
    modules['vggt_slam.rknn_depth_head'].install_rknn_depth_head = install_head('depth')
    modules['vggt_slam.rknn_grouped_dino'].install_grouped_dino = install('grouped')
    modules['vggt_slam.rknn_dino_blocks'].install_rknn_dino_blocks = lambda *a, **k: pytest.fail('wrong DINO dispatch')
    modules['vggt_slam.rknn_half_io_dino'].install_half_io_dino = lambda *a, **k: pytest.fail('wrong half DINO dispatch')
    arguments = ['generated', '--manifest', str(tmp_path / 'manifest.placeholder'), '--manifest-sha256', 'a' * 64,
        '--paired-aggregator', '--model-threads', '1', '--output', str(tmp_path / 'out'),
        '--dino-manifest', 'generated', '--dino-manifest-sha256', 'b' * 64,
        '--reference-run', str(reference)]
    if half_io:
        arguments += ['--half-io-library', str(tmp_path / 'neon.placeholder'), '--half-io-library-sha256', 'd' * 64]
    if grouped:
        arguments += ['--grouped-dino', '--depth-manifest', 'generated', '--depth-manifest-sha256', 'c' * 64]
    original_read_bytes = Path.read_bytes
    def read_bytes(path):
        return b'rockchip,rk3588s' if str(path) == '/proc/device-tree/compatible' else original_read_bytes(path)
    with patch.dict(sys.modules, modules), patch.object(sys, 'argv', arguments), \
            patch.object(Path, 'read_bytes', read_bytes), patch.object(runner.os, 'sched_setaffinity'), \
            contextlib.redirect_stdout(io.StringIO()):
        message = ('Backend cleanup failed' if fail == 'close_grouped' else
                   'generated prediction failure' if fail == 'predict' else 'generated installer failure')
        with pytest.raises(RuntimeError, match=message):
            runner.main()
    call = next(event for event in events if isinstance(event, tuple) and event[0] in ('ordinary', 'half'))
    assert call[0] == ('half' if half_io else 'ordinary')
    assert call[1] == (model, tmp_path / 'manifest.placeholder', 'a' * 64)
    assert call[2] == (dict(half_io_library=tmp_path / 'neon.placeholder', half_io_library_sha256='d' * 64) if half_io else {})
    assert events.count('remove') == 8
    report = json.loads((tmp_path / 'out/result.json').read_text())
    assert report['status'] == 'failed'
    assert bool(report['cleanup_errors']) is (fail == 'close_grouped')
    assert report['quality_accepted'] is False and report['realtime_validated'] is False
    assert report.get('half_io_experiment', False) is half_io
    assert report['grouped_dino'] is grouped
    if grouped:
        grouped_call = next(event for event in events if isinstance(event, tuple) and event[0] == 'grouped')
        assert grouped_call[1] == (model, Path('generated'), 'b' * 64)
        assert grouped_call[2] == dict(half_io_library=tmp_path / 'neon.placeholder', half_io_library_sha256='d' * 64)
        assert 'rknn_grouped_dino.py' in report['runtime_source_sha256']
        closes = [event for event in events if isinstance(event, str) and event.startswith('close_')]
        assert closes == (['close_camera', 'close_aggregator'] if fail == 'grouped' else
                          ['close_cache', 'close_depth', 'close_grouped', 'close_camera', 'close_aggregator'])
        assert events[-8:] == ['remove'] * 8
        assert ('predict' in events) is (fail != 'grouped')
    else:
        assert 'rknn_grouped_dino.py' not in report['runtime_source_sha256']


@pytest.mark.parametrize('args,message', [
    (['--grouped-dino'], 'requires --dino-manifest and --half-io-library'),
    (['--grouped-dino', '--dino-manifest', 'generated', '--dino-manifest-sha256', 'b' * 64],
     'requires --dino-manifest and --half-io-library'),
    (['--grouped-dino', '--half-io-library', 'generated', '--half-io-library-sha256', 'd' * 64],
     'requires --dino-manifest and --half-io-library'),
    (['--grouped-dino', '--dino-manifest', 'generated', '--dino-manifest-sha256', 'b' * 64,
      '--half-io-library', 'generated', '--half-io-library-sha256', 'd' * 64], 'requires --paired-aggregator'),
])
def test_grouped_dependencies_reject_before_board_or_model(args, message):
    runner = load_runner()
    base = ['generated', '--manifest', 'generated', '--manifest-sha256', 'a' * 64, '--output', 'generated']
    error = io.StringIO()
    with patch.object(sys, 'argv', base + args), contextlib.redirect_stderr(error), \
            patch.object(Path, 'read_bytes', side_effect=AssertionError('must reject before board read')):
        with pytest.raises(SystemExit) as exc:
            runner.main()
    assert exc.value.code == 2 and message in error.getvalue()
