"""Generated service-double checks only; no Torch, VGGT, RKNN or board run."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCES = ('patch_cache.py', 'rknn_aggregator_blocks.py', 'rknn_camera_head.py',
                   'rknn_dino_blocks.py', 'rknn_depth_head.py', 'rknn_aggregator_pairs.py')
HALF_IO_SOURCES = ('rknn_half_io_pairs.py', 'native/fp16_neon.c')


def load_replay():
    path = ROOT / 'scripts' / 'replay_original_vggt.py'
    spec = importlib.util.spec_from_file_location('_generated_replay', path)
    module = importlib.util.module_from_spec(spec)
    # The production script disables bytecode globally; keep this test isolated.
    original_bytecode_setting = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original_bytecode_setting
    return module


def generated_run(*, npu=False, paired=False, camera=False, dino=False, depth=False,
                  half_io=False, no_cache=False, fail=None, model_threads=None):
    replay = load_replay()
    events = []
    modules = {name: ModuleType(name) for name in (
        'cv2', 'torch', 'vggt_slam', 'vggt_slam.runtime', 'vggt_slam.solver',
        'vggt_slam.timestamp_replay', 'vggt_slam.edge_mapping',
        'vggt_slam.patch_cache', 'vggt_slam.rknn_aggregator_blocks', 'vggt_slam.rknn_camera_head',
        'vggt_slam.rknn_dino_blocks', 'vggt_slam.rknn_depth_head',
        'vggt_slam.rknn_aggregator_pairs', 'vggt_slam.rknn_half_io_pairs')}
    modules['vggt_slam'].__path__ = []
    modules['torch'].__version__ = 'generated-service-double'
    for name in ('set_num_threads', 'set_num_interop_threads'):
        setattr(modules['torch'], name, lambda value, field=name: events.append(f'{field}={value}'))
    modules['cv2'].setNumThreads = lambda _: None

    class Handle:
        def __init__(self, name):
            self.name, self.closed = name, False

        def close(self):
            events.append('close_' + self.name)
            self.closed = True
            if fail == 'close_' + self.name:
                raise RuntimeError('generated close failure')

        def metrics(self):
            return {'generated_service_double': True, 'closed': self.closed}

    class Model:
        def __init__(self):
            self.before = self.after = None

        def register_forward_pre_hook(self, hook):
            self.before = hook
            return Handle('pre_hook')

        def register_forward_hook(self, hook):
            self.after = hook
            return Handle('post_hook')

    Handle.remove = Handle.close
    model = Model()
    modules['vggt_slam.runtime'].load_model = lambda *a, **k: model

    def install_rknn(*args, **kwargs):
        events.append('install_rknn')
        if fail == 'install_rknn':
            raise RuntimeError('generated install failure')
        assert args[0] is model and args[2] == 'a' * 64
        assert kwargs['source_root'] == replay.BASE / 'src'
        return Handle('rknn')

    def install_paired_rknn(*args, **kwargs):
        events.append('install_paired_rknn')
        if fail == 'install_paired_rknn':
            raise RuntimeError('generated pair install failure')
        assert args[0] is model and args[2] == 'a' * 64
        assert kwargs['source_root'] == replay.BASE / 'src'
        return Handle('rknn')

    def install_cache(arg):
        assert arg is model
        events.append('install_cache')
        return Handle('cache')

    def install_half_io(*args, **kwargs):
        events.append('install_half_io_rknn')
        if fail == 'install_half_io_rknn':
            raise RuntimeError('generated half I/O install failure')
        assert args[0] is model and args[2] == 'a' * 64
        assert kwargs == dict(source_root=replay.BASE / 'src',
            half_io_library=replay.BASE.parent / 'neon.placeholder',
            half_io_library_sha256='d' * 64)
        return Handle('rknn')

    def install_camera(arg):
        assert arg is model
        events.append('install_camera_rknn')
        if fail == 'install_camera_rknn':
            raise RuntimeError('generated camera install failure')
        return Handle('camera_rknn')

    modules['vggt_slam.rknn_aggregator_blocks'].install_rknn_aggregator_blocks = install_rknn
    modules['vggt_slam.rknn_aggregator_pairs'].install_rknn_aggregator_pairs = install_paired_rknn
    modules['vggt_slam.rknn_half_io_pairs'].install_half_io_pairs = install_half_io
    modules['vggt_slam.patch_cache'].install_patch_cache = install_cache
    modules['vggt_slam.rknn_camera_head'].install_rknn_camera_head = install_camera
    def install_dino(*args, **kwargs):
        assert args[0] is model and args[2] == 'b' * 64
        assert kwargs['source_root'] == replay.BASE / 'src'
        events.append('install_dino_rknn')
        if fail == 'install_dino_rknn':
            raise RuntimeError('generated DINO install failure')
        return Handle('dino_rknn')
    modules['vggt_slam.rknn_dino_blocks'].install_rknn_dino_blocks = install_dino
    def install_depth(*args, **kwargs):
        assert args[0] is model and args[2] == 'c' * 64
        assert kwargs['source_root'] == replay.BASE / 'src'
        events.append('install_depth_rknn')
        if fail == 'install_depth_rknn':
            raise RuntimeError('generated depth install failure')
        return Handle('depth_rknn')
    modules['vggt_slam.rknn_depth_head'].install_rknn_depth_head = install_depth
    rows = [dict(timestamp=float(i), path=f'/generated/{i}.png') for i in (1, 2, 3)]
    modules['vggt_slam.timestamp_replay'].read_rgb_index = lambda _: rows

    class Replay(Handle):
        def __init__(self, given, *, cpus):
            super().__init__('replay')
            assert given is rows and cpus == {0}
            self.index = 0

        def start(self):
            self.start_monotonic = time.monotonic() - 1.

        def read(self):
            if self.index == len(rows):
                return None
            now = time.monotonic()
            item = dict(rows[self.index], frame='generated decoded pixel placeholder',
                        sequence=self.index, captured_monotonic=now-.001,
                        decode_started_monotonic=now-.0008, ready_monotonic=now-.0007)
            self.index += 1
            return item

        @property
        def exhausted(self):
            return self.index == len(rows)

    modules['vggt_slam.timestamp_replay'].TimestampReplay = Replay

    class Solver:
        def __init__(self, **kwargs):
            assert kwargs['max_loops'] == 0 and kwargs['offline']
            self.map = SimpleNamespace(submaps={}, write_poses_to_file=self.write_poses)
            self.graph = SimpleNamespace(optimize=lambda: None,
                graph=SimpleNamespace(error=lambda _: 0.), values=SimpleNamespace(size=lambda: 4),
                get_homography=lambda _: np.eye(4))
            self.calls = 0

        def run_predictions(self, paths, arg_model, *, max_loops):
            assert arg_model is model and max_loops == 0
            events.append('predict')
            if fail == 'predict':
                raise RuntimeError('generated prediction failure')
            model.before(model, ())
            self.calls += 1
            index = (self.calls - 1) * 2
            self.current_working_submap = SimpleNamespace(
                frame_ids=[float(Path(x).stem) for x in paths],
                pointclouds=[np.zeros((1, 3)), np.ones((1, 3))],
                colors=[np.zeros((1, 3), np.uint8)] * 2,
                conf_masks=[np.array([30.])] * 2, conf_threshold=25., get_id=lambda: index)
            model.after(model, (), {})
            return {'generated': True}

        def add_points(self, prediction):
            assert prediction == {'generated': True}
            submap = self.current_working_submap
            self.map.submaps[submap.get_id()] = submap

        @staticmethod
        def write_poses(path, graph, *, kitti_format):
            assert not kitti_format
            Path(path).write_text('# generated pose placeholder\n')

    modules['vggt_slam.solver'].Solver = Solver
    modules['vggt_slam.edge_mapping'].platform_sample = lambda: {'generated_service_double': True}
    modules['vggt_slam.edge_mapping'].atomic_json = lambda path, data: path.write_text(json.dumps(data))
    original_is_file, original_read_bytes = Path.is_file, Path.read_bytes
    def is_file(path):
        return str(path) == '/proc/device-tree/compatible' or original_is_file(path)
    def read_bytes(path):
        return b'rockchip,rk3588s\0' if str(path) == '/proc/device-tree/compatible' else original_read_bytes(path)
    generated_affinity = set(range(8))
    def get_affinity(pid):
        assert pid == 0
        return generated_affinity.copy()
    def set_affinity(pid, cpus):
        assert pid == 0
        generated_affinity.clear()
        generated_affinity.update(cpus)
        events.append('set_affinity=' + ','.join(map(str, sorted(cpus))))
    original_sys_path = sys.path[:]
    try:
        with tempfile.TemporaryDirectory(prefix='generated-replay-check-') as temporary:
            root = Path(temporary)
            replay.BASE = root / 'board-source-placeholder'
            runtime_directory = replay.BASE / 'src/vggt_slam'
            runtime_directory.mkdir(parents=True)
            # Hash real source bytes in the generated board tree; imports still
            # resolve exclusively to the service doubles above.
            source_hashes = {}
            for name in RUNTIME_SOURCES + (HALF_IO_SOURCES if half_io else ()):
                content = (ROOT / 'vggt_slam' / name).read_bytes()
                (runtime_directory / name).parent.mkdir(parents=True, exist_ok=True)
                (runtime_directory / name).write_bytes(content)
                if name != 'rknn_aggregator_pairs.py' or paired:
                    source_hashes[name] = hashlib.sha256(content).hexdigest()
            args = ['generated-replay', '--rgb-index', str(root/'rgb.txt'),
                    '--checkpoint', str(root/'checkpoint.placeholder'), '--output', str(root/'out')]
            if npu:
                args += ['--rknn-manifest', str(root/'manifest.placeholder'), '--rknn-manifest-sha256', 'a'*64]
            if paired:
                args += ['--rknn-paired-aggregator']
            if half_io:
                args += ['--half-io-library', str(root / 'neon.placeholder'),
                         '--half-io-library-sha256', 'd' * 64]
            if no_cache:
                args += ['--no-patch-cache']
            if camera:
                args += ['--rknn-camera-head']
            if dino:
                args += ['--rknn-dino-manifest', str(root/'dino.placeholder'),
                         '--rknn-dino-manifest-sha256', 'b'*64]
            if depth:
                args += ['--rknn-depth-manifest', str(root/'depth.placeholder'),
                         '--rknn-depth-manifest-sha256', 'c'*64]
            if model_threads is not None:
                args += ['--model-threads', str(model_threads)]
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', args), \
                    patch.object(replay.platform, 'machine', return_value='aarch64'), \
                    patch.object(replay.sys, 'platform', 'linux'), \
                    patch.object(Path, 'is_file', is_file), patch.object(Path, 'read_bytes', read_bytes), \
                    patch.object(replay.os, 'sched_getaffinity', get_affinity), \
                    patch.object(replay.os, 'sched_setaffinity', set_affinity), contextlib.redirect_stdout(io.StringIO()):
                code = replay.main()
            report = json.loads((root/'out/results.json').read_text())
            assert report['runtime_source_sha256'] == source_hashes
            assert report['rknn_paired_aggregator'] is paired
            if half_io:
                assert report['half_io_experiment'] is True and report['internal_precision_changed'] is False
                assert report['half_io_library_sha256'] == 'd' * 64
                assert report['half_io_source_sha256'] == source_hashes['native/fp16_neon.c']
            else:
                assert 'half_io_experiment' not in report
    finally:
        sys.path[:] = original_sys_path
    return code, report, events


@pytest.mark.parametrize('npu,camera,no_cache', [
    (False, False, False), (True, False, False), (True, False, True),
    (True, True, False), (True, True, True),
], ids=['cpu-cache', 'aggregator-cache', 'aggregator', 'aggregator-camera-cache', 'aggregator-camera'])
def test_generated_lifecycle_and_complete_metrics(npu, camera, no_cache):
    code, report, events = generated_run(npu=npu, camera=camera, no_cache=no_cache)
    assert code == 0 and report['status'] == 'completed' and report['fresh_pose_count'] == 2
    assert report['candidate_accepted'] is False and report['trajectory_quality_validated'] is False
    expected = ('NPU_FP16_aggregator_camera_CPU_FP32_DINO_depth' if camera else
                'NPU_FP16_aggregator_CPU_FP32_DINO_heads' if npu else 'CPU_FP32')
    assert report['precision'] == expected
    assert report['rknn_camera_head'] is camera
    assert report['model_threads'] == 4 and report['cpu_affinity'] == [4, 5, 6, 7]
    assert events.count('set_num_threads=4') == 1
    for row in report['updates']:
        for key in ('decode_seconds', 'queue_seconds', 'full_model_seconds', 'prediction_seconds',
                    'SL4_and_points_seconds', 'publication_seconds', 'processing_seconds',
                    'input_to_pose_seconds', 'graph_error', 'graph_nodes'):
            assert key in row
    assert 'fresh_pose_hz' in report and 'input_to_pose_p95_seconds' in report
    if npu:
        assert report['rknn']['closed'] and events.index('close_replay') < events.index('close_rknn')
        if not no_cache:
            assert events.index('install_rknn') < events.index('install_cache')
            assert events.index('close_cache') < events.index('close_rknn')
    else:
        assert 'rknn' not in report and 'install_rknn' not in events
    if camera:
        assert report['camera_rknn']['closed']
        assert events.index('install_rknn') < events.index('install_camera_rknn')
        assert events.index('close_replay') < events.index('close_camera_rknn') < events.index('close_rknn')
        if not no_cache:
            assert events.index('install_camera_rknn') < events.index('install_cache')
            assert events.index('close_cache') < events.index('close_camera_rknn')
    else:
        assert 'camera_rknn' not in report and 'install_camera_rknn' not in events


@pytest.mark.parametrize('threads,cpus', [(1, [4, 5, 6, 7]), (2, [6, 7])])
def test_generated_model_threads_preserve_decoder_and_restore_affinity(threads, cpus):
    code, report, events = generated_run(npu=True, camera=True, model_threads=threads)
    assert code == 0 and report['status'] == 'completed'
    assert report['model_threads'] == threads and report['cpu_affinity'] == cpus
    assert report['decoder_cpu_affinity'] == [0]
    assert events.count(f'set_num_threads={threads}') == 1 and 'set_num_threads=4' not in events
    assert events.count('set_num_interop_threads=1') == 1
    affinity_calls = [event for event in events if event.startswith('set_affinity=')]
    assert affinity_calls == ['set_affinity=' + ','.join(map(str, cpus)), 'set_affinity=0,1,2,3,4,5,6,7']


@pytest.mark.parametrize('fail', ['install_rknn', 'predict', 'close_cache'])
def test_generated_failures_restore_installed_handles(fail):
    code, report, events = generated_run(npu=True, fail=fail)
    assert code == 1 and report['status'] == 'failed'
    if fail == 'install_rknn':
        assert 'install_cache' not in events and 'predict' not in events
    else:
        assert events.index('close_cache') < events.index('close_rknn')
        assert report['rknn']['closed']


def test_generated_camera_installation_failure_restores_aggregator():
    code, report, events = generated_run(npu=True, camera=True, fail='install_camera_rknn')
    assert code == 1 and report['status'] == 'failed'
    assert events.index('install_rknn') < events.index('install_camera_rknn') < events.index('close_rknn')
    assert report['rknn']['closed'] and 'install_cache' not in events and 'predict' not in events


@pytest.mark.parametrize('fail', [None, 'install_dino_rknn', 'predict'])
def test_generated_dino_lifecycle_with_all_backends(fail):
    code, report, events = generated_run(npu=True, camera=True, dino=True, fail=fail)
    assert code == (0 if fail is None else 1)
    assert report['precision'] == 'NPU_FP16_DINO_aggregator_camera_remaining_CPU_FP32'
    assert report['rknn']['closed'] and report['camera_rknn']['closed']
    if fail == 'install_dino_rknn':
        assert 'install_cache' not in events and 'predict' not in events
        assert 'dino_rknn' not in report
    else:
        assert report['dino_rknn']['closed']
        assert events.index('install_dino_rknn') < events.index('install_cache')
        assert events.index('close_replay') < events.index('close_cache') < events.index('close_dino_rknn')
        assert events.index('close_dino_rknn') < events.index('close_camera_rknn') < events.index('close_rknn')


@pytest.mark.parametrize('fail', [None, 'install_depth_rknn', 'predict'])
def test_generated_depth_lifecycle_with_all_backends(fail):
    code, report, events = generated_run(npu=True, camera=True, dino=True, depth=True, fail=fail)
    assert code == (0 if fail is None else 1)
    assert report['precision'] == 'NPU_FP16_DINO_aggregator_camera_depth_logits_remaining_CPU_FP32'
    assert all(report[name]['closed'] for name in ('rknn', 'camera_rknn', 'dino_rknn'))
    if fail == 'install_depth_rknn':
        assert 'install_cache' not in events and 'predict' not in events
        assert 'depth_rknn' not in report
    else:
        assert report['depth_rknn']['closed']
        assert events.index('install_dino_rknn') < events.index('install_depth_rknn') < events.index('install_cache')
        assert events.index('close_cache') < events.index('close_depth_rknn') < events.index('close_dino_rknn')


@pytest.mark.parametrize('fail', [None, 'predict'])
def test_generated_paired_aggregator_lifecycle_with_all_backends(fail):
    code, report, events = generated_run(npu=True, paired=True, camera=True,
                                         dino=True, depth=True, fail=fail)
    assert code == (0 if fail is None else 1)
    assert report['status'] == ('completed' if fail is None else 'failed')
    assert report['precision'] == 'NPU_FP16_DINO_aggregator_camera_depth_logits_remaining_CPU_FP32'
    assert report['candidate_accepted'] is False and report['trajectory_quality_validated'] is False
    assert all(report[name]['closed'] for name in ('rknn', 'camera_rknn', 'dino_rknn', 'depth_rknn', 'cache'))
    assert [event for event in events if event.startswith('install_')] == [
        'install_paired_rknn', 'install_camera_rknn', 'install_dino_rknn',
        'install_depth_rknn', 'install_cache']
    assert [event for event in events if event.startswith('close_') and event not in
            ('close_pre_hook', 'close_post_hook')] == [
        'close_replay', 'close_cache', 'close_depth_rknn', 'close_dino_rknn',
        'close_camera_rknn', 'close_rknn']


@pytest.mark.parametrize('args,message', [
    (['--rknn-manifest', 'generated'], 'supplied together'),
    (['--rknn-camera-head'], 'requires --rknn-manifest'),
    (['--rknn-paired-aggregator'], 'requires --rknn-manifest'),
    (['--rknn-manifest-sha256', 'a'*64], 'supplied together'),
    (['--rknn-manifest', 'generated', '--rknn-manifest-sha256', 'BAD'], 'lowercase SHA256'),
    (['--rknn-manifest', 'generated', '--rknn-manifest-sha256', 'a'*64, '--image-size', '336'], 'image-size 112'),
], ids=['missing-hash', 'camera-without-manifest', 'pair-without-manifest',
        'missing-manifest', 'malformed-hash', 'wrong-image-size'])
def test_generated_invalid_backend_arguments_rejected_before_loading(args, message):
    base = ['generated-replay', '--rgb-index', 'generated', '--checkpoint', 'generated', '--output', 'generated']
    replay = load_replay()
    stderr = io.StringIO()
    with patch.object(sys, 'argv', base+args), contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as exc:
            replay.main()
    assert exc.value.code == 2 and message in stderr.getvalue()


@pytest.mark.parametrize('fail', [None, 'install_half_io_rknn', 'predict'])
def test_generated_half_io_lifecycle_and_unchanged_precision(fail):
    code, report, events = generated_run(npu=True, paired=True, camera=True, dino=True,
                                         depth=True, half_io=True, fail=fail)
    assert code == (0 if fail is None else 1)
    assert report['precision'] == 'NPU_FP16_DINO_aggregator_camera_depth_logits_remaining_CPU_FP32'
    assert report['candidate_accepted'] is False and report['trajectory_quality_validated'] is False
    assert 'install_paired_rknn' not in events and 'install_rknn' not in events
    if fail == 'install_half_io_rknn':
        assert 'install_cache' not in events and 'predict' not in events
    else:
        assert [event for event in events if event.startswith('install_')] == [
            'install_half_io_rknn', 'install_camera_rknn', 'install_dino_rknn',
            'install_depth_rknn', 'install_cache']
        assert [event for event in events if event.startswith('close_') and event not in
                ('close_pre_hook', 'close_post_hook')] == [
            'close_replay', 'close_cache', 'close_depth_rknn', 'close_dino_rknn',
            'close_camera_rknn', 'close_rknn']


@pytest.mark.parametrize('args,message', [
    (['--half-io-library', 'generated'], 'supplied together'),
    (['--half-io-library-sha256', 'd' * 64], 'supplied together'),
    (['--half-io-library', 'generated', '--half-io-library-sha256', 'd' * 64],
     'requires --rknn-paired-aggregator'),
    (['--rknn-paired-aggregator', '--half-io-library', 'generated',
      '--half-io-library-sha256', 'BAD'], 'lowercase SHA256'),
])
def test_generated_half_io_arguments_rejected_before_loading(args, message):
    replay = load_replay()
    stderr = io.StringIO()
    base = ['generated-replay', '--rgb-index', 'generated', '--checkpoint', 'generated', '--output', 'generated']
    with patch.object(sys, 'argv', base + args), contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as exc:
            replay.main()
    assert exc.value.code == 2 and message in stderr.getvalue()
