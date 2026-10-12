"""Generated replay service doubles only; no model, Torch, SDK or madvise execution."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('aggregator_installed', 'camera_installed', 'DINO_installed', 'depth_installed')
SOURCES = ('patch_cache.py', 'rknn_aggregator_blocks.py', 'rknn_camera_head.py',
           'rknn_dino_blocks.py', 'rknn_depth_head.py', 'rknn_aggregator_pairs.py',
           'rknn_half_io_pairs.py', 'native/fp16_neon.c', 'rknn_half_io_dino.py',
           'rknn_grouped_dino.py', 'rknn_retained_aggregator.py',
           'checkpoint_residency.py', 'runtime.py', 'checkpoint_identity.py')


def load_replay():
    spec = importlib.util.spec_from_file_location('_checkpoint_replay_double',
                                                 ROOT / 'scripts/replay_original_vggt.py')
    module = importlib.util.module_from_spec(spec)
    before = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def arguments(root, *, enabled=True, omit=(), no_cache=False):
    options = {'--rknn-manifest': str(root/'agg'), '--rknn-manifest-sha256': 'a'*64,
               '--rknn-paired-aggregator': None, '--rknn-retained-aggregator': None,
               '--half-io-library': str(root/'neon'), '--half-io-library-sha256': 'd'*64,
               '--rknn-camera-head': None, '--rknn-camera-split4': None,
               '--rknn-dino-manifest': str(root/'dino'), '--rknn-dino-manifest-sha256': 'b'*64,
               '--rknn-grouped-dino': None, '--rknn-depth-manifest': str(root/'depth'),
               '--rknn-depth-manifest-sha256': 'c'*64}
    args = ['generated', '--rgb-index', str(root/'rgb.txt'), '--checkpoint', str(root/'model.fixture'),
            '--output', str(root/'output'), '--model-threads', '1']
    for option, value in options.items():
        if option not in omit:
            args.append(option)
            if value is not None:
                args.append(value)
    if enabled:
        args.append('--checkpoint-pageout')
    if no_cache:
        args.append('--no-patch-cache')
    return args


def generated_run(root, *, enabled=True, fail=None):
    replay = load_replay()
    replay.BASE = root/'board'
    directory = replay.BASE/'src/vggt_slam'
    directory.mkdir(parents=True)
    for name in SOURCES:
        target = directory/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/'vggt_slam'/name).read_bytes())
    events, live = [], set()
    now = [1000.]
    def clock():
        now[0] += .001
        return now[0]
    replay.time = SimpleNamespace(monotonic=clock)
    names = ('cv2', 'torch', 'vggt_slam', 'vggt_slam.runtime', 'vggt_slam.solver',
             'vggt_slam.timestamp_replay', 'vggt_slam.edge_mapping', 'vggt_slam.patch_cache',
             'vggt_slam.rknn_retained_aggregator', 'vggt_slam.rknn_camera_head',
             'vggt_slam.rknn_grouped_dino', 'vggt_slam.rknn_depth_head',
             'vggt_slam.checkpoint_residency')
    modules = {name: ModuleType(name) for name in names}
    modules['vggt_slam'].__path__ = []
    modules['cv2'].setNumThreads = lambda _: None
    modules['torch'].__version__ = 'generated service double'
    modules['torch'].set_num_threads = modules['torch'].set_num_interop_threads = lambda _: None

    class Handle:
        def __init__(self, name):
            self.name, self.closed = name, False
            live.add(name)
        def close(self):
            events.append('close_'+self.name)
            live.remove(self.name)
            self.closed = True
        def remove(self):
            events.append('remove_'+self.name)
        def metrics(self):
            return dict(closed=self.closed, generated_service_double=True)

    class Model:
        def register_forward_pre_hook(self, fn):
            self.before = fn
            return SimpleNamespace(remove=lambda: events.append('remove_pre_hook'))
        def register_forward_hook(self, fn):
            self.after = fn
            return SimpleNamespace(remove=lambda: events.append('remove_post_hook'))
    model = Model()
    def load_model(*args, **kwargs):
        events.append('verified_load_double')
        return model
    modules['vggt_slam.runtime'].load_model = load_model
    def install(name):
        def function(arg, *args, **kwargs):
            assert arg is model
            events.append('install_'+name)
            if fail == 'install_'+name:
                raise RuntimeError('generated installation failure')
            if name == 'camera':
                assert kwargs == dict(split_fc1=True)
            return Handle(name)
        return function
    for module, function, name in (
            ('rknn_retained_aggregator', 'install_retained_aggregator', 'aggregator'),
            ('rknn_camera_head', 'install_rknn_camera_head', 'camera'),
            ('rknn_grouped_dino', 'install_grouped_dino', 'DINO'),
            ('rknn_depth_head', 'install_rknn_depth_head', 'depth'),
            ('patch_cache', 'install_patch_cache', 'cache')):
        setattr(modules['vggt_slam.'+module], function, install(name))
    stat_calls, snapshot_calls = [], []
    def checkpoint_stat(path):
        stat_calls.append(path)
        events.append('checkpoint_stat')
        return dict(fixture_identity=2 if fail == 'stat_changed' and len(stat_calls)==2 else 1)
    def snapshot(path, stat):
        assert path == root/'model.fixture' and stat == dict(fixture_identity=1)
        label = 'startup' if not snapshot_calls else 'precleanup'
        snapshot_calls.append(label)
        events.append('snapshot_'+label)
        if fail == 'snapshot_'+label:
            raise RuntimeError('generated snapshot failure')
        now[0] += .25
        return dict(generated_service_double=True, snapshot_monotonic=now[0])
    def pageout(arg, path, stat, receipt_path, *, source_root, stage):
        assert arg is model and stat == dict(fixture_identity=1)
        assert path == root/'model.fixture' and source_root == replay.BASE/'src'
        assert stage in STAGES and stage.removesuffix('_installed') in live
        persisted = json.loads((root/'output/results.json').read_text())
        assert persisted['checkpoint_pageout_stages'][-1]['stage'] == stage
        assert persisted['checkpoint_pageout_stages'][-1]['status'] == 'running'
        events.append('pageout_'+stage)
        result = dict(stage=stage, status='generated_advice_completed', seconds=1.,
                      advice_calls=[dict(seconds=.9)], tensor_storage_VA_version_unchanged=True,
                      checkpoint_VMA_layout_unchanged=True,
                      before=dict(smaps_rollup_kB=dict(Rss=4096)),
                      after_immediate=dict(smaps_rollup_kB=dict(Rss=2048)))
        if fail == stage:
            result['status'] = 'generated_failed_advice'
        receipt_path.write_text(json.dumps(result))
        now[0] += 1.
        if fail == stage:
            raise RuntimeError('generated advice failure')
        return result
    helper = modules['vggt_slam.checkpoint_residency']
    helper.checkpoint_stat, helper.memory_snapshot, helper.pageout_checkpoint = checkpoint_stat, snapshot, pageout
    rows = [dict(timestamp=float(i), path=f'/generated/{i}.png') for i in (1, 2, 3)]
    modules['vggt_slam.timestamp_replay'].read_rgb_index = lambda _: rows
    class Replay(Handle):
        def __init__(self, given, *, cpus):
            assert given is rows and cpus == {0}
            super().__init__('replay')
            self.index = 0
        def start(self):
            self.start_monotonic = clock()
            events.append('start_replay')
        def read(self):
            if self.index == len(rows):
                return None
            t = clock()
            item = dict(rows[self.index], sequence=self.index, frame=None,
                        captured_monotonic=t-.001, decode_started_monotonic=t-.0008,
                        ready_monotonic=t-.0007)
            self.index += 1
            return item
        @property
        def exhausted(self):
            return self.index == len(rows)
    modules['vggt_slam.timestamp_replay'].TimestampReplay = Replay
    class Solver:
        def __init__(self, **kwargs):
            assert kwargs['max_loops'] == 0 and kwargs['offline']
            self.map = SimpleNamespace(submaps={}, write_poses_to_file=lambda path, *_a, **_k:
                                       Path(path).write_text('# generated pose fixture\n'))
            self.graph = SimpleNamespace(optimize=lambda: None, graph=SimpleNamespace(error=lambda _: 0.),
                                         values=SimpleNamespace(size=lambda: 4), get_homography=lambda _: np.eye(4))
            self.calls = 0
        def run_predictions(self, paths, arg, *, max_loops):
            assert arg is model and max_loops == 0
            events.append('predict')
            model.before(model, ())
            now[0] += .01
            model.after(model, (), {})
            node = self.calls*2
            self.calls += 1
            self.current_working_submap = SimpleNamespace(
                frame_ids=[float(Path(p).stem) for p in paths], pointclouds=[np.ones((1, 3))]*2,
                colors=[np.zeros((1, 3), np.uint8)]*2, conf_masks=[np.array([30.])]*2,
                conf_threshold=25., get_id=lambda: node)
            return {}
        def add_points(self, prediction):
            submap = self.current_working_submap
            self.map.submaps[submap.get_id()] = submap
    modules['vggt_slam.solver'].Solver = Solver
    def atomic_json(path, value):
        events.append('publish' if path.name == 'map.json' else 'persist_stage')
        path.write_text(json.dumps(value))
    modules['vggt_slam.edge_mapping'].atomic_json = atomic_json
    modules['vggt_slam.edge_mapping'].platform_sample = lambda: {}
    real_is_file, real_read_bytes = Path.is_file, Path.read_bytes
    def is_file(path):
        return str(path) == '/proc/device-tree/compatible' or real_is_file(path)
    def read_bytes(path):
        return b'rockchip,rk3588s\0' if str(path) == '/proc/device-tree/compatible' else real_read_bytes(path)
    affinity = set(range(8))
    def set_affinity(pid, value):
        affinity.clear()
        affinity.update(value)
        events.append('affinity_'+','.join(map(str, sorted(value))))
    before_path = sys.path[:]
    try:
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', arguments(root, enabled=enabled)), \
                patch.object(replay.platform, 'machine', return_value='aarch64'), \
                patch.object(Path, 'is_file', is_file), patch.object(Path, 'read_bytes', read_bytes), \
                patch.object(replay.os, 'sched_getaffinity', side_effect=lambda _: affinity.copy()), \
                patch.object(replay.os, 'sched_setaffinity', side_effect=set_affinity), \
                contextlib.redirect_stdout(io.StringIO()):
            code = replay.main()
    finally:
        sys.path[:] = before_path
    return code, json.loads((root/'output/results.json').read_text()), events, live


@pytest.mark.parametrize('omit,no_cache', [
    (('--rknn-retained-aggregator',), False), (('--rknn-paired-aggregator',), False),
    (('--half-io-library', '--half-io-library-sha256'), False),
    (('--rknn-grouped-dino',), False),
    (('--rknn-dino-manifest', '--rknn-dino-manifest-sha256'), False),
    (('--rknn-camera-head',), False), (('--rknn-camera-split4',), False),
    (('--rknn-depth-manifest', '--rknn-depth-manifest-sha256'), False), ((), True),
])
def test_incomplete_pageout_configuration_rejects_before_board_or_loading(tmp_path, omit, no_cache):
    replay = load_replay()
    with patch.object(sys, 'argv', arguments(tmp_path, omit=omit, no_cache=no_cache)), \
            patch.object(Path, 'is_file', side_effect=AssertionError('must reject before board read')), \
            patch.object(replay.os, 'sched_setaffinity', side_effect=AssertionError('must reject before setup')), \
            contextlib.redirect_stderr(io.StringIO()) as stderr:
        with pytest.raises(SystemExit) as exc:
            replay.main()
    assert exc.value.code == 2 and '--checkpoint-pageout requires' in stderr.getvalue()
    assert not (tmp_path/'output').exists()


def test_four_stage_reclamation_precedes_arrivals_and_preserves_latency_accounting(tmp_path):
    code, report, events, live = generated_run(tmp_path)
    assert code == 0 and not live and report['checkpoint_pageout']
    setup = [x for x in events if x.startswith(('install_', 'pageout_', 'persist_stage'))]
    assert setup == [item for stage in STAGES for item in
                     ('install_'+stage.removesuffix('_installed'), 'persist_stage', 'pageout_'+stage)] + ['install_cache']
    assert events.index('checkpoint_stat') < events.index('verified_load_double')
    assert events[:4] == ['affinity_4,5,6,7', 'checkpoint_stat', 'verified_load_double', 'checkpoint_stat']
    assert report['startup_seconds'] >= 4.25
    assert events.count('snapshot_startup') == events.count('snapshot_precleanup') == 1
    assert events.index('snapshot_startup') < events.index('start_replay')
    assert events.index('snapshot_precleanup') > max(i for i,x in enumerate(events) if x == 'publish')
    assert report['checkpoint_pageout_precleanup_diagnostic_seconds'] >= .25
    assert report['checkpoint_pageout_precleanup_memory']['snapshot_monotonic'] > report['updates'][-1]['published_monotonic']
    assert report['fresh_pose_count'] == 2 and len(report['updates']) == 2
    for row in report['updates']:
        assert row['input_to_pose_seconds'] == row['published_monotonic']-row['captured_monotonic']
        assert row['input_to_pose_seconds'] < .1  # Startup advice and final diagnostics occur outside arrivals.
    assert [entry['stage'] for entry in report['checkpoint_pageout_stages']] == list(STAGES)
    for entry in report['checkpoint_pageout_stages']:
        path = tmp_path/'output'/entry['receipt']
        assert entry['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry['tensor_storage_VA_version_unchanged'] and entry['checkpoint_VMA_layout_unchanged']
    assert report['checkpoint_residency_source_sha256'] == hashlib.sha256((ROOT/'vggt_slam/checkpoint_residency.py').read_bytes()).hexdigest()
    assert not report['real_time_validated'] and not report['candidate_accepted']


@pytest.mark.parametrize('stage', STAGES)
def test_stage_failure_records_receipt_and_closes_every_installed_backend(tmp_path, stage):
    code, report, events, live = generated_run(tmp_path, fail=stage)
    assert code == 1 and report['status'] == 'failed' and not live
    count = STAGES.index(stage)+1
    assert [x for x in events if x.startswith('close_')] == [
        'close_'+x.removesuffix('_installed') for x in reversed(STAGES[:count])]
    assert 'start_replay' not in events and 'predict' not in events and 'install_cache' not in events
    entry = report['checkpoint_pageout_stages'][-1]
    assert entry['stage'] == stage and entry['status'] == 'failed'
    assert entry['sha256'] == hashlib.sha256((tmp_path/'output'/entry['receipt']).read_bytes()).hexdigest()
    assert events[-1] == 'affinity_0,1,2,3,4,5,6,7'


def test_checkpoint_change_during_load_stops_before_backends_or_reclamation(tmp_path):
    code, report, events, live = generated_run(tmp_path, fail='stat_changed')
    assert code == 1 and not live and 'stat changed during verified loading' in report['error']
    assert not any(x.startswith(('install_', 'pageout_', 'snapshot_')) for x in events)


def test_disabled_default_does_not_call_helper_or_change_service_flow(tmp_path):
    code, report, events, live = generated_run(tmp_path, enabled=False)
    assert code == 0 and not live and report['checkpoint_pageout'] is False
    assert 'checkpoint_residency.py' not in report['runtime_source_sha256']
    assert 'checkpoint_pageout_stages' not in report
    assert not any(x.startswith(('checkpoint_stat', 'pageout_', 'snapshot_', 'persist_stage')) for x in events)
    assert report['fresh_pose_count'] == 2 and events.count('predict') == 2


@pytest.mark.parametrize('failure', ['install_camera', 'snapshot_precleanup'])
def test_install_or_final_diagnostic_error_still_uses_existing_cleanup(tmp_path, failure):
    code, report, events, live = generated_run(tmp_path, fail=failure)
    assert code == 1 and not live and report['status'] == 'failed'
    closed = [x for x in events if x.startswith('close_')]
    assert closed == (['close_aggregator'] if failure == 'install_camera' else
                      ['close_replay', 'close_cache', 'close_depth', 'close_DINO', 'close_camera', 'close_aggregator'])
