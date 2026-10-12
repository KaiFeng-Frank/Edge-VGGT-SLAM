"""Saved receipt and fake SDK checks; no Torch, ORT, model or SDK execution."""
import copy
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def adapter():
    spec = importlib.util.spec_from_file_location('_camera_split4_generated',
        ROOT / 'vggt_slam/rknn_camera_head.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipts(module):
    rows = []
    for name, digest in [('split4-compile.json', module.SPLIT_COMPILE_SHA),
                         ('split4-ort.json', module.SPLIT_CHECK_SHA),
                         ('original-camera-capture.json', module.CAPTURE_SHA)]:
        path = (ROOT / 'deploy/rk3588/validation/vggt-v15-camera-capture112.json'
                if name == 'original-camera-capture.json' else ROOT / 'tests/fixtures' / name)
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        rows.append(json.loads(raw))
    return rows


@pytest.mark.parametrize('mutation', [None, 'config', 'operation', 'weight', 'channels', 'onnx'])
def test_saved_receipts_and_mutation_gate(monkeypatch, mutation):
    m = adapter()
    compiled, checked, capture = receipts(m)
    weights = capture['artifacts']['camera-state.npz']['tensors']
    if mutation == 'config':
        compiled['configuration']['quantize_weight'] = True
    elif mutation == 'operation':
        compiled['operations'][2]['code'] = -1
    elif mutation == 'weight':
        compiled['graph_proof']['replacements'][0]['weight_raw_sha256'] = '0' * 64
    elif mutation == 'channels':
        compiled['graph_proof']['replacements'][0]['output_channel_ranges'][3] = [0, 2048]
    elif mutation == 'onnx':
        checked['onnx_checks'][0]['comparison']['strict_pass'] = False
    pins = []
    def fake_pin(path, expected):
        pins.append((Path(path).name, expected))
        if 'camera-step-split4-r1' in str(path):
            assert expected == m.SPLIT_COMPILE_SHA
            return SimpleNamespace(read_text=lambda: json.dumps(compiled))
        if 'head-ort-r1' in str(path):
            assert expected == m.SPLIT_CHECK_SHA
            return SimpleNamespace(read_text=lambda: json.dumps(checked))
        return Path(path)
    monkeypatch.setattr(m, 'pin', fake_pin)
    if mutation is not None:
        with pytest.raises(ValueError):
            m._split4_artifact(ROOT, weights)
        assert len(pins) == 2  # Rejected before source imports or SDK initialization.
    else:
        assert m._split4_artifact(ROOT, weights).name == 'camera-step-split4.rknn'
        assert pins[-3:] == [('compile_camera_step_split_board.py', m.SPLIT_BUILDER_SHA),
                            ('compile_camera_fc1_splits_board.py', m.SPLIT_HELPER_SHA),
                            ('probe_camera_step_split_board.py', m.SPLIT_PROBE_SHA)]


@pytest.mark.parametrize('destroy_code', [0, -1])
def test_destroy_failure_keeps_unreleased_handle(destroy_code):
    m = adapter()
    handle = object.__new__(m.RKNNOriginalCameraHead)
    handle.lock, handle.closed, handle.failed = threading.RLock(), False, False
    handle.patch = handle.head = None
    handle.context = ctypes.c_uint64(7)
    handle.stats = dict(contexts_created=1, contexts_destroyed=0, cleanup_errors=[])
    calls = []
    def destroy(context):
        calls.append(context.value)
        return destroy_code
    def check(code, label):
        if code:
            raise RuntimeError(label + ' failed')
    handle.api = SimpleNamespace(rknn_destroy=destroy)
    handle.base = SimpleNamespace(check_call=check)
    if destroy_code:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            handle.close()
        assert handle.context.value == 7 and handle.metrics()['live_contexts'] == 1
        assert handle.stats['contexts_destroyed'] == 0 and handle.stats['cleanup_errors']
    else:
        handle.close()
        handle.close()
        assert handle.context.value == 0 and handle.metrics()['live_contexts'] == 0
        assert handle.stats['contexts_destroyed'] == 1 and not handle.stats['cleanup_errors']
    assert calls == [7]


def test_original_four_iteration_forward_source_unchanged():
    text = (ROOT / 'vggt_slam/rknn_camera_head.py').read_text()
    original = text[text.index('    def _make_forward(self):'):text.index('    def close(self):')]
    assert hashlib.sha256(original.encode()).hexdigest() == '22eebb32371b420e3e29b1e08dfb48f255c516e120d1408949af3e9a04e2a1af'
