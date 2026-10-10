"""Synthetic checks only: no released model/checkpoint/image execution."""
import importlib.util
from pathlib import Path
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch


HELPER_ROOT = Path(os.environ.get('V24_HELPER_ROOT', Path(__file__).resolve().parents[1]/'deploy/rk3588/validation'))
PATH = HELPER_ROOT/'export-v24-da3-board.py'
if not PATH.is_file():
    PATH = Path(__file__).with_name('export-v24-da3-board.py')
spec = importlib.util.spec_from_file_location('v24_export_synthetic', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_output_contract():
    assert probe.SHAPE == (2, 3, 252, 336)
    assert [(r['name'], r['reference_key'], r['shape']) for r in probe.OUTPUTS] == [
        ('depth', 'depth', [2, 252, 336]),
        ('depth_conf', 'conf', [2, 252, 336]),
        ('pose_enc', 'pose', [2, 9]),
    ]


def test_raw_metadata_order():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert probe.record(a[:, ::-1]) == probe.record(a[:, ::-1].copy(order='C'))
    assert probe.record(a)['bytes'] == 24


@pytest.mark.parametrize('value', [np.ones(3, np.float16), np.ones(3, np.float64),
                                  np.array([], np.float32), np.array([np.nan], np.float32),
                                  np.array([np.inf], np.float32)])
def test_reject_nonfinite_or_nonfp32(value):
    with pytest.raises(ValueError):
        probe.record(value)


def test_comparison_preserves_strict_gate():
    original = np.ones((2, 3), dtype=np.float32)
    assert probe.compare(original.copy(), original)['array_equal']
    perturbed = original.copy()
    perturbed[0, 0] += 0.0003
    evidence = probe.compare(perturbed, original)
    assert not evidence['elementwise_gate_passed']
    assert evidence['failed_elements'] == 1
    assert evidence['rtol'] == 1e-4 and evidence['atol'] == 1e-5
    assert evidence['relative_l2'] > 0


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        probe.compare(np.ones(3, np.float32), np.ones((1, 3), np.float32))


def test_zero_reference_has_no_fake_relative_error():
    evidence = probe.compare(np.ones(3, np.float32), np.zeros(3, np.float32))
    assert evidence['relative_l2'] is None
    assert evidence['failed_elements'] == 3


def test_wrapper_preserves_original_features_and_camera_token_order():
    class Backbone(torch.nn.Module):
        def forward(self, images, **kwargs):
            assert kwargs == dict(cam_token=None, export_feat_layers=[], ref_view_strategy='saddle_balanced')
            assert tuple(images.shape) == (1, 2, 3, 2, 3)
            raw_camera = images.mean(dim=(-1, -2, -3))[..., None].expand(1, 2, 9)
            return [(images, raw_camera)], []

    class Head(torch.nn.Module):
        def forward(self, feats, h, w, patch_start_idx):
            assert (h, w, patch_start_idx) == (252, 336, 0)
            pixels = feats[0][0][:, :, 0]
            return SimpleNamespace(depth=pixels + 10, depth_conf=pixels + 20)

    class Decoder(torch.nn.Module):
        def forward(self, camera):
            return camera + 30

    net = torch.nn.Module()
    net.backbone, net.head, net.cam_dec = Backbone(), Head(), Decoder()
    net.register_parameter('synthetic_parameter', torch.nn.Parameter(torch.tensor([7.0])))
    before = probe.weight_manifest(net)
    inputs = torch.arange(36, dtype=torch.float32).reshape(2, 3, 2, 3)
    with torch.no_grad():
        depth, conf, pose = probe.make_wrapper(net)(inputs)
    assert torch.equal(depth, inputs[:, 0] + 10)
    assert torch.equal(conf, inputs[:, 0] + 20)
    assert torch.equal(pose, inputs.mean(dim=(-1, -2, -3))[:, None].expand(2, 9) + 30)
    assert probe.weight_manifest(net) == before


def test_source_and_config_pins_are_complete(tmp_path):
    source = tmp_path/'official'
    source.mkdir()
    (source/'one.py').write_text('synthetic = True\n')
    config = tmp_path/'model-metadata'/'config.json'
    config.parent.mkdir()
    config.write_text('{"synthetic": true}\n')
    digest = probe.sha(config)
    common = SimpleNamespace(MODEL_CONFIG_SHA256=digest, verify_source_manifest=lambda *_:
                             {'source_hashes': {'one.py': probe.sha(source/'one.py')}})
    pins = {}
    probe.bind_source_pins(SimpleNamespace(source_dir=source, manifest=None, manifest_sha256=None), common, pins)
    assert pins == {str(source/'one.py'): probe.sha(source/'one.py'), str(config): digest}


def test_host_guard_precedes_output_and_model_side_effects(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.platform, 'machine', lambda: 'x86_64')
    output = tmp_path/'not-created'
    with pytest.raises(RuntimeError, match='Board-only'):
        probe.main(['--mode', 'export', '--output', str(output)])
    assert not output.exists()


def test_cleanup_source_hash_error_keeps_receipt_and_restores_affinity(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, 'BASE', tmp_path)
    monkeypatch.setattr(probe, 'require_board', lambda: 'synthetic-RK')
    affinity = {0, 1}
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda _: set(affinity))
    def set_affinity(_pid, value):
        affinity.clear()
        affinity.update(value)
    monkeypatch.setattr(probe.os, 'sched_setaffinity', set_affinity)
    monkeypatch.setattr(torch, 'set_num_threads', lambda _: None)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda _: None)
    monkeypatch.setattr(torch, 'get_num_interop_threads', lambda: 1)
    def fake_export(_args, report, pins):
        pins[str(tmp_path/'missing-input.npy')] = 'f'*64
        report['status'] = 'exported_not_runtime_validated'
    monkeypatch.setattr(probe, 'export_model', fake_export)
    output = tmp_path/'experiments'/'synthetic-error'
    assert probe.main(['--mode', 'export', '--output', str(output), '--cpus', '0', '--threads', '1']) == 1
    import json
    evidence = json.loads((output/'export.json').read_text())
    assert evidence['status'] == 'failed' and evidence['sources_and_inputs_unchanged'] is False
    assert evidence['affinity_restored'] and affinity == {0, 1}
    assert any('SHA verification failed' in s for s in evidence['cleanup_errors'])
