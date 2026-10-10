"""Synthetic lifecycle/protocol tests only; no original weights, images or NPU."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'deploy/rk3588/validation/run-v20-aggregator-attention-board.py'
spec = importlib.util.spec_from_file_location('v20_attention_synthetic', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
protocol_spec = importlib.util.spec_from_file_location('v20_r3_synthetic_sdk',
                                                     Path(__file__).with_name('test_v19_attention_runtime_r3.py'))
protocol = importlib.util.module_from_spec(protocol_spec)
protocol_spec.loader.exec_module(protocol)


def toy_inputs(shape):
    # Same shape but noncontiguous, like original post-QKV head views.
    return tuple((torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)+i*1000)
                 .transpose(1, 2).contiguous().transpose(1, 2) for i in range(3))


def toy_functional():
    calls = []
    def synthetic(query, key, value, **kwargs):
        calls.append(((query, key, value), kwargs))
        return query+2*key+3*value  # Protocol arithmetic, explicitly not a real SDPA reference.
    return SimpleNamespace(scaled_dot_product_attention=synthetic), calls


def test_four_interleaved_forwards_reuse_contexts_and_keep_live_qkv_and_output_order():
    base = m.load_attention()
    runtime = base.load_helper('run-v12-rknn-block-board.py', base.RUNTIME_SHA)
    shape = (2, 5, 7, 64)
    sdk, sessions = protocol.sessions_for(base, runtime, [(1, 4, 7, 64), (1, 3, 7, 64), (1, 3, 7, 64)])
    functional, cpu_calls = toy_functional()
    original = functional.scaled_dot_product_attention
    values = toy_inputs(shape)
    assert all(not tensor.is_contiguous() for tensor in values)
    pool = ThreadPoolExecutor(max_workers=3)
    report = {'cleanup_errors': []}
    groups, rows = [], []
    def native(flat, original_shape, group):
        groups.append(group)
        return base.execute_round(sdk, runtime, sessions, flat, original_shape, 30000, group, pool)
    bridge = m.AttentionBridge(functional, torch, shape, native)
    try:
        with bridge:
            for index in range(4):
                for mode in m.order_for_round(index):
                    row = {'mode': mode, 'round_index': index}
                    rows.append(row)
                    bridge.begin(mode, row)
                    actual = functional.scaled_dot_product_attention(*values, dropout_p=0.)
                    torch.testing.assert_close(actual, values[0]+2*values[1]+3*values[2], rtol=0, atol=0)
                    assert all(a is b for a, b in zip(bridge.operands, values))
                    assert row['sdpa_calls'] == 1 and row['attention_seconds'] > 0
                    if mode == 'hybrid':
                        assert actual.is_contiguous() and actual.dtype == torch.float32
                        assert row['tensor_to_numpy_seconds'] >= 0
                        assert row['native_attention']['run_api_overlap']['max_simultaneous'] == 3
    finally:
        m.cleanup_native(base, runtime, sdk, pool, sessions, report)
    assert bridge.restored and functional.scaled_dot_product_attention is original
    assert len(cpu_calls) == 4 and len(groups) == 4
    assert len(sdk.states) == 3 and len(sdk.requests) == 12 and len(sdk.releases) == 12
    assert report['workers_joined'] and not report['cleanup_errors']
    assert sdk.destroyed == [3, 2, 1] and sdk.active == 0
    assert [row['mode'] for row in rows] == ['cpu','hybrid','hybrid','cpu','cpu','hybrid','hybrid','cpu']


@pytest.mark.parametrize('kwargs', [{'dropout_p': .5}, {'dropout_p': 0., 'attn_mask': None}, {},
                                  {'dropout_p': 0., 'is_causal': True}])
def test_changed_original_sdpa_signature_is_rejected_and_callable_restored(kwargs):
    functional, calls = toy_functional()
    original = functional.scaled_dot_product_attention
    bridge = m.AttentionBridge(functional, torch, (1, 3, 4, 64), lambda *args: pytest.fail('No NPU fallback'))
    with pytest.raises(ValueError, match='unmasked eval'):
        with bridge:
            bridge.begin('cpu', {})
            functional.scaled_dot_product_attention(*toy_inputs((1, 3, 4, 64)), **kwargs)
    assert not calls and bridge.restored and functional.scaled_dot_product_attention is original


def test_native_failure_joins_all_workers_restores_sdpa_and_destroys_contexts():
    base = m.load_attention()
    runtime = base.load_helper('run-v12-rknn-block-board.py', base.RUNTIME_SHA)
    shape = (1, 6, 4, 64)
    sdk, sessions = protocol.sessions_for(base, runtime, [(1, 2, 4, 64)]*3, failure='run')
    functional, calls = toy_functional()
    original = functional.scaled_dot_product_attention
    pool = ThreadPoolExecutor(max_workers=3)
    report = {'cleanup_errors': []}
    def native(flat, original_shape, group):
        return base.execute_round(sdk, runtime, sessions, flat, original_shape, 30000, group, pool)
    bridge = m.AttentionBridge(functional, torch, shape, native)
    row = {}
    try:
        with pytest.raises(RuntimeError, match='rknn_run'):
            with bridge:
                bridge.begin('hybrid', row)
                functional.scaled_dot_product_attention(*toy_inputs(shape), dropout_p=0.)
    finally:
        m.cleanup_native(base, runtime, sdk, pool, sessions, report)
    assert not calls and bridge.restored and functional.scaled_dot_product_attention is original
    assert report['workers_joined'] and not report['cleanup_errors']
    assert sdk.active == 0 and sdk.destroyed == [3, 2, 1] and sorted(sdk.releases) == [2, 3]
    assert row['native_attention']['worker_errors']


def test_native_error_logs_are_captured_even_if_sdk_returned_success(tmp_path):
    path = tmp_path/'sdk.log'
    before = {fd: os.fstat(fd) for fd in (1, 2)}
    log = m.NativeLog(path)
    with log:
        os.write(1, b'I RKNN runtime version 2.3.2\n')
        os.write(2, b'E RKNN Meet unsupported src layout for normalize: NCHW, only support NHWC src layout!\n')
        os.write(2, b'E RKNN rknn_inputs_set, normalize error(-1) index=0\n')
        inspected = log.inspect()
    assert log.restored and inspected['error_line_count'] == 2
    assert inspected['sha256'] == m.sha(path)
    for fd in (1, 2):
        after = os.fstat(fd)
        assert (after.st_dev, after.st_ino) == (before[fd].st_dev, before[fd].st_ino)


def test_partial_fd_redirection_failure_restores_both_streams(tmp_path, monkeypatch):
    duplicate = os.dup2
    before = {fd: os.fstat(fd) for fd in (1, 2)}
    count = 0
    def fail_second(source, destination):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('synthetic second-stream redirection failure')
        return duplicate(source, destination)
    monkeypatch.setattr(m.os, 'dup2', fail_second)
    log = m.NativeLog(tmp_path/'partial.log')
    with pytest.raises(OSError, match='second-stream'):
        with log:
            pytest.fail('No body after incomplete native-log setup')
    assert log.restored and not log.saved and log.fd is None
    for fd in (1, 2):
        after = os.fstat(fd)
        assert (after.st_dev, after.st_ino) == (before[fd].st_dev, before[fd].st_ino)


@pytest.mark.parametrize('key', ['block', 'profile_sha256', 'selected_pair', 'checkpoint',
                              'original_block_array_metadata', 'original_block_parameter_metadata'])
def test_different_profile_input_weight_or_source_pair_cannot_use_attention_capture(tmp_path, key):
    path = tmp_path/'profile.json'
    path.write_text('{}')
    profile = dict(_path=str(path), selected_pair={'frame_ids':[5,78]}, checkpoint={'sha256':'a'*64})
    row = dict(arrays={'input':{'sha256':'b'*64}}, weights={'norm1.weight':{'sha256':'c'*64}})
    capture = dict(block='global0', profile_sha256=m.sha(path), selected_pair=profile['selected_pair'],
                   checkpoint=profile['checkpoint'], original_block_array_metadata=row['arrays'],
                   original_block_parameter_metadata=row['weights'])
    m.validate_capture_pair(profile, row, 'global0', capture)
    changed = copy.deepcopy(capture)
    changed[key] = 'different'
    with pytest.raises(ValueError, match='identical original input/weights'):
        m.validate_capture_pair(profile, row, 'global0', changed)


def test_frozen_r3_source_guard_and_host_guard_before_output(tmp_path, monkeypatch):
    (tmp_path/m.R3_NAME).write_bytes(b'not frozen')
    monkeypatch.setattr(m, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='Frozen R3'):
        m.load_attention()
    monkeypatch.undo()
    args = ['--block','global0','--profile-dir',str(tmp_path/'profile'),'--profile-sha256','a'*64,
            '--capture-dir',str(tmp_path/'capture'),'--capture-sha256','b'*64,
            '--export-dir',str(tmp_path/'export'),'--export-sha256','c'*64,'--output',str(tmp_path/'new'),
            '--models','synthetic.rknn','--model-sha256','d'*64,
            '--compile-receipts','synthetic.json','--compile-sha256','e'*64]
    with pytest.raises(RuntimeError, match='Board-only'):
        m.main(args)
    assert not (tmp_path/'new').exists()
