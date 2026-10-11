#!/usr/bin/env python3
"""Tiny generated/fake checks only: no SDK, tensor model or calibration imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


H = load('v50_hybrid_api.py')
M = load('prepare-v50-hybrid-step1-board.py')


class FakeContracts:
    CONFIGURATION = {'target_platform': 'rk3588', 'float_dtype': 'float16',
                     'optimization_level': 3, 'quantized_dtype': 'w8a8',
                     'quantized_algorithm': 'normal', 'quantized_method': 'channel',
                     'do_quantization': True}
    @staticmethod
    def require_board():
        return 'fake tiny test, not a board identity assertion'
    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeToolkit:
    def __init__(self, failed=None, code=0):
        self.calls, self.failed, self.code = [], failed, code
    def __getattr__(self, name):
        if name not in {'config', 'load_onnx', 'hybrid_quantization_step1',
                        'hybrid_quantization_step2', 'export_rknn', 'release'}:
            raise AssertionError('Forbidden or unsupported API: ' + name)
        def operation(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.code if name == self.failed else 0
        return operation


def report():
    return {'operations': [], 'cleanup_errors': [], 'released': None,
            'precision_policy_approved': False, 'calibration_api_invoked': False,
            'calibration_completed_on_board': False}


class HybridChecks(unittest.TestCase):
    def test_exact_step1_calls_and_configuration(self):
        sdk, row = FakeToolkit(), report()
        H.step1(sdk, HERE / 'generated-graph.onnx', HERE / 'generated-dataset.txt', row, FakeContracts)
        H.release(sdk, row)
        self.assertEqual([call[0] for call in sdk.calls],
                         ['config', 'load_onnx', 'hybrid_quantization_step1', 'release'])
        expected = dict(FakeContracts.CONFIGURATION)
        expected.pop('do_quantization')
        expected.update(mean_values=None, std_values=None, quant_img_RGB2BGR=False)
        self.assertEqual(sdk.calls[0][2], expected)
        self.assertEqual(sdk.calls[1][2], {'model': str(HERE / 'generated-graph.onnx')})
        self.assertEqual(sdk.calls[2][2], {'dataset': str(HERE / 'generated-dataset.txt'),
                         'rknn_batch_size': None, 'proposal': False,
                         'proposal_dataset_size': 1, 'custom_hybrid': None})
        self.assertTrue(row['calibration_completed_on_board'])
        self.assertTrue(row['released'])
        self.assertTrue(all(item['status'] == 'completed' for item in row['operations']))

    def test_board_guard_precedes_sdk_calls(self):
        class RejectedBoard(FakeContracts):
            @staticmethod
            def require_board():
                raise ValueError('generated host rejection')
        for operation in (
                lambda s, r: H.step1(s, 'graph', 'dataset', r, RejectedBoard),
                lambda s, r: H.step2(s, 'model', 'data', 'cfg', 'output', r, RejectedBoard)):
            sdk, row = FakeToolkit(), report()
            with self.assertRaises(ValueError):
                operation(sdk, row)
            self.assertEqual(sdk.calls, [])

    def test_api_failures_stop_and_record_release(self):
        for name in ('config', 'load_onnx', 'hybrid_quantization_step1'):
            for code in (-1, True, False):
                sdk, row = FakeToolkit(name, code), report()
                with self.assertRaises(RuntimeError):
                    H.step1(sdk, 'graph', 'dataset', row, FakeContracts)
                H.release(sdk, row)
                self.assertEqual(row['operations'][-1]['status'], 'failed')
                self.assertFalse(row['calibration_completed_on_board'])
                self.assertEqual(sdk.calls[-1][0], 'release')
                self.assertTrue(row['released'])
        sdk, row = FakeToolkit('release', -1), report()
        H.release(sdk, row)
        self.assertFalse(row['released'])
        self.assertEqual(len(row['cleanup_errors']), 1)

    def test_supported_none_return(self):
        sdk, row = FakeToolkit('hybrid_quantization_step1', None), report()
        H.step1(sdk, 'graph', 'dataset', row, FakeContracts)
        self.assertTrue(row['calibration_completed_on_board'])
        self.assertIsNone(row['operations'][-1]['code'])

    def test_step2_requires_approval_and_exact_cfg(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / 'generated.quantization.cfg'
            cfg.write_text('tiny generated config; not SDK data\n')
            sdk, row = FakeToolkit(), report()
            with self.assertRaises(ValueError):
                H.step2(sdk, 'model', 'data', cfg, 'output', row, FakeContracts)
            self.assertEqual(sdk.calls, [])
            row.update(precision_policy_approved=True, approved_configuration_sha256='0' * 64)
            with self.assertRaises(ValueError):
                H.step2(sdk, 'model', 'data', cfg, 'output', row, FakeContracts)
            self.assertEqual(sdk.calls, [])
            row['approved_configuration_sha256'] = FakeContracts.sha(cfg)
            H.step2(sdk, 'model', 'data', cfg, Path(directory) / 'output.rknn', row, FakeContracts)
            self.assertEqual([item[0] for item in sdk.calls], ['hybrid_quantization_step2', 'export_rknn'])
            self.assertEqual(sdk.calls[0][2], {'model_input': str(Path('model').resolve()),
                             'data_input': str(Path('data').resolve()), 'model_quantization_cfg': str(cfg)})

    def test_generated_intermediates_preserved_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            work, out = Path(directory) / 'work', Path(directory) / 'out'
            work.mkdir(); out.mkdir()
            for suffix in ('.model', '.data', '.quantization.cfg'):
                (work / ('generated' + suffix)).write_bytes(('tiny ' + suffix).encode())
            before = {p.name: p.read_bytes() for p in work.iterdir()}
            rows = M.collect_intermediates(work, out, FakeContracts)
            self.assertEqual(set(rows), {'model', 'data', 'configuration'})
            for item in rows.values():
                self.assertEqual((out / item['file']).read_bytes(), before[item['file']])
                self.assertEqual(FakeContracts.sha(out / item['file']), item['sha256'])
            self.assertEqual({p.name: p.read_bytes() for p in work.iterdir()}, before)
            with self.assertRaises(ValueError):
                M.collect_intermediates(work, out, FakeContracts)

    def test_intermediate_ambiguity_empty_name_and_escape_rejected(self):
        for failure in ('duplicate', 'empty', 'stem', 'symlink'):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                work, out = base / 'work', base / 'out'
                work.mkdir(); out.mkdir()
                for suffix in ('.model', '.data', '.quantization.cfg'):
                    (work / ('generated' + suffix)).write_bytes(b'tiny generated data')
                if failure == 'duplicate':
                    (work / 'duplicate.model').write_bytes(b'tiny')
                elif failure == 'empty':
                    (work / 'generated.data').write_bytes(b'')
                elif failure == 'stem':
                    (work / 'generated.data').rename(work / 'changed.data')
                else:
                    outside = base / 'outside.data'
                    outside.write_bytes(b'tiny outside')
                    (work / 'generated.data').unlink()
                    (work / 'generated.data').symlink_to(outside)
                with self.assertRaises(ValueError):
                    M.collect_intermediates(work, out, FakeContracts)

    def test_helper_sha_is_frozen_and_sdk_is_lazy(self):
        self.assertEqual(FakeContracts.sha(HERE / 'v50_hybrid_api.py'), M.HYBRID_API_SHA)
        import ast
        tree = ast.parse((HERE / 'prepare-v50-hybrid-step1-board.py').read_text())
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module == 'rknn.api'
                             for node in tree.body))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        self.assertTrue(any(isinstance(node, ast.ImportFrom) and node.module == 'rknn.api'
                            for node in ast.walk(main)))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HybridChecks))
    print(json.dumps({'schema': 'v50_original_hybrid_tiny_fake_checks', 'tests': result.testsRun,
                      'passed': result.wasSuccessful(), 'actual_SDK_import': False,
                      'actual_model_execution': False, 'actual_calibration': False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
