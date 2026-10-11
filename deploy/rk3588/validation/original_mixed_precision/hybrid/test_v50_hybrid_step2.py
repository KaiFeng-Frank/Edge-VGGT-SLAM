#!/usr/bin/env python3
"""Five tiny source/generated checks, no real SDK/model/calibration execution."""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


S = load(HERE / 'v50_hybrid_step2_contracts.py')
H = load(HERE / 'v50_hybrid_api.py')
M = load(HERE / 'compile-v50-hybrid-step2-board.py')
layout_candidates = [HERE.parent / 'layout/v50_original_mlp_scope_contracts.py',
                     HERE / 'v50_original_mlp_scope_contracts.py']
L = load(next(path for path in layout_candidates if path.is_file()))


def synthetic_receipt():
    """Only in-memory invented compile metadata; never saved as actual evidence."""
    r = {'schema': S.SCHEMA, 'status': 'compiled_not_runtime_validated',
         'script_sha256': 'a' * 64, 'contracts_sha256': 'b' * 64, 'precision_policy_sha256': S.POLICY_SHA,
         'step1_receipt_sha256': S.STEP1_SHA, 'step1_script_sha256': S.STEP1_SOURCE_SHA,
         'layout_contracts_sha256': S.LAYOUT_CONTRACTS_SHA, 'pair_contracts_sha256': S.PAIR_CONTRACTS_SHA,
         'pair_export_sha256': S.PAIR_EXPORT_SHA, 'pair_graph_sha256': S.PAIR_GRAPH_SHA,
         'quant_contracts_sha256': S.QUANT_CONTRACTS_SHA, 'capture_contracts_sha256': S.CAPTURE_CONTRACTS_SHA,
         'capture_manifest_sha256': S.CAPTURE_SHA, 'capture_script_sha256': S.CAPTURE_SOURCE_SHA,
         'hybrid_api_sha256': S.HYBRID_API_SHA, 'SDK_API_source_sha256': S.SDK_API_SHA,
         'configuration': dict(S.CONFIGURATION), 'versions': dict(S.VERSIONS),
         'input_transform': dict(S.INPUT_TRANSFORM),
         'board': {'machine': 'aarch64', 'device_tree_compatible': 'synthetic rk3588 metadata only'},
         'candidate_accepted': False, 'model_inference_on_host': False,
         'historical_FP32_rejections_retained': True, 'whole_model_quality_validated': False,
         'real_time_validated': False, 'fixed_FP32_gate': {'rtol': 1e-4, 'atol': 1e-5},
         'operations': [{'name': name, 'status': 'completed', 'code': 0, 'seconds': 0.01}
                        for name in ('hybrid_quantization_step2', 'export_rknn')],
         'released': True, 'release_code': None, 'cleanup_errors': [], 'source_and_inputs_unchanged': True,
         'cwd_restored': True, 'environment_restored': True, 'precision_policy_approved': True,
         'runtime_inference_API_called': False, 'calibration_executed_on_board_only': True,
         'new_calibration_dataset_supplied_to_step2': False,
         'original_Torch_FP32_reference_is_sole_numerical_reference': True,
         'raw_generated_artifacts': copy.deepcopy(S.RAW_ARTIFACTS),
         'original_checkpoint': {'sha256': S.CHECKPOINT_SHA, 'bytes': 5026874952},
         'approved_configuration': dict(S.APPROVED_CONFIGURATION),
         'approved_configuration_sha256': S.APPROVED_CONFIGURATION['sha256'],
         'diagnostic_requested_scope': 'four actual fc1/GELU selectors INT8; entire fc2/LayerScale/residual group FP16',
         'final_fc1_weight_or_arithmetic_precision_proven': False, 'synthetic_mock': True}
    r['precision_edit_gate'] = {'only_custom_quantize_layers_changed': True,
        'quantize_parameters_and_other_fields_unchanged': True, 'custom_block_only_bytes_changed': True,
        'exact_original_MLP_only_precision_proven': False, 'candidate_accepted': False,
        'runtime_accuracy_or_speed_proven': False, 'scope_label': 'MLP-focused precision allocation diagnostic',
        'untouched_prefix_bytes': 0, 'untouched_prefix_sha256': S.raw_sha(b''),
        'untouched_suffix_bytes': 33709, 'untouched_suffix_sha256': S.UNTOUCHED_SUFFIX_SHA,
        'quantize_parameters_catalog_sha256': S.CATALOG_SHA, 'integer_requests': list(S.INTEGER_SELECTORS),
        'FP16_requests': ['synthetic_protected_' + str(i) for i in range(91)],
        'edited_configuration_sha256': S.APPROVED_CONFIGURATION['sha256'],
        'raw_configuration_sha256': S.RAW_ARTIFACTS['configuration']['sha256']}
    pins = [r['script_sha256'], r['contracts_sha256'], S.POLICY_SHA, S.STEP1_SHA,
            S.PAIR_EXPORT_SHA, S.PAIR_GRAPH_SHA, S.PAIR_CONTRACTS_SHA, S.CAPTURE_CONTRACTS_SHA,
            S.CAPTURE_SHA, S.QUANT_CONTRACTS_SHA, S.LAYOUT_CONTRACTS_SHA, S.SDK_API_SHA, S.HYBRID_API_SHA,
            *[item['sha256'] for item in S.RAW_ARTIFACTS.values()], S.APPROVED_CONFIGURATION['sha256']]
    r['source_and_input_hashes'] = {'synthetic_pin_' + str(i): value for i, value in enumerate(pins)}
    r['source_and_input_hashes_after'] = dict(r['source_and_input_hashes'])
    r['output'] = {'file': 'pair.rknn', 'bytes': 12, 'sha256': 'c' * 64}
    r['output_after'], r['output_unchanged'] = dict(r['output']), True
    return r


def gate(row):
    return S.check_step2_compile(row, 'a' * 64, 'b' * 64, S.POLICY_SHA, S.STEP1_SHA, 'c' * 64)


class Step2Checks(unittest.TestCase):
    def test_custom_block_surgery_preserves_generated_range_bytes(self):
        raw = b'# generated tiny fixture\ncustom_quantize_layers: {}\nquantize_parameters:\n    tiny: {dtype: int8, min: [-1.2500], max: [2.0], scale: [0.02]}\n'
        edited, evidence = S.edit_cfg(raw, {'tiny': 'float16'}, L)
        old, start, end = S.parse_cfg(raw)
        new, next_start, next_end = S.parse_cfg(edited)
        self.assertEqual(raw[:start], edited[:next_start])
        self.assertEqual(raw[end:], edited[next_end:])
        self.assertEqual(old['quantize_parameters'], new['quantize_parameters'])
        self.assertTrue(evidence['custom_block_only_bytes_changed'])
        with self.assertRaises(ValueError):
            L.check_cfg_edit(old, {**new, 'quantize_parameters': {'tiny': {'scale': [0.03]}}}, {'tiny': 'float16'})
        with self.assertRaises(ValueError):
            S.edit_cfg(raw, {'unknown': 'int8'}, L)

    def test_unsafe_ambiguous_yaml_and_json_rejected(self):
        malformed = [b'custom_quantize_layers: {}\ncustom_quantize_layers: {}\n',
            b'custom_quantize_layers: &a {}\nquantize_parameters: *a\n',
            b'custom_quantize_layers: {}\nquantize_parameters: {<<: {x: 1}}\n',
            b'custom_quantize_layers: {}\nquantize_parameters: {x: .nan}\n',
            b'custom_quantize_layers: {}\nquantize_parameters: !!python/object/apply:os.system [false]\n']
        for raw in malformed:
            with self.assertRaises(Exception):
                S.parse_cfg(raw)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'generated.json'
            p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                S.unique_json(p)

    def test_compile_gate_rejects_scope_hash_return_cleanup_changes(self):
        self.assertFalse(gate(synthetic_receipt())['candidate_accepted'])
        for mutate in (lambda r: r.update(candidate_accepted=True),
                       lambda r: r.update(environment_restored=False),
                       lambda r: r.update(pair_graph_sha256='0' * 64),
                       lambda r: r.update(output_unchanged=False),
                       lambda r: r['operations'][0].update(code=True),
                       lambda r: r['precision_edit_gate'].update(untouched_suffix_bytes=1),
                       lambda r: r['precision_edit_gate']['integer_requests'].append('fused_fc2'),
                       lambda r: r['source_and_input_hashes_after'].clear()):
            row = synthetic_receipt(); mutate(row)
            with self.assertRaises(ValueError):
                gate(row)

    def test_sdk_error_stops_export_and_releases_fake_only(self):
        class FakeC:
            @staticmethod
            def require_board():
                return 'fake source test only'
            sha = staticmethod(S.sha)
        class FakeSDK:
            def __init__(self): self.calls = []
            def hybrid_quantization_step2(self, **kw): self.calls.append('step2'); return -1
            def export_rknn(self, *args): self.calls.append('export'); return 0
            def release(self): self.calls.append('release'); return None
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / 'generated.cfg'; cfg.write_text('tiny fake config\n')
            row = {'operations': [], 'cleanup_errors': [], 'precision_policy_approved': True,
                   'approved_configuration_sha256': S.sha(cfg)}
            sdk = FakeSDK()
            with self.assertRaises(RuntimeError):
                H.step2(sdk, 'generated.model', 'generated.data', cfg, 'generated.rknn', row, FakeC)
            H.release(sdk, row)
            self.assertEqual(sdk.calls, ['step2', 'release'])
            self.assertTrue(row['released'])

    def test_main_host_gate_before_sdk_and_frozen_source_binding(self):
        self.assertEqual(S.sha(HERE / 'v50_hybrid_step2_contracts.py'), M.CONTRACTS_SHA)
        tree = ast.parse((HERE / 'compile-v50-hybrid-step2-board.py').read_text())
        self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.module == 'rknn.api' for n in tree.body))
        with tempfile.TemporaryDirectory() as directory, patch.object(M.platform, 'machine', return_value='generated-host'):
            output, work = Path(directory) / 'out', Path(directory) / 'work'
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                M.main(['--output-dir', str(output), '--work-dir', str(work)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())
            self.assertFalse(work.exists())


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Step2Checks))
    print(json.dumps({'schema': 'v50_original_hybrid_step2_tiny_fake_checks', 'tests': result.testsRun,
        'passed': result.wasSuccessful(), 'actual_SDK_import': False, 'actual_model_execution': False,
        'actual_calibration': False, 'compile_mock_is_synthetic_not_actual_evidence': True}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
