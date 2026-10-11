#!/usr/bin/env python3
"""Three bounded r2 regression checks, synthetic metadata only, no SDK imports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


S = load(HERE / 'v50_hybrid_step2_r2_contracts.py')
M = load(HERE / 'compile-v50-hybrid-step2-r2-board.py')
OLD = load(HERE / 'test_v50_hybrid_step2.py')
L = OLD.L


def generated_policy():
    catalog = {name: {'dtype': 'int8', 'scale': [0.025]} for name in S.INTERNAL_SELECTORS}
    catalog.update({'synthetic_initializer_' + str(i): {'dtype': 'int8'} for i in range(24)})
    catalog.update({'input': {'dtype': 'float32'}, 'output': {'dtype': 'float32'}})
    catalog.update({'synthetic_absent_sdpa_' + str(i): {'dtype': 'int8'} for i in range(4)})
    raw = ('custom_quantize_layers: {}\nquantize_parameters: ' + json.dumps(catalog) + '\n').encode()
    requested = {key: ('int8' if key in S.INTEGER_SELECTORS else 'float16') for key in S.INTERNAL_SELECTORS}
    edited, gate = S.edit_cfg(raw, requested, L)
    policy = {'schema': 'v50_original_VGGT_four_selector_diagnostic_precision_policy_r2',
        'revision': S.REVISION, 'pair_graph_sha256': S.PAIR_GRAPH_SHA, 'pair_export_sha256': S.PAIR_EXPORT_SHA,
        'checkpoint_sha256': S.CHECKPOINT_SHA, 'capture_manifest_sha256': S.CAPTURE_SHA,
        'step1_receipt_sha256': S.STEP1_SHA, 'failed_r1_receipt_sha256': S.FAILED_R1_SHA,
        'selector_scope_proof_sha256': S.SELECTOR_PROOF_SHA, 'layout_contracts_sha256': S.LAYOUT_CONTRACTS_SHA,
        'requested_custom_layers': requested, 'integer_selector_count': 4, 'floating_selector_count': 61,
        'excluded_initializer_count': 24, 'excluded_public_IO': ['input', 'output'],
        'excluded_absent_SDPA_estimate_count': 4, 'external_IO_initializer_and_hidden_attention_precision_unproven': True,
        'SDK_acceptance_of_static_selectors_unproven': True, 'candidate_accepted': False,
        'final_fc1_weight_or_arithmetic_precision_proven': False, 'precision_edit_gate': gate,
        'approved_configuration': {'file': S.APPROVED_CONFIGURATION['file'], 'bytes': len(edited), 'sha256': S.raw_sha(edited)}}
    return policy, raw, edited


def synthetic_r2_receipt():
    r = OLD.synthetic_receipt()
    r.update(revision=S.REVISION, precision_policy_sha256=S.POLICY_SHA, r1_contracts_sha256=S.R1_CONTRACTS_SHA,
        failed_r1_receipt_sha256=S.FAILED_R1_SHA, failed_r1_rejection_retained=True,
        selector_scope_proof_sha256=S.SELECTOR_PROOF_SHA,
        external_IO_initializer_and_hidden_attention_precision_unproven=True,
        approved_configuration=dict(S.APPROVED_CONFIGURATION),
        approved_configuration_sha256=S.APPROVED_CONFIGURATION['sha256'], diagnostic_requested_scope=S.DIAGNOSTIC_SCOPE)
    g = r['precision_edit_gate']
    g.update(FP16_requests=sorted(set(S.INTERNAL_SELECTORS) - set(S.INTEGER_SELECTORS)),
             edited_configuration_sha256=S.APPROVED_CONFIGURATION['sha256'])
    r['source_and_input_hashes'].update(r2_policy=S.POLICY_SHA, failed_r1=S.FAILED_R1_SHA,
        r1_contracts=S.R1_CONTRACTS_SHA, r2_config=S.APPROVED_CONFIGURATION['sha256'])
    r['source_and_input_hashes_after'] = dict(r['source_and_input_hashes'])
    return r


def gate(r):
    return S.check_step2_compile(r, 'a' * 64, 'b' * 64, S.POLICY_SHA, S.STEP1_SHA, 'c' * 64)


class R2Checks(unittest.TestCase):
    def test_generated_scope_preserves_excluded_ranges_and_rejects_public_input(self):
        policy, raw, edited = generated_policy()
        evidence = S.check_policy(policy, raw, edited, L)
        self.assertEqual(len(evidence['integer_requests']), 4)
        self.assertEqual(len(evidence['FP16_requests']), 61)
        before, _, end = S.parse_cfg(raw); after, _, next_end = S.parse_cfg(edited)
        self.assertEqual(raw[end:], edited[next_end:])
        self.assertEqual(before['quantize_parameters'], after['quantize_parameters'])
        policy['requested_custom_layers']['input'] = 'float16'
        with self.assertRaises(ValueError): S.check_policy(policy, raw, edited, L)

    def test_r2_receipt_retains_failed_r1_and_rejects_unproven_extra_selector(self):
        self.assertTrue(gate(synthetic_r2_receipt())['failed_r1_retained'])
        r = synthetic_r2_receipt(); r['failed_r1_rejection_retained'] = False
        with self.assertRaises(ValueError): gate(r)
        r = synthetic_r2_receipt(); r['precision_edit_gate']['FP16_requests'][0] = 'input'
        with self.assertRaises(ValueError): gate(r)
        r = synthetic_r2_receipt(); r['external_IO_initializer_and_hidden_attention_precision_unproven'] = False
        with self.assertRaises(ValueError): gate(r)

    def test_new_main_uses_pinned_r2_and_unmodified_r1_sources(self):
        self.assertEqual(S.sha(HERE / 'v50_hybrid_step2_r2_contracts.py'), M.CONTRACTS_SHA)
        self.assertEqual(S.sha(S.R1_SOURCE), S.R1_CONTRACTS_SHA)
        self.assertEqual(len(set(S.INTERNAL_SELECTORS)), 65)
        self.assertTrue(set(S.INTEGER_SELECTORS) <= set(S.INTERNAL_SELECTORS))
        self.assertFalse({'input', 'output'} & set(S.INTERNAL_SELECTORS))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(R2Checks))
    print(json.dumps({'schema': 'v50_original_hybrid_step2_r2_tiny_generated_checks', 'tests': result.testsRun,
        'passed': result.wasSuccessful(), 'actual_SDK_import': False, 'actual_model_execution': False,
        'actual_calibration': False, 'compile_mock_is_synthetic_not_actual_evidence': True}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
