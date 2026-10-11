#!/usr/bin/env python3
"""Four generated source/config rejection checks; never execute a real model."""
from __future__ import annotations

import copy
import unittest

from v50_original_mlp_scope_contracts import (
    check_cfg_edit, check_diagnostic_declarations,
)


class OriginalScopeChecks(unittest.TestCase):
    def diagnostic(self):
        return {
            'candidate_accepted': False, 'real_time_validated': False,
            'whole_model_quality_validated': False, 'rtol': 1e-4, 'atol': 1e-5,
            'original_Torch_FP32_reference_is_sole_numerical_reference': True,
            'historical_FP32_rejections_retained': True,
        }

    def configuration(self):
        # Generated tiny metadata, unrelated to any captured tensor or weights.
        return {'custom_quantize_layers': {}, 'quantize_parameters': {
            'generated_mlp_output': {'dtype': 'int8', 'min': [-1.0], 'max': [1.0]},
            'generated_boundary': {'dtype': 'float16', 'min': [-2.0], 'max': [2.0]},
        }}

    def test_preserved_diagnostic_is_not_adoption(self):
        result = check_diagnostic_declarations(self.diagnostic())
        self.assertFalse(result['candidate_accepted'])

    def test_hidden_acceptance_is_rejected(self):
        row = self.diagnostic()
        row['candidate_accepted'] = True
        with self.assertRaises(ValueError):
            check_diagnostic_declarations(row)

    def test_custom_requests_do_not_prove_fusion_precision(self):
        before = self.configuration()
        request = {'generated_mlp_output': 'int8', 'generated_boundary': 'float16'}
        after = copy.deepcopy(before)
        after['custom_quantize_layers'] = request
        result = check_cfg_edit(before, after, request)
        self.assertTrue(result['quantize_parameters_and_other_fields_unchanged'])
        self.assertFalse(result['exact_original_MLP_only_precision_proven'])
        self.assertFalse(result['runtime_accuracy_or_speed_proven'])

    def test_generated_ranges_mutation_is_rejected(self):
        before = self.configuration()
        request = {'generated_mlp_output': 'int8', 'generated_boundary': 'float16'}
        after = copy.deepcopy(before)
        after['custom_quantize_layers'] = request
        after['quantize_parameters']['generated_mlp_output']['min'][0] = -3.0
        with self.assertRaises(ValueError):
            check_cfg_edit(before, after, request)


if __name__ == '__main__':
    unittest.main(verbosity=2)
