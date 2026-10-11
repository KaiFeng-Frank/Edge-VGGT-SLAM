"""Generated saved records only: motion admission isolation and truthful gates."""
import copy
import importlib.util
from pathlib import Path
import unittest


def load():
    path = Path(__file__).with_name('evaluate-v45-half-motion.py')
    spec = importlib.util.spec_from_file_location('_v45_tiny_evaluation', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load()

    def records(self):
        baseline = dict(frames=[dict(index=0, timestamp=100., arrival_monotonic=200.,
            result=dict(observation_id=0, admitted=True, fresh_pose_valid=False,
                        pose_w2c=None, motion_probe=None, reason='initialization_wait')),
            dict(index=1, timestamp=101., arrival_monotonic=201.,
            result=dict(observation_id=1, admitted=False, fresh_pose_valid=False,
                        pose_w2c=None, motion_probe=1.0345186591148376,
                        reason='initial_motion_below_2'))])
        candidate = copy.deepcopy(baseline)
        candidate['frames'][0]['arrival_monotonic'] += 10
        candidate['frames'][1]['result'].update(admitted=True, reason='initialization_wait')
        return candidate, baseline

    def ratios(self, value=.99, direction=1.7):
        return {key: dict(statistics=dict(rmse=dict(candidate_over_baseline=ratio)))
                for key, ratio in [(key, value) for key in self.evaluator.PRIMARY_METRICS] +
                                  [(self.evaluator.DIRECTION_METRIC, direction)]}

    def test_same_probe_different_admission_is_isolated(self):
        candidate, baseline = self.records()
        result = self.evaluator.check_first_divergence(candidate, baseline)
        self.assertTrue(result['trial_isolation_verified'])
        self.assertEqual(result['first_admission_decision_difference'], 1)
        self.assertFalse(result['old_gold_or_capture_comparison_after_admission_divergence'])

    def test_probe_number_change_breaks_isolation(self):
        candidate, baseline = self.records()
        candidate['frames'][1]['result']['motion_probe'] = 1.05
        self.assertFalse(self.evaluator.check_first_divergence(candidate, baseline)['trial_isolation_verified'])

    def test_source_timestamp_is_not_ignored_as_a_clock(self):
        candidate, baseline = self.records()
        candidate['frames'][0]['timestamp'] += .001
        self.assertFalse(self.evaluator.check_first_divergence(candidate, baseline)['trial_isolation_verified'])

    def test_primary_limits_fixed_direction_extra_and_missing_is_not_pass(self):
        gates = self.evaluator.quality_gates(self.ratios(value=1.15))
        self.assertFalse(gates['10_percent']['primary_ATE_rotation_RPE_translation_RPE_pass'])
        self.assertTrue(gates['20_percent']['primary_ATE_rotation_RPE_translation_RPE_pass'])
        self.assertIn('additional_direction_diagnostic_not_acceptance_standard', gates)
        missing = self.evaluator.quality_gates(self.ratios(value=None))
        self.assertFalse(missing['20_percent']['primary_ATE_rotation_RPE_translation_RPE_pass'])

    def test_failed_or_previous_schema_rejected(self):
        schema, status, driver = self.evaluator.CONTRACTS['candidate']
        for override in (dict(status='failed'),
                         dict(schema='v44_half_keyframe7p5_original_Torch_CPU_research_replay')):
            run = dict(schema=schema, status=status, script_sha256=driver)
            run.update(override)
            with self.assertRaises(ValueError):
                self.evaluator.check_completed_run(run, 'candidate')


if __name__ == '__main__':
    unittest.main()
