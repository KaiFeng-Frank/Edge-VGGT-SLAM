"""Generated JSON/numbers only: truthful trial isolation and primary gates."""
import copy
import importlib.util
from pathlib import Path
import unittest


def load():
    path = Path(__file__).with_name('evaluate-v44-half-keyframe.py')
    spec = importlib.util.spec_from_file_location('_v44_tiny_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SavedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load()

    def runs(self):
        frames = [dict(index=i, timestamp=100 + i / 30,
                       arrival_monotonic=200 + i, processing_seconds=.03,
                       result=dict(observation_id=i, pose_w2c=[0., 0., 0., 0., 0., 0., 1.],
                                   captured_monotonic=200 + i, process_seconds=.025,
                                   keyframe=None, diagnostic=dict(pose_update_norm=.01)))
                  for i in range(55)]
        baseline = dict(frames=frames)
        baseline['frames'][54]['result']['keyframe'] = dict(
            mean_bidirectional_flow=8.138959407806396, removed_observation_id=53)
        candidate = copy.deepcopy(baseline)
        for row in candidate['frames']:
            row['arrival_monotonic'] += 30
            row['result']['captured_monotonic'] += 30
            row['processing_seconds'] = .05
        candidate['frames'][54]['result']['keyframe']['removed_observation_id'] = None
        return candidate, baseline

    def ratios(self, primary=.99, direction=1.7):
        return {key: dict(statistics=dict(rmse=dict(candidate_over_baseline=value)))
                for key, value in [(key, primary) for key in self.evaluator.PRIMARY_METRICS] +
                                  [(self.evaluator.DIRECTION_METRIC, direction)]}

    def test_expected_isolated_first_decision(self):
        candidate, baseline = self.runs()
        result = self.evaluator.check_first_divergence(candidate, baseline)
        self.assertTrue(result['trial_isolation_verified'])
        self.assertEqual(result['first_nonclock_frame_difference'], 54)

    def test_earlier_numeric_difference_is_not_a_clock(self):
        candidate, baseline = self.runs()
        candidate['frames'][3]['result']['diagnostic']['pose_update_norm'] = .02
        result = self.evaluator.check_first_divergence(candidate, baseline)
        self.assertFalse(result['trial_isolation_verified'])
        self.assertEqual(result['first_nonclock_frame_difference'], 3)

    def test_direction_is_diagnostic_and_limits_remain_fixed(self):
        gates = self.evaluator.quality_gates(self.ratios(primary=1.15))
        self.assertFalse(gates['10_percent']['primary_ATE_rotation_RPE_translation_RPE_pass'])
        self.assertTrue(gates['20_percent']['primary_ATE_rotation_RPE_translation_RPE_pass'])
        self.assertIn('additional_direction_diagnostic_not_acceptance_standard', gates)

    def test_missing_primary_ratio_is_not_success(self):
        gates = self.evaluator.quality_gates(self.ratios(primary=None))
        self.assertFalse(gates['20_percent']['primary_ATE_rotation_RPE_translation_RPE_pass'])

    def test_failed_or_wrong_schema_is_rejected_before_trajectory(self):
        schema, status, driver = self.evaluator.CONTRACTS['candidate']
        for override in (dict(status='failed'), dict(schema='v39_DPVO_half_original_Torch_CPU_research_replay')):
            run = dict(schema=schema, status=status, script_sha256=driver)
            run.update(override)
            with self.assertRaises(ValueError):
                self.evaluator.check_completed_run(run, 'candidate')


if __name__ == '__main__':
    unittest.main()
