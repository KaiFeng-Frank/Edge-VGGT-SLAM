"""Analytical tiny tests; no real checkpoint, features, images or native code."""
import unittest
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HELPER = HERE / "corr_numpy_oracle.py"
if not HELPER.is_file():
    HELPER = HERE.parent / "deploy/rk3588/validation/dpvo_native/corr_numpy_oracle.py"
if not HELPER.is_file():
    raise RuntimeError("Missing published dpvo_native/corr_numpy_oracle.py dependency")
SPEC = importlib.util.spec_from_file_location("independent_corr_numpy_oracle", HELPER)
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def inputs(h=8, w=8, b=1, patches=1, views=1, ph=1, pw=1, edges=1):
    return (np.zeros((b, patches, 8, ph, pw), np.float32),
            np.zeros((b, views, 8, h, w), np.float32),
            np.zeros((b, edges, 2, ph, pw), np.float32),
            np.zeros(edges, np.int64), np.zeros(edges, np.int64))


class OracleTests(unittest.TestCase):
    def test_integer_channel_dot(self):
        g, f, c, k, j = inputs()
        g.fill(2); f.fill(3)
        got = oracle.tiny_full_oracle(g, f, c, k, j, radius=0)
        np.testing.assert_array_equal(got, [[[48., 48.]]])

    def test_offset_x_outermost_y_inner(self):
        g, f, c, k, j = inputs()
        g[:, :, 0] = 1
        y, x = np.mgrid[:8, :8]
        f[:, :, 0] = 100*y+x
        c[:, :, 0] = 2; c[:, :, 1] = 3
        actual = oracle.selected_elements(g, f, c, k, j,
                   [(0, 0, 0, 2, 0, 0, 0), (0, 0, 2, 0, 0, 0, 0)], radius=1)
        np.testing.assert_array_equal(actual, [401., 203.])

    def test_fractional_negative_padding(self):
        g, f, c, k, j = inputs(h=4, w=4)
        g[:, :, 0] = 1
        y, x = np.mgrid[:4, :4]
        f[:, :, 0] = 1+10*y+x
        c[:, :, 0] = -.25; c[:, :, 1] = .5
        actual = oracle.tiny_full_oracle(g, f, c, k, j, radius=0)
        expected_pool = .9375*.875*17.5
        np.testing.assert_array_equal(actual, [[[4.5, expected_pool]]])

    def test_level_coordinate_division_precedes_floor(self):
        g, f, c, k, j = inputs(h=8, w=8)
        g[:, :, 0] = 1
        f[:, :, 0, :4, :4] = 2
        f[:, :, 0, :4, 4:] = 6
        f[:, :, 0, 4:, :4] = 10
        f[:, :, 0, 4:, 4:] = 14
        c[:, :, 0] = 1.5; c[:, :, 1] = 2.5
        got = oracle.tiny_full_oracle(g, f, c, k, j, radius=0)
        # level4 coordinates (.375,.625), not floor(original)/4.
        self.assertEqual(got[0, 0, 1], 2*(1-.375)*(1-.625)+6*.375*(1-.625)+10*(1-.375)*.625+14*.375*.625)

    def test_pooling_discards_incomplete_rows_and_columns(self):
        g, f, c, k, j = inputs(h=9, w=10)
        g[:, :, 0] = 1
        f[:, :, 0, :8, :8] = 2
        f[:, :, 0, 8:, :] = 999; f[:, :, 0, :, 8:] = 999
        got = oracle.tiny_full_oracle(g, f, c, k, j, radius=0)
        np.testing.assert_array_equal(got, [[[2., 2.]]])

    def test_batch_patch_view_and_nonsquare_patch_indices(self):
        g, f, c, k, j = inputs(b=2, patches=2, views=2, ph=1, pw=2, edges=2)
        g[0, 0, 0] = [1, 2]; g[0, 1, 0] = [3, 4]
        g[1, 0, 0] = [5, 6]; g[1, 1, 0] = [7, 8]
        f[0, 0, 0] = 10; f[0, 1, 0] = 20
        f[1, 0, 0] = 30; f[1, 1, 0] = 40
        k[:] = [1, 0]; j[:] = [0, 1]
        got = oracle.tiny_full_oracle(g, f, c, k, j, radius=0)
        np.testing.assert_array_equal(got, [[[30, 30, 40, 40], [20, 20, 40, 40]],
                                          [[210, 210, 240, 240], [200, 200, 240, 240]]])

    def test_level_is_innermost_flat_dimension(self):
        g, f, c, k, j = inputs(ph=1, pw=2)
        g[:, :, 0] = [1, 3]
        f[:, :, 0] = np.arange(64, dtype=np.float32).reshape(8, 8)
        c[:, :, 0] = 1; c[:, :, 1] = 1
        got = oracle.tiny_full_oracle(g, f, c, k, j, radius=0)
        # FP32 pooled first cell=13.5; interpolation at (.25,.25)=22.5.
        np.testing.assert_array_equal(got, [[[9., 22.5, 27., 67.5]]])

    def test_all_outside_is_zero(self):
        g, f, c, k, j = inputs()
        g.fill(1); f.fill(2); c.fill(-200)
        self.assertFalse(oracle.tiny_full_oracle(g, f, c, k, j, radius=3).any())

    def test_empty_edges(self):
        got = oracle.tiny_full_oracle(*inputs(edges=0), radius=3)
        self.assertEqual(got.shape, (1, 0, 882//9))

    def test_invalid_inputs_are_rejected(self):
        for mutate in (lambda x: x[2].fill(np.nan), lambda x: x[2].fill(2**30),
                       lambda x: x[3].fill(-1), lambda x: x[4].fill(1),
                       lambda x: x[0].fill(np.inf)):
            value = inputs(); mutate(value)
            with self.assertRaises(ValueError): oracle.tiny_full_oracle(*value, radius=0)
        value = list(inputs()); value[3] = value[3].astype(np.int32)
        with self.assertRaises(ValueError): oracle.tiny_full_oracle(*value, radius=0)

    def test_inputs_unchanged_and_result_independent(self):
        value = inputs(); value[0].fill(1); value[1].fill(1)
        before = [oracle.raw_metadata(x) for x in value]
        result = oracle.tiny_full_oracle(*value, radius=0)
        result.fill(777)
        self.assertEqual(before, [oracle.raw_metadata(x) for x in value])

    def test_float64_dot_oracle_and_fixed_threshold(self):
        g, f, c, k, j = inputs()
        g[:] = np.array([1e8, 1, -1e8, 0, 0, 0, 0, 0], np.float32)[None, None, :, None, None]
        f.fill(1)
        np.testing.assert_array_equal(oracle.tiny_full_oracle(g, f, c, k, j, radius=0), [[[1., 1.]]])
        expected = np.array([1.], np.float64)
        lo = oracle.ATOL+oracle.RTOL
        self.assertEqual(oracle.error_stats(expected+lo*.999, expected)["outside_fixed_FP32_gate"], 0)
        self.assertEqual(oracle.error_stats(expected+lo*1.001, expected)["outside_fixed_FP32_gate"], 1)
        self.assertEqual(oracle.error_stats(np.zeros(1), np.zeros(1))["relative_L2"], None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
