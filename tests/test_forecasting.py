"""预测边界的行为测试：未来答案改变，当前预测与选择必须保持不变。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from forecasting import (METHODS, forecast_at_origin, completed_window_scores,
                         select_by_completed_windows)


def assert_future_invariant(testcase, predictor, series, cutoff):
    """同一检验同时检查正常入口和故意泄漏入口，避免测试永远只会报通过。"""
    expected = predictor(series, cutoff)
    variants = [np.zeros(len(series) - cutoff),
                np.linspace(10000, 90000, len(series) - cutoff),
                series.iloc[cutoff:].to_numpy()[::-1]]
    for future in variants:
        changed = series.astype(float).copy()
        changed.iloc[cutoff:] = future
        actual = predictor(changed, cutoff)
        testcase.assertEqual(set(expected), set(actual))
        for method in expected:
            np.testing.assert_array_equal(actual[method], expected[method])


class ForecastBoundaryTests(unittest.TestCase):
    def setUp(self):
        # 用有变化的历史防止“预测总返回常数0”的无意义实现碰巧过关。
        day = np.arange(112)
        dates = pd.date_range("2025-01-01", periods=len(day))
        self.series = pd.Series(30 + .4 * day + np.array([0, 1, 2, 3, 5, 8, 4])[dates.dayofweek],
                                index=dates)

    def test_prediction_is_invariant_to_all_future_values(self):
        for cutoff in [56, 70, 84, 98]:
            with self.subTest(cutoff=cutoff):
                assert_future_invariant(self, forecast_at_origin, self.series, cutoff)

    def test_same_check_rejects_deliberate_future_leak(self):
        def leaking_predictor(series, cutoff):
            # 故意把预测期真实均值当预测；这是反例，不在正式练习中调用。
            return {"leaked": np.repeat(series.iloc[cutoff:cutoff + 14].mean(), 14)}
        with self.assertRaises(AssertionError):
            assert_future_invariant(self, leaking_predictor, self.series, 56)

    def test_selection_uses_only_completed_history(self):
        cutoff = 56
        expected = select_by_completed_windows(self.series.iloc[:cutoff])
        changed = self.series.copy()
        changed.iloc[cutoff:] = 1000000
        self.assertEqual(expected, select_by_completed_windows(changed.iloc[:cutoff]))
        used_cutoffs = {row["window_cutoff"] for row in expected[2]}
        self.assertEqual(used_cutoffs, {28, 42})
        self.assertTrue(all(pd.Timestamp(row["window_end"]) <= self.series.index[55]
                            for row in expected[2]))

    def test_unfinished_window_is_not_eligible(self):
        with self.assertRaisesRegex(ValueError, "尚未完整结束"):
            completed_window_scores(self.series.iloc[:55], [42])

    def test_simple_baseline_has_known_values(self):
        predictions = forecast_at_origin(self.series, 56)
        expected = sum(self.series.iloc[49:56]) / 7
        np.testing.assert_allclose(predictions["recent_mean"], expected, rtol=0, atol=1e-12)
        self.assertEqual(set(predictions), set(METHODS))
        for offset, value in enumerate(predictions["weekday_mean"]):
            weekday = self.series.index[56 + offset].dayofweek
            source = [self.series.iloc[i] for i in range(28, 56)
                      if self.series.index[i].dayofweek == weekday]
            self.assertAlmostEqual(value, sum(source) / len(source), places=12)


if __name__ == "__main__":
    unittest.main()
