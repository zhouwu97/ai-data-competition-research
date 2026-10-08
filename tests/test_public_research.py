"""公开配送练习的两个小答案：标签可用时点与小时分组回退。

运行：python -m unittest tests.test_public_research -v
"""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lade_public_research", ROOT / "research/lade-jilin/run.py")
research = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(research)


class PublicResearchChecks(unittest.TestCase):
    def test_unfinished_and_boundary_labels_are_excluded(self):
        origin = pd.Timestamp("2022-01-02 12:00:00")
        frame = pd.DataFrame({
            "accept_time": pd.to_datetime(["2022-01-01 09:00", "2022-01-02 11:00",
                                            "2022-01-02 13:00", "2022-01-02 10:00"]),
            "delivery_time": pd.to_datetime(["2022-01-01 11:00", "2022-01-02 13:00",
                                              "2022-01-02 14:00", "2022-01-02 12:00"]),
            "minutes": [120, 120, 60, 120],
        })
        # 已接单但尚未完成、未来接单、恰在起点完成，都不能进训练集。
        self.assertEqual(research.mature_history(frame, origin).index.tolist(), [0])

    def test_hour_median_and_small_group_fallback(self):
        # 30条10分钟、29条100分钟、30条40分钟：89条的总体中位数是40。
        # 8点刚好30条可分组，9点29条需回退；11点没有历史，也回退。
        frame = pd.DataFrame({
            "accept_time": pd.to_datetime(["2022-01-01 08:00"] * 30
                                            + ["2022-01-01 09:00"] * 29
                                            + ["2022-01-01 10:00"] * 30),
            "minutes": [10] * 30 + [100] * 29 + [40] * 30,
        })
        estimates = research.duration_estimates(frame, [8, 9, 10, 11])
        self.assertEqual(estimates["global_median"].tolist(), [40, 40, 40, 40])
        self.assertEqual(estimates["hour_median"].tolist(), [10, 40, 40, 40])


if __name__ == "__main__":
    unittest.main()
