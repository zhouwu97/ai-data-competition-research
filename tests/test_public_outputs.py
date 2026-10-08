"""篡改逐条结果、汇总或报告，真实研究检查必须发现；只改临时副本。"""
from pathlib import Path
import shutil
import re
import tempfile
import unittest

from scripts.check_public_research import check_public_research
from scripts.check_calculations import CalculationError
from tests.test_calculations import rewrite_csv, rewrite_json

ROOT = Path(__file__).resolve().parents[1]


class PublicOutputChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "results"
        shutil.copytree(ROOT / "research/lade-jilin/results", self.output)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_results(self):
        check_public_research(self.output)

    def test_false_overall_mae(self):
        def change(data):
            data["duration_record_mean_mae_minutes"]["hour_median"] = 1.
        rewrite_json(self.output / "summary.json", change)
        with self.assertRaisesRegex(CalculationError, "时长MAE"):
            check_public_research(self.output)

    def test_false_window_and_summary_even_when_consistent(self):
        def change(data):
            for row in data:
                row["mae_minutes"] = float(row["mae_minutes"]) / 10
        rewrite_csv(self.output / "duration_metrics.csv", change)
        def summary(data):
            data["duration_record_mean_mae_minutes"] = {
                key: value / 10 for key, value in data["duration_record_mean_mae_minutes"].items()}
        rewrite_json(self.output / "summary.json", summary)
        with self.assertRaisesRegex(CalculationError, "duration_metrics.csv.*MAE"):
            check_public_research(self.output)

    def test_false_tail_mae(self):
        rewrite_csv(self.output / "duration_metrics.csv",
                    lambda rows: rows[0].update(mae_over_360_minutes="1.0"))
        with self.assertRaisesRegex(CalculationError, "长尾MAE"):
            check_public_research(self.output)

    def test_wrong_test_count(self):
        rewrite_csv(self.output / "duration_metrics.csv", lambda rows: rows[0].update(test_records="1"))
        with self.assertRaisesRegex(CalculationError, "测试条数"):
            check_public_research(self.output)

    def test_changed_actual_in_one_method(self):
        rewrite_csv(self.output / "duration_predictions.csv", lambda rows: rows[0].update(actual_minutes="1"))
        with self.assertRaisesRegex(CalculationError, "相同逐条对象"):
            check_public_research(self.output)

    def test_changed_prediction(self):
        rewrite_csv(self.output / "duration_predictions.csv", lambda rows: rows[0].update(prediction_minutes="10000"))
        with self.assertRaisesRegex(CalculationError, "MAE"):
            check_public_research(self.output)

    def test_false_volume_window(self):
        rewrite_csv(self.output / "volume_metrics.csv", lambda rows: rows[0].update(mae_records_per_day="1.0"))
        with self.assertRaisesRegex(CalculationError, "volume_metrics.csv.*MAE"):
            check_public_research(self.output)

    def test_wrong_report_with_number_still_elsewhere(self):
        path = self.output / "report.md"
        content = path.read_text(encoding="utf-8")
        line = next(line for line in content.splitlines() if line.startswith("| hour_median |"))
        path.write_text(content.replace(line, "| hour_median | 1.0000 |") + "\n" + line + "\n", encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(CalculationError, "report.md.*MAE"):
            check_public_research(self.output)

    def test_false_diagnostic_group(self):
        path = self.output / "duration_diagnostics.csv"
        self.assertTrue(path.exists(), "先重新运行吉林研究生成诊断表。")
        rewrite_csv(path, lambda rows: rows[0].update(mae_minutes="10000"))
        with self.assertRaisesRegex(CalculationError, "时长诊断.*MAE"):
            check_public_research(self.output)

    def test_false_report_gain_even_with_correct_phrase_elsewhere(self):
        path = self.output / "report.md"
        content = path.read_text(encoding="utf-8")
        phrase = re.search(r"按小时分组相对总体中位数的MAE变化为[^（\n]+", content).group()
        path.write_text(content.replace(phrase, "按小时分组相对总体中位数的MAE变化为99.00%")
                        + "\n" + phrase + "\n", encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(CalculationError, "时长MAE变化"):
            check_public_research(self.output)

    def test_false_report_tail(self):
        path = self.output / "report.md"
        content = path.read_text(encoding="utf-8")
        changed = re.sub(r"超过6小时的运单，按小时分组误差仍为[\d.]+分钟",
                         "超过6小时的运单，按小时分组误差仍为1.00分钟", content)
        path.write_text(changed, encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(CalculationError, "长尾误差"):
            check_public_research(self.output)


if __name__ == "__main__":
    unittest.main()
