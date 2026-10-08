"""运行失败、中文文件编码及合法退化数据的回归检查。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import validate_research
from scripts.check_calculations import check_customers
from scripts.check_public_research import check_public_research
from scripts.predict_demand import main as predict_demand, demand_history
from common import read_input, write_json, write_csv
from tests.test_public_research import research

ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


segments = load_example("04_customer_segments")
runner = load_example("run_all")


class RuntimeChecks(unittest.TestCase):
    def test_generated_files_are_utf8_with_lf(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_json({"城市": "吉林"}, folder / "summary.json")
            write_csv(pd.DataFrame({"城市": ["吉林"]}), folder / "cities.csv")
            for path in folder.iterdir():
                self.assertIn("吉林", path.read_bytes().decode("utf-8"))
                self.assertNotIn(b"\r\n", path.read_bytes())

    def test_failure_replaces_old_success_and_removes_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "examples/results"
            output.mkdir(parents=True)
            (output / "report.md").write_text("OLD SUCCESS", encoding="utf-8", newline="\n")
            (output / "run_manifest.json").write_text('{"status":"success"}', encoding="utf-8", newline="\n")
            (output / "claims.json").write_text("[]", encoding="utf-8", newline="\n")
            failed = subprocess.CompletedProcess([], 1, "中文输出\n", "故意失败\n")
            with patch.object(runner, "RESULT", output), patch.object(runner.subprocess, "run", return_value=failed) as run:
                with self.assertRaisesRegex(RuntimeError, "失败"):
                    runner.main()
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_steps"], [])
            self.assertTrue(any("不是success" in error for error in validate_research.evidence_errors(Path(folder))))
            self.assertNotIn("OLD SUCCESS", (output / "report.md").read_text(encoding="utf-8"))
            self.assertFalse((output / "claims.json").exists())
            self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
            self.assertIn("故意失败", (output / "run.log").read_text(encoding="utf-8"))

    def test_gbk_default_for_resource_search_and_validation(self):
        # 强制未指定编码的Path.open使用GBK，文件本身仍为UTF-8。
        original = Path.open
        def gbk_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            if "b" not in mode and encoding is None:
                encoding = "gbk"
            return original(path, mode, buffering, encoding, errors, newline)
        with patch.object(Path, "open", gbk_open):
            self.assertEqual(validate_research.evidence_errors(ROOT), [])
            import runpy
            with patch("sys.argv", ["find_resources.py", "--keyword", "配送"]):
                runpy.run_path(str(ROOT / "scripts/find_resources.py"), run_name="__main__")


class CustomerBoundaryChecks(unittest.TestCase):
    def test_numeric_customer_ids_keep_leading_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "business_volume.csv").write_text(
                "customer_id,business_volume\n001,10\n1,20\n", encoding="utf-8", newline="\n")
            data = read_input("business_volume.csv", ["customer_id", "business_volume"], path)
            self.assertEqual(data.customer_id.tolist(), ["001", "1"])

    def run_customers(self, count, identical=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, output = root / "examples/sample_inputs", root / "examples/results/task04"
            inputs.mkdir(parents=True)
            bills = []
            for i in range(count):
                orders = 1 if identical else 1 + i % 3
                for j in range(orders):
                    bills.append([f"C{i}", f"W{i}_{j}", "2025-04-01", 60 / orders])
            pd.DataFrame(bills, columns=["customer_id", "waybill_id", "date", "quantity"]).to_csv(
                inputs / "waybills.csv", index=False, encoding="utf-8")
            pd.DataFrame([[f"C{i}", 60] for i in range(count)], columns=["customer_id", "business_volume"]).to_csv(
                inputs / "business_volume.csv", index=False, encoding="utf-8")
            segments.main(["--input-dir", str(inputs), "--output-dir", str(output)])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["volume_rule_silhouette_reason"])
            self.assertTrue((output / "profiles.csv").exists())
            check_customers(root)
            return summary

    def test_sixty_equal_business_volumes(self):
        self.run_customers(60)

    def test_one_customer(self):
        self.assertIsNone(self.run_customers(1)["silhouette"])

    def test_two_customers(self):
        self.run_customers(2)

    def test_all_features_identical(self):
        self.assertEqual(self.run_customers(60, identical=True)["actual_clusters"], 1)


class CustomDataChecks(unittest.TestCase):
    def test_future_export_ignores_later_values(self):
        dates = pd.date_range("2025-01-01", periods=40)
        raw = pd.DataFrame({"date": dates, "region": "R", "sku": "S", "quantity": np.arange(40)})
        as_of = pd.Timestamp("2025-02-05")
        history = demand_history(raw, as_of)
        raw.loc[raw.date >= as_of, "quantity"] = 999999
        pd.testing.assert_series_equal(history, demand_history(raw, as_of))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data = folder / "demand.csv"
            raw.to_csv(data, index=False, encoding="utf-8")
            predict_demand(["--data", str(data), "--output-dir", str(folder / "out"),
                            "--as-of", "2025-02-05", "--horizon-days", "7"])
            predictions = pd.read_csv(folder / "out/future_predictions.csv", encoding="utf-8")
            self.assertEqual(len(predictions), 21)
            self.assertEqual(predictions.date.min(), "2025-02-05")
            self.assertTrue((predictions.prediction < 100).all())

    def test_custom_delivery_zero_error_and_no_long_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data, output = folder / "records.csv", folder / "out"
            days = pd.date_range("2026-01-01", periods=60)
            raw = pd.DataFrame({"order_id": range(60), "city": "自有城市", "ds": days.strftime("%Y-%m-%d"),
                                "accept_time": (days + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                                "delivery_time": (days + pd.Timedelta(hours=8, minutes=10)).strftime("%Y-%m-%d %H:%M:%S")})
            raw.to_csv(data, index=False, encoding="utf-8-sig")
            research.main(["--custom-data", "--data", str(data), "--output-dir", str(output),
                           "--horizon-days", "7", "--windows", "1", "--hour-min-records", "1",
                           "--observation-end", "2026-02-19"])
            check_public_research(output)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["data_kind"], "user_supplied_records")
            self.assertEqual(summary["duration_record_mean_mae_minutes"]["hour_median"], 0)
            audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["rows"], 60)
            self.assertEqual(audit["observed_completed_rows"], 50)
            status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["input_file"], "records.csv")
            self.assertEqual(status["source_sha256"], summary["source_sha256"])
            self.assertNotIn("data_path", status)
            self.assertNotIn("command", status)
            self.assertNotIn(str(folder), json.dumps(status, ensure_ascii=False))
            # 同一输出目录再次失败，应替换成功报告并让校验拒绝。
            with self.assertRaises(ValueError):
                research.main(["--data", str(data), "--output-dir", str(output)])
            self.assertIn("failed", (output / "report.md").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "未成功"):
                check_public_research(output)

    def test_slow_order_crossing_observation_end_cannot_be_dropped_for_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data, output = folder / "records.csv", folder / "out"
            days = pd.date_range("2026-01-01", periods=60)
            records = [{"order_id": str(i), "city": "Test", "ds": str(day.date()),
                        "accept_time": str(day + pd.Timedelta(hours=8)),
                        "delivery_time": str(day + pd.Timedelta(hours=8, minutes=10))}
                       for i, day in enumerate(days)]
            records.append({"order_id": "slow", "city": "Test", "ds": "2026-02-22",
                            "accept_time": "2026-02-19 08:00:00", "delivery_time": "2026-02-22 08:00:00"})
            pd.DataFrame(records).to_csv(data, index=False, encoding="utf-8")
            command = ["--custom-data", "--data", str(data), "--output-dir", str(output),
                       "--horizon-days", "7", "--windows", "1", "--hour-min-records", "1"]
            # 截止日当天的快单完成，慢单三天后完成；旧实现只剩快单，MAE为0。
            with self.assertRaisesRegex(ValueError, "1条在观察截止日后才完成"):
                research.main(command + ["--observation-end", "2026-02-19"])
            maturity = pd.read_csv(output / "duration_label_maturity.csv", encoding="utf-8")
            self.assertEqual(maturity.accepted_records.tolist(), [8])
            self.assertEqual(maturity.mature_records.tolist(), [7])
            self.assertEqual(maturity.pending_records.tolist(), [1])
            self.assertFalse((output / "summary.json").exists())
            self.assertIn("failed", (output / "report.md").read_text(encoding="utf-8"))

            # 选择早一天结束的完整接单窗口，可以合法评分。
            research.main(command + ["--observation-end", "2026-02-19", "--evaluation-end", "2026-02-18"])
            check_public_research(output)
            predicted = pd.read_csv(output / "duration_predictions.csv", encoding="utf-8")
            self.assertEqual(predicted.accept_date.max(), "2026-02-18")

            # 延后观察、仍评价同一接单窗口时，慢单必须计入误差。
            research.main(command + ["--observation-end", "2026-02-22", "--evaluation-end", "2026-02-19"])
            check_public_research(output)
            metrics = pd.read_csv(output / "duration_metrics.csv", encoding="utf-8")
            self.assertEqual(metrics.test_records.tolist(), [8, 8])
            self.assertEqual(metrics.mae_minutes.tolist(), [538.75, 538.75])

    def test_failure_paths_stay_in_ignored_local_log(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data, output = folder / "private.csv", folder / "out"
            error = OSError(f"cannot open {str(data.resolve())!r}")
            with patch.object(research, "execute", side_effect=error):
                with self.assertRaises(OSError):
                    research.main(["--custom-data", "--data", str(data), "--output-dir", str(output)])
            status_text = (output / "run_status.json").read_text(encoding="utf-8")
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertNotIn(str(folder), status_text + report)
            self.assertNotIn(str(folder).replace("\\", "\\\\"), status_text + report)
            local_log = json.loads((output / "run.local.log").read_text(encoding="utf-8"))
            self.assertEqual(local_log["data_path"], str(data.resolve()))
            self.assertEqual(local_log["error"], f"OSError: {error}")


if __name__ == "__main__":
    unittest.main()
