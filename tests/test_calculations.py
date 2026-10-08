"""运行：python -m unittest tests.test_calculations -v

所有改错发生在临时副本，不改仓库输入或结果。
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import check_calculations as check
from scripts import validate_research


ROOT = Path(__file__).resolve().parents[1]


def rewrite_csv(path, change):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, records = reader.fieldnames, list(reader)
    change(records)
    # 兼容旧结果没有RMSE列、而新测试需要补充该列的情况。
    fields += [field for row in records for field in row if field not in fields]
    fields = list(dict.fromkeys(fields))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def rewrite_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sync_manifest(root):
    """同步摘要也救不了错误公式；独立检查根本不读取该清单。"""
    path = root / "examples/results/run_manifest.json"
    if path.exists():
        def update(data):
            for section in ["input_sha256", "source_sha256", "result_sha256"]:
                for name in data.get(section, {}):
                    target = root / name
                    if target.exists():
                        data[section][name] = hashlib.sha256(target.read_bytes()).hexdigest()
        rewrite_json(path, update)


class KnownAnswers(unittest.TestCase):
    def test_mae_rmse(self):
        mae, rmse = check.error_metrics([0, 10, 10], [0, 0, 20])
        self.assertAlmostEqual(mae, 20 / 3)
        self.assertAlmostEqual(rmse, (200 / 3) ** .5)

    def test_eiq_repeated_sku_and_daily_order_key(self):
        raw = [dict(date="day1", order_id="O1", sku="A", quantity="2"),
               dict(date="day1", order_id="O1", sku="A", quantity="3"),
               dict(date="day1", order_id="O1", sku="B", quantity="1"),
               dict(date="day1", order_id="O2", sku="A", quantity="4"),
               dict(date="day2", order_id="O1", sku="A", quantity="7")]
        orders, items, normalized = check.daily_eiq(raw)
        self.assertEqual(orders[("day1", "O1")], {"EQ": 6, "EN": 2})
        self.assertEqual(items[("day1", "A")], {"IQ": 9, "IK": 2})
        self.assertEqual(orders[("day2", "O1")], {"EQ": 7, "EN": 1})
        self.assertEqual(normalized, 4)

    def test_abc_exact_boundary_and_crossing_item(self):
        labels, shares = check.abc_classification([80, 15, 5, 0])
        self.assertEqual(labels, ["A", "B", "C", "ZERO"])
        self.assertEqual(shares, [.8, .95, 1., 1.])
        self.assertEqual(check.abc_classification([85, 10, 5])[0], ["A", "B", "C"])

    def test_location_denominators_and_radius_boundary(self):
        cities = [dict(city_id="a", x_km=0, y_km=0, demand=1),
                  dict(city_id="b", x_km=3, y_km=4, demand=9)]
        center = [dict(cluster=0, x_km=0, y_km=0)]
        _, metrics = check.location_metrics(cities, center, 4)
        self.assertEqual(metrics, {"city_coverage": .5, "demand_coverage": .1,
                                   "weighted_distance_km": 4.5})
        self.assertEqual(check.location_metrics(cities, center, 5)[1]["city_coverage"], 1)

    def test_fixed_centers_cover_no_less_when_radius_grows(self):
        cities = [dict(city_id=str(i), x_km=i, y_km=0, demand=i + 1) for i in range(8)]
        center = [dict(cluster=0, x_km=2, y_km=0)]
        scores = [check.location_metrics(cities, center, radius)[1] for radius in [0, 1, 2, 5]]
        for previous, current in zip(scores, scores[1:]):
            self.assertLessEqual(previous["city_coverage"], current["city_coverage"])
            self.assertLessEqual(previous["demand_coverage"], current["demand_coverage"])

    def test_silhouette_hand_answer_and_singleton(self):
        self.assertAlmostEqual(check.silhouette([[0], [2], [10], [12]], [0, 0, 1, 1]), 79 / 99)
        self.assertAlmostEqual(check.silhouette([[0], [2], [10]], [0, 0, 1]), (0.8 + .75) / 3)

    def test_standardization_uses_population_std_and_zero_column(self):
        self.assertEqual(check.standardized([[1, 7], [3, 7]]), [[-1., 0.], [1., 0.]])

    def test_ari_label_permutation_and_disagreement(self):
        self.assertEqual(check.adjusted_rand([0, 0, 1, 1], [7, 7, 9, 9]), 1)
        self.assertAlmostEqual(check.adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]), -.5)
        self.assertEqual(check.adjusted_rand([0, 0], [8, 8]), 1)


class BrokenOutputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "examples/sample_inputs", self.root / "examples/sample_inputs")
        shutil.copytree(ROOT / "examples/results", self.root / "examples/results")
        manifest = json.loads((self.root / "examples/results/run_manifest.json").read_text(encoding="utf-8"))
        for name in manifest.get("source_sha256", {}):
            target = self.root / name
            if (ROOT / name).exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_outputs(self):
        for run_check in check.CHECKS.values():
            run_check(self.root)

    def test_mae_divided_by_ten_even_with_synced_results(self):
        output = self.root / "examples/results/task01"
        def break_metrics(records):
            for row in records:
                row["mae"] = float(row["mae"]) / 10
        rewrite_csv(output / "metrics.csv", break_metrics)
        def break_summary(data):
            data["mean_mae"] = {key: value / 10 for key, value in data["mean_mae"].items()}
        rewrite_json(output / "summary.json", break_summary)
        sync_manifest(self.root)
        self.assertEqual(validate_research.evidence_errors(self.root), [])
        with self.assertRaisesRegex(check.CalculationError, "MAE"):
            check.check_forecast(self.root)

    def test_counting_rows_as_distinct_sku_is_detected(self):
        path = self.root / "examples/results/task02/order_eiq.csv"
        rewrite_csv(path, lambda data: data[0].update(EN=float(data[0]["EN"]) + 1))
        sync_manifest(self.root)
        with self.assertRaisesRegex(check.CalculationError, "EN"):
            check.check_warehouse(self.root)

    def test_wrong_saved_rmse_is_detected(self):
        output = self.root / "examples/results/task01"
        predicted = check.rows(output / "predictions.csv")
        def break_rmse(records):
            for record in records:
                members = [row for row in predicted if row["cutoff_day"] == record["cutoff_day"]
                           and row["model"] == record["model"]]
                record["rmse"] = check.error_metrics([row["actual"] for row in members],
                                                     [row["prediction"] for row in members])[1]
            records[0]["rmse"] /= 10
        rewrite_csv(output / "metrics.csv", break_rmse)
        sync_manifest(self.root)
        with self.assertRaisesRegex(check.CalculationError, "RMSE"):
            check.check_forecast(self.root)

    def test_wrong_abc_class_is_detected(self):
        path = self.root / "examples/results/task02/item_eiq_abc.csv"
        rewrite_csv(path, lambda data: data[0].update(ABC="C"))
        sync_manifest(self.root)
        with self.assertRaisesRegex(check.CalculationError, "ABC"):
            check.check_warehouse(self.root)

    def test_city_fraction_misreported_as_demand_fraction(self):
        path = self.root / "examples/results/task03/coverage.csv"
        rewrite_csv(path, lambda data: data[0].update(demand_coverage=data[0]["city_coverage"]))
        sync_manifest(self.root)
        with self.assertRaisesRegex(check.CalculationError, "demand_coverage"):
            check.check_locations(self.root)

    def test_wrong_silhouette_in_summary_and_table_is_detected(self):
        output = self.root / "examples/results/task04"
        rewrite_json(output / "summary.json", lambda data: data.update(silhouette_k3=.99))
        def change(records):
            for row in records:
                if row["method"] == "kmeans" and row["k"] == "3":
                    row["silhouette"] = .99
        rewrite_csv(output / "comparison.csv", change)
        sync_manifest(self.root)
        with self.assertRaisesRegex(check.CalculationError, "轮廓系数"):
            check.check_customers(self.root)

    def test_wrong_ari_is_detected(self):
        output = self.root / "examples/results/task04"
        rewrite_json(output / "summary.json", lambda data: data.update(ari_to_volume_rule=.123))
        sync_manifest(self.root)
        with self.assertRaisesRegex(check.CalculationError, "ARI"):
            check.check_customers(self.root)

    def test_stress_mae_divided_by_ten_with_synced_summary(self):
        output = self.root / "examples/results/stress"
        def break_scores(records):
            for row in records:
                row["mae"] = float(row["mae"]) / 10
        rewrite_csv(output / "prediction_metrics.csv", break_scores)
        def break_summary(data):
            for by_model in data["forecast_mean_mae"].values():
                for model in by_model:
                    by_model[model] /= 10
        rewrite_json(output / "summary.json", break_summary)
        sync_manifest(self.root)
        with self.assertRaisesRegex(check.CalculationError, "MAE"):
            check.check_stress(self.root)

    def test_stress_customer_time_ari_changed_in_table_and_summary(self):
        output = self.root / "examples/results/stress"
        def change(records):
            for row in records:
                if row["period"] == "later" and row["method"] == "kmeans_fixed":
                    row["ari_to_first_period"] = .99
        rewrite_csv(output / "customer_metrics.csv", change)
        rewrite_json(output / "summary.json", lambda data: data.update(customer_fixed_model_time_ari=.99))
        sync_manifest(self.root)
        with self.assertRaisesRegex(check.CalculationError, "ari_to_first_period"):
            check.check_stress(self.root)


if __name__ == "__main__":
    unittest.main()
