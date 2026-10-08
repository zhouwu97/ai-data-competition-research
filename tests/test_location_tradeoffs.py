"""几个小答案保护目标方向、半径边界与损失预算，不重复整套实验。"""
import importlib.util
from pathlib import Path
import sys
import unittest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
spec = importlib.util.spec_from_file_location("location_tradeoffs", EXAMPLES / "08_location_tradeoffs.py")
tradeoffs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tradeoffs)


def plan(centers, loss, cover, city=.5, distance=1):
    return {"centers": centers, "weighted_squared_loss": loss, "demand_coverage": cover,
            "city_coverage": city, "weighted_distance_km": distance}


class LocationTradeoffs(unittest.TestCase):
    def test_weighted_metrics_and_radius_boundary(self):
        points = [dict(city_id="A", x_km=0, y_km=0, demand=1),
                  dict(city_id="B", x_km=3, y_km=4, demand=9)]
        strict = tradeoffs.evaluate(points, [points[0]], 4)
        self.assertEqual(strict["weighted_squared_loss"], 225)
        self.assertEqual(strict["weighted_distance_km"], 4.5)
        self.assertEqual(strict["city_coverage"], .5)
        self.assertEqual(strict["demand_coverage"], .1)
        self.assertEqual(tradeoffs.evaluate(points, [points[0]], 5)["demand_coverage"], 1)

    def test_two_objective_direction_and_equal_points(self):
        cheap, service, inferior = plan("A", 10, .5), plan("B", 11, .8), plan("C", 12, .5)
        self.assertFalse(tradeoffs.dominates(cheap, service))
        self.assertFalse(tradeoffs.dominates(service, cheap))
        self.assertTrue(tradeoffs.dominates(cheap, inferior))
        self.assertFalse(tradeoffs.dominates(cheap, plan("D", 10, .5)))
        self.assertFalse(tradeoffs.dominates(plan("E", 10 - 1e-10, .5), cheap))

    def test_budget_is_relative_and_ties_are_deterministic(self):
        rows = [plan("A", 100, .5), plan("C", 100.2, .8), plan("B", 100.2, .8),
                plan("D", 100.201, .9)]
        self.assertEqual(tradeoffs.select(rows, "budgeted_demand_coverage", 0)["centers"], "A")
        self.assertEqual(tradeoffs.select(rows, "budgeted_demand_coverage", .002)["centers"], "B")
        self.assertEqual(tradeoffs.select(rows, "max_demand_coverage")["centers"], "D")
        with self.assertRaises(ValueError):
            tradeoffs.select(rows, "budgeted_demand_coverage", -.01)


if __name__ == "__main__":
    unittest.main()
