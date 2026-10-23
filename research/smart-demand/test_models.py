"""核查未知目标不会变零、测试输入不会改变训练及种子复现。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import fit_predict, masked_objective, ridge_predict


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(52)
        self.x = rng.normal(size=(48, 4)).astype(np.float32)
        self.y = np.repeat(np.maximum(self.x[:, :1] + 3., 0.), 14, axis=1)
        self.base = np.full(48, 3., dtype=np.float32)
        self.px = np.zeros((2, 4), dtype=np.float32)
        self.pb = np.full(2, 3., dtype=np.float32)

    def fit(self, **kwargs):
        options = {"seed": 12, "epochs": 3, "batch_size": 16}
        options.update(kwargs)
        return fit_predict(self.x, self.y, self.base, self.px, self.pb, **options)

    def test_unknown_targets_have_no_loss(self):
        pred = torch.tensor([[2., 99.], [4., 3.]])
        target = torch.tensor([[1., float("nan")], [7., 3.]])
        self.assertAlmostEqual(float(masked_objective(pred, target, "mae", 2.)), 4 / 6)
        changed = pred.clone()
        changed[0, 1] = 100000.
        self.assertEqual(float(masked_objective(pred, target, "poisson", 2.)),
                         float(masked_objective(changed, target, "poisson", 2.)))

    def test_initial_predictions_are_bases(self):
        pred, trace = self.fit(epochs=0)
        np.testing.assert_allclose(pred, self.pb[:, None] * np.ones((2, 14)), rtol=1e-6)
        self.assertEqual(trace["parameters"], (4 * 64 + 64) + (64 * 32 + 32) + (32 * 14 + 14))

    def test_seed_reproduces_weights(self):
        first, t1 = self.fit()
        second, t2 = self.fit()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(t1["weights_sha256"], t2["weights_sha256"])

    def test_prediction_inputs_do_not_change_fit(self):
        first, t1 = self.fit()
        second, t2 = fit_predict(self.x, self.y, self.base,
                                np.vstack([self.px, np.full((1, 4), 1e9)]),
                                np.r_[self.pb, 1.].astype(np.float32),
                                seed=12, epochs=3, batch_size=16)
        np.testing.assert_array_equal(first, second[:2])
        self.assertEqual(t1["scaler_mean"], t2["scaler_mean"])
        self.assertEqual(t1["weights_sha256"], t2["weights_sha256"])

    def test_unobserved_rows_do_not_change_training(self):
        first, t1 = self.fit()
        second, t2 = fit_predict(np.vstack([self.x, np.full((1, 4), 1e6)]),
                                np.vstack([self.y, np.full((1, 14), np.nan)]),
                                np.r_[self.base, 1.], self.px, self.pb,
                                seed=12, epochs=3, batch_size=16)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(t1["scaler_mean"], t2["scaler_mean"])
        self.assertEqual(t2["dropped_unobserved_rows"], 1)

    def test_scales_must_be_positive(self):
        for value in [0., -1., np.inf, np.nan]:
            base = self.base.copy()
            base[0] = value
            with self.assertRaises(ValueError):
                fit_predict(self.x, self.y, base, self.px, self.pb, seed=12)

    def test_poisson_fits_with_partial_targets(self):
        self.y[::2, 3:7] = np.nan
        pred, trace = self.fit(loss="poisson")
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred >= 0).all())
        self.assertEqual(trace["observed_target_cells"], 48 * 14 - 24 * 4)

    def test_ridge_masks_targets_and_ignores_future_features(self):
        self.y[::2, 3:7] = np.nan
        first, t1 = ridge_predict(self.x, self.y, self.px)
        second, t2 = ridge_predict(self.x, self.y,
                                  np.vstack([self.px, np.full((1, 4), 1e9)]))
        np.testing.assert_allclose(first, second[:2], rtol=1e-6)
        self.assertTrue((first >= 0).all())
        self.assertEqual(t1["observed_rows_by_horizon"][3], 24)
        self.assertEqual(t1["scaler_mean"], t2["scaler_mean"])


if __name__ == "__main__":
    unittest.main()
