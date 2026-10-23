"""用小型已知表检查预测时点；不靠真实数据的偶然完整性通过。"""
from copy import deepcopy
from contextlib import closing
import gc
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from data import BEHAVIOR, baselines, calendar_features, history_features, load_panel, make_fold


class DataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # SQLite上下文提交事务，并不自动关闭连接；先回收加载器的短期读连接。
        self.addCleanup(gc.collect)
        self.path = Path(self.temporary.name) / 'fixture.sqlite'
        self.days = pd.date_range('2015-08-01', periods=160)
        rows = []
        for item in range(4):
            if item == 0:
                positions = range(160)
            elif item == 1:
                positions = [t for t in range(40, 160) if t not in (93, 124, 130)]
            elif item == 2:
                # 预测时只有27次观测；未来再完整也不能提前获得资格。
                positions = [*range(94, 121), *range(121, 160)]
            else:
                # 预测时够28次，但之后没有记录；仍须保留预测资格。
                positions = range(65, 93)
            for t in positions:
                q = t % 11 if item == 0 else float((t + item) % 7 + 1)
                row = {'date': str(self.days[t].date()), 'item_id': str(item),
                       'cate_id': 'c' + str(item % 2), 'qty_alipay': q}
                row.update({name: float(t + 2 * item + j + 1) for j, name in enumerate(BEHAVIOR)})
                rows.append(row)
        self.frame = pd.DataFrame(rows)
        with closing(sqlite3.connect(self.path)) as connection:
            self.frame.to_sql('Item_feature', connection, index=False)
        self.panel = load_panel(self.path)

    def changed_future(self, origin):
        changed = deepcopy(self.panel)
        future = changed['values'][:, origin + 1:, :]
        future[:] = 999999.
        future[1::2, ::2, :] = np.nan
        changed['total'] = np.nansum(changed['values'], axis=0, dtype=np.float64).astype(np.float32)[None]
        changed['daily_count'] = np.isfinite(changed['values'][:, :, 0]).sum(axis=0)
        return changed

    def test_future_values_and_missingness_do_not_change_training_or_inputs(self):
        for scope in ('sku', 'national'):
            with self.subTest(scope=scope):
                original = make_fold(self.panel, 120, scope=scope, behavior=True)
                changed = make_fold(self.changed_future(120), 120, scope=scope, behavior=True)
                for key in ('x', 'y', 'base', 'train_origins', 'train_items',
                            'current_x', 'current_base', 'indices'):
                    np.testing.assert_equal(original[key], changed[key], err_msg=key)
                self.assertEqual(original['levels'], changed['levels'])
                self.assertFalse(np.array_equal(original['truth'], changed['truth'], equal_nan=True))

    def test_no_training_target_passes_current_cutoff(self):
        for scope in ('sku', 'national'):
            fold = make_fold(self.panel, 120, scope=scope, behavior=True, stride=1)
            self.assertLessEqual(fold['latest_label'], 120)
            self.assertTrue(np.all(fold['train_origins'] + 14 <= 120))
            raw = self.panel['values'] if scope == 'sku' else self.panel['total']
            for row, (o, item) in enumerate(zip(fold['train_origins'], fold['train_items'])):
                np.testing.assert_equal(fold['y'][row], raw[item, o + 1:o + 15, 0])

    def test_eligibility_uses_history_and_keeps_unobserved_future(self):
        fold = make_fold(self.panel, 120, scope='sku')
        self.assertEqual(self.panel['items'][fold['indices']].tolist(), ['0', '1', '3'])
        unknown_row = np.flatnonzero(self.panel['items'][fold['indices']] == '3')[0]
        self.assertTrue(np.isnan(fold['truth'][unknown_row]).all())
        self.assertTrue(np.isfinite(fold['current_x']).all())

    def test_missing_target_and_recorded_zero_remain_different(self):
        fold = make_fold(self.panel, 120, scope='sku')
        positions = {item: j for j, item in enumerate(self.panel['items'][fold['indices']])}
        self.assertEqual(float(fold['truth'][positions['0'], 0]), 0.)  # 日期121有记录，且数量为零。
        self.assertTrue(np.isnan(fold['truth'][positions['1'], 3]))  # 日期124没有商品记录。
        self.assertTrue(np.isnan(fold['y']).any())

    def test_national_truth_is_raw_record_sum_not_masked_eligible_sku_sum(self):
        national = make_fold(self.panel, 120, scope='national')
        sku = make_fold(self.panel, 120, scope='sku')
        expected = self.frame.groupby('date').qty_alipay.sum().reindex(
            [str(d.date()) for d in self.days[121:135]]).to_numpy()
        np.testing.assert_allclose(national['truth'][0], expected)
        self.assertTrue(np.isfinite(national['truth']).all())
        # 商品2历史不足，却已经出现在实际总记录里；全国目标不能被SKU资格筛掉。
        self.assertTrue(np.all(national['truth'][0] > np.nansum(sku['truth'], axis=0)))

    def test_startup_history_has_unknown_left_padding_and_no_tail_wrap(self):
        values = self.panel['values'][[0]]
        x, base = history_features(values, 27, self.days, behavior=True)
        altered = values.copy()
        altered[:, 28:, :] = 1e9
        x_changed, base_changed = history_features(altered, 27, self.days, behavior=True)
        np.testing.assert_equal(x, x_changed)
        np.testing.assert_equal(base, base_changed)
        self.assertAlmostEqual(float(base[0]), float(np.mean(values[0, :28, 0])), places=6)
        self.assertEqual(float(x[0, 57]), .5)
        fold = make_fold(self.panel, 55, scope='national', stride=1)
        self.assertEqual(int(fold['train_origins'].min()), 27)
        self.assertEqual(int(fold['train_origins'].max()) + 14, 55)

    def test_startup_baseline_never_reads_negative_calendar_indices(self):
        original = baselines(self.panel, 27, scope='national')
        changed = baselines(self.changed_future(27), 27, scope='national')
        for name in original:
            np.testing.assert_equal(original[name], changed[name], err_msg=name)
        expected = np.median(self.panel['total'][0, :28, 0])
        np.testing.assert_allclose(original['median56'], expected)

    def test_calendar_flags_are_fixed_by_date(self):
        dates = pd.DatetimeIndex(['2015-11-10', '2015-11-11', '2015-12-12'])
        features = calendar_features(dates).reshape(3, 6)
        np.testing.assert_array_equal(features[:, 4], [0, 1, 0])
        np.testing.assert_array_equal(features[:, 5], [0, 0, 1])
        np.testing.assert_allclose(features[:, 0],
            [np.sin(d.to_pydatetime().weekday() * 2 * np.pi / 7) for d in dates], atol=1e-7)
        self.assertEqual(calendar_features(dates[:1]).shape, (6,))

    def test_history_bounds_are_explicit(self):
        for origin in (26, len(self.days) - 14):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                history_features(self.panel['values'], origin, self.days)

    def test_loader_rejects_duplicate_product_day(self):
        with closing(sqlite3.connect(self.path)) as connection:
            self.frame.iloc[[0]].to_sql('Item_feature', connection, if_exists='append', index=False)
        with self.assertRaisesRegex(ValueError, '唯一'):
            load_panel(self.path)

    def test_loader_rejects_time_varying_category(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("UPDATE Item_feature SET cate_id='new' WHERE item_id='0' AND date=?",
                               (str(self.days[-1].date()),))
            connection.commit()
        with self.assertRaisesRegex(ValueError, '类别'):
            load_panel(self.path)

    def test_loader_does_not_turn_an_unrecorded_day_into_national_zero(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute('DELETE FROM Item_feature WHERE date=?', (str(self.days[80].date()),))
            connection.commit()
        with self.assertRaisesRegex(ValueError, '全国日期'):
            load_panel(self.path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
