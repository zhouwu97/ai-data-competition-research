"""从原库和逐条预测独立核对研究数字，不导入训练入口或数据构造函数。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ('median14', 'median28', 'median56', 'mean28', 'weekday8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def raw_data(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        frame = pd.read_sql_query('SELECT date,item_id,qty_alipay FROM Item_feature', connection)
    frame['date'] = pd.to_datetime(frame['date'])
    if frame.duplicated(['date', 'item_id']).any():
        raise AssertionError('原始商品日重复。')
    days = pd.date_range(frame.date.min(), frame.date.max(), freq='D')
    items = np.sort(frame.item_id.unique())
    matrix = frame.pivot(index='item_id', columns='date', values='qty_alipay').reindex(index=items, columns=days).to_numpy(dtype=float)
    # 全国目标直接由原始记录求和；不复用商品队列或未来观测掩码。
    totals = frame.groupby('date').qty_alipay.sum().reindex(days)
    if totals.isna().any():
        raise AssertionError('原始日总量含完全无记录的日子。')
    return days, items, matrix, totals.to_numpy(dtype=float)[None, :]


def baseline_from_history(history, past_dates, future_dates, method):
    rows = []
    for values in history:
        if method.startswith('median'):
            width = int(method[6:])
            known = values[-width:]
            known = known[np.isfinite(known)]
            rows.append(np.full(len(future_dates), np.median(known) if len(known) else 0.))
        elif method == 'mean28':
            known = values[-28:]
            known = known[np.isfinite(known)]
            rows.append(np.full(len(future_dates), known.mean() if len(known) else 0.))
        else:
            row = []
            for day in future_dates:
                known = values[-56:][past_dates[-56:].dayofweek == day.dayofweek]
                known = known[np.isfinite(known)]
                row.append(known.mean() if len(known) else 0.)
            rows.append(row)
    return np.asarray(rows, dtype=float)


def independent_metrics(truth, prediction):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise AssertionError('预测维度、有限性或非负性不满足要求。')
    observed = np.isfinite(truth)
    difference = prediction.astype(np.float64)[observed] - truth.astype(np.float64)[observed]
    absolute = sum(abs(float(value)) for value in difference)
    target_sum = sum(float(value) for value in truth[observed])
    count = int(observed.sum())
    if count == 0:
        raise AssertionError('当前窗没有可评分目标。')
    return {'n': count, 'abs_error': absolute, 'mae': absolute / count,
            'bias': sum(float(value) for value in difference) / count,
            'wape': absolute / target_sum if target_sum else None,
            'truth_sum': target_sum,
            'pred_observed_sum': sum(float(value) for value in prediction[observed])}


def close(actual, expected, label, *, total=False):
    if expected is None:
        if actual is not None and not pd.isna(actual):
            raise AssertionError(f'{label}: 应为空。')
        return
    # 保存的是float32预测，CSV中的总和也可能按float32累计；仅容许相应舍入量。
    np.testing.assert_allclose(float(actual), float(expected), rtol=2e-6,
                               atol=.1 if total else 1e-5, err_msg=label)


def pooled(rows):
    count = sum(row['n'] for row in rows)
    return {'n': count, 'abs_error': sum(row['abs_error'] for row in rows),
            'mae': sum(row['abs_error'] for row in rows) / count,
            'bias': sum(row['bias'] * row['n'] for row in rows) / count}


def verify(output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'success':
        raise ValueError('实验尚未完成，等待 manifest.status=success 后再独立复核。')
    config = manifest['config']
    input_path = Path(manifest['input'])
    if sha256(input_path) != manifest['input_sha256']:
        raise AssertionError('原始输入摘要已改变。')
    if sha256(Path(__file__).with_name('config.json')) != manifest['config_sha256']:
        raise AssertionError('配置与执行版本不一致。')
    for name, digest in manifest['source_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise AssertionError(f'源码与执行完成时版本不一致：{name}')
    days, items, sku, national = raw_data(input_path)
    metrics = pd.read_csv(output / 'window_metrics.csv', dtype={'seed': str})
    metric_keys = ['scope', 'phase', 'start', 'method', 'seed']
    if metrics.duplicated(metric_keys).any():
        raise AssertionError('保存分数出现重复键。')
    indexed = metrics.set_index(metric_keys)
    selection = json.loads((output / 'selection_before_test.json').read_text(encoding='utf-8'))
    checks = defaultdict(list)
    scores = defaultdict(list)
    expected_metric_keys = set()
    baselines_checked = 0
    arrays_checked = 0
    ensemble_checks = 0
    dates_by_phase = {phase: [pd.Timestamp(value) for value in config[phase + '_starts']]
                      for phase in ('development', 'test')}
    for phase, starts in dates_by_phase.items():
        if len(starts) != 6 or any((b - a).days != 14 for a, b in zip(starts, starts[1:])):
            raise AssertionError('每阶段必须是六个非重叠14天窗口。')
    if dates_by_phase['development'][-1] + pd.Timedelta(days=14) != dates_by_phase['test'][0]:
        raise AssertionError('开发与测试窗口边界不一致。')

    for scope in config['scopes']:
        raw = sku if scope == 'sku' else national
        for phase, starts in dates_by_phase.items():
            candidate_names = ([candidate['name'] for candidate in config['candidates']]
                               if phase == 'development' else [selection[scope]['neural']])
            seeds = [str(seed) for seed in config[phase + '_seeds']]
            for start in starts:
                label = str(start.date())
                origin = int(days.get_loc(start)) - 1
                folder = output / 'predictions' / scope / phase / label
                with np.load(folder / 'truth.npz') as saved:
                    truth = saved['truth']
                    indices = saved['indices']
                    close(saved['origin'], origin, f'{scope}/{label}/origin')
                    history_counts = np.isfinite(raw[:, max(0, origin - 55):origin + 1]).sum(axis=1)
                    expected_indices = np.flatnonzero(history_counts >= 28)
                    np.testing.assert_array_equal(indices, expected_indices)
                    np.testing.assert_equal(truth, raw[indices, origin + 1:origin + 15])
                    base = []
                    for row in raw[indices, origin - 27:origin + 1]:
                        known = row[np.isfinite(row)]
                        base.append(max(float(known.mean()) if len(known) else 0., 1.))
                    np.testing.assert_allclose(saved['base'], base, rtol=1e-6, atol=1e-5)
                expected_files = {'truth.npz', 'ridge.npz'} | {name + '.npz' for name in BASELINES}
                methods = [(name, 'ensemble', folder / (name + '.npz')) for name in (*BASELINES, 'ridge')]
                for candidate in candidate_names:
                    seed_predictions = []
                    for seed in seeds:
                        filename = candidate + '_' + seed + '.npz'
                        expected_files.add(filename)
                        seed_predictions.append(np.load(folder / filename)['prediction'])
                        methods.append((candidate, seed, folder / filename))
                    filename = candidate + '_ensemble.npz'
                    expected_files.add(filename)
                    ensemble = np.load(folder / filename)['prediction']
                    np.testing.assert_allclose(ensemble, np.stack(seed_predictions).mean(axis=0),
                                               rtol=1e-6, atol=1e-5)
                    ensemble_checks += 1
                    methods.append((candidate, 'ensemble', folder / filename))
                if {path.name for path in folder.glob('*.npz')} != expected_files:
                    raise AssertionError(f'{scope}/{phase}/{label}: 预测文件集合不符。')
                for method, seed, file in methods:
                    prediction = np.load(file)['prediction']
                    if method in BASELINES:
                        expected = baseline_from_history(raw[indices, :origin + 1], days[:origin + 1],
                                                         days[origin + 1:origin + 15], method)
                        np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-5)
                        baselines_checked += 1
                    key = (scope, phase, label, method, seed)
                    expected_metric_keys.add(key)
                    saved_row = indexed.loc[key]
                    calculated = independent_metrics(truth, prediction)
                    for field, expected in calculated.items():
                        close(saved_row[field], expected, '/'.join(key) + '/' + field,
                              total=field in ('pred_observed_sum', 'truth_sum', 'abs_error'))
                    total14 = abs(float(prediction.astype(float).sum()) - float(truth.astype(float).sum())) if scope == 'national' else None
                    close(saved_row['total14_ae'], total14, '/'.join(key) + '/total14_ae', total=True)
                    close(saved_row['eligible_items'], len(indices), '/'.join(key) + '/eligible_items')
                    row = {'start': label, 'total14_ae': total14, **calculated}
                    scores[(scope, phase, method, seed)].append(row)
                    checks[scope].append({'phase': phase, 'start': label, 'method': method,
                                          'seed': seed, **row})
                    arrays_checked += 1
    if set(indexed.index.tolist()) != expected_metric_keys:
        raise AssertionError('CSV分数与预测文件键集合不一致。')

    fit_records = json.loads((output / 'fits.json').read_text(encoding='utf-8'))
    for row in fit_records:
        if pd.Timestamp(row['latest_label']) >= pd.Timestamp(row['start']):
            raise AssertionError('训练标签越过预测时点。')
    if len(fit_records) != manifest['neural_fits']:
        raise AssertionError('拟合次数与manifest不符。')
    summary = {}
    for scope in config['scopes']:
        dev = {method: pooled(rows)['mae'] for (s, phase, method, seed), rows in scores.items()
               if s == scope and phase == 'development' and seed == 'ensemble'}
        selected_neural = min(sorted(candidate['name'] for candidate in config['candidates']), key=lambda name: dev[name])
        selected_median = min(('median14', 'median28', 'median56'), key=lambda name: dev[name])
        if selection[scope]['neural'] != selected_neural or selection[scope]['median'] != selected_median:
            raise AssertionError('开发选择与独立重算不一致。')
        for name, value in dev.items():
            close(selection[scope]['development_mae'][name], value, f'{scope}/development/{name}')
        baseline_rows = scores[(scope, 'test', selected_median, 'ensemble')]
        neural_rows = scores[(scope, 'test', selected_neural, 'ensemble')]
        baseline_total, neural_total = pooled(baseline_rows), pooled(neural_rows)
        if len(baseline_rows) != 6 or len(neural_rows) != 6:
            raise AssertionError('选定方法缺少六个测试窗。')
        windows = []
        for baseline_row, neural_row in zip(baseline_rows, neural_rows):
            if baseline_row['start'] != neural_row['start']:
                raise AssertionError('配对窗口日期不一致。')
            windows.append({'start': baseline_row['start'], 'median_mae': baseline_row['mae'],
                            'neural_mae': neural_row['mae'], 'difference': neural_row['mae'] - baseline_row['mae'],
                            'median_total14_ae': baseline_row['total14_ae'],
                            'neural_total14_ae': neural_row['total14_ae']})
        seed_metrics = {str(seed): pooled(scores[(scope, 'test', selected_neural, str(seed))])
                        for seed in config['test_seeds']}
        lower_overall = neural_total['mae'] < baseline_total['mae']
        no_window_worse = all(row['difference'] <= 0 for row in windows)
        all_seeds_lower = all(row['mae'] < baseline_total['mae'] for row in seed_metrics.values())
        summary[scope] = {'neural': selected_neural, 'median': selected_median,
                          'development_mae': dev, 'test_median': baseline_total,
                          'test_neural_ensemble': neural_total, 'test_neural_seeds': seed_metrics,
                          'windows': windows, 'lower_overall': lower_overall,
                          'no_test_window_worse': no_window_worse, 'all_seed_pooled_maes_lower': all_seeds_lower,
                          'wins': sum(row['difference'] < 0 for row in windows),
                          'losses': sum(row['difference'] > 0 for row in windows),
                          'stable': lower_overall and no_window_worse and all_seeds_lower}
    result = {'status': 'success', 'verified_utc': datetime.now(timezone.utc).isoformat(),
              'input_sha256': manifest['input_sha256'], 'config_sha256': manifest['config_sha256'],
              'prediction_arrays_checked': arrays_checked, 'ensemble_formulas_checked': ensemble_checks,
              'baseline_arrays_recomputed_from_sqlite': baselines_checked,
              'neural_fit_records_checked': len(fit_records), 'scopes': summary}
    destination = output / 'verification.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'success', 'prediction_arrays_checked': arrays_checked,
                      'baseline_arrays_recomputed_from_sqlite': baselines_checked,
                      'stable': {scope: row['stable'] for scope, row in summary.items()},
                      'output': str(destination)}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/smart-demand-20261023')
    args = parser.parse_args()
    verify(args.output)
