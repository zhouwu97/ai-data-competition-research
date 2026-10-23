"""题一成交记录：保留缺商品日，并只从每个预测起点之前构造输入。"""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'outputs/changfeng-deep-search/competition-code/ai-journey-changfeng/source/长风杯项目/2025年长风杯竞赛数据/智能分仓数据集.sqlite'
BEHAVIOR = ['pv_uv', 'cart_uv', 'collect_uv', 'qty_gmv', 'num_alipay']


def load_panel(path=DEFAULT_INPUT):
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as con:
        frame = pd.read_sql_query('SELECT date,item_id,cate_id,qty_alipay,' + ','.join(BEHAVIOR) + ' FROM Item_feature', con)
    frame['date'] = pd.to_datetime(frame['date'])
    if frame.duplicated(['date', 'item_id']).any():
        raise ValueError('商品日不是唯一，不能静默汇总。')
    days = pd.date_range(frame.date.min(), frame.date.max(), freq='D')
    items = np.sort(frame.item_id.unique())
    values = np.full((len(items), len(days), 1 + len(BEHAVIOR)), np.nan, dtype=np.float32)
    ii, dd = pd.Index(items).get_indexer(frame.item_id), days.get_indexer(frame.date)
    values[ii, dd] = frame[['qty_alipay'] + BEHAVIOR].to_numpy(dtype=np.float32)
    if not np.isfinite(frame[['qty_alipay'] + BEHAVIOR].to_numpy()).all() or (values[np.isfinite(values)] < 0).any():
        raise ValueError('本实验要求已记录值非负且有限。')
    if frame.groupby('item_id').cate_id.nunique().max() != 1:
        raise ValueError('类别随时间变化时必须改用各起点当时的类别。')
    category = frame.groupby('item_id').cate_id.first().reindex(items).astype(str).to_numpy()
    total = np.nansum(values, axis=0, dtype=np.float64).astype(np.float32)[None, :, :]
    daily_count = np.isfinite(values[:, :, 0]).sum(axis=0)
    if (daily_count == 0).any():
        raise ValueError('没有任何记录的全国日期不能用nansum解释成零成交。')
    return {'days': days, 'items': items, 'values': values, 'category': category,
            'total': total, 'daily_count': daily_count}


def calendar_features(days):
    """未来星期和固定日期已知；这里没有使用未来浏览、加购或成交。"""
    angle = np.asarray(days.dayofweek) * 2 * np.pi / 7
    annual = np.asarray(days.dayofyear) * 2 * np.pi / 365.25
    return np.stack([np.sin(angle), np.cos(angle), np.sin(annual), np.cos(annual),
                     ((days.month == 11) & (days.day == 11)).astype(float),
                     ((days.month == 12) & (days.day == 12)).astype(float)], axis=1).astype(np.float32).ravel()


def _mean(values):
    mask = np.isfinite(values)
    return np.divide(np.nansum(values, axis=1), mask.sum(axis=1),
                     out=np.zeros(values.shape[0], dtype=float), where=mask.sum(axis=1) > 0)


def _median(values):
    result = np.zeros(values.shape[0], dtype=float)
    for i, row in enumerate(values):
        finite = row[np.isfinite(row)]
        result[i] = np.median(finite) if len(finite) else 0.
    return result


def history_features(values, origin, days, *, behavior=False, category=None, levels=None, counts=None):
    """origin 是最后已知日期的位置。缺测只在输入中放占位零，另给观测掩码。"""
    if origin < 27 or origin + 14 >= values.shape[1]:
        raise ValueError('至少需要28天历史以及14天可评价日期。')
    h = np.full((len(values), 56), np.nan, dtype=np.float32)
    recent = values[:, max(0, origin-55):origin+1, 0]
    h[:, -recent.shape[1]:] = recent
    base = np.maximum(_mean(h[:, -28:]), 1.).astype(np.float32)
    observed = np.isfinite(h)
    x = [np.log1p(np.nan_to_num(h[:, -28:], nan=0.) / base[:, None]), observed[:, -28:].astype(float),
         np.log1p(base)[:, None], np.full((len(values), 1), min(origin+1, 56)/56)]
    for width in (7, 14, 28, 56):
        part = h[:, -width:]
        known = np.isfinite(part)
        maximum = np.max(np.where(known, part, 0.), axis=1)
        x += [np.stack([np.log1p(_mean(part) / base), np.log1p(_median(part) / base),
                        np.log1p(maximum / base), known.mean(axis=1),
                        ((part == 0) & known).sum(axis=1) / np.maximum(known.sum(axis=1), 1)], axis=1)]
    # 两类日历在所有对照里相同，行为删减比较只改变历史行为输入。
    future = calendar_features(days[origin+1:origin+15])
    x.append(np.broadcast_to(future, (len(values), len(future))))
    if category is not None:
        x.append(np.stack([(category == level).astype(float) for level in levels], axis=1))
    if counts is not None:
        recent = counts[origin-27:origin+1]
        x.append(np.broadcast_to(np.log1p(recent), (len(values), 28)))
    if behavior:
        for j in range(1, values.shape[2]):
            for width in (7, 28):
                part = values[:, max(0, origin-width+1):origin+1, j]
                x.append(np.log1p(_mean(part) / base)[:, None])
    return np.concatenate(x, axis=1).astype(np.float32), base


def make_fold(panel, origin, *, scope, behavior=False, stride=7):
    """训练目标最晚结束于origin；测试资格只由历史决定，未知目标保持NaN。"""
    values = panel['values'] if scope == 'sku' else panel['total']
    days = panel['days']
    category = panel['category'] if scope == 'sku' else None
    if category is not None:
        seen = np.isfinite(values[:, :origin+1, 0]).any(axis=1)
        levels = sorted(set(category[seen]))
    else:
        levels = None
    counts = panel['daily_count'] if scope == 'national' else None
    xs, ys, bs, origins, item_indices = [], [], [], [], []
    # 不用未来是否存在14条记录筛选商品；仅排除完全没有监督信号的训练行。
    for o in range(27, origin - 13, stride):
        x, b = history_features(values, o, days, behavior=behavior, category=category, levels=levels, counts=counts)
        y = values[:, o+1:o+15, 0]
        eligible = (np.isfinite(values[:, max(0, o-55):o+1, 0]).sum(axis=1) >= 28) & np.isfinite(y).any(axis=1)
        xs.append(x[eligible]); ys.append(y[eligible]); bs.append(b[eligible])
        origins.extend([o] * int(eligible.sum())); item_indices.extend(np.flatnonzero(eligible))
    current_x, current_base = history_features(values, origin, days, behavior=behavior, category=category, levels=levels, counts=counts)
    eligible = np.isfinite(values[:, max(0, origin-55):origin+1, 0]).sum(axis=1) >= 28
    idx = np.flatnonzero(eligible)
    return {'x': np.concatenate(xs), 'y': np.concatenate(ys), 'base': np.concatenate(bs),
            'train_origins': np.asarray(origins), 'train_items': np.asarray(item_indices),
            'current_x': current_x[eligible], 'current_base': current_base[eligible],
            'truth': values[eligible, origin+1:origin+15, 0], 'indices': idx,
            'levels': levels, 'latest_label': int(max(origins) + 14)}


def baselines(panel, origin, *, scope, indices=None):
    values = panel['values'][:, :, 0] if scope == 'sku' else panel['total'][:, :, 0]
    if indices is not None:
        values = values[indices]
    result = {}
    for w in (14, 28, 56):
        result[f'median{w}'] = np.repeat(_median(values[:, max(0, origin-w+1):origin+1])[:, None], 14, axis=1)
    result['mean28'] = np.repeat(_mean(values[:, max(0, origin-27):origin+1])[:, None], 14, axis=1)
    weekday = []
    for step in range(1, 15):
        positions = [t for t in range(max(0, origin-55), origin+1) if (origin+step-t) % 7 == 0]
        weekday.append(_mean(values[:, positions]))
    result['weekday8'] = np.stack(weekday, axis=1)
    return result
