"""三个已有预测方法的共用入口：只接收历史数量和未来日期。

这里不读取文件、不写报告，也不接收未来真实销量。01、06、07都调用同一实现，
这样“修改未来销量不应改变当前预测”可以直接在实际预测入口上测试。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# 排序只用于成绩完全相同的平局；不会根据后来哪个模型赢修改这个顺序。
METHODS = ("recent_mean", "weekday_mean", "calendar_ridge")


def calendar_features(dates, origin):
    """预测前可知的日期差和星期，不包含未来销量。"""
    dates = pd.DatetimeIndex(dates)
    return np.column_stack([(dates - origin).days.to_numpy(), np.eye(7)[dates.dayofweek]])


def predict_methods(history, forecast_dates):
    """用历史序列生成三个方法的预测，不让模型接触未来答案。

    history是日期索引的Series，至少28个连续日；forecast_dates只提供未来日历。
    三方法仍是最近7日均值、最近28日同星期均值、日历Ridge，没有增加模型。
    """
    if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("历史输入应为日期索引的Series。")
    if len(history) < 28 or history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("需要至少28个按日期升序且不重复的历史日。")
    if not history.index.equals(pd.date_range(history.index.min(), history.index.max())):
        raise ValueError("历史日期不连续；先解释缺失日，不能偷偷填0。")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("历史数量需要是有限非负数。")
    dates = pd.DatetimeIndex(forecast_dates)
    if not len(dates) or dates.has_duplicates or (dates <= history.index[-1]).any():
        raise ValueError("预测日期必须严格晚于最后一个历史日。")
    recent = history.iloc[-28:]
    weekday_means = recent.groupby(recent.index.dayofweek).mean()
    predictions = {
        "recent_mean": np.repeat(history.iloc[-7:].mean(), len(dates)),
        "weekday_mean": np.array([weekday_means[d] for d in dates.dayofweek]),
    }
    model = Ridge(alpha=0.1)
    model.fit(calendar_features(history.index, history.index[0]), values)
    predictions["calendar_ridge"] = np.maximum(
        0, model.predict(calendar_features(dates, history.index[0])))
    return predictions


def forecast_at_origin(series, cutoff, horizon=14):
    """完整序列的便利入口，但只把cutoff之前的数量交给预测函数。

    cutoff=56表示已观察第0~55天。未来日期由第55天之后的日历生成，
    而不是从未来真实数量或未来数据行推导，因此可以修改未来数据来检验边界。
    """
    if cutoff < 28 or cutoff > len(series) or horizon < 1:
        raise ValueError("预测起点应有至少28天历史，且预测期必须为正整数。")
    history = series.iloc[:cutoff]
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return predict_methods(history, dates)


def completed_window_scores(history, window_cutoffs, horizon=14):
    """只评分已经完整结束的历史14日窗口；未完成窗口直接拒绝。

    返回逐窗口、逐方法的MAE。每次历史回测仍只用当时更早的训练数量。
    """
    rows = []
    for cutoff in window_cutoffs:
        if cutoff + horizon > len(history):
            raise ValueError("这个历史窗口尚未完整结束，不能用于当前选方法。")
        actual = history.iloc[cutoff:cutoff + horizon].to_numpy(dtype=float)
        predictions = forecast_at_origin(history, cutoff, horizon)
        for method in METHODS:
            rows.append({"window_cutoff": int(cutoff), "model": method,
                         "window_start": str(history.index[cutoff].date()),
                         "window_end": str(history.index[cutoff + horizon - 1].date()),
                         "mae": float(np.mean(np.abs(actual - predictions[method])))})
    return rows


def select_by_completed_windows(history, horizon=14, lookback_windows=2):
    """固定选择规则：最近两个已完成窗口平均MAE最小；平局按METHODS顺序。

    第一个历史预测起点固定为28，此后每14天一个窗口。新情景出现之后，
    至少等一个14日窗口完整发生，才能根据它的误差改变选择。
    """
    if lookback_windows < 1:
        raise ValueError("至少使用一个已完成窗口。")
    cutoffs = list(range(28, len(history) - horizon + 1, horizon))[-lookback_windows:]
    if not cutoffs:
        raise ValueError("还没有一个完整历史窗口可供选择。")
    rows = completed_window_scores(history, cutoffs, horizon)
    scores = {method: float(np.mean([row["mae"] for row in rows if row["model"] == method]))
              for method in METHODS}
    selected = min(METHODS, key=lambda method: scores[method])
    return selected, scores, rows
