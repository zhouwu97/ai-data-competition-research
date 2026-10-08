"""配对重算已有时长预测，并寻找可以继续核验的时间与任务组成线索。

这里不训练新模型。MAE下降的来源先用相同运单的两次误差相减解释，
再核对时钟、同日完成与配送员组成；这些关联不等于作业机制的因果证据。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = HERE.parent / "results/duration_predictions.csv"


def _write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6f")


def _stats(group, total_gain):
    tail = group.actual_minutes > 360
    return {
        "test_records": int(len(group)),
        "baseline_mae_minutes": float(group.baseline_error.mean()),
        "hour_mae_minutes": float(group.hour_error.mean()),
        "mean_improvement_minutes": float(group.improvement.mean()),
        "total_improvement_minutes": float(group.improvement.sum()),
        "share_of_total_improvement": float(group.improvement.sum() / total_gain) if total_gain else None,
        "actual_median_minutes": float(group.actual_minutes.median()),
        "mean_prediction_shift_minutes": float(group.prediction_shift.mean()),
        "tail_records": int(tail.sum()),
        "tail_share": float(tail.mean()),
        "same_day_share": float((group.completion_day_offset == 0).mean()),
        "distinct_couriers": int(group.courier_id.nunique()),
        "distinct_regions": int(group.region_id.nunique()),
    }


def _paired_predictions(raw, predictions_path):
    predictions = pd.read_csv(predictions_path, encoding="utf-8")
    keys = ["window_start", "sample_index"]
    if set(predictions.method) != {"global_median", "hour_median"}:
        raise ValueError("配对分析只接受当前两种中位数方法。")
    baseline = predictions[predictions.method == "global_median"].set_index(keys).sort_index()
    hourly = predictions[predictions.method == "hour_median"].set_index(keys).sort_index()
    if not baseline.index.is_unique or not hourly.index.is_unique or not baseline.index.equals(hourly.index):
        raise ValueError("两种方法不能按窗口与样本序号一对一对齐。")
    for column in ["actual_minutes", "accept_date", "accept_hour"]:
        if not baseline[column].equals(hourly[column]):
            raise ValueError(f"配对对象的{column}不同。")
    summary_path = predictions_path.parent / "summary.json"
    horizon = 14
    if summary_path.exists():
        horizon = json.loads(summary_path.read_text(encoding="utf-8"))["parameters"]["horizon_days"]
    paired = baseline.rename(columns={"prediction_minutes": "baseline_prediction"}).copy()
    paired["hour_prediction"] = hourly.prediction_minutes
    cohorts = []
    # 原结果的样本序号来自输入行顺序，不能排序原表后再回连。
    for window, group in paired.groupby(level="window_start", sort=True):
        origin = pd.Timestamp(window)
        cohort = raw[(raw.accept_time >= origin) &
                     (raw.accept_time < origin + pd.Timedelta(days=horizon))].copy()
        if len(cohort) != len(group) or group.index.get_level_values("sample_index").tolist() != list(range(len(cohort))):
            raise ValueError(f"{window}窗口的原始对象数或行顺序不能回连。")
        if (not np.allclose(cohort.minutes.to_numpy(), group.actual_minutes.to_numpy(), atol=1e-6, rtol=0)
                or cohort.accept_time.dt.strftime("%Y-%m-%d").tolist() != group.accept_date.tolist()
                or cohort.accept_time.dt.hour.tolist() != group.accept_hour.tolist()):
            raise ValueError(f"{window}窗口的日期、小时或实际时长与原始数据不同。")
        cohort.index = group.index
        cohorts.append(cohort)
    source = pd.concat(cohorts).sort_index()
    for column in ["accept_time", "delivery_time", "courier_id", "region_id"]:
        paired[column] = source[column]
    paired["completion_day_offset"] = (source.delivery_time.dt.normalize() -
                                        source.accept_time.dt.normalize()).dt.days
    paired["baseline_error"] = (paired.actual_minutes - paired.baseline_prediction).abs()
    paired["hour_error"] = (paired.actual_minutes - paired.hour_prediction).abs()
    paired["improvement"] = paired.baseline_error - paired.hour_error
    paired["prediction_shift"] = paired.hour_prediction - paired.baseline_prediction
    paired["duration_band"] = np.where(paired.actual_minutes > 360, "over_360", "up_to_360")
    return paired.reset_index(), len(predictions)


def _composition_contrasts(paired):
    rows = []
    # 对同一配送员、同一区域，乃至同一配送员同一天分别比较，检查组成变化能否独自解释差异。
    # 每组都需要两时段有足够对象；覆盖低时，不能把局部结果外推到整个小时。
    for hour in [9, 10, 13, 14, 15]:
        target = paired[paired.accept_hour == hour]
        reference = paired[paired.accept_hour == 8]
        for by, columns, minimum in [("courier", ["courier_id"], 5),
                                     ("region", ["region_id"], 5),
                                     ("courier_day", ["courier_id", "accept_date"], 3)]:
            subset = paired[paired.accept_hour.isin([8, hour])]
            grouped = subset.groupby(columns + ["accept_hour"]).actual_minutes.agg(["size", "median"])
            table = grouped.unstack("accept_hour")
            supported = table[(table["size"][8] >= minimum) & (table["size"][hour] >= minimum)]
            differences = supported["median"][hour] - supported["median"][8]
            weights = supported["size"][hour]
            rows.append({"grouping": by, "reference_hour": 8, "target_hour": hour,
                         "minimum_records_per_hour_in_group": minimum,
                         "supported_groups": int(len(supported)),
                         "target_records_covered": int(weights.sum()),
                         "target_records": int(len(target)),
                         "target_coverage_share": float(weights.sum() / len(target)),
                         "pooled_median_difference_minutes": float(target.actual_minutes.median() -
                                                                     reference.actual_minutes.median()),
                         "weighted_within_group_median_difference_minutes":
                             float(np.average(differences, weights=weights)) if len(supported) else None,
                         "groups_with_shorter_target_median": int((differences < 0).sum()),
                         "groups_with_equal_target_median": int((differences == 0).sum()),
                         "groups_with_longer_target_median": int((differences > 0).sum())})
    return pd.DataFrame(rows)


def analyze(raw: pd.DataFrame, output: Path, predictions_path: Path | None = None) -> dict:
    """raw保留原输入行顺序，包含完整时间、minutes及配送员/区域字段；公开结果只保存聚合。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(predictions_path or DEFAULT_PREDICTIONS)
    paired, prediction_rows = _paired_predictions(raw, predictions_path)
    total_gain = float(paired.improvement.sum())
    group_rows = [{"by": "overall", "group": "all", "duration_band": "all", **_stats(paired, total_gain)}]
    for band, group in paired.groupby("duration_band", sort=True):
        group_rows.append({"by": "overall", "group": "all", "duration_band": band, **_stats(group, total_gain)})
    for by in ["window_start", "accept_hour", "accept_date"]:
        for label, group in paired.groupby(by, sort=True):
            group_rows.append({"by": by, "group": str(label), "duration_band": "all", **_stats(group, total_gain)})
            for band, part in group.groupby("duration_band", sort=True):
                group_rows.append({"by": by, "group": str(label), "duration_band": band, **_stats(part, total_gain)})
    for label, low, high in [("7_to_9", 7, 9), ("13_to_15", 13, 15)]:
        group = paired[paired.accept_hour.between(low, high)]
        group_rows.append({"by": "accept_period", "group": label, "duration_band": "all", **_stats(group, total_gain)})
        for band, part in group.groupby("duration_band", sort=True):
            group_rows.append({"by": "accept_period", "group": label, "duration_band": band, **_stats(part, total_gain)})
    groups = pd.DataFrame(group_rows)
    _write_csv(groups, output / "duration_groups.csv")

    geometry_rows = []
    for (window, hour), group in paired.groupby(["window_start", "accept_hour"], sort=True):
        if group.baseline_prediction.nunique() != 1 or group.hour_prediction.nunique() != 1:
            raise ValueError("当前方法每个窗口/小时应有一个固定预测值。")
        baseline, hourly = float(group.baseline_prediction.iloc[0]), float(group.hour_prediction.iloc[0])
        low, high = min(baseline, hourly), max(baseline, hourly)
        shift = hourly - baseline
        below = group.actual_minutes <= low
        above = (group.actual_minutes >= high) & ~below
        between = ~below & ~above
        # 两预测值外侧，每条对象的变化只由预测位移决定。中间则取决于离两端的距离。
        geometric_gain = np.where(below, -shift, np.where(above, shift,
                                      np.sign(shift) * (2 * group.actual_minutes - baseline - hourly)))
        if not np.allclose(group.improvement, geometric_gain, atol=1e-9, rtol=0):
            raise ValueError("绝对损失的几何恒等式不成立。")
        tail = group.actual_minutes > 360
        geometry_rows.append({"window_start": window, "accept_hour": int(hour), "test_records": len(group),
                              "baseline_prediction_minutes": baseline, "hour_prediction_minutes": hourly,
                              "prediction_shift_minutes": shift, "at_or_below_lower_prediction": int(below.sum()),
                              "between_predictions": int(between.sum()), "at_or_above_upper_prediction": int(above.sum()),
                              "gain_below_minutes": float(group.loc[below, "improvement"].sum()),
                              "gain_between_minutes": float(group.loc[between, "improvement"].sum()),
                              "gain_above_minutes": float(group.loc[above, "improvement"].sum()),
                              "tail_records": int(tail.sum()),
                              "tail_gain_minutes": float(group.loc[tail, "improvement"].sum()),
                              "tail_gain_from_shift_minutes": float(group.loc[tail, "prediction_shift"].sum())})
    geometry = pd.DataFrame(geometry_rows)
    _write_csv(geometry, output / "duration_loss_geometry.csv")

    paired["delivery_hour"] = paired.delivery_time.dt.hour
    clock = paired.groupby(["accept_hour", "duration_band", "completion_day_offset", "delivery_hour"]).size()
    clock = clock.rename("test_records").reset_index()
    clock["share_within_accept_hour_and_band"] = clock.test_records / clock.groupby(
        ["accept_hour", "duration_band"]).test_records.transform("sum")
    _write_csv(clock, output / "duration_completion_clock.csv")
    profiles = []
    for label, group in paired.groupby("accept_hour", sort=True):
        courier_sizes, region_sizes = group.courier_id.value_counts(), group.region_id.value_counts()
        profiles.append({"accept_hour": int(label), "test_records": len(group),
                         "accept_clock_median_minutes": float((group.accept_time.dt.hour * 60 +
                                                                 group.accept_time.dt.minute).median()),
                         "delivery_clock_median_minutes": float((group.delivery_time.dt.hour * 60 +
                                                                   group.delivery_time.dt.minute).median()),
                         "duration_median_minutes": float(group.actual_minutes.median()),
                         "duration_p90_minutes": float(group.actual_minutes.quantile(.9)),
                         "same_day_records": int((group.completion_day_offset == 0).sum()),
                         "cross_day_records": int((group.completion_day_offset > 0).sum()),
                         "distinct_couriers": int(len(courier_sizes)), "distinct_regions": int(len(region_sizes)),
                         "largest_courier_share": float(courier_sizes.iloc[0] / len(group)),
                         "courier_concentration_sum_squared_shares": float(((courier_sizes / len(group)) ** 2).sum()),
                         "largest_region_share": float(region_sizes.iloc[0] / len(group))})
    _write_csv(pd.DataFrame(profiles), output / "duration_hour_profiles.csv")
    contrasts = _composition_contrasts(paired)
    _write_csv(contrasts, output / "duration_composition_contrasts.csv")

    tails = paired[paired.actual_minutes > 360]
    daily_gain = paired.groupby("accept_date").improvement.mean()
    window_gain = paired.groupby("window_start").improvement.mean()
    examples = []
    for window, hour in [("2022-08-14", 8), ("2022-08-14", 13)]:
        group = paired[(paired.window_start == window) & (paired.accept_hour == hour)]
        if len(group):
            examples.append({"window_start": window, "accept_hour": hour,
                             "baseline_prediction_minutes": float(group.baseline_prediction.iloc[0]),
                             "hour_prediction_minutes": float(group.hour_prediction.iloc[0]),
                             "bands": {band: _stats(part, total_gain)
                                       for band, part in group.groupby("duration_band", sort=True)}})
    summary = {
        "prediction_rows": prediction_rows, "paired_records": len(paired),
        "pairing_keys": ["window_start", "sample_index"],
        "verified_pair_fields": ["actual_minutes", "accept_date", "accept_hour"],
        "raw_row_order_backlink_verified": True,
        "overall": _stats(paired, total_gain), "over_360": _stats(tails, total_gain),
        "up_to_360": _stats(paired[paired.actual_minutes <= 360], total_gain),
        "tail_accept_7_to_9_records": int(tails.accept_hour.between(7, 9).sum()),
        "tail_accept_7_to_9_share": float(tails.accept_hour.between(7, 9).mean()),
        "same_day_records": int((paired.completion_day_offset == 0).sum()),
        "cross_day_records": int((paired.completion_day_offset > 0).sum()),
        "scored_accept_dates": int(len(daily_gain)),
        "dates_with_negative_improvement": daily_gain[daily_gain < 0].index.tolist(),
        "negative_date_mean_improvement_minutes": {str(k): float(v) for k, v in daily_gain[daily_gain < 0].items()},
        "window_mean_improvement_minutes": {str(k): float(v) for k, v in window_gain.items()},
        "maximum_prediction_minutes": float(paired[["baseline_prediction", "hour_prediction"]].max().max()),
        "all_tail_gain_equals_prediction_shift": bool(np.allclose(tails.improvement, tails.prediction_shift)),
        "loss_geometry_verified": True,
        "concrete_examples": examples,
        "interpretation": "长尾均高于两预测值，其误差变化等于预测位移，不能据此声称模型识别了长尾机制。",
        "composition_contrast_definition": "相对8时；对两小时均满足最小样本数的共同组，以目标小时对象数加权组内中位数差；不是 pooled 中位数差的精确分解。",
        "limits": ["长尾按实际完成时长事后划分，预测时未知。",
                   "四窗口已经用于首轮评价，逐日逐窗口结果是内部一致性核对，不是新数据外部验证。",
                   "同配送员/区域或同配送员同日对照仍未控制任务类型、接单批次、路线、工作安排和输入采样。",
                   "本模块仅回连当前四窗口；10月9日以后的稀疏记录不构成后续验证。"],
    }
    (output / "duration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE.parent / "data/delivery_jl.csv")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    for column in ["accept_time", "delivery_time"]:
        raw[column] = pd.to_datetime("2022-" + raw[column], format="%Y-%m-%d %H:%M:%S", errors="raise")
    raw["minutes"] = (raw.delivery_time - raw.accept_time).dt.total_seconds() / 60
    raw["accept_date"] = raw.accept_time.dt.normalize()
    raw["delivery_date"] = raw.delivery_time.dt.normalize()
    summary = analyze(raw, args.output_dir, args.predictions)
    print(json.dumps({key: summary[key] for key in ["paired_records", "overall", "over_360", "up_to_360"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
