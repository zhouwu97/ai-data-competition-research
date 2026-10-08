"""把记录量下降拆成活跃配送员日和每配送员日记录数；不把缺记录日补零。

直接运行：.venv/Scripts/python.exe research/lade-jilin/round2/volume_structure.py
公开结果只含聚合量，配送员集合在内存中用于核对两期对象。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPECTED_SHA = "12e2cf4664dd5b4475d39dddee8872f5a03b3082f08f0eece7f103baee6c6e73"
COMPARISONS = [
    ("primary_7d", "2022-09-24", "2022-09-30", "2022-10-01", "2022-10-07"),
    ("earlier_baseline_7d", "2022-09-17", "2022-09-23", "2022-10-01", "2022-10-07"),
    ("equal_6d", "2022-09-25", "2022-09-30", "2022-10-01", "2022-10-06"),
    ("shift_after_7d", "2022-09-24", "2022-09-30", "2022-10-02", "2022-10-08"),
]


def _write_csv(frame, output):
    frame.to_csv(output, index=False, encoding="utf-8", lineterminator="\n", float_format="%.8f")


def _period(frame, date_col, start, end):
    dates = pd.date_range(start, end)
    part = frame[frame[date_col].between(dates[0], dates[-1])]
    daily = part.groupby(date_col).agg(records=("courier_id", "size"),
                                      active_couriers=("courier_id", "nunique"))
    courier_days = int(daily.active_couriers.sum())
    observed = int(len(daily))
    complete = observed == len(dates)
    return part, {
        "start": start, "end": end, "requested_days": len(dates), "observed_days": observed,
        "missing_dates": dates.difference(daily.index).strftime("%Y-%m-%d").tolist(),
        "complete": complete, "records": len(part), "unique_couriers": int(part.courier_id.nunique()),
        "courier_days": courier_days,
        "records_per_day": len(part) / len(dates) if complete else None,
        "active_couriers_per_day": courier_days / len(dates) if complete else None,
        "records_per_active_courier_day": len(part) / courier_days if courier_days else None,
    }


def _decompose(before, after):
    # 例：每天10人各20条得到200条。人数和每人条数同时改变时，对称分解不偏向先算哪项。
    if not before["complete"] or not after["complete"]:
        return {key: None for key in ("delta_records_per_day", "active_contribution_per_day",
                                      "productivity_contribution_per_day", "active_share_of_delta",
                                      "productivity_share_of_delta", "change_pct")}
    a0, a1 = before["active_couriers_per_day"], after["active_couriers_per_day"]
    p0, p1 = before["records_per_active_courier_day"], after["records_per_active_courier_day"]
    delta = after["records_per_day"] - before["records_per_day"]
    active = (a1 - a0) * (p0 + p1) / 2
    productivity = (p1 - p0) * (a0 + a1) / 2
    if not np.isclose(active + productivity, delta, atol=1e-8):
        raise AssertionError("人数和每配送员日条数的分解不闭合")
    return {"delta_records_per_day": delta, "active_contribution_per_day": active,
            "productivity_contribution_per_day": productivity,
            "active_share_of_delta": active / delta if delta else None,
            "productivity_share_of_delta": productivity / delta if delta else None,
            "change_pct": 100 * delta / before["records_per_day"]}


def _aggregate_group(frame, date_col, days):
    count = len(frame)
    courier_days = len(frame[[date_col, "courier_id"]].drop_duplicates())
    return {"records": count, "unique_couriers": int(frame.courier_id.nunique()),
            "courier_days": courier_days, "records_per_day": count / days if days else None,
            "records_per_active_courier_day": count / courier_days if courier_days else None}


def analyze(raw: pd.DataFrame, output: Path) -> dict:
    """原字段及已解析的accept_date/delivery_date进入；只保存匿名区域和集合聚合。"""
    required = {"courier_id", "region_id", "accept_date", "delivery_date"}
    if raw.empty or not required.issubset(raw.columns) or raw[list(required)].isna().any().any():
        raise ValueError("记录量分析需要完整配送员、区域和两个日期字段")
    frame = raw.copy()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    regions = sorted(frame.region_id.unique(), key=lambda value: str(value))
    region_labels = {region: f"region_{i + 1}" for i, region in enumerate(regions)}
    frame["region_label"] = frame.region_id.map(region_labels)
    comparison_rows, cohort_rows, region_rows, overlap_rows, daily_rows, daily_region_rows = [], [], [], [], [], []
    summaries = {}
    for clock, date_col in [("delivery", "delivery_date"), ("accept", "accept_date")]:
        observed_daily = frame.groupby(date_col).agg(
            records=("courier_id", "size"), active_couriers=("courier_id", "nunique"),
            observed_regions=("region_id", "nunique"))
        dates = pd.date_range(observed_daily.index.min(), observed_daily.index.max())
        expanded = observed_daily.reindex(dates)
        expanded["records_per_active_courier"] = expanded.records / expanded.active_couriers
        # 城市整日无记录时所有量保留缺失；某区域在城市有记录日没出现则是文件内的0条。
        for date, row in expanded.iterrows():
            daily_rows.append({"clock": clock, "date": date.strftime("%Y-%m-%d"),
                               "observed": bool(date in observed_daily.index), **row.to_dict()})
        region_daily = frame.groupby([date_col, "region_label"]).agg(
            records=("courier_id", "size"), active_couriers=("courier_id", "nunique"))
        for date in dates:
            for label in region_labels.values():
                observed = date in observed_daily.index
                if not observed:
                    n, active = np.nan, np.nan
                elif (date, label) in region_daily.index:
                    n, active = region_daily.loc[(date, label), ["records", "active_couriers"]]
                else:
                    n, active = 0, 0
                daily_region_rows.append({"clock": clock, "date": date.strftime("%Y-%m-%d"),
                                          "region": label, "city_date_observed": observed,
                                          "records": n, "active_couriers": active})
        for name, b_start, b_end, a_start, a_end in COMPARISONS:
            before_part, before = _period(frame, date_col, b_start, b_end)
            after_part, after = _period(frame, date_col, a_start, a_end)
            decomposition = _decompose(before, after)
            row = {"clock": clock, "comparison": name,
                   **{f"before_{key}": value for key, value in before.items() if key != "missing_dates"},
                   **{f"after_{key}": value for key, value in after.items() if key != "missing_dates"},
                   "same_weekday_set": sorted(pd.date_range(b_start, b_end).weekday.tolist()) ==
                                       sorted(pd.date_range(a_start, a_end).weekday.tolist()),
                   **decomposition}
            comparison_rows.append(row)
            before_ids, after_ids = set(before_part.courier_id), set(after_part.courier_id)
            sets = {"shared": before_ids & after_ids, "before_only": before_ids - after_ids,
                    "after_only": after_ids - before_ids}
            cohort_deltas = []
            for cohort, ids in sets.items():
                b_group = _aggregate_group(before_part[before_part.courier_id.isin(ids)], date_col,
                                           before["requested_days"] if before["complete"] else None)
                a_group = _aggregate_group(after_part[after_part.courier_id.isin(ids)], date_col,
                                           after["requested_days"] if after["complete"] else None)
                delta = (a_group["records_per_day"] - b_group["records_per_day"]
                         if before["complete"] and after["complete"] else None)
                cohort_deltas.append(delta)
                cohort_rows.append({"clock": clock, "comparison": name, "cohort": cohort,
                                    **{f"before_{key}": value for key, value in b_group.items()},
                                    **{f"after_{key}": value for key, value in a_group.items()},
                                    "delta_records_per_day": delta})
            if before["complete"] and after["complete"] and not np.isclose(sum(cohort_deltas), decomposition["delta_records_per_day"]):
                raise AssertionError("配送员集合分解不闭合")
            b_courier_regions = before_part.groupby("courier_id").region_id.apply(set).to_dict()
            a_courier_regions = after_part.groupby("courier_id").region_id.apply(set).to_dict()
            overlap_rows.append({"clock": clock, "comparison": name,
                                 "shared_couriers": len(sets["shared"]),
                                 "before_only_couriers": len(sets["before_only"]),
                                 "after_only_couriers": len(sets["after_only"]),
                                 "shared_with_changed_region_set": sum(
                                     b_courier_regions[c] != a_courier_regions[c] for c in sets["shared"]),
                                 "before_multiple_region_couriers": sum(len(v) > 1 for v in b_courier_regions.values()),
                                 "after_multiple_region_couriers": sum(len(v) > 1 for v in a_courier_regions.values())})
            regional_deltas, comparison_region_rows = [], []
            for region, label in region_labels.items():
                b = _aggregate_group(before_part[before_part.region_id == region], date_col,
                                     before["requested_days"] if before["complete"] else None)
                a = _aggregate_group(after_part[after_part.region_id == region], date_col,
                                     after["requested_days"] if after["complete"] else None)
                delta = (a["records_per_day"] - b["records_per_day"]
                         if before["complete"] and after["complete"] else None)
                regional_deltas.append(delta)
                comparison_region_rows.append({"clock": clock, "comparison": name, "region": label,
                                    **{f"before_{key}": value for key, value in b.items()},
                                    **{f"after_{key}": value for key, value in a.items()},
                                    "before_share": b["records"] / before["records"] if before["records"] else None,
                                    "after_share": a["records"] / after["records"] if after["records"] else None,
                                    "delta_records_per_day": delta,
                                    "share_of_total_decline": delta / decomposition["delta_records_per_day"]
                                    if decomposition["delta_records_per_day"] else None})
            if before["complete"] and after["complete"] and not np.isclose(sum(regional_deltas), decomposition["delta_records_per_day"]):
                raise AssertionError("区域分解不闭合")
            # 区域权重用配送员日份额，才与全体每配送员日条数相乘；记录份额会重复使用结果本身。
            region_mix = {"applicable": False,
                          "reason": "区域无记录或同一配送员日在多个区域出现时，不作这一加权分解。"}
            can_mix = (before["complete"] and after["complete"] and
                       sum(r["before_courier_days"] for r in comparison_region_rows) == before["courier_days"] and
                       sum(r["after_courier_days"] for r in comparison_region_rows) == after["courier_days"] and
                       all(r["before_courier_days"] and r["after_courier_days"] for r in comparison_region_rows))
            if can_mix:
                for r in comparison_region_rows:
                    w0, w1 = (r["before_courier_days"] / before["courier_days"],
                              r["after_courier_days"] / after["courier_days"])
                    p0, p1 = (r["before_records_per_active_courier_day"], r["after_records_per_active_courier_day"])
                    r.update(before_courier_day_share=w0, after_courier_day_share=w1,
                             mix_contribution_to_productivity=(w1 - w0) * (p0 + p1) / 2,
                             within_region_contribution_to_productivity=(p1 - p0) * (w0 + w1) / 2)
                mix = sum(r["mix_contribution_to_productivity"] for r in comparison_region_rows)
                within = sum(r["within_region_contribution_to_productivity"] for r in comparison_region_rows)
                delta_p = after["records_per_active_courier_day"] - before["records_per_active_courier_day"]
                if not np.isclose(mix + within, delta_p):
                    raise AssertionError("区域权重与区域内条数分解不闭合")
                region_mix = {"applicable": True, "delta_records_per_active_courier_day": delta_p,
                              "region_weight_contribution": mix, "within_region_contribution": within,
                              "region_weight_share_of_delta": mix / delta_p if delta_p else None,
                              "within_region_share_of_delta": within / delta_p if delta_p else None}
            region_rows.extend(comparison_region_rows)
            if name == "primary_7d":
                summaries[clock] = {"before": before, "after": after, "decomposition": decomposition,
                                    "courier_groups": {cohort: len(ids) for cohort, ids in sets.items()},
                                    "region_productivity_decomposition": region_mix}

    # 只输出最后出现日期的分布，用于识别记录终止是否集中；它不是离职日期。
    last_seen_rows = []
    for clock, date_col in [("delivery", "delivery_date"), ("accept", "accept_date")]:
        last_dates = frame.groupby("courier_id")[date_col].max()
        relevant = set(frame.loc[frame[date_col].between("2022-09-24", "2022-09-30"), "courier_id"])
        for date, count in last_dates[last_dates.index.isin(relevant)].value_counts().sort_index().items():
            last_seen_rows.append({"clock": clock, "last_observed_date": date.strftime("%Y-%m-%d"),
                                   "couriers_active_in_september24_30": int(count)})
    _write_csv(pd.DataFrame(daily_rows), output / "volume_daily.csv")
    _write_csv(pd.DataFrame(daily_region_rows), output / "volume_daily_regions.csv")
    _write_csv(pd.DataFrame(comparison_rows), output / "volume_comparisons.csv")
    _write_csv(pd.DataFrame(cohort_rows), output / "volume_courier_cohorts.csv")
    _write_csv(pd.DataFrame(region_rows), output / "volume_region_comparisons.csv")
    _write_csv(pd.DataFrame(overlap_rows), output / "volume_courier_region_overlap.csv")
    _write_csv(pd.DataFrame(last_seen_rows), output / "volume_last_seen.csv")
    summary = {"rows": len(frame), "unique_couriers": int(frame.courier_id.nunique()),
               "anonymous_region_count": len(regions),
               "couriers_observed_in_multiple_regions_over_full_input": int(
                   (frame.groupby("courier_id").region_id.nunique() > 1).sum()),
               "cross_day_records": int((frame.accept_date != frame.delivery_date).sum()),
               "primary_comparison": summaries,
               "region_labels": "匿名序号按输入region_id字符串排序；不是实际行政区名称。",
               "missing_date_rule": "城市整日无记录保留缺失，不解释为0条；比较期缺日时不计算平均每日分解。",
               "interpretation": ["活跃配送员指该日出现在这份文件中的不同courier_id，不代表完整在岗人数。",
                                  "每配送员日条数=总记录数/总活跃配送员日；一个配送员两天出现计2个配送员日。",
                                  "共同配送员、前期独有、后期独有均由比较期是否有记录决定，不推断入职离职。",
                                  "对称分解是记录恒等式的数量分解，不能从它确定需求、作业或采样变化的原因。",
                                  "两个时钟反映同一份已完成记录的日期归属，接单统计不是完整新增需求或未完成对象统计。"]}
    (output / "volume_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE.parent / "data/delivery_jl.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError("本次二轮分析只复现已记录的固定吉林数据版本")
    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    for column in ["accept_time", "delivery_time"]:
        raw[column] = pd.to_datetime("2022-" + raw[column], format="%Y-%m-%d %H:%M:%S")
    raw["minutes"] = (raw.delivery_time - raw.accept_time).dt.total_seconds() / 60
    raw["accept_date"] = raw.accept_time.dt.normalize()
    raw["delivery_date"] = raw.delivery_time.dt.normalize()
    summary = analyze(raw, args.output_dir)
    for clock, result in summary["primary_comparison"].items():
        print(clock, json.dumps(result["decomposition"], ensure_ascii=False))


if __name__ == "__main__":
    main()
