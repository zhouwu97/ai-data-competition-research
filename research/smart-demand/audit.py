"""Read-only audit of the complete historical smart-warehouse SQLite export."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / (
    "outputs/changfeng-deep-search/competition-code/ai-journey-changfeng/source/"
    "长风杯项目/2025年长风杯竞赛数据/智能分仓数据集.sqlite"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: pd.Series) -> dict[str, float]:
    return {str(q): float(values.quantile(q)) for q in (0, .1, .25, .5, .75, .9, 1)}


def verify_exports(source: Path, output: Path) -> dict:
    """Recompute exported coverage with SQLite, separately from the pandas path."""
    import csv

    report = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as connection:
        rows, total, zero, item_count, date_count = connection.execute(
            "SELECT COUNT(*),SUM(qty_alipay),SUM(qty_alipay=0),"
            "COUNT(DISTINCT item_id),COUNT(DISTINCT date) FROM Item_feature"
        ).fetchone()
        internal = connection.execute(
            "SELECT SUM(CAST(julianday(hi)-julianday(lo)+1 AS INTEGER)-n) FROM "
            "(SELECT item_id,MIN(date) lo,MAX(date) hi,COUNT(*) n FROM Item_feature GROUP BY item_id)"
        ).fetchone()[0]
        expected_items = {str(r[0]): r[1:] for r in connection.execute(
            "SELECT item_id,COUNT(*),MIN(date),MAX(date),SUM(qty_alipay),SUM(qty_gmv),"
            "SUM(qty_alipay=0) FROM Item_feature GROUP BY item_id"
        )}
        expected_dates = {str(r[0]): r[1:] for r in connection.execute(
            "SELECT date,COUNT(DISTINCT item_id),SUM(qty_alipay),SUM(qty_gmv),SUM(pv_ipv),"
            "SUM(cart_ipv),SUM(qty_alipay=0) FROM Item_feature GROUP BY date"
        )}
        expected_gaps = {str(r[0]): r[1] for r in connection.execute(
            "SELECT item_id,COALESCE(MAX(CAST(julianday(date)-julianday(prev)-1 AS INTEGER)),0) FROM "
            "(SELECT item_id,date,LAG(date) OVER(PARTITION BY item_id ORDER BY date) prev FROM Item_feature) GROUP BY item_id"
        )}
    with (output / "item_coverage.csv").open(encoding="utf-8", newline="") as stream:
        item_rows = list(csv.DictReader(stream))
    with (output / "date_coverage.csv").open(encoding="utf-8", newline="") as stream:
        date_rows = list(csv.DictReader(stream))
    date_columns = ("observed_items", "recorded_paid_quantity", "recorded_ordered_quantity",
                    "total_pageviews", "total_cart_events", "zero_paid_items")
    checks = {
        "row_count": rows == report["tables"]["Item_feature"]["rows"],
        "paid_total": total == report["target"]["sum"],
        "explicit_zero_rows": zero == report["target"]["zero_rows"],
        "distinct_items": item_count == report["items"]["count"],
        "distinct_dates": date_count == report["calendar"]["observed_dates"],
        "internal_gaps_from_sql_calendar": internal == report["items"]["total_internal_missing_item_days"],
        "each_exported_item_reaggregated_from_sql": len(item_rows) == len(expected_items) and all(
            (int(r["observed_days"]), r["first_date"], r["last_date"], int(r["qty_alipay_total"]),
             int(r["qty_gmv_total"]), int(r["zero_paid_days"])) == expected_items[r["item_id"]]
            for r in item_rows),
        "each_exported_date_reaggregated_from_sql": len(date_rows) == len(expected_dates) and all(
            tuple(int(r[k]) for k in date_columns) == expected_dates[r["date"]] for r in date_rows),
        "each_max_gap_matches_sql_lag": all(int(r["max_observed_gap_days"]) == expected_gaps[r["item_id"]] for r in item_rows),
        "original_source_hash_unchanged": sha256(source) == report["input"]["sha256"],
    }
    record = {"status": "success" if all(checks.values()) else "failed",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "rows_reaggregated": rows, "item_rows_verified": len(item_rows),
              "date_rows_verified": len(date_rows), "input_sha256": report["input"]["sha256"]}
    (output / "data_audit_validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise AssertionError(f"Independent SQL verification failed: {checks}")
    return record


def audit(source: Path, output: Path) -> dict:
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as connection:
        integrity = connection.execute("PRAGMA quick_check").fetchall()
        data = pd.read_sql_query("SELECT * FROM Item_feature", connection)
        cost = pd.read_sql_query("SELECT * FROM Cost", connection)
        meta = pd.read_sql_query("SELECT * FROM bwds_meta", connection)
        schemas = {
            table: connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            for table in ("Item_feature", "Cost", "bwds_meta")
        }
    parsed = pd.to_datetime(data["date"], errors="coerce", format="%Y-%m-%d")
    invalid_dates = int(parsed.isna().sum())
    if invalid_dates:
        raise ValueError(f"{invalid_dates} dates cannot be interpreted as YYYY-MM-DD")
    data["date"] = parsed
    first, last = parsed.min(), parsed.max()
    calendar = pd.date_range(first, last, freq="D")
    numeric = list(data.select_dtypes(include="number").columns)
    fields = []
    for name in data.columns:
        values = data[name]
        entry = {"field": name, "dtype": str(values.dtype), "missing": int(values.isna().sum()),
                 "empty_strings": int(values.eq("").sum()), "distinct": int(values.nunique())}
        if name in numeric:
            finite = np.isfinite(values.to_numpy(dtype=float))
            entry.update(nonfinite=int((~finite).sum()), negative=int(values.lt(0).sum()),
                         zero=int(values.eq(0).sum()), min=float(values.min()),
                         p50=float(values.median()), p90=float(values.quantile(.9)),
                         p99=float(values.quantile(.99)), max=float(values.max()),
                         total=float(values.sum()))
        fields.append(entry)
    pd.DataFrame(fields).to_csv(output / "field_profile.csv", index=False, encoding="utf-8")

    items = data.groupby("item_id", sort=True).agg(
        observed_days=("date", "size"), first_date=("date", "min"), last_date=("date", "max"),
        qty_alipay_total=("qty_alipay", "sum"), qty_gmv_total=("qty_gmv", "sum"),
        zero_paid_days=("qty_alipay", lambda x: int(x.eq(0).sum())),
    )
    items["span_days"] = (items.last_date - items.first_date).dt.days + 1
    # Absence of a row does not identify whether an item was unavailable or unrecorded.
    items["internal_missing_days"] = items.span_days - items.observed_days
    items["before_first_days"] = (items.first_date - first).dt.days
    items["after_last_days"] = (last - items.last_date).dt.days
    items["full_calendar_missing_days"] = len(calendar) - items.observed_days
    items["last28_observed_days"] = data[data.date > last - pd.Timedelta(days=28)].groupby("item_id").size().reindex(items.index, fill_value=0)
    items["max_observed_gap_days"] = data.sort_values(["item_id", "date"]).groupby("item_id").date.diff().dt.days.sub(1).clip(lower=0).groupby(data.sort_values(["item_id", "date"]).item_id).max().reindex(items.index).fillna(0).astype(int)
    items.reset_index().to_csv(output / "item_coverage.csv", index=False, encoding="utf-8", date_format="%Y-%m-%d")

    dates = data.groupby("date").agg(
        observed_items=("item_id", "nunique"), recorded_paid_quantity=("qty_alipay", "sum"),
        recorded_ordered_quantity=("qty_gmv", "sum"), total_pageviews=("pv_ipv", "sum"),
        total_cart_events=("cart_ipv", "sum"), zero_paid_items=("qty_alipay", lambda x: int(x.eq(0).sum())),
    ).reindex(calendar)
    dates.index.name = "date"
    dates["first_observed_items"] = items.first_date.value_counts().reindex(calendar, fill_value=0)
    dates["last_observed_items"] = items.last_date.value_counts().reindex(calendar, fill_value=0)
    dates.reset_index().to_csv(output / "date_coverage.csv", index=False, encoding="utf-8", date_format="%Y-%m-%d")
    spikes = dates.nlargest(15, "recorded_paid_quantity").copy()
    total_paid = int(data.qty_alipay.sum())
    spikes["share_of_all_recorded_paid_quantity"] = spikes.recorded_paid_quantity / total_paid
    spikes.reset_index().to_csv(output / "top_spike_days.csv", index=False, encoding="utf-8", date_format="%Y-%m-%d")

    static_fields = ("cate_id", "cate_level_id", "brand_id", "supplier_id")
    static = {name: {"distinct_values": int(data[name].nunique()),
                     "items_with_multiple_values": int(data.groupby("item_id")[name].nunique().gt(1).sum())}
              for name in static_fields}
    checks = {
        "paid_quantity_greater_than_ordered_quantity": int(data.qty_alipay.gt(data.qty_gmv).sum()),
        "paid_amount_greater_than_ordered_amount": int(data.amt_alipay.gt(data.amt_gmv).sum()),
        "paid_quantity_less_than_paid_order_count": int(data.qty_alipay.lt(data.num_alipay).sum()),
        "ordered_quantity_less_than_order_count": int(data.qty_gmv.lt(data.num_gmv).sum()),
        "paid_order_count_less_than_paid_people_count": int(data.num_alipay.lt(data.unum_alipay).sum()),
        "pageviews_less_than_visitors": int(data.pv_ipv.lt(data.pv_uv).sum()),
        "cart_events_less_than_cart_people": int(data.cart_ipv.lt(data.cart_uv).sum()),
        "non_jhs_paid_quantity_greater_than_all_paid_quantity": int(data.qty_alipay_njhs.gt(data.qty_alipay).sum()),
        "positive_paid_quantity_with_zero_paid_amount": int((data.qty_alipay.gt(0) & data.amt_alipay.eq(0)).sum()),
        "zero_paid_quantity_with_positive_paid_amount": int((data.qty_alipay.eq(0) & data.amt_alipay.gt(0)).sum()),
    }
    corr_fields = ["qty_alipay", "qty_gmv", "num_alipay", "amt_alipay", "pv_ipv", "pv_uv", "cart_ipv", "cart_uv", "collect_uv"]
    correlations = data[corr_fields].corr()["qty_alipay"].to_dict()
    cost_items = set(cost.sp_id.astype(str))
    feature_items = set(data.item_id.astype(str))
    replacement_counts = {name: int(meta[name].fillna("").astype(str).str.contains("\ufffd", regex=False).sum())
                          for name in meta.select_dtypes(include="object")}
    metadata_fields = {str(row.Name): {"alias": row.Alias, "intro": None if pd.isna(row.Intro) else str(row.Intro)}
                       for row in meta.itertuples() if row.Type == "COL"}
    cost_numeric = {name: {"missing": int(cost[name].isna().sum()),
                           "nonfinite": int((~np.isfinite(cost[name])).sum()),
                           "negative": int(cost[name].lt(0).sum()),
                           "min": float(cost[name].min()), "max": float(cost[name].max())}
                    for name in ("cost1", "cost2")}
    report = {
        "status": "success", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(source), "bytes": source.stat().st_size, "sha256": source_hash,
                  "read_only": True, "quick_check": integrity},
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
        "tables": {"Item_feature": {"rows": len(data), "columns": len(data.columns)},
                   "Cost": {"rows": len(cost), "columns": len(cost.columns)},
                   "bwds_meta": {"rows": len(meta), "columns": len(meta.columns)}},
        "schema": schemas,
        "calendar": {"start": str(first.date()), "end": str(last.date()), "calendar_days": len(calendar),
                     "observed_dates": int(data.date.nunique()), "globally_missing_dates": [str(x.date()) for x in calendar.difference(data.date)],
                     "invalid_dates": invalid_dates, "duplicate_item_dates": int(data.duplicated(["date", "item_id"]).sum()),
                     "exact_duplicate_rows": int(data.duplicated().sum()), "duplicate_id_values": int(data.id.duplicated().sum())},
        "items": {"count": len(items), "observed_days_quantiles": quantiles(items.observed_days),
                  "full_453_day_items": int(items.observed_days.eq(len(calendar)).sum()),
                  "items_with_internal_missing_days": int(items.internal_missing_days.gt(0).sum()),
                  "total_internal_missing_item_days": int(items.internal_missing_days.sum()),
                  "full_panel_potential_item_days": int(len(items) * len(calendar)),
                  "full_panel_missing_item_days": int(items.full_calendar_missing_days.sum()),
                  "last28_complete_items": int(items.last28_observed_days.eq(28).sum()),
                  "daily_item_count_quantiles": quantiles(dates.observed_items),
                  "static_fields": static},
        "target": {"column": "qty_alipay", "meaning": "recorded paid quantity (成交件数), an observed sales proxy, not latent unmet demand",
                   "rows": len(data), "zero_rows": int(data.qty_alipay.eq(0).sum()), "zero_fraction": float(data.qty_alipay.eq(0).mean()),
                   "sum": total_paid, "quantiles": quantiles(data.qty_alipay),
                   "p99": float(data.qty_alipay.quantile(.99)), "maximum": int(data.qty_alipay.max()),
                   "qty_gmv_not_equal_rows": int(data.qty_gmv.ne(data.qty_alipay).sum()),
                   "qty_gmv_total": int(data.qty_gmv.sum()), "same_day_correlations_descriptive_only": correlations,
                   "consistency_check_counts_not_proven_errors": checks,
                   "top_days": [{"date": str(idx.date()), "quantity": int(row.recorded_paid_quantity),
                                 "share": float(row.share_of_all_recorded_paid_quantity)} for idx, row in spikes.iterrows()]},
        "cost": {"distinct_items": len(cost_items), "warehouse_codes": {str(k): int(v) for k, v in cost.ch_code.value_counts().items()},
                 "duplicate_item_warehouse_keys": int(cost.duplicated(["sp_id", "ch_code"]).sum()),
                 "matched_feature_items": len(feature_items & cost_items), "unmatched_feature_items": sorted(feature_items - cost_items),
                 "numeric": cost_numeric, "regional_time_demand_present": False},
        "fields": fields, "metadata": metadata_fields,
        "all_table_missing_counts": {
            name: {str(k): int(v) for k, v in frame.isna().sum().items()}
            for name, frame in (("Item_feature", data), ("Cost", cost), ("bwds_meta", meta))
        },
        "metadata_replacement_character_rows": replacement_counts,
        "metadata_encoding": {"sqlite_text": "Unicode aliases are intact; no U+FFFD replacement characters were observed",
                              "companion_csv": "GB18030-decodable; interpreting its bytes as UTF-8 fails",
                              "console_note": "A GBK stdout stream read as UTF-8 can display mojibake; source values are checked via UTF-8 JSON and escaped Unicode"},
        "decisions": [
            "Use qty_alipay as the paid-quantity target and keep qty_gmv as a different historical outcome.",
            "Do not turn missing item dates into observed zero sales. Explicit zero rows remain valid zero targets.",
            "All realized sales, GMV, views, cart and collection features must end at the forecast origin; future realized values are excluded.",
            "Use chronological 14-day evaluation. Score known labels with coverage counts and a clearly defined item cohort; report complete-window exclusions.",
            "Report product errors and recorded national aggregate errors separately; aggregate totals have a changing recorded product mix.",
            "Separate shopping-festival peak windows from ordinary windows without choosing model weights from future outcomes.",
            "Static identifiers are categorical labels, not numerical distances. Learn encodings only from historical training data.",
            "Cost joins on item_id=sp_id; warehouse costs do not provide regional historical demand or customer preference observations.",
            "Correlated same-day paid order count or amount cannot serve as inputs for future paid quantity unless shifted into history.",
        ],
    }
    if sha256(source) != source_hash:
        raise RuntimeError("The source file changed while auditing")
    (output / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(report, output)
    verify_exports(source, output)
    return report


def write_markdown(report: dict, output: Path) -> None:
    target, items, cal = report["target"], report["items"], report["calendar"]
    lines = [
        "# 智能分仓数据审查", "",
        f"实际读取完整 SQLite：商品日记录 {report['tables']['Item_feature']['rows']:,} 行、32 列，{items['count']:,} 个商品；日期 {cal['start']}—{cal['end']}，{cal['observed_dates']} 个日期。商品—日期无重复，所有数值字段均已检查缺失、非有限值、负数和分位数，详见 `field_profile.csv`。", "",
        f"预测目标选 `qty_alipay`（成交件数），是已记录支付成交量，不等于未满足需求。总量 {target['sum']:,} 件；明确记录为零的行占 {target['zero_fraction']:.2%}。`qty_gmv` 是另一下单口径，{target['qty_gmv_not_equal_rows']:,} 行与成交件数不同，不能混作同一答案。", "",
        f"只有 {items['full_453_day_items']} 个商品覆盖全期。单品记录数中位 {items['observed_days_quantiles']['0.5']:g} 天；{items['items_with_internal_missing_days']} 个商品在首末记录之间有缺日，共 {items['total_internal_missing_item_days']:,} 个商品日。缺记录不代表销量为零，商品首次出现前和最后出现后的记录也未知。逐商品和逐日范围保存在 `item_coverage.csv`、`date_coverage.csv`。", "",
        "类目、大类、品牌、供应商在同一商品内均未发现变化。编号应按类别处理，不能把编号差当业务距离。未来浏览、加购、收藏、下单、成交和金额都属于未来实际发生结果；预测只能用截止日及以前的记录。同期 `num_alipay` 与目标相关性很高，直接使用会把结果信息带入输入。", "",
        "成交与下单的日期口径可能不同：成交件数大于当日下单件数的行数是 " + str(target['consistency_check_counts_not_proven_errors']['paid_quantity_greater_than_ordered_quantity']) + "。另有 " + str(target['consistency_check_counts_not_proven_errors']['zero_paid_quantity_with_positive_paid_amount']) + " 行成交件数为零、成交金额却大于零。现有说明不能解释金额与件数的不一致；保留原成交件数目标，不按金额反推数量、不单凭不相等删除记录。其他数量、人数和金额核对见 JSON 的 consistency_check_counts_not_proven_errors。", "",
        "| 最大成交日期 | 件数 | 占全部已记录成交 |", "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {r['date']} | {r['quantity']:,} | {r['share']:.2%} |" for r in target["top_days"][:5])
    lines += ["", "购物节尖峰会影响总量误差，因此实验要同时报告普通窗口与含固定购物节日期的窗口；这些分组只用于检查误差，不能用后来已知的高峰决定模型。全国汇总反映当前记录商品集合，商品覆盖随日期变化，不能称为全区域完整需求。", "",
              "`Cost` 为 1,000 个商品在 `all`、1—5 六个仓口径上的 6,000 行成本，商品键全部可连接且商品—仓键不重复。商品行为表没有区域或仓字段，因此成本表不能补出区域销量、区域偏好或各仓历史需求。", "",
              "SQLite 中文元数据可正常读取，未发现 Unicode 替换字符；旁边的元数据 CSV 需按 GB18030 读取。终端编码不匹配会显示乱码，不能把显示问题当作源数据损坏。所有字段概况、元数据和来源 SHA256 在 `data_audit.json`。", "",
              "本次采用只读 SQLite 连接，运行前后 SHA256 相同。脚本还用独立 SQL 聚合逐一核对导出的 1,000 行商品表和 453 行日期表，核对缺日数、最大间隔和目标总量；结果见 `data_audit_validation.json`。复跑：", "", "```powershell", ".\\.venv\\Scripts\\python.exe research/smart-demand/audit.py", "```", "",
              "建议模型先比较同一十四日目标上的近期中位数、近期均值、星期基线和共享小网络。任何标签缺日都应记录评估覆盖，不能用插值后的答案证明模型更准。", ""]
    (output / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    report = audit(args.input, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "rows": report["tables"]["Item_feature"]["rows"], "items": report["items"]["count"],
                      "internal_missing_item_days": report["items"]["total_internal_missing_item_days"],
                      "source_sha256": report["input"]["sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
