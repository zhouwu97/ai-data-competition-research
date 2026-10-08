"""用自有需求CSV导出未来预测；输入字段：date、region、sku、quantity。"""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from common import read_input, write_csv, write_json
from forecasting import predict_methods


def demand_history(raw, as_of):
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.isna().any() or (frame.date != frame.date.dt.normalize()).any():
        raise ValueError("date必须是完整日期，不能含时分秒或空值。")
    if frame.duplicated(["date", "region", "sku"]).any():
        raise ValueError("日期-区域-SKU必须唯一。")
    # 观察日期当天及之后的数据不参与拟合。
    frame = frame[frame.date < as_of].copy()
    frame["quantity"] = pd.to_numeric(frame.quantity, errors="raise")
    if frame.empty or not np.isfinite(frame.quantity).all() or (frame.quantity < 0).any():
        raise ValueError("观察日期以前需要有限非负需求数据。")
    history = frame.groupby("date").quantity.sum().sort_index()
    if history.index[-1] != as_of - pd.Timedelta(days=1):
        raise ValueError("历史必须覆盖观察日期的前一天；缺失日不能自动填0。")
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", required=True, help="预测开始日期YYYY-MM-DD，仅使用此前历史")
    parser.add_argument("--horizon-days", type=int, default=14)
    args = parser.parse_args(argv)
    as_of = pd.Timestamp(args.as_of)
    if pd.isna(as_of) or as_of != as_of.normalize() or args.horizon_days < 1:
        parser.error("观察日期必须有效，预测期必须是正整数。")
    raw = read_input(args.data.name, ["date", "region", "sku", "quantity"], args.data.parent)
    history = demand_history(raw, as_of)
    dates = pd.date_range(as_of, periods=args.horizon_days)
    predictions = predict_methods(history, dates)
    records = [[day.date(), method, float(value)] for method, values in predictions.items()
               for day, value in zip(dates, values)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(records, columns=["date", "model", "prediction"]),
              args.output_dir / "future_predictions.csv")
    write_json({"data_kind": "user_supplied_records", "input_path": str(args.data.resolve()),
                "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "as_of": str(as_of.date()), "history_days": len(history),
                "history_start": str(history.index[0].date()),
                "history_end": str(history.index[-1].date()), "horizon_days": args.horizon_days,
                "target": "total_quantity", "models": list(predictions)},
               args.output_dir / "summary.json")
    (args.output_dir / "report.md").write_text(
        f"# 未来需求预测导出\n\n观察日期：{as_of.date()}；仅使用此前{len(history)}天历史。"
        f"预测{args.horizon_days}天，按日期汇总所有区域与SKU的quantity。\n\n"
        "[预测表](future_predictions.csv)分别列出最近均值、星期均值与日历模型；"
        "没有未来实际值，因此没有声称未来MAE或库存收益。商品级预测需要另行定义。\n\n"
        "[输入与参数](summary.json)保存输入摘要和历史截止日期。\n", encoding="utf-8", newline="\n")
    print(f"未来需求预测已保存到{args.output_dir / 'future_predictions.csv'}。")


if __name__ == "__main__":
    main()
