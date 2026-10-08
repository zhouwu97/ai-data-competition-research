"""题一练习：用过去预测未来14天；比较简单基线和日历趋势模型。"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from common import read_input, output_dir, write_csv, write_json, save_figure
from common import assert_future_split, plt
from forecasting import forecast_at_origin

def main():
    out = output_dir("task01")
    raw = read_input("demand.csv", ["date", "region", "sku", "quantity"])
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    if raw.duplicated(["date", "region", "sku"]).any() or (raw.quantity < 0).any():
        raise ValueError("本练习要求每个日期-区域-SKU唯一，且数量非负。")
    # 小型区域-SKU描述统计；没有品牌、浏览等字段，不能据此解释真实消费偏好。
    regional = raw.groupby(["region", "sku"]).agg(
        total_quantity=("quantity", "sum"), mean_daily_quantity=("quantity", "mean"),
        std_daily_quantity=("quantity", "std")).reset_index()
    write_csv(regional, out / "regional_summary.csv")
    # 本练习预测全国汇总序列；商品级预测是正式赛题需要另行确认的粒度。
    series = raw.groupby("date")["quantity"].sum().sort_index()
    if not series.index.equals(pd.date_range(series.index.min(), series.index.max())):
        raise ValueError("日期不连续；不能偷偷把缺失日填0，先解释缺失。")
    predictions, metrics = [], []
    # 四个不重叠的14天验证窗口；后一个训练集可以包含此前已经发生的日期。
    for cutoff in [56, 70, 84, 98]:
        train, valid = series.iloc[:cutoff], series.iloc[cutoff:cutoff + 14]
        assert_future_split(train.index, valid.index)
        # 共用入口只把cutoff之前的历史交给模型；未来数量留到下面评分时使用。
        candidates = forecast_at_origin(series, cutoff, len(valid))
        for name, pred in candidates.items():
            mae = float(mean_absolute_error(valid.to_numpy(), pred))
            # RMSE把较大的误差罚得更重；和MAE一起看，不直接当库存损失。
            rmse = float(np.sqrt(np.mean((valid.to_numpy() - pred) ** 2)))
            metrics.append([cutoff, str(valid.index[0].date()), name, mae, rmse])
            for date, actual, estimate in zip(valid.index, valid.to_numpy(), pred):
                predictions.append([cutoff, date.date(), name, int(actual), float(estimate)])
    frame = pd.DataFrame(predictions, columns=["cutoff_day", "date", "model",
                                              "actual", "prediction"])
    scores = pd.DataFrame(metrics, columns=["cutoff_day", "valid_start", "model", "mae", "rmse"])
    write_csv(frame, out / "predictions.csv")
    write_csv(scores, out / "metrics.csv")
    means = scores.groupby("model")["mae"].mean().to_dict()
    summary = {"data_kind": "synthetic", "windows": 4, "horizon_days": 14,
               "target": "national_total_quantity", "mean_mae": means,
               "mean_rmse": scores.groupby("model")["rmse"].mean().to_dict()}
    write_json(summary, out / "summary.json")
    fig, ax = plt.subplots(figsize=(9, 4))
    latest = frame[frame.cutoff_day == 98]
    actual = latest[latest.model == "recent_mean"]
    ax.plot(pd.to_datetime(actual.date), actual.actual, color="#263238", label="Actual")
    for name, group in latest.groupby("model"):
        ax.plot(pd.to_datetime(group.date), group.prediction, label=name)
    ax.set(title="Synthetic data: final 14-day backtest", ylabel="Quantity")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    save_figure(fig, out / "forecast.svg")
    text = "# 题一教学实验记录\n\n全部数据由00_make_data.py生成，无真实比赛成绩。\n\n"
    text += "| 方法 | 四个窗口平均MAE（件/天） |\n| --- | --- |\n"
    for name, value in means.items():
        text += f"| {name} | {value:.4f} |\n"
    text += ("\n数据含人为设置的周规律和线性趋势，适合日历模型；"
             "这不证明该模型在真实赛题优于其他方法。平均值之外，请查看metrics.csv的逐窗口结果。"
             "没有库存成本、缺货或补货决策数据，不能推导库存节省。\n\n"
             "证据：[区域-SKU统计](regional_summary.csv)、[逐窗口指标](metrics.csv)、[逐日预测](predictions.csv)、"
             "[图表](forecast.svg)、[汇总](summary.json)。\n")
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print("题一完成：4个14天窗口，3种方法；真实赛题粒度待确认。")

if __name__ == "__main__":
    main()
