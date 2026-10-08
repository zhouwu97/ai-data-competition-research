"""公开真实配送数据的两条小路线：记录量预测与接单至完成时长。

从仓库根目录：python research/lade-jilin/run.py --download
也可以用 --data 指向自行取得的同版本CSV。原始数据不提交仓库。
先读plan.md和before-model.md，了解建模前的想法为什么改变。
"""
import argparse
import hashlib
from pathlib import Path
import platform
import sys
import time
from urllib.request import urlopen
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from common import write_csv, write_json, save_figure, plt
from forecasting import METHODS, predict_methods, select_by_completed_windows

REVISION = "be2cec02775cafc8d52230303f32134382bcc50b"
URL = f"https://huggingface.co/datasets/Cainiao-AI/LaDe/resolve/{REVISION}/delivery/delivery_jl.csv"
EXPECTED_SHA = "12e2cf4664dd5b4475d39dddee8872f5a03b3082f08f0eece7f103baee6c6e73"


def longest_block(daily):
    """空日期的含义未知，选连续有记录的最长区间；不把空日补0。"""
    groups = (daily.index.to_series().diff() != pd.Timedelta(days=1)).cumsum()
    blocks = [block for _, block in daily.groupby(groups.to_numpy())]
    return max(blocks, key=lambda block: (len(block), -block.index[0].value))


def mature_history(frame, origin):
    """接单早不等于标签已知：到预测时尚未完成的运单不能用于训练。"""
    return frame[(frame.accept_time < origin) & (frame.delivery_time < origin)]


def duration_estimates(history, hours):
    """同样已完成历史：全局中位数，对照接单小时中位数；少样本回退。"""
    if history.empty:
        raise ValueError("缺少已完成的历史时长。")
    overall = float(history.minutes.median())
    by_hour = history.groupby(history.accept_time.dt.hour).minutes.agg(["median", "count"])
    reliable = by_hour[by_hour["count"] >= 30]["median"].to_dict()
    return {"global_median": np.repeat(overall, len(hours)),
            "hour_median": np.array([reliable.get(int(h), overall) for h in hours])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/delivery_jl.csv")
    parser.add_argument("--download", action="store_true", help="从官方固定版本下载一次")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.download and not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=45) as response:
            args.data.write_bytes(response.read())
    if not args.data.exists():
        raise SystemExit("未找到数据；加 --download，或用 --data 指向官方吉林CSV。")
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError("数据与本次固定版本不同；不要悄悄覆盖已记录的实验输入。")
    raw = pd.read_csv(args.data)
    required = ["order_id", "city", "ds", "accept_time", "delivery_time"]
    if not set(required).issubset(raw.columns) or raw[required].isna().any().any():
        raise ValueError("必需字段缺失，先解释数据口径。")
    if raw.order_id.duplicated().any() or set(raw.city) != {"Jilin"}:
        raise ValueError("本次要求吉林运单ID唯一。")
    frame = raw[["accept_time", "delivery_time"]].copy()
    # CSV时间没有年。2022来自作者README；不能用机器当前年份拼接。
    for column in frame:
        frame[column] = pd.to_datetime("2022-" + frame[column], format="%Y-%m-%d %H:%M:%S")
    ds_text = raw.ds.astype(str).str.zfill(4)
    ds = pd.to_datetime("2022-" + ds_text.str[:2] + "-" + ds_text.str[2:])
    frame["minutes"] = (frame.delivery_time - frame.accept_time).dt.total_seconds() / 60
    if (frame.minutes < 0).any():
        raise ValueError("出现负时长，先核对跨日和时间字段，不能自动改成正值。")
    daily = frame.groupby(frame.delivery_time.dt.normalize()).size().sort_index()
    observed = longest_block(daily)
    if len(observed) < 98:
        raise ValueError("连续可评分区间不足，不能完成本次四个14日窗口。")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    write_csv(daily.rename("completed_records").rename_axis("date").reset_index(),
              out / "daily_counts.csv")
    missing = pd.date_range(daily.index.min(), daily.index.max()).difference(daily.index)
    audit = {"source_url": URL, "source_revision": REVISION, "source_sha256": digest,
             "source_bytes": args.data.stat().st_size, "data_kind": "public_real_records",
             "license_on_current_data_card": "Apache-2.0; research purposes",
             "rows": len(raw), "unique_orders": raw.order_id.nunique(),
             "source_columns": raw.columns.tolist(),
             "missing_field_counts": raw.isna().sum().loc[lambda x: x > 0].to_dict(),
             "ds_days": ds.nunique(), "ds_finish_day_disagreements":
             int((ds != frame.delivery_time.dt.normalize()).sum()),
             "ds_accept_day_disagreements": int((ds != frame.accept_time.dt.normalize()).sum()),
             "no_finish_records_dates": missing.strftime("%Y-%m-%d").tolist(),
             "observation_block": {"start": str(observed.index[0].date()),
                                   "end": str(observed.index[-1].date()), "days": len(observed)},
             "minutes_quantiles": frame.minutes.quantile([0, .5, .9, .99, 1]).to_dict()}
    # numpy整数转成Python整数，避免JSON依赖某一环境的隐式转换。
    audit["unique_orders"], audit["ds_days"] = int(audit["unique_orders"]), int(audit["ds_days"])
    write_json(audit, out / "data_audit.json")
    windows = range(len(observed) - 56, len(observed), 14)
    volume_predictions, volume_metrics, choices = [], [], []
    eta_predictions, eta_metrics = [], []
    for cutoff in windows:
        history, future = observed.iloc[:cutoff], observed.iloc[cutoff:cutoff + 14]
        origin, end = future.index[0], future.index[-1] + pd.Timedelta(days=1)
        chosen, past_scores, completed = select_by_completed_windows(history)
        estimates = predict_methods(history, future.index)
        estimates["sequential_select"] = estimates[chosen]
        choices.append([origin.date(), history.index[-1].date(),
                        max(row["window_end"] for row in completed), chosen,
                        *[past_scores[name] for name in METHODS]])
        for method, prediction in estimates.items():
            error = np.abs(future.to_numpy() - prediction)
            volume_metrics.append([origin.date(), future.index[-1].date(), method, float(error.mean())])
            volume_predictions.extend([origin.date(), day.date(), method, int(actual), float(pred)]
                                      for day, actual, pred in zip(future.index, future, prediction))
        # 时长路线使用同样的四个日历窗口，按接单日选未来对象；训练标签须已完成。
        mature = mature_history(frame, origin)
        valid = frame[(frame.accept_time >= origin) & (frame.accept_time < end)]
        duration_predictions = duration_estimates(mature, valid.accept_time.dt.hour)
        known_after = str(valid.delivery_time.max())
        for method, prediction in duration_predictions.items():
            errors = np.abs(valid.minutes.to_numpy() - prediction)
            long_mask = valid.minutes.to_numpy() > 360
            eta_metrics.append([origin.date(), future.index[-1].date(), method, len(mature),
                                len(valid), float(errors.mean()), int(long_mask.sum()),
                                float(errors[long_mask].mean()) if long_mask.any() else 0., known_after])
            eta_predictions.extend([origin.date(), method, int(hour), float(actual), float(pred)]
                                   for hour, actual, pred in zip(valid.accept_time.dt.hour,
                                                                valid.minutes, prediction))
    volume_table = pd.DataFrame(volume_metrics, columns=["window_start", "window_end", "method", "mae_records_per_day"])
    eta_table = pd.DataFrame(eta_metrics, columns=["window_start", "window_end", "method", "known_training_records",
                                                 "test_records", "mae_minutes", "records_over_360_minutes",
                                                 "mae_over_360_minutes", "all_test_labels_known_after"])
    write_csv(pd.DataFrame(volume_predictions, columns=["window_start", "date", "method", "actual", "prediction"]),
              out / "volume_predictions.csv")
    write_csv(volume_table, out / "volume_metrics.csv")
    write_csv(pd.DataFrame(choices, columns=["forecast_date", "history_end", "latest_completed_backtest_end",
                                            "selected_model", "past_mae_recent_mean", "past_mae_weekday_mean",
                                            "past_mae_calendar_ridge"]), out / "volume_choices.csv")
    write_csv(pd.DataFrame(eta_predictions, columns=["window_start", "method", "accept_hour", "actual_minutes", "prediction_minutes"]),
              out / "duration_predictions.csv")
    write_csv(eta_table, out / "duration_metrics.csv")
    volume_means = volume_table.groupby("method").mae_records_per_day.mean().to_dict()
    # 四窗口运单数不同，时长按所有测试运单汇总；不把窗口均值再简单平均。
    eta_means = {name: float(np.average(group.mae_minutes, weights=group.test_records))
                 for name, group in eta_table.groupby("method")}
    summary = {"data_kind": "public_real_records", "volume_window_mean_mae": volume_means,
               "duration_record_mean_mae_minutes": eta_means,
               "volume_choices": [row[3] for row in choices], "python": platform.python_version(),
               "source_sha256": digest,
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "forecasting_sha256": hashlib.sha256((ROOT / "examples/forecasting.py").read_bytes()).hexdigest()}
    write_json(summary, out / "summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    calendar = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    axes[0].plot(calendar.index, calendar.values, color="#218380", linewidth=1)
    axes[0].axvspan(observed.index[0], observed.index[-1], color="#218380", alpha=.08)
    axes[0].set(title="Observed completion records; gaps remain missing", ylabel="Records/day")
    for name, group in eta_table.groupby("method"):
        axes[1].plot(pd.to_datetime(group.window_start), group.mae_minutes, marker="o", label=name)
    axes[1].set(title="Future duration errors", ylabel="MAE (minutes)")
    axes[1].legend(fontsize=8)
    fig.autofmt_xdate()
    save_figure(fig, out / "research_comparison.svg")
    text = ("# 真实数据试做：从字段审查到两条路线\n\n"
            "输入来自菜鸟公开LaDe吉林配送记录。本次自行实现两条小路线；"
            "[建模前计划](../plan.md)与[看数据后的改判](../before-model.md)保留原文。\n\n"
            "## 实际数据先改变了什么\n\n"
            f"取得{len(raw)}条记录、{audit['source_bytes']}字节，无重复运单ID。ds与完成日期有"
            f"{audit['ds_finish_day_disagreements']}条不同，所以完成量按delivery_time日期统计。"
            f"完成日期区间内{len(missing)}天无记录，空日含义不明，没有补零。最长连续观察段为"
            f"{observed.index[0].date()}至{observed.index[-1].date()}，共{len(observed)}天。\n\n"
            "日期取2022依据作者说明；文件与字段核查见[data_audit.json](data_audit.json)。"
            "逐日记录图的断线表示没有记录，不表示业务量为零。\n\n"
            "## 路线一：预测已记录完成量\n\n"
            "连续段按完整文件的日期覆盖情况选定，评价它最后四个14天窗口；"
            "这是该观察段的回测。每次只用更早日期预测；顺序选择仍采用"
            "最近两个已完成窗口平均MAE最小，历史不足两个时只使用已有完整窗口。"
            "[选择记录](volume_choices.csv)保存当时可用日期及历史成绩。\n\n"
            "| 方法 | 四窗口平均MAE（记录/天） |\n| --- | ---: |\n")
    for name, value in volume_means.items():
        text += f"| {name} | {value:.4f} |\n"
    text += ("\n[逐日预测](volume_predictions.csv)与[窗口结果](volume_metrics.csv)可以重算。"
             "这个目标测量文件中完成记录的条数；它不覆盖取消单、未完成单或平台之外的需求。"
             "尾部下降无法仅凭此文件区分真实业务变化与采样变化，当前不给排班或库存收益结论。\n\n"
             "## 路线二：接单时预测完成时长\n\n"
             "比较已完成历史的总体中位数与接单小时中位数，小时组少于30条时回退总体值。"
             "两法使用相同历史和未来运单，训练要求接单和完成时间均早于窗口起点。"
             "模型在起点冻结，到每张运单接单时读取当时可见的小时；"
             "未来完成时间仅在评分时使用。极长样本保留，另列超过6小时的误差。\n\n"
             "| 方法 | 全部测试运单MAE（分钟） |\n| --- | ---: |\n")
    for name, value in eta_means.items():
        text += f"| {name} | {value:.4f} |\n"
    gain = 100 * (eta_means["global_median"] - eta_means["hour_median"]) / eta_means["global_median"]
    tail_means = {name: float(np.average(group.mae_over_360_minutes,
                                         weights=group.records_over_360_minutes))
                  for name, group in eta_table.groupby("method")}
    text += (f"\n按小时分组相对总体中位数的MAE变化为{gain:.2f}%（正数表示减少）。"
             f"改进后的全部运单误差仍为{eta_means['hour_median']:.2f}分钟；"
             f"超过6小时的运单，按小时分组误差仍为{tail_means['hour_median']:.2f}分钟。"
             "这次发现是接单时段有一点预测信息，但还没有解决长耗时问题。"
             "查看[逐窗口与长尾结果](duration_metrics.csv)，确认改善是否只集中于某一时段；"
             "[逐条时长预测](duration_predictions.csv)不含原运单、配送员ID与坐标。"
             "时长包含接单后的等待和作业，不能当成两点之间纯行驶时间。\n\n"
             "![实际记录与未来误差](research_comparison.svg)\n\n"
             "## 现在怎样选择继续研究的方向\n\n"
             "完成量路线首先卡在记录覆盖口径，继续堆预测模型无法解释月底断档。"
             "下一步需要数据方说明采样与业务覆盖，才能把预测连接到业务决策。"
             "时长路线可以继续查接单时段与长尾的关系，但是否能据此改善服务，"
             "还要补批次、承诺时限或独立后续数据。两条路线的误差单位不同，不直接排名。\n\n"
             "因此本次建议：若仅用当前文件做一份小研究，优先深化时长诊断，"
             "保留完成量基线及它暴露的数据问题。若取得完整采样说明，或长尾时间口径被推翻，"
             "这项选择需要重新评估。它是公开真实数据练习的路线取舍，不是正式竞赛选题。\n\n"
             "## 复现与答辩\n\n"
             "从仓库根目录运行 `python research/lade-jilin/run.py --download`；"
             "已有文件可用 `--data /path/to/delivery_jl.csv`。程序固定数据版本及SHA256，"
             "原CSV由官方获取，仓库只保存本次统计与结果。\n\n"
             "> 为什么不把没记录的天填成0？\n> 因为数据没有说明缺记录等于零业务。"
             "我们选择连续可观察段评分，并把覆盖口径列为下一步最需要的材料。\n\n"
             "> 分组误差降低，就能说提高配送效率吗？\n> 还不能。"
             "目前只是接单时长预测的误差对照，尚未改变配送策略，也没有测量服务收益。\n\n"
             "来源：[官方数据卡](https://huggingface.co/datasets/Cainiao-AI/LaDe)、"
             "[作者时间范围说明](https://github.com/wenhaomin/LaDe)。数据卡标Apache-2.0与研究用途；"
             "本次引用Wu等，*LaDe: The First Comprehensive Last-mile Express Dataset from Industry*, "
             "KDD 2024, DOI 10.1145/3637528.3671548。\n")
    (out / "report.md").write_text(text, encoding="utf-8")
    print(f"公开真实数据两条路线完成；用时{time.perf_counter()-started:.2f}秒。")
    print("volume MAE:", volume_means)
    print("duration MAE:", eta_means)


if __name__ == "__main__":
    main()
