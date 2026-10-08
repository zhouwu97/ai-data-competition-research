"""公开真实配送数据的两条小路线：记录量预测与接单至完成时长。

从仓库根目录：python research/lade-jilin/run.py --download
也可以用 --data 指向自行取得的同版本CSV，或用--custom-data分析同字段自有数据。
先读plan.md和before-model.md，了解建模前的想法为什么改变。
"""
import argparse
from datetime import datetime, timezone
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


def duration_estimates(history, hours, min_records=30):
    """同样已完成历史：全局中位数，对照接单小时中位数；少样本回退。"""
    if history.empty:
        raise ValueError("缺少已完成的历史时长。")
    overall = float(history.minutes.median())
    by_hour = history.groupby(history.accept_time.dt.hour).minutes.agg(["median", "count"])
    reliable = by_hour[by_hour["count"] >= min_records]["median"].to_dict()
    return {"global_median": np.repeat(overall, len(hours)),
            "hour_median": np.array([reliable.get(int(h), overall) for h in hours])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--download", action="store_true", help="从官方固定版本下载一次")
    parser.add_argument("--custom-data", action="store_true", help="分析自有同字段数据，记录其摘要和来源类型")
    parser.add_argument("--output-dir", type=Path, help="结果目录；自有数据必须指定")
    parser.add_argument("--year", type=int, help="输入没有年份时明确提供年份；官方固定为2022")
    parser.add_argument("--observation-end", help="可选观察结束日期YYYY-MM-DD，只使用此前已完成记录（含当天）")
    parser.add_argument("--horizon-days", type=int, default=14)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--hour-min-records", type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.horizon_days, args.windows, args.hour_min_records) < 1:
        parser.error("窗口长度、窗口数和小时最小样本数必须是正整数。")
    if args.custom_data and (args.download or args.output_dir is None or args.data is None):
        parser.error("自有数据需要--data和--output-dir，不能使用--download。")
    if not args.custom_data and args.year not in [None, 2022]:
        parser.error("官方固定版本的年份为2022。")
    args.data = args.data or HERE / "data/delivery_jl.csv"
    args.output_dir = args.output_dir or HERE / "results"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "run_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": ["python", "research/lade-jilin/run.py", *(sys.argv[1:] if argv is None else argv)],
              "data_path": str(args.data.resolve())}
    write_json(status, args.output_dir / "run_status.json")
    (args.output_dir / "report.md").write_text(
        "# 配送研究本次运行：running\n\n正在运行，分项文件可能仍属于旧结果。\n", encoding="utf-8", newline="\n")
    try:
        execute(args)
        status["status"] = "success"
        write_json(status, args.output_dir / "run_status.json")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(status, args.output_dir / "run_status.json")
        (args.output_dir / "report.md").write_text(
            f"# 配送研究本次运行：failed\n\n{status['error']}\n\n"
            "本次未成功完成，分项文件可能属于旧运行或未完成运行。\n", encoding="utf-8", newline="\n")
        raise


def execute(args):
    started = time.perf_counter()
    if args.download and not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=45) as response:
            args.data.write_bytes(response.read())
    if not args.data.exists():
        raise SystemExit("未找到数据；加 --download，或用 --data 指向官方吉林CSV。")
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if not args.custom_data and digest != EXPECTED_SHA:
        raise ValueError("数据与本次固定版本不同；不要悄悄覆盖已记录的实验输入。")
    raw = pd.read_csv(args.data, encoding="utf-8-sig", dtype={"order_id": "string", "city": "string"})
    required = ["order_id", "city", "ds", "accept_time", "delivery_time"]
    if raw.empty or not set(required).issubset(raw.columns) or raw[required].isna().any().any():
        raise ValueError("必需字段缺失，先解释数据口径。")
    if raw.order_id.duplicated().any() or (not args.custom_data and set(raw.city) != {"Jilin"}):
        raise ValueError("运单ID必须唯一；官方模式还要求城市为Jilin。")
    frame = raw[["accept_time", "delivery_time"]].copy()
    # 官方CSV缺年份，2022来自作者说明；自有数据使用完整时间或显式提供的年份。
    year = args.year if args.custom_data else 2022
    for column in frame:
        values = frame[column].astype(str)
        if year is not None:
            values = str(year) + "-" + values
        frame[column] = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if year is None:
        ds = pd.to_datetime(raw.ds.astype(str), format="%Y-%m-%d", errors="raise")
    else:
        ds_text = raw.ds.astype(str).str.zfill(4)
        ds = pd.to_datetime(str(year) + "-" + ds_text.str[:2] + "-" + ds_text.str[2:],
                            format="%Y-%m-%d", errors="raise")
    if frame.isna().any().any() or ds.isna().any():
        raise ValueError("时间字段不能为NaT。")
    frame["minutes"] = (frame.delivery_time - frame.accept_time).dt.total_seconds() / 60
    if (frame.minutes < 0).any():
        raise ValueError("出现负时长，先核对跨日和时间字段，不能自动改成正值。")
    if args.observation_end:
        end = pd.Timestamp(args.observation_end)
        if pd.isna(end) or end != end.normalize():
            raise ValueError("观察结束日期需要是完整日期。")
        frame = frame[frame.delivery_time < end + pd.Timedelta(days=1)]
    if frame.empty:
        raise ValueError("观察区间没有已完成记录。")
    daily = frame.groupby(frame.delivery_time.dt.normalize()).size().sort_index()
    observed = longest_block(daily)
    if len(observed) < 28 + args.horizon_days * (args.windows + 1):
        raise ValueError("连续可评分区间不足，至少需28天训练、1个选模历史窗口和指定测试窗口。")
    out = args.output_dir
    write_csv(daily.rename("completed_records").rename_axis("date").reset_index(),
              out / "daily_counts.csv")
    missing = pd.date_range(daily.index.min(), daily.index.max()).difference(daily.index)
    data_kind = "user_supplied_records" if args.custom_data else "public_real_records"
    audit = {"source_url": None if args.custom_data else URL,
             "source_revision": None if args.custom_data else REVISION, "source_sha256": digest,
             "source_bytes": args.data.stat().st_size, "data_kind": data_kind,
             "license_on_current_data_card": None if args.custom_data else "Apache-2.0; research purposes",
             "rows": len(raw), "unique_orders": raw.order_id.nunique(),
             "observed_completed_rows": len(frame),
             "source_columns": raw.columns.tolist(),
             "missing_field_counts": raw.isna().sum().loc[lambda x: x > 0].to_dict(),
             "ds_days": ds.nunique(), "ds_finish_day_disagreements":
             int((ds.loc[frame.index] != frame.delivery_time.dt.normalize()).sum()),
             "ds_accept_day_disagreements": int((ds.loc[frame.index] != frame.accept_time.dt.normalize()).sum()),
             "no_finish_records_dates": missing.strftime("%Y-%m-%d").tolist(),
             "observation_block": {"start": str(observed.index[0].date()),
                                   "end": str(observed.index[-1].date()), "days": len(observed)},
             "minutes_quantiles": frame.minutes.quantile([0, .5, .9, .99, 1]).to_dict()}
    # numpy整数转成Python整数，避免JSON依赖某一环境的隐式转换。
    audit["unique_orders"], audit["ds_days"] = int(audit["unique_orders"]), int(audit["ds_days"])
    write_json(audit, out / "data_audit.json")
    windows = range(len(observed) - args.windows * args.horizon_days, len(observed), args.horizon_days)
    volume_predictions, volume_metrics, choices = [], [], []
    eta_predictions, eta_metrics = [], []
    for cutoff in windows:
        history, future = observed.iloc[:cutoff], observed.iloc[cutoff:cutoff + args.horizon_days]
        origin, end = future.index[0], future.index[-1] + pd.Timedelta(days=1)
        chosen, past_scores, completed = select_by_completed_windows(history, horizon=args.horizon_days)
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
        if valid.empty:
            raise ValueError(f"{origin.date()}窗口没有接单测试对象，不能计算时长MAE。")
        duration_predictions = duration_estimates(mature, valid.accept_time.dt.hour, args.hour_min_records)
        known_after = str(valid.delivery_time.max())
        for method, prediction in duration_predictions.items():
            errors = np.abs(valid.minutes.to_numpy() - prediction)
            long_mask = valid.minutes.to_numpy() > 360
            eta_metrics.append([origin.date(), future.index[-1].date(), method, len(mature),
                                len(valid), float(errors.mean()), int(long_mask.sum()),
                                float(errors[long_mask].mean()) if long_mask.any() else None, known_after])
            eta_predictions.extend([origin.date(), method, int(accepted.hour), float(actual), float(pred),
                                    accepted.date(), i]
                                   for i, (accepted, actual, pred) in enumerate(zip(valid.accept_time,
                                                                                 valid.minutes, prediction)))
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
    eta_frame = pd.DataFrame(eta_predictions, columns=["window_start", "method", "accept_hour", "actual_minutes",
                                                       "prediction_minutes", "accept_date", "sample_index"])
    write_csv(eta_frame, out / "duration_predictions.csv")
    eta_frame["absolute_error"] = (eta_frame.actual_minutes - eta_frame.prediction_minutes).abs()
    eta_frame["duration_band"] = np.where(eta_frame.actual_minutes > 360, "over_360", "up_to_360")
    diagnostic_tables = []
    for by in ["accept_hour", "accept_date", "duration_band"]:
        table = eta_frame.groupby([by, "method"]).absolute_error.agg(test_records="size", mae_minutes="mean").reset_index()
        table = table.rename(columns={by: "group"})
        table.insert(0, "by", by)
        diagnostic_tables.append(table)
    write_csv(pd.concat(diagnostic_tables, ignore_index=True), out / "duration_diagnostics.csv")
    write_csv(eta_table, out / "duration_metrics.csv")
    volume_means = volume_table.groupby("method").mae_records_per_day.mean().to_dict()
    # 四窗口运单数不同，时长按所有测试运单汇总；不把窗口均值再简单平均。
    eta_means = {name: float(np.average(group.mae_minutes, weights=group.test_records))
                 for name, group in eta_table.groupby("method")}
    summary = {"data_kind": data_kind, "volume_window_mean_mae": volume_means,
               "parameters": {"horizon_days": args.horizon_days, "windows": args.windows,
                              "hour_min_records": args.hour_min_records, "year": year,
                              "observation_end": args.observation_end},
               "duration_diagnostics": "duration_diagnostics.csv",
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
    introduction = ("输入由使用者提供，来源类型为user_supplied_records；数据来源与使用范围由提供者说明。"
                    if args.custom_data else "输入来自菜鸟公开LaDe吉林配送记录。")
    text = ("# 配送记录研究：从字段审查到两条路线\n\n"
            f"{introduction}本次比较完成量与时长两条路线。\n\n"
            "## 实际数据先改变了什么\n\n"
            f"取得{len(raw)}条记录、{audit['source_bytes']}字节，无重复运单ID；观察范围内有{len(frame)}条已完成记录。"
            "观察范围内ds与完成日期有"
            f"{audit['ds_finish_day_disagreements']}条不同，所以完成量按delivery_time日期统计。"
            f"完成日期区间内{len(missing)}天无记录，空日含义不明，没有补零。最长连续观察段为"
            f"{observed.index[0].date()}至{observed.index[-1].date()}，共{len(observed)}天。\n\n"
            f"年份口径：{year if year is not None else '输入完整时间中的年份'}；"
            "官方固定版本的2022来自作者说明。文件与字段核查见[data_audit.json](data_audit.json)。"
            "逐日记录图的断线表示没有记录，不表示业务量为零。\n\n"
            "## 路线一：预测已记录完成量\n\n"
            f"连续段按观察范围的日期覆盖情况选定，评价它最后{args.windows}个{args.horizon_days}天窗口；"
            "这是该观察段的回测。每次只用更早日期预测；顺序选择仍采用"
            "最近两个已完成窗口平均MAE最小，历史不足两个时只使用已有完整窗口。"
            "[选择记录](volume_choices.csv)保存当时可用日期及历史成绩。\n\n"
            "| 方法 | 窗口平均MAE（记录/天） |\n| --- | ---: |\n")
    for name, value in volume_means.items():
        text += f"| {name} | {value:.4f} |\n"
    text += ("\n[逐日预测](volume_predictions.csv)与[窗口结果](volume_metrics.csv)可以重算。"
             "这个目标测量文件中完成记录的条数；它不覆盖取消单、未完成单或平台之外的需求。"
             "尾部下降无法仅凭此文件区分真实业务变化与采样变化，当前不给排班或库存收益结论。\n\n"
             "## 路线二：接单时预测完成时长\n\n"
             f"比较已完成历史的总体中位数与接单小时中位数，小时组少于{args.hour_min_records}条时回退总体值。"
             "两法使用相同历史和未来运单，训练要求接单和完成时间均早于窗口起点。"
             "模型在起点冻结，到每张运单接单时读取当时可见的小时；"
             "未来完成时间仅在评分时使用。极长样本保留，另列超过6小时的误差。\n\n"
             "| 方法 | 全部测试运单MAE（分钟） |\n| --- | ---: |\n")
    for name, value in eta_means.items():
        text += f"| {name} | {value:.4f} |\n"
    gain = (100 * (eta_means["global_median"] - eta_means["hour_median"]) / eta_means["global_median"]
            if eta_means["global_median"] else None)
    gain_text = f"{gain:.2f}%" if gain is not None else "不可计算（总体MAE为0）"
    tail = eta_frame[(eta_frame.method == "hour_median") & (eta_frame.actual_minutes > 360)]
    tail_text = (f"超过6小时的运单，按小时分组误差仍为{tail.absolute_error.mean():.2f}分钟。"
                 if not tail.empty else "没有超过6小时的测试运单，长尾MAE不可计算。")
    text += (f"\n按小时分组相对总体中位数的MAE变化为{gain_text}（正数表示减少）。"
             f"改进后的全部运单误差仍为{eta_means['hour_median']:.2f}分钟；"
             f"{tail_text}接单时段是否有预测价值需要逐窗口和后续数据确认。"
             "查看[逐窗口与长尾结果](duration_metrics.csv)，确认改善是否只集中于某一时段；"
             "[逐条时长预测](duration_predictions.csv)不含原运单、配送员ID与坐标。"
             "[按接单小时、日期与长尾分组](duration_diagnostics.csv)可定位改善或退步集中在哪些对象；"
             "其中长尾分组仅用于事后诊断，预测时不能预知实际时长。"
             "时长包含接单后的等待和作业，不能当成两点之间纯行驶时间。\n\n"
             "![实际记录与未来误差](research_comparison.svg)\n\n"
             "## 现在怎样选择继续研究的方向\n\n"
             "完成量路线首先卡在记录覆盖口径，继续堆预测模型无法解释月底断档。"
             "下一步需要数据方说明采样与业务覆盖，才能把预测连接到业务决策。"
             "时长路线可以继续查接单时段与长尾的关系，但是否能据此改善服务，"
             "还要补批次、承诺时限或独立后续数据。两条路线的误差单位不同，不直接排名。\n\n"
             "继续研究时先检查每个窗口的改善是否保持，再决定增加特征或模型。"
             "保留完成量基线及它暴露的数据问题；路线取舍需结合提供者的业务口径。\n\n"
             "## 复现与答辩\n\n"
             "从仓库根目录运行 `python research/lade-jilin/run.py --download`；"
             "已有官方文件可用 `--data /path/to/delivery_jl.csv`。自有数据用"
             "`--custom-data --data /path/to/records.csv --output-dir outputs/my-delivery`；"
             "完整时间含年份，缺年份时显式加`--year`。窗口和观察日期见[参数](summary.json)。"
             "官方模式核对固定SHA256；自有模式记录实际SHA256和来源类型。\n\n"
             "> 为什么不把没记录的天填成0？\n> 因为数据没有说明缺记录等于零业务。"
             "我们选择连续可观察段评分，并把覆盖口径列为下一步最需要的材料。\n\n"
             "> 分组误差降低，就能说提高配送效率吗？\n> 还不能。"
             "目前只是接单时长预测的误差对照，尚未改变配送策略，也没有测量服务收益。\n\n"
             "输入来源类型、实际摘要和字段检查见[data_audit.json](data_audit.json)。\n")
    if not args.custom_data:
        text += ("\n来源：[官方数据卡](https://huggingface.co/datasets/Cainiao-AI/LaDe)、"
                 "[作者时间范围说明](https://github.com/wenhaomin/LaDe)。数据卡标Apache-2.0与研究用途；"
                 "引用Wu等，*LaDe: The First Comprehensive Last-mile Express Dataset from Industry*, "
                 "KDD 2024, DOI 10.1145/3637528.3671548。\n")
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"配送记录研究完成（{data_kind}）；用时{time.perf_counter()-started:.2f}秒。")
    print("volume MAE:", volume_means)
    print("duration MAE:", eta_means)


if __name__ == "__main__":
    main()
