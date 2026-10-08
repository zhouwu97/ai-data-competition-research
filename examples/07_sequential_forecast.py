"""事前怎样选预测方法：只用已完成窗口，比较固定方法与顺序选择。

从仓库根目录运行 python examples/07_sequential_forecast.py；先生成00教学数据。
三种机制和选择规则在代码中固定，不用预测期答案替当前选择找理由。
"""
import numpy as np
import pandas as pd
from common import INPUT, output_dir, write_csv, write_json, save_figure, plt
from forecasting import METHODS, forecast_at_origin, select_by_completed_windows


def make_scenarios(original):
    """第一种沿用原数据；后两种前56天相同，分别下降、下降后再增长。"""
    weekly = np.array([0, 8, 4, 12, 28, 60, 40])
    decline = original.copy()
    rng = np.random.default_rng(1701)
    anchor = float(original.iloc[49:56].mean() - weekly.mean())
    # 与06的下降情景一致，独立于哪个模型将获胜。
    for day in range(56, len(decline)):
        decline.iloc[day] = max(5, round(anchor - 2 * (day - 55)
                                       + weekly[decline.index[day].dayofweek]
                                       + rng.normal(0, 5)))
    regrowth = decline.copy()
    rng = np.random.default_rng(2701)
    # 第84天开始，机制由下降换成增长；此前数据和下降情景完全相同。
    anchor = float(decline.iloc[83] - weekly[decline.index[83].dayofweek])
    for day in range(84, len(regrowth)):
        regrowth.iloc[day] = max(5, round(anchor + 3 * (day - 83)
                                        + weekly[regrowth.index[day].dayofweek]
                                        + rng.normal(0, 5)))
    return {"original": original, "decline": decline, "regrowth": regrowth}


def main():
    out = output_dir("sequential")
    raw = pd.read_csv(INPUT / "demand.csv", parse_dates=["date"])
    original = raw.groupby("date").quantity.sum().sort_index()
    if len(original) < 112:
        raise ValueError("本教学比较需要112个日期，先运行00_make_data.py。")
    scenarios = make_scenarios(original)
    inputs, predictions, past_scores, decisions, metric_rows = [], [], [], [], []
    for scenario, series in scenarios.items():
        inputs.extend([scenario, date.date(), int(qty)] for date, qty in series.items())
        for cutoff in [56, 70, 84, 98]:
            history = series.iloc[:cutoff]
            # 先选方法和生成预测，再接触这次预测期的真实值评分。
            chosen, known_scores, completed = select_by_completed_windows(history)
            candidates = forecast_at_origin(series, cutoff)
            decision_date = history.index[-1] + pd.Timedelta(days=1)
            used_cutoffs = sorted({row["window_cutoff"] for row in completed})
            for row in completed:
                known_at = pd.Timestamp(row["window_end"]) + pd.Timedelta(days=1)
                assert known_at <= decision_date
                past_scores.append([scenario, cutoff, str(decision_date.date()),
                                    row["window_cutoff"], row["window_start"], row["window_end"],
                                    str(known_at.date()), row["model"], row["mae"]])
            # 到这里选择已经冻结。下面的actual用于事后评价，不重新改变chosen。
            valid = series.iloc[cutoff:cutoff + 14]
            evaluated = dict(candidates)
            evaluated["sequential_select"] = candidates[chosen]
            maes = {}
            for method, estimates in evaluated.items():
                error = valid.to_numpy() - estimates
                mae = float(np.mean(np.abs(error)))
                maes[method] = mae
                metric_rows.append([scenario, cutoff, method, mae,
                                    float(np.sqrt(np.mean(error ** 2)))])
                predictions.extend([scenario, cutoff, date.date(), method,
                                    chosen if method == "sequential_select" else method,
                                    int(actual), float(prediction)]
                                   for date, actual, prediction in zip(valid.index, valid, estimates))
            explanation = (f"只看已完成窗口{used_cutoffs}的平均MAE；"
                           f"{chosen}最小（{known_scores[chosen]:.4f}），平局按固定方法顺序")
            decisions.append([scenario, cutoff, str(decision_date.date()),
                              str(history.index[-1].date()), "|".join(map(str, used_cutoffs)),
                              max(row["window_end"] for row in completed), chosen,
                              *[known_scores[name] for name in METHODS], explanation,
                              str(valid.index[-1].date()),
                              str((valid.index[-1] + pd.Timedelta(days=1)).date()),
                              maes["sequential_select"]])
    write_csv(pd.DataFrame(inputs, columns=["scenario", "date", "quantity"]),
              out / "scenario_inputs.csv")
    write_csv(pd.DataFrame(predictions, columns=["scenario", "cutoff_day", "date", "method",
                                                "selected_model", "actual", "prediction"]),
              out / "predictions.csv")
    past = pd.DataFrame(past_scores, columns=["scenario", "decision_cutoff_day", "decision_date",
                                             "scored_window_cutoff", "window_start", "window_end",
                                             "known_at_date", "model", "mae"])
    write_csv(past, out / "past_scores.csv")
    decision_frame = pd.DataFrame(decisions, columns=[
        "scenario", "cutoff_day", "decision_date", "last_observed_date", "used_window_cutoffs",
        "latest_completed_window_end", "selected_model", "past_mae_recent_mean",
        "past_mae_weekday_mean", "past_mae_calendar_ridge", "reason_before_forecast",
        "future_window_end", "future_mae_known_at_date", "future_mae"])
    write_csv(decision_frame, out / "decisions.csv")
    metrics = pd.DataFrame(metric_rows, columns=["scenario", "cutoff_day", "method", "mae", "rmse"])
    write_csv(metrics, out / "metrics.csv")
    means = metrics.groupby(["scenario", "method"]).mae.mean().unstack(0)
    summary = {
        "data_kind": "synthetic_sequential", "horizon_days": 14,
        "decision_cutoffs": [56, 70, 84, 98], "first_history_window_cutoff": 28,
        "selection_rule": "minimum_mean_mae_of_last_two_completed_14_day_windows",
        "tie_order": list(METHODS), "mean_mae": means.to_dict(),
        "selected_models": {name: group.selected_model.tolist()
                            for name, group in decision_frame.groupby("scenario", sort=False)},
    }
    write_json(summary, out / "summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    labels = {"calendar_ridge": "Calendar Ridge", "recent_mean": "Recent mean",
              "weekday_mean": "Weekday mean", "sequential_select": "Sequential choice"}
    colors = {"calendar_ridge": "#687baf", "recent_mean": "#218380",
              "weekday_mean": "#d88b47", "sequential_select": "#263238"}
    for ax, scenario in zip(axes, scenarios):
        for method in [*METHODS, "sequential_select"]:
            group = metrics[(metrics.scenario == scenario) & (metrics.method == method)]
            ax.plot(group.cutoff_day, group.mae, marker="o", color=colors[method],
                    label=labels[method], linewidth=2 if method == "sequential_select" else 1,
                    linestyle="--" if method == "sequential_select" else "-")
        ax.set(title=scenario, xlabel="Forecast origin: completed days", xticks=[56, 70, 84, 98])
    axes[0].set_ylabel("Future 14-day MAE (units/day)")
    axes[0].legend(fontsize=7)
    save_figure(fig, out / "sequential_comparison.svg")
    initial_decision = decision_frame.iloc[0]
    text = ("# 事前选方法：选择不是看到结果后再解释\n\n"
            "问题：当我们不知道接下来14天销量时，应怎样在三个已有方法中选一个？"
            "本次固定一个小规则，再让它面对原趋势、下降突变、下降后重新增长三种教学机制。\n\n"
            "## 决策在什么时间形成\n\n"
            "规则在执行前固定：从第28天开始每14天做一个历史预测，"
            "当前只取最近两个完整结束窗口的平均MAE，选最小者。完全同分按最近均值、"
            "同星期均值、日历Ridge顺序；不根据后来成绩改规则。\n\n"
            "第一次决策已有56天历史：cutoff=28的窗口覆盖第28~41天，"
            "cutoff=42覆盖第42~55天。两者都已完整发生，因此可用于第56天起的预测。"
            "第56~69天的成绩要等第69天结束才知道，只能从第70天起参与选择；"
            "在第56天决策时，它没有投票权。CSV使用从0起的天编号。\n\n"
            "三种机制前56天相同，因此首次选择必须相同；再增长机制到第83天结束前"
            "与下降机制相同，因此第84天的选择也必须相同。模型不能预先知道下一天换机制。\n\n"
            f"最初两个已完成窗口的平均MAE：最近均值{initial_decision.past_mae_recent_mean:.4f}、"
            f"同星期均值{initial_decision.past_mae_weekday_mean:.4f}、日历Ridge"
            f"{initial_decision.past_mae_calendar_ridge:.4f}，因此初次选"
            f"{initial_decision.selected_model}。这一步不需要知道第56天之后的任何数量。\n\n"
            "## 实际选择与未来结果\n\n"
            "| 情景 | 已观察天数 | 使用的已完成窗口起点 | 当时选择 | 后来14天MAE |\n"
            "| --- | ---: | --- | --- | ---: |\n")
    for row in decision_frame.itertuples():
        text += (f"| {row.scenario} | {row.cutoff_day} | {row.used_window_cutoffs} | "
                 f"{row.selected_model} | {row.future_mae:.4f} |\n")
    text += ("\n[decisions.csv](decisions.csv)把事前理由和事后成绩放在不同字段，"
             "[past_scores.csv](past_scores.csv)列出当时可知的每个成绩及可用日期。"
             "未来成绩直到future_mae_known_at_date才可用于下一轮；它不参与当前选择。\n\n"
             "## 顺序选择是否真的更好\n\n"
             "| 方法/固定规则 | 原趋势平均MAE | 下降突变平均MAE | 重新增长平均MAE |\n"
             "| --- | ---: | ---: | ---: |\n")
    for method in [*METHODS, "sequential_select"]:
        row = means.loc[method]
        text += f"| {method} | {row.original:.4f} | {row.decline:.4f} | {row.regrowth:.4f} |\n"
    text += ("\n![三种机制的逐窗口比较](sequential_comparison.svg)\n\n"
             f"原趋势中，规则一直选日历Ridge，平均MAE为{means.loc['sequential_select', 'original']:.4f}。"
             f"下降情景中，顺序规则平均MAE为{means.loc['sequential_select', 'decline']:.4f}，"
             f"比固定最近均值{means.loc['recent_mean', 'decline']:.4f}更差；"
             f"重新增长时，顺序规则为{means.loc['sequential_select', 'regrowth']:.4f}，"
             f"也高于固定同星期均值{means.loc['weekday_mean', 'regrowth']:.4f}。"
             "因此这条自动选择规则并没有在三个情景中普遍胜出。\n\n"
             "顺序选择也有反应延迟：突变发生的第一个窗口，还没有新机制的完整误差可用。"
             "后续又可能被两个窗口的平均值拖慢。以上是固定规则的实际成绩，不是"
             "挑出每个未来窗口事后最佳方法拼成的推荐；顺序规则是否值得用，需要看它"
             "相较固定方法在哪些窗口改善、在哪些窗口变差。这里不增加模型，也不凭这三种造数机制推荐线上部署。\n\n"
             "如果想尝试只看一个窗口或更长窗口，请先另记规则，再增加未看过的情景比较，"
             "不要改完规则后只展示它最有利的同一段数据。\n\n"
             "## 怎样复查\n\n"
             "从[三套逐日输入](scenario_inputs.csv)检查历史边界；"
             "用[逐日预测](predictions.csv)重算[MAE/RMSE](metrics.csv)。"
             "每个sequential_select预测应等于当时selected_model的预测。"
             "共享预测函数只接收历史数量；测试会保持历史不变、把预测期及之后销量"
             "分别改成0、大数和打乱顺序，要求预测与选择不变。测试还故意放入读取"
             "未来销量的函数，确认同一个检验会拒绝它。\n\n"
             "执行：`python examples/07_sequential_forecast.py`；"
             "测试：`python -m unittest tests.test_forecasting -v`。\n")
    (out / "report.md").write_text(text, encoding="utf-8")
    print("顺序预测完成：三种机制、四个起点；只按最近两个已完成窗口选择已有方法。")


if __name__ == "__main__":
    main()
