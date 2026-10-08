"""从已保存逐条预测重算真实配送研究；只用标准库，不调用建模代码。"""
import argparse
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import re
import sys

try:
    from .check_calculations import (CalculationError, close, require, rows, read_json,
                                     unique_index, error_metrics)
except ImportError:
    from check_calculations import (CalculationError, close, require, rows, read_json,
                                    unique_index, error_metrics)


def report_table(report, header):
    result, active, found = {}, False, False
    for line in report.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells == header:
            require(not found, f"report.md: 重复指标表{header}")
            active, found = True, True
            continue
        if not line.startswith("|"):
            active = False
        if not active or all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        require(len(cells) == 2 and cells[0] not in result, "report.md: 指标行重复或列数不符")
        result[cells[0]] = cells[1]
    require(found, f"report.md: 缺少指标表{header}")
    return result


def check_report(output, volume, duration, tail):
    report = (output / "report.md").read_text(encoding="utf-8")
    for header, values in [(["方法", "窗口平均MAE（记录/天）"], volume),
                           (["方法", "全部测试运单MAE（分钟）"], duration)]:
        # 旧报告的四窗口表名也明确检查，兼容既有产物。
        if header[1] == "窗口平均MAE（记录/天）" and "| 四窗口平均MAE（记录/天） |" in report:
            header = ["方法", "四窗口平均MAE（记录/天）"]
        saved = report_table(report, header)
        require(set(saved) == set(values), "report.md: 方法集合不符")
        for method, value in values.items():
            require(saved[method] == f"{value:.4f}", f"report.md: {method} MAE数值不符")
    gain = None if duration["global_median"] == 0 else (
        100 * (duration["global_median"] - duration["hour_median"]) / duration["global_median"])
    expected = "不可计算" if gain is None else f"{gain:.2f}%"
    require(re.findall(r"按小时分组相对总体中位数的MAE变化为([^（\n]+)", report) == [expected],
            "report.md: 时长MAE变化不符或重复")
    require(re.findall(r"改进后的全部运单误差仍为([\d.]+)分钟", report) ==
            [f"{duration['hour_median']:.2f}"],
            "report.md: 总体时长误差不符")
    if tail["hour_median"] is None:
        require("没有超过6小时的测试运单" in report, "report.md: 缺少长尾不可计算说明")
    else:
        require(re.findall(r"超过6小时的运单，按小时分组误差仍为([\d.]+)分钟", report) ==
                [f"{tail['hour_median']:.2f}"],
                "report.md: 长尾误差不符")


def check_public_research(output):
    output = Path(output)
    if (output / "run_status.json").exists():
        require(read_json(output / "run_status.json")["status"] == "success",
                "真实研究本次运行未成功；查看report.md与run_status.json")
    summary = read_json(output / "summary.json")
    parameters = summary.get("parameters", {})
    horizon, windows = parameters.get("horizon_days", 14), parameters.get("windows", 4)
    daily = unique_index(rows(output / "daily_counts.csv"), ["date"], "daily_counts.csv")
    volume_rows = rows(output / "volume_predictions.csv")
    unique_index(volume_rows, ["window_start", "date", "method"], "volume_predictions.csv")
    volume_metrics = unique_index(rows(output / "volume_metrics.csv"),
                                  ["window_start", "method"], "volume_metrics.csv")
    volume_groups, means = defaultdict(list), defaultdict(list)
    for row in volume_rows:
        volume_groups[(row["window_start"], row["method"])].append(row)
        close(row["actual"], daily[(row["date"],)]["completed_records"], "完成量逐条实际值")
    starts = sorted({key[0] for key in volume_groups})
    require(len(starts) == windows, "真实研究回测窗口数量不符")
    require(set(volume_groups) == set(volume_metrics) == {
        (start, method) for start in starts for method in
        ["recent_mean", "weekday_mean", "calendar_ridge", "sequential_select"]},
        "完成量预测和指标的窗口/方法集合不符")
    for key, members in volume_groups.items():
        start = date.fromisoformat(key[0])
        days = [(start + timedelta(days=i)).isoformat() for i in range(horizon)]
        require(sorted(row["date"] for row in members) == days, "完成量窗口日期不符")
        require(volume_metrics[key]["window_end"] == days[-1], "完成量窗口结束日期不符")
        value, _ = error_metrics([row["actual"] for row in members], [row["prediction"] for row in members])
        close(volume_metrics[key]["mae_records_per_day"], value, f"volume_metrics.csv {key} MAE")
        means[key[1]].append(value)
    volume = {method: sum(values) / len(values) for method, values in means.items()}
    require(set(summary["volume_window_mean_mae"]) == set(volume), "完成量汇总方法集合不符")
    for method, value in volume.items():
        close(summary["volume_window_mean_mae"][method], value, f"summary.json {method} 完成量MAE")

    predicted = rows(output / "duration_predictions.csv")
    groups, all_errors, tail_errors = defaultdict(list), defaultdict(list), defaultdict(list)
    metrics = unique_index(rows(output / "duration_metrics.csv"),
                           ["window_start", "method"], "duration_metrics.csv")
    diagnostics = defaultdict(list)
    for row in predicted:
        actual, estimate = float(row["actual_minutes"]), float(row["prediction_minutes"])
        # error_metrics也拒绝NaN和无穷；不能用非法输入制造看似很小的误差。
        error_metrics([actual], [estimate])
        require(actual >= 0 and estimate >= 0 and 0 <= int(row["accept_hour"]) <= 23,
                "时长预测包含非法小时或负数")
        error = abs(actual - estimate)
        method = row["method"]
        groups[(row["window_start"], method)].append(row)
        all_errors[method].append(error)
        if actual > 360:
            tail_errors[method].append(error)
        if "accept_date" in row:
            require(key_date_in_window(row["accept_date"], row["window_start"], horizon),
                    "时长接单日期不在窗口内")
            for by, group in [("accept_hour", str(int(row["accept_hour"]))),
                              ("accept_date", row["accept_date"]),
                              ("duration_band", "over_360" if actual > 360 else "up_to_360")]:
                diagnostics[(by, group, method)].append(error)
    require(set(groups) == set(metrics) == {(start, method) for start in starts
                                           for method in ["global_median", "hour_median"]},
            "时长预测和指标的窗口/方法集合不符")
    for start in starts:
        first, second = groups[(start, "global_median")], groups[(start, "hour_median")]
        fields = ["accept_hour", "actual_minutes"]
        if "accept_date" in first[0]:
            fields += ["accept_date", "sample_index"]
            unique_index(first, ["sample_index"], "时长预测总体法逐条编号")
            unique_index(second, ["sample_index"], "时长预测小时法逐条编号")
        require([[row[field] for field in fields] for row in first] ==
                [[row[field] for field in fields] for row in second],
                "两种时长方法未使用相同逐条对象/实际值")
        require(metrics[(start, "global_median")]["known_training_records"] ==
                metrics[(start, "hour_median")]["known_training_records"], "时长训练样本数不一致")
    for key, members in groups.items():
        saved = metrics[key]
        expected_end = (date.fromisoformat(key[0]) + timedelta(days=horizon - 1)).isoformat()
        require(saved["window_end"] == expected_end, "时长窗口结束日期不符")
        value, _ = error_metrics([row["actual_minutes"] for row in members],
                                [row["prediction_minutes"] for row in members])
        close(saved["test_records"], len(members), f"duration_metrics.csv {key} 测试条数")
        close(saved["mae_minutes"], value, f"duration_metrics.csv {key} MAE")
        tail_members = [row for row in members if float(row["actual_minutes"]) > 360]
        close(saved["records_over_360_minutes"], len(tail_members), f"{key} 长尾条数")
        if tail_members:
            tail_value, _ = error_metrics([row["actual_minutes"] for row in tail_members],
                                         [row["prediction_minutes"] for row in tail_members])
            close(saved["mae_over_360_minutes"], tail_value, f"{key} 长尾MAE")
        else:
            require(saved["mae_over_360_minutes"] in ["", "0", "0.000000"], "无长尾样本时指标应为空")
    duration = {method: sum(values) / len(values) for method, values in all_errors.items()}
    tail = {method: sum(tail_errors[method]) / len(tail_errors[method])
            if tail_errors[method] else None for method in duration}
    require(set(summary["duration_record_mean_mae_minutes"]) == set(duration), "时长汇总方法集合不符")
    for method, value in duration.items():
        close(summary["duration_record_mean_mae_minutes"][method], value, f"summary.json {method} 时长MAE")
    if "duration_diagnostics" in summary:
        recorded = unique_index(rows(output / summary["duration_diagnostics"]),
                                ["by", "group", "method"], "duration_diagnostics.csv")
        require(set(recorded) == set(diagnostics), "时长诊断分组集合不符")
        for key, errors in diagnostics.items():
            close(recorded[key]["test_records"], len(errors), f"时长诊断{key} 条数")
            close(recorded[key]["mae_minutes"], sum(errors) / len(errors), f"时长诊断{key} MAE")
    check_report(output, volume, duration, tail)
    return f"真实研究：重算{len(predicted)}行时长预测、完成量窗口、长尾、总体MAE与报告数字。"


def key_date_in_window(day, start, horizon):
    return date.fromisoformat(start) <= date.fromisoformat(day) < date.fromisoformat(start) + timedelta(days=horizon)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "research/lade-jilin/results")
    args = parser.parse_args()
    try:
        print("OK " + check_public_research(args.output_dir))
    except (CalculationError, OSError, KeyError, ValueError, IndexError, TypeError) as error:
        print(f"FAIL 真实研究：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
