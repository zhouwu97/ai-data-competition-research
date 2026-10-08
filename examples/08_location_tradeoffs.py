"""用84个候选方案研究选址权衡，不增加新的优化算法。

运行：python examples/08_location_tradeoffs.py
输入沿用06压力练习的九个教学城市，先运行06_method_stress.py。
三个中心、候选地址和需求保持不变，只改变服务半径与可接受损失。
"""
from hashlib import sha256
from itertools import combinations
from math import hypot

import pandas as pd

from common import HERE, RESULT, output_dir, write_csv, write_json, save_figure, plt


TOLERANCE = 1e-9
RADII = (3, 5, 7)
BUDGETS = (0, .002, .01)


def evaluate(points, centers, radius):
    """需求分配给最近中心；恰好等于半径的点也算被覆盖。"""
    distances = [min(hypot(p["x_km"] - c["x_km"], p["y_km"] - c["y_km"])
                     for c in centers) for p in points]
    total = sum(p["demand"] for p in points)
    covered = [d <= radius + TOLERANCE for d in distances]
    return {
        "city_coverage": sum(covered) / len(points),
        "demand_coverage": sum(p["demand"] for p, yes in zip(points, covered) if yes) / total,
        "weighted_distance_km": sum(p["demand"] * d for p, d in zip(points, distances)) / total,
        "weighted_squared_loss": sum(p["demand"] * d ** 2 for p, d in zip(points, distances)),
    }


def dominates(left, right):
    """平方损失越小、需求覆盖越大越好，至少一项严格更好才叫支配。"""
    loss_left, loss_right = left["weighted_squared_loss"], right["weighted_squared_loss"]
    cover_left, cover_right = left["demand_coverage"], right["demand_coverage"]
    no_worse = loss_left <= loss_right + TOLERANCE and cover_left >= cover_right - TOLERANCE
    better = loss_left < loss_right - TOLERANCE or cover_left > cover_right + TOLERANCE
    return no_worse and better


def demand_key(row):
    # 同覆盖率时优先少损失，再比较城市覆盖、平均距离和地址编号，保证可复现。
    return (-row["demand_coverage"], row["weighted_squared_loss"],
            -row["city_coverage"], row["weighted_distance_km"], row["centers"])


def select(rows, policy, extra_loss=None):
    """预算是相对平方损失的上限，例如.002表示允许比最优增加0.2%。"""
    minimum = min(row["weighted_squared_loss"] for row in rows)
    if policy == "min_squared_loss":
        return min(rows, key=lambda row: (row["weighted_squared_loss"],
                                         -row["demand_coverage"], row["centers"]))
    if policy == "max_city_coverage":
        return min(rows, key=lambda row: (-row["city_coverage"], *demand_key(row)))
    if policy == "max_demand_coverage":
        return min(rows, key=demand_key)
    if policy == "budgeted_demand_coverage":
        if extra_loss is None or extra_loss < 0:
            raise ValueError("平方损失增幅上限必须是非负数。")
        eligible = [row for row in rows
                    if row["weighted_squared_loss"] <= minimum * (1 + extra_loss) + TOLERANCE]
        return min(eligible, key=demand_key)
    raise ValueError(f"未知选址目标：{policy}")


def enumerate_candidates(points, k, radius):
    rows = []
    for chosen in combinations(points, k):
        rows.append({"radius_km": radius, "k": k,
                     "centers": "|".join(p["city_id"] for p in chosen),
                     **evaluate(points, chosen, radius)})
    minimum = min(row["weighted_squared_loss"] for row in rows)
    for row in rows:
        row["squared_loss_increase_pct"] = (row["weighted_squared_loss"] / minimum - 1) * 100
        row["is_pareto"] = not any(dominates(other, row) for other in rows)
    return rows


def plot_candidates(groups, out):
    # 前三幅展示全部方案；第四幅放大近最优范围，否则0.16%的差异看不清。
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, radius in zip(axes.flat, RADII):
        rows = groups[radius]
        frontier = sorted([r for r in rows if r["is_pareto"]],
                          key=lambda r: r["weighted_squared_loss"])
        ax.scatter([r["squared_loss_increase_pct"] for r in rows],
                   [r["demand_coverage"] * 100 for r in rows],
                   color="#b8c4cd", s=22, alpha=.65, label="All 84 candidates")
        ax.plot([r["squared_loss_increase_pct"] for r in frontier],
                [r["demand_coverage"] * 100 for r in frontier],
                "o-", color="#147d83", linewidth=2, markersize=6, label="Pareto frontier")
        ax.set_title(f"Service radius = {radius} km")
        ax.set_xlabel("Squared-loss increase from minimum (%)")
        ax.set_ylabel("Demand coverage (%)")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc="lower right")
    ax = axes.flat[3]
    colors = ["#d57d32", "#147d83", "#6c73a9"]
    for radius, color in zip(RADII, colors):
        frontier = sorted([r for r in groups[radius] if r["is_pareto"]],
                          key=lambda r: r["weighted_squared_loss"])
        near = [r for r in frontier if r["squared_loss_increase_pct"] <= 1]
        ax.plot([r["squared_loss_increase_pct"] for r in near],
                [r["demand_coverage"] * 100 for r in near], "o-",
                color=color, linewidth=2, label=f"Radius {radius} km")
    ax.axvline(.2, color="#59616b", linestyle="--", linewidth=1, label="0.2% budget")
    ax.set(xlim=(-.03, 1.03), ylim=(60, 100), title="Near-optimal frontier (0–1% extra loss)",
           xlabel="Squared-loss increase from minimum (%)", ylabel="Demand coverage (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.18)
    save_figure(fig, out / "location_tradeoffs.svg")


def write_report(groups, out):
    five = groups[5]
    baseline = select(five, "min_squared_loss")
    alternative = select(five, "max_demand_coverage")
    three = groups[3]
    strict_best = select(three, "max_demand_coverage")
    strict_baseline = select(three, "min_squared_loss")
    loose_budget = select(groups[7], "budgeted_demand_coverage", .002)
    lines = [
        "# 选址权衡：多花一点平方损失，能换来多少服务？", "",
        "先固定问题：九个教学城市既是需求点，也是允许建设的候选位置；选择三个不同中心。"
        "三种服务半径各穷举84个方案，改变的是目标与服务阈值，候选位置、距离口径和需求相同。", "",
        "平方损失是 `Σ(城市需求 × 最近中心距离²)`，单位为需求单位·km²。"
        "它惩罚远距离，但不是建站费用、运费或投资回报。需求覆盖是半径内需求占全部需求的比例；"
        "城市覆盖则每个城市各算一票，两者不应混用。", "",
        "## 5公里下，审查提出的权衡确实存在", "",
        "| 选择目标 | 中心 | 平方损失 | 城市覆盖 | 需求覆盖 | 加权平均距离 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, row in [("最小平方损失", baseline), ("最大城市/需求覆盖", alternative)]:
        lines.append(f"| {label} | {row['centers'].replace('|', ', ')} | {row['weighted_squared_loss']:.0f} | "
                     f"{row['city_coverage']:.2%} | {row['demand_coverage']:.2%} | "
                     f"{row['weighted_distance_km']:.3f} km |")
    loss_gain = alternative["squared_loss_increase_pct"]
    city_gain = (alternative["city_coverage"] - baseline["city_coverage"]) * 100
    demand_gain = (alternative["demand_coverage"] - baseline["demand_coverage"]) * 100
    distance_change = baseline["weighted_distance_km"] - alternative["weighted_distance_km"]
    lines += ["", f"平方损失只增加 **{loss_gain:.4f}%**，城市覆盖提高 **{city_gain:.2f}个百分点**，"
              f"需求覆盖提高 **{demand_gain:.2f}个百分点**，平均距离还下降 **{distance_change:.3f} km**。"
              "百分点是两个百分比直接相减，例如60%变成70%是增加10个百分点，并非增加10%。", "",
              "覆盖方案仍然没有覆盖全部需求。它适合提出一个条件式判断：若企业容许平方损失增加0.2%，"
              "并优先服务半径内需求，这个方案更合适；若平方损失必须严格最优，就保留原方案。", "",
              "## 非支配方案是什么？", "",
              "把平方损失越小和需求覆盖越大作为两个目标。如果另一方案损失不更大、覆盖不更少，"
              "而且至少一项更好，原方案就被‘支配’。找不到这种替代的方案，组成 Pareto（帕累托）前沿。"
              "它是一组需要做取舍的候选，不是自动选出的最终答案。城市覆盖和平均距离保留供解释，"
              "本次没有把它们偷偷加入支配判断。", "",
              "| 半径 | 非支配中心 | 平方损失增幅 | 需求覆盖 | 城市覆盖 |", "| --- | --- | ---: | ---: | ---: |"]
    for radius in RADII:
        for row in sorted([r for r in groups[radius] if r["is_pareto"]],
                          key=lambda r: r["weighted_squared_loss"]):
            lines.append(f"| {radius} km | {row['centers'].replace('|', ', ')} | {row['squared_loss_increase_pct']:.4f}% | "
                         f"{row['demand_coverage']:.2%} | {row['city_coverage']:.2%} |")
    lines += ["", "## 半径与容忍度改变后，结论并不总是漂亮", "",
              "下表每格都是：在相对最优平方损失的指定增幅内，选需求覆盖最高的方案。", "",
              "| 半径 | 0%上限 | 0.2%上限 | 1%上限 | 无增幅上限的最大需求覆盖 |",
              "| --- | --- | --- | --- | --- |"]
    for radius in RADII:
        picks = [select(groups[radius], "budgeted_demand_coverage", budget) for budget in BUDGETS]
        unrestricted = select(groups[radius], "max_demand_coverage")
        cells = [f"{row['demand_coverage']:.2%}（{row['centers'].replace('|', ', ')}）" for row in picks + [unrestricted]]
        lines.append(f"| {radius} km | " + " | ".join(cells) + " |")
    strict_gain = (strict_best["demand_coverage"] - strict_baseline["demand_coverage"]) * 100
    lines += ["", f"**3公里**时，0.2%和1%的容忍度都没有换来更多需求覆盖；若追求最高覆盖，"
              f"平方损失要增加 **{strict_best['squared_loss_increase_pct']:.2f}%**，"
              f"需求覆盖只提高 **{strict_gain:.2f}个百分点**。此时保留最小损失方案可能更合理。"
              f"**7公里**时，0.2%的方案达到{loose_budget['demand_coverage']:.2%}需求覆盖，仍不能写成全部覆盖。"
              "服务承诺的半径必须有业务依据，不能为了漂亮结果而调大。", "",
              "图的前三幅展示全部方案，右下角放大1%增幅内的前沿，避免小差异被横轴范围淹没。"
              "连线只辅助阅读，线上的中间位置不代表本候选集存在对应方案。", "",
              "![目标与阈值权衡](location_tradeoffs.svg)", "",
              "## 怎样用它做一次事前决定", "",
              "先确定服务半径、优先覆盖城市还是需求、可容忍的平方损失增幅，再查看对应方案。"
              "不要看完结果后挑最有利的目标。在真实项目中，还要把距离替换成适当的道路距离，"
              "加入实际候选地址、容量与费用后再检验这组取舍。这里只完成给定候选集的精确比较。", "",
              "- `all_candidates.csv`：每个半径全部84个方案及非支配标记。",
              "- `selected_plans.csv`：三个单目标方案及三个损失上限方案，含所覆盖城市。",
              "- `pareto_plans.csv`：仅保留两个目标下的前沿。",
              "- `input_points.csv`、`input_record.json`：实际输入、摘要、单位与平局规则。",
              "- `summary.json`：本次结果和5公里的直接差值，报告数字从这些计算生成。", "",
              "多目标背景：[pymoo官方教程](https://pymoo.org/getting_started/part_3.html)。"
              "本例只有84个组合，用穷举即可，不需要遗传算法或安装pymoo。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    source = RESULT / "stress/dc_points.csv"
    if not source.exists():
        raise FileNotFoundError("先运行 python examples/06_method_stress.py，生成九个教学城市。")
    frame = pd.read_csv(source).sort_values("city_id").reset_index(drop=True)
    required = ["city_id", "x_km", "y_km", "demand"]
    if len(frame) != 9 or frame[required].isna().any().any() or frame.city_id.duplicated().any():
        raise ValueError("本练习需要九个唯一、字段完整的候选城市。")
    if (frame.demand <= 0).any():
        raise ValueError("需求必须大于0，不能把缺失当作0。")
    points = frame[required].to_dict("records")
    out = output_dir("tradeoffs")
    write_csv(frame[required], out / "input_points.csv")
    write_json({"source": "examples/results/stress/dc_points.csv",
                "source_sha256": sha256(source.read_bytes()).hexdigest(),
                "scenario": "nine_synthetic_cities", "k": 3, "candidate_count": 9,
                "radii_km": list(RADII), "extra_loss_budgets": list(BUDGETS),
                "objectives": {"minimize": "weighted_squared_loss", "maximize": "demand_coverage"},
                "distance": "euclidean_km", "coverage_boundary": "distance <= radius",
                "loss_unit": "synthetic_demand_unit * km^2", "numeric_tolerance": TOLERANCE,
                "tie_break": "demand coverage, squared loss, city coverage, mean distance, center IDs",
                "min_squared_tie_break": "squared loss, demand coverage, center IDs",
                "max_city_tie_break": "city coverage, then same demand-selection tie break"},
               out / "input_record.json")
    groups = {radius: enumerate_candidates(points, 3, radius) for radius in RADII}
    all_rows, selected = [], []
    for radius, rows in groups.items():
        all_rows.extend(rows)
        policies = [("min_squared_loss", None), ("max_city_coverage", None),
                    ("max_demand_coverage", None)] + [("budgeted_demand_coverage", b) for b in BUDGETS]
        for policy, budget in policies:
            row = select(rows, policy, budget).copy()
            centers = [p for p in points if p["city_id"] in row["centers"].split("|")]
            covered = [p["city_id"] for p in points
                       if min(hypot(p["x_km"] - c["x_km"], p["y_km"] - c["y_km"])
                              for c in centers) <= radius + TOLERANCE]
            row.update(policy=policy, extra_loss_budget="" if budget is None else budget,
                       covered_cities="|".join(covered))
            selected.append(row)
    write_csv(pd.DataFrame(all_rows), out / "all_candidates.csv")
    write_csv(pd.DataFrame(selected), out / "selected_plans.csv")
    write_csv(pd.DataFrame([r for r in all_rows if r["is_pareto"]]), out / "pareto_plans.csv")
    baseline, alternative = select(groups[5], "min_squared_loss"), select(groups[5], "max_demand_coverage")
    summary = {"candidate_solutions_per_radius": len(groups[5]), "total_evaluated": len(all_rows),
               "pareto_counts": {str(r): sum(p["is_pareto"] for p in rows) for r, rows in groups.items()},
               "selected_plans": selected,
               "radius5_tradeoff": {
                   "squared_loss_increase_pct": alternative["squared_loss_increase_pct"],
                   "city_coverage_gain_percentage_points": 100 * (alternative["city_coverage"] - baseline["city_coverage"]),
                   "demand_coverage_gain_percentage_points": 100 * (alternative["demand_coverage"] - baseline["demand_coverage"]),
                   "weighted_distance_reduction_km": baseline["weighted_distance_km"] - alternative["weighted_distance_km"]}}
    write_json(summary, out / "summary.json")
    write_report(groups, out)
    plot_candidates(groups, out)
    print(f"已穷举{len(all_rows)}个半径—选址组合，结果：{out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
