"""四个方法压力练习：故意换掉有利条件，看看原来的判断还能不能成立。

运行：python examples/06_method_stress.py
先运行00_make_data.py及01~04基础练习。全部输入仍是教学数据。
这里保存逐条数据，而不是只保存一个好看的分数，便于其他程序独立重算。
"""
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from common import INPUT, RESULT, output_dir, write_csv, write_json, save_figure, plt
from forecasting import forecast_at_origin


def prediction_stress(out):
    raw = pd.read_csv(INPUT / "demand.csv", parse_dates=["date"])
    original = raw.groupby("date").quantity.sum().sort_index()
    changed = original.copy()
    # 前56天完全不动；其后换成下降机制，不读取任何模型的预测来造答案。
    # 降幅2件/天、周规律及噪声都是本练习的明确设定，不是假定真实市场会这样变。
    rng = np.random.default_rng(1701)
    weekday = np.array([0, 8, 4, 12, 28, 60, 40])
    anchor = float(original.iloc[49:56].mean() - weekday.mean())
    for day in range(56, len(changed)):
        changed.iloc[day] = max(5, round(anchor - 2 * (day - 55)
                                       + weekday[changed.index[day].dayofweek]
                                       + rng.normal(0, 5)))
    inputs, predictions, metrics = [], [], []
    for scenario, series in [("original", original), ("trend_change", changed)]:
        inputs.extend([scenario, date.date(), int(qty)] for date, qty in series.items())
        for cutoff in [56, 70, 84, 98]:
            train, valid = series.iloc[:cutoff], series.iloc[cutoff:cutoff + 14]
            assert train.index.max() < valid.index.min()
            estimates = forecast_at_origin(series, cutoff, len(valid))
            for name, pred in estimates.items():
                metrics.append([scenario, cutoff, name,
                                float(np.mean(np.abs(valid.to_numpy() - pred)))])
                predictions.extend([scenario, cutoff, date.date(), name, int(actual), float(y)]
                                   for date, actual, y in zip(valid.index, valid, pred))
    write_csv(pd.DataFrame(inputs, columns=["scenario", "date", "quantity"]),
              out / "prediction_input.csv")
    write_csv(pd.DataFrame(predictions, columns=["scenario", "cutoff_day", "date", "model",
                                               "actual", "prediction"]),
              out / "prediction_predictions.csv")
    scores = pd.DataFrame(metrics, columns=["scenario", "cutoff_day", "model", "mae"])
    write_csv(scores, out / "prediction_metrics.csv")
    return scores.groupby(["scenario", "model"]).mae.mean().unstack(0)


def abc_labels(values):
    """与基础题二一致：越过80%/95%的那一项仍留在前一档。"""
    before = (values.cumsum() - values) / values.sum()
    return ["A" if p < .8 else "B" if p < .95 else "C" for p in before]


def warehouse_stress(out):
    rows = []
    # 两天一轮，A、B交替成为大需求品项，C保持4件。前两天作为布局观察期。
    for day in range(1, 9):
        for sku, qty in zip(["A", "B", "C"], [20, 2, 4] if day % 2 else [2, 20, 4]):
            rows.append([day, sku, qty])
    demand = pd.DataFrame(rows, columns=["day", "sku", "quantity"])
    classified = []
    for _, daily in demand.groupby("day"):
        daily = daily.sort_values(["quantity", "sku"], ascending=[False, True]).copy()
        daily["ABC"] = abc_labels(daily.quantity)
        classified.append(daily)
    write_csv(pd.concat(classified), out / "warehouse_demand.csv")
    # 玩具仓库：只有1个近货位，距起点2米；另两品项10米。
    # 每件商品各走一次往返，没有容量/合单/拥堵。调换近货位另走60米。
    # 这里相加的是行走米数，不是工时、成本或真实效率。
    near_distance, far_distance, swap_distance = 2.0, 10.0, 60.0
    write_json({"near_distance_m": near_distance, "far_distance_m": far_distance,
                "swap_distance_m": swap_distance, "initial_near_sku": "A",
                "observation_days": [1, 2], "evaluation_days": [3, 4, 5, 6, 7, 8],
                "pick_rule": "one_unit_one_round_trip"}, out / "warehouse_parameters.json")
    decisions = []
    for strategy in ["fixed_two_day", "daily_previous_day", "hold_two_observations"]:
        current = "A"  # 前两天A、B合计相同，按SKU排序平局选A。
        for day in range(3, 9):
            previous = demand[demand.day == day - 1].sort_values(
                ["quantity", "sku"], ascending=[False, True]).iloc[0].sku
            before_previous = demand[demand.day == day - 2].sort_values(
                ["quantity", "sku"], ascending=[False, True]).iloc[0].sku
            chosen = current
            if strategy == "daily_previous_day":
                chosen = previous
            elif strategy == "hold_two_observations" and previous == before_previous:
                chosen = previous
            # 决策仅使用day之前的信息；当天数量只用于事后评价。
            moving = swap_distance if chosen != current else 0.0
            daily = demand[demand.day == day]
            picking = sum(row.quantity * 2 * (near_distance if row.sku == chosen
                                              else far_distance)
                          for row in daily.itertuples())
            decisions.append([strategy, day, day - 1, chosen, picking, moving,
                              picking + moving])
            current = chosen
    operations = pd.DataFrame(decisions, columns=["strategy", "day", "last_observed_day",
                                                   "near_sku", "pick_walk_m", "move_walk_m",
                                                   "total_walk_m"])
    write_csv(operations, out / "warehouse_operations.csv")
    return operations.groupby("strategy")[["pick_walk_m", "move_walk_m", "total_walk_m"]].sum()


def location_metrics(points, weights, centers, radius):
    """先求每城最近距离，再分别求城市覆盖、需求覆盖、平均距离和平方损失。"""
    nearest = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
    covered = nearest <= radius
    return (float(covered.mean()), float(weights[covered].sum() / weights.sum()),
            float(np.average(nearest, weights=weights)), float(np.sum(weights * nearest ** 2)))


def optimal_1d_squared_centers(x, weights, k):
    """此处仅9个有序一维点，枚举连续分段可核对平方损失的全局最优。

    一维平方距离的最优簇可取连续区间，每段最优中心是加权均值。
    这不是通用大规模求解器，只用于排除“碰巧初始化不好”的解释。
    """
    best = None
    for cuts in combinations(range(1, len(x)), k - 1):
        groups = np.split(np.arange(len(x)), cuts)
        centers = np.array([np.average(x[g], weights=weights[g]) for g in groups])
        loss = np.sum(weights * np.abs(x[:, None] - centers).min(axis=1) ** 2)
        if best is None or loss < best[0]:
            best = (loss, centers)
    return np.column_stack([best[1], np.zeros(k)])


def location_stress(out):
    # 固定反例，不在每次执行时搜索一个恰好好看的样本。
    x = np.array([7, 8, 15, 29, 41, 46, 49, 64, 72], dtype=float)
    weights = np.array([77, 57, 61, 63, 76, 36, 57, 9, 13], dtype=float)
    points = np.column_stack([x, np.zeros(len(x))])
    radius = 5.0
    write_csv(pd.DataFrame({"city_id": [f"P{i}" for i in range(9)], "x_km": x,
                           "y_km": 0.0, "demand": weights}), out / "dc_points.csv")
    records, center_rows, continuous = [], [], {}
    for k in [2, 3]:
        model = KMeans(n_clusters=k, n_init=50, random_state=42).fit(points, sample_weight=weights)
        continuous[k] = model.cluster_centers_
        # 将K-means结果映射到互不重复的城市候选点，之后才与同约束覆盖法比较。
        # 选候选组合及中心一一配对的最小距离；小样本直接枚举，不做贪心去重。
        from itertools import permutations
        snapped = min(combinations(range(len(points)), k), key=lambda ids: min(
            np.sum((points[list(ids)] - model.cluster_centers_[list(order)]) ** 2)
            for order in permutations(range(k))))
        candidates = []
        for ids in combinations(range(len(points)), k):
            metric = location_metrics(points, weights, points[list(ids)], radius)
            candidates.append((ids, metric))
        min_squared = min(candidates, key=lambda item: (item[1][3], item[0]))
        # 最大城市覆盖优先，平局时看需求覆盖、加权平均距离，再按ID确定顺序。
        max_coverage = min(candidates, key=lambda item: (-item[1][0], -item[1][1],
                                                        item[1][2], item[0]))
        methods = [("continuous_kmeans", model.cluster_centers_, "continuous_plane", 0),
                   ("continuous_squared_exact", optimal_1d_squared_centers(x, weights, k),
                    "continuous_plane", len(list(combinations(range(1, 9), k - 1)))),
                   ("candidate_kmeans_snapped", points[list(snapped)], "nine_city_candidates", 0),
                   ("candidate_min_squared_exact", points[list(min_squared[0])],
                    "nine_city_candidates", len(candidates)),
                   ("candidate_max_city_cover_exact", points[list(max_coverage[0])],
                    "nine_city_candidates", len(candidates))]
        for method, centers, constraint, enumerated in methods:
            metric = location_metrics(points, weights, centers, radius)
            records.append([method, k, constraint, radius, *metric, enumerated])
            center_rows.extend([method, k, label, float(cx), float(cy)]
                               for label, (cx, cy) in enumerate(centers))
    scores = pd.DataFrame(records, columns=["method", "k", "constraint", "radius_km",
                                           "city_coverage", "demand_coverage",
                                           "weighted_distance_km", "weighted_squared_loss",
                                           "enumerated_solutions"])
    write_csv(scores, out / "dc_evaluation.csv")
    write_csv(pd.DataFrame(center_rows, columns=["method", "k", "center_id", "x_km", "y_km"]),
              out / "dc_centers.csv")
    # 基础例子的实际中心拿来比较100/270/540公里，不重新挑一个有利初始化。
    base = pd.read_csv(INPUT / "cities.csv")
    center_file = RESULT / "task03" / "centers.csv"
    if not center_file.exists():
        raise FileNotFoundError("先运行03_dc_locations.py，压力练习会复用其保存的中心。")
    base_centers = pd.read_csv(center_file)
    tradeoffs = []
    for k in [1, 2, 3, 4]:
        centers = base_centers[base_centers.k == k][["x_km", "y_km"]].to_numpy()
        for r in [100, 270, 540]:
            tradeoffs.append([k, r, *location_metrics(base[["x_km", "y_km"]].to_numpy(),
                                                     base.demand.to_numpy(), centers, r)])
    tradeoffs = pd.DataFrame(tradeoffs, columns=["k", "radius_km", "city_coverage",
                                               "demand_coverage", "weighted_distance_km",
                                               "weighted_squared_loss"])
    write_csv(tradeoffs, out / "base_dc_tradeoffs.csv")
    return scores, tradeoffs, points, weights, continuous


def customer_stress(out):
    rng = np.random.default_rng(1903)
    n = 90
    # 一团连续变化的客户，没有预先分出三个等级，K=3只是与基础例保持可比较。
    orders = rng.integers(5, 24, n)
    volume = np.maximum(1, orders * rng.lognormal(2.0, .5, n))
    recency = rng.integers(1, 31, n)
    initial = np.column_stack([orders, volume, recency]).astype(float)
    later = initial.copy()
    later[:, 0] = np.maximum(1, np.round(initial[:, 0] + rng.normal(0, 2, n)))
    later[:, 1] = np.maximum(1, initial[:, 1] * rng.lognormal(0, .12, n))
    later[:, 2] = np.clip(initial[:, 2] + rng.integers(-3, 4, n), 1, 30)
    changed = rng.choice(n, 30, replace=False)
    later[changed, 0] = rng.integers(2, 32, len(changed))
    later[changed, 1] = rng.uniform(10, 340, len(changed))
    later[changed, 2] = rng.integers(1, 31, len(changed))
    scaler = StandardScaler().fit(initial)
    initial_scaled, later_scaled = scaler.transform(initial), scaler.transform(later)
    fitted = KMeans(n_clusters=3, n_init=10, random_state=42).fit(initial_scaled)
    refit = KMeans(n_clusters=3, n_init=10, random_state=42).fit(later_scaled)
    fixed_threshold = np.quantile(initial[:, 1], [1 / 3, 2 / 3])
    first_labels, rows, metric_rows = {}, [], []
    seed_rows, seed_scores = [], []
    for seed in range(6):
        assigned = KMeans(n_clusters=3, n_init=1, random_state=seed).fit_predict(initial_scaled)
        seed_scores.append(float(adjusted_rand_score(fitted.labels_, assigned)))
        seed_rows.extend([seed, f"C{i:03}", int(label)] for i, label in enumerate(assigned))
    seed_ari = min(seed_scores)
    write_csv(pd.DataFrame(seed_rows, columns=["seed", "customer_id", "cluster"]),
              out / "customer_seed_assignments.csv")
    for period, features, scaled in [("first", initial, initial_scaled),
                                     ("later", later, later_scaled)]:
        labels = {
            "kmeans_fixed": fitted.predict(scaled),
            # 保持第一期的缩放器，只改变中心；避免把缩放器变化混进中心变化。
            "kmeans_refit": fitted.labels_ if period == "first" else refit.labels_,
            "volume_fixed": np.digitize(features[:, 1], fixed_threshold, right=True),
            "volume_refit": np.digitize(features[:, 1],
                                         np.quantile(features[:, 1], [1 / 3, 2 / 3]), right=True),
        }
        for method, assigned in labels.items():
            if period == "first":
                first_labels[method] = assigned
            metric_rows.append([period, method, 3, float(silhouette_score(scaled, assigned)),
                                float(adjusted_rand_score(first_labels[method], assigned)),
                                float(adjusted_rand_score(labels["volume_refit"], assigned))])
            rows.extend([period, f"C{i:03}", method, int(label)]
                        for i, label in enumerate(assigned))
    features = []
    for period, values in [("first", initial), ("later", later)]:
        features.extend([period, f"C{i:03}", *map(float, row), int(i in changed)]
                        for i, row in enumerate(values))
    write_csv(pd.DataFrame(features, columns=["period", "customer_id", "orders",
                                              "business_volume", "recency_days",
                                              "designed_large_change"]), out / "customer_features.csv")
    write_csv(pd.DataFrame(rows, columns=["period", "customer_id", "method", "cluster"]),
              out / "customer_assignments.csv")
    scores = pd.DataFrame(metric_rows, columns=["period", "method", "k", "silhouette",
                                               "ari_to_first_period", "ari_to_volume_refit"])
    write_csv(scores, out / "customer_metrics.csv")
    # 对账是独立的5行口径练习，不用于偷偷修改上面90位客户的特征。
    reconciliation = pd.DataFrame([
        ["R0", "Jan", 120, "kg", "Jan", 12, "10kg", "教学可换算", 10,
         "单位已知时120=12*10；转换另存字段，保留原值"],
        ["R1", "Jan", 190, "kg", "Jan", 180, "kg", "待核实", None,
         "同周期同单位仍差10；检查退货和截单，不能直接覆盖"],
        ["R2", "Jan", 75, "kg", "Jan", None, "kg", "待核实", None,
         "缺值不是0；记录缺失原因，暂不作为完整业务量"],
        ["R3", "Jan", 90, "kg", "Jan-Feb", 210, "kg", "待核实", None,
         "先取相同统计周期，不能把两个周期强行对齐"],
        ["R4", "Jan", 40, "kg", "Jan", 40, "kg", "教学一致", 1,
         "本行一致；仍保留两个来源及原始字段"],
    ], columns=["customer_id", "waybill_period", "waybill_quantity", "waybill_unit",
                "external_period", "external_quantity", "external_unit", "review_status",
                "known_unit_factor", "review_note"])
    write_csv(reconciliation, out / "reconciliation.csv")
    return scores, float(seed_ari), initial, later, changed


def main():
    out = output_dir("stress")
    forecasts = prediction_stress(out)
    warehouse = warehouse_stress(out)
    location, tradeoffs, points, weights, centers = location_stress(out)
    customers, seed_ari, initial, later, changed = customer_stress(out)
    original_best = forecasts.original.idxmin()
    changed_best = forecasts.trend_change.idxmin()
    km = location[location.method == "continuous_kmeans"].set_index("k")
    first = customers[(customers.period == "first") & (customers.method == "kmeans_fixed")].iloc[0]
    later_row = customers[(customers.period == "later") & (customers.method == "kmeans_fixed")].iloc[0]
    base = tradeoffs[tradeoffs.radius_km == 100].set_index("k")
    summary = {
        "data_kind": "synthetic_stress", "original_best_forecast": original_best,
        "changed_best_forecast": changed_best,
        "forecast_mean_mae": forecasts.to_dict(),
        "warehouse_walk_m": warehouse.total_walk_m.to_dict(),
        "kmeans_city_coverage": {str(k): float(km.loc[k, "city_coverage"]) for k in [2, 3]},
        "first_customer_silhouette": float(first.silhouette),
        "customer_fixed_model_time_ari": float(later_row.ari_to_first_period),
        "single_initialization_minimum_ari": seed_ari,
        "interpretation": "压力情景检验假设边界，不是正式比赛成绩或通用算法排名",
    }
    write_json(summary, out / "summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    forecasts.rename(columns={"original": "Original", "trend_change": "Trend change"}).plot(
        kind="bar", ax=axes[0, 0], color=["#218380", "#d88b47"], rot=15)
    axes[0, 0].set(title="A. Forecast assumption changed", ylabel="Mean MAE (units/day)", xlabel="")
    axes[0, 0].legend(fontsize=8)
    warehouse.reindex(["fixed_two_day", "daily_previous_day", "hold_two_observations"])[
        ["pick_walk_m", "move_walk_m"]].rename(
        columns={"pick_walk_m": "Picking", "move_walk_m": "Relocation"}).plot(
            kind="bar", stacked=True, ax=axes[0, 1], color=["#218380", "#d88b47"], rot=15)
    axes[0, 1].set(title="B. Previous-day relocation", ylabel="Toy walk distance (m)", xlabel="")
    axes[0, 1].set_xticklabels(["Fixed", "Daily", "Hold"], rotation=0)
    axes[0, 1].legend(fontsize=8)
    ax = axes[1, 0]
    for row, k in enumerate([2, 3]):
        nearest = np.abs(points[:, 0, None] - centers[k][:, 0]).min(axis=1)
        colors = np.where(nearest <= 5, "#218380", "#909b9f")
        ax.scatter(points[:, 0], np.repeat(row, len(points)), c=colors, s=weights, alpha=.8)
        ax.scatter(centers[k][:, 0], np.repeat(row, k), marker="X", s=100,
                   color="#d88b47", label="Centers" if row == 0 else None)
        for center in centers[k][:, 0]:
            ax.plot([center - 5, center + 5], [row - .15, row - .15], color="#218380")
    ax.set(yticks=[0, 1], yticklabels=["k=2", "k=3"], xlabel="x (fictional km)",
           title="C. Lower squared loss, fewer cities covered", ylim=(-.4, 1.4))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.scatter(initial[:, 0], initial[:, 1], s=18, color="#218380", alpha=.55, label="First period")
    ax.scatter(later[changed, 0], later[changed, 1], s=25, color="#d88b47", marker="x",
               label="Changed customers later")
    ax.set(title="D. Continuous customer cloud and drift", xlabel="Orders", ylabel="Volume")
    ax.legend(fontsize=8)
    save_figure(fig, out / "stress_overview.svg")
    text = "# 四个方法压力练习\n\n先看发生了什么，再决定是否需要换方法。全部为合成教学情景。\n\n"
    text += "![四个压力情景](stress_overview.svg)\n\n"
    text += "## 1. 预测：适合原机制的模型，换机制后会怎样？\n\n"
    text += "| 方法 | 原机制平均MAE | 趋势突变平均MAE |\n| --- | ---: | ---: |\n"
    for name, row in forecasts.iterrows():
        text += f"| {name} | {row.original:.4f} | {row.trend_change:.4f} |\n"
    text += (f"\n前56天相同，后56天改成下降趋势。原机制最好的是{original_best}，"
             f"新机制最好的是{changed_best}。这说明日历线性趋势的适用假设出了问题，"
             "并不证明某个基线总会赢。先查看哪个窗口开始恶化，再考虑短训练窗、变化检测或更新频率；"
             "新增方案仍应在未调参的窗口比较。\n\n"
             "重算：[两种输入](prediction_input.csv) → [逐日预测](prediction_predictions.csv) → "
             "[逐窗口MAE](prediction_metrics.csv)。MAE是每窗口14个绝对误差的平均，不是除以10。\n\n")
    text += "## 2. 仓储：每天发现波动，不等于每天搬货\n\n"
    text += "| 布局策略 | 拣选行走米数 | 调位行走米数 | 合计米数 |\n| --- | ---: | ---: | ---: |\n"
    for name, row in warehouse.iterrows():
        text += f"| {name} | {row.pick_walk_m:.0f} | {row.move_walk_m:.0f} | {row.total_walk_m:.0f} |\n"
    text += ("\n前2天用于初始观察，比较第3~8天。只有1个近位（2米），其他位置10米，"
             "每件商品单独往返，换近位额外行走60米。每日策略用昨天最高IQ选今天近位，"
             "持有策略需连续两次观察指向同一SKU才换。A/B隔日反转，因此追昨天反而放错近位。"
             "这是上述玩具流程的真实计算；真实仓库需改用订单拣选、货位容量与搬运工时。"
             "ABC仍按天做诊断，布局应另设观察周期和调整条件，不能从每日类别直接下搬货指令。\n\n"
             "重算：[场景参数](warehouse_parameters.json)、[每日数量及ABC](warehouse_demand.csv)、"
             "[每次决策与代价](warehouse_operations.csv)。\n\n")
    text += "## 3. 选址：平方距离目标与覆盖目标会冲突\n\n"
    text += "| 连续坐标K-means | 城市覆盖率（5km） | 加权平均距离km | 加权平方损失 |\n| --- | ---: | ---: | ---: |\n"
    for k, row in km.iterrows():
        text += f"| k={k} | {row.city_coverage:.1%} | {row.weighted_distance_km:.4f} | {row.weighted_squared_loss:.2f} |\n"
    text += ("\n9个固定一维点，各方法使用同一组需求权重。增加中心后平方损失下降，"
             "城市覆盖却下降；枚举所有一维连续分段得到相同最优平方损失，因此不能仅归因于初始化。\n\n"
             "进一步比较时，先把K-means映射到9个城市候选点（不重复），再与同样候选点、同K、同半径的"
             "穷举最大城市覆盖比较。另列候选点平方损失最优解，以分开目标差异和搜索误差。"
             "连续坐标原始K-means与离散候选覆盖法约束不同，不能直接用分数判谁更强。"
             "最大覆盖法穷举每个组合，以城市覆盖优先，平局依次看需求覆盖和距离。"
             "在允许保留旧中心并增加新中心、没有额外冲突约束时，最优覆盖随K不降；"
             "这个性质不适用于每次重拟合的K-means覆盖率。\n\n")
    text += (f"基础12城、100km下：2中心覆盖{base.loc[2, 'city_coverage']:.1%}、"
             f"加权平均距离{base.loc[2, 'weighted_distance_km']:.2f}km；3中心覆盖"
             f"{base.loc[3, 'city_coverage']:.1%}、距离{base.loc[3, 'weighted_distance_km']:.2f}km。"
             f"多建一个中心仅在本例减少{base.loc[2, 'weighted_distance_km'] - base.loc[3, 'weighted_distance_km']:.2f}km"
             "平均距离，不知道建站及运输成本就不能判断值不值。若目标只是该半径全覆盖，"
             "这次保存的2中心方案已满足，但这不是对所有连续可行方案的最少站数证明。\n\n"
             "每日约270km、两日的背景另作两种代理：全部路程用来单程行驶时540km；"
             "两日还需完成往返时，单程270km。两者都忽略装卸、多点配送和路网，不能作为两日送达保证。"
             "基础12城在270/540km下均已全覆盖，阈值在这组小图上失去区分力，"
             "提示正式数据应先核验道路距离和服务模型，而不是追求漂亮覆盖差异。\n\n"
             "重算：[点与需求](dc_points.csv)、[中心](dc_centers.csv)、[同约束比较](dc_evaluation.csv)、"
             "[基础12城代理情景](base_dc_tradeoffs.csv)。\n\n")
    text += "## 4. 客户：三个簇不等于三个天然等级\n\n"
    text += (f"90位客户由连续云团生成。第一期K=3轮廓系数{first.silhouette:.4f}；"
             f"只跑一次初始化的6个种子与默认分组最小ARI为{seed_ari:.4f}。"
             f"第二期30位客户变化较大，固定第一期模型前后分组ARI为{later_row.ari_to_first_period:.4f}。"
             "这三个指标分别谈形状、初始化和时间变化，不能互相替代。单次初始化与基础例的10次初始化设置不同；"
             "低跨期ARI也可能反映真实客户变化，先查看哪些客户改变及原因。\n\n"
             "列出固定模型、重拟合中心、固定业务量阈值、重算分位阈值四种路线。"
             "重拟合K-means仍用第一期缩放器；否则缩放变化也会混进比较。"
             "若客户服务需要稳定等级，要研究调整频率和变化原因，不能仅选择当期轮廓系数最高的方法。"
             "简单业务量规则也可作为可解释起点，方法复杂程度不能替代业务检验。\n\n"
             "另有5行独立口径对账练习：单位换算、缺值、同口径差额、统计周期不同、一致记录。"
             "[对账记录](reconciliation.csv)保留原值，3行待核实；不自动把差额抹平，"
             "也不把这个独立例子偷偷拼进90位客户的特征。\n\n"
             "重算：[两期特征](customer_features.csv)、[逐客户分组](customer_assignments.csv)、"
             "[比较指标](customer_metrics.csv)、[单次初始化分组](customer_seed_assignments.csv)。\n\n"
             "## 下一步怎样使用\n\n"
             "选一条失败现象，先画出原因，提出一个可以反驳的改进，再留出新窗口比较。"
             "本练习展示的是路线改变的依据；正式数据的机制、成本和授权需分别核实。"
             "[汇总数字](summary.json)便于检查，逐条CSV才是重算入口。\n")
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print("压力练习完成：预测机制反转、仓储次日调位、选址目标冲突、客户弱结构与漂移。")


if __name__ == "__main__":
    main()
