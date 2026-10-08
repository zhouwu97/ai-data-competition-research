"""从原始输入和逐条结果独立重算四个教学实验，不导入建模脚本。

运行：python scripts/check_calculations.py
只用 Python 标准库；CSV 保存六位小数，所以比较允许小量舍入误差。
"""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
import json
import math
from pathlib import Path
import sys


ABS_TOL = 2e-6
REL_TOL = 1e-7


class CalculationError(ValueError):
    """结果与独立计算不一致，错误信息说明需要查看的文件或字段。"""


def close(actual, expected, name):
    actual, expected = float(actual), float(expected)
    if not math.isfinite(actual) or not math.isclose(
            actual, expected, abs_tol=ABS_TOL, rel_tol=REL_TOL):
        raise CalculationError(f"{name}: 文件值 {actual:.9g}，独立重算 {expected:.9g}")


def require(condition, message):
    if not condition:
        raise CalculationError(message)


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        result = list(csv.DictReader(handle))
    require(bool(result), f"{path}: 没有数据行")
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unique_index(records, fields, name):
    result = {}
    for record in records:
        key = tuple(record[field] for field in fields)
        require(key not in result, f"{name}: 重复键 {key}")
        result[key] = record
    return result


def error_metrics(actual, predicted):
    """MAE 是绝对误差平均；RMSE 是平方误差平均后开根号。"""
    require(len(actual) == len(predicted) and len(actual) > 0, "误差输入长度不符")
    differences = [float(a) - float(p) for a, p in zip(actual, predicted)]
    require(all(math.isfinite(x) for x in differences), "误差输入包含非有限数")
    return (sum(abs(x) for x in differences) / len(differences),
            math.sqrt(sum(x * x for x in differences) / len(differences)))


def daily_eiq(raw):
    """用字典和集合重算：多行同一订单-SKU求和，EN/IK计不同编号。"""
    lines = defaultdict(float)
    for record in raw:
        qty = float(record["quantity"])
        require(qty > 0, "EIQ 本练习要求出库数量为正")
        lines[(record["date"], record["order_id"], record["sku"])] += qty
    orders, items = {}, {}
    for (day, order, sku), qty in lines.items():
        orders.setdefault((day, order), {"EQ": 0.0, "sku_set": set()})
        items.setdefault((day, sku), {"IQ": 0.0, "order_set": set()})
        orders[(day, order)]["EQ"] += qty
        orders[(day, order)]["sku_set"].add(sku)
        items[(day, sku)]["IQ"] += qty
        items[(day, sku)]["order_set"].add(order)
    return ({key: {"EQ": value["EQ"], "EN": len(value["sku_set"])}
             for key, value in orders.items()},
            {key: {"IQ": value["IQ"], "IK": len(value["order_set"])}
             for key, value in items.items()}, len(lines))


def abc_classification(quantities, a_limit=.80, b_limit=.95):
    """quantities 已按数量降序、SKU升序排列；跨阈值品项仍归前一类。"""
    total = sum(quantities)
    require(total > 0 and all(q >= 0 for q in quantities), "ABC 数量非法")
    cumulative, labels, shares = 0.0, [], []
    for quantity in quantities:
        before = cumulative / total
        labels.append("ZERO" if quantity == 0 else
                      "A" if before < a_limit else "B" if before < b_limit else "C")
        cumulative += quantity
        shares.append(cumulative / total)
    return labels, shares


def location_metrics(cities, centers, radius):
    """独立遍历所有中心，计算最近距离、两种覆盖率和需求加权距离。"""
    require(bool(cities) and bool(centers), "选址输入不能为空")
    allocations = {}
    total_demand, covered_demand, covered_cities, weighted_distance = 0., 0., 0, 0.
    for city in cities:
        weight = float(city["demand"])
        require(weight > 0, "城市需求权重必须为正")
        distances = [(math.hypot(float(city["x_km"]) - float(center["x_km"]),
                                 float(city["y_km"]) - float(center["y_km"])),
                      int(center["cluster"])) for center in centers]
        distance, cluster = min(distances)
        allocations[city["city_id"]] = (cluster, distance)
        total_demand += weight
        weighted_distance += weight * distance
        if distance <= radius:
            covered_cities += 1
            covered_demand += weight
    return allocations, {"city_coverage": covered_cities / len(cities),
                         "demand_coverage": covered_demand / total_demand,
                         "weighted_distance_km": weighted_distance / total_demand}


def standardized(points, reference=None):
    """按列减均值、除总体标准差；reference可固定为第一期客户。"""
    require(bool(points) and bool(points[0]), "特征不能为空")
    reference = points if reference is None else reference
    dimensions = len(points[0])
    require(all(len(row) == dimensions for row in points + reference), "特征列数不一致")
    means = [sum(row[j] for row in reference) / len(reference) for j in range(dimensions)]
    scales = [math.sqrt(sum((row[j] - means[j]) ** 2 for row in reference) / len(reference))
              for j in range(dimensions)]
    return [[(row[j] - means[j]) / (scales[j] or 1.) for j in range(dimensions)]
            for row in points]


def silhouette(points, labels):
    """直接按两两欧氏距离求 a、b 和 (b-a)/max(a,b)；单样本群记0。"""
    require(len(points) == len(labels), "轮廓系数输入长度不符")
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    require(2 <= len(groups) < len(points), "轮廓系数需要2至n-1个群")
    values = []
    for i, point in enumerate(points):
        own = groups[labels[i]]
        if len(own) == 1:
            values.append(0.)
            continue
        distances = [math.dist(point, other) for other in points]
        a = sum(distances[j] for j in own if j != i) / (len(own) - 1)
        b = min(sum(distances[j] for j in members) / len(members)
                for label, members in groups.items() if label != labels[i])
        values.append((b - a) / max(a, b) if max(a, b) else 0.)
    return sum(values) / len(values)


def adjusted_rand(labels_a, labels_b):
    """由列联表中的样本对数量重算ARI；不使用群编号的大小。"""
    require(len(labels_a) == len(labels_b), "ARI 输入长度不符")
    n = len(labels_a)
    if n < 2:
        return 1.
    choose2 = lambda x: x * (x - 1) / 2
    both = sum(choose2(x) for x in Counter(zip(labels_a, labels_b)).values())
    side_a = sum(choose2(x) for x in Counter(labels_a).values())
    side_b = sum(choose2(x) for x in Counter(labels_b).values())
    expected = side_a * side_b / choose2(n)
    denominator = (side_a + side_b) / 2 - expected
    return (both - expected) / denominator if denominator else 1.


def quantile(values, fraction):
    """线性插值分位点；与本例三等分规则的定义一致。"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def check_forecast(root):
    inputs, output = root / "examples/sample_inputs", root / "examples/results/task01"
    totals = defaultdict(float)
    for row in rows(inputs / "demand.csv"):
        totals[row["date"]] += float(row["quantity"])
    summary = read_json(output / "summary.json")
    horizon = int(summary["horizon_days"])
    dated = sorted(totals)
    grouped = defaultdict(list)
    predicted = rows(output / "predictions.csv")
    unique_index(predicted, ["cutoff_day", "date", "model"], "predictions.csv")
    for row in predicted:
        close(row["actual"], totals[row["date"]], f"predictions.csv {row['date']} actual")
        grouped[(row["cutoff_day"], row["model"])].append(row)
    recorded = unique_index(rows(output / "metrics.csv"), ["cutoff_day", "model"], "metrics.csv")
    require(set(grouped) == set(recorded), "metrics.csv 与预测的窗口/方法集合不一致")
    by_model, recomputed = defaultdict(list), {}
    for key, records in grouped.items():
        cutoff = int(key[0])
        require(sorted(row["date"] for row in records) == dated[cutoff:cutoff + horizon]
                and len(records) == horizon, f"预测窗口 {key} 的日期或长度不符")
        mae, rmse = error_metrics([row["actual"] for row in records],
                                  [row["prediction"] for row in records])
        close(recorded[key]["mae"], mae, f"metrics.csv {key} MAE")
        require(recorded[key]["valid_start"] == dated[cutoff], f"metrics.csv {key} 起始日期不符")
        if "rmse" in recorded[key]:
            close(recorded[key]["rmse"], rmse, f"metrics.csv {key} RMSE")
        by_model[key[1]].append((mae, rmse))
    require(len({key[0] for key in grouped}) == int(summary["windows"]), "预测窗口数不符")
    for model, values in by_model.items():
        mae = sum(value[0] for value in values) / len(values)
        rmse = sum(value[1] for value in values) / len(values)
        close(summary["mean_mae"][model], mae, f"summary.json {model} 平均MAE")
        if "mean_rmse" in summary:
            close(summary["mean_rmse"][model], rmse, f"summary.json {model} 平均RMSE")
        recomputed[model] = (mae, rmse)
    verified = "MAE、RMSE已核对" if all("rmse" in row for row in recorded.values()) else "MAE已核对"
    return f"题一：{len(grouped)}组窗口×方法的{verified}；独立计算的平均RMSE为 " + "; ".join(
        f"{model}={values[1]:.4f}" for model, values in sorted(recomputed.items()))


def check_warehouse(root):
    raw = rows(root / "examples/sample_inputs/warehouse_orders.csv")
    output = root / "examples/results/task02"
    orders, items, line_count = daily_eiq(raw)
    recorded_orders = unique_index(rows(output / "order_eiq.csv"), ["date", "order_id"], "order_eiq.csv")
    recorded_items = unique_index(rows(output / "item_eiq_abc.csv"), ["date", "sku"], "item_eiq_abc.csv")
    require(set(orders) == set(recorded_orders), "订单EIQ的日期-订单键不一致")
    require(set(items) == set(recorded_items), "商品EIQ的日期-SKU键不一致")
    for key, values in orders.items():
        for field, value in values.items():
            close(recorded_orders[key][field], value, f"order_eiq.csv {key} {field}")
    for key, values in items.items():
        for field, value in values.items():
            close(recorded_items[key][field], value, f"item_eiq_abc.csv {key} {field}")
    compared = unique_index(rows(output / "threshold_comparison.csv"), ["date", "sku"], "threshold_comparison.csv")
    require(set(compared) == set(items), "ABC阈值比较的日期-SKU键不一致")
    changed = 0
    for day in {key[0] for key in items}:
        keys = sorted((key for key in items if key[0] == day), key=lambda key: (-items[key]["IQ"], key[1]))
        quantities = [items[key]["IQ"] for key in keys]
        labels, shares = abc_classification(quantities)
        alternative, _ = abc_classification(quantities, .70, .90)
        for key, label, other, share in zip(keys, labels, alternative, shares):
            require(recorded_items[key]["ABC"] == label, f"item_eiq_abc.csv {key} ABC应为{label}")
            close(recorded_items[key]["cum_share"], share, f"item_eiq_abc.csv {key} 累计占比")
            require(compared[key]["ABC"] == label and compared[key]["ABC_70_90"] == other,
                    f"threshold_comparison.csv {key} ABC阈值结果不符")
            for field in ["IQ", "IK"]:
                close(compared[key][field], items[key][field], f"threshold_comparison.csv {key} {field}")
            changed += label != other
    summary = read_json(output / "summary.json")
    expected = {"days": len({key[0] for key in items}), "source_lines": len(raw),
                "normalized_order_sku_lines": line_count,
                "total_quantity": sum(float(row["quantity"]) for row in raw),
                "threshold_changed_rows": changed}
    for field, value in expected.items():
        close(summary[field], value, f"task02 summary.json {field}")
    return "题二：从原始订单重算每日EQ/EN/IQ/IK、两组ABC边界及数量汇总。"


def check_locations(root):
    cities = rows(root / "examples/sample_inputs/cities.csv")
    unique_index(cities, ["city_id"], "cities.csv")
    output = root / "examples/results/task03"
    center_rows = rows(output / "centers.csv")
    unique_index(center_rows, ["k", "cluster"], "centers.csv")
    grouped = defaultdict(list)
    for center in center_rows:
        grouped[center["k"]].append(center)
    allocated = unique_index(rows(output / "allocations.csv"), ["k", "city_id"], "allocations.csv")
    scores = unique_index(rows(output / "coverage.csv"), ["k", "radius_km"], "coverage.csv")
    expected_keys = {(k, city["city_id"]) for k in grouped for city in cities}
    require(set(allocated) == expected_keys, "allocations.csv 城市/中心数量集合不符")
    require({key[0] for key in scores} == set(grouped), "coverage.csv 与 centers.csv 的k集合不符")
    rechecked = {}
    for (k, radius), score in scores.items():
        centers = sorted(grouped[k], key=lambda row: int(row["cluster"]))
        require(len(centers) == int(k), f"centers.csv k={k}中心数量不符")
        nearest, metrics = location_metrics(cities, centers, float(radius))
        for city, (label, distance) in nearest.items():
            record = allocated[(k, city)]
            close(record["distance_km"], distance, f"allocations.csv k={k} {city} 最近距离")
            # 距离并列时允许分给任一最近中心，不强制相同编号。
            chosen = next((center for center in centers if int(center["cluster"]) == int(record["cluster"])), None)
            require(chosen is not None, f"allocations.csv k={k} {city} 中心编号不存在")
            raw_city = next(row for row in cities if row["city_id"] == city)
            chosen_distance = math.hypot(float(raw_city["x_km"]) - float(chosen["x_km"]),
                                         float(raw_city["y_km"]) - float(chosen["y_km"]))
            close(chosen_distance, distance, f"allocations.csv k={k} {city} 分配非最近中心")
        for field, value in metrics.items():
            close(score[field], value, f"coverage.csv k={k} 半径{radius} {field}")
        rechecked[(int(k), float(radius))] = metrics
    summary = read_json(output / "summary.json")
    key = (int(summary["illustration_k"]), float(summary["illustration_radius_km"]))
    require(key in rechecked, "选址汇总的演示情景不存在")
    for field, value in rechecked[key].items():
        close(summary[field], value, f"task03 summary.json {field}")
    return "题三：独立重算已存中心的最近分配、城市/需求覆盖率及加权距离。"


def check_customers(root):
    inputs, output = root / "examples/sample_inputs", root / "examples/results/task04"
    summary = read_json(output / "summary.json")
    cutoff = date.fromisoformat(summary["cutoff"])
    raw = rows(inputs / "waybills.csv")
    unique_index(raw, ["waybill_id"], "waybills.csv")
    businesses = unique_index(rows(inputs / "business_volume.csv"), ["customer_id"], "business_volume.csv")
    customers = unique_index(rows(output / "customer_features.csv"), ["customer_id"], "customer_features.csv")
    aggregated = defaultdict(lambda: {"orders": 0, "summed_quantity": 0., "last_date": date.min})
    for record in raw:
        day = date.fromisoformat(record["date"])
        require(day < cutoff, "运单包含观察时点之后的日期")
        value = aggregated[(record["customer_id"],)]
        value["orders"] += 1
        value["summed_quantity"] += float(record["quantity"])
        value["last_date"] = max(day, value["last_date"])
    require(set(customers) == set(aggregated) == set(businesses), "客户集合不一致")
    for key, value in aggregated.items():
        expected = {"orders": value["orders"], "summed_quantity": value["summed_quantity"],
                    "business_volume": float(businesses[key]["business_volume"]),
                    "recency_days": (cutoff - value["last_date"]).days}
        for field, number in expected.items():
            close(customers[key][field], number, f"customer_features.csv {key} {field}")
    records = list(customers.values())
    volumes = [float(row["business_volume"]) for row in records]
    q1, q2 = quantile(volumes, 1 / 3), quantile(volumes, 2 / 3)
    rule = [0 if volume <= q1 else 1 if volume <= q2 else 2 for volume in volumes]
    for row, label in zip(records, rule):
        require(int(row["volume_rule"]) == label, f"客户 {row['customer_id']} 业务量分档不符")
    features = summary["features"]
    require(features == ["orders", "business_volume", "recency_days"],
            "本例核对的特征应为orders、business_volume、recency_days")
    points = standardized([[float(row[field]) for field in features] for row in records])
    labels = [int(row["cluster"]) for row in records]
    require(len(set(labels)) == int(summary["illustration_k"]), "客户演示群数量不符")
    score = silhouette(points, labels)
    rule_score = silhouette(points, rule)
    ari = adjusted_rand(labels, rule)
    close(summary["customers"], len(records), "task04 summary.json 客户数")
    close(summary["silhouette_k3"], score, "task04 summary.json 轮廓系数")
    close(summary["ari_to_volume_rule"], ari, "task04 summary.json ARI")
    comparison = unique_index(rows(output / "comparison.csv"), ["method", "k"], "comparison.csv")
    close(comparison[("kmeans", str(summary["illustration_k"]))]["silhouette"], score,
          "comparison.csv 演示k轮廓系数")
    close(comparison[("volume_rule", "3")]["silhouette"], rule_score, "comparison.csv 规则分组轮廓系数")
    profiles = unique_index(rows(output / "profiles.csv"), ["cluster"], "profiles.csv")
    require(set(profiles) == {(str(label),) for label in labels}, "画像群编号集合不符")
    for key, profile in profiles.items():
        members = [row for row in records if row["cluster"] == key[0]]
        close(profile["customer_count"], len(members), f"profiles.csv 群{key[0]}人数")
        for field, source in [("mean_orders", "orders"), ("mean_volume", "business_volume"),
                              ("mean_recency", "recency_days")]:
            close(profile[field], sum(float(row[source]) for row in members) / len(members),
                  f"profiles.csv 群{key[0]} {field}")
    return "题四：从运单核对客户特征；独立重算演示k与规则的轮廓系数、ARI及画像。"


def check_stress_forecast(root, output, summary):
    source = unique_index(rows(output / "prediction_input.csv"), ["scenario", "date"],
                          "stress prediction_input.csv")
    scenarios = {key[0] for key in source}
    require(scenarios == {"original", "trend_change"}, "压力预测应含原机制与趋势变化机制")
    days = sorted(key[1] for key in source if key[0] == "original")
    require(days == sorted(key[1] for key in source if key[0] == "trend_change"),
            "两机制的日期集合不一致")
    original = defaultdict(float)
    for record in rows(root / "examples/sample_inputs/demand.csv"):
        original[record["date"]] += float(record["quantity"])
    require(days == sorted(original), "压力原机制的日期与基础输入不一致")
    for day in days:
        close(source[("original", day)]["quantity"], original[day], f"压力原机制 {day} 数量")
    for day in days[:56]:
        close(source[("trend_change", day)]["quantity"], original[day], f"压力前56天 {day} 数量")
    predicted = rows(output / "prediction_predictions.csv")
    unique_index(predicted, ["scenario", "cutoff_day", "date", "model"], "压力逐日预测")
    grouped = defaultdict(list)
    for record in predicted:
        key = (record["scenario"], record["cutoff_day"], record["model"])
        grouped[key].append(record)
        close(record["actual"], source[(record["scenario"], record["date"])]["quantity"],
              f"压力预测 {key} {record['date']} actual")
    recorded = unique_index(rows(output / "prediction_metrics.csv"),
                            ["scenario", "cutoff_day", "model"], "压力预测指标")
    require(set(grouped) == set(recorded), "压力预测的逐条记录与指标键不一致")
    expected = {(scenario, str(cutoff), model) for scenario in scenarios
                for cutoff in [56, 70, 84, 98]
                for model in ["recent_mean", "weekday_mean", "calendar_ridge"]}
    require(set(grouped) == expected, "压力预测缺少相同预算下的窗口或方法")
    means = defaultdict(list)
    for key, records in grouped.items():
        cutoff = int(key[1])
        require(sorted(record["date"] for record in records) == days[cutoff:cutoff + 14],
                f"压力预测 {key} 的14天日期不符")
        mae, _ = error_metrics([record["actual"] for record in records],
                               [record["prediction"] for record in records])
        close(recorded[key]["mae"], mae, f"stress prediction_metrics.csv {key} MAE")
        means[(key[0], key[2])].append(mae)
    for scenario in scenarios:
        by_model = {model: sum(values) / len(values) for (mechanism, model), values in means.items()
                    if mechanism == scenario}
        for model, value in by_model.items():
            close(summary["forecast_mean_mae"][scenario][model], value,
                  f"压力汇总 {scenario} {model} 平均MAE")
        field = "original_best_forecast" if scenario == "original" else "changed_best_forecast"
        require(summary[field] == min(by_model, key=by_model.get), f"压力汇总 {field} 与误差排名不符")


def check_stress_warehouse(output, summary):
    parameters = read_json(output / "warehouse_parameters.json")
    require(parameters["pick_rule"] == "one_unit_one_round_trip", "压力仓储拣选口径未知")
    near, far, moving = (float(parameters[key]) for key in
                         ["near_distance_m", "far_distance_m", "swap_distance_m"])
    require(0 <= near <= far and moving >= 0, "压力仓储距离参数非法")
    demand = unique_index(rows(output / "warehouse_demand.csv"), ["day", "sku"], "压力仓储数量")
    by_day = defaultdict(dict)
    for (day, sku), record in demand.items():
        by_day[int(day)][sku] = float(record["quantity"])
    for day, quantities in by_day.items():
        keys = sorted(quantities, key=lambda sku: (-quantities[sku], sku))
        labels, _ = abc_classification([quantities[sku] for sku in keys])
        for sku, label in zip(keys, labels):
            require(demand[(str(day), sku)]["ABC"] == label, f"压力仓储 第{day}天 {sku} ABC不符")
    initial_quantities = defaultdict(float)
    for day in parameters["observation_days"]:
        for sku, quantity in by_day[int(day)].items():
            initial_quantities[sku] += quantity
    initial = min(initial_quantities, key=lambda sku: (-initial_quantities[sku], sku))
    require(initial == parameters["initial_near_sku"], "初始近位与观察期数量/平局规则不符")
    operations = unique_index(rows(output / "warehouse_operations.csv"), ["strategy", "day"],
                              "压力仓储操作")
    strategies = ["fixed_two_day", "daily_previous_day", "hold_two_observations"]
    days = list(map(int, parameters["evaluation_days"]))
    require(days == sorted(days) and len(days) == len(set(days)), "仓储评价日期重复或未排序")
    require(set(operations) == {(strategy, str(day)) for strategy in strategies for day in days},
            "压力仓储策略/评价日期集合不一致")
    for strategy in strategies:
        current, total = initial, 0.
        for day in days:
            operation = operations[(strategy, str(day))]
            previous = min(by_day[day - 1], key=lambda sku: (-by_day[day - 1][sku], sku))
            before = min(by_day[day - 2], key=lambda sku: (-by_day[day - 2][sku], sku))
            chosen = (previous if strategy == "daily_previous_day" or
                      strategy == "hold_two_observations" and previous == before else current)
            require(int(operation["last_observed_day"]) == day - 1, "仓储操作的信息时点不符")
            require(operation["near_sku"] == chosen, f"压力仓储 {strategy} 第{day}天近位决策不符")
            picking = sum(2 * quantity * (near if sku == chosen else far)
                          for sku, quantity in by_day[day].items())
            relocation = moving if chosen != current else 0.
            for field, value in [("pick_walk_m", picking), ("move_walk_m", relocation),
                                  ("total_walk_m", picking + relocation)]:
                close(operation[field], value, f"压力仓储 {strategy} 第{day}天 {field}")
            total += picking + relocation
            current = chosen
        close(summary["warehouse_walk_m"][strategy], total, f"压力仓储 {strategy} 合计距离")


def extended_location_metrics(cities, centers, radius):
    allocations, metrics = location_metrics(cities, centers, radius)
    metrics["weighted_squared_loss"] = sum(float(city["demand"]) * allocations[city["city_id"]][1] ** 2
                                           for city in cities)
    return metrics


def check_stress_locations(root, output, summary):
    cities = rows(output / "dc_points.csv")
    unique_index(cities, ["city_id"], "压力选址点")
    center_rows = unique_index(rows(output / "dc_centers.csv"), ["method", "k", "center_id"], "压力选址中心")
    groups = defaultdict(list)
    for (method, k, center_id), record in center_rows.items():
        groups[(method, k)].append({**record, "cluster": center_id})
    scores = unique_index(rows(output / "dc_evaluation.csv"), ["method", "k"], "压力选址指标")
    require(set(groups) == set(scores), "压力选址的中心与指标键不一致")
    recomputed = {}
    for key, record in scores.items():
        centers = groups[key]
        require(len(centers) == int(key[1]), f"压力选址 {key} 中心数量不符")
        radius = float(record["radius_km"])
        metric = extended_location_metrics(cities, centers, radius)
        for field, value in metric.items():
            close(record[field], value, f"stress dc_evaluation.csv {key} {field}")
        recomputed[key] = metric
        if record["constraint"] == "nine_city_candidates":
            require(len({(center["x_km"], center["y_km"]) for center in centers}) == len(centers),
                    f"候选选址 {key} 有重复中心")
            for center in centers:
                require(any(math.isclose(float(center["x_km"]), float(city["x_km"]), abs_tol=ABS_TOL)
                            and math.isclose(float(center["y_km"]), float(city["y_km"]), abs_tol=ABS_TOL)
                            for city in cities), f"候选选址 {key} 中心未落在给定候选点")
        if key[0] in ["candidate_max_city_cover_exact", "candidate_min_squared_exact"]:
            candidates = []
            for selected in combinations(cities, int(key[1])):
                candidate_centers = [{**city, "cluster": i} for i, city in enumerate(selected)]
                candidates.append(extended_location_metrics(cities, candidate_centers, radius))
            close(record["enumerated_solutions"], len(candidates), f"压力选址 {key} 枚举数量")
            if key[0] == "candidate_max_city_cover_exact":
                best = min(candidates, key=lambda value: (-value["city_coverage"],
                            -value["demand_coverage"], value["weighted_distance_km"]))
                for field in ["city_coverage", "demand_coverage", "weighted_distance_km"]:
                    close(metric[field], best[field], f"压力选址 {key} 候选最优 {field}")
            else:
                close(metric["weighted_squared_loss"], min(value["weighted_squared_loss"] for value in candidates),
                      f"压力选址 {key} 候选最优平方损失")
    for k in ["2", "3"]:
        close(summary["kmeans_city_coverage"][k], recomputed[("continuous_kmeans", k)]["city_coverage"],
              f"压力汇总 K={k} 城市覆盖")
    base_cities = rows(root / "examples/sample_inputs/cities.csv")
    base_centers = rows(root / "examples/results/task03/centers.csv")
    tradeoffs = unique_index(rows(output / "base_dc_tradeoffs.csv"), ["k", "radius_km"], "固定中心代理情景")
    for (k, radius), record in tradeoffs.items():
        centers = [center for center in base_centers if center["k"] == k]
        for field, value in extended_location_metrics(base_cities, centers, float(radius)).items():
            close(record[field], value, f"base_dc_tradeoffs.csv K={k} 半径{radius} {field}")


def check_stress_customers(output, summary):
    features = unique_index(rows(output / "customer_features.csv"), ["period", "customer_id"], "压力客户特征")
    ids = sorted(key[1] for key in features if key[0] == "first")
    require(ids == sorted(key[1] for key in features if key[0] == "later"), "压力客户两期ID不一致")
    fields = ["orders", "business_volume", "recency_days"]
    points = {period: [[float(features[(period, customer)][field]) for field in fields] for customer in ids]
              for period in ["first", "later"]}
    scaled = {period: standardized(value, points["first"]) for period, value in points.items()}
    records = unique_index(rows(output / "customer_assignments.csv"), ["period", "customer_id", "method"],
                           "压力客户分组")
    methods = ["kmeans_fixed", "kmeans_refit", "volume_fixed", "volume_refit"]
    require(set(records) == {(period, customer, method) for period in points for customer in ids for method in methods},
            "压力客户分组缺少方法、客户或时期")
    labels = {(period, method): [int(records[(period, customer, method)]["cluster"]) for customer in ids]
              for period in points for method in methods}
    fixed_thresholds = [quantile([row[1] for row in points["first"]], fraction) for fraction in [1 / 3, 2 / 3]]
    for period in points:
        dynamic_thresholds = [quantile([row[1] for row in points[period]], fraction) for fraction in [1 / 3, 2 / 3]]
        for method, thresholds in [("volume_fixed", fixed_thresholds), ("volume_refit", dynamic_thresholds)]:
            rule = [0 if row[1] <= thresholds[0] else 1 if row[1] <= thresholds[1] else 2 for row in points[period]]
            require(labels[(period, method)] == rule, f"压力客户 {period} {method} 分位分档不符")
    recorded = unique_index(rows(output / "customer_metrics.csv"), ["period", "method"], "压力客户指标")
    require(set(recorded) == set(labels), "压力客户的指标与分组键不一致")
    for key, assigned in labels.items():
        period, method = key
        metric = {"silhouette": silhouette(scaled[period], assigned),
                  "ari_to_first_period": adjusted_rand(labels[("first", method)], assigned),
                  "ari_to_volume_refit": adjusted_rand(labels[(period, "volume_refit")], assigned)}
        close(recorded[key]["k"], len(set(assigned)), f"压力客户 {key} 群数量")
        for field, value in metric.items():
            close(recorded[key][field], value, f"stress customer_metrics.csv {key} {field}")
    seeds = unique_index(rows(output / "customer_seed_assignments.csv"), ["seed", "customer_id"], "压力客户初始化")
    seed_ids = {key[0] for key in seeds}
    require(set(seeds) == {(seed, customer) for seed in seed_ids for customer in ids}, "初始化分组的客户集合不一致")
    minimum_ari = min(adjusted_rand(labels[("first", "kmeans_fixed")],
                                  [int(seeds[(seed, customer)]["cluster"]) for customer in ids]) for seed in seed_ids)
    close(summary["single_initialization_minimum_ari"], minimum_ari, "压力汇总 单次初始化最小ARI")
    close(summary["first_customer_silhouette"], silhouette(scaled["first"], labels[("first", "kmeans_fixed")]),
          "压力汇总 首期轮廓系数")
    close(summary["customer_fixed_model_time_ari"],
          adjusted_rand(labels[("first", "kmeans_fixed")], labels[("later", "kmeans_fixed")]),
          "压力汇总 固定模型时间ARI")


def check_stress(root):
    output = root / "examples/results/stress"
    summary = read_json(output / "summary.json")
    check_stress_forecast(root, output, summary)
    check_stress_warehouse(output, summary)
    check_stress_locations(root, output, summary)
    check_stress_customers(output, summary)
    return "压力练习：重算两机制MAE、玩具拣选/调位距离、同候选覆盖、客户轮廓与时间/种子ARI。"


CHECKS = {"01": check_forecast, "02": check_warehouse,
          "03": check_locations, "04": check_customers, "stress": check_stress}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="仓库路径；通常不需要填写")
    parser.add_argument("--task", choices=CHECKS, help="只检查一道题，例如 --task 01")
    args = parser.parse_args()
    failures = []
    for task in ([args.task] if args.task else CHECKS):
        try:
            print("OK " + CHECKS[task](args.root))
        except (CalculationError, OSError, KeyError, ValueError, IndexError, ZeroDivisionError) as error:
            failures.append(task)
            print(f"FAIL 题{task}：{error}", file=sys.stderr)
    if failures:
        return 1
    print(f"数值比较容限：绝对{ABS_TOL:g}，相对{REL_TOL:g}（处理CSV舍入）。")
    print("检查不读取文件摘要。基础题四其他k/种子未保存标签，未独立核对；不判断真实业务收益。")
    print("压力选址只穷举给定9城候选；连续坐标全局最优、模型拟合过程和口径对账文字未审计。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
