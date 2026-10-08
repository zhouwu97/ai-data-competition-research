"""题三练习：虚构平面上的需求加权K-means及覆盖评价。不是全国真实选址。"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from common import read_input, output_dir, write_csv, write_json, save_figure, plt

def main():
    out = output_dir("task03")
    cities = read_input("cities.csv", ["city_id", "x_km", "y_km", "demand"])
    if cities.city_id.duplicated().any() or (cities.demand <= 0).any():
        raise ValueError("城市ID应唯一，需求权重应为正。")
    xy = cities[["x_km", "y_km"]].to_numpy()
    weights = cities.demand.to_numpy()
    metrics, allocations, centers_rows = [], [], []
    selected_centers = None
    for k in [1, 2, 3, 4]:
        # demand是样本权重，不是与公里坐标混在一起的第三个坐标。
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(xy, sample_weight=weights)
        centers = model.cluster_centers_
        distances = np.sqrt(((xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        nearest = distances.min(axis=1)
        labels = distances.argmin(axis=1)
        weighted_distance = float(np.average(nearest, weights=weights))
        for radius in [50, 100, 150]:
            covered = nearest <= radius
            # 两种分母都报告，不能把城市覆盖率当成需求覆盖率。
            metrics.append([k, radius, float(covered.mean()),
                            float(weights[covered].sum() / weights.sum()),
                            weighted_distance])
        for row, label, distance in zip(cities.itertuples(index=False), labels, nearest):
            allocations.append([k, row.city_id, int(label), float(distance)])
        for label, (x, y) in enumerate(centers):
            centers_rows.append([k, label, float(x), float(y)])
        if k == 3:
            selected_centers = centers
    scores = pd.DataFrame(metrics, columns=["k", "radius_km", "city_coverage",
                                           "demand_coverage", "weighted_distance_km"])
    write_csv(scores, out / "coverage.csv")
    write_csv(pd.DataFrame(allocations, columns=["k", "city_id", "cluster", "distance_km"]),
              out / "allocations.csv")
    write_csv(pd.DataFrame(centers_rows, columns=["k", "cluster", "x_km", "y_km"]),
              out / "centers.csv")
    row = scores[(scores.k == 3) & (scores.radius_km == 100)].iloc[0]
    summary = {"data_kind": "synthetic", "coordinate_system": "fictional_planar_km",
               "illustration_k": 3, "illustration_radius_km": 100,
               "city_coverage": float(row.city_coverage),
               "demand_coverage": float(row.demand_coverage),
               "weighted_distance_km": float(row.weighted_distance_km)}
    write_json(summary, out / "summary.json")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(cities.x_km, cities.y_km, s=cities.demand * 2, alpha=0.7, label="Demand")
    ax.scatter(selected_centers[:, 0], selected_centers[:, 1], marker="X",
               s=140, color="#c44e52", label="K-means centers")
    for x, y in selected_centers:
        ax.add_patch(plt.Circle((x, y), 100, fill=False, color="#218380", alpha=0.5))
    ax.set(xlabel="x (km)", ylabel="y (km)", title="Synthetic plane: k=3, radius=100 km")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    save_figure(fig, out / "locations.svg")
    text = ("# 题三教学实验记录\n\n12个虚构平面城市，两个坐标均为公里，"
            "距离是直线欧氏距离，没有道路或真实地理资料。\n\n"
            f"仅作演示的k=3、半径100公里：城市覆盖率{row.city_coverage:.4f}，"
            f"需求覆盖率{row.demand_coverage:.4f}，需求加权平均距离"
            f"{row.weighted_distance_km:.4f}公里。\n\n"
            "k=3只是画图示例，并非最优中心数量。K-means优化加权平方距离，"
            "没有直接优化覆盖率、建站成本、容量或运输时效。覆盖表比较4个k和3个半径；"
            "中心是连续坐标，尚未约束到可建站地点。正式题必须重新定义距离与服务阈值。\n\n"
            "证据：[覆盖表](coverage.csv)、[城市分配](allocations.csv)、"
            "[中心](centers.csv)、[图表](locations.svg)、[汇总](summary.json)。\n")
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print("题三完成：4个中心数量、3个半径，区分城市覆盖与需求覆盖。")

if __name__ == "__main__":
    main()
