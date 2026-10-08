"""题四练习：聚合运单、合并业务量、比较规则分组和K-means。"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from common import read_input, output_dir, write_csv, write_json, save_figure, plt

def main():
    out = output_dir("task04")
    bills = read_input("waybills.csv", ["customer_id", "waybill_id", "date", "quantity"])
    volume = read_input("business_volume.csv", ["customer_id", "business_volume"])
    bills["date"] = pd.to_datetime(bills.date, errors="raise")
    cutoff = pd.Timestamp("2025-05-01")  # 只使用观察时点以前的记录。
    if bills.waybill_id.duplicated().any() or (bills.date >= cutoff).any():
        raise ValueError("运单ID重复或有观察时点以后的记录。")
    if (bills.quantity <= 0).any() or (volume.business_volume <= 0).any():
        raise ValueError("本例只处理正向业务量，退货口径需另行制定。")
    aggregated = bills.groupby("customer_id").agg(
        orders=("waybill_id", "nunique"), last_date=("date", "max"),
        summed_quantity=("quantity", "sum")).reset_index()
    if set(aggregated.customer_id) != set(volume.customer_id):
        raise ValueError("两张表的客户集合不一致，先确认缺失客户的处理。")
    # validate防止重复客户ID导致合并后行数膨胀；这是实际常见错误。
    customers = aggregated.merge(volume, on="customer_id", validate="one_to_one")
    if not np.allclose(customers.summed_quantity, customers.business_volume):
        raise ValueError("运单数量与业务量表口径不一致。")
    customers["recency_days"] = (cutoff - customers.last_date).dt.days
    features = ["orders", "business_volume", "recency_days"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(customers[features])
    # 这是同一观察时点的描述分析，可以拟合这一批客户；
    # 若评估未来客户，必须另外固定训练时期和外推验证。
    candidates, labels3 = [], None
    for k in [2, 3, 4, 5]:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        candidates.append(["kmeans", k, float(silhouette_score(scaled, labels))])
        if k == 3:
            labels3 = labels
    # 简单对照：仅按业务量三等分。分位点是本教学数据的规则，不是官方标准。
    q1, q2 = customers.business_volume.quantile([1 / 3, 2 / 3])
    rule_labels = np.where(customers.business_volume <= q1, 0,
                           np.where(customers.business_volume <= q2, 1, 2))
    candidates.append(["volume_rule", 3, float(silhouette_score(scaled, rule_labels))])
    customers["cluster"] = labels3
    customers["volume_rule"] = rule_labels
    profiles = customers.groupby("cluster").agg(
        customer_count=("customer_id", "size"), mean_orders=("orders", "mean"),
        mean_volume=("business_volume", "mean"), mean_recency=("recency_days", "mean")
    ).reset_index()
    # ARI按“哪些人同组”比较，因此不怕0、1、2编号互换。
    seeds = []
    for seed in [0, 1, 2, 3, 4]:
        other = KMeans(n_clusters=3, n_init=10, random_state=seed).fit_predict(scaled)
        seeds.append([seed, float(adjusted_rand_score(labels3, other))])
    score = float(silhouette_score(scaled, labels3))
    summary = {"data_kind": "synthetic", "customers": len(customers),
               "illustration_k": 3, "silhouette_k3": score,
               "ari_to_volume_rule": float(adjusted_rand_score(labels3, rule_labels)),
               "minimum_seed_ari": min(x[1] for x in seeds),
               "features": features, "cutoff": str(cutoff.date())}
    write_csv(customers.drop(columns=["last_date"]), out / "customer_features.csv")
    write_csv(profiles, out / "profiles.csv")
    write_csv(pd.DataFrame(candidates, columns=["method", "k", "silhouette"]),
              out / "comparison.csv")
    write_csv(pd.DataFrame(seeds, columns=["seed", "ari_to_seed42"]),
              out / "seed_stability.csv")
    write_json(summary, out / "summary.json")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in customers.groupby("cluster"):
        ax.scatter(group.orders, group.business_volume, label=f"cluster {label}", alpha=0.7)
    ax.set(xlabel="Orders", ylabel="Business volume",
           title="Synthetic customers: only 2 of the 3 features shown")
    ax.legend(fontsize=8)
    save_figure(fig, out / "segments.svg")
    text = ("# 题四教学实验记录\n\n合成运单和合成业务量表，一对一合并后"
            f"{len(customers)}位客户。\n\n"
            f"演示k=3：轮廓系数{score:.4f}，与仅按业务量分组的ARI"
            f"{summary['ari_to_volume_rule']:.4f}，5个额外初始化种子与种子42的最小ARI"
            f"{summary['minimum_seed_ari']:.4f}。\n\n"
            "三个数分别描述几何结构、与规则的分组相似性、初始化稳定性。"
            "它们都不表示利润、营销效果或真实客户等级；没有干预结果，也没验证未来稳定性。"
            "k=3只是固定示范，同时列出k=2至5的指标，不能自动认定最高分最适合业务。\n\n"
            "先读profiles.csv的原始单位均值，再为群体提出服务假设；"
            "编号0不是低价值，编号2也不是高价值。\n\n"
            "证据：[客户特征](customer_features.csv)、[画像](profiles.csv)、"
            "[方法比较](comparison.csv)、[种子稳定性](seed_stability.csv)、"
            "[图表](segments.svg)、[汇总](summary.json)。\n")
    (out / "report.md").write_text(text, encoding="utf-8")
    print("题四完成：60位合成客户，规则对照、4个k和5个初始化检查。")

if __name__ == "__main__":
    main()
