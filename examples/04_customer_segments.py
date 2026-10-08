"""题四练习：聚合运单、合并业务量、比较规则分组和K-means。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from common import read_input, RESULT, write_csv, write_json, save_figure, plt

def silhouette_or_reason(points, labels):
    groups = len(np.unique(labels))
    if not 2 <= groups < len(points):
        return None, f"实际分组数为{groups}，轮廓系数要求2至样本数减1个组。"
    return float(silhouette_score(points, labels)), ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="含waybills.csv和business_volume.csv的目录")
    parser.add_argument("--output-dir", type=Path, help="结果目录；自有数据必须指定")
    parser.add_argument("--cutoff", default="2025-05-01", help="观察时点YYYY-MM-DD，仅使用此前记录")
    args = parser.parse_args(argv)
    if args.input_dir is not None and args.output_dir is None:
        parser.error("自有数据请指定--output-dir，单独保存结果。")
    out = args.output_dir or RESULT / "task04"
    out.mkdir(parents=True, exist_ok=True)
    data_kind = "synthetic" if args.input_dir is None else "user_supplied_records"
    bills = read_input("waybills.csv", ["customer_id", "waybill_id", "date", "quantity"], args.input_dir)
    volume = read_input("business_volume.csv", ["customer_id", "business_volume"], args.input_dir)
    bills["date"] = pd.to_datetime(bills.date, errors="raise")
    cutoff = pd.Timestamp(args.cutoff)
    if pd.isna(cutoff) or cutoff != cutoff.normalize():
        raise ValueError("观察时点需要是有效日期。")
    if bills.waybill_id.duplicated().any() or (bills.date >= cutoff).any():
        raise ValueError("运单ID重复或有观察时点以后的记录。")
    bills["quantity"] = pd.to_numeric(bills.quantity, errors="raise")
    volume["business_volume"] = pd.to_numeric(volume.business_volume, errors="raise")
    if (not np.isfinite(bills.quantity).all() or not np.isfinite(volume.business_volume).all()
            or (bills.quantity <= 0).any() or (volume.business_volume <= 0).any()):
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
    candidates, illustration_labels = [], None
    distinct = len(np.unique(scaled, axis=0))
    illustration_k = min(3, distinct, max(1, len(customers) - 1))
    for k in [2, 3, 4, 5]:
        if k > min(len(customers), distinct):
            candidates.append(["kmeans", k, 0, None,
                               f"样本数{len(customers)}、不同特征行数{distinct}，无法形成{k}组。"])
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        score, reason = silhouette_or_reason(scaled, labels)
        candidates.append(["kmeans", k, len(np.unique(labels)), score, reason])
        if k == illustration_k:
            illustration_labels = labels
    if illustration_labels is None:
        illustration_labels = np.zeros(len(customers), dtype=int)
        candidates.append(["kmeans", 1, 1, None, "只有一个演示组，轮廓系数不可计算。"])
    # 简单对照：仅按业务量三等分。分位点是本教学数据的规则，不是官方标准。
    q1, q2 = customers.business_volume.quantile([1 / 3, 2 / 3])
    rule_labels = np.where(customers.business_volume <= q1, 0,
                           np.where(customers.business_volume <= q2, 1, 2))
    rule_score, rule_reason = silhouette_or_reason(scaled, rule_labels)
    candidates.append(["volume_rule", 3, len(np.unique(rule_labels)), rule_score, rule_reason])
    customers["cluster"] = illustration_labels
    customers["volume_rule"] = rule_labels
    profiles = customers.groupby("cluster").agg(
        customer_count=("customer_id", "size"), mean_orders=("orders", "mean"),
        mean_volume=("business_volume", "mean"), mean_recency=("recency_days", "mean")
    ).reset_index()
    # ARI按“哪些人同组”比较，因此不怕0、1、2编号互换。
    seeds = []
    for seed in [0, 1, 2, 3, 4]:
        other = KMeans(n_clusters=illustration_k, n_init=10, random_state=seed).fit_predict(scaled)
        seeds.append([seed, float(adjusted_rand_score(illustration_labels, other))])
    score, reason = silhouette_or_reason(scaled, illustration_labels)
    summary = {"data_kind": data_kind, "customers": len(customers),
               "illustration_k": illustration_k, "actual_clusters": len(np.unique(illustration_labels)),
               "silhouette": score, "silhouette_reason": reason,
               "silhouette_k3": score if illustration_k == 3 else None,
               "volume_rule_silhouette_reason": rule_reason,
               "ari_to_volume_rule": float(adjusted_rand_score(illustration_labels, rule_labels)),
               "minimum_seed_ari": min(x[1] for x in seeds),
               "features": features, "cutoff": str(cutoff.date())}
    write_csv(customers.drop(columns=["last_date"]), out / "customer_features.csv")
    write_csv(profiles, out / "profiles.csv")
    write_csv(pd.DataFrame(candidates, columns=["method", "k", "actual_clusters", "silhouette", "reason"]),
              out / "comparison.csv")
    write_csv(pd.DataFrame(seeds, columns=["seed", "ari_to_seed42"]),
              out / "seed_stability.csv")
    write_json(summary, out / "summary.json")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in customers.groupby("cluster"):
        ax.scatter(group.orders, group.business_volume, label=f"cluster {label}", alpha=0.7)
    ax.set(xlabel="Orders", ylabel="Business volume",
           title=f"{data_kind}: only 2 of the 3 features shown")
    ax.legend(fontsize=8)
    save_figure(fig, out / "segments.svg")
    displayed_score = f"{score:.4f}" if score is not None else f"不可计算（{reason}）"
    text = (f"# 客户分群实验记录\n\n数据类型：{data_kind}。运单与业务量表一对一合并后"
            f"{len(customers)}位客户。\n\n"
            f"演示k={illustration_k}：轮廓系数{displayed_score}，与仅按业务量分组的ARI"
            f"{summary['ari_to_volume_rule']:.4f}，5个额外初始化种子与种子42的最小ARI"
            f"{summary['minimum_seed_ari']:.4f}。\n\n"
            "三个数分别描述几何结构、与规则的分组相似性、初始化稳定性。"
            "它们都不表示利润、营销效果或真实客户等级；没有干预结果，也没验证未来稳定性。"
            "优先演示k=3；样本或不同特征不足时降低组数。k=2至5及规则分组的"
            "不可计算指标保存为空并附原因；全部相同特征或单个客户仍输出画像。"
            "退化分组的ARI可能为1，不代表存在有效客户区分。\n\n"
            "先读profiles.csv的原始单位均值，再为群体提出服务假设；"
            "编号0不是低价值，编号2也不是高价值。\n\n"
            "证据：[客户特征](customer_features.csv)、[画像](profiles.csv)、"
            "[方法比较](comparison.csv)、[种子稳定性](seed_stability.csv)、"
            "[图表](segments.svg)、[汇总](summary.json)。\n")
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"题四完成：{len(customers)}位客户，演示k={illustration_k}；不可计算指标见comparison.csv的reason。")

if __name__ == "__main__":
    main()
