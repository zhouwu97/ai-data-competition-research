"""题二练习：按天计算EIQ和ABC，先确保每一个数都能手算核对。"""
import pandas as pd
from common import read_input, output_dir, write_csv, write_json, save_figure, plt

def abc_labels(values, a_limit=0.80, b_limit=0.95):
    # 排序在调用前完成。跨过阈值的那个SKU仍归前一类，避免最大SKU被排除出A。
    # 零数量单独标记；小样本不保证A、B、C三类都存在。
    total = values.sum()
    if total <= 0:
        raise ValueError("总数量必须大于0。")
    before = (values.cumsum() - values) / total
    return ["ZERO" if qty == 0 else ("A" if prev < a_limit else
            "B" if prev < b_limit else "C") for qty, prev in zip(values, before)]

def main():
    out = output_dir("task02")
    raw = read_input("warehouse_orders.csv", ["date", "order_id", "sku", "quantity"])
    pd.to_datetime(raw.date, errors="raise")
    if (raw.quantity <= 0).any():
        raise ValueError("本练习只含正向出库；退货和0数量需另定口径。")
    # 同订单同SKU多行先求和；真实数据要先分清拆行与错误重复，不能机械去重。
    lines = raw.groupby(["date", "order_id", "sku"], as_index=False)["quantity"].sum()
    orders = lines.groupby(["date", "order_id"]).agg(
        EQ=("quantity", "sum"), EN=("sku", "nunique")).reset_index()
    items = lines.groupby(["date", "sku"]).agg(
        IQ=("quantity", "sum"), IK=("order_id", "nunique")).reset_index()
    # 跨日订单号允许复用，因为订单键在这里是date+order_id。
    for date, group in lines.groupby("date"):
        assert int(group.quantity.sum()) == int(orders[orders.date == date].EQ.sum())
        assert int(group.quantity.sum()) == int(items[items.date == date].IQ.sum())
    classified = []
    for date, group in items.groupby("date"):
        group = group.sort_values(["IQ", "sku"], ascending=[False, True]).copy()
        group["cum_share"] = group.IQ.cumsum() / group.IQ.sum()
        group["ABC"] = abc_labels(group.IQ)
        classified.append(group)
    abc = pd.concat(classified, ignore_index=True)
    write_csv(orders, out / "order_eiq.csv")
    write_csv(abc, out / "item_eiq_abc.csv")
    comparison = abc[["date", "sku", "IQ", "IK", "ABC"]].copy()
    comparison["ABC_70_90"] = ""
    for date, group in abc.groupby("date", sort=False):
        comparison.loc[group.index, "ABC_70_90"] = abc_labels(group.IQ, 0.70, 0.90)
    changed = int((comparison.ABC != comparison.ABC_70_90).sum())
    write_csv(comparison, out / "threshold_comparison.csv")
    summary = {"data_kind": "synthetic", "days": int(lines.date.nunique()),
               "source_lines": len(raw), "normalized_order_sku_lines": len(lines),
               "total_quantity": int(lines.quantity.sum()),
               "threshold_changed_rows": changed}
    write_json(summary, out / "summary.json")
    fig, ax = plt.subplots(figsize=(7, 4))
    latest = abc[abc.date == abc.date.max()]
    ax.bar(latest.sku, latest.IQ, color="#218380")
    ax.set(title="Synthetic data: daily SKU outbound quantity", ylabel="IQ (units)")
    save_figure(fig, out / "abc.svg")
    text = ("# 题二教学实验记录\n\n合成订单；没有真实仓库布局或实测效率数据。\n\n"
            f"原始{len(raw)}行，按日期-订单-SKU汇总后{len(lines)}行，"
            f"总数量{summary['total_quantity']}；EQ合计与IQ合计按天核对通过。\n\n"
            "EQ是每单件数，EN是每单不同SKU数，IQ是每SKU件数，IK是含该SKU的不同订单数。"
            "数量高和订单出现频繁不是一回事。ABC本例按IQ，不是金额或利润。\n\n"
            f"80%/95%改为70%/90%后，有{changed}个日期-SKU记录改变类别。"
            "两组阈值都是教学假设，不能当作统一赛规。\n\n"
            "策略假设：高IK品项可优先研究易取放位置；高EN订单可比较分区或批量拣选。"
            "需补充尺寸、通道、容量、路线与实际作业数据才能证明提效。\n\n"
            "证据：[订单EIQ](order_eiq.csv)、[每日商品EIQ-ABC](item_eiq_abc.csv)、"
            "[阈值比较](threshold_comparison.csv)、[图表](abc.svg)、[汇总](summary.json)。\n")
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print("题二完成：按天计算EQ/EN/IQ/IK，核对数量并比较ABC阈值。")

if __name__ == "__main__":
    main()
