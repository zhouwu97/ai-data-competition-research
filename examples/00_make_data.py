"""生成原创、固定随机种子的教学数据。所有客户、仓库、城市都是虚构的。"""
import numpy as np
import pandas as pd
from common import INPUT, write_csv

def main():
    INPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)  # 固定种子是为了学习时能复查，不代表结果真实。
    # 题一：112天、两个区域、两种商品。数量有周规律、缓慢趋势和噪声。
    rows = []
    weekly = np.array([0, 2, 1, 3, 7, 15, 10])
    for region in ["North", "South"]:
        for sku, base in [("A", 25), ("B", 12)]:
            for day, date in enumerate(pd.date_range("2025-01-01", periods=112)):
                quantity = max(0, round(base + 0.12 * day + weekly[date.dayofweek]
                                       + (3 if region == "South" else 0)
                                       + rng.normal(0, 3)))
                rows.append([date.date(), region, sku, quantity])
    write_csv(pd.DataFrame(rows, columns=["date", "region", "sku", "quantity"]),
              INPUT / "demand.csv")
    # 题二：同一订单的同一SKU可以分成多行。不能把行数当成品项数。
    orders = [
        ["2025-01-01", "O01", "A", 6], ["2025-01-01", "O01", "B", 1],
        ["2025-01-01", "O02", "A", 2], ["2025-01-01", "O02", "C", 1],
        ["2025-01-01", "O03", "B", 3], ["2025-01-01", "O03", "B", 1],
        ["2025-01-01", "O03", "C", 1],
        ["2025-01-02", "O04", "A", 10], ["2025-01-02", "O04", "D", 1],
        ["2025-01-02", "O05", "A", 8], ["2025-01-02", "O05", "C", 1],
        ["2025-01-02", "O06", "B", 2], ["2025-01-02", "O06", "C", 2],
        ["2025-01-02", "O07", "C", 1], ["2025-01-02", "O07", "E", 1]]
    write_csv(pd.DataFrame(orders, columns=["date", "order_id", "sku", "quantity"]),
              INPUT / "warehouse_orders.csv")
    # 题三：在虚构平面上用“公里”作为两个坐标的单位，不是经纬度。
    cities = []
    for group, (x, y) in enumerate([(0, 0), (150, 35), (330, 0)]):
        for city in range(4):
            cities.append([f"T{group}{city}", x + rng.normal(0, 15),
                           y + rng.normal(0, 15), int(rng.integers(10, 100))])
    write_csv(pd.DataFrame(cities, columns=["city_id", "x_km", "y_km", "demand"]),
              INPUT / "cities.csv")
    # 题四：先生成运单，再生成业务量表，练习按客户聚合与一对一合并。
    # 分组参数只是造数据的机制；不输出所谓“真实客户等级”作为答案。
    shipments = []
    for i in range(60):
        customer = f"C{i:03}"
        group = i // 20
        count = int(rng.integers(*[(2, 6), (8, 16), (4, 10)][group]))
        for j in range(count):
            days_ago = int(rng.integers(*[(15, 45), (1, 15), (4, 25)][group]))
            quantity = int(rng.integers(*[(2, 12), (5, 20), (35, 65)][group]))
            date = pd.Timestamp("2025-05-01") - pd.Timedelta(days=days_ago)
            shipments.append([customer, f"W{i:03}-{j:02}", date.date(), quantity])
    waybills = pd.DataFrame(shipments,
                           columns=["customer_id", "waybill_id", "date", "quantity"])
    write_csv(waybills, INPUT / "waybills.csv")
    volume = waybills.groupby("customer_id", as_index=False)["quantity"].sum()
    volume = volume.rename(columns={"quantity": "business_volume"})
    write_csv(volume, INPUT / "business_volume.csv")
    print("已生成5份合成教学CSV；正式赛题数据仍未取得。")

if __name__ == "__main__":
    main()
