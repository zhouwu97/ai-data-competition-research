"""可重复的错误示范：检查器应当发现故意制造的问题。"""
import pandas as pd
from common import read_input, output_dir, write_csv, assert_future_split
from common import assert_same_columns

def require_available(available_at, prediction_at):
    if pd.Timestamp(available_at) > pd.Timestamp(prediction_at):
        raise ValueError("字段在预测时点之后才可用：不能进入当时的模型。")

def main():
    out = output_dir("failure_cases")
    rows = []
    def rejected(name, function, fix):
        try:
            function()
        except (ValueError, pd.errors.MergeError) as error:
            rows.append([name, "caught", str(error), fix])
        else:
            raise AssertionError(f"错误样例未被发现：{name}")
    rejected("future_feature", lambda: require_available("2025-01-03", "2025-01-02"),
             "只使用预测时点前能取得的字段；补录时间也要审查")
    rejected("future_split", lambda: assert_future_split(
        ["2025-01-01", "2025-01-03"], ["2025-01-02"]),
        "未来预测用训练日期严格早于验证日期的切分")
    rejected("stacking_column_order", lambda: assert_same_columns(
        ["model_a", "model_b"], ["model_b", "model_a"]),
        "固定元模型输入名称和顺序；另审查OOF与推理的生成流程")
    left = pd.DataFrame({"id": [1, 2]})
    right = pd.DataFrame({"id": [1, 1, 2], "value": [3, 4, 5]})
    rejected("duplicated_join_key", lambda: left.merge(right, on="id", validate="one_to_one"),
             "确认业务主键；选择正确合并关系，不能默默接受行数膨胀")
    orders = read_input("warehouse_orders.csv", ["date", "order_id", "sku", "quantity"])
    o3 = orders[(orders.date == "2025-01-01") & (orders.order_id == "O03")]
    wrong, correct = len(o3), o3.sku.nunique()
    assert wrong == 3 and correct == 2
    rows.append(["EN_counts_rows", "caught", f"按行数得到{wrong}，不同SKU数应为{correct}",
                 "先明确订单键；EN按不同SKU计数"])
    # 故意把答案当预测：0误差只是作弊演示，不是一个可用预测模型。
    actual = [10, 20, 30]
    illegal_prediction = actual.copy()
    legal_constant_prediction = [20, 20, 20]
    illegal_mae = sum(abs(a - p) for a, p in zip(actual, illegal_prediction)) / 3
    legal_mae = sum(abs(a - p) for a, p in zip(actual, legal_constant_prediction)) / 3
    assert illegal_mae == 0 and legal_mae > 0
    rows.append(["answer_as_feature", "illustrated",
                 f"直接抄答案MAE={illegal_mae:.4f}；常数对照MAE={legal_mae:.4f}",
                 "指标好不等于流程正确；需要审核字段来源和可用时点"])
    write_csv(pd.DataFrame(rows, columns=["case", "status", "observation", "fix"]),
              out / "checks.csv")
    text = ("# 六个错误示范的实际执行记录\n\n这些是故意构造的教学反例。"
            "前四项验证检查器会拒绝输入；第五项核对EN；第六项说明抄答案能制造0误差。\n\n"
            "见[结果表](checks.csv)。这些检查覆盖指定情形，不能自动保证没有全部泄漏、"
            "业务口径错误或OOF污染。正式数据仍需人工审核。\n")
    (out / "report.md").write_text(text, encoding="utf-8")
    print("错误示范完成：4项拒绝检查、1项EIQ口径检查、1项答案泄漏示范。")

if __name__ == "__main__":
    main()
