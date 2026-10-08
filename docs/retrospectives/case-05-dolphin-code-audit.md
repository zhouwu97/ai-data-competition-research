# 案例 05：海豚杯三夏项目——从介绍回查实际代码

对象：qyyyy09315/DophinCup2025ofSanX。作者的 Top3 和成绩声明未独立核验。本次读取 README、清洗、BERT 表示、融合、标签检查与合成数据共 6 个文件；不是全仓库审计。固定来源见[证据清单](sources.md)。

## 能确认的路线

所读代码能串出“清洗 → 冻结 BERT 表示 → 表格分类 → 概率平均”的大致方法。但以下差异影响复现：

| 位置 | 静态观察 |
| --- | --- |
| [清洗](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/preprocessing/clean_data.py) | 使用当前工作目录的 train.csv / clean.csv，与说明的 raw / processed 不一致 |
| [BERT 表示](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/feature_engineering/bert_embedding/bert_convert_efficient.py) | 依赖本地模型目录，输出 Parquet；下游所读脚本输入 CSV |
| [名为 Stacking 的脚本](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/models/ensemble/stacking/st_brf_xgboost_catboost_stacking.py) | AdaBoost、XGBoost、CatBoost 三模型概率等权平均；未见 OOF 元学习器 |
| 同一分类脚本 | 二分类，随机分层留出 5%；写有固定阈值和综合评分 |
| [标签检查](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/preprocessing/label_detection.py)、[合成数据](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/preprocessing/synthetic_data_generator.py) | 前者只检查 target 列存在；后者检测离群值，所读实现未生成新样本 |

README 的多标签、Stacking、标签修正、数据合成等描述，不能替代这些文件的实际行为；也不能由单个脚本推断作者所有参赛代码。

## 本仓库分析

这份项目适合做“复现前审计”练习。需要先修通数据接口并核对赛题目标，才能比较架构。分类脚本先留出再拟合填补、筛选、缩放和重采样，这个局部顺序值得保留；若上游清洗先在全量数据上拟合统计量，又用于验证，则需要改为折内处理。

BERT 文件对所有 token 直接取均值，没有用 attention mask 排除 padding。作为静态推断，不同批次的填充长度可能影响表示；实际影响尚未测量。可用同一文本不同批次组成比较向量作为小实验。

## 我们的修正与复盘计划

1. 统一数据路径、文件格式、主键与目标定义；逐步记录行数、列数和输出文件。
2. 先建立 TF-IDF 或简单表格基线，再测试 BERT 表示，避免一次同时改多个模块。
3. 按主体与测试场景选择验证；在训练折内完成统计拟合和采样，验证侧保持原分布。
4. 把均值融合准确命名；如再做 Stacking，保存 OOF 输入与元模型的独立评估。
5. 评分权重与阈值以当届规程为准，不能从旧代码推定；报告区分 F1、accuracy 和综合分。

这些是本仓库的待执行计划。尚未运行上游模型、测量 padding 影响或确认当前赛规。

