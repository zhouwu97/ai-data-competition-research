# 题一智能分仓：先检查数据，再比较深度学习

使用本机已下载的完整往届 SQLite，研究未来14天成交件数。来源、适用范围与本轮证据集中见[证据说明](../../docs/research/evidence.md#2026-10-23题一智能分仓深度学习实验)。先读[数据审计](results/data_audit.md)，再看[实验结果](results/report.md)。

本轮失误怎样转为下次比赛的研究方法，见[两套物流数据复盘](../../docs/research/two-dataset-retrospective.md)：从要支持的决定出发，说明每一步的问题、产物、用途和采用依据。

这次分开回答两个问题：全国模型直接预测数据中每日已记录成交总量；共享商品模型让多个商品共用一套网络，只在未来有记录的商品日上检查误差。缺商品日保持未知，不用未来记录名单拼出全国预测。

原始SQLite需自行从[作者公开固定目录](https://github.com/hzy-sunstone/ai-journey-notebooks/tree/09d9ec99415ddde7c817e9fb83b4706bb572b17f/%E9%95%BF%E9%A3%8E%E6%9D%AF%E9%A1%B9%E7%9B%AE)取得，不随本仓库提交。可以保留在任意位置，再给audit.py与run.py传入`--input "D:/data/智能分仓数据集.sqlite"`；不要用旁边仅1,000行的CSV替代完整库。

## 输入与特征为什么这样选

| 信息 | 实际使用 | 原因 |
| --- | --- | --- |
| 目标 | `qty_alipay`成交件数 | 与下单件数、金额分开，不反推缺失销量 |
| 历史成交 | 最近28天逐日值与观测掩码；7/14/28/56天均值、中位数、最大值、零值与覆盖比例 | 同时表示近期水平、波动和缺记录；零值与未知分开 |
| 数量尺度 | 当时近28天已记录成交均值，至少1件 | 让大、小商品共享网络；尺度只用自己的过去 |
| 类别 | 商品模型的类目one-hot | 编号大小不代表距离；类别集合按当前历史确定 |
| 历史行为对照 | `pv_uv/cart_uv/collect_uv/qty_gmv/num_alipay`过去7/28天汇总 | 检验历史浏览、加购等是否提供成交历史之外的信号；从不输入未来实际行为 |
| 已知日历 | 未来14天星期、年内周期、11月11日与12月12日标记 | 日历提前已知；不假定未来活动力度已知 |
| 全国覆盖 | 过去28天每日记录商品数 | 全国记录的商品集合在变化 |
| 未纳入 | 金额、商品ID数值、品牌/供应商、Cost | 金额与件数口径有不一致；首轮控制维数。Cost缺少历史地域需求，不能补出分仓销量 |

共享商品的资格是在预测前56天至少有28条记录；未来缺标签只是不参与误差计算，不回头改变预测资格。输入最早允许28天历史，不足56天的启动段保留掩码，因此2014双11没有被长窗口机械排除。

## 实验怎样比较

先固定[配置](config.json)：开发期为2015-07-13—10-04的六个14天窗口；后段为2015-10-05—12-27的六个不重叠窗口。每次训练的14天答案必须在当时预测起点以前全部结束。后段每次可用此前新发生的数据重新拟合，属于滚动预测。

三个深度候选分别是“成交历史+MAE”“再加历史行为+MAE”“历史行为+Poisson损失”，均为两层MLP、一次输出14天；商品模型在所有历史达标商品之间共享。MAE让模型减少绝对件数误差，Poisson对照检查另一种计数损失是否有利。还有14/28/56天中位数、28天均值、同星期均值，以及使用完整行为特征的Ridge线性模型；Ridge与行为MLP才是同输入对照。

开发只用固定一个种子比较三种深度候选，各任务按开发误差选一个深度候选和一个中位数。后段仅运行选中的深度候选，用预先固定的三个种子平均，同时保留各自结果。选中“深度候选”不代表它超过了中位数。

事先规定“稳定超过”：后段整体MAE更低、六窗均不退步，并且三个种子的整体MAE都更低。14天合计误差另报，不能用合计误差抵消每天的错误。数据审计已看过全期覆盖和极值，这次属于冻结设置后的历史时间评价，不称为未来盲测。

## 复跑与检查

在仓库根目录运行。原库位置见`data.py`，可用`--input`指定其他本地副本。本机系统Python已有PyTorch CPU；仓库原`.venv`没有PyTorch。实际版本见输出`manifest.json`与[依赖记录](requirements.txt)。

```powershell
python research/smart-demand/audit.py
python research/smart-demand/run.py --output outputs/smart-demand-reproduce
python research/smart-demand/verify.py --output outputs/smart-demand-reproduce
python research/smart-demand/report.py --output outputs/smart-demand-reproduce --results research/smart-demand/results-reproduce
python -m unittest discover -s research/smart-demand -p 'test_*.py' -v
```

新实验使用新输出目录；中断后用相同输入、配置加`--resume`恢复。逐窗预测、训练记录和选择文件保存在输出目录；本专题`results/`保留精简报告、数据审计和图。独立检查从原库和保存的预测重算，不把文件摘要检查当作准确性证明。
