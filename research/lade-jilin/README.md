# 吉林配送：第一次公开真实数据小研究

想从教学数据走到真实数据，从这里开始。使用菜鸟公开LaDe吉林配送记录，自行实现两条小路线：预测每天的完成记录数、接单时预测完成时长。先读[建模前计划](plan.md)，再读[看数据后的改判](before-model.md)，最后看[实际实验报告](results/report.md)。前两份记录保留原文。

这次先遇到日期缺口、跨日完成和长耗时，再决定怎样评价。最终继续研究哪条路线，是看过实验后的判断；你可以找反例推翻它。

## 怎么运行

从仓库根目录，按[新手运行说明](../../docs/beginner/python-first-steps.md)安装依赖，再运行：

```bash
python research/lade-jilin/run.py --download
```

程序从官方固定版本下载约4.7MB文件，保存到本专题的`data/`目录并核对SHA256。原始CSV不提交仓库。已经下载同一文件时：

```bash
python research/lade-jilin/run.py --data /path/to/delivery_jl.csv
```

Windows示例：`python research/lade-jilin/run.py --data "D:\data\delivery_jl.csv"`。结果会重建到`results/`；先复制旧结果，再改代码比较。

## 先认识五个字段

| 字段 | 含义 | 本次怎么用 |
| --- | --- | --- |
| order_id | 一条配送记录的ID | 检查重复，不作为预测特征 |
| ds | 文件的日期字段 | 先核对它对应接单日还是完成日 |
| accept_time | 接单时间 | 当时可见的接单小时；计算训练标签是否已成熟 |
| delivery_time | 完成时间 | 按天统计完成记录；完成后才能得到时长标签 |
| city | 城市 | 核对本次确实为吉林记录 |

“标签成熟”就是答案已经发生：昨天接单、明天才完成的运单，今天不能用其最终时长训练。MAE是预测和实际差值取绝对值后平均；这里完成量的单位是记录/天，时长的单位是分钟，不能直接比两个数字选择路线。

## 结果读什么

- [report.md](results/report.md)：问题、方法、发现、下一步与模拟答辩。
- [data_audit.json](results/data_audit.json)：输入来源、版本、缺失、日期与时长范围。
- [volume_choices.csv](results/volume_choices.csv)：每次选模时已知的历史成绩。
- [duration_metrics.csv](results/duration_metrics.csv)：每窗口误差及超过6小时记录的误差。
- 两份逐条预测CSV供重算；输出不含原运单ID、配送员ID或坐标。

原数据只包含公开配送记录，未说明所有日期的采样覆盖，不能直接等同市场需求。正式赛题另从[赛事入口](../../competition-requirements/data-access.md)取得；本专题是一份研究练习。

来源：[官方数据卡](https://huggingface.co/datasets/Cainiao-AI/LaDe)、[作者说明与论文](https://github.com/wenhaomin/LaDe)。引用：Wu等，*LaDe: The First Comprehensive Last-mile Express Dataset from Industry*，KDD 2024，DOI 10.1145/3637528.3671548。官方卡标Apache-2.0及研究用途，本次仅下载其公开吉林文件，不复制上游实现。
