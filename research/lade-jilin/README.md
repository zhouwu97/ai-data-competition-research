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

换成自有同字段CSV时，显式选择自有模式并单独指定结果目录：

```bash
python research/lade-jilin/run.py --custom-data --data "D:/data/records.csv" --output-dir outputs/my-delivery --horizon-days 14 --windows 4 --hour-min-records 30
python scripts/check_public_research.py --output-dir outputs/my-delivery
```

字段仍为下面的五列；运单ID唯一、必需字段非空、完成时间不早于接单时间。`accept_time`与`delivery_time`使用完整`YYYY-MM-DD HH:MM:SS`，`ds`使用`YYYY-MM-DD`。若沿用LaDe无年份格式，显式加`--year 2022`：时间为`MM-DD HH:MM:SS`，ds为四位MMDD。输入CSV使用UTF-8或UTF-8 BOM。自有模式保存实际SHA256与`user_supplied_records`来源类型，不沿用官方文件的来源与许可声明。

`--observation-end 2026-09-30`指定观察截止日：完成量只统计当日及以前的完成记录；时长仍按接单日期从完整输入选择测试对象，逐窗口记录已成熟与未成熟条数到`duration_label_maturity.csv`。例如截止日当天两单接单，一单10分钟完成、一单三天后完成，程序会保留这两个对象并停止评分，不能删除慢单后只给快单计算MAE。

使用`--evaluation-end 2026-09-23`可提前结束回测，让测试对象有更多时间完成；或者等标签成熟后延后观察截止日。提前几天本身不保证标签成熟，程序仍检查完整接单对象。未指定观察截止日时，使用输入中已提供的全部完成标签。输入需保留跨截止日完成的记录及其时间；如果文件本身删除了未完成单，这项检查无法恢复漏掉的对象，评价仍可能偏乐观。

程序在回测结束日以前的最长连续完成日期段，取最后指定数量的窗口，至少需要28天训练、一个完整的选模历史窗口和全部测试窗口。缺记录日保持缺失，不能自动解释为零业务；测试窗口没有接单对象时会说明原因并停止。

每次运行将`run_status.json`与报告标为`running`，全部成功后改为`success`；出错时报告显示`failed`和原因，避免误读旧成功报告。分项文件在失败时可能是旧结果，检查器会拒绝非成功运行。公开状态只记录输入文件名、实际SHA256、来源类型与运行状态；完整命令、路径和错误详情保存在`run.local.log`，该日志已由Git忽略。

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
- [duration_label_maturity.csv](results/duration_label_maturity.csv)：各窗口按接单日选出的对象数及观察截止时已成熟、未成熟条数。
- 两份逐条预测CSV供重算；输出不含原运单ID、配送员ID或坐标。
- [duration_diagnostics.csv](results/duration_diagnostics.csv)：按接单小时、接单日期、是否超过6小时拆开条数与MAE；长尾分类是事后诊断，不能用于预测时选对象。

先看逐窗口改善是否一致，再从接单小时和日期找误差集中对象。用后续日期的新数据在另一输出目录运行，才能检验改善是否保持；当前四个窗口的4.61%改善不证明后续时间仍有效。

`python scripts/validate_research.py`会同时检查教学结果和仓库内的真实研究。自有目录使用上面的独立检查命令，它从逐条预测重算完成量、时长和长尾的窗口MAE、总体MAE、分组诊断与报告数字。它不依赖原始数据下载，也不验证模型拟合、原始时间标签或数据覆盖说明；范围见[测试说明](../../tests/README.md)。

原数据只包含公开配送记录，未说明所有日期的采样覆盖，不能直接等同市场需求。正式赛题另从[赛事入口](../../competition-requirements/data-access.md)取得；本专题是一份研究练习。

来源：[官方数据卡](https://huggingface.co/datasets/Cainiao-AI/LaDe)、[作者说明与论文](https://github.com/wenhaomin/LaDe)。引用：Wu等，*LaDe: The First Comprehensive Last-mile Express Dataset from Industry*，KDD 2024，DOI 10.1145/3637528.3671548。官方卡标Apache-2.0及研究用途，本次仅下载其公开吉林文件，不复制上游实现。
