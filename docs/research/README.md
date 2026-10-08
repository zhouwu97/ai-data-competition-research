# 按你遇到的问题研究

先选一个问题，读解释、跑相应代码，再找结果会改变哪项判断。基础练习和困难情景是同一套方法的两种观察。

| 现在的问题 | 先读 | 接着做 |
| --- | --- | --- |
| K-means究竟在做什么？ | [四数字手算](../beginner/kmeans-explained.md) | 运行 `examples/05_kmeans_by_hand.py`，对照三轮中心 |
| 一个模型表现好，换一段时间还可靠吗？ | [数据与验证](../beginner/data-and-validation.md) | 对照基础预测与[趋势突变](../../examples/results/stress/report.md) |
| 不知道未来时该选哪个预测模型？ | [顺序选择报告](../../examples/results/sequential/report.md) | 看当时可用的回测成绩与后来误差，保留反应滞后的失败 |
| 每日ABC改变了，要马上搬货吗？ | [EIQ和ABC](../beginner/eiq-abc-explained.md) | 看困难情景的每日观察、次日决策和搬移成本 |
| 多建一个中心，服务一定更好吗？ | [题三档案](../../changfeng/task-03/README.md) | 比较平方距离目标、覆盖目标及相同候选集 |
| 愿意放宽一点距离目标，能换多少覆盖？ | [权衡报告](../../examples/results/tradeoffs/report.md) | 比较三个半径的非支配方案，不把平方损失当成本 |
| 分群能帮我提出什么策略？ | [题四档案](../../changfeng/task-04/README.md) | 对照规则分组、弱结构和时间变化，找额外信息 |
| 我怎么知道报告里的数字算对了？ | [检查说明](../../tests/README.md) | 运行独立重算，再看故意改错后失败的测试 |
| 该选哪题，怎样向评委解释？ | [从结果到选择](decision-cases.md) | 提出基础路线和一条针对瓶颈的增强路线 |
| 正式数据从哪里来？ | [数据获取](../../competition-requirements/data-access.md) | 确认报名、课程权限、字段和统计口径 |
| 想研究真实数据，先做哪条路线？ | [吉林配送小研究](../../research/lade-jilin/README.md) | 按计划、字段改判、实验报告顺序，研究记录量与时长两条路线 |

[精选方法来源](method-sources.md)说明每篇资料读哪部分；[证据状态](evidence.md)集中说明哪些结论已经验证。
