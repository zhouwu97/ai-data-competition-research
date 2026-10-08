# AI / 数据分析竞赛预研

面向个人与算法社团的竞赛资料库：Kaggle、海豚杯、长风杯，以及天池、讯飞等可继续关注的平台。

资料与实验更新于 **2026-10-08**。这里既教方法，也研究方法什么时候失效、怎样依据结果选路线。

## 零基础先从这里开始

先打开[从零开始](docs/beginner/README.md)。不需要先懂K-means，也不需要先收藏全部资源：

1. [术语解释](docs/beginner/glossary.md)：样本、特征、标签、基线、训练、验证是什么。
2. [K-means手算解释](docs/beginner/kmeans-explained.md)：把1、2、8、9分两组，看中心怎样移动。
3. [EIQ与ABC解释](docs/beginner/eiq-abc-explained.md)：用几张订单核对件数、品项数与频次。
4. [运行说明](docs/beginner/python-first-steps.md)：Windows和macOS/Linux命令、结果位置与报错处理。
5. [可运行练习](examples/README.md)与[已执行结果](examples/results/report.md)：带中文注释的代码、CSV、图表、报告和日志。
6. [四题研究档案](changfeng/README.md)、[选题表](templates/topic-selection-matrix.md)和[AI审核流程](docs/ai-assisted-competition.md)：从学习转到正式选题。

练习使用合成数据，已有实际运行结果；正式数据和当届完整要求的获取状态集中见[证据说明](docs/research/evidence.md)与[赛事要求](competition-requirements/README.md)。

## 已经会基础，接着研究什么？

打开[按问题研究](docs/research/README.md)，直接进入你遇到的问题：预测突然失准、每天ABC变化、多建中心却覆盖变差、分群只是规模分档。

[困难情景的实际结果](examples/results/stress/report.md)与[决策案例](docs/research/decision-cases.md)把这些问题接到路线取舍和答辩。先解释失败，再考虑增加算法。

## 先看结论

- **练基础：** 先完成本站小练习，再用Datawhale补清洗与分析，熟悉后做Kaggle Titanic或House Prices的完整流程。
- **准备海豚杯：** 先确认数智分析还是应用创新赛道，再准备数据依据、实验、报告和展示。2026 辽宁赛通知已找到，日期存在来源差异，见赛事档案。
- **准备长风杯：** 2025 辽宁本科生赛道已有四个指定题目的公开说明，可练需求预测、仓储分析、选址和客户分群；本次尚未确认 2026 届完整规程。
- **选代码：** 先复现简单 baseline，再读高排名方案。当前找到的海豚杯代码项目与长风杯展示项目，完整程度差别很大。

## 阅读导航

| 内容 | 文件 |
| --- | --- |
| 从零理解术语与方法 | [新手教程](docs/beginner/README.md)、[精选现成教程](docs/beginner/learning-resources.md) |
| 带注释代码、输入、实际结果与图表 | [练习说明](examples/README.md)、[运行汇总](examples/results/report.md) |
| 方法为什么会失败、结果怎样改变选择 | [问题入口](docs/research/README.md)、[困难情景](examples/results/stress/report.md)、[决策案例](docs/research/decision-cases.md) |
| 数字是否算对、文件是否对应本次运行 | [独立计算检查](tests/README.md)、[版本与证据检查](scripts/validate_research.py) |
| 规则、正式数据与使用权限 | [要求清单](competition-requirements/README.md)、[数据获取](competition-requirements/data-access.md) |
| 四题用同一标准审查 | [四题档案](changfeng/README.md)、[选题表](templates/topic-selection-matrix.md) |
| AI分析、代码与报告怎样审核 | [AI辅助参赛](docs/ai-assisted-competition.md)、[字段时点表](templates/feature-availability.md) |
| 选题比较、我们的判断与改判条件 | [从选题到分析](docs/retrospectives/topic-selection-analysis.md) |
| 六个参赛路线案例、失败原因和迁移建议 | [比赛复盘分析专项](docs/retrospectives/README.md) |
| 从选题到赛后复盘的具体动作 | [跨案例参赛路线](docs/retrospectives/playbook.md) |
| 复盘时记录证据、根因与决策 | [复盘模板](templates/postmortem.md)、[决策日志](templates/decision-log.csv) |
| 赛事比较、候选平台 | [赛事总览](docs/competitions/overview.md) |
| Kaggle 入门和方案检索 | [Kaggle](docs/competitions/kaggle.md) |
| 2026 海豚杯与辽宁赛的核查记录 | [海豚杯](docs/competitions/dolphin-cup.md) |
| 长风杯往届题目和准备方法 | [长风杯](docs/competitions/changfeng-cup.md) |
| 每条共享资源的用途、限制和阅读优先级 | [资源导读](resources/README.md) |
| 可筛选的链接目录与核查证据 | [catalog.json](resources/catalog.json) |
| 验证、特征、实验和报告的经验 | [参赛工作方法](docs/experience/workflow.md) |
| 学习计划、研究记录与报告 | [学习路线](docs/experience/roadmap.md)、[赛事卡](templates/competition-card.md)、[复现卡](templates/solution-review.md)、[实验日志](templates/experiments.csv)、[报告提纲](templates/report-outline.md) |
| 已检索的范围与仍需确认的事项 | [检索记录](resources/search-log.md) |

## 熟悉基础后再看的资源

1. [Datawhale competition-baseline](https://github.com/datawhalechina/competition-baseline)：中文比赛 baseline，适合先跑通一套流程。
2. [Datawhale 动手学数据分析](https://github.com/datawhalechina/hands-on-data-analysis)：数据观察、清洗、建模和评估。
3. [Kaggle Solutions](https://github.com/faridrashidi/kaggle-solutions)：按赛题找方案、讨论和代码的索引；[网页入口](https://kaggle.farid.one/)。
4. [2025 海豚杯三夏团队项目](https://github.com/qyyyy09315/DophinCup2025ofSanX)：文本处理、BERT 特征与多种分类模型；作者成绩声明尚未独立核实。
5. [长风杯心理状态可视化](https://github.com/1065374244/changfeng)：历史展示参考，当前根目录主要是前端编译产物，未见完整分析流程。

## 如何使用和更新

每次只选一场比赛、一份方案。填赛事卡，确定能用的数据与评分方式；填复现卡，记录实际运行命令和结果。新资源写入 `resources/catalog.json`，注明核查时间与依据，资料修订运行 `python scripts/check_catalog.py` 即可。计算改动先重建结果，再核对：

```bash
python examples/run_all.py
python scripts/validate_research.py
python -m unittest discover -s tests -v
```

本仓库收录链接和原创导读，不搬运第三方完整代码、论文或竞赛数据。上游代码的公开访问不等于允许用于当前比赛；引用和改编时分别检查许可证与赛规。未见获奖证明的作者成绩，统一标作作者自述。

独立计算检查从逐条数据重算指标，版本检查核对摘要和报告引用。检查范围见[说明](tests/README.md)。

写作遵循[仓库原则](AGENTS.md)：术语有例子，结论有依据，重复提醒集中写；不靠复杂算法、流程数量或主观打分制造进步感。
