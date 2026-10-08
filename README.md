# AI / 数据分析竞赛预研

面向个人与算法社团的竞赛资料库：Kaggle、海豚杯、长风杯，以及天池、讯飞等可继续关注的平台。

资料按条目记录核查日期；赛事入口主要核查于 **2026-10-07**，比赛路线与复盘专项更新于 **2026-10-08**。这里汇总公开入口、共享方案和参赛方法；实际报名资格、赛题、日程与提交格式以当届规程为准。

## 先看结论

- **练基础：** 从 Kaggle Titanic 或 House Prices 做一次数据读取、验证、建模、提交，再用 Datawhale 补清洗与分析。
- **准备海豚杯：** 先确认数智分析还是应用创新赛道，再准备数据依据、实验、报告和展示。2026 辽宁赛通知已找到，日期存在来源差异，见赛事档案。
- **准备长风杯：** 2025 辽宁本科生赛道已有四个指定题目的公开说明，可练需求预测、仓储分析、选址和客户分群；本次尚未确认 2026 届完整规程。
- **选代码：** 先复现简单 baseline，再读高排名方案。当前找到的海豚杯代码项目与长风杯展示项目，完整程度差别很大。

上述学习顺序是本仓库的建议，不是赛事规定。不要将训练赛排名当作正式竞赛奖项，也不要把往届要求套到新一届。

## 阅读导航

| 内容 | 文件 |
| --- | --- |
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
| 从零做起的四周计划 | [学习路线](docs/experience/roadmap.md) |
| 赛事信息收集模板 | [赛事卡](templates/competition-card.md) |
| 阅读并复现别人方案 | [方案复现卡](templates/solution-review.md) |
| 实验与展示 | [实验日志](templates/experiments.csv)、[报告提纲](templates/report-outline.md) |
| 已检索的范围与仍需确认的事项 | [检索记录](resources/search-log.md) |

## 最值得先打开的五条

1. [Datawhale competition-baseline](https://github.com/datawhalechina/competition-baseline)：中文比赛 baseline，适合先跑通一套流程。
2. [Datawhale 动手学数据分析](https://github.com/datawhalechina/hands-on-data-analysis)：数据观察、清洗、建模和评估。
3. [Kaggle Solutions](https://github.com/faridrashidi/kaggle-solutions)：按赛题找方案、讨论和代码的索引；[网页入口](https://kaggle.farid.one/)。
4. [2025 海豚杯三夏团队项目](https://github.com/qyyyy09315/DophinCup2025ofSanX)：文本处理、BERT 特征与多种分类模型；作者成绩声明尚未独立核实。
5. [长风杯心理状态可视化](https://github.com/1065374244/changfeng)：历史展示参考，当前根目录主要是前端编译产物，未见完整分析流程。

## 如何使用和更新

每次只选一场比赛、一份方案。填赛事卡，确定能用的数据与评分方式；填复现卡，记录实际运行命令和结果。新资源写入 `resources/catalog.json`，注明核查时间与依据，再运行：

```bash
python scripts/check_catalog.py
```

本仓库收录链接和原创导读，不搬运第三方完整代码、论文或竞赛数据。上游代码的公开访问不等于允许用于当前比赛；引用和改编时分别检查许可证与赛规。未见获奖证明的作者成绩，统一标作作者自述。

GitHub 仓库：[zhouwu97/ai-data-competition-research](https://github.com/zhouwu97/ai-data-competition-research)，私有。2026-10-08 新增复盘专项：六个案例、参赛路线、复盘模板与15个固定版本来源。仅做资料和局部代码审读，尚未运行上游训练。
