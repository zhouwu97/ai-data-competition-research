# Kaggle：从完整提交开始

## 入口

- [竞赛列表](https://www.kaggle.com/competitions)
- [竞赛文档](https://www.kaggle.com/docs/competitions)
- [Titanic](https://www.kaggle.com/competitions/titanic)
- [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)

官方将 Getting Started 用于入门学习。此次部分官方页面只返回动态页面壳，因此具体规则、每日提交次数、算力配额不在本仓库写死。

## 推荐做法（本仓库建议）

第一题用 Titanic：读清目标、样本粒度和输出格式，建立本地验证，比较一个常数或多数类基线与简单模型，提交后记录本地分数与榜单分数。

第二题用 House Prices：练数值缺失、类别编码和回归误差。模型选择从线性模型和树模型对比开始；如果对目标取对数，必须让本地评估与竞赛定义一致，不能混用变换前后指标。

第三题按方向选择：中文文本分析可先用 TF-IDF 分类练习，再看预训练模型；时间序列可读 M5 方案，先建立简单季节基线与时间切分，再考虑复杂网络。

## 如何找别人分享

先在 [faridrashidi/kaggle-solutions](https://github.com/faridrashidi/kaggle-solutions) 搜同类问题，再进入作者讨论或代码仓库。[EliotAndres/kaggle-past-solutions](https://github.com/EliotAndres/kaggle-past-solutions) 可作为补充历史索引。

读到名次声明时，区分作者自述与官方结果；读到代码时，确认它是最终模型、简化版还是赛后改进版。首次复现不要把大型集成的耗时当作自己的投入预算。

## 推荐记录

每次提交记录：对应 Git 提交、切分方式、随机种子、指标、训练时间、内存、提交文件、榜单反馈。调参用训练 / 验证数据；最后留出集尽量少看，避免把它调成另一个训练集。

## 规则核对

单场赛事卡至少填：资格、队伍人数、截止日、评估方式、外部数据与预训练模型、在线 API、公开分享与 AI 辅助政策、最终代码与复现要求。官方未说明的字段标待确认，不依据其他赛题猜测。
