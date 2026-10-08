# 遇到问题再读的七份资料

核查日期：2026-10-08。下面是直接读过的官方教程或作者论文，按问题挑一份；不用先读完。scikit-learn链接固定到1.8，与教学程序的依赖版本对应。

| 你遇到的问题 | 资料与优先阅读部分 | 能支持什么判断 | 读完做什么 |
| --- | --- | --- | --- |
| 增加配送中心，为什么覆盖率未必增加？ | [scikit-learn：K-means目标](https://scikit-learn.org/1.8/modules/clustering.html#k-means)，读2.3.2目标公式、局部最小值、sample weights | K-means减小的是簇内平方距离；它的目标不是服务半径内覆盖量。需求加权也不改变这一点 | 同时列平方距离、平均距离、城市覆盖和需求覆盖；看是不是目标冲突 |
| 预测模型以前很好，后来突然失效？ | [Lu等：Learning under Concept Drift: A Review](https://arxiv.org/html/2004.05785v1)，作者预印本2020-04-13；先看II定义与突变/渐变图，再看V-A及VI评估 | 数据关系变化会使过去学到的规律不再适用；重训、窗口和适应机制需要在变化后的时段比较 | 将误差按回测窗口画出来，比较长历史与近期数据；保留变化前后的结果，而非只报一个均值 |
| 时间序列的分数好得异常？ | [scikit-learn：Lagged features for time series forecasting](https://scikit-learn.org/1.8/auto_examples/applications/plot_time_series_lagged_features.html)，读Naive evaluation、TimeSeriesSplit和Conclusion | 该官方示范展示随机切分造成过于乐观的结果，时间切分更贴近此示范的预测任务 | 标出预测时点和特征可用时点；两周预测还要确认窗口内不能取得的真实值有没有混入特征 |
| 每日ABC变了，货就要每天搬吗？ | [Xu与Ren：Dynamic Storage Location Assignment](https://onlinelibrary.wiley.com/doi/10.1155/2020/1621828)，2020-11-29；读第3节、第5.2.2节滚动热度、第5.2.3节节省距离 | 文中在特定仓型和路线下考虑小步调整、需求变化与额外作业；搬移有成本，不能把分类变化直接等同收益 | 比较固定、每日与滚动策略的未来拣选代价和搬移次数；只有已知布局或明确模拟假设才能计算距离收益 |
| K-means给出彩色分组，怎样看出不合适？ | [scikit-learn：Demonstration of k-means assumptions](https://scikit-learn.org/1.8/auto_examples/cluster/plot_kmeans_assumptions.html)，先看四幅图及Fit models | 拉长形状、方差不同等数据会产生不直观分组；增加初始化只能处理部分局部解问题 | 画特征分布，加入弱分群或异常客户情景，再与规则分档比较；不要因为图有颜色就认定发现了三类真实客户 |
| 轮廓系数均值高，分群就能用于业务吗？ | [scikit-learn：Silhouette analysis](https://scikit-learn.org/1.8/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)，读开头、不同K的轮廓图和各群大小 | 轮廓系数描述所用特征空间的分离程度，均值可能掩盖群内差异；该示范在2与4之间也有歧义 | 查看每群样本与特征，再问分组是否改变服务行动。指标不能直接换算成利润或客户价值 |
| 服务覆盖才是目标，怎样比较另一条路线？ | [PySAL/spopt：Maximal Coverage Location Problem](https://pysal.org/spopt/notebooks/mclp.html)，示范更新2025-04-07；读开头目标、cost matrix与欧氏/路网距离对照 | MCLP给定候选设施、数量及服务阈值，以覆盖需求为目标；距离口径会影响结果 | 先在小候选集穷举并对照K-means。2025题三要求的K-means仍须完成，覆盖模型作为有问题证据后的扩展 |

上面最后一列是本仓库的练法，不是论文已验证的长风杯方案。仓储论文研究的是特定picker-to-parts仓型，概念漂移论文主要讨论流式学习；迁移到自己的数据时，先明确实验对象与评价口径。

## 怎样把资料变成自己的发现

记下一个问题即可：**原方法哪里失败，另一条路线改变了哪个目标或假设？** 然后做最小对照，记录会让自己改判的结果。先不用增加模型数量，也不用把每份资料写成长复盘。

赛规与数据获取见[规则入口](../../competition-requirements/README.md)；术语不懂时返回[零基础学习入口](../beginner/README.md)。
