# K-means：从四个数字开始理解

K-means常译为K均值聚类。K表示我们想分成几组，means表示用每组的均值作为中心。它根据你选的特征和距离分组，不会自己知道哪类客户重要，也不会自己判断仓库是否值得建设。

## 1. 用1、2、8、9手算分两组

这四个数字可以想成四位客户的某个单一特征。只用于理解算法，不能据此建立真实客户价值体系。

先指定K=2，故意把两个中心放在1和2：

| 轮次 | 分组前的中心 | 分到第0组 | 分到第1组 | 重新求均值 |
| --- | --- | --- | --- | --- |
| 1 | 1、2 | 1 | 2、8、9 | 1；19/3约为6.3333 |
| 2 | 1、6.3333 | 1、2 | 8、9 | 1.5；8.5 |
| 3 | 1.5、8.5 | 1、2 | 8、9 | 1.5；8.5，不再变化 |

每轮只有两个动作：把点分给最近的中心，再用组内均值更新中心。四个点到最终中心的平方距离之和为0.25+0.25+0.25+0.25=1。

手算程序：[05_kmeans_by_hand.py](../../examples/05_kmeans_by_hand.py)。实际运行轨迹：[trace.csv](../../examples/results/kmeans_hand/trace.csv)。

## 2. 有多个特征时，怎样算“近”？

客户A和B都有订单数、业务量两个特征，就像平面上的两个点。欧氏距离可以想成纸上直线距离。K-means通常让每个点更靠近自己组的中心，优化的是组内平方距离之和。

如果订单数差10，业务量差1000，直接计算距离时业务量很容易主导结果。客户特征练习先用StandardScaler：每列减去该列均值，再除以该列标准差。这个变换改变了“相似”的定义，需要有业务解释；不是数据越标准化越好。

尤其不能把地理坐标随意分别标准化：两个轴如果都是公里，原有距离有物理意义。我们的选址练习保留公里坐标，并把需求作为样本权重；它和客户特征分群的处理目的不同。

## 3. 我必须自己指定K吗？

这个算法需要你给K。练习比较2、3、4、5，不代表真实客户必然有3层。K越大，组内距离通常可以更小；不能只追求最低组内距离。

轮廓系数是一个几何参考：大致检查同组是否较近、与其他组是否较远，范围通常在-1到1。接近1表示在所选距离下分离较好，接近0表示重叠。它不是准确率，不是利润指标，也不能跨不同特征和距离直接比较来决定赢家。

还要看每组人数、画像是否稳定，以及是否真的需要分别采取行动。只有一个组或每人各一组时，轮廓系数不适用。

## 4. 代码里的几行是什么意思？

~~~python
# K=3是本次示范配置，不是自动发现了三个真实等级。
model = KMeans(n_clusters=3, n_init=10, random_state=42)
# scaled每行是一个客户，每列是一个已经明确含义的特征。
labels = model.fit_predict(scaled)
~~~

| 写法 | 含义 |
| --- | --- |
| n_clusters=3 | 要求分成3组 |
| n_init=10 | 用不同起始中心尝试10次，按组内平方距离保留结果 |
| random_state=42 | 固定初始化随机过程，便于复查 |
| fit_predict | 先学习中心，再返回这批对象所属组的编号 |
| cluster_centers_ | 学出来的中心坐标；标准化空间里的中心不是原始单位 |
| labels | 分组代号；0、1、2没有高低顺序 |

调用接口依据：[scikit-learn KMeans文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)、[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)。本仓库显式指定n_init，减少不同版本默认值造成的困惑。

## 5. 学完后研究什么？

打开[客户分群代码](../../examples/04_customer_segments.py)和[实际报告](../../examples/results/task04/report.md)。先看profiles.csv的原始单位，再看comparison.csv。我们还拿仅按业务量分档作对照。

如果两个分组很相似，先问复杂处理增加了什么。规则分组仍可能有用，不能因为简单就丢弃。

ARI用来比较两份分组是否把同样的人放在一起，编号互换不会影响它；1表示分组完全对应。本练习拿它检查不同初始化与规则对照，不把它叫作真实类别准确率。只测初始化，不代表已经检验时间变化或抽样变化。

## 6. 常见误解

K-means会按指定K分组，即使现实里没有清楚的群体。离群值、尺度、所选特征和起始中心都可能影响结果。复杂、不规则的群体未必适合它；正式赛题规定该方法时先完成要求，再按允许范围比较扩展。

二维图只显示两个特征，本练习实际上用了三个；图上重叠不一定等于模型错误，图上分开也不等于业务有效。

继续读：[数据与验证](data-and-validation.md)。进一步参考：[轮廓系数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)、[ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)、[官方课程](https://inria.github.io/scikit-learn-mooc/python_scripts/clustering_kmeans.html)。

以上技术页面核查于2026-10-08；四数字与客户教学代码为本仓库原创。
