# 可运行练习：从输入到结果

全部数据由本仓库原创生成，是合成教学数据，人物与城市均为虚构。没有官方赛题数据，没有比赛提交或获奖验证。先读[新手入口](../docs/beginner/README.md)，安装步骤见[运行说明](../docs/beginner/python-first-steps.md)。

## 运行

从仓库根目录使用已装依赖的Python：

~~~bash
python examples/05_kmeans_by_hand.py
python examples/run_all.py
python scripts/check_catalog.py
python scripts/validate_research.py
python -m unittest discover -s tests -v
python scripts/test_validate_research.py
~~~

run_all会重建教学输入和结果，不要在这些目录存正式数据。[requirements.txt](requirements.txt)是本次实际运行的依赖版本。

现在 `run_all.py` 还会执行四类困难情景，再用独立公式检查基础指标。先读[困难情景报告](results/stress/report.md)与[决策案例](../docs/research/decision-cases.md)，看结果如何影响方法选择。

## 文件与研究问题

| 程序 | 输入与分析 | 先研究什么 | 结果 |
| --- | --- | --- | --- |
| [00_make_data.py](00_make_data.py) | 创建5份合成CSV | 字段、单位、造数据假设 | sample_inputs目录 |
| [05_kmeans_by_hand.py](05_kmeans_by_hand.py) | 四数字、两中心 | 为什么分组与均值要循环 | [轨迹](results/kmeans_hand/trace.csv) |
| [01_demand_forecast.py](01_demand_forecast.py) | 全国汇总的14天回测 | 同星期基线为什么比近期均值更有针对性 | [报告](results/task01/report.md) |
| [02_warehouse_eiq_abc.py](02_warehouse_eiq_abc.py) | 每日订单与品项统计 | EN为什么不按行数算，ABC阈值改变什么 | [报告](results/task02/report.md) |
| [03_dc_locations.py](03_dc_locations.py) | 虚构平面选址 | 城市覆盖与需求覆盖为何不同 | [报告](results/task03/report.md) |
| [04_customer_segments.py](04_customer_segments.py) | 运单聚合与客户分群 | 聚类是否增加了规则分组之外的信息 | [报告](results/task04/report.md) |
| [failure_cases.py](failure_cases.py) | 故意构造错误 | 检查应如何失败，0误差为什么也会骗人 | [记录](results/failure_cases/report.md) |
| [06_method_stress.py](06_method_stress.py) | 趋势突变、周期波动、目标冲突与客户变化 | 方法假设失效时怎样改判 | [压力报告](results/stress/report.md) |
| [common.py](common.py) | 共用读写、图与输入检查 | 第一遍可跳过 | 被各练习调用 |
| [run_all.py](run_all.py) | 顺序执行并记录产物 | 只有全部成功才写汇总 | [汇总](results/report.md) |

注释重点解释业务对象、处理理由和结论边界，不仅翻译函数名。SVG图的轴使用英文，中文含义在对应报告与代码里说明；不用安装额外中文字体。

## 输入数据字典

| CSV | 一行代表什么 | 关键字段与单位 |
| --- | --- | --- |
| demand.csv | 某日-区域-商品 | date日期；region区域；sku品项；quantity件数 |
| warehouse_orders.csv | 订单中的一个品项明细 | date出库日；order_id订单；sku品项；quantity件数；允许拆行 |
| cities.csv | 一个虚构城市 | x_km/y_km平面公里；demand正需求权重；不是经纬度 |
| waybills.csv | 一张虚构运单 | customer_id客户；waybill_id运单；date记录日；quantity件数 |
| business_volume.csv | 同一观察周期的一位客户 | business_volume汇总件数；在本例应与运单合计一致 |

## 从输出读回代码

先打开报告，再找到结果表列名，然后搜索生成该列的代码。最后回到输入核对一两个对象。

例如题二第一天EQ合计15、IQ合计15；全期合计41。题四cluster编号不代表价值高低，profiles.csv用原始单位解释。题三演示k=3，不宣称最优。题一日历模型适合人为设置的趋势，不能推广为真实比赛结论。

## 自己动手修改一次

保存原结果后，只改变一个条件：题一改历史窗口，题二改ABC阈值，题三改覆盖半径，题四改特征。先写预期再运行，保留成功和失败。数据或代码变更后重新运行run_all，摘要检查才会对应新版本。

这些修改是教学探索，不能对已见过结果的合成数据声称严格盲分析或获奖创新。
