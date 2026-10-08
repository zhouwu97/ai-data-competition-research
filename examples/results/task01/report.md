# 题一教学实验记录

全部数据由00_make_data.py生成，无真实比赛成绩。

| 方法 | 四个窗口平均MAE（件/天） |
| --- | --- |
| calendar_ridge | 4.8184 |
| recent_mean | 16.5944 |
| weekday_mean | 9.8571 |

数据含人为设置的周规律和线性趋势，适合日历模型；这不证明该模型在真实赛题优于其他方法。平均值之外，请查看metrics.csv的逐窗口结果。没有库存成本、缺货或补货决策数据，不能推导库存节省。

证据：[区域-SKU统计](regional_summary.csv)、[逐窗口指标](metrics.csv)、[逐日预测](predictions.csv)、[图表](forecast.svg)、[汇总](summary.json)。
