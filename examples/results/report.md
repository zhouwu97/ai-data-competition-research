# 四类教学练习：实际执行结果

2026-10-08建立的原创教学练习。全部输入是合成数据，没有取得正式赛题数据，没有提交比赛或证明获奖效果。

| 证据编号 | 本次结果 | 数值 | 结果文件 |
| --- | --- | --- | --- |
| C01 | 题一14天回测窗口数 | 4 | [task01/summary.json](task01/summary.json) |
| C02 | 题二出库总件数 | 41 | [task02/summary.json](task02/summary.json) |
| C03 | 题三演示k=3的需求加权距离（公里） | 16.1933 | [task03/summary.json](task03/summary.json) |
| C04 | 题四演示k=3的轮廓系数 | 0.6326 | [task04/summary.json](task04/summary.json) |

这些量回答不同问题，不能据此排列四个赛题的获奖机会。选题表中的真实数据和评分证据仍是待取得状态。

详细记录：[题一](task01/report.md)、[题二](task02/report.md)、[题三](task03/report.md)、[题四](task04/report.md)、[错误示范](failure_cases/report.md)。

复查：[命令日志](run.log)、[版本与SHA256](run_manifest.json)、[结论对应数值](claims.json)。全部步骤实际返回0后，才会写入本汇总。
