# 检索范围与核查记录

核查时间：2026-10-07，北京时间。

## 做了哪些检索

公开网页：海豚杯2026、长风杯2026、辽宁赛事与校内通知、Kaggle方案集、Datawhale baseline、M5及推荐系统方案。

GitHub仓库搜索：海豚杯、长风杯、kaggle solutions、datawhale competition。中文关键词的扩展搜索返回大量无关项目，未将这些噪声加入目录。

直接读文件：两个国内杯赛项目的README与根目录，海豚杯LICENSE、长风杯index.html；Kaggle两个索引与Datawhale三个教程/方案README；M5 README、Web Traffic Readme.md、NVIDIA RecSys2021 README。其余资料按catalog的verification字段标注读取范围。

此次找到一份直接相关海豚杯代码项目、一份直接相关长风杯展示项目。这个数量只代表本次检索结果，不能推断其他共享项目不存在。

## 核查中发现的区别

- 海豚杯仓库名称是 `DophinCup2025ofSanX`，不是标准拼写 `Dolphin`。保留真实路径。
- Web Traffic 的说明文件是 `Readme.md`，不能把 `README.md` 的404误判为仓库消失。
- `dayeren/Kaggle_Competition_Treasure` 与 `team-learning-data-mining` 没有根 README，但根目录可读。
- 尝试的 `datawhalechina/easy-sql` 路径返回404，未列入推荐。
- 长风杯展示仓库已读到的是编译产物，不能假定有开发源码或可复现分析。
- 高校通知有适用范围和日期差异；未把明显笔误改写成官方结论。

## 下一次应补齐

1. 海豚杯官网正文与2026操作指南中的赛题、评分表和提交格式。
2. 你所在学校的报名 / 推荐状态，是否允许补报名或已进入后续阶段。
3. 长风杯2026当届规程；没有它就不填写本届截止日。
4. 选定题目后审读相应代码，核查许可证、数据授权、标签处理与验证方法。
5. 实际复现至少一份简单baseline，记录环境、资源和结果。

## 发布记录

2026-10-08：已在用户确认的 `zhouwu97` 账户下创建私有仓库 `ai-data-competition-research`，通过网页上传资料。研究内容核查日期仍为2026-10-07。
