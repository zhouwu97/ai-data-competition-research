# 添加资料

写作和实现遵循[仓库原则](AGENTS.md)：先说问题与证据，术语给例子；限制集中说明，避免反复提醒或增加无用流程。

先确认赛事身份和年份，再添加资源。每条至少包含 URL、类型、用途、限制、核查日期和已读依据。

名次注明作者声明或官方证据；入口失效与内容不完整分别标记。新增资源不能仅由星数、标题或 README 宣传判断质量。

资料修订运行 `python scripts/check_catalog.py`。计算改动运行 `python examples/run_all.py`、`python scripts/validate_research.py`和 `python -m unittest discover -s tests -v`；元数据测试仍见 `python scripts/test_validate_research.py`。独立重算与版本一致性分别检查，说明见[检查入口](tests/README.md)。

教学源码改动后运行 `python examples/run_all.py`，提交输入、结果、报告、日志与摘要。注明合成或真实数据，未执行不能写成实测；不以合成结果推断比赛获奖机会。

术语第一次出现给白话说明或新手文档链接。第三方代码与数据只收录链接；如确需引入代码，保留其许可和引用，另核查赛规。
