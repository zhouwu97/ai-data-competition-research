# 专项来源与证据

核查日期：2026-10-08。下表固定到已读取的提交，避免分支更新后无法定位。逐一重新读取固定版本并核对文件 blob SHA，共 15 个文件。完整值见[机器清单](../../resources/retrospective-sources.json)。

| 证据 ID | 仓库 / 固定文件 | 文件 blob SHA（前12位） |
| --- | --- | --- |
| ev-log | [avoid137/kaggle-s6e9-postmortem / docs/experiment-log.md](https://github.com/avoid137/kaggle-s6e9-postmortem/blob/e156407799d789f2bb30a989a5802caec099ce5b/docs/experiment-log.md) | 0113fd22daf3 |
| metric-log | [msusol/kaggle-playground-series-s6e7 / docs/plans/leaderboard.md](https://github.com/msusol/kaggle-playground-series-s6e7/blob/80422e8a096284748524a1407c326109a2560986/docs/plans/leaderboard.md) | 38efb52ebd89 |
| hms-readme | [zlin7dev/kaggle-3 / README.md](https://github.com/zlin7dev/kaggle-3/blob/9d9502ba8841bcdf81d3313e7920259d4b2abe59/README.md) | ba77ac13b9ff |
| hms-filter | [zlin7dev/kaggle-3 / scripts/filter_train.py](https://github.com/zlin7dev/kaggle-3/blob/9d9502ba8841bcdf81d3313e7920259d4b2abe59/scripts/filter_train.py) | 2498aa6e288e |
| polymer-readme | [jday96314/NeurIPS-polymer-prediction / README.md](https://github.com/jday96314/NeurIPS-polymer-prediction/blob/a7a2bc90e6f5e6f1f3e00ab556e4f7ff555aae53/README.md) | 5a99d5c6b91c |
| polymer-run | [jday96314/NeurIPS-polymer-prediction / train_winning_ensemble.sh](https://github.com/jday96314/NeurIPS-polymer-prediction/blob/a7a2bc90e6f5e6f1f3e00ab556e4f7ff555aae53/train_winning_ensemble.sh) | df401edc4cf0 |
| dolphin-readme | [qyyyy09315/DophinCup2025ofSanX / README.md](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/README.md) | 49b393a48c1d |
| dolphin-clean | [qyyyy09315/DophinCup2025ofSanX / src/preprocessing/clean_data.py](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/preprocessing/clean_data.py) | fde4aedeb911 |
| dolphin-bert | [qyyyy09315/DophinCup2025ofSanX / src/feature_engineering/bert_embedding/bert_convert_efficient.py](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/feature_engineering/bert_embedding/bert_convert_efficient.py) | 08c265aa34ed |
| dolphin-stack | [qyyyy09315/DophinCup2025ofSanX / src/models/ensemble/stacking/st_brf_xgboost_catboost_stacking.py](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/models/ensemble/stacking/st_brf_xgboost_catboost_stacking.py) | 332ccc962c28 |
| dolphin-label | [qyyyy09315/DophinCup2025ofSanX / src/preprocessing/label_detection.py](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/preprocessing/label_detection.py) | 2b1b2fdc2e56 |
| dolphin-synthetic | [qyyyy09315/DophinCup2025ofSanX / src/preprocessing/synthetic_data_generator.py](https://github.com/qyyyy09315/DophinCup2025ofSanX/blob/5e498d0536693e1915aac62835986675fc8e0169/src/preprocessing/synthetic_data_generator.py) | a55447a9ea9e |
| changfeng-index | [1065374244/changfeng / index.html](https://github.com/1065374244/changfeng/blob/fc8499803641c683d6949e6c9795405f5a4ab86e/index.html) | 0e958c4e762e |
| ev-readme | [avoid137/kaggle-s6e9-postmortem / README.md](https://github.com/avoid137/kaggle-s6e9-postmortem/blob/e156407799d789f2bb30a989a5802caec099ce5b/README.md) | eda8b7318630 |
| changfeng-readme | [1065374244/changfeng / README.md](https://github.com/1065374244/changfeng/blob/fc8499803641c683d6949e6c9795405f5a4ab86e/README.md) | 57126ea6e833 |

另读取国内两仓库根目录、海豚杯递归目录及聚合物根目录，核查文件组织；目录观察不等于全仓库代码审计。HMS 和聚合物称第 3 / 第 1 名，属于作者声明。本轮没有独立成绩证明，没有下载比赛数据，没有安装或执行训练入口。

## 分析边界

- 分数表和参赛路线来自作者记录；本仓库提出的迁移、改进和复现计划是原创分析。
- S6E9 的根因解释未经过建议修复后的实验验证；不照搬“线性免疫”或其他过强结论。
- S6E7 的固定增益门槛不构成普遍显著性检验；换模型与换早停指标同时发生时，不能只归因于指标。
- 国内杯赛两作品不包含可核验的完整时间线；海豚杯为局部代码审读，长风杯为交付完整性审读。
- 不完整材料不能证明作者没做相关工作；公开代码也不能代替当届规则和数据授权。

## 检索与筛选

本轮检索参赛者 postmortem、experiment log、leaderboard、winning solution 和两项国内杯赛的 GitHub 分享。中文杯赛搜索有大量同名、书籍和无关项目，未收录；也排除了金海豚、配音等不同赛事。六个案例按材料可追溯性与方法覆盖选择，不按星数或声明名次排序。
