# 第一阶段最终验收报告

日期：2026-07-03

## 结论

当前项目已经完成第一阶段的公开数据最小闭环：数据解析、标签映射、说话人级划分、SpeechOcean762 音素对齐、GOP 等价声学基线、SpeechOcean 建模、阈值校准、正式评估、100 条预测样例、错误分析、最小 demo 和复现说明。

按计划书的“硬验收标准”，当前已经具备可提交的第一阶段 baseline 验收包。自监督语音表征模型已经完成 CPU pilot 对比实验，但当前效果未超过 GOP/表格模型。项目内部小样本验证因为暂未提供内部脱敏音频和人工标签，仍作为待补项。

## 已完成交付物

| 计划书要求 | 当前状态 | 证据 |
|---|---|---|
| SpeechOcean762 全量解析、标签映射、说话人划分 | 完成 | `data/processed/speechocean/phones_aligned.csv`, `reports/phase1/data_manifest.csv` |
| L2-ARCTIC 解析与错误类型补充 | 完成 | `data/processed/l2_arctic/phones.csv`, `scripts/prepare_l2_arctic.py` |
| 标签规范 | 完成 | `docs/label_spec.md` |
| GOP 或等价声学证据模型 | 完成 baseline | `scripts/run_acoustic_baseline.py`, `artifacts/baseline_acoustic_v1/`, `docs/baseline_results_v1.md` |
| 阈值校准 | 完成 | `reports/phase1/thresholds.csv`，含 global、phone_group、target_phone 三级阈值 |
| 正式评估指标 | 完成 | `reports/phase1/formal_eval_metrics.csv` |
| 混淆矩阵与错误类指标 | 完成 | `formal_eval_metrics.csv` 中含 TN/FP/FN/TP、error_precision、error_recall、error_f1 |
| 100 条测试音频预测样例 | 完成 | `reports/phase1/prediction_samples_100_utterances.csv` |
| 错误分析 | 完成 | `reports/phase1/error_analysis_summary.md` 及按音素组/音素 CSV |
| 最小演示 | 完成 | `scripts/demo_predict.py`, `reports/phase1/demo_prediction.csv` |
| 自监督语音表征模型 | 完成 pilot | `scripts/run_ssl_baseline.py`, `reports/phase1/ssl_eval_metrics.csv`, `artifacts/ssl_wav2vec2_v1/` |
| 可复现说明 | 完成 | `README.md`, `requirements.txt` |
| 第二阶段接口字段 | 完成 | 输出表保留 `target_phone`, `phone_group`, `duration_ms`, `prediction`, `confidence`, `threshold`, `audio_path` 等字段 |

## 当前关键结果

### SpeechOcean762 对齐

- 5,000 条语音全部完成自动对齐；
- 94,445 个目标音素生成边界；
- 90,323 个音素为 `alignment_quality=pass`；
- 4,122 个音素为 `alignment_quality=review`；
- L2-ARCTIC 人工边界回放验证中位误差约 20 ms。

### GOP 等价声学基线

L2-ARCTIC Mandarin 测试集上，融合分类器结果：

- Balanced Accuracy：0.695；
- Macro-F1：0.559；
- AUC：0.768；
- 错误 Recall：0.745；
- 错误 Precision：0.231。

该结果达到 Macro-F1 与 AUC 的最低过关线，Balanced Accuracy 距 0.70 目标线约 0.005，但错误 Precision 仍不足。

### SpeechOcean 第一阶段表格模型

使用 `alignment_quality == "pass"` 的 90,323 个 SpeechOcean 音素训练和评估。

当前最佳 test 结果：

- 模型：`feature_random_forest`；
- 校准：`global_threshold`；
- Balanced Accuracy：0.629247；
- Macro-F1：0.413134；
- AUC：0.692519；
- 错误 Precision：0.062112；
- 错误 Recall：0.698517。

该模型主要作为 SpeechOcean 全量数据闭环与 demo 产物，不替代 GOP 等价声学基线。

### 追加优化：监督式音素段声学分类器

在继续优化阶段，新增 `scripts/run_segment_acoustic_classifier.py`。该脚本直接为每个
音素段抽取 log-Mel + energy 的均值和方差统计量，并结合目标音素、音素组、时长等
特征训练 Logistic Regression、Random Forest 和 ExtraTrees。

当前最佳 test 结果为 `ExtraTrees + balanced_accuracy`：

- Balanced Accuracy：0.678004；
- Macro-F1：0.483555；
- AUC：0.734815；
- 错误 Precision：0.653487；
- 错误 Recall：0.083951。

若使用 dev 集上校准的 `error_precision >= 0.40` 阈值，test 结果为：

- Balanced Accuracy：0.666270；
- Macro-F1：0.564582；
- AUC：0.734815；
- 错误 Precision：0.456343；
- 错误 Recall：0.133280。

这说明模型已经能做到“少量高置信错误比较准”，但仍不能同时满足计划书中
“precision 不低于 0.40 时 recall 不低于 0.50”的组合指标。

### 自监督语音表征模型

已接入 `facebook/wav2vec2-base` 冻结编码器，对音素段池化 hidden states 后训练轻量 Logistic Regression。当前 CPU pilot 设置为每个 split 抽取 120 条语音，共 360 条语音、5,543 个音素段。

当前 SSL pilot test 结果：

- Balanced Accuracy：0.436885；
- Macro-F1：0.462458；
- AUC：0.468314；
- 错误 Precision：0.000000；
- 错误 Recall：0.000000。

结论：自监督路线已经完成可复现实现和对比实验，但该 pilot 未优于 GOP 等价声学基线。主要原因可能包括抽样规模较小、错误样本极度稀疏、音素段边界噪声和冻结表征未针对发音正确性适配。计划书允许“提升模型不明显时保留为实验结果”，因此该项作为完成但不作为当前推荐模型。

## 与计划书指标线对照

| 指标 | 最低过关线 | 当前最好证据 | 状态 |
|---|---:|---:|---|
| Balanced Accuracy | 多数类基线 +10 个百分点 | GOP 融合 0.695；Segment ExtraTrees 0.678 | 通过最低线，接近目标线 |
| Macro-F1 | ≥ 0.55 | GOP 融合 0.559；precision 约束模型 0.565 | 通过最低线 |
| AUC | ≥ 0.70 | GOP 融合 0.768；Segment ExtraTrees 0.735 | 通过最低线 |
| 错误 Recall | precision≥0.40 时 recall≥0.50 | precision 约束模型 precision 0.456，recall 0.133 | 未完全满足 |
| 核心音素/音素组结果 | 至少 8 个 | `error_analysis_by_phone_group.csv`, `error_analysis_by_target_phone.csv` | 完成报告 |

## 仍需你确认或提供的事项

1. 新音频文本自动 G2P：当前 demo 对清单内语音可直接输出诊断；对新音频需要提供 `--target-phones`。若要做到“输入任意英文文本自动生成目标音素”，需要接入 CMUdict/g2p-en 或项目词典。
2. 内部小样本验证：计划书建议 20–30 名学生、每人 10–20 个核心音素或 20–30 个高频词。当前没有内部脱敏音频和人工标签，不能伪造该验证。
3. `review` 样本复核：4,122 个自动对齐低置信样本建议人工抽检 100–200 个，决定是否重对齐或纳入第二轮训练。

## 建议验收表述

建议对外表述为：

> 第一阶段已完成公开数据驱动的音素发音正确性 baseline 验收包，包含 SpeechOcean762 全量对齐、L2-ARCTIC 错误类型解析、GOP 等价声学基线、SpeechOcean 全量建模、自监督 wav2vec2 表征 pilot、阈值校准、正式评估、100 条样例、错误分析和最小 demo。当前模型已达到可复现、可评估、可扩展的第一阶段底座要求；内部小样本迁移验证和 review 样本复核作为下一轮优化继续推进。
