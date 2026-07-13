# 模型优化记录

日期：2026-07-04

## 优化目标

计划书第六节中，当前最难满足的是：

```text
错误音素召回率：在 precision 不低于 0.40 时 recall 不低于 0.50
```

此前模型能获得较高错误 recall，但错误 precision 很低；继续优化的重点是降低误报。

## 已尝试方法

### 1. SpeechOcean 自训练 GOP 等价声学模型

使用 SpeechOcean train split 的正确音素段训练 39 个对角高斯音素模型，然后在 dev/test 上计算目标音素与竞争音素的似然差。

最佳融合结果：

- Balanced Accuracy：0.623299；
- Macro-F1：0.414365；
- AUC：0.671143；
- 错误 Precision：0.063697；
- 错误 Recall：0.686047。

结论：recall 高，但误报过多。

### 2. 声学 GOP + 表格特征融合

将 `gop_score`、`target_log_likelihood`、`competitor_log_likelihood` 等声学特征并入 SpeechOcean 建模脚本。

最佳结果：

- Balanced Accuracy：0.675260；
- Macro-F1：0.452763；
- AUC：0.748487；
- 错误 Precision：0.075341；
- 错误 Recall：0.718287。

结论：排序能力明显提升，AUC 达标，但错误 precision 仍低。

### 3. Macro-F1 阈值优化

使用声学融合特征，以 Macro-F1 为 dev 阈值目标。

最佳结果：

- Balanced Accuracy：0.643513；
- Macro-F1：0.553569；
- AUC：0.712928；
- 错误 Precision：0.120339；
- 错误 Recall：0.412960。

结论：Macro-F1 和 AUC 达标，但错误 precision 仍低。

### 4. 监督式音素段声学分类器

新增 `scripts/run_segment_acoustic_classifier.py`，为每个音素段抽取 log-Mel + energy 的均值/方差统计量，训练 Logistic Regression、Random Forest 和 ExtraTrees。

最佳 Balanced Accuracy 结果：

- 模型：ExtraTrees；
- Balanced Accuracy：0.678004；
- Macro-F1：0.483555；
- AUC：0.734815；
- 错误 Precision：0.653487；
- 错误 Recall：0.083951。

precision 约束结果：

- 模型：ExtraTrees；
- Balanced Accuracy：0.666270；
- Macro-F1：0.564582；
- AUC：0.734815；
- 错误 Precision：0.456343；
- 错误 Recall：0.133280。

结论：已经可以获得高 precision 错误报警，但 recall 不足。

## 当前最优验收口径

如果强调总体排序和召回：

- 使用 `reports/phase1_acoustic_fusion/model_comparison.csv`；
- AUC 0.748，BA 0.675，错误 recall 0.718，但 precision 低。

如果强调高置信错误报警：

- 使用 `reports/phase1_segment_acoustic/model_comparison.csv`；
- Macro-F1 0.565，AUC 0.735，错误 precision 0.456，但 recall 0.133。

## 仍未完全达标的原因判断

1. SpeechOcean 错误样本占比低，test 中错误类约 4%，极易出现误报。
2. 公开数据的音素分数标签和真实“可教学纠错错误”并不完全一致。
3. 自动对齐虽整体可用，但边界误差会影响短音素的声学特征。
4. 冻结 wav2vec2 pilot 没有针对发音正确性做微调，效果未超过传统声学特征。

## 建议下一步

1. 人工复核一批高置信 false alarm 和 missed error，确认是模型错还是原始标签口径过宽。
2. 将训练目标从全音素整体模型改为核心音素专项模型，例如 /r-l/、/v-w/、/th/、元音组。
3. 对错误样本做更强的数据增强或重采样。
4. 如果有 GPU，再尝试 wav2vec2/WavLM 的监督微调，而不是只用冻结 embedding。
5. 若要严格达到 precision≥0.40 且 recall≥0.50，可能需要内部专家复核标签或更多真实错误样本。
