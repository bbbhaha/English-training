# wav2vec2-MDD 全量正式结果

日期：2026-07-05

## 数据

使用 `data/processed/mdd/speechocean_manifest.csv`，仅包含 `alignment_quality=pass` 的 SpeechOcean 音素段。

- 总音素段：90,323；
- train：36,064；
- dev：8,788；
- test：45,471；
- test correct：43,650；
- test mispronounced：1,821；
- positive class：`label=1`，mispronounced。

训练、验证、测试按 speaker 隔离。

## 主路线

```text
wav -> wav2vec2 hidden states
alignment start/end -> phone-span mean pooling
target_phone + phone_group + duration + wav2vec2 embedding
-> phone-level binary classifier
```

预训练模型：`facebook/wav2vec2-base`。

## Logistic Regression

模型文件：

```text
artifacts/mdd_wav2vec2/mdd_classifier.joblib
```

测试集结果：

- Accuracy：0.731147；
- Balanced Accuracy：0.687096；
- Precision：0.091423；
- Recall：0.639209；
- F1：0.159967；
- AUC：0.745912。

在 `Precision >= 0.40` 的阈值中，最大 recall 点为：

- Precision：0.416667；
- Recall：0.002746；
- F1：0.005456；
- threshold：0.999164。

## MLP

模型文件：

```text
artifacts/mdd_wav2vec2/mdd_classifier_mlp.joblib
```

测试集结果：

- Accuracy：0.714191；
- Balanced Accuracy：0.696420；
- Precision：0.090389；
- Recall：0.677100；
- F1：0.159488；
- AUC：0.759781。

在 `Precision >= 0.40` 的阈值中，最大 recall 点为：

- Precision：0.400000；
- Recall：0.002197；
- F1：0.004369；
- threshold：0.997105。

## 是否达到验收指标

未达到。

计划书重点验收条件为：

```text
错误类别 precision >= 0.40
在此前提下 recall >= 0.50
```

当前 wav2vec2-MDD 全量模型在追求 Balanced Accuracy 的阈值下 recall 可到 0.64–0.68，但 precision 只有约 0.09；当强制 precision 达到 0.40 时，recall 接近 0。

这说明模型可以捕捉一部分误发模式，但高置信误发排序仍不够好，不能同时做到“报得准”和“抓得多”。

## 当前最好可汇报指标

如果按论文式主路线汇报：

- 使用 MLP：Balanced Accuracy 0.696，AUC 0.760，Recall 0.677；
- 但 precision 仅 0.090。

如果按验收 precision 口径汇报：

- precision 可达到 0.400；
- 但 recall 仅 0.002。

## 建议下一步

1. 进行核心音素专项模型，而不是全音素统一模型；
2. 引入更强的音素对齐/CTC posterior/GOP 作为额外特征；
3. 使用 GPU 微调 wav2vec2/WavLM，而不是冻结 embedding；
4. 人工复核一批 false positives 和 false negatives，判断 SpeechOcean 标签是否与“教学误发”口径一致；
5. 扩充真实误发样本或做错误类重采样/数据增强。
