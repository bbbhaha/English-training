# wav2vec2-MDD 主模型流程

目标路线参考《Explore Wav2vec 2.0 for Mispronunciation Detection》的音素级误发检测思想：

```text
学生音频 + canonical phone + alignment
-> wav2vec2 hidden states
-> phone span mean pooling
-> target_phone / phone_group / duration / optional GOP
-> phone-level binary classifier
```

## 标签定义

- `label=1`：mispronounced，误发，positive class；
- `label=0`：correct，正确或可接受。

这和旧项目里的 `gold_binary` 相反：旧字段 `gold_binary=1` 表示正确。

## 主要脚本

1. `scripts/prepare_mdd_manifest.py`
2. `scripts/extract_wav2vec2_features.py`
3. `scripts/train_mdd_classifier.py`
4. `scripts/evaluate_mdd.py`
5. `scripts/predict_mdd.py`

GOP 相关脚本继续保留，作为 baseline，不删除。

## 评估口径

评估必须以 mispronounced 为 positive class，输出：

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- AUC
- PR curve
- Precision >= 0.40 时 recall 最大的阈值和对应 Precision/Recall/F1

项目验收重点不是 accuracy，而是：

```text
错误类别 precision >= 0.40
在这个前提下 recall >= 0.50
```
