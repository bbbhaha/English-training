# MDD 系统诊断与阈值校准报告

日期：2026-07-05

## 1. 标签与评估口径检查

当前 MDD manifest 使用：

```text
label=1: mispronounced / error
label=0: correct / acceptable
```

评估脚本 `scripts/evaluate_mdd.py`、诊断脚本 `scripts/diagnose_mdd_data_and_scores.py`、阈值校准脚本 `scripts/calibrate_mdd_thresholds.py` 均以 `label=1` 作为 positive class。

SpeechOcean762 当前采用识别性错误检测口径：

```text
phone score < 1.0 -> error / mispronounced
phone score >= 1.0 -> acceptable / correct
```

当前正式 MDD 训练只使用 SpeechOcean762 pass 对齐样本，未混用 L2-ARCTIC。

## 2. 数据分布诊断

诊断输出目录：

```text
outputs/mdd_diagnostics_mlp_dev/
```

关键数据：

- 总样本：90,323；
- correct：86,180；
- mispronounced：4,143；
- 错误样本比例：约 4.59%；
- `duration < 0.03s` 的音素段：6,635；
- `end <= start` 异常：0；
- 最大音素时长：0.50s。

phone_group 错误比例：

| phone_group | correct | mispronounced | error rate |
|---|---:|---:|---:|
| affricate | 627 | 25 | 0.038 |
| fricative | 15,204 | 616 | 0.039 |
| glide | 4,099 | 96 | 0.023 |
| liquid | 5,631 | 336 | 0.056 |
| nasal | 9,410 | 428 | 0.044 |
| stop | 16,689 | 488 | 0.028 |
| vowel | 34,520 | 2,154 | 0.059 |

结论：类别极度不平衡，且大量短音素会影响 wav2vec2 frame pooling。

## 3. Score histogram 与 PR 曲线

重点文件：

```text
outputs/mdd_diagnostics_mlp_dev/score_histogram_correct_vs_error.png
outputs/mdd_diagnostics_mlp_dev/pr_curve.csv
outputs/mdd_diagnostics_mlp_dev/confusion_matrices.json
```

dev set 上 MLP score 诊断：

| threshold | precision | recall | F1 | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.154 | 0.558 | 0.241 | 1,286 | 185 | 234 |
| best F1 | 0.177 | 0.439 | 0.252 | 856 | 235 | 184 |
| precision >= 0.40 | 0.500 | 0.002 | 0.005 | 1 | 418 | 1 |

结论：模型分数可以找到一部分错误，但 correct 与 error 的高分区域重叠严重；一旦要求 precision >= 0.40，recall 几乎归零。

## 4. 阈值策略诊断

阈值输出：

```text
outputs/thresholds_global.json
outputs/thresholds_by_phone_group.json
outputs/thresholds_by_target_phone.json
outputs/mdd_thresholds_mlp/threshold_strategy_comparison.csv
```

test set 上三种阈值策略：

| strategy | precision | recall | F1 | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|
| global | 0.294 | 0.003 | 0.005 | 12 | 1816 | 5 |
| phone_group | 0.165 | 0.056 | 0.084 | 518 | 1719 | 102 |
| target_phone | 0.161 | 0.144 | 0.152 | 1368 | 1559 | 262 |

结论：phone/group 阈值能提升 recall，但 precision 仍达不到 0.40；问题不是单一全局阈值，而是模型 score 排序能力不足。

## 5. False positive 来源

重点文件：

```text
outputs/mdd_thresholds_mlp/false_positive_by_phone.csv
outputs/mdd_thresholds_mlp/false_positive_examples.csv
outputs/mdd_thresholds_mlp/per_phone_pr_summary.csv
```

false positive 主要来自：

| target_phone | group | FP | TP | precision | recall |
|---|---|---:|---:|---:|---:|
| L | liquid | 228 | 36 | 0.136 | 0.400 |
| N | nasal | 228 | 27 | 0.106 | 0.270 |
| IH | vowel | 157 | 28 | 0.151 | 0.250 |
| R | liquid | 151 | 23 | 0.132 | 0.371 |
| AY | vowel | 128 | 34 | 0.210 | 0.523 |
| AE | vowel | 61 | 6 | 0.090 | 0.120 |
| AO | vowel | 52 | 10 | 0.161 | 0.244 |

结论：precision 崩掉主要来自 liquid、nasal、vowel，尤其 L/N/IH/R/AY。

## 6. 哪些 phone 达到验收

按 `precision >= 0.40` 且 `recall >= 0.50`：

| phone | group | n | positive | precision | recall | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| ZH | fricative | 19 | 5 | 0.400 | 0.800 | 0.777 |
| OY | vowel | 20 | 2 | 0.500 | 0.500 | 0.567 |

但这两个 phone 样本量太小，不能作为稳定验收依据。

## 7. 训练流程修改与实验

已做修改：

1. `diagnose_mdd_data_and_scores.py`：系统诊断数据、标签、score、PR、混淆矩阵；
2. `calibrate_mdd_thresholds.py`：global / phone_group / target_phone 阈值；
3. `train_mdd_classifier.py`：
   - 支持 `--min-duration` 过滤短音素；
   - 加入 `normalized_duration_by_phone`；
   - 加入 `--negative-weight`；
   - 加入 hard negative mining 参数。

实验结果：

| model | setting | precision | recall | AUC | 结论 |
|---|---|---:|---:|---:|---|
| MLP | 原始全量 | 0.090 | 0.677 | 0.760 | recall 高，FP 太多 |
| LogReg | min_duration=0.03, negative_weight=1.5 | 0.089 | 0.677 | 0.749 | 无明显改善 |
| LogReg | min_duration=0.03, negative_weight=2.0 | 0.089 | 0.677 | 0.749 | 无明显改善 |
| LogReg | min_duration=0.03, negative_weight=3.0 | 0.089 | 0.678 | 0.749 | 无明显改善 |
| LogReg + hard negative mining | min_duration=0.03 | 0.084 | 0.672 | 0.736 | 变差 |

结论：单纯权重、短音素过滤和 hard negative mining 没有解决 score 排序问题。

## 8. 当前判断

严格总体指标仍未达到：

```text
mispronounced precision >= 0.40
且 recall >= 0.50
```

主要原因不是阈值搜索写错，而是：

1. 错误样本比例太低；
2. SpeechOcean 的“可接受/轻微口音/错误”边界本身模糊；
3. liquid、nasal、vowel 的 false positive 过多；
4. 短音素和自动对齐误差会污染 phone-level pooling；
5. 冻结 wav2vec2 embedding 对细粒度发音正确性区分仍不足。

## 9. 建议验收策略

不建议继续强行让全音素统一模型达标。建议下一步改为：

1. 先限定 8–12 个核心音素或音素组；
2. 优先处理 /r-l/、/v-w/、/th/、核心元音组；
3. 对 L/N/IH/R/AY 的 false positive 做人工复核；
4. 如果复核发现原标签把轻微口音也压成 error，应重新调整标签口径；
5. 如果复核发现边界错误，应先改对齐或排除低置信样本。
