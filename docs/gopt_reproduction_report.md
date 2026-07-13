# GOPT / HierTFR 下一阶段复现报告

日期：2026-07-05

## 路线切换说明

冻结 wav2vec2 embedding + MLP/LogReg 的全音素二分类路线已诊断为无法达到当前验收指标。下一阶段主线切换为更匹配 SpeechOcean762 的 GOPT / HierTFR 风格路线。

保留原有结果作为 baseline：

1. wav2vec2 frozen embedding + MLP；
2. GOP equivalent baseline；
3. wav2vec2-MDD threshold calibration diagnostics。

## 当前实现

新增文件：

```text
configs/gopt_speechocean762.yaml
scripts/prepare_speechocean_gopt_features.py
scripts/train_gopt_phone_score.py
scripts/evaluate_gopt_as_mdd.py
docs/gopt_reproduction_report.md
```

## GOPT 输入特征

当前输入包括：

- GOP feature：`gop_score`
- target phone likelihood：`target_log_likelihood`
- competitor likelihood：`competitor_log_likelihood`
- canonical phone embedding：`target_phone`
- phone group embedding：`phone_group`
- position embedding：`phone_index`
- duration feature：`duration`
- normalized duration by phone：`normalized_duration_by_phone`

## 模型结构

当前实现是轻量 GOPT-style Transformer：

```text
numeric GOP/duration projection
+ canonical phone embedding
+ phone_group embedding
+ position embedding
-> TransformerEncoder
-> phone-level score regression head
```

输出范围为 `[0, 2]`，对应 SpeechOcean762 phone score。

## 二分类转换

回归输出 `predicted_score` 后转 MDD：

```text
predicted_score < threshold -> label=1 error
predicted_score >= threshold -> label=0 correct
```

评估时使用 `error_score = -predicted_score` 搜索 PR 阈值。

## score=1 设置

已支持两种设置：

1. 训练时保留 score=1，作为连续回归分数；
2. 二分类评估时可丢弃 score=1，只评估 score=0 vs score=2。

命令中使用：

```powershell
python scripts/evaluate_gopt_as_mdd.py --binary-drop-score-one
```

## 已跑通 smoke test

特征准备：

```powershell
python scripts/prepare_speechocean_gopt_features.py --config configs/gopt_speechocean762.yaml
```

输出：

- utterances：5,000；
- phones after duration filter：85,800；
- feature file：`artifacts/gopt_speechocean762/features.npz`。

1 epoch smoke train：

```powershell
python scripts/train_gopt_phone_score.py --config configs/gopt_speechocean762.yaml --epochs 1
```

结果：

- dev MSE：0.161848；
- test MSE：0.134926；
- test PCC：0.121203。

GOPT-as-MDD smoke eval：

```powershell
python scripts/evaluate_gopt_as_mdd.py --config configs/gopt_speechocean762.yaml --binary-drop-score-one
```

1 epoch 仅用于验证流程，不代表正式模型效果。

## 正式训练结果

已运行完整配置训练：

```powershell
python scripts/train_gopt_phone_score.py --config configs/gopt_speechocean762.yaml
```

训练早停在第 18 轮，最佳 dev MSE 出现在第 14 轮。

回归指标：

| split | MSE | PCC |
|---|---:|---:|
| dev best | 0.153967 | - |
| test | 0.131975 | 0.225813 |

## GOPT-as-MDD 正式评估

### 全音素，二分类评估丢弃 score=1

命令：

```powershell
python scripts/evaluate_gopt_as_mdd.py `
  --config configs/gopt_speechocean762.yaml `
  --output-dir reports/gopt_speechocean762 `
  --binary-drop-score-one
```

结果：

| metric | value |
|---|---:|
| phone score MSE | 0.131975 |
| phone score PCC | 0.225813 |
| Precision | 0.316327 |
| Recall | 0.017455 |
| F1 | 0.033084 |
| AUC | 0.716356 |
| AUPRC | 0.112509 |
| FPR | 0.001630 |
| FP | 67 |
| FN | 1745 |

未找到满足 `precision >= 0.40` 的有效 recall 点；fallback 最高 precision 点为 precision 0.316。

### 核心音素集，二分类评估丢弃 score=1

核心音素：

```text
R L V W TH DH IH IY AE EH AH N
```

结果：

| metric | value |
|---|---:|
| phone score MSE | 0.142072 |
| phone score PCC | 0.194009 |
| Precision | 0.309859 |
| Recall | 0.024609 |
| F1 | 0.045596 |
| AUC | 0.698062 |
| AUPRC | 0.110821 |
| FPR | 0.002571 |
| FP | 49 |
| FN | 872 |

核心音素集也未达到 `precision >= 0.40, recall >= 0.50`。

### score=1 保留评估

保留 score=1 进入二分类评估时，结果与丢弃 score=1 基本一致：

| setting | Precision | Recall | AUC | AUPRC |
|---|---:|---:|---:|---:|
| all phones, keep score=1 | 0.313131 | 0.017455 | 0.715983 | 0.111902 |
| core phones, keep score=1 | 0.309859 | 0.024609 | 0.697950 | 0.110405 |

## 当前结论

GOPT phone-score regression 比 1 epoch smoke 有明显提升，test PCC 从约 0.121 提高到 0.226；但转成 binary MDD 后仍未达到验收线。

当前失败模式和 wav2vec2-MDD 类似：

1. 回归分数能排序一部分发音质量；
2. 但高 precision 的 error 区域太小；
3. 一旦要求 precision 接近 0.40，recall 仍很低；
4. 核心音素集也未明显改善。

下一步若继续 GOPT 路线，建议优先尝试：

1. HierTFR：同时建模 word/phone 层级，而不是只做 phone-level Transformer；
2. 使用更接近论文的 ASR posterior/GOP，而不是当前高斯 GOP；
3. 对 score=1 的样本做更细口径处理，例如训练保留，阈值搜索时只用 0/2；
4. 按 phone_group 分别训练 score regression，而不仅是阈值分组；
5. 对 L/N/IH/R/AY 做单独模型和人工复核。

## 正式实验命令

训练完整 GOPT：

```powershell
python scripts/train_gopt_phone_score.py --config configs/gopt_speechocean762.yaml
```

评估全音素：

```powershell
python scripts/evaluate_gopt_as_mdd.py `
  --config configs/gopt_speechocean762.yaml `
  --output-dir reports/gopt_speechocean762 `
  --binary-drop-score-one
```

评估核心音素：

```powershell
python scripts/evaluate_gopt_as_mdd.py `
  --config configs/gopt_speechocean762.yaml `
  --output-dir reports/gopt_speechocean762_core `
  --binary-drop-score-one `
  --core-phones R L V W TH DH IH IY AE EH AH N
```

## 下一步对比表

正式实验完成后，需要比较：

| model | threshold strategy | required outputs |
|---|---|---|
| wav2vec2 frozen + MLP | PR threshold | MDD metrics |
| GOP baseline | global/group threshold | MDD metrics |
| GOPT phone-score regression | dev PR threshold | score + MDD metrics |
| GOPT + core phone sets | dev PR threshold | core phone MDD metrics |

GOPT 每行必须报告：

- phone score MSE；
- phone score PCC；
- binary Precision；
- binary Recall；
- F1；
- AUC；
- AUPRC；
- FPR；
- precision>=0.40 时的最大 recall。
