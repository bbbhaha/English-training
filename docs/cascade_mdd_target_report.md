# Cascade MDD 项目指标报告

## 目标

本阶段不追求论文复现 fidelity，只优化项目验收指标：

```text
在所有 precision >= 0.40 的阈值中，找到最大 recall
验收目标：precision >= 0.40 且 recall >= 0.50
```

## 为什么单阶段 all-phone 模型失败

当前 SpeechOcean phone-level MDD 极度不平衡。test 集中正样本比例很低，模型只要稍微激进就会产生大量 false positives；一旦强行把阈值调到 precision >= 0.40，recall 就会接近 0。

已有结果显示：

- GOP baseline 能捞到很多错误，但 precision 不够；
- frozen wav2vec2 + MLP/LogReg 排序能力不足；
- GOPT-style score regression 能学到一点 phone score 排序，但无法形成稳定高置信 error 区域；
- HierTFR-minimal smoke 只证明流程可跑，未解决项目指标。

因此本阶段改为两阶段 cascade：

1. Stage 1：高召回 candidate generator；
2. Stage 2：false-positive verifier；
3. 最终预测：

```text
final_error = candidate_generator_predicts_error AND verifier_accepts
```

## 已实现文件

- `configs/candidate_verifier.yaml`
- `configs/core_phone_sets.json`
- `scripts/generate_mdd_candidates.py`
- `scripts/train_candidate_verifier.py`
- `scripts/evaluate_cascade_mdd.py`
- `outputs/final_cascade_mdd_comparison.csv`
- `outputs/cascade_false_positive_by_phone.csv`
- `outputs/cascade_false_positive_examples.csv`
- `outputs/cascade_per_phone_pr_summary.csv`
- `outputs/cascade_removed_false_positives.csv`

## Stage 1：GOP aggressive candidate generator

候选器使用 GOP aggressive threshold，在 dev set 上选择阈值，目标 candidate recall >= 0.70。

dev 阈值选择结果：

| split | threshold | candidate precision | candidate recall | FP | FN |
|---|---:|---:|---:|---:|---:|
| dev | 2.363949 | 0.0603 | 0.7408 | 4719 | 106 |

test 结果：

| split | candidate precision | candidate recall | FP | FN | candidate rows |
|---|---:|---:|---:|---:|---:|
| test | 0.0509 | 0.7337 | 23670 | 461 | 24940 |

重要发现：

此前“GOP precision 约 0.231、recall 约 0.745”的观察主要来自 L2-ARCTIC baseline，不直接等价于 SpeechOcean test。当前 SpeechOcean 上 GOP candidate recall 达到 0.7337，但 precision 只有 0.0509，候选池仍然非常脏。

这意味着 verifier 面临的问题仍然很难：候选池正样本率只从全量约 4% 提升到约 5.1%，没有达到理想中的 23.1%。

## Stage 2：Candidate verifier

第一版 verifier 使用：

- 预提取 wav2vec2 phone embedding；
- target phone embedding；
- phone group embedding；
- duration；
- normalized duration；
- GOP score；
- candidate score；
- 简化 TextGate：

```text
gate = sigmoid(W_text(text_repr))
gated_audio = audio_repr * gate
```

训练只在 GOP candidates 上进行。

### 无 contrastive loss

dev candidate pool：

| metric | value |
|---|---:|
| candidate train positive rate | 0.0663 |
| candidate dev positive rate | 0.0603 |
| best dev AUPRC | 0.2512 |

test cascade：

| metric | value |
|---|---:|
| candidate precision | 0.0509 |
| candidate recall | 0.7337 |
| verifier precision inside candidate pool | 0.3975 |
| verifier recall on true candidate errors | 0.0504 |
| final recall product | 0.0370 |
| final precision | 0.3975 |
| final recall | 0.0370 |
| final F1 | 0.0677 |
| AUC | 0.6902 |
| AUPRC | 0.1385 |
| FPR | 0.0024 |
| FP | 97 |
| FN | 1667 |

该模型几乎碰到 precision 0.40，但 verifier 太保守，只保留了约 5.0% 的真错误候选，所以 final recall 只有 0.037。

关系式：

```text
final_recall = candidate_recall * verifier_recall_on_true_candidate_errors
             = 0.7337 * 0.0504
             = 0.0370
```

### Contrastive loss

contrastive 版本：

| metric | value |
|---|---:|
| best dev AUPRC | 0.2529 |
| final precision | 0.3148 |
| final recall | 0.0098 |
| AUC | 0.6877 |
| AUPRC | 0.1296 |
| FP | 37 |
| FN | 1714 |

结论：当前简单 supervised contrastive loss 没有帮助项目指标，反而让模型更保守，recall 更低。

## FPR 约束结果

无 contrastive cascade：

| constraint | precision | recall | FPR | FP | FN |
|---|---:|---:|---:|---:|---:|
| FPR <= 0.03 | 0.2231 | 0.1987 | 0.0298 | 1198 | 1387 |
| FPR <= 0.04 | 0.2067 | 0.2409 | 0.0398 | 1600 | 1314 |
| FPR <= 0.05 | 0.1922 | 0.2750 | 0.0497 | 2000 | 1255 |

这说明：如果允许较多 false positives，recall 可以上升到 0.20–0.28；但 precision 仍然远低于 0.40。

## 是否达到验收目标？

没有。

当前最佳 cascade 在 precision 约束附近的结果是：

```text
precision = 0.3975
recall    = 0.0370
```

它接近 precision 0.40，但 recall 离 0.50 很远。

## Verifier 是否成功移除了 false positives？

是，但过度保守。

GOP candidate test 有 23,670 个 false positives。Cascade 最终 FP 降到 97，说明 verifier 确实大量移除了 false positives。

问题是它同时也移除了太多 true positives：

- candidate true positives: 1270
- final true positives: 64

也就是说 false positive removal 成功了，但 true error retention 失败了。

## 哪个 candidate source 最好？

目前只完成最小实验：

- GOP aggressive candidate + verifier
- GOP aggressive candidate + contrastive verifier

当前最好的是：

```text
GOP aggressive candidate + non-contrastive TextGate verifier
```

但仍未达标。

下一步应实现并比较：

- GOP + wav2vec2 union candidates；
- GOP + GOPT union candidates；
- phone_group-specific aggressive candidates；
- hard_fp_phones candidates；
- core phone set candidates。

## L/N/IH/R/AY 是否仍是主要 false-positive 来源？

已输出：

- `outputs/cascade_false_positive_by_phone.csv`
- `outputs/cascade_false_positive_examples.csv`
- `outputs/cascade_removed_false_positives.csv`

需要基于这些表继续分析。当前 cascade 最终 FP 只有 97，因此可以逐条人工抽查；但 candidate 阶段的 false positives 仍然非常多。

## 当前结论

Cascade 方向是合理的，但第一版 candidate generator 不够好。

关键不是 verifier 架构还不够复杂，而是 SpeechOcean 上 GOP candidate precision 太低。候选池正样本率只有 5.1%，导致 verifier 仍然接近在极不平衡数据上训练。

要达到：

```text
precision >= 0.40
recall >= 0.50
```

需要同时满足：

1. candidate recall 至少 0.70；
2. candidate precision 明显高于 0.05；
3. verifier 保留至少约 68% 的 true error candidates；
4. verifier 移除足够多 false positives。

当前第 1 点满足，第 2/3 点失败。

## 建议下一步

优先做 candidate generator，而不是继续加深 verifier：

1. union candidate：
   - GOP + wav2vec2 MLP；
   - GOP + GOPT score；
2. phone_group-specific aggressive thresholds；
3. core phone set cascade：
   - th；
   - rl；
   - hard_fp_phones；
4. 对 L/N/IH/R/AY 单独训练 verifier 或单独阈值；
5. 如果要继续训练神经 verifier，目标不是更高 AUC，而是：

```text
在 verifier true-candidate recall >= 0.67 时，最大化 false positive removal
```

## 给导师展示的当前证据

可以展示：

1. 单阶段模型失败不是偶然，而是 precision-recall tradeoff 在高不平衡下崩塌；
2. Cascade verifier 能显著移除 false positives：
   - FP 从 23670 降到 97；
3. 但当前 candidate precision 太低，导致 true error 也被大量删掉；
4. 下一步技术重点应从“训练更复杂分类器”转到“更好的候选生成 + phone/core-set 专门策略”。
