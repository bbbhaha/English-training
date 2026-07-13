# 第一阶段音素标签规范（v0.1）

## 1. 主验收：识别性错误检测

- `gold_binary = 1`：目标音素正确，或 SpeechOcean762 音素 accuracy 分数不低于 1.0；
- `gold_binary = 0`：错误、漏发，或 SpeechOcean762 音素 accuracy 分数低于 1.0。

该口径回答“学习者是否发成了可识别的目标音”，是第一阶段主标签。

## 2. 标准性不足检测

- `attention_binary = 0`：SpeechOcean762 音素 accuracy 分数不低于 1.8；
- `attention_binary = 1`：分数低于 1.8，需要纠音关注。

L2-ARCTIC 没有与 0–2 专家评分直接等价的标签，因此该字段留空。

## 3. 三分类

| 标签 | 含义 | SpeechOcean762 映射 | L2-ARCTIC 映射 |
|---|---|---|---|
| `correct` | 标准或接近标准 | score >= 1.8 | 未标为错误 |
| `acceptable` | 可识别但口音较重 | 1.0 <= score < 1.8 | 无等价标注，留空 |
| `incorrect` | 错误或漏发 | score < 1.0 | substitution/deletion/addition |

不同数据集的三分类监督不完全同质，训练和报告时必须保留 `dataset_source`。

## 4. L2-ARCTIC 错误类型

- `substitution`：`CPL,PPL,s`
- `deletion`：`CPL,sil,d`
- `addition`：`sil,PPL,a`
- `correct`：原强制对齐标签未改变

`CPL` 写入 `target_phone_raw`，`PPL` 写入 `perceived_phone_raw`。用于建模的
`target_phone` 和 `perceived_phone` 去除重音数字和偏离符号 `*`。

## 5. 静音处理

普通静音 `sil/sp/spn` 不作为目标音素样本。人工标注的 addition 虽以 `sil` 为
目标占位符，但必须保留，因为它是有效错误事件。

