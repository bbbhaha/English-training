# 数据字典（v0.1）

| 字段 | 类型 | 说明 |
|---|---|---|
| `utterance_id` | string | 数据集内唯一语句标识 |
| `speaker_id` | string | 说话人标识 |
| `speaker_gender` | string | `M/F/unknown` |
| `native_language` | string | 母语 |
| `word` | string | 与音素时间重叠最大的单词 |
| `target_phone_raw` | string | 数据集原始目标音素 |
| `target_phone` | string | 统一后的 ARPAbet 音素 |
| `perceived_phone_raw` | string | 实际感知音素原始标注 |
| `perceived_phone` | string | 统一后的实际音素 |
| `phone_index` | integer | 语句内有效音素序号，从 0 开始 |
| `start_ms/end_ms` | float/null | 音素边界，毫秒；SpeechOcean762 原始标注无边界，强制对齐后回填 |
| `duration_ms` | float/null | 音素时长 |
| `source_score` | float/null | 原数据集音素人工评分 |
| `gold_binary` | 0/1 | 识别性错误检测标签 |
| `attention_binary` | 0/1/null | 标准性不足检测标签 |
| `gold_three_class` | string/null | `correct/acceptable/incorrect` |
| `error_type` | string | `correct/substitution/deletion/addition/unknown` |
| `phone_group` | string | 发音类别粗分组 |
| `dataset_source` | string | 数据来源 |
| `split` | string | `train/dev/test` |
| `audio_path` | string | 相对项目根目录的音频路径 |
| `annotation_path` | string | 相对项目根目录的标注路径 |
| `alignment_method` | string/null | 自动边界生成方法；人工边界数据为空 |
| `alignment_score` | float/null | 语句级平均对齐目标分数，用于排序抽检，不跨模型比较 |
| `alignment_quality` | string/null | `pass/review`；小于 20 ms 或大于 500 ms 的自动边界标为 `review` |

模型预测阶段在上述结构后追加：

`prediction, confidence, evidence_score, gop_score, error_type_hint,
model_version`
