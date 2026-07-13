# SpeechOcean762 强制对齐结果 v1

日期：2026-06-23

## 方法

使用 L2-ARCTIC train 说话人的人工音素边界训练 39 个对角高斯音素声学模型，
并以 L2-ARCTIC 音素时长中位数作为时长先验。对 SpeechOcean762 的已知目标音素
序列执行分段 Viterbi 强制对齐。持续低能量区作为停顿移除，并对跨停顿的音素段施加
惩罚，使边界优先落在词间停顿处。

运行命令：

```powershell
python scripts/align_speechocean.py
python scripts/validate_manifest.py data/processed/speechocean/phones_aligned.csv
```

## 全量结果

- 5,000/5,000 条语音对齐成功，失败 0 条；
- 94,445/94,445 个目标音素已回填 `start_ms/end_ms/duration_ms`；
- 90,323 个音素标记为 `pass`，4,122 个标记为 `review`；
- 输出：`data/processed/speechocean/phones_aligned.csv`；
- 逐语句质量信息：`data/processed/speechocean/alignment_report.json`；
- 数据清单校验通过，train/dev/test 说话人保持隔离。

自动音素段小于 20 ms 或大于 500 ms 时，`alignment_quality` 标记为 `review`。
模型训练应优先使用 `pass`，待人工抽检后再决定是否纳入 `review` 样本。

## 人工边界回放验证

在未参与声学模型训练的 L2-ARCTIC dev/test 语音中固定抽取 100 条，将自动边界
与人工 TextGrid 比较：

- 边界绝对误差中位数：20 ms；
- 85.3% 的边界误差不超过 50 ms；
- 93.4% 的边界误差不超过 100 ms；
- 100 条语音全部成功对齐。

复现命令：

```powershell
python scripts/evaluate_alignment.py --limit 100
```

这些数字验证的是同类普通话母语学习者语音上的边界定位能力。SpeechOcean762 没有
人工时间边界，因此仍应抽检低声学分数、超长音素和长停顿样本，不能把零运行失败
等同于零边界错误。
