# 给模型开发（B）的交接说明

## 可直接使用的输入

- SpeechOcean762 对齐清单：`data/processed/speechocean/phones_aligned.csv`
- L2-ARCTIC 人工边界清单：`data/processed/l2_arctic/phones.csv`
- 两套数据合并清单：`data/processed/combined/phones.csv`
- SpeechOcean 对齐质量报告：
  `data/processed/speechocean/alignment_report.json`

合并清单共 114,429 行，并保持说话人级 train/dev/test 隔离。

## 建议的首轮训练过滤

SpeechOcean762 首轮只使用：

```text
alignment_quality == "pass"
```

这会保留 90,323 个音素，暂时排除 4,122 个小于 20 ms 或大于 500 ms 的
`review` 音素。L2-ARCTIC 使用人工边界，不受该过滤条件限制。

不要重新随机划分数据，应直接使用清单中的 `split`。阈值和超参数只能在 dev
上确定，test 保留用于最终报告。

## 复现与校验

```powershell
python scripts/run_acoustic_baseline.py
python scripts/align_speechocean.py
python scripts/evaluate_alignment.py --limit 100
python scripts/combine_manifests.py
python scripts/validate_manifest.py data/processed/combined/phones.csv
```

人工边界回放验证结果：边界绝对误差中位数 20 ms，85.3% 不超过 50 ms，
93.4% 不超过 100 ms。

## B 的下一步

1. 将 SpeechOcean `pass` 数据接入声学特征训练；
2. 加入说话人/语句级归一化；
3. 在 dev 上以错误 Precision 不低于 0.40 为约束校准阈值；
4. 报告 Balanced Accuracy、Macro-F1、AUC、错误 Precision/Recall；
5. 单独分析 `review` 样本，不把它们直接并入首轮正式训练。
