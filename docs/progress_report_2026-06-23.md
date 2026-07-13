# 第一阶段进展报告：数据底座 v0.1

日期：2026-06-23

## 已完成

- 固化识别性二分类、标准性关注二分类和三分类标签口径；
- 下载并解压 SpeechOcean762；
- 解析 SpeechOcean762 全部 5,000 条语音；
- 解析 L2-ARCTIC Mandarin 全部 600 份人工 TextGrid；
- 建立统一音素数据字段和数据字典；
- 建立说话人隔离的 train/dev/test 划分；
- 建立自动数据验证和基础单元测试。

## 首版数据统计

### SpeechOcean762

- 250 名说话人，5,000 条语音；
- 94,445 个目标音素；
- 识别性标签：89,903 个可接受，4,542 个错误；
- 三分类：82,343 correct，7,560 acceptable，4,542 incorrect；
- 划分：100 名训练、25 名验证、125 名官方测试说话人。

错误类占比约 4.8%，后续训练不能使用普通 Accuracy 作为主要选择指标。必须采用
类别权重或重采样，并以 Balanced Accuracy、Macro-F1、错误召回和 AUC 为主。

### L2-ARCTIC Mandarin

- 4 名说话人，每人 150 份人工标注，共 600 条语音；
- 19,984 个有效音素/错误事件；
- 16,740 correct，2,433 substitution，605 deletion，206 addition；
- 按说话人进行 2/1/1 train/dev/test 管线划分。

L2-ARCTIC 的错误比例和错误类型信息更丰富，但只有 4 名普通话说话人。它适合作为
错误类型补充和跨数据集测试，不适合单独支撑稳定的普通话学习者效果结论。

## 已识别的数据问题

1. SpeechOcean762 不提供音素时间边界，`start_ms/end_ms` 暂时为空，需在 GOP 阶段
   通过强制对齐回填。
2. 两个数据集的三分类并不完全同质：L2-ARCTIC 没有“重口音但可接受”的评分。
3. SpeechOcean762 错误类严重稀少，训练和阈值校准必须在验证集完成。
4. L2-ARCTIC 只有 4 名普通话说话人，独立测试指标方差会较大。

## 下一步

进入计划书第 4 周任务：选择可复现的声学模型与强制对齐路线，生成音素边界和
GOP/等价声学证据；先在 SpeechOcean762 子集跑通，再扩展全量。

## 第 4 周基线更新

已完成基于 L2-ARCTIC 人工边界的 GOP 等价声学似然基线及轻量融合分类器。
独立测试说话人结果为 Balanced Accuracy 0.695、Macro-F1 0.559、AUC 0.768、
错误 Recall 0.745。错误 Precision 0.231，尚未达到验收要求，下一轮重点是利用
SpeechOcean762 扩大说话人覆盖并降低跨说话人误报。

详细结果见 `docs/baseline_results_v1.md`。

## SpeechOcean762 强制对齐更新

已使用 L2-ARCTIC 人工边界训练的 39 个音素声学模型和分段 Viterbi 算法完成
SpeechOcean762 全量强制对齐。5,000 条语音、94,445 个目标音素全部成功生成
`start_ms/end_ms/duration_ms`，其中 90,323 个标记为 `pass`、4,122 个超短或
超长边界标记为 `review`，数据清单校验通过。

在 L2-ARCTIC 隔离的 dev/test 语音中固定抽取 100 条进行人工边界回放验证：
边界绝对误差中位数 20 ms，85.3% 不超过 50 ms，93.4% 不超过 100 ms。
详细结果与复现命令见 `docs/alignment_results_v1.md`。
