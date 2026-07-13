# HierTFR / GOPT 官方特征复现与 MDD 辅助头阶段报告

## 当前状态

本阶段已停止继续盲调 frozen wav2vec2 embedding + MLP/LogReg 和 phone-only GOPT，改为：

1. 验证 GOPT 官方 intermediate GOP feature 能否接入统一序列格式；
2. 对比 `official_gop_features` / `current_gaussian_gop_features` / `no_gop_features`；
3. 实现 HierTFR-minimal：phone Transformer + phone-to-word pooling + word Transformer；
4. 加入 phone score regression、word score regression、MDD auxiliary head；
5. 输出 binary MDD 的 precision/FPR 约束阈值搜索和 per-phone false-positive 分析。

## 新增/修改文件

- `scripts/prepare_speechocean_gopt_features.py`
  - 新增 `word_ids`、`word_scores`，并在 `features.phones.csv` 中保留 `word_index`、`word_accuracy`。
- `scripts/prepare_gopt_official_features.py`
  - 生成官方复现对照特征变体。
  - 当前支持 `current_gaussian_gop_features`、`no_gop_features`、外部传入的 `official_gop_features`。
- `scripts/train_gopt_official_repro.py`
  - 批量训练 GOPT 官方复现 feature variants。
- `scripts/evaluate_gopt_official_repro.py`
  - 输出 `outputs/gopt_official_repro_results.csv`。
- `configs/hiertfr_minimal_speechocean.yaml`
  - HierTFR-minimal 配置。
- `scripts/train_hiertfr_minimal.py`
  - phone-level Transformer、phone-to-word pooling、word-level Transformer。
  - 支持 `--alpha-mdd`、`--beta-word`、`--drop-score1-for-mdd`、`--drop-score1-all`、`--min-duration`、`--core-phone-set`。
- `scripts/evaluate_hiertfr_as_mdd.py`
  - 输出 phone MSE/PCC、word PCC、binary Precision/Recall/F1/AUC/AUPRC/FPR/FP/FN。
  - 输出 `max recall at precision >= 0.40`。
  - 输出 `max recall at FPR <= 0.03/0.04/0.05`。
  - 输出 `per_phone_pr_summary.csv`、`false_positive_by_phone.csv`、`false_positive_examples.csv`。
- `scripts/run_hiertfr_core_phone_sets.py`
  - 对核心音素组分别训练/评估，并输出 `outputs/hiertfr_core_phone_set_results.csv`。

## 已完成的最小可行测试

### 特征重建

命令：

```powershell
python scripts\prepare_speechocean_gopt_features.py
```

结果：

- utterances: 5,000
- phones after duration/alignment filtering: 85,800
- train/dev/test 按 speaker split 使用原有 split 字段。

### 官方特征变体准备

命令：

```powershell
python scripts\prepare_gopt_official_features.py
```

输出：

- `artifacts/gopt_official_repro/current_gaussian_gop_features.npz`
- `artifacts/gopt_official_repro/no_gop_features.npz`
- `artifacts/gopt_official_repro/feature_variants.csv`

当前本地没有 GOPT 官方 intermediate GOP feature 文件，所以 `official_gop_features` 已被标记为 `missing_official_intermediate_features`。

### GOPT 官方复现对照 smoke

命令：

```powershell
python scripts\train_gopt_official_repro.py --epochs 1
python scripts\evaluate_gopt_official_repro.py
```

输出：

- `outputs/gopt_official_repro_results.csv`

1 epoch smoke 结果不是正式结论，只用于验证流程：

| variant | phone PCC | word PCC | binary precision | binary recall | precision>=0.40 found |
|---|---:|---:|---:|---:|---|
| current_gaussian_gop_features | 0.1212 | 0.1050 | 0.2308 | 0.0017 | false |
| no_gop_features | 0.1740 | 0.1099 | 0.1511 | 0.0265 | false |
| official_gop_features | missing | missing | missing | missing | skipped |

### HierTFR-minimal smoke

命令：

```powershell
python scripts\train_hiertfr_minimal.py --epochs 1 --output-dir artifacts\hiertfr_minimal_smoke
python scripts\evaluate_hiertfr_as_mdd.py --checkpoint artifacts\hiertfr_minimal_smoke\hiertfr_minimal.pt --output-dir reports\hiertfr_minimal_smoke_test --split test
```

test smoke 结果：

| metric | value |
|---|---:|
| phone MSE | 0.1381 |
| phone PCC | 0.0836 |
| word PCC | 0.1042 |
| AUC | 0.5913 |
| AUPRC | 0.0638 |
| precision>=0.40 found | true |
| precision at selected point | 0.5000 |
| recall at selected point | 0.0006 |
| FPR at selected point | 0.000024 |
| max recall at FPR<=0.03 | 0.0743 |
| max recall at FPR<=0.04 | 0.0963 |
| max recall at FPR<=0.05 | 0.1126 |

这说明脚本链路可跑通，但 1 epoch 模型还没有学习充分，不能当正式验收结果。

### 单元测试

命令：

```powershell
python -m unittest discover -s tests -v
```

结果：10/10 passed。

## 如何运行正式实验

### 1. 如果拿到了官方 GOPT intermediate feature

把官方 `.npz` 放到项目中，然后运行：

```powershell
python scripts\prepare_gopt_official_features.py --official-feature-file path\to\official_features.npz
python scripts\train_gopt_official_repro.py
python scripts\evaluate_gopt_official_repro.py
```

如果官方文件里的 key 不是 `numeric`、`features` 或 `gop_features`，需要先按脚本注释调整 key mapping。

### 2. 训练正式 HierTFR-minimal

```powershell
python scripts\train_hiertfr_minimal.py
python scripts\evaluate_hiertfr_as_mdd.py
```

### 3. 对比 MDD auxiliary head 与 score threshold

```powershell
python scripts\evaluate_hiertfr_as_mdd.py --score-source mdd_head
python scripts\evaluate_hiertfr_as_mdd.py --score-source score_threshold --output-dir reports\hiertfr_minimal_score_threshold
```

### 4. 跑核心音素组实验

```powershell
python scripts\run_hiertfr_core_phone_sets.py
```

输出：

- `outputs/hiertfr_core_phone_set_results.csv`

## 对用户提出的 9 个问题的当前回答

1. 当前自制 GOP 和官方 GOPT GOP 特征差距多大？
   - 还不能量化，因为本地没有官方 intermediate GOP features。现在已完成接入口；拿到官方文件后可直接对比 `outputs/gopt_official_repro_results.csv`。

2. GOPT official reproduction 是否接近论文结果？
   - 当前还不能判断。`official_gop_features` 尚未接入真实官方文件；当前结果只代表自制 Gaussian GOP / no-GOP smoke。

3. HierTFR-minimal 是否提升 phone PCC？
   - smoke 不能下结论。1 epoch HierTFR phone PCC 为 0.0836，低于此前正式 GOPT phone-only 的 0.2258；但这是未正式训练的流程验证。

4. 加 word-level context 后，binary MDD 的 precision 是否改善？
   - smoke 下在极保守阈值可达到 precision 0.50，但 recall 只有 0.0006，实用上没有改善。需要正式训练和官方 GOP feature 后再判断。

5. MDD auxiliary head 是否比单纯 score threshold 更好？
   - 评估脚本已支持两种 score source。当前只完成 `mdd_head` smoke；正式对比需要分别跑 `--score-source mdd_head` 和 `--score-source score_threshold`。

6. score=1 是否是 precision 崩掉的主要来源？
   - 现在训练/评估脚本已支持 `--drop-score1-for-mdd` 和 `--drop-score1-all`。已有 GOPT 结果显示丢弃 score=1 对全局 precision 没有根本性改善；HierTFR 正式实验仍需比较。

7. L/N/IH/R/AY 是否仍然贡献最多 false positive？
   - 脚本已输出 `false_positive_by_phone.csv`。需要正式 HierTFR 模型后读取该表判断；smoke 模型不适合作错误来源结论。

8. 哪些核心音素组达到 precision>=0.40 recall>=0.50？
   - 批量实验脚本已实现。正式运行 `scripts/run_hiertfr_core_phone_sets.py` 后查看 `outputs/hiertfr_core_phone_set_results.csv`。

9. 是否建议第一阶段只验收核心音素组？
   - 是。基于此前 frozen wav2vec2、GOP baseline、phone-only GOPT 的诊断，全音素统一验收很难；更合理的是先限定核心音素组，尤其 `rl`、`front_vowels`、`final_consonants`，确认 precision>=0.40 recall>=0.50 后再扩展。

## 当前最关键缺口

真正能决定下一阶段成败的不是再盲调超参，而是：

1. 拿到 GOPT 官方 SpeechOcean762 intermediate GOP features，或严格复现其 ASR posterior/GOP 特征；
2. 用正式 epoch 训练 HierTFR-minimal；
3. 分别评估 `mdd_head` 和 `score_threshold`；
4. 跑核心音素组单独实验；
5. 基于 `false_positive_by_phone.csv` 和 `per_phone_pr_summary.csv` 决定第一阶段验收范围。
