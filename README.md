# 英语语音智能评测：第一阶段

本仓库按照《第一阶段计划书》建设音素发音正确性模型的数据与实验底座。

当前主路线已调整为论文复现风格的 wav2vec2-MDD：

```text
audio + canonical phone + alignment -> wav2vec2 phone embedding
-> target_phone / phone_group / duration / optional GOP
-> phone-level binary classifier
```

其中 `label=1` 表示 mispronounced，`label=0` 表示 correct。原 GOP 等价声学模型继续保留为 baseline。

当前实现：

- L2-ARCTIC 人工标注 TextGrid 解析；
- 统一音素级数据清单；
- 二分类、三分类标签规范；
- 按说话人隔离的数据划分；
- SpeechOcean762 数据下载和解析入口；
- SpeechOcean762 音素级强制对齐；
- GOP 等价声学似然基线；
- SpeechOcean762 第一阶段可复现建模、阈值校准、错误分析与最小演示；
- 数据质量检查与统计报告。

## 快速开始

要求 Python 3.10+。

```powershell
pip install -r requirements.txt
```

```powershell
python scripts/prepare_l2_arctic.py
python scripts/validate_manifest.py data/processed/l2_arctic/phones.csv
```

下载并准备 SpeechOcean762：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_speechocean.ps1
python scripts/prepare_speechocean.py
python scripts/align_speechocean.py
python scripts/validate_manifest.py data/processed/speechocean/phones_aligned.csv
```

所有处理结果默认写入 `data/processed/`。原始数据不会被修改。

运行 CPU 可复现的 GOP 等价声学似然基线：

```powershell
python scripts/run_acoustic_baseline.py
```

该基线使用 L2-ARCTIC train 说话人的正确音素段训练音素高斯声学模型，在 dev
说话人上校准阈值，并仅在隔离的 test 说话人上报告最终指标。

首版结果见 [docs/baseline_results_v1.md](docs/baseline_results_v1.md)。
强制对齐结果见 [docs/alignment_results_v1.md](docs/alignment_results_v1.md)。

运行第一阶段 SpeechOcean 建模、阈值校准、正式评估、100 条样例和错误分析：

```powershell
python scripts/run_phase1_modeling.py --alignment-quality pass
```

主要输出在 `reports/phase1/`：

- `formal_eval_metrics.csv`：Accuracy、Balanced Accuracy、Macro-F1、AUC、错误类 Precision/Recall/F1；
- `model_comparison.csv`：test 集模型排序；
- `thresholds.csv`：全局、音素组、target_phone 三级阈值；
- `prediction_samples_100_utterances.csv`：100 条测试语音的音素级诊断样例；
- `error_analysis_summary.md` 与相关 CSV：错误分析；
- `demo_prediction.csv`：最小演示输出样例。

运行最小演示：

```powershell
python scripts/demo_predict.py --utterance-id speechocean_000030012 --output reports/phase1/demo_prediction.csv
```

对不在清单里的新音频，当前脚本需要显式提供目标音素序列：

```powershell
python scripts/demo_predict.py --audio-path path/to/audio.wav --text "WE CALL IT BEAR" --target-phones "W IY K AO L IH T B EH R"
```

说明：当前 demo 已具备音素级诊断表输出接口；新文本自动 G2P 尚未接入，需要后续接入 CMUdict、g2p-en 或项目自有词典。

## wav2vec2-MDD 主路线

### 1. 生成 phone-level MDD manifest

SpeechOcean 主数据：

```powershell
python scripts/prepare_mdd_manifest.py `
  --input data/processed/speechocean/phones_aligned.csv `
  --output data/processed/mdd/speechocean_manifest.csv `
  --alignment-quality pass
```

L2-ARCTIC 数据也可在准备脚本中同步生成：

```powershell
python scripts/prepare_l2_arctic.py `
  --mdd-output data/processed/mdd/l2_arctic_manifest.csv
```

统一 manifest 字段：

```text
utt_id, speaker_id, wav_path, word, target_phone, phone_group,
start, end, duration, label, split, dataset_source, phone_index, gop_score
```

### 2. 提取 wav2vec2 音素段 embedding

首次运行如未缓存模型，会下载 `facebook/wav2vec2-base`。如果已下载，可加 `--local-files-only`。

```powershell
python scripts/extract_wav2vec2_features.py `
  --manifest data/processed/mdd/speechocean_manifest.csv `
  --output artifacts/mdd_wav2vec2/features.npz `
  --local-files-only
```

快速小样本测试：

```powershell
python scripts/extract_wav2vec2_features.py `
  --manifest data/processed/mdd/speechocean_manifest.csv `
  --output artifacts/mdd_wav2vec2_smoke/features.npz `
  --max-utterances-per-split 20 `
  --local-files-only
```

### 3. 训练 phone-level MDD 分类器

默认 Logistic Regression，使用 `class_weight=balanced` 处理错误样本少的问题。

```powershell
python scripts/train_mdd_classifier.py `
  --features artifacts/mdd_wav2vec2/features.npz `
  --output artifacts/mdd_wav2vec2/mdd_classifier.joblib
```

### 4. 评估，并搜索 Precision >= 0.40 下 Recall 最大的阈值

评估时 positive class 是 mispronounced。

```powershell
python scripts/evaluate_mdd.py `
  --features artifacts/mdd_wav2vec2/features.npz `
  --model artifacts/mdd_wav2vec2/mdd_classifier.joblib `
  --output-dir reports/mdd_wav2vec2 `
  --min-precision 0.40
```

输出：

- `reports/mdd_wav2vec2/metrics.json`
- `reports/mdd_wav2vec2/pr_curve.csv`
- `reports/mdd_wav2vec2/predictions.csv`
- `reports/mdd_wav2vec2/summary.md`

验收重点：

```text
mispronounced precision >= 0.40
在此前提下 recall >= 0.50
```

### 5. 单条或小批量预测

如果已有 MDD manifest 行：

```powershell
python scripts/predict_mdd.py `
  --model artifacts/mdd_wav2vec2/mdd_classifier.joblib `
  --manifest path/to/predict_manifest.csv `
  --output reports/mdd_wav2vec2/predict_output.csv `
  --local-files-only
```

如果是一条新音频，需要提供音素级 alignment CSV，至少包含：

```text
target_phone,start,end
```

然后运行：

```powershell
python scripts/predict_mdd.py `
  --model artifacts/mdd_wav2vec2/mdd_classifier.joblib `
  --wav-path path/to/audio.wav `
  --alignment-csv path/to/alignment.csv `
  --output reports/mdd_wav2vec2/predict_output.csv `
  --local-files-only
```

运行自监督语音表征 baseline。首次运行会从 HuggingFace 下载 `facebook/wav2vec2-base`：

```powershell
python scripts/run_ssl_baseline.py
```

当前默认设置为 CPU 友好的 pilot：每个 split 抽取 120 条语音，共 360 条语音。输出：

- `reports/phase1/ssl_eval_metrics.csv`
- `reports/phase1/ssl_predictions.csv`
- `reports/phase1/ssl_baseline_summary.md`
- `artifacts/ssl_wav2vec2_v1/`

如需跑全量，可使用：

```powershell
python scripts/run_ssl_baseline.py --max-utterances-per-split 0
```

运行监督式音素段声学特征分类器：

```powershell
python scripts/run_segment_acoustic_classifier.py
```

该脚本为每个音素段抽取 log-Mel 均值/方差特征，并训练 Logistic Regression、
Random Forest 和 ExtraTrees。输出在 `reports/phase1_segment_acoustic/`。

## 统一输出

音素清单的核心字段包括：

`utterance_id, speaker_id, word, target_phone, perceived_phone, phone_index,
start_ms, end_ms, gold_binary, gold_three_class, error_type, phone_group,
dataset_source, split, audio_path`

详细定义见 [docs/data_dictionary.md](docs/data_dictionary.md) 和
[docs/label_spec.md](docs/label_spec.md)。

第一阶段最终验收状态见 [docs/phase1_final_report.md](docs/phase1_final_report.md)。

## 当前数据划分说明

SpeechOcean762 应优先保留官方 train/test，说话人级验证集从官方训练部分生成。
本地 L2-ARCTIC Mandarin 只有 4 名说话人，因此默认使用 2/1/1 名说话人作为
train/dev/test。该划分适合管线验证和跨说话人测试，不应被当作稳定的模型效果估计。
## End-to-end pronunciation prediction

### Mandarin-L1 word-deletion evidence

真实音频的漏读判断使用三路独立证据：无提示英语 ASR 的词级编辑对齐、alignment-free
Wav2Vec2 CTC 的“完整句/删词句”似然比较，以及强制对齐时长与边界。单独一路异常只进入
`possible_deletion`，ASR 与 CTC 或极端时长证据一致时才确认 `deletion`。首次使用先准备模型：

```powershell
python scripts\download_deletion_models.py
```

中文母语 L2-ARCTIC 漏读子集的可复现诊断评估：

```powershell
python scripts\evaluate_mandarin_l2_arctic_deletions.py `
  --phones data\processed\l2_arctic\phones.csv `
  --audio-root . `
  --output-dir outputs\mandarin_deletion_validation
```

如果词典和 G2P 都无法提供可靠目标发音，后端输出 `g2p_issue`，网页显示“单词暂未收录”，
不会把该词误报为普通发音错误。研究依据与局部验证范围见
`docs/mandarin_l1_word_deletion_strategy.md`。

This workflow is for real `wav + target text` pronunciation diagnosis. It is
reported separately from public dataset `gold_binary` evaluation.

### 1. Text to phones

```powershell
python scripts/g2p_text.py --text "WE CALL IT BEAR" --output outputs/demo/g2p.json
```

The JSON contains word-level and phone-level ARPAbet mappings. Each word keeps
`phones`, `g2p_source`, `word_index`, `phone_index_start`, and
`phone_index_end`. The G2P module uses CMUdict when available, then `g2p-en`,
then marks unresolved words as `oov`.

### 2. Audio to target-phone alignment

```powershell
python scripts/align_audio.py `
  --audio data/demo/learner.wav `
  --text "WE CALL IT BEAR" `
  --output outputs/demo/alignment.csv `
  --g2p-output outputs/demo/g2p.json
```

The alignment CSV contains `word`, `target_phone`, `phone_index`, `start_ms`,
`end_ms`, `duration_ms`, and `alignment_quality`. If alignment fails, phones are
kept for traceability but `alignment_quality` is set to `bad`.

Bad alignment is never allowed to become a high-confidence pronunciation error;
it is routed to `uncertain_review`.

### 3. One-command prediction

```powershell
python scripts/predict_pronunciation.py `
  --audio data/demo/learner.wav `
  --text "WE CALL IT BEAR" `
  --output outputs/demo/prediction.csv `
  --alignment-output outputs/demo/alignment.csv `
  --g2p-output outputs/demo/g2p.json
```

Output columns:

`utterance_id, speaker_id, word, word_index, target_phone, phone_index,
start_ms, end_ms, duration_ms, model_error_score, prob_correct,
manual_calibrated_error_probability, decision, confidence, error_type,
alignment_quality, review_reason, g2p_source`

Allowed decisions:

- `correct`
- `acceptable_accent`
- `true_error`
- `uncertain_review`

### 4. Manual-label end-to-end evaluation

Create or update:

```text
data/manual_review/manual_review_labels.csv
```

Allowed `manual_review_label` values:

- `correct`
- `acceptable_accent`
- `true_error`
- `bad_alignment`

Then run:

```powershell
python scripts/evaluate_e2e_prediction.py `
  --prediction outputs/demo/prediction.csv `
  --manual-labels data/manual_review/manual_review_labels.csv `
  --output outputs/demo/e2e_evaluation_report.json
```

The report includes `true_error_precision`, `true_error_recall`,
`true_error_f1`, `acceptable_accent_false_positive_rate`,
`alignment_failure_rate`, `uncertain_review_rate`, per-phone metrics, and
per-speaker metrics. Do not mix this manual-label end-to-end report with public
dataset `gold_binary` evaluation.

### 5. Web demo

This project also includes a local web demo adapted from the partner
`English_assessment_project` frontend. The backend has been changed to call this
project's end-to-end pipeline instead of the partner project's uniform
segmentation demo.

Start the server:

```powershell
python webapp\app.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

The page supports microphone recording or audio upload, target-text input, and
phone-level diagnosis display. API outputs and trace files are saved under:

```text
outputs/webapp/
```

The web demo preserves the same safety rule as the CLI:

```text
alignment_quality == bad -> decision = uncertain_review
```

It does not convert a failed alignment into a high-confidence pronunciation
error.

### Mandarin-L1 deletion fusion model

The default backend loads `models/mandarin_deletion_fusion_v2.joblib`. The
model combines Whisper word edits, alignment-free CTC full-vs-deleted logit
margins, an independent CTC greedy transcript, and lexical context. It is
trained with speaker-isolated Mandarin L2-ARCTIC splits and controlled
whole-word deletion augmentation.

```powershell
python scripts/train_mandarin_deletion_fusion.py `
  --phones C:/path/to/data/processed/l2_arctic/phones.csv `
  --source-project-root C:/path/to/project `
  --output-dir outputs/mandarin_deletion_training_v1 `
  --model-output models/mandarin_deletion_fusion_v1.joblib
```

See `docs/mandarin_l1_word_deletion_strategy.md` for the literature basis,
speaker split, metrics, limitations, and the separate real-deletion check.

Evaluate the paired real recordings listed in the local manifest:

```powershell
python scripts/evaluate_manual_deletion_pairs.py `
  --manifest data/manual_deletion_pairs/manifest.csv `
  --model models/mandarin_deletion_fusion_v2.joblib `
  --output-dir outputs/manual_deletion_pairs/final_v2
```

## Phone-level three-state diagnosis

The default web backend and `--decision-mode phone_diagnosis` now use two
independent phoneme CTC models plus a speaker-isolated Mandarin-L1 calibration
model. The second model gates deletion and substitution false alarms. Every
supported target phone receives one deployable state:

```text
correct / mispronounced / deleted
读对 / 读错 / 漏读
```

Download the runtime models once:

```powershell
python scripts\download_deletion_models.py
```

Run an arbitrary WAV and target sentence:

```powershell
python scripts\predict_pronunciation.py `
  --audio learner.wav `
  --text "SHE SEES THE BLUE BIRD" `
  --output outputs\demo\prediction.csv `
  --word-summary-output outputs\demo\word_summary.csv `
  --decision-mode phone_diagnosis `
  --enable-asr `
  --enable-ctc-deletion
```

See `docs/phone_three_state_diagnosis.md` for the CTC hypothesis method,
Mandarin-L1 speaker split, held-out metrics, correct-audio sanity result,
limitations, and reproduction commands.
