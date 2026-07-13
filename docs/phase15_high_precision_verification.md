# Phase-1.5 High-Precision Verification

## 2026-07-12 Label Policy Update

The manual review labels have been updated for a correction-focused project goal:
all rows previously labeled `acceptable_accent` are now treated as `true_error`.
The current 200-row manual label distribution is:

| label | count |
|---|---:|
| `true_error` | 177 |
| `correct` | 14 |
| `bad_alignment` | 9 |

Use the refreshed E2E report in `outputs/e2e_alpha_hardset_calibrated/metric_optimization_report.md`
for current metrics. Older sections below are kept as experiment history.

Phase-1.5 adds a non-destructive verification layer on top of existing Phase-1 phone-level predictions. Its purpose is to improve error precision and `max_recall_at_precision_0_40`, not to optimize global accuracy.

## Inputs

Required prediction columns:

- `target_phone`

Strongly recommended columns:

- `gold_binary`: Phase-1 convention, `0=error`, `1=correct`
- `prediction`: Phase-1 convention, `0=predicted error`, `1=predicted correct`
- `prob_correct` or another probability/confidence column
- `phone_group`, `phone_index`, `duration_ms`, `utterance_id`, `speaker_id`, `word`

Training manifest columns for retrieval and one-class verification:

- `target_phone`
- one of `gold_binary`, `label`, or `gold_label`
- numeric/acoustic columns if available, such as `duration_ms`, `gop_score`, `evidence_score`, or wav2vec columns named `w2v_*`

If optional columns are missing, the scripts print validation messages and continue with weaker signals.

## Run

```powershell
python scripts/phase15_run_verifiers.py `
  --input reports/phase1_acoustic_fusion_macro/best_model_predictions.csv `
  --train-manifest data/processed/speechocean/phones_aligned.csv `
  --config configs/phase15/aggregator.yaml `
  --output outputs/phase15_verification/test_verified_predictions.csv
```

Evaluate:

```powershell
python scripts/phase15_evaluate.py `
  --input outputs/phase15_verification/test_verified_predictions.csv `
  --label-col gold_binary `
  --score-col final_error_score `
  --decision-col final_decision `
  --output outputs/phase15_verification/evaluation_report.json
```

## Output Columns

- `expected_attribute_vector`
- `attribute_risk_score`
- `attribute_mismatch_count`
- `attribute_verifier_decision`
- `proto_same_phone_sim_top1`
- `proto_same_phone_sim_topk_mean`
- `proto_confusion_phone_sim_top1`
- `proto_margin`
- `retrieval_verifier_decision`
- `oneclass_anomaly_score`
- `oneclass_verifier_decision`
- `final_decision`
- `final_error_score`

`final_decision` has four classes:

- `correct`
- `acceptable_accent`
- `uncertain_review`
- `high_confidence_error`

For high-precision binary evaluation, only `high_confidence_error` is treated as error. A second analysis-only policy treats `high_confidence_error + uncertain_review` as error.

## Current Smoke-Test Result

Command run on `reports/phase1_acoustic_fusion_macro/best_model_predictions.csv` with `data/processed/speechocean/phones_aligned.csv` as the training manifest:

| Policy | Error Precision | Error Recall | Balanced Accuracy | Macro-F1 | AUC |
|---|---:|---:|---:|---:|---:|
| current best baseline | 0.126369 | 0.412054 | 0.644693 | 0.557821 | 0.726766 |
| baseline + attribute verifier | 0.135000 | 0.192857 | 0.569823 | 0.557215 | 0.639984 |
| baseline + retrieval verifier | 0.130257 | 0.099554 | 0.535465 | 0.539663 | 0.526453 |
| baseline + oneclass verifier | 0.125855 | 0.320536 | 0.612333 | 0.558012 | 0.554695 |
| baseline + all verifiers | 0.130800 | 0.173661 | 0.561984 | 0.553136 | 0.717358 |

The first rule-based version is therefore a working verification scaffold, not yet a tuned solution for the `precision >= 0.40` target. The next useful tasks are to add stronger acoustic evidence to the retrieval vectors, tune verifier thresholds on dev only, and inspect high-confidence false positives by core phone.

## Phase-1.5B Calibration and Error Audit

Phase-1.5B adds analysis and calibration tools for understanding why `high_confidence_error` still contains too many false positives. These tools do not train a new large model.

### False Positive Audit

```powershell
python scripts/phase15_audit_false_positives.py `
  --input outputs/phase15_verification/test_verified_predictions.csv `
  --output-dir outputs/phase15_verification/audit
```

Outputs:

- `outputs/phase15_verification/audit/high_confidence_false_positives.csv`
- `outputs/phase15_verification/audit/high_confidence_true_positives.csv`
- `outputs/phase15_verification/audit/false_negatives.csv`
- `outputs/phase15_verification/audit/uncertain_review_samples.csv`
- `outputs/phase15_verification/audit/manual_review_packet.csv`

The `audit_reason` column summarizes why a row was treated as high-confidence error, such as `main_model_high_prob + attribute_high_risk`, `main_model_high_prob + retrieval_negative_margin`, or `main_model_only_no_verifier_support`.

If `manual_review_packet.csv` already exists, rerunning the audit script preserves existing `manual_review_label` and `manual_review_notes` values by matching `utterance_id`, `speaker_id`, `word`, `target_phone`, `start_ms`, and `end_ms`.

### Analysis Tables

The audit and evaluation scripts write:

- `outputs/phase15_verification/analysis/per_phone_metrics.csv`
- `outputs/phase15_verification/analysis/per_phone_group_metrics.csv`
- `outputs/phase15_verification/analysis/core_phone_metrics.csv`
- `outputs/phase15_verification/analysis/evidence_pattern_metrics.csv`
- `outputs/phase15_verification/analysis/alignment_quality_metrics.csv`

Use `per_phone_metrics.csv` sorted by `false_positive_count` to find phones that dominate false alarms. Use `alignment_quality_metrics.csv` to check whether `review` alignments are contributing disproportionate false positives.

### Evidence Pattern Metrics

`evidence_pattern_metrics.csv` groups rows by verifier agreement patterns:

- `main_only`
- `main+attribute`
- `main+retrieval`
- `main+oneclass`
- `main+attribute+retrieval`
- `main+attribute+retrieval+oneclass`

High false-positive counts in `main_only` mean the aggregator is too permissive and should downgrade those rows to `uncertain_review`. Better high-confidence decisions should usually have at least two independent verifier signals.

### Threshold Sweep

```powershell
python scripts/phase15_sweep_aggregator.py `
  --input outputs/phase15_verification/test_verified_predictions.csv `
  --config configs/phase15/aggregator.yaml `
  --output-dir outputs/phase15_verification/calibration
```

Outputs:

- `outputs/phase15_verification/calibration/aggregator_sweep_results.csv`
- `outputs/phase15_verification/calibration/best_aggregator_config.yaml`
- `outputs/phase15_verification/calibration/best_threshold_summary.json`

The primary selection metric is `max_recall_at_precision_0_40`. Tie breakers are higher precision, higher recall, higher macro-F1, and fewer false positives on core phones.

Current Phase-1.5B sweep smoke-test best setting:

| Metric | Value |
|---|---:|
| precision | 0.149649 |
| recall | 0.085714 |
| balanced_accuracy | 0.532371 |
| macro_f1 | 0.539551 |
| max_recall_at_precision_0_40 | 0.000893 |
| max_recall_at_precision_0_50 | 0.000000 |

This is a meaningful calibration scaffold, but it still does not achieve the project target. Do not report `precision >= 0.40` unless a later evaluation actually reaches it with non-trivial recall.

### Strict Consensus Mode

`configs/phase15/aggregator.yaml` now supports:

```yaml
aggregator:
  aggregator_mode: strict_consensus
strict_consensus:
  min_evidence_count: 2
  allow_review_alignment: false
  require_main_model_error: true
```

In strict consensus mode, `high_confidence_error` requires main-model error evidence, enough verifier support, allowed alignment quality, and an enabled target phone or phone group. Weak evidence is downgraded to `uncertain_review`.

The sweep script writes `best_aggregator_config.yaml`; compare it against the baseline by rerunning `phase15_run_verifiers.py` with that config, then rerun `phase15_evaluate.py`.

### Manual Review Packet

Open `outputs/phase15_verification/audit/manual_review_packet.csv` and annotate:

- `manual_review_label`
- `manual_review_notes`

The packet prioritizes high-confidence false positives, high-confidence true positives, high-score false negatives, and boundary `uncertain_review` rows, with core phones such as `R/L/V/W/TH/DH/N/NG` placed first.

### Manual-Calibrated Verifier

After filling `manual_review_label`, run:

```powershell
python scripts/phase15_analyze_manual_review.py `
  --input outputs/phase15_verification/audit/manual_review_packet.csv `
  --output-dir outputs/phase15_verification/manual_review_analysis
```

Then train a small calibration layer:

```powershell
python scripts/phase15_manual_calibrated_verifier.py `
  --predictions outputs/phase15_verification/test_verified_predictions.csv `
  --manual-review outputs/phase15_verification/audit/manual_review_packet.csv `
  --config configs/phase15/aggregator.yaml `
  --output-dir outputs/phase15_verification/manual_calibration `
  --target-precision 0.40
```

Outputs:

- `outputs/phase15_verification/manual_calibration/manual_calibrated_predictions.csv`
- `outputs/phase15_verification/manual_calibration/manual_calibration_oof_predictions.csv`
- `outputs/phase15_verification/manual_calibration/manual_calibration_report.json`
- `outputs/phase15_verification/manual_calibration/manual_calibration_summary.csv`

Current 200-row manual review result:

| Label | Count |
|---|---:|
| `true_error` | 52 |
| `acceptable_accent` | 125 |
| `correct` | 14 |
| `bad_alignment` | 9 |

On the 200 manually reviewed high-confidence rows, the out-of-fold calibration threshold can reach precision `0.50` with recall `0.038462`. This is a very conservative setting: it proves some high-precision region exists under the manual label definition, but it has low recall and should not be reported as final held-out performance.

When evaluated against the original public `gold_binary`, the same calibrated decisions do not improve precision. That mismatch is expected because the manual review splits `acceptable_accent` and `bad_alignment` away from `true_error`, while the public binary labels do not capture that distinction cleanly.
