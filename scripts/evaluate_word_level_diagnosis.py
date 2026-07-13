#!/usr/bin/env python
"""Evaluate deletion and mispronunciation diagnosis at word level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


LABEL_COLUMNS = ("word_label", "manual_review_label", "gold_label", "label")
DELETION_LABELS = {"deletion", "deletion_or_missing", "possible_deletion", "missing_word"}
MISPRONUNCIATION_LABELS = {"mispronounced", "true_error", "pronunciation_error"}
EXCLUDED_LABELS = {"alignment_issue", "bad_alignment"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final word-level pronunciation diagnoses.")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/word_level_evaluation"))
    args = parser.parse_args()

    prediction = pd.read_csv(args.prediction, encoding="utf-8-sig", keep_default_na=False)
    labels = pd.read_csv(args.labels, encoding="utf-8-sig", keep_default_na=False)
    merged, label_column = merge_word_labels(prediction, labels)
    if merged.empty:
        raise SystemExit("No word-level prediction rows matched the labels.")
    report = evaluate_word_level(merged, label_column)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "word_level_diagnosis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _group_metrics(merged, label_column, "word").to_csv(
        args.output_dir / "per_word_metrics.csv", index=False, encoding="utf-8-sig"
    )
    _group_metrics(merged, label_column, "phone_group").to_csv(
        args.output_dir / "per_phone_group_metrics.csv", index=False, encoding="utf-8-sig"
    )
    _group_metrics(merged, label_column, "mandarin_confusion_type").to_csv(
        args.output_dir / "per_confusion_type_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def merge_word_labels(prediction: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    label_column = next((column for column in LABEL_COLUMNS if column in labels.columns), "")
    if not label_column:
        raise ValueError(f"Labels must contain one of: {', '.join(LABEL_COLUMNS)}")
    left = prediction.copy()
    if "final_word_decision" not in left.columns:
        raise ValueError("Prediction is missing final_word_decision.")
    keys = [column for column in ("utterance_id", "word_index") if column in left.columns and column in labels.columns]
    if "word_index" not in keys:
        keys = [column for column in ("utterance_id", "word") if column in left.columns and column in labels.columns]
    if not keys:
        raise ValueError("Prediction and labels need word_index or word as a shared key.")
    for key in keys:
        left[key] = left[key].astype(str).str.strip()
        labels[key] = labels[key].astype(str).str.strip()
    left = left.drop_duplicates(keys, keep="last")
    label_fields = list(dict.fromkeys(keys + [label_column]))
    return left.merge(labels[label_fields].drop_duplicates(keys, keep="last"), on=keys, how="inner"), label_column


def evaluate_word_level(frame: pd.DataFrame, label_column: str) -> dict[str, object]:
    labels = frame[label_column].astype(str).str.strip().str.lower()
    decisions = frame["final_word_decision"].astype(str).str.strip().str.lower()
    valid = ~labels.isin(EXCLUDED_LABELS)
    deletion = _binary_metrics(labels[valid].isin(DELETION_LABELS), decisions[valid].eq("deletion"))
    mispronunciation = _binary_metrics(
        labels[valid].isin(MISPRONUNCIATION_LABELS), decisions[valid].eq("mispronounced")
    )
    correct = labels.isin({"correct", "word_correct"})
    false_alarm = correct & ~decisions.eq("correct")
    return {
        "matched_words": int(len(frame)),
        "evaluated_words": int(valid.sum()),
        "deletion_precision": deletion["precision"],
        "deletion_recall": deletion["recall"],
        "deletion_f1": deletion["f1"],
        "mispronunciation_precision": mispronunciation["precision"],
        "mispronunciation_recall": mispronunciation["recall"],
        "mispronunciation_f1": mispronunciation["f1"],
        "alignment_issue_rate": _rate(decisions.eq("alignment_issue")),
        "correct_false_alarm_rate": _rate(false_alarm[correct]) if correct.any() else 0.0,
        "label_counts": _counts(labels),
        "decision_counts": _counts(decisions),
        "note": "Word-level diagnosis metrics are separate from phone-level/public-corpus metrics.",
    }


def _group_metrics(frame: pd.DataFrame, label_column: str, group_column: str) -> pd.DataFrame:
    if group_column not in frame.columns:
        return pd.DataFrame(columns=[group_column, "support"])
    rows = []
    values = frame[group_column].fillna("unknown").astype(str)
    for value, group in frame.groupby(values, dropna=False):
        report = evaluate_word_level(group, label_column)
        rows.append({group_column: value, "support": len(group), **{
            key: report[key] for key in (
                "deletion_precision", "deletion_recall", "deletion_f1",
                "mispronunciation_precision", "mispronunciation_recall", "mispronunciation_f1",
                "alignment_issue_rate", "correct_false_alarm_rate",
            )
        }})
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


def _rate(values: pd.Series) -> float:
    return round(float(values.mean()), 6) if len(values) else 0.0


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts().to_dict().items()}


if __name__ == "__main__":
    main()
