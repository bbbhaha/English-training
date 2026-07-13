#!/usr/bin/env python
"""Evaluate end-to-end pronunciation predictions against manual review labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


KEY_COLUMNS = ["utterance_id", "speaker_id", "word", "target_phone", "start_ms", "end_ms"]
PRONUNCIATION_ERROR_LABELS = {"true_error", "acceptable_accent"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wav+text end-to-end pronunciation predictions.")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--manual-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/e2e/evaluation_report.json"))
    parser.add_argument("--per-phone-output", type=Path)
    parser.add_argument("--per-speaker-output", type=Path)
    args = parser.parse_args()
    pred = pd.read_csv(args.prediction, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    labels = pd.read_csv(args.manual_labels, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    merged = _merge(pred, labels)
    if merged.empty:
        raise SystemExit("No prediction rows matched manual labels.")
    eval_frame = _true_error_eval_frame(merged)
    y_true = eval_frame["manual_review_label"].astype(str).isin(PRONUNCIATION_ERROR_LABELS).to_numpy(dtype=int)
    y_pred = eval_frame["decision"].astype(str).eq("true_error").to_numpy(dtype=int)
    acceptable = merged["manual_review_label"].astype(str).eq("acceptable_accent").to_numpy(dtype=bool)
    bad_alignment = merged["manual_review_label"].astype(str).eq("bad_alignment").to_numpy(dtype=bool)
    metrics = _binary_metrics(y_true, y_pred)
    report = {
        "matched_rows": int(len(merged)),
        "true_error_eval_rows": int(len(eval_frame)),
        "excluded_bad_alignment_rows": int(bad_alignment.sum()),
        "true_error_precision": metrics["precision"],
        "true_error_recall": metrics["recall"],
        "true_error_f1": metrics["f1"],
        "bad_alignment_predicted_true_error_count": int((merged["decision"].astype(str).eq("true_error").to_numpy(dtype=bool) & bad_alignment).sum()),
        "acceptable_accent_false_positive_rate": round(float(merged.loc[acceptable, "decision"].astype(str).eq("true_error").mean()), 6) if acceptable.any() else 0.0,
        "alignment_failure_rate": round(float(merged["alignment_quality"].astype(str).str.lower().eq("bad").mean()), 6) if "alignment_quality" in merged.columns else 0.0,
        "uncertain_review_rate": round(float(merged["decision"].astype(str).eq("uncertain_review").mean()), 6),
        "label_counts": {str(k): int(v) for k, v in merged["manual_review_label"].value_counts().to_dict().items()},
        "decision_counts": {str(k): int(v) for k, v in merged["decision"].value_counts().to_dict().items()},
        "note": "This is end-to-end manual-label evaluation and should be reported separately from public dataset gold_binary evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    per_phone = _group_metrics(merged, "target_phone")
    per_speaker = _group_metrics(merged, "speaker_id")
    (args.per_phone_output or args.output.with_name("per_phone_metrics.csv")).parent.mkdir(parents=True, exist_ok=True)
    per_phone.to_csv(args.per_phone_output or args.output.with_name("per_phone_metrics.csv"), index=False, encoding="utf-8-sig")
    per_speaker.to_csv(args.per_speaker_output or args.output.with_name("per_speaker_metrics.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _merge(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    left = pred.copy()
    right = labels.copy()
    left["_key"] = _key(left)
    right["_key"] = _key(right)
    cols = ["_key", "manual_review_label", "manual_review_notes", "reviewer", "review_date"]
    for col in cols:
        if col not in right.columns:
            right[col] = ""
    return left.merge(right[cols].drop_duplicates("_key", keep="last"), on="_key", how="inner")


def _key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for col in KEY_COLUMNS:
        if col not in frame.columns:
            value = pd.Series("", index=frame.index)
        elif col in {"start_ms", "end_ms"}:
            value = pd.to_numeric(frame[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        else:
            value = frame[col].astype(str).str.strip()
        parts.append(value)
    key = parts[0]
    for part in parts[1:]:
        key = key + "||" + part
    return key


def _group_metrics(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    if col not in frame.columns:
        return pd.DataFrame()
    for key, group in frame.groupby(col):
        eval_group = _true_error_eval_frame(group)
        y_true = eval_group["manual_review_label"].astype(str).isin(PRONUNCIATION_ERROR_LABELS).to_numpy(dtype=int)
        y_pred = eval_group["decision"].astype(str).eq("true_error").to_numpy(dtype=int)
        metrics = _binary_metrics(y_true, y_pred)
        acceptable = group["manual_review_label"].astype(str).eq("acceptable_accent").to_numpy(dtype=bool)
        rows.append(
            {
                col: str(key),
                "support": int(len(group)),
                "true_error_eval_support": int(len(eval_group)),
                "bad_alignment_support": int(group["manual_review_label"].astype(str).eq("bad_alignment").sum()),
                "true_error_support": int(y_true.sum()),
                "true_error_precision": metrics["precision"],
                "true_error_recall": metrics["recall"],
                "true_error_f1": metrics["f1"],
                "acceptable_accent_false_positive_rate": round(float(group.loc[acceptable, "decision"].astype(str).eq("true_error").mean()), 6) if acceptable.any() else 0.0,
                "uncertain_review_rate": round(float(group["decision"].astype(str).eq("uncertain_review").mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def _true_error_eval_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[~frame["manual_review_label"].astype(str).eq("bad_alignment")].copy()


def _binary_metrics(y_true, y_pred) -> dict[str, float]:
    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


if __name__ == "__main__":
    main()
