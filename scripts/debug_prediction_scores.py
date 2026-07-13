#!/usr/bin/env python
"""Debug score saturation and decision-rule failures in prediction CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCORE_COLUMNS = [
    "model_error_score",
    "prob_correct",
    "manual_calibrated_error_probability",
    "confidence",
]


def analyze_prediction_scores(path: Path) -> dict:
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    report: dict[str, object] = {
        "path": str(path),
        "rows": int(len(frame)),
        "decision_distribution": _counts(frame, "decision"),
        "alignment_quality_distribution": _counts(frame, "alignment_quality"),
        "score_describe": {},
    }
    for col in SCORE_COLUMNS:
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            report["score_describe"][col] = _describe(values)
        else:
            report["score_describe"][col] = {"missing": True}

    prob_correct = pd.to_numeric(frame.get("prob_correct", 0.5), errors="coerce").fillna(0.5)
    error_prob = pd.to_numeric(frame.get("manual_calibrated_error_probability", 0.5), errors="coerce").fillna(0.5)
    model_error = pd.to_numeric(frame.get("model_error_score", 0.5), errors="coerce").fillna(0.5)
    decision = frame.get("decision", pd.Series("", index=frame.index)).astype(str)
    report["high_prob_correct_but_true_error_count"] = int(((prob_correct >= 0.75) & decision.eq("true_error")).sum())
    report["error_probability_eq_1_count"] = int((error_prob == 1.0).sum())
    report["possible_probability_saturation_bug"] = bool(len(frame) > 0 and (error_prob == 1.0).all())
    report["possible_fallback_model_bug"] = bool(len(frame) > 1 and model_error.nunique(dropna=False) == 1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug prediction score distributions.")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_prediction_scores(args.prediction)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


def _counts(frame: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[col].value_counts(dropna=False).to_dict().items()}


def _describe(values: pd.Series) -> dict[str, float | int]:
    desc = values.describe()
    out = {str(k): round(float(v), 6) for k, v in desc.to_dict().items()}
    out["nan_count"] = int(values.isna().sum())
    return out


if __name__ == "__main__":
    main()

