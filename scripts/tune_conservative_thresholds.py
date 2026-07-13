#!/usr/bin/env python
"""Search conservative decision thresholds under a correct-audio false-positive cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.decision import DecisionConfig, apply_decision_rules


def tune_thresholds(
    *,
    sanity_prediction: Path,
    hardset_prediction: Path | None,
    hardset_labels: Path | None,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sanity = pd.read_csv(sanity_prediction) if sanity_prediction.exists() else pd.DataFrame()
    hardset = _load_hardset(hardset_prediction, hardset_labels)
    rows = []
    for error_thr in [0.90, 0.92, 0.95, 0.98]:
        for conf_thr in [0.70, 0.75, 0.80, 0.85]:
            for prob_correct_thr in [0.40, 0.35, 0.30, 0.25]:
                config = DecisionConfig(
                    mode="conservative",
                    true_error_probability_threshold=error_thr,
                    confidence_threshold=conf_thr,
                    max_prob_correct_for_true_error=prob_correct_thr,
                )
                sanity_eval = apply_decision_rules(sanity, config) if not sanity.empty else sanity
                fp_rate = float(sanity_eval["decision"].eq("true_error").mean()) if len(sanity_eval) else 0.0
                precision, recall, f1 = _hardset_metrics(hardset, config)
                rows.append(
                    {
                        "manual_calibrated_error_probability_threshold": error_thr,
                        "confidence_threshold": conf_thr,
                        "max_prob_correct_for_true_error": prob_correct_thr,
                        "correct_audio_true_error_rate": fp_rate,
                        "true_error_precision": precision,
                        "true_error_recall": recall,
                        "true_error_f1": f1,
                    }
                )
    result = pd.DataFrame(rows)
    feasible = result[
        (result["correct_audio_true_error_rate"] <= 0.05)
        & (result["true_error_precision"] >= 0.40)
    ].copy()
    if feasible.empty:
        feasible = result[result["correct_audio_true_error_rate"] <= 0.05].copy()
    best = feasible.sort_values(["true_error_recall", "true_error_precision"], ascending=False).head(1)
    best_row = best.iloc[0].to_dict() if not best.empty else {}
    result.to_csv(output_dir / "threshold_tradeoff_table.csv", index=False, encoding="utf-8-sig")
    (output_dir / "best_conservative_thresholds.json").write_text(
        json.dumps(best_row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "threshold_tradeoff_report.md").write_text(_report(best_row, result), encoding="utf-8")
    return best_row


def _load_hardset(prediction: Path | None, labels: Path | None) -> pd.DataFrame:
    if prediction is None or labels is None or not prediction.exists() or not labels.exists():
        return pd.DataFrame()
    pred = pd.read_csv(prediction)
    lab = pd.read_csv(labels)
    keys = [col for col in ["utterance_id", "speaker_id", "word", "target_phone", "start_ms", "end_ms"] if col in pred.columns and col in lab.columns]
    if not keys:
        return pd.DataFrame()
    merged = pred.merge(lab, on=keys, how="inner", suffixes=("", "_manual"))
    return merged


def _hardset_metrics(frame: pd.DataFrame, config: DecisionConfig) -> tuple[float, float, float]:
    if frame.empty or "manual_review_label" not in frame.columns:
        return 0.0, 0.0, 0.0
    decided = apply_decision_rules(frame, config)
    pred = decided["decision"].eq("true_error")
    gold = frame["manual_review_label"].astype(str).eq("true_error")
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _report(best: dict[str, object], table: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Conservative Threshold Tradeoff Report",
            "",
            "Goal: correct_audio_true_error_rate <= 5%, then hard-set precision >= 0.40, recall as high as possible.",
            "",
            "## Best Thresholds",
            "```json",
            json.dumps(best, indent=2, ensure_ascii=False),
            "```",
            "",
            f"Scanned combinations: {len(table)}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune conservative thresholds.")
    parser.add_argument("--sanity-prediction", type=Path, default=ROOT / "outputs/sanity_correct/prediction_all.csv")
    parser.add_argument("--hardset-prediction", type=Path)
    parser.add_argument("--hardset-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/conservative_thresholds")
    args = parser.parse_args()
    best = tune_thresholds(
        sanity_prediction=args.sanity_prediction,
        hardset_prediction=args.hardset_prediction,
        hardset_labels=args.hardset_labels,
        output_dir=args.output_dir,
    )
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
