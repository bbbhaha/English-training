#!/usr/bin/env python
"""Evaluate wav2vec2-MDD with mispronounced as the positive class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_feature_frame(features_path: Path, min_duration: float = 0.0) -> pd.DataFrame:
    matrix = np.load(features_path)["embeddings"]
    metadata = pd.read_csv(features_path.with_suffix(".metadata.csv"), encoding="utf-8-sig", keep_default_na=False)
    if min_duration > 0:
        duration = pd.to_numeric(metadata["duration"], errors="coerce")
        keep = duration >= min_duration
        metadata = metadata.loc[keep].reset_index(drop=True)
        matrix = matrix[keep.to_numpy()]
    feature_columns = pd.DataFrame(matrix, columns=[f"w2v_{i}" for i in range(matrix.shape[1])])
    frame = pd.concat([metadata.reset_index(drop=True), feature_columns], axis=1)
    frame["label"] = frame["label"].astype(int)
    return frame


def predict_scores(model_path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    bundle = joblib.load(model_path)
    if "normalized_duration_by_phone" in bundle.get("numeric_features", []):
        frame = apply_duration_stats(frame, bundle.get("duration_stats_by_phone", {}))
    pipeline = bundle["pipeline"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    scores = pipeline.predict_proba(frame[numeric + categorical])[:, positive_index]
    return scores, bundle


def apply_duration_stats(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = frame.copy()
    durations = pd.to_numeric(out["duration"], errors="coerce")
    global_mean = float(durations.mean())
    global_std = float(durations.std(ddof=0)) or 1.0
    values = []
    for _, row in out.iterrows():
        item = stats.get(str(row["target_phone"]), {"mean": global_mean, "std": global_std})
        values.append((float(row["duration"]) - item["mean"]) / item["std"])
    out["normalized_duration_by_phone"] = values
    return out


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y_true, pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 6),
        "auc": round(float(auc), 6),
        "tn_correct_predicted_correct": int(tn),
        "fp_correct_predicted_mispronounced": int(fp),
        "fn_mispronounced_predicted_correct": int(fn),
        "tp_mispronounced_predicted_mispronounced": int(tp),
    }


def best_balanced_accuracy_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best = (-1.0, 0.5)
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 501), np.quantile(scores, np.linspace(0.0, 1.0, 501))]))
    for threshold in candidates:
        value = balanced_accuracy_score(y_true, scores >= threshold)
        if value > best[0]:
            best = (float(value), float(threshold))
    return best[1]


def best_recall_at_precision(y_true: np.ndarray, scores: np.ndarray, min_precision: float) -> dict[str, object]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return {
            "found": False,
            "min_precision": min_precision,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "threshold": None,
        }
    index = valid[np.argmax(recall[:-1][valid])]
    threshold = float(thresholds[index])
    metrics = threshold_metrics(y_true, scores, threshold)
    return {
        "found": True,
        "min_precision": min_precision,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "threshold": metrics["threshold"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("artifacts/mdd_wav2vec2/features.npz"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/mdd_wav2vec2/mdd_classifier.joblib"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-precision", type=float, default=0.40)
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mdd_wav2vec2"))
    args = parser.parse_args()

    frame = load_feature_frame(args.features, args.min_duration)
    split_frame = frame[frame["split"] == args.split].copy()
    if split_frame.empty:
        raise SystemExit(f"No rows for split={args.split!r}")
    scores, bundle = predict_scores(args.model, split_frame)
    y_true = split_frame["label"].to_numpy(dtype=int)
    threshold = best_balanced_accuracy_threshold(y_true, scores)
    default_metrics = threshold_metrics(y_true, scores, threshold)
    precision_point = best_recall_at_precision(y_true, scores, args.min_precision)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_curve = pd.DataFrame(
        {
            "precision": precision[:-1],
            "recall": recall[:-1],
            "threshold": thresholds,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pr_curve.to_csv(args.output_dir / "pr_curve.csv", index=False, encoding="utf-8-sig")
    predictions = split_frame[["utt_id", "speaker_id", "wav_path", "word", "target_phone", "phone_group", "start", "end", "duration", "label", "split"]].copy()
    predictions["mispronounced_probability"] = scores
    predictions["prediction"] = (scores >= threshold).astype(int)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "positive_class": "mispronounced",
        "split": args.split,
        "rows": int(len(split_frame)),
        "min_duration": args.min_duration,
        "labels": {str(k): int(v) for k, v in split_frame["label"].value_counts().to_dict().items()},
        "balanced_accuracy_threshold_metrics": default_metrics,
        f"max_recall_at_precision_{args.min_precision}": precision_point,
        "model": {
            "classifier": bundle.get("classifier"),
            "label_definition": bundle.get("label_definition"),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = [
        "# wav2vec2-MDD Evaluation",
        "",
        "Positive class: mispronounced (`label=1`)",
        "",
        "## Balanced-Accuracy Threshold",
    ]
    for key, value in default_metrics.items():
        summary.append(f"- {key}: {value}")
    summary.extend(["", f"## Max Recall with Precision >= {args.min_precision}"])
    for key, value in precision_point.items():
        summary.append(f"- {key}: {value}")
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
