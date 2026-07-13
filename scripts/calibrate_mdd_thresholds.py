#!/usr/bin/env python
"""Calibrate global, phone-group, and target-phone thresholds for MDD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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


def add_scores(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    if "normalized_duration_by_phone" in bundle.get("numeric_features", []):
        frame = apply_duration_stats(frame, bundle.get("duration_stats_by_phone", {}))
    pipeline = bundle["pipeline"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    for column in numeric:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in categorical:
        if column not in frame.columns:
            frame[column] = ""
    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    out = frame.copy()
    out["score"] = pipeline.predict_proba(out[numeric + categorical])[:, positive_index]
    return out


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


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else 0.5,
        "auprc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float(np.mean(y_true)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def select_threshold(y_true: np.ndarray, scores: np.ndarray, min_precision: float) -> dict[str, object]:
    if len(y_true) == 0:
        return {"threshold": 1.0, "meets_precision": False, "reason": "empty"}
    if len(np.unique(y_true)) < 2:
        # With no positives or no negatives, PR calibration is not meaningful.
        default = 1.0 if y_true.sum() == 0 else 0.0
        return {"threshold": default, "meets_precision": False, "reason": "single_class", **metrics(y_true, scores, default)}
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid):
        index = valid[np.argmax(recall[:-1][valid])]
        threshold = float(thresholds[index])
        return {"meets_precision": True, "reason": "max_recall_at_min_precision", **metrics(y_true, scores, threshold)}
    # Fallback: choose threshold with highest precision, then highest recall.
    index = int(np.lexsort((recall[:-1], precision[:-1]))[-1])
    threshold = float(thresholds[index])
    return {"meets_precision": False, "reason": "precision_target_unreachable", **metrics(y_true, scores, threshold)}


def build_global(dev: pd.DataFrame, min_precision: float) -> dict[str, object]:
    return select_threshold(dev["label"].to_numpy(), dev["score"].to_numpy(), min_precision)


def build_group(dev: pd.DataFrame, global_threshold: dict[str, object], min_precision: float, min_total: int) -> dict[str, object]:
    result = {"fallback_threshold": global_threshold["threshold"], "groups": {}}
    for group, subset in sorted(dev.groupby("phone_group")):
        if len(subset) < min_total or subset["label"].nunique() < 2:
            result["groups"][str(group)] = {"threshold": global_threshold["threshold"], "fallback": "global", "reason": "insufficient_group_data"}
        else:
            result["groups"][str(group)] = {**select_threshold(subset["label"].to_numpy(), subset["score"].to_numpy(), min_precision), "fallback": None}
    return result


def build_phone(
    dev: pd.DataFrame,
    group_thresholds: dict[str, object],
    global_threshold: dict[str, object],
    min_precision: float,
    min_positive: int,
    min_total: int,
) -> dict[str, object]:
    result = {
        "fallback_global_threshold": global_threshold["threshold"],
        "phones": {},
    }
    for phone, subset in sorted(dev.groupby("target_phone")):
        group = str(subset["phone_group"].iloc[0])
        group_info = group_thresholds["groups"].get(group, {"threshold": global_threshold["threshold"]})
        positives = int(subset["label"].sum())
        if len(subset) < min_total or positives < min_positive or subset["label"].nunique() < 2:
            result["phones"][str(phone)] = {
                "threshold": group_info["threshold"],
                "fallback": "phone_group",
                "phone_group": group,
                "reason": "insufficient_phone_data",
                "dev_total": int(len(subset)),
                "dev_positive": positives,
            }
        else:
            result["phones"][str(phone)] = {
                **select_threshold(subset["label"].to_numpy(), subset["score"].to_numpy(), min_precision),
                "fallback": None,
                "phone_group": group,
                "dev_total": int(len(subset)),
                "dev_positive": positives,
            }
    return result


def threshold_for_row(row: pd.Series, strategy: str, global_threshold: dict, group_thresholds: dict, phone_thresholds: dict) -> tuple[float, str]:
    if strategy == "global":
        return float(global_threshold["threshold"]), "global"
    if strategy == "phone_group":
        group = str(row["phone_group"])
        info = group_thresholds["groups"].get(group)
        if info:
            return float(info["threshold"]), "phone_group" if not info.get("fallback") else info["fallback"]
        return float(global_threshold["threshold"]), "global"
    if strategy == "target_phone":
        phone = str(row["target_phone"])
        info = phone_thresholds["phones"].get(phone)
        if info:
            return float(info["threshold"]), "target_phone" if not info.get("fallback") else info["fallback"]
        return threshold_for_row(row, "phone_group", global_threshold, group_thresholds, phone_thresholds)
    raise ValueError(strategy)


def evaluate_strategy(frame: pd.DataFrame, strategy: str, global_threshold: dict, group_thresholds: dict, phone_thresholds: dict) -> tuple[dict, pd.DataFrame]:
    out = frame.copy()
    thresholds = [threshold_for_row(row, strategy, global_threshold, group_thresholds, phone_thresholds) for _, row in out.iterrows()]
    out["threshold"] = [item[0] for item in thresholds]
    out["threshold_source"] = [item[1] for item in thresholds]
    out["prediction"] = (out["score"] >= out["threshold"]).astype(int)
    metric = metrics(out["label"].to_numpy(), out["score"].to_numpy(), 0.5)
    # Replace threshold-based counts with row-specific threshold counts.
    tn, fp, fn, tp = confusion_matrix(out["label"], out["prediction"], labels=[0, 1]).ravel()
    metric.update(
        {
            "threshold": strategy,
            "accuracy": float(accuracy_score(out["label"], out["prediction"])),
            "balanced_accuracy": float(balanced_accuracy_score(out["label"], out["prediction"])),
            "precision": float(precision_score(out["label"], out["prediction"], zero_division=0)),
            "recall": float(recall_score(out["label"], out["prediction"], zero_division=0)),
            "f1": float(f1_score(out["label"], out["prediction"], zero_division=0)),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        }
    )
    return metric, out


def write_error_analysis(predictions: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for phone, group in predictions.groupby("target_phone"):
        y = group["label"].to_numpy()
        pred = group["prediction"].to_numpy()
        scores = group["score"].to_numpy()
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "target_phone": phone,
                "phone_group": group["phone_group"].iloc[0],
                "fp_count": fp,
                "tp_count": tp,
                "precision": precision,
                "recall": recall,
                "avg_score_correct": float(scores[y == 0].mean()) if (y == 0).any() else np.nan,
                "avg_score_error": float(scores[y == 1].mean()) if (y == 1).any() else np.nan,
            }
        )
    pd.DataFrame(rows).sort_values("fp_count", ascending=False).to_csv(output_dir / "false_positive_by_phone.csv", index=False, encoding="utf-8-sig")
    fp_examples = predictions[(predictions["label"] == 0) & (predictions["prediction"] == 1)].copy()
    fp_examples = fp_examples.sort_values("score", ascending=False)
    keep = ["utt_id", "speaker_id", "word", "target_phone", "start", "end", "label", "score", "prediction", "threshold", "wav_path"]
    fp_examples[keep].head(300).rename(columns={"label": "gold_label", "score": "predicted_score", "prediction": "predicted_label"}).to_csv(output_dir / "false_positive_examples.csv", index=False, encoding="utf-8-sig")


def per_phone_pr_summary(frame: pd.DataFrame, output_dir: Path, min_precision: float) -> None:
    rows = []
    for phone, group in frame.groupby("target_phone"):
        y = group["label"].to_numpy()
        scores = group["score"].to_numpy()
        if len(np.unique(y)) < 2:
            rows.append({"target_phone": phone, "phone_group": group["phone_group"].iloc[0], "n": len(group), "positive": int(y.sum()), "precision_at_min": 0, "recall_at_min": 0, "f1_at_min": 0, "auprc": float(y.mean()), "meets_acceptance": False, "reason": "single_class"})
            continue
        selected = select_threshold(y, scores, min_precision)
        rows.append(
            {
                "target_phone": phone,
                "phone_group": group["phone_group"].iloc[0],
                "n": len(group),
                "positive": int(y.sum()),
                "precision_at_min": selected.get("precision", 0),
                "recall_at_min": selected.get("recall", 0),
                "f1_at_min": selected.get("f1", 0),
                "auprc": selected.get("auprc", 0),
                "threshold": selected.get("threshold"),
                "meets_acceptance": bool(selected.get("precision", 0) >= min_precision and selected.get("recall", 0) >= 0.5),
                "reason": selected.get("reason"),
            }
        )
    pd.DataFrame(rows).sort_values(["meets_acceptance", "recall_at_min"], ascending=False).to_csv(output_dir / "per_phone_pr_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("artifacts/mdd_wav2vec2/features.npz"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/mdd_wav2vec2/mdd_classifier.joblib"))
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--min-precision", type=float, default=0.40)
    parser.add_argument("--min-group-total", type=int, default=50)
    parser.add_argument("--min-phone-total", type=int, default=50)
    parser.add_argument("--min-phone-positive", type=int, default=10)
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mdd_thresholds"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_scores(load_feature_frame(args.features, args.min_duration), args.model)
    dev = frame[frame["split"] == args.dev_split].copy()
    evaluation = frame[frame["split"] == args.eval_split].copy()
    global_threshold = build_global(dev, args.min_precision)
    group_thresholds = build_group(dev, global_threshold, args.min_precision, args.min_group_total)
    phone_thresholds = build_phone(dev, group_thresholds, global_threshold, args.min_precision, args.min_phone_positive, args.min_phone_total)

    (args.output_dir / "thresholds_global.json").write_text(json.dumps(global_threshold, indent=2), encoding="utf-8")
    (args.output_dir / "thresholds_by_phone_group.json").write_text(json.dumps(group_thresholds, indent=2), encoding="utf-8")
    (args.output_dir / "thresholds_by_target_phone.json").write_text(json.dumps(phone_thresholds, indent=2), encoding="utf-8")
    # Also satisfy requested generic outputs/ filenames.
    generic = Path("outputs")
    generic.mkdir(exist_ok=True)
    (generic / "thresholds_global.json").write_text(json.dumps(global_threshold, indent=2), encoding="utf-8")
    (generic / "thresholds_by_phone_group.json").write_text(json.dumps(group_thresholds, indent=2), encoding="utf-8")
    (generic / "thresholds_by_target_phone.json").write_text(json.dumps(phone_thresholds, indent=2), encoding="utf-8")

    comparison_rows = []
    prediction_outputs = {}
    for strategy in ["global", "phone_group", "target_phone"]:
        metric, preds = evaluate_strategy(evaluation, strategy, global_threshold, group_thresholds, phone_thresholds)
        metric["threshold_strategy"] = strategy
        metric["precision_at_least_0_40_max_recall"] = metric["recall"] if metric["precision"] >= args.min_precision else 0.0
        comparison_rows.append(metric)
        prediction_outputs[strategy] = preds
        preds[["utt_id", "speaker_id", "wav_path", "word", "target_phone", "phone_group", "start", "end", "duration", "label", "score", "threshold", "threshold_source", "prediction"]].to_csv(args.output_dir / f"predictions_{strategy}.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "threshold_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    best_strategy = max(comparison_rows, key=lambda row: (row["precision"] >= args.min_precision, row["recall"], row["f1"]))
    best_predictions = prediction_outputs[best_strategy["threshold_strategy"]]
    write_error_analysis(best_predictions, args.output_dir)
    per_phone_pr_summary(evaluation, args.output_dir, args.min_precision)
    summary = {"best_strategy": best_strategy, "rows": comparison_rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
