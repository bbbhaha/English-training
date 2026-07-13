#!/usr/bin/env python
"""Run phase-1 SpeechOcean modeling, calibration, evaluation, and samples.

This script is intentionally CPU-friendly.  It uses the aligned SpeechOcean
phone manifest, trains simple tabular baselines, calibrates global,
phone-group, and target-phone thresholds on the dev split, then reports test
metrics with the error class called out explicitly.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]

CATEGORICAL_FEATURES = [
    "target_phone",
    "phone_group",
    "speaker_gender",
]
NUMERIC_FEATURES = [
    "duration_ms",
    "phone_index",
    "word_index",
    "speaker_age",
]
OPTIONAL_NUMERIC_FEATURES = [
    "gop_score",
    "evidence_score",
    "target_log_likelihood",
    "competitor_log_likelihood",
    "alignment_score",
]
MODEL_NAMES = ["majority_class", "feature_logreg", "feature_random_forest"]
CALIBRATIONS = ["majority_class", "global_threshold", "phone_group_threshold", "target_phone_threshold"]


@dataclass
class Thresholds:
    global_threshold: float
    phone_group: dict[str, float]
    target_phone: dict[str, float]
    target_phone_fallback: dict[str, str]


def active_numeric_features(df: pd.DataFrame) -> list[str]:
    return NUMERIC_FEATURES + [
        col
        for col in OPTIONAL_NUMERIC_FEATURES
        if col in df.columns and col not in NUMERIC_FEATURES
    ]


def load_manifest(path: Path, alignment_quality: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if alignment_quality.lower() != "all":
        df = df[df["alignment_quality"].str.lower() == alignment_quality.lower()].copy()
    if df.empty:
        raise SystemExit(f"No rows after alignment_quality={alignment_quality!r} filter.")
    df["gold_binary"] = df["gold_binary"].astype(int)
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    for col in NUMERIC_FEATURES + [c for c in OPTIONAL_NUMERIC_FEATURES if c in df.columns]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_models(random_state: int, numeric_features: list[str]) -> dict[str, Pipeline]:
    return {
        "feature_logreg": Pipeline(
            [
                ("preprocess", build_preprocessor(numeric_features)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "feature_random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor(numeric_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def probability_of_correct(model: Pipeline, frame: pd.DataFrame, numeric_features: list[str]) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    return model.predict_proba(frame[numeric_features + CATEGORICAL_FEATURES])[:, classes.index(1)]


def score_at_threshold(y_true: pd.Series, scores: np.ndarray, threshold: float, objective: str) -> float:
    y_pred = (scores >= threshold).astype(int)
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if objective == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(objective)


def find_best_threshold(y_true: pd.Series, scores: np.ndarray, objective: str) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(scores, np.linspace(0.01, 0.99, 99))]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        value = score_at_threshold(y_true, scores, float(threshold), objective)
        if value > best_score:
            best_score = value
            best_threshold = float(threshold)
    return best_threshold


def calibrate_thresholds(
    dev: pd.DataFrame,
    score_col: str,
    objective: str,
    min_target_samples: int,
    min_group_samples: int,
) -> Thresholds:
    global_threshold = find_best_threshold(dev["gold_binary"], dev[score_col].to_numpy(), objective)
    group_thresholds: dict[str, float] = {}
    for group, group_dev in dev.groupby("phone_group"):
        y = group_dev["gold_binary"]
        if len(group_dev) >= min_group_samples and y.nunique() == 2:
            group_thresholds[str(group)] = find_best_threshold(y, group_dev[score_col].to_numpy(), objective)
        else:
            group_thresholds[str(group)] = global_threshold

    target_thresholds: dict[str, float] = {}
    target_fallback: dict[str, str] = {}
    for phone, phone_dev in dev.groupby("target_phone"):
        y = phone_dev["gold_binary"]
        group = str(phone_dev["phone_group"].iloc[0])
        if len(phone_dev) >= min_target_samples and y.nunique() == 2:
            target_thresholds[str(phone)] = find_best_threshold(y, phone_dev[score_col].to_numpy(), objective)
            target_fallback[str(phone)] = "target_phone"
        else:
            target_thresholds[str(phone)] = group_thresholds.get(group, global_threshold)
            target_fallback[str(phone)] = "phone_group"
    return Thresholds(global_threshold, group_thresholds, target_thresholds, target_fallback)


def apply_calibration(frame: pd.DataFrame, thresholds: Thresholds, calibration: str, score_col: str) -> pd.DataFrame:
    out = frame.copy()
    if calibration == "global_threshold":
        out["threshold"] = thresholds.global_threshold
        out["threshold_source"] = "global"
    elif calibration == "phone_group_threshold":
        out["threshold"] = out["phone_group"].map(thresholds.phone_group).fillna(thresholds.global_threshold)
        out["threshold_source"] = "phone_group"
    elif calibration == "target_phone_threshold":
        out["threshold"] = out["target_phone"].map(thresholds.target_phone)
        out["threshold"] = out["threshold"].fillna(out["phone_group"].map(thresholds.phone_group)).fillna(thresholds.global_threshold)
        out["threshold_source"] = out["target_phone"].map(thresholds.target_phone_fallback).fillna("phone_group")
    else:
        raise ValueError(calibration)
    out["prediction"] = (out[score_col].astype(float) >= out["threshold"].astype(float)).astype(int)
    out["confidence"] = np.maximum(out[score_col].astype(float), 1.0 - out[score_col].astype(float)).round(6)
    out["calibration"] = calibration
    return out


def metric_row(frame: pd.DataFrame, model: str, calibration: str, score_col: str) -> dict[str, object]:
    y_true = frame["gold_binary"].astype(int)
    y_pred = frame["prediction"].astype(int)
    scores = frame[score_col].astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    # gold_binary=0 is the error class.  These two fields are the plan-book
    # "error phone precision/recall" rather than the sklearn positive class.
    error_precision = tn / (tn + fn) if (tn + fn) else 0.0
    error_recall = tn / (tn + fp) if (tn + fp) else 0.0
    error_f1 = 2 * error_precision * error_recall / (error_precision + error_recall) if (error_precision + error_recall) else 0.0
    return {
        "split": str(frame["split"].iloc[0]),
        "model": model,
        "calibration": calibration,
        "n": len(frame),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision_correct_class": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_correct_class": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "auc": round(float(auc), 6),
        "error_precision": round(float(error_precision), 6),
        "error_recall": round(float(error_recall), 6),
        "error_f1": round(float(error_f1), 6),
        "tn_error_predicted_error": int(tn),
        "fp_correct_predicted_error": int(fp),
        "fn_error_predicted_correct": int(fn),
        "tp_correct_predicted_correct": int(tp),
    }


def majority_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["model"] = "majority_class"
    out["prob_correct"] = 1.0
    out["threshold"] = 0.5
    out["threshold_source"] = "majority_class"
    out["prediction"] = 1
    out["confidence"] = 1.0
    out["calibration"] = "majority_class"
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_error_analysis(predictions: pd.DataFrame, report_dir: Path) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    test["case_type"] = np.select(
        [
            (test["gold_binary"] == 0) & (test["prediction"] == 0),
            (test["gold_binary"] == 0) & (test["prediction"] == 1),
            (test["gold_binary"] == 1) & (test["prediction"] == 0),
            (test["gold_binary"] == 1) & (test["prediction"] == 1),
        ],
        ["true_error_detected", "missed_error", "false_alarm", "true_correct"],
        default="unknown",
    )
    group_rows = []
    for group, g in test.groupby("phone_group"):
        row = {"phone_group": group, "n": len(g)}
        row.update(g["case_type"].value_counts().to_dict())
        group_rows.append(row)
    phone_rows = []
    for phone, g in test.groupby("target_phone"):
        row = {"target_phone": phone, "phone_group": g["phone_group"].iloc[0], "n": len(g)}
        row.update(g["case_type"].value_counts().to_dict())
        phone_rows.append(row)
    pd.DataFrame(group_rows).fillna(0).sort_values("n", ascending=False).to_csv(report_dir / "error_analysis_by_phone_group.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phone_rows).fillna(0).sort_values("n", ascending=False).to_csv(report_dir / "error_analysis_by_target_phone.csv", index=False, encoding="utf-8-sig")
    cases = test[test["case_type"].isin(["missed_error", "false_alarm"])].copy()
    keep = [
        "utterance_id",
        "speaker_id",
        "transcript",
        "word",
        "target_phone",
        "phone_group",
        "start_ms",
        "end_ms",
        "gold_binary",
        "prediction",
        "prob_correct",
        "confidence",
        "case_type",
        "audio_path",
    ]
    cases[keep].head(300).to_csv(report_dir / "error_cases.csv", index=False, encoding="utf-8-sig")
    summary = [
        "# Error Analysis",
        "",
        f"Test rows: {len(test)}",
        "",
        "Case counts:",
        "",
    ]
    for name, value in test["case_type"].value_counts().items():
        summary.append(f"- {name}: {value}")
    summary.extend(["", "Largest phone groups by sample count are in `error_analysis_by_phone_group.csv`."])
    (report_dir / "error_analysis_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/speechocean/phones_aligned.csv")
    parser.add_argument("--alignment-quality", default="pass")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/phase1")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/phase1_models")
    parser.add_argument("--objective", choices=["balanced_accuracy", "macro_f1"], default="balanced_accuracy")
    parser.add_argument("--min-target-samples", type=int, default=40)
    parser.add_argument("--min-group-samples", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.input, args.alignment_quality)
    numeric_features = active_numeric_features(df)
    train = df[df["split"] == "train"].copy()
    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or dev.empty or test.empty:
        raise SystemExit("Input must contain non-empty train/dev/test splits.")

    data_manifest = [
        {
            "file": str(args.input.relative_to(ROOT) if args.input.is_relative_to(ROOT) else args.input),
            "role": "primary_aligned_model_input",
            "alignment_quality_filter": args.alignment_quality,
            "rows": len(df),
            "train_rows": len(train),
            "dev_rows": len(dev),
            "test_rows": len(test),
            "gold_binary_0_error": int((df["gold_binary"] == 0).sum()),
            "gold_binary_1_correct": int((df["gold_binary"] == 1).sum()),
            "speakers_train": train["speaker_id"].nunique(),
            "speakers_dev": dev["speaker_id"].nunique(),
            "speakers_test": test["speaker_id"].nunique(),
        }
    ]
    write_csv(args.report_dir / "data_manifest.csv", data_manifest)

    metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    majority = pd.concat([majority_predictions(dev), majority_predictions(test)], ignore_index=True)
    for split in ["dev", "test"]:
        frame = majority[majority["split"] == split]
        metrics.append(metric_row(frame, "majority_class", "majority_class", "prob_correct"))
    prediction_frames.append(majority)

    threshold_rows: list[dict[str, object]] = []
    for model_name, model in build_models(args.random_state, numeric_features).items():
        model.fit(train[numeric_features + CATEGORICAL_FEATURES], train["gold_binary"])
        joblib.dump(model, args.model_dir / f"{model_name}.joblib")
        scored_frames = []
        for split_name, split_frame in [("dev", dev), ("test", test)]:
            scored = split_frame.copy()
            scored["model"] = model_name
            scored["prob_correct"] = probability_of_correct(model, scored, numeric_features)
            scored_frames.append(scored)
        scored_all = pd.concat(scored_frames, ignore_index=True)
        scored_dev = scored_all[scored_all["split"] == "dev"].copy()
        thresholds = calibrate_thresholds(
            scored_dev,
            "prob_correct",
            args.objective,
            args.min_target_samples,
            args.min_group_samples,
        )
        threshold_rows.append(
            {
                "model": model_name,
                "level": "global",
                "key": "*",
                "threshold": round(thresholds.global_threshold, 6),
                "fallback": "",
            }
        )
        for group, value in sorted(thresholds.phone_group.items()):
            threshold_rows.append(
                {
                    "model": model_name,
                    "level": "phone_group",
                    "key": group,
                    "threshold": round(value, 6),
                    "fallback": "global" if value == thresholds.global_threshold else "",
                }
            )
        for phone, value in sorted(thresholds.target_phone.items()):
            threshold_rows.append(
                {
                    "model": model_name,
                    "level": "target_phone",
                    "key": phone,
                    "threshold": round(value, 6),
                    "fallback": thresholds.target_phone_fallback.get(phone, ""),
                }
            )
        for calibration in ["global_threshold", "phone_group_threshold", "target_phone_threshold"]:
            calibrated = apply_calibration(scored_all, thresholds, calibration, "prob_correct")
            for split in ["dev", "test"]:
                frame = calibrated[calibrated["split"] == split]
                metrics.append(metric_row(frame, model_name, calibration, "prob_correct"))
            prediction_frames.append(calibrated)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.report_dir / "formal_eval_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_rows).to_csv(args.report_dir / "thresholds.csv", index=False, encoding="utf-8-sig")
    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values(["balanced_accuracy", "macro_f1", "auc"], ascending=False)
    test_metrics.to_csv(args.report_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    best = test_metrics.iloc[0].to_dict()

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    best_predictions = all_predictions[
        (all_predictions["model"] == best["model"])
        & (all_predictions["calibration"] == best["calibration"])
    ].copy()
    prediction_cols = [
        "utterance_id",
        "speaker_id",
        "transcript",
        "word",
        "target_phone",
        "phone_index",
        "start_ms",
        "end_ms",
        "duration_ms",
        "gold_binary",
        "phone_group",
        "split",
        "audio_path",
        "prob_correct",
        "threshold",
        "threshold_source",
        "prediction",
        "confidence",
        "model",
        "calibration",
    ]
    best_predictions[prediction_cols].to_csv(args.report_dir / "best_model_predictions.csv", index=False, encoding="utf-8-sig")
    sample_utterances = best_predictions[best_predictions["split"] == "test"]["utterance_id"].drop_duplicates().head(100)
    best_predictions[best_predictions["utterance_id"].isin(sample_utterances)][prediction_cols].to_csv(
        args.report_dir / "prediction_samples_100_utterances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"utterance_id": sample_utterances}).to_csv(
        args.report_dir / "prediction_sample_utterances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_error_analysis(best_predictions, args.report_dir)

    summary = [
        "# Phase-1 Formal Evaluation Summary",
        "",
        f"Input: `{args.input}`",
        f"Alignment filter: `{args.alignment_quality}`",
        f"Threshold objective: `{args.objective}`",
        "",
        "## Current Best Test Result",
        "",
        f"- Model: `{best['model']}`",
        f"- Calibration: `{best['calibration']}`",
        f"- Balanced Accuracy: {best['balanced_accuracy']}",
        f"- Macro-F1: {best['macro_f1']}",
        f"- AUC: {best['auc']}",
        f"- Error Precision: {best['error_precision']}",
        f"- Error Recall: {best['error_recall']}",
        f"- Error F1: {best['error_f1']}",
        "",
        "See `formal_eval_metrics.csv`, `model_comparison.csv`, `thresholds.csv`, and `best_model_predictions.csv` for details.",
    ]
    (args.report_dir / "formal_eval_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
