#!/usr/bin/env python
"""Train a supervised classifier on per-phone acoustic segment statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.acoustic import read_wav_mono, segment_features


CAT_FEATURES = ["target_phone", "phone_group", "speaker_gender"]
BASE_NUM_FEATURES = ["duration_ms", "phone_index", "word_index", "speaker_age"]


def extract_segment_stats(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_audio: dict[str, pd.DataFrame] = {
        audio_path: group.sort_values("phone_index")
        for audio_path, group in df.groupby("audio_path")
    }
    for index, (audio_path, group) in enumerate(by_audio.items(), start=1):
        rate, signal = read_wav_mono(ROOT / audio_path)
        for _, row in group.iterrows():
            features = segment_features(signal, rate, float(row["start_ms"]), float(row["end_ms"]))
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            out = row.to_dict()
            for i, value in enumerate(mean):
                out[f"seg_mean_{i}"] = float(value)
            for i, value in enumerate(std):
                out[f"seg_std_{i}"] = float(value)
            rows.append(out)
        if index % 250 == 0:
            print(f"Extracted segment stats for {index}/{len(by_audio)} utterances")
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def load_or_extract(input_path: Path, features_path: Path, alignment_quality: str) -> pd.DataFrame:
    if features_path.exists():
        df = pd.read_csv(features_path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    else:
        df = pd.read_csv(input_path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
        if alignment_quality.lower() != "all":
            df = df[df["alignment_quality"].str.lower() == alignment_quality.lower()].copy()
        df = extract_segment_stats(df, features_path)
    if alignment_quality.lower() != "all" and "alignment_quality" in df.columns:
        df = df[df["alignment_quality"].str.lower() == alignment_quality.lower()].copy()
    df["gold_binary"] = df["gold_binary"].astype(int)
    return df


def build_pipeline(model_name: str, numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                CAT_FEATURES,
            ),
        ]
    )
    if model_name == "logreg":
        clf = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000, random_state=42)
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    elif model_name == "extra":
        clf = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", pre), ("model", clf)])


def best_threshold(y_true: np.ndarray, scores: np.ndarray, objective: str) -> float:
    best = (-1.0, 0.5)
    for threshold in np.unique(np.concatenate([np.linspace(0.01, 0.99, 199), np.quantile(scores, np.linspace(0.01, 0.99, 199))])):
        pred = (scores >= threshold).astype(int)
        if objective == "macro_f1":
            value = f1_score(y_true, pred, average="macro", zero_division=0)
        else:
            value = balanced_accuracy_score(y_true, pred)
        if value > best[0]:
            best = (float(value), float(threshold))
    return best[1]


def precision_constrained_threshold(y_true: np.ndarray, scores: np.ndarray, min_error_precision: float) -> float | None:
    best: tuple[float, float] | None = None
    for threshold in np.unique(np.concatenate([np.linspace(0.0, 1.0, 501), np.quantile(scores, np.linspace(0.0, 1.0, 501))])):
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, _tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        error_precision = tn / (tn + fp) if (tn + fp) else 0.0
        error_recall = tn / (tn + fn) if (tn + fn) else 0.0
        if error_precision >= min_error_precision:
            candidate = (error_recall, float(threshold))
            if best is None or candidate[0] > best[0]:
                best = candidate
    return None if best is None else best[1]


def metrics(frame: pd.DataFrame, threshold: float, model_name: str, calibration: str) -> dict[str, object]:
    y_true = frame["gold_binary"].astype(int).to_numpy()
    scores = frame["prob_correct"].astype(float).to_numpy()
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    ep = tn / (tn + fp) if (tn + fp) else 0.0
    er = tn / (tn + fn) if (tn + fn) else 0.0
    ef = 2 * ep * er / (ep + er) if (ep + er) else 0.0
    return {
        "split": str(frame["split"].iloc[0]),
        "model": model_name,
        "calibration": calibration,
        "threshold": round(float(threshold), 6),
        "n": len(frame),
        "accuracy": round(float(accuracy_score(y_true, pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 6),
        "precision_correct_class": round(float(precision_score(y_true, pred, zero_division=0)), 6),
        "recall_correct_class": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 6),
        "auc": round(float(roc_auc_score(y_true, scores)), 6),
        "error_precision": round(float(ep), 6),
        "error_recall": round(float(er), 6),
        "error_f1": round(float(ef), 6),
        "tn_error_predicted_error": int(tn),
        "fp_correct_predicted_error": int(fp),
        "fn_error_predicted_correct": int(fn),
        "tp_correct_predicted_correct": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/speechocean/phones_aligned.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/segment_acoustic_v1/segment_features.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/segment_acoustic_v1")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/phase1_segment_acoustic")
    parser.add_argument("--alignment-quality", default="pass")
    parser.add_argument("--objective", choices=["balanced_accuracy", "macro_f1"], default="balanced_accuracy")
    parser.add_argument("--min-error-precision", type=float, default=0.4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    df = load_or_extract(args.input, args.features, args.alignment_quality)
    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    numeric = BASE_NUM_FEATURES + [c for c in df.columns if c.startswith("seg_mean_") or c.startswith("seg_std_")]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    train = df[df["split"] == "train"].copy()
    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()

    all_metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name in ["logreg", "rf", "extra"]:
        pipe = build_pipeline(model_name, numeric)
        pipe.fit(train[numeric + CAT_FEATURES], train["gold_binary"])
        joblib.dump(pipe, args.output_dir / f"{model_name}.joblib")
        classes = list(pipe.named_steps["model"].classes_)
        correct_index = classes.index(1)
        for split_name, split_frame in [("dev", dev), ("test", test)]:
            split_frame = split_frame.copy()
            split_frame["prob_correct"] = pipe.predict_proba(split_frame[numeric + CAT_FEATURES])[:, correct_index]
            split_frame["model"] = model_name
            prediction_frames.append(split_frame)
        model_dev = prediction_frames[-2]
        threshold = best_threshold(model_dev["gold_binary"].to_numpy(), model_dev["prob_correct"].to_numpy(), args.objective)
        constrained = precision_constrained_threshold(
            model_dev["gold_binary"].to_numpy(),
            model_dev["prob_correct"].to_numpy(),
            args.min_error_precision,
        )
        for split_frame in prediction_frames[-2:]:
            all_metrics.append(metrics(split_frame, threshold, model_name, args.objective))
            if constrained is not None:
                all_metrics.append(metrics(split_frame, constrained, model_name, f"error_precision_{args.min_error_precision}"))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.report_dir / "segment_acoustic_metrics.csv", index=False, encoding="utf-8-sig")
    test_metrics = metrics_df[metrics_df["split"] == "test"].sort_values(["balanced_accuracy", "macro_f1", "auc"], ascending=False)
    test_metrics.to_csv(args.report_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    best = test_metrics.iloc[0].to_dict()
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.report_dir / "segment_acoustic_predictions.csv", index=False, encoding="utf-8-sig")
    summary = [
        "# Segment Acoustic Classifier",
        "",
        f"Rows: {len(df)}",
        "",
        "## Best Test Result",
        "",
        f"- Model: `{best['model']}`",
        f"- Calibration: `{best['calibration']}`",
        f"- Balanced Accuracy: {best['balanced_accuracy']}",
        f"- Macro-F1: {best['macro_f1']}",
        f"- AUC: {best['auc']}",
        f"- Error Precision: {best['error_precision']}",
        f"- Error Recall: {best['error_recall']}",
    ]
    (args.report_dir / "segment_acoustic_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
