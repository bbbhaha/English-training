#!/usr/bin/env python
"""Use manual review labels to calibrate Phase-1.5 high-confidence decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase15_verification.analysis import add_evidence_columns
from phase15_verification.config import load_config
from phase15_verification.labels import infer_error_labels


ERROR_LABELS = {"true_error", "acceptable_accent"}
NON_ERROR_LABELS = {"correct", "bad_alignment"}
NUMERIC_FEATURES = [
    "final_error_score",
    "main_model_error_score",
    "attribute_risk_score",
    "attribute_mismatch_count",
    "proto_same_phone_sim_top1",
    "proto_same_phone_sim_topk_mean",
    "proto_confusion_phone_sim_top1",
    "proto_margin",
    "oneclass_anomaly_score",
    "verification_support_votes",
    "evidence_count",
    "verifier_evidence_count",
    "duration_ms",
    "prob_correct",
    "confidence",
]
CATEGORICAL_FEATURES = [
    "target_phone",
    "phone_group",
    "attribute_verifier_decision",
    "retrieval_verifier_decision",
    "oneclass_verifier_decision",
    "evidence_pattern",
    "audit_reason",
]
KEY_COLUMNS = ["utterance_id", "speaker_id", "word", "target_phone", "start_ms", "end_ms"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small manual-label calibration layer for Phase-1.5.")
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/phase15_verification/test_verified_predictions.csv")
    parser.add_argument("--manual-review", type=Path, default=ROOT / "outputs/phase15_verification/audit/manual_review_packet.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase15/aggregator.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase15_verification/manual_calibration")
    parser.add_argument("--target-precision", type=float, default=0.40)
    parser.add_argument("--acceptable-threshold", type=float, default=0.20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred = pd.read_csv(args.predictions, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    pred = add_evidence_columns(pred, cfg)
    manual = pd.read_csv(args.manual_review, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    labeled = _merge_manual_labels(pred, manual)
    if labeled.empty:
        raise SystemExit("No labeled manual review rows could be merged into predictions.")
    usable = labeled[labeled["manual_review_label"].isin(ERROR_LABELS | NON_ERROR_LABELS)].copy()
    if usable["manual_review_label"].nunique() < 2:
        raise SystemExit("Manual review labels need at least one true_error and one non-error label.")
    usable["manual_is_true_error"] = usable["manual_review_label"].isin(ERROR_LABELS).astype(int)

    numeric = [c for c in NUMERIC_FEATURES if c in usable.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in usable.columns]
    model = _build_model(numeric, categorical)
    oof_scores = _cross_val_scores(model, usable, numeric, categorical, usable["manual_is_true_error"].to_numpy())
    threshold_info = _select_threshold(usable["manual_is_true_error"].to_numpy(), oof_scores, args.target_precision)
    model.fit(usable[numeric + categorical], usable["manual_is_true_error"])
    train_scores = model.predict_proba(usable[numeric + categorical])[:, list(model.named_steps["classifier"].classes_).index(1)]

    all_scores = model.predict_proba(pred[numeric + categorical])[:, list(model.named_steps["classifier"].classes_).index(1)]
    calibrated = pred.copy()
    calibrated["manual_calibrated_error_probability"] = np.round(all_scores, 6)
    threshold = float(threshold_info["threshold"] if threshold_info["threshold"] is not None else 1.01)
    calibrated["manual_calibrated_threshold"] = threshold
    original_hce = calibrated["final_decision"].astype(str) == "high_confidence_error"
    keep_hce = original_hce & (calibrated["manual_calibrated_error_probability"] >= threshold)
    downgraded_to_acceptable = original_hce & ~keep_hce & (calibrated["manual_calibrated_error_probability"] < args.acceptable_threshold)
    calibrated["manual_calibrated_decision"] = calibrated["final_decision"]
    calibrated.loc[original_hce & ~keep_hce, "manual_calibrated_decision"] = "uncertain_review"
    calibrated.loc[downgraded_to_acceptable, "manual_calibrated_decision"] = "acceptable_accent"
    calibrated.loc[keep_hce, "manual_calibrated_decision"] = "high_confidence_error"
    calibrated["manual_calibration_action"] = np.select(
        [keep_hce, original_hce & ~keep_hce, ~original_hce],
        ["keep_high_confidence_error", "downgrade_high_confidence_error", "unchanged_non_hce"],
        default="unchanged",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "manual_calibrated_verifier.joblib"
    joblib.dump(
        {
            "model": model,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "threshold": threshold,
            "target_precision": args.target_precision,
            "error_labels": sorted(ERROR_LABELS),
            "non_error_labels": sorted(NON_ERROR_LABELS),
            "label_definition": "true_error/acceptable_accent=positive pronunciation error under correction-focused policy; correct/bad_alignment=non_error",
        },
        model_path,
    )
    calibrated.to_csv(args.output_dir / "manual_calibrated_predictions.csv", index=False, encoding="utf-8-sig")
    oof = usable[KEY_COLUMNS + ["manual_review_label", "manual_review_notes", "final_decision", "final_error_score"]].copy()
    oof["manual_is_true_error"] = usable["manual_is_true_error"]
    oof["manual_calibrated_oof_probability"] = np.round(oof_scores, 6)
    oof["manual_calibrated_oof_prediction"] = (oof_scores >= threshold).astype(int)
    oof.to_csv(args.output_dir / "manual_calibration_oof_predictions.csv", index=False, encoding="utf-8-sig")
    _write_pr_curve(usable["manual_is_true_error"].to_numpy(), oof_scores, args.output_dir / "manual_calibration_oof_pr_curve.csv")

    report = {
        "model_path": str(model_path),
        "manual_review_rows": int(len(manual)),
        "merged_labeled_rows": int(len(usable)),
        "manual_label_counts": {str(k): int(v) for k, v in usable["manual_review_label"].value_counts().to_dict().items()},
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target_precision": args.target_precision,
        "selected_threshold": threshold_info,
        "manual_train_metrics_apparent": _metrics(
            usable["manual_is_true_error"].to_numpy(),
            (train_scores >= threshold).astype(int),
            train_scores,
        ),
        "manual_oof_metrics": _metrics(usable["manual_is_true_error"].to_numpy(), oof["manual_calibrated_oof_prediction"].to_numpy(), oof_scores),
        "gold_binary_metrics_after_manual_calibration": _gold_metrics(calibrated),
        "decision_counts_after_manual_calibration": {str(k): int(v) for k, v in calibrated["manual_calibrated_decision"].value_counts().to_dict().items()},
        "caution": "OOF metrics are based on the 200 manually reviewed high-confidence rows. Treat them as calibration evidence, not final held-out proof.",
    }
    (args.output_dir / "manual_calibration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([_flat_summary(report)]).to_csv(args.output_dir / "manual_calibration_summary.csv", index=False, encoding="utf-8-sig")
    _write_model_card(args.output_dir / "manual_calibrated_model_card.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _merge_manual_labels(pred: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    left = pred.copy()
    right = manual.copy()
    for frame in [left, right]:
        for col in KEY_COLUMNS:
            if col not in frame.columns:
                frame[col] = ""
        frame["_merge_key"] = _merge_key(frame)
    keep = ["_merge_key", "manual_review_label", "manual_review_notes"]
    merged = left.merge(right[keep], on="_merge_key", how="inner")
    merged["manual_review_label"] = merged["manual_review_label"].astype(str).str.strip()
    return merged[merged["manual_review_label"] != ""].copy()


def _merge_key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for col in KEY_COLUMNS:
        if col in {"start_ms", "end_ms"}:
            value = pd.to_numeric(frame[col], errors="coerce").map(_format_time_key)
        else:
            value = frame[col].astype(str).str.strip()
        parts.append(value)
    key = parts[0]
    for part in parts[1:]:
        key = key + "||" + part
    return key


def _format_time_key(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _build_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)),
        ]
    )


def _cross_val_scores(model: Pipeline, frame: pd.DataFrame, numeric: list[str], categorical: list[str], y: np.ndarray) -> np.ndarray:
    min_class = int(pd.Series(y).value_counts().min())
    if min_class < 2:
        model.fit(frame[numeric + categorical], y)
        return model.predict_proba(frame[numeric + categorical])[:, list(model.named_steps["classifier"].classes_).index(1)]
    n_splits = min(5, min_class)
    scores = np.zeros(len(frame), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(frame, y):
        fold_model = _build_model(numeric, categorical)
        fold_model.fit(frame.iloc[train_idx][numeric + categorical], y[train_idx])
        scores[test_idx] = fold_model.predict_proba(frame.iloc[test_idx][numeric + categorical])[:, list(fold_model.named_steps["classifier"].classes_).index(1)]
    return scores


def _select_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= target_precision)[0]
    if len(valid):
        idx = valid[np.argmax(recall[:-1][valid])]
        return {
            "found_target_precision": True,
            "precision": round(float(precision[idx]), 6),
            "recall": round(float(recall[idx]), 6),
            "threshold": round(float(thresholds[idx]), 6),
        }
    best_idx = int(np.argmax(precision[:-1])) if len(thresholds) else 0
    return {
        "found_target_precision": False,
        "precision": round(float(precision[:-1][best_idx]), 6) if len(thresholds) else 0.0,
        "recall": round(float(recall[:-1][best_idx]), 6) if len(thresholds) else 0.0,
        "threshold": round(float(thresholds[best_idx]), 6) if len(thresholds) else None,
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "auc": round(float(auc), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _write_pr_curve(y_true: np.ndarray, scores: np.ndarray, path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    frame = pd.DataFrame(
        {
            "precision": precision[:-1],
            "recall": recall[:-1],
            "threshold": thresholds,
        }
    )
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_model_card(path: Path, report: dict) -> None:
    lines = [
        "# Manual-Calibrated Phase-1.5 Verifier",
        "",
        "Purpose: calibrate Phase-1.5 `high_confidence_error` decisions using manual review labels.",
        "",
        "Label definition:",
        "",
        "- Positive: `true_error`",
        "- Non-error: `correct`, `bad_alignment`",
        "",
        f"Model artifact: `{report['model_path']}`",
        f"Training rows: {report['merged_labeled_rows']}",
        f"Target precision: {report['target_precision']}",
        "",
        "## Selected Threshold",
        "",
    ]
    for key, value in report["selected_threshold"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Out-of-Fold Manual Metrics", ""])
    for key, value in report["manual_oof_metrics"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Public Gold Binary Metrics After Calibration", ""])
    for key, value in report["gold_binary_metrics_after_manual_calibration"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Caution: out-of-fold metrics are based on the 200 manually reviewed high-confidence rows.",
            "The public `gold_binary` metric uses a different label granularity and should not be treated as the final manual-label objective.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gold_metrics(frame: pd.DataFrame) -> dict:
    if "gold_binary" not in frame.columns:
        return {}
    y_true = infer_error_labels(frame, "gold_binary")
    y_pred = (frame["manual_calibrated_decision"].astype(str) == "high_confidence_error").to_numpy(dtype=int)
    scores = pd.to_numeric(frame["manual_calibrated_error_probability"], errors="coerce").fillna(0.0).to_numpy()
    return _metrics(y_true, y_pred, scores)


def _flat_summary(report: dict) -> dict:
    row = {
        "merged_labeled_rows": report["merged_labeled_rows"],
        "target_precision": report["target_precision"],
        "selected_threshold": report["selected_threshold"]["threshold"],
        "oof_found_target_precision": report["selected_threshold"]["found_target_precision"],
    }
    for prefix, metrics in [("manual_oof", report["manual_oof_metrics"]), ("gold_after", report["gold_binary_metrics_after_manual_calibration"])]:
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    for key, value in report["manual_train_metrics_apparent"].items():
        row[f"manual_train_{key}"] = value
    return row


if __name__ == "__main__":
    main()
