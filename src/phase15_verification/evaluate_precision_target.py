from __future__ import annotations

import json
from pathlib import Path

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

from .labels import infer_error_labels


def evaluate_frame(
    frame: pd.DataFrame,
    label_col: str,
    score_col: str,
    decision_col: str,
    error_value: str = "auto",
    review_as_error: bool = False,
    core_phones: list[str] | None = None,
) -> dict:
    y_true = infer_error_labels(frame, label_col, error_value)
    scores = pd.to_numeric(frame[score_col], errors="coerce").fillna(0.0).to_numpy()
    decisions = frame[decision_col].astype(str)
    positive = ["high_confidence_error", "error", "likely_error"]
    if review_as_error:
        positive += ["uncertain_review"]
    y_pred = decisions.isin(positive).to_numpy(dtype=int)
    return _metrics(y_true, y_pred, scores) | {
        "rows": int(len(frame)),
        "positive_prediction_policy": "high_confidence_error_plus_uncertain_review" if review_as_error else "high_confidence_error_only",
        "max_recall_at_precision_0_40": _best_recall_at_precision(y_true, scores, 0.40),
        "max_recall_at_precision_0_50": _best_recall_at_precision(y_true, scores, 0.50),
        "per_phone": _group_metrics(frame, y_true, y_pred, "target_phone"),
        "per_phone_group": _group_metrics(frame, y_true, y_pred, "phone_group"),
        "core_phone_set": _core_metrics(frame, y_true, y_pred, core_phones or []),
        "alignment_quality": _alignment_metrics(frame, y_true, y_pred),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "auc": round(float(auc), 6),
        "error_precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "error_recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": {
            "tn_non_error_predicted_non_error": int(tn),
            "fp_non_error_predicted_error": int(fp),
            "fn_error_predicted_non_error": int(fn),
            "tp_error_predicted_error": int(tp),
        },
    }


def _best_recall_at_precision(y_true: np.ndarray, scores: np.ndarray, min_precision: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return {"found": False, "precision": 0.0, "recall": 0.0, "threshold": None}
    index = valid[np.argmax(recall[:-1][valid])]
    return {
        "found": True,
        "precision": round(float(precision[index]), 6),
        "recall": round(float(recall[index]), 6),
        "threshold": round(float(thresholds[index]), 6),
    }


def _group_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, col: str) -> list[dict]:
    if col not in frame.columns:
        return []
    rows = []
    for key, idx in frame.groupby(col).groups.items():
        idx_list = list(idx)
        if len(idx_list) == 0:
            continue
        m = _metrics(y_true[idx_list], y_pred[idx_list], y_pred[idx_list])
        rows.append({"key": str(key), "n": len(idx_list), "error_rate": round(float(y_true[idx_list].mean()), 6), **m})
    return sorted(rows, key=lambda r: r["n"], reverse=True)


def _core_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, core_phones: list[str]) -> dict:
    if "target_phone" not in frame.columns or not core_phones:
        return {}
    mask = frame["target_phone"].astype(str).str.upper().isin([p.upper() for p in core_phones]).to_numpy()
    if not mask.any():
        return {"n": 0}
    return {"n": int(mask.sum()), **_metrics(y_true[mask], y_pred[mask], y_pred[mask])}


def _alignment_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if "alignment_quality" not in frame.columns:
        return {}
    out = {}
    for key, idx in frame.groupby("alignment_quality").groups.items():
        idx_list = list(idx)
        out[str(key)] = {"n": len(idx_list), **_metrics(y_true[idx_list], y_pred[idx_list], y_pred[idx_list])}
    return out


def write_evaluation(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
