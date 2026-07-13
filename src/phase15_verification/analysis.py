from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .labels import infer_error_labels, main_error_decision, main_error_score


CORE_PHONE_DEFAULTS = ["R", "L", "V", "W", "TH", "DH", "N", "NG", "IY", "IH", "EH", "AE", "UW", "UH"]


def normalize_group_key(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "liquids": "liquid",
        "fricatives": "fricative",
        "vowels": "vowel",
        "stops": "stop",
        "nasals": "nasal",
        "affricates": "affricate",
        "glides": "glide",
        "final_consonants": "final_consonant",
    }
    return aliases.get(text, text)


def add_evidence_columns(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or {}
    attr_cfg = cfg.get("attribute", {})
    retrieval_cfg = cfg.get("retrieval", {})
    oneclass_cfg = cfg.get("oneclass", {})
    out = frame.copy()
    out["main_model_error_score"] = main_error_score(out).round(6)
    out["main_model_error_flag"] = main_error_decision(out).astype(int)
    attr_threshold = float(attr_cfg.get("high_risk_threshold", 0.55))
    attr_decision = _series(out, "attribute_verifier_decision", "")
    retrieval_decision = _series(out, "retrieval_verifier_decision", "")
    oneclass_decision = _series(out, "oneclass_verifier_decision", "")
    out["attribute_error_flag"] = (
        (attr_decision.astype(str) == "high_confidence_error")
        | (pd.to_numeric(_series(out, "attribute_risk_score", 0.0), errors="coerce").fillna(0.0) >= attr_threshold)
    ).astype(int)
    margin_threshold = float(retrieval_cfg.get("negative_margin_threshold", 0.0))
    out["retrieval_error_flag"] = (
        (retrieval_decision.astype(str) == "likely_error")
        | (pd.to_numeric(_series(out, "proto_margin", 0.0), errors="coerce").fillna(0.0) < margin_threshold)
    ).astype(int)
    anomaly_threshold = float(oneclass_cfg.get("anomaly_threshold", 0.55))
    out["oneclass_error_flag"] = (
        (oneclass_decision.astype(str) == "high_confidence_error")
        | (pd.to_numeric(_series(out, "oneclass_anomaly_score", 0.0), errors="coerce").fillna(0.0) >= anomaly_threshold)
    ).astype(int)
    out["duration_outlier_flag"] = _duration_outlier_flag(out, float(cfg.get("analysis", {}).get("duration_z_threshold", 2.5))).astype(int)
    if "alignment_quality" in out.columns:
        out["alignment_review_flag"] = (out["alignment_quality"].astype(str).str.lower() != "pass").astype(int)
    else:
        out["alignment_review_flag"] = 0
    verifier_cols = ["attribute_error_flag", "retrieval_error_flag", "oneclass_error_flag", "duration_outlier_flag"]
    all_cols = ["main_model_error_flag"] + verifier_cols
    out["verifier_evidence_count"] = out[verifier_cols].sum(axis=1).astype(int)
    out["evidence_count"] = out[all_cols].sum(axis=1).astype(int)
    out["evidence_pattern"] = [
        _pattern(row)
        for _, row in out[all_cols + ["alignment_review_flag"]].iterrows()
    ]
    out["audit_reason"] = [audit_reason(row) for _, row in out.iterrows()]
    return out


def audit_reason(row: pd.Series) -> str:
    final_decision = str(row.get("final_decision", ""))
    main_score = float(row.get("main_model_error_score", 0.0) or 0.0)
    if final_decision == "high_confidence_error":
        parts = []
        if main_score >= 0.50 or int(row.get("main_model_error_flag", 0)) == 1:
            parts.append("main_model_high_prob")
        if int(row.get("attribute_error_flag", 0)) == 1:
            parts.append("attribute_high_risk")
        if int(row.get("retrieval_error_flag", 0)) == 1:
            parts.append("retrieval_negative_margin")
        if int(row.get("oneclass_error_flag", 0)) == 1:
            parts.append("oneclass_anomaly")
        if int(row.get("duration_outlier_flag", 0)) == 1:
            parts.append("duration_outlier")
        verifier_count = int(row.get("verifier_evidence_count", 0))
        if verifier_count == 0:
            return "main_model_only_no_verifier_support"
        if main_score < 0.35 and verifier_count < 2:
            return "weak_evidence_but_marked_error"
        return " + ".join(parts) if parts else "weak_evidence_but_marked_error"
    if final_decision == "uncertain_review":
        return "weak_or_conflicting_verifier_evidence"
    if final_decision == "acceptable_accent":
        return "non_error_with_moderate_error_score"
    return "likely_correct"


def _pattern(row: pd.Series) -> str:
    names = []
    if int(row.get("main_model_error_flag", 0)) == 1:
        names.append("main")
    if int(row.get("attribute_error_flag", 0)) == 1:
        names.append("attribute")
    if int(row.get("retrieval_error_flag", 0)) == 1:
        names.append("retrieval")
    if int(row.get("oneclass_error_flag", 0)) == 1:
        names.append("oneclass")
    if int(row.get("duration_outlier_flag", 0)) == 1:
        names.append("duration")
    if int(row.get("alignment_review_flag", 0)) == 1:
        names.append("alignment_review")
    if names == ["main"]:
        return "main_only"
    return "+".join(names) if names else "none"


def _duration_outlier_flag(frame: pd.DataFrame, z_threshold: float) -> pd.Series:
    duration_col = "duration_ms" if "duration_ms" in frame.columns else "duration" if "duration" in frame.columns else None
    if duration_col is None:
        return pd.Series(False, index=frame.index)
    durations = pd.to_numeric(frame[duration_col], errors="coerce").fillna(0.0)
    if "target_phone" in frame.columns:
        means = durations.groupby(frame["target_phone"].astype(str)).transform("mean")
        stds = durations.groupby(frame["target_phone"].astype(str)).transform("std").replace(0, np.nan)
    else:
        means = pd.Series(durations.mean(), index=frame.index)
        stds = pd.Series(durations.std() or 1.0, index=frame.index)
    z = ((durations - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
    return z >= z_threshold


def _series(frame: pd.DataFrame, col: str, default: object) -> pd.Series:
    if col in frame.columns:
        return frame[col]
    return pd.Series(default, index=frame.index)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    scores_arr = scores if scores is not None else y_pred
    try:
        auc = roc_auc_score(y_true, scores_arr)
    except ValueError:
        auc = 0.5
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "auc": round(float(auc), 6),
        "false_positive_count": int(fp),
        "true_positive_count": int(tp),
        "false_negative_count": int(fn),
        "true_negative_count": int(tn),
    }


def grouped_metric_table(
    frame: pd.DataFrame,
    label_col: str,
    group_col: str,
    decision_col: str = "final_decision",
    score_col: str = "final_error_score",
    error_value: str = "auto",
) -> pd.DataFrame:
    if group_col not in frame.columns:
        return pd.DataFrame()
    frame = frame.reset_index(drop=True)
    y_true_all = infer_error_labels(frame, label_col, error_value)
    y_pred_all = (frame[decision_col].astype(str) == "high_confidence_error").to_numpy(dtype=int)
    scores_all = pd.to_numeric(_series(frame, score_col, 0.0), errors="coerce").fillna(0.0).to_numpy()
    rows = []
    for key, idx in frame.groupby(group_col).groups.items():
        indices = np.array(list(idx), dtype=int)
        y_true = y_true_all[indices]
        y_pred = y_pred_all[indices]
        scores = scores_all[indices]
        metrics = binary_metrics(y_true, y_pred, scores)
        subset = frame.iloc[indices]
        rows.append(
            {
                group_col: str(key),
                "support": int(len(indices)),
                "error_support": int(y_true.sum()),
                **metrics,
                "threshold_used": _threshold_summary(subset),
                "mean_final_error_score": _mean_col(subset, score_col),
                "mean_attribute_risk_score": _mean_col(subset, "attribute_risk_score"),
                "mean_proto_margin": _mean_col(subset, "proto_margin"),
                "mean_oneclass_anomaly_score": _mean_col(subset, "oneclass_anomaly_score"),
            }
        )
    out = pd.DataFrame(rows)
    if "false_positive_count" in out.columns:
        out = out.sort_values(["false_positive_count", "support"], ascending=[False, False])
    return out


def core_phone_metric_table(
    frame: pd.DataFrame,
    label_col: str,
    core_phones: list[str] | None = None,
    decision_col: str = "final_decision",
    score_col: str = "final_error_score",
    error_value: str = "auto",
) -> pd.DataFrame:
    if "target_phone" not in frame.columns:
        return pd.DataFrame()
    phones = [p.upper() for p in (core_phones or CORE_PHONE_DEFAULTS)]
    subset = frame[frame["target_phone"].astype(str).str.upper().isin(phones)].copy()
    if subset.empty:
        return pd.DataFrame()
    return grouped_metric_table(subset, label_col, "target_phone", decision_col, score_col, error_value)


def evidence_pattern_metrics(frame: pd.DataFrame, label_col: str, error_value: str = "auto") -> pd.DataFrame:
    if "evidence_pattern" not in frame.columns:
        frame = add_evidence_columns(frame)
    y_true = infer_error_labels(frame, label_col, error_value)
    pred = (frame["final_decision"].astype(str) == "high_confidence_error").to_numpy(dtype=int)
    total_errors = max(int(y_true.sum()), 1)
    rows = []
    for pattern, idx in frame.groupby("evidence_pattern").groups.items():
        indices = np.array(list(idx), dtype=int)
        tp = int(((y_true[indices] == 1) & (pred[indices] == 1)).sum())
        fp = int(((y_true[indices] == 0) & (pred[indices] == 1)).sum())
        predicted = int(pred[indices].sum())
        rows.append(
            {
                "evidence_pattern": str(pattern),
                "sample_count": int(len(indices)),
                "predicted_error_count": predicted,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "precision": round(float(tp / predicted), 6) if predicted else 0.0,
                "recall_contribution": round(float(tp / total_errors), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(["false_positive_count", "sample_count"], ascending=[False, False])


def alignment_quality_metrics(frame: pd.DataFrame, label_col: str, error_value: str = "auto") -> pd.DataFrame:
    if "alignment_quality" not in frame.columns:
        return pd.DataFrame()
    return grouped_metric_table(frame, label_col, "alignment_quality", error_value=error_value)


def write_analysis_tables(
    frame: pd.DataFrame,
    output_dir: Path,
    label_col: str = "gold_binary",
    error_value: str = "auto",
    core_phones: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_metric_table(frame, label_col, "target_phone", error_value=error_value).to_csv(
        output_dir / "per_phone_metrics.csv", index=False, encoding="utf-8-sig"
    )
    grouped_metric_table(frame, label_col, "phone_group", error_value=error_value).to_csv(
        output_dir / "per_phone_group_metrics.csv", index=False, encoding="utf-8-sig"
    )
    core_phone_metric_table(frame, label_col, core_phones, error_value=error_value).to_csv(
        output_dir / "core_phone_metrics.csv", index=False, encoding="utf-8-sig"
    )
    evidence_pattern_metrics(frame, label_col, error_value).to_csv(
        output_dir / "evidence_pattern_metrics.csv", index=False, encoding="utf-8-sig"
    )
    alignment_quality_metrics(frame, label_col, error_value).to_csv(
        output_dir / "alignment_quality_metrics.csv", index=False, encoding="utf-8-sig"
    )


def _threshold_summary(frame: pd.DataFrame) -> str:
    if "final_error_score_threshold" in frame.columns:
        values = pd.to_numeric(frame["final_error_score_threshold"], errors="coerce").dropna().unique()
        return ";".join(str(round(float(v), 6)) for v in sorted(values)[:5])
    if "threshold" in frame.columns:
        values = pd.to_numeric(frame["threshold"], errors="coerce").dropna().unique()
        return ";".join(str(round(float(v), 6)) for v in sorted(values)[:5])
    return ""


def _mean_col(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns:
        return 0.0
    return round(float(pd.to_numeric(frame[col], errors="coerce").fillna(0.0).mean()), 6)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
