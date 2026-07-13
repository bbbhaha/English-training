#!/usr/bin/env python
"""Sweep Phase-1.5 strict-consensus aggregator thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase15_verification.analysis import CORE_PHONE_DEFAULTS, add_evidence_columns, dump_json
from phase15_verification.config import load_config
from phase15_verification.labels import infer_error_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep strict-consensus thresholds for Phase-1.5 aggregation.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/phase15_verification/test_verified_predictions.csv")
    parser.add_argument("--test-input", type=Path, help="Optional held-out verified prediction table to evaluate the best setting.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase15/aggregator.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase15_verification/calibration")
    parser.add_argument("--label-col", default="gold_binary")
    parser.add_argument("--error-value", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frame = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    frame = add_evidence_columns(frame, cfg)
    rows = []
    for params in _param_grid():
        scored = _apply_params(frame, cfg, params)
        metrics = _score_setting(scored, args.label_col, args.error_value, cfg)
        rows.append({**params, **metrics})
    results = pd.DataFrame(rows).sort_values(
        ["max_recall_at_precision_0_40", "precision", "recall", "macro_f1", "core_false_positive_count"],
        ascending=[False, False, False, False, True],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "aggregator_sweep_results.csv", index=False, encoding="utf-8-sig")
    best = results.iloc[0].to_dict()
    best_config = _best_config(cfg, best)
    _write_simple_yaml(args.output_dir / "best_aggregator_config.yaml", best_config)
    summary = {
        "selection_metric": "max_recall_at_precision_0_40",
        "input": str(args.input),
        "best": _jsonable(best),
    }
    if args.test_input:
        test = pd.read_csv(args.test_input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
        test = add_evidence_columns(test, cfg)
        scored_test = _apply_params(test, cfg, best)
        summary["optional_test_metrics"] = _score_setting(scored_test, args.label_col, args.error_value, cfg)
    dump_json(args.output_dir / "best_threshold_summary.json", summary)
    print(f"Sweep rows: {len(results)}")
    print(f"Saved sweep results: {args.output_dir / 'aggregator_sweep_results.csv'}")
    print(json.dumps(summary["best"], indent=2))


def _param_grid() -> list[dict]:
    score_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    evidence_counts = [1, 2, 3]
    attr_thresholds = [0.45, 0.55, 0.65]
    margin_thresholds = [0.0, 0.05, 0.10]
    anomaly_thresholds = [0.55, 0.65]
    return [
        {
            "final_error_score_threshold": score,
            "min_evidence_count": evidence,
            "attribute_risk_score_threshold": attr,
            "proto_margin_threshold": margin,
            "oneclass_anomaly_score_threshold": anomaly,
        }
        for score in score_thresholds
        for evidence in evidence_counts
        for attr in attr_thresholds
        for margin in margin_thresholds
        for anomaly in anomaly_thresholds
    ]


def _apply_params(frame: pd.DataFrame, cfg: dict, params: dict) -> pd.DataFrame:
    out = frame.copy()
    attr_decision = out["attribute_verifier_decision"] if "attribute_verifier_decision" in out.columns else pd.Series("", index=out.index)
    retrieval_decision = out["retrieval_verifier_decision"] if "retrieval_verifier_decision" in out.columns else pd.Series("", index=out.index)
    oneclass_decision = out["oneclass_verifier_decision"] if "oneclass_verifier_decision" in out.columns else pd.Series("", index=out.index)
    out["attribute_error_flag"] = (
        (attr_decision.astype(str) == "high_confidence_error")
        | (_numeric(out, "attribute_risk_score", 0.0) >= float(params["attribute_risk_score_threshold"]))
    ).astype(int)
    out["retrieval_error_flag"] = (
        (retrieval_decision.astype(str) == "likely_error")
        | (_numeric(out, "proto_margin", 0.0) < float(params["proto_margin_threshold"]))
    ).astype(int)
    out["oneclass_error_flag"] = (
        (oneclass_decision.astype(str) == "high_confidence_error")
        | (_numeric(out, "oneclass_anomaly_score", 0.0) >= float(params["oneclass_anomaly_score_threshold"]))
    ).astype(int)
    out["verifier_evidence_count"] = out[["attribute_error_flag", "retrieval_error_flag", "oneclass_error_flag", "duration_outlier_flag"]].sum(axis=1)
    score = _numeric(out, "final_error_score", 0.0)
    main = out["main_model_error_flag"].astype(int) == 1
    evidence = out["verifier_evidence_count"].astype(int) >= int(params["min_evidence_count"])
    alignment_ok = out.get("alignment_review_flag", pd.Series(0, index=out.index)).astype(int) == 0
    score_ok = score >= float(params["final_error_score_threshold"])
    out["_sweep_adjusted_score"] = np.where(main & evidence & alignment_ok, score, 0.0)
    out["_sweep_decision"] = np.where(main & evidence & alignment_ok & score_ok, "high_confidence_error", "correct")
    out["_sweep_pred"] = (out["_sweep_decision"] == "high_confidence_error").astype(int)
    return out


def _score_setting(frame: pd.DataFrame, label_col: str, error_value: str, cfg: dict) -> dict:
    y_true = infer_error_labels(frame, label_col, error_value)
    y_pred = frame["_sweep_pred"].to_numpy(dtype=int)
    scores = pd.to_numeric(frame["_sweep_adjusted_score"], errors="coerce").fillna(0.0).to_numpy()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    core_phones = set(str(p).upper() for p in cfg.get("evaluation", {}).get("core_phones", CORE_PHONE_DEFAULTS))
    target_phone = frame["target_phone"] if "target_phone" in frame.columns else pd.Series("", index=frame.index)
    core_mask = target_phone.astype(str).str.upper().isin(core_phones).to_numpy()
    core_fp = int(((y_true == 0) & (y_pred == 1) & core_mask).sum())
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "auc": round(float(auc), 6),
        "max_recall_at_precision_0_40": _max_recall_at_precision(y_true, scores, 0.40),
        "max_recall_at_precision_0_50": _max_recall_at_precision(y_true, scores, 0.50),
        "predicted_error_count": int(y_pred.sum()),
        "false_positive_count": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_positive_count": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_negative_count": int(((y_true == 1) & (y_pred == 0)).sum()),
        "core_false_positive_count": core_fp,
    }


def _max_recall_at_precision(y_true: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return 0.0
    return round(float(recall[:-1][valid].max()), 6)


def _numeric(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index)


def _best_config(base: dict, best: dict) -> dict:
    cfg = dict(base)
    cfg["aggregator"] = {
        **cfg.get("aggregator", {}),
        "aggregator_mode": "strict_consensus",
        "final_error_score_threshold": float(best["final_error_score_threshold"]),
    }
    cfg["attribute"] = {**cfg.get("attribute", {}), "high_risk_threshold": float(best["attribute_risk_score_threshold"])}
    cfg["retrieval"] = {**cfg.get("retrieval", {}), "negative_margin_threshold": float(best["proto_margin_threshold"])}
    cfg["oneclass"] = {**cfg.get("oneclass", {}), "anomaly_threshold": float(best["oneclass_anomaly_score_threshold"])}
    cfg["strict_consensus"] = {
        **cfg.get("strict_consensus", {}),
        "min_evidence_count": int(best["min_evidence_count"]),
        "allow_review_alignment": False,
        "require_main_model_error": True,
    }
    return cfg


def _write_simple_yaml(path: Path, data: dict, indent: int = 0) -> None:
    lines = _yaml_lines(data, indent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_lines(data, indent: int = 0) -> list[str]:
    prefix = " " * indent
    lines = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_scalar(value)}")
    elif isinstance(data, list):
        for value in data:
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(value, indent + 2))
            else:
                lines.append(f"{prefix}- {_scalar(value)}")
    return lines


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _jsonable(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
