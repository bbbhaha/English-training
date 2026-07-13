#!/usr/bin/env python
"""Evaluate Phase-1.5 high-precision decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase15_verification.config import load_config
from phase15_verification.analysis import add_evidence_columns, write_analysis_tables
from phase15_verification.evaluate_precision_target import evaluate_frame, write_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase-1.5 verifier outputs against precision-target metrics.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/phase15_verification/test_verified_predictions.csv")
    parser.add_argument("--label-col", default="gold_binary")
    parser.add_argument("--score-col", default="final_error_score")
    parser.add_argument("--decision-col", default="final_decision")
    parser.add_argument("--error-value", default="auto")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase15/aggregator.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase15_verification/evaluation_report.json")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "outputs/phase15_verification/evaluation_summary.csv")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "outputs/phase15_verification/analysis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frame = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    frame = add_evidence_columns(frame, cfg)
    core_phones = cfg.get("evaluation", {}).get("core_phones", [])
    comparison_specs = _add_comparison_columns(frame)
    comparisons = {}
    rows = []
    for name, spec in comparison_specs.items():
        metrics = evaluate_frame(
            frame,
            label_col=args.label_col,
            score_col=spec["score_col"],
            decision_col=spec["decision_col"],
            error_value=args.error_value,
            review_as_error=False,
            core_phones=core_phones,
        )
        comparisons[name] = metrics
        rows.append(_summary_row(name, metrics))
    high_precision = evaluate_frame(
        frame,
        label_col=args.label_col,
        score_col=args.score_col,
        decision_col=args.decision_col,
        error_value=args.error_value,
        review_as_error=False,
        core_phones=core_phones,
    )
    review_as_error = evaluate_frame(
        frame,
        label_col=args.label_col,
        score_col=args.score_col,
        decision_col=args.decision_col,
        error_value=args.error_value,
        review_as_error=True,
        core_phones=core_phones,
    )
    report = {
        "selection_metric": "max_recall_at_precision_0_40",
        "input": str(args.input),
        "label_col": args.label_col,
        "score_col": args.score_col,
        "decision_col": args.decision_col,
        "comparisons": comparisons,
        "binary_high_confidence_error_only": high_precision,
        "binary_high_confidence_error_plus_uncertain_review_analysis_only": review_as_error,
    }
    write_evaluation(report, args.output)
    rows.append(_summary_row("final_high_confidence_error_only", high_precision))
    rows.append(_summary_row("final_high_confidence_error_plus_uncertain_review", review_as_error))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
    write_analysis_tables(frame, args.analysis_dir, args.label_col, args.error_value, core_phones)
    print(json.dumps({k: report[k] for k in ["selection_metric", "input"]}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Analysis tables saved under: {args.analysis_dir}")


def _summary_row(name: str, metrics: dict) -> dict:
    return {
        "policy": name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "auc": metrics["auc"],
        "error_precision": metrics["error_precision"],
        "error_recall": metrics["error_recall"],
        "max_recall_at_precision_0_40": metrics["max_recall_at_precision_0_40"]["recall"],
        "max_recall_at_precision_0_50": metrics["max_recall_at_precision_0_50"]["recall"],
    }


def _add_comparison_columns(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    specs: dict[str, dict[str, str]] = {}
    if "main_model_error_decision" in frame.columns:
        frame["_cmp_baseline_decision"] = np.where(
            pd.to_numeric(frame["main_model_error_decision"], errors="coerce").fillna(0).astype(int) == 1,
            "high_confidence_error",
            "correct",
        )
        specs["current_best_baseline"] = {
            "decision_col": "_cmp_baseline_decision",
            "score_col": "main_model_error_score" if "main_model_error_score" in frame.columns else "final_error_score",
        }
    if "attribute_verifier_decision" in frame.columns:
        frame["_cmp_attribute_decision"] = np.where(
            frame["attribute_verifier_decision"].astype(str) == "high_confidence_error",
            "high_confidence_error",
            "correct",
        )
        specs["baseline_plus_attribute_verifier"] = {
            "decision_col": "_cmp_attribute_decision",
            "score_col": "attribute_risk_score",
        }
    if "retrieval_verifier_decision" in frame.columns:
        proto_margin = frame["proto_margin"] if "proto_margin" in frame.columns else pd.Series(0.0, index=frame.index)
        frame["_cmp_retrieval_score"] = (0.5 - pd.to_numeric(proto_margin, errors="coerce").fillna(0.0)).clip(0, 1)
        frame["_cmp_retrieval_decision"] = np.where(
            frame["retrieval_verifier_decision"].astype(str) == "likely_error",
            "high_confidence_error",
            "correct",
        )
        specs["baseline_plus_retrieval_verifier"] = {
            "decision_col": "_cmp_retrieval_decision",
            "score_col": "_cmp_retrieval_score",
        }
    if "oneclass_verifier_decision" in frame.columns:
        frame["_cmp_oneclass_decision"] = np.where(
            frame["oneclass_verifier_decision"].astype(str) == "high_confidence_error",
            "high_confidence_error",
            "correct",
        )
        specs["baseline_plus_oneclass_verifier"] = {
            "decision_col": "_cmp_oneclass_decision",
            "score_col": "oneclass_anomaly_score",
        }
    specs["baseline_plus_all_verifiers"] = {
        "decision_col": "final_decision",
        "score_col": "final_error_score",
    }
    return specs


if __name__ == "__main__":
    main()
