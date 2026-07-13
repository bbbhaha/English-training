#!/usr/bin/env python
"""Sweep end-to-end thresholds and write E2E Alpha optimization analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score


KEY_COLUMNS = ["utterance_id", "speaker_id", "word", "target_phone", "start_ms", "end_ms"]
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
PRONUNCIATION_ERROR_LABELS = {"true_error", "acceptable_accent"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep E2E true_error thresholds under manual labels.")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--manual-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e2e_alpha"))
    parser.add_argument("--precision-target", type=float, default=0.40)
    args = parser.parse_args()

    pred = pd.read_csv(args.prediction, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    labels = pd.read_csv(args.manual_labels, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    merged = _merge(pred, labels)
    if merged.empty:
        raise SystemExit("No prediction rows matched manual labels.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep = _sweep(merged, args.precision_target)
    sweep.to_csv(args.output_dir / "threshold_sweep_results.csv", index=False, encoding="utf-8-sig")
    best = _best_row(sweep, args.precision_target)
    (args.output_dir / "best_thresholds.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_pred = _apply_thresholds(merged, best)
    acceptable = _acceptable_accent_analysis(best_pred)
    acceptable.to_csv(args.output_dir / "acceptable_accent_false_positive_analysis.csv", index=False, encoding="utf-8-sig")
    alignment_report, alignment_rows = _alignment_failure_analysis(best_pred)
    (args.output_dir / "alignment_failure_summary.json").write_text(
        json.dumps(alignment_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    alignment_rows.to_csv(args.output_dir / "alignment_failure_samples.csv", index=False, encoding="utf-8-sig")
    bins, calibration = _calibration(best_pred)
    bins.to_csv(args.output_dir / "calibration_bins.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = _markdown_report(best_pred, sweep, best, alignment_report, calibration, args.precision_target)
    (args.output_dir / "metric_optimization_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"matched_rows": len(merged), "best": best}, indent=2, ensure_ascii=False))


def _merge(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    left = pred.copy()
    right = labels.copy()
    left["_key"] = _key(left)
    right["_key"] = _key(right)
    cols = ["_key", "manual_review_label", "manual_review_notes", "reviewer", "review_date"]
    for col in cols:
        if col not in right.columns:
            right[col] = ""
    merged = left.merge(right[cols].drop_duplicates("_key", keep="last"), on="_key", how="inner")
    merged["manual_review_label"] = merged["manual_review_label"].astype(str).str.strip()
    return merged[merged["manual_review_label"] != ""].copy()


def _key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for col in KEY_COLUMNS:
        if col not in frame.columns:
            value = pd.Series("", index=frame.index)
        elif col in {"start_ms", "end_ms"}:
            value = pd.to_numeric(frame[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        else:
            value = frame[col].astype(str).str.strip()
        parts.append(value)
    key = parts[0]
    for part in parts[1:]:
        key = key + "||" + part
    return key


def _sweep(frame: pd.DataFrame, precision_target: float) -> pd.DataFrame:
    rows = []
    groups = sorted(frame.get("phone_group", pd.Series("", index=frame.index)).astype(str).unique())
    for model_t in THRESHOLDS:
        for manual_t in THRESHOLDS:
            for conf_t in [0.0, 0.50, 0.70, 0.90]:
                scored = _apply_thresholds(frame, {
                    "threshold_mode": "global",
                    "model_error_score_threshold": model_t,
                    "manual_calibrated_error_probability_threshold": manual_t,
                    "confidence_threshold": conf_t,
                    "phone_group_thresholds": {},
                })
                rows.append(_metric_row(scored, precision_target))
                per_group = {group: manual_t for group in groups}
                for group in groups:
                    per_group[group] = min(0.99, manual_t + 0.10)
                scored_group = _apply_thresholds(frame, {
                    "threshold_mode": "per_phone_group",
                    "model_error_score_threshold": model_t,
                    "manual_calibrated_error_probability_threshold": manual_t,
                    "confidence_threshold": conf_t,
                    "phone_group_thresholds": per_group,
                })
                rows.append(_metric_row(scored_group, precision_target))
    return pd.DataFrame(rows)


def _apply_thresholds(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = frame.copy()
    model_score = _num(out, "model_error_score", 0.0)
    manual_score = _num(out, "manual_calibrated_error_probability", 0.0)
    confidence = _num(out, "confidence", 0.0)
    group_thresholds = thresholds.get("phone_group_thresholds", {}) or {}
    group = out.get("phone_group", pd.Series("", index=out.index)).astype(str)
    manual_t = group.map(group_thresholds).fillna(float(thresholds.get("manual_calibrated_error_probability_threshold", 0.95))).astype(float)
    pred = (
        (model_score >= float(thresholds.get("model_error_score_threshold", 0.5)))
        & (manual_score >= manual_t)
        & (confidence >= float(thresholds.get("confidence_threshold", 0.0)))
    )
    bad_align = out.get("alignment_quality", pd.Series("", index=out.index)).astype(str).str.lower().eq("bad")
    pred = pred & ~bad_align
    out["sweep_true_error_prediction"] = pred.astype(int)
    out["sweep_decision"] = np.where(pred, "true_error", out.get("decision", "uncertain_review"))
    out.loc[bad_align, "sweep_decision"] = "uncertain_review"
    out["sweep_model_error_score_threshold"] = float(thresholds.get("model_error_score_threshold", 0.5))
    out["sweep_manual_probability_threshold"] = manual_t
    out["sweep_confidence_threshold"] = float(thresholds.get("confidence_threshold", 0.0))
    out.attrs["thresholds"] = thresholds
    return out


def _metric_row(frame: pd.DataFrame, precision_target: float) -> dict:
    eval_frame = _true_error_eval(frame)
    y_true = eval_frame["manual_review_label"].isin(PRONUNCIATION_ERROR_LABELS).to_numpy(dtype=int)
    y_pred = eval_frame["sweep_true_error_prediction"].to_numpy(dtype=int)
    metrics = _binary_metrics(y_true, y_pred)
    thresholds = frame.attrs.get("thresholds", {})
    acceptable = frame["manual_review_label"].eq("acceptable_accent")
    bad_alignment = frame["manual_review_label"].eq("bad_alignment")
    return {
        "threshold_mode": thresholds.get("threshold_mode", "global"),
        "model_error_score_threshold": thresholds.get("model_error_score_threshold", 0.5),
        "manual_calibrated_error_probability_threshold": thresholds.get("manual_calibrated_error_probability_threshold", 0.95),
        "confidence_threshold": thresholds.get("confidence_threshold", 0.0),
        "phone_group_thresholds_json": json.dumps(thresholds.get("phone_group_thresholds", {}), ensure_ascii=False, sort_keys=True),
        "precision_target_met": metrics["precision"] >= precision_target,
        **metrics,
        "predicted_true_error_count": int(y_pred.sum()),
        "false_positive_count": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_positive_count": int(((y_true == 1) & (y_pred == 1)).sum()),
        "acceptable_accent_predicted_true_error_count": int((frame.loc[acceptable, "sweep_true_error_prediction"] == 1).sum()),
        "acceptable_accent_false_positive_rate": round(float((frame.loc[acceptable, "sweep_true_error_prediction"] == 1).mean()), 6) if acceptable.any() else 0.0,
        "bad_alignment_predicted_true_error_count": int((frame.loc[bad_alignment, "sweep_true_error_prediction"] == 1).sum()),
    }


def _best_row(sweep: pd.DataFrame, precision_target: float) -> dict:
    valid = sweep[sweep["precision"] >= precision_target].copy()
    if valid.empty:
        valid = sweep.copy()
    valid = valid.sort_values(
        ["recall", "precision", "f1", "false_positive_count", "predicted_true_error_count", "acceptable_accent_predicted_true_error_count"],
        ascending=[False, False, False, True, True, True],
    )
    row = valid.iloc[0].to_dict()
    row["phone_group_thresholds"] = json.loads(row.pop("phone_group_thresholds_json"))
    for key, value in list(row.items()):
        if isinstance(value, (np.integer, np.floating)):
            row[key] = value.item()
    return row


def _acceptable_accent_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    acceptable = frame[frame["manual_review_label"].eq("acceptable_accent")].copy()
    rows = []
    for level in ["target_phone", "phone_group", "speaker_id"]:
        if level not in acceptable.columns:
            continue
        for key, group in acceptable.groupby(level):
            rows.append({
                "level": level,
                "key": str(key),
                "acceptable_accent_count": int(len(group)),
                "predicted_true_error_count": int(group["sweep_true_error_prediction"].sum()),
                "false_positive_rate": round(float(group["sweep_true_error_prediction"].mean()), 6) if len(group) else 0.0,
            })
    columns = ["level", "key", "acceptable_accent_count", "predicted_true_error_count", "false_positive_rate"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["predicted_true_error_count", "acceptable_accent_count"], ascending=[False, False])


def _alignment_failure_analysis(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    bad = frame.get("alignment_quality", pd.Series("", index=frame.index)).astype(str).str.lower().eq("bad")
    duration = _num(frame, "duration_ms", 0.0)
    duration_outlier = (duration < 20.0) | (duration > 500.0)
    missing_phone = frame.get("target_phone", pd.Series("", index=frame.index)).astype(str).str.strip().eq("")
    report = {
        "rows": int(len(frame)),
        "alignment_failure_count": int(bad.sum()),
        "alignment_failure_rate": round(float(bad.mean()), 6) if len(frame) else 0.0,
        "bad_alignment_predicted_true_error_count": int((bad & frame["sweep_true_error_prediction"].astype(bool)).sum()),
        "duration_outlier_count": int(duration_outlier.sum()),
        "missing_phone_count": int(missing_phone.sum()),
    }
    samples = frame[bad | duration_outlier | missing_phone].copy()
    samples["duration_outlier_flag"] = duration_outlier[bad | duration_outlier | missing_phone].to_numpy(dtype=int)
    samples["missing_phone_flag"] = missing_phone[bad | duration_outlier | missing_phone].to_numpy(dtype=int)
    return report, samples


def _calibration(frame: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    eval_frame = _true_error_eval(frame)
    y_true = eval_frame["manual_review_label"].isin(PRONUNCIATION_ERROR_LABELS).to_numpy(dtype=int)
    score = _num(eval_frame, "manual_calibrated_error_probability", 0.0).clip(0, 1).to_numpy()
    if len(eval_frame) == 0:
        return pd.DataFrame(), {"brier_score": 0.0, "ece": 0.0}
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (score >= low) & (score < high if i < bins - 1 else score <= high)
        count = int(mask.sum())
        if count:
            mean_score = float(score[mask].mean())
            frac_pos = float(y_true[mask].mean())
            ece += (count / len(score)) * abs(mean_score - frac_pos)
        else:
            mean_score = 0.0
            frac_pos = 0.0
        rows.append({
            "bin_low": round(float(low), 6),
            "bin_high": round(float(high), 6),
            "count": count,
            "mean_predicted_probability": round(mean_score, 6),
            "true_error_rate": round(frac_pos, 6),
            "abs_gap": round(abs(mean_score - frac_pos), 6),
        })
    return pd.DataFrame(rows), {
        "brier_score": round(float(brier_score_loss(y_true, score)), 6) if len(np.unique(y_true)) > 1 else 0.0,
        "ece": round(float(ece), 6),
        "calibration_eval_rows": int(len(eval_frame)),
    }


def _markdown_report(frame: pd.DataFrame, sweep: pd.DataFrame, best: dict, alignment: dict, calibration: dict, precision_target: float) -> str:
    current_eval = _true_error_eval(frame)
    current_metrics = _binary_metrics(
        current_eval["manual_review_label"].isin(PRONUNCIATION_ERROR_LABELS).to_numpy(dtype=int),
        current_eval["decision"].eq("true_error").to_numpy(dtype=int),
    )
    best_recall = best["recall"] if best["precision"] >= precision_target else 0.0
    acceptable_fp = best.get("acceptable_accent_predicted_true_error_count", 0)
    next_focus = "label thresholding/model calibration"
    if alignment["alignment_failure_rate"] > 0.10 or alignment["bad_alignment_predicted_true_error_count"] > 0:
        next_focus = "alignment"
    elif acceptable_fp > 0:
        next_focus = "label definition and acceptable_accent separation"
    return "\n".join([
        "# E2E Alpha Metric Optimization Report",
        "",
        "## Current Metrics",
        "",
        f"- true_error precision: {current_metrics['precision']}",
        f"- true_error recall: {current_metrics['recall']}",
        f"- true_error f1: {current_metrics['f1']}",
        "",
        "## Best Sweep Result",
        "",
        f"- precision target: {precision_target}",
        f"- best precision: {best['precision']}",
        f"- best recall at precision >= {precision_target}: {best_recall}",
        f"- best f1: {best['f1']}",
        f"- threshold mode: {best['threshold_mode']}",
        f"- model_error_score threshold: {best['model_error_score_threshold']}",
        f"- manual probability threshold: {best['manual_calibrated_error_probability_threshold']}",
        f"- confidence threshold: {best['confidence_threshold']}",
        "",
        "## Acceptable Accent False Positives",
        "",
        f"- acceptable_accent predicted true_error count under best thresholds: {acceptable_fp}",
        f"- acceptable_accent false positive rate: {best.get('acceptable_accent_false_positive_rate', 0.0)}",
        "",
        "## Alignment Failure",
        "",
        f"- alignment failure rate: {alignment['alignment_failure_rate']}",
        f"- bad alignment predicted true_error count: {alignment['bad_alignment_predicted_true_error_count']}",
        f"- duration outlier count: {alignment['duration_outlier_count']}",
        f"- missing phone count: {alignment['missing_phone_count']}",
        "",
        "## Calibration",
        "",
        f"- Brier score: {calibration['brier_score']}",
        f"- ECE: {calibration['ece']}",
        "",
        "## Recommendation",
        "",
        f"Next optimization focus: {next_focus}.",
        "",
        "Public dataset metrics and real end-to-end manual-label metrics must remain separate.",
        "",
    ]) + "\n"


def _true_error_eval(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[~frame["manual_review_label"].eq("bad_alignment")].copy()


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


def _num(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


if __name__ == "__main__":
    main()
