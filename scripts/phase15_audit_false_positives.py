#!/usr/bin/env python
"""Export Phase-1.5 false-positive and manual-review audit packets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase15_verification.analysis import CORE_PHONE_DEFAULTS, add_evidence_columns, write_analysis_tables
from phase15_verification.config import load_config
from phase15_verification.labels import infer_error_labels


AUDIT_COLUMNS = [
    "utterance_id",
    "speaker_id",
    "word",
    "target_phone",
    "phone_group",
    "start_ms",
    "end_ms",
    "gold_label",
    "gold_binary",
    "prediction",
    "prob_correct",
    "mispronounced_probability",
    "confidence",
    "gop_score",
    "duration_ms",
    "duration",
    "alignment_quality",
    "main_model_error_score",
    "main_model_error_flag",
    "attribute_risk_score",
    "attribute_mismatch_count",
    "attribute_verifier_decision",
    "proto_same_phone_sim_top1",
    "proto_same_phone_sim_topk_mean",
    "proto_confusion_phone_sim_top1",
    "proto_margin",
    "retrieval_verifier_decision",
    "oneclass_anomaly_score",
    "oneclass_verifier_decision",
    "evidence_count",
    "verifier_evidence_count",
    "evidence_pattern",
    "audit_reason",
    "final_error_score",
    "final_decision",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase-1.5 high-confidence false-positive audit files.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/phase15_verification/test_verified_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase15_verification/audit")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "outputs/phase15_verification/analysis")
    parser.add_argument("--label-col", default="gold_binary")
    parser.add_argument("--error-value", default="auto")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase15/aggregator.yaml")
    parser.add_argument("--manual-review-limit", type=int, default=200)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    cfg = load_config(args.config)
    frame = add_evidence_columns(frame, cfg)
    y_true = infer_error_labels(frame, args.label_col, args.error_value)
    pred_hce = (frame["final_decision"].astype(str) == "high_confidence_error").to_numpy(dtype=int)
    frame["_is_error"] = y_true
    frame["_pred_hce"] = pred_hce
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(frame[(frame["_pred_hce"] == 1) & (frame["_is_error"] == 0)], args.output_dir / "high_confidence_false_positives.csv")
    _write(frame[(frame["_pred_hce"] == 1) & (frame["_is_error"] == 1)], args.output_dir / "high_confidence_true_positives.csv")
    _write(frame[(frame["_pred_hce"] == 0) & (frame["_is_error"] == 1)], args.output_dir / "false_negatives.csv")
    _write(frame[frame["final_decision"].astype(str) == "uncertain_review"], args.output_dir / "uncertain_review_samples.csv")
    packet_path = args.output_dir / "manual_review_packet.csv"
    packet = _manual_review_packet(frame, args.manual_review_limit)
    packet = _preserve_manual_annotations(packet, packet_path)
    packet.to_csv(packet_path, index=False, encoding="utf-8-sig")
    core_phones = cfg.get("evaluation", {}).get("core_phones", CORE_PHONE_DEFAULTS)
    write_analysis_tables(frame, args.analysis_dir, args.label_col, args.error_value, core_phones)
    print(f"Audit files saved under: {args.output_dir}")
    print(f"Analysis tables saved under: {args.analysis_dir}")


def _write(frame: pd.DataFrame, path: Path) -> None:
    out = _audit_view(frame)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _audit_view(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in AUDIT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[AUDIT_COLUMNS]


def _manual_review_packet(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    out = frame.copy()
    core = set(CORE_PHONE_DEFAULTS)
    target_phone = out["target_phone"] if "target_phone" in out.columns else pd.Series("", index=out.index)
    main_score = _numeric(out, "main_model_error_score", 0.0)
    final_score = _numeric(out, "final_error_score", 0.0)
    out["_core_priority"] = target_phone.astype(str).str.upper().isin(core).astype(int)
    out["_priority"] = np.select(
        [
            (out["_pred_hce"] == 1) & (out["_is_error"] == 0),
            (out["_pred_hce"] == 1) & (out["_is_error"] == 1),
            (out["_pred_hce"] == 0) & (out["_is_error"] == 1) & (main_score >= 0.35),
            out["final_decision"].astype(str) == "uncertain_review",
        ],
        [1, 2, 3, 4],
        default=5,
    )
    out["_boundary_distance"] = (final_score - 0.35).abs()
    out = out.sort_values(["_priority", "_core_priority", "final_error_score", "_boundary_distance"], ascending=[True, False, False, True])
    keep = [
        "utterance_id",
        "speaker_id",
        "word",
        "target_phone",
        "phone_group",
        "start_ms",
        "end_ms",
        "gold_label",
        "gold_binary",
        "final_decision",
        "final_error_score",
        "evidence_pattern",
        "audit_reason",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = ""
    packet = out[keep].head(limit).copy()
    packet["manual_review_label"] = ""
    packet["manual_review_notes"] = ""
    return packet


def _preserve_manual_annotations(packet: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    if not existing_path.exists():
        return packet
    existing = pd.read_csv(existing_path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if "manual_review_label" not in existing.columns and "manual_review_notes" not in existing.columns:
        return packet
    left = packet.copy()
    right = existing.copy()
    left["_review_key"] = _review_key(left)
    right["_review_key"] = _review_key(right)
    annotation_cols = ["_review_key"]
    for col in ["manual_review_label", "manual_review_notes"]:
        if col not in right.columns:
            right[col] = ""
        annotation_cols.append(col)
    merged = left.drop(columns=["manual_review_label", "manual_review_notes"], errors="ignore").merge(
        right[annotation_cols].drop_duplicates("_review_key", keep="last"),
        on="_review_key",
        how="left",
    )
    for col in ["manual_review_label", "manual_review_notes"]:
        merged[col] = merged[col].fillna("")
    return merged.drop(columns=["_review_key"])


def _review_key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["utterance_id", "speaker_id", "word", "target_phone", "start_ms", "end_ms"]:
        if col not in frame.columns:
            value = pd.Series("", index=frame.index)
        elif col in {"start_ms", "end_ms"}:
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


def _numeric(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index)


if __name__ == "__main__":
    main()
