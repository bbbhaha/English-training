#!/usr/bin/env python
"""Build an E2E hard-set prediction/label pair from the Phase-1.5 manual review packet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["utterance_id", "speaker_id", "word", "target_phone", "start_ms", "end_ms"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Phase-1.5 manual review rows into E2E Alpha hard-set files.")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/phase15_verification/test_verified_predictions.csv"))
    parser.add_argument("--manual-review", type=Path, default=Path("outputs/phase15_verification/audit/manual_review_packet.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e2e_alpha_hardset"))
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    manual = pd.read_csv(args.manual_review, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    manual = manual[manual["manual_review_label"].astype(str).str.strip() != ""].copy()
    if manual.empty:
        raise SystemExit("No labeled manual review rows found.")
    pred["_key"] = _key(pred)
    manual["_key"] = _key(manual)
    matched = pred.merge(manual[["_key", "manual_review_label", "manual_review_notes"]], on="_key", how="inner")
    if matched.empty:
        raise SystemExit("No manual review rows matched prediction rows.")
    prediction = _to_e2e_prediction(matched)
    labels = _to_e2e_labels(matched)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output_dir / "hardset_prediction.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(args.output_dir / "hardset_manual_labels.csv", index=False, encoding="utf-8-sig")
    print(f"Matched rows: {len(matched)}")
    print(f"Wrote {args.output_dir / 'hardset_prediction.csv'}")
    print(f"Wrote {args.output_dir / 'hardset_manual_labels.csv'}")


def _to_e2e_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model_error_score" not in out.columns:
        if "main_model_error_score" in out.columns:
            out["model_error_score"] = out["main_model_error_score"]
        elif "prob_correct" in out.columns:
            out["model_error_score"] = 1.0 - pd.to_numeric(out["prob_correct"], errors="coerce").fillna(1.0)
        else:
            out["model_error_score"] = pd.to_numeric(out.get("final_error_score", 0.0), errors="coerce").fillna(0.0)
    if "manual_calibrated_error_probability" not in out.columns:
        out["manual_calibrated_error_probability"] = pd.to_numeric(out.get("final_error_score", 0.0), errors="coerce").fillna(0.0)
    if "confidence" not in out.columns:
        out["confidence"] = pd.to_numeric(out.get("final_error_score", 0.0), errors="coerce").fillna(0.0)
    if "decision" not in out.columns:
        out["decision"] = out.get("final_decision", "uncertain_review")
    out["decision"] = out["decision"].replace({"high_confidence_error": "true_error"})
    if "error_type" not in out.columns:
        out["error_type"] = out["decision"].map(
            {
                "true_error": "pronunciation_error",
                "acceptable_accent": "acceptable_accent",
                "correct": "none",
                "uncertain_review": "uncertain",
            }
        ).fillna("uncertain")
    if "alignment_quality" not in out.columns:
        out["alignment_quality"] = "unknown"
    if "review_reason" not in out.columns:
        out["review_reason"] = out.get("audit_reason", "")
    manual_bad_alignment = out.get("manual_review_label", pd.Series("", index=out.index)).astype(str).str.strip().eq("bad_alignment")
    out.loc[manual_bad_alignment, "alignment_quality"] = "bad"
    out.loc[manual_bad_alignment, "decision"] = "uncertain_review"
    out.loc[manual_bad_alignment, "confidence"] = 0.0
    out.loc[manual_bad_alignment, "error_type"] = "bad_alignment"
    out.loc[manual_bad_alignment, "review_reason"] = out.loc[manual_bad_alignment, "review_reason"].map(
        lambda value: _merge_reason(value, "manual_bad_alignment")
    )
    if "g2p_source" not in out.columns:
        out["g2p_source"] = "dataset_manifest"
    keep = [
        "utterance_id",
        "speaker_id",
        "word",
        "word_index",
        "target_phone",
        "phone_index",
        "start_ms",
        "end_ms",
        "duration_ms",
        "model_error_score",
        "prob_correct",
        "manual_calibrated_error_probability",
        "decision",
        "confidence",
        "error_type",
        "alignment_quality",
        "review_reason",
        "g2p_source",
        "phone_group",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = ""
    return out[keep]


def _merge_reason(existing: object, extra: str) -> str:
    parts = []
    for value in [existing, extra]:
        for part in str(value or "").split(";"):
            item = part.strip()
            if item and item not in parts:
                parts.append(item)
    return ";".join(parts)


def _to_e2e_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    labels = pd.DataFrame()
    for col in KEY_COLUMNS:
        labels[col] = out[col] if col in out.columns else ""
    labels["old_prediction"] = out.get("final_decision", out.get("decision", ""))
    labels["manual_review_label"] = out["manual_review_label"].astype(str).str.strip()
    labels["manual_review_notes"] = out.get("manual_review_notes", "")
    labels["reviewer"] = "phase15_manual_review"
    labels["review_date"] = "2026-07-08"
    return labels


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


if __name__ == "__main__":
    main()
