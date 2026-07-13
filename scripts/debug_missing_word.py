#!/usr/bin/env python
"""Inspect word-level deletion signals in a prediction CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.deletion_detector import build_word_summary, detect_word_deletions


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug possible missing-word/deletion detection.")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--word", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.prediction)
    frame, _ = detect_word_deletions(frame, mode="deletion_only")
    mask = frame["word"].astype(str).str.upper().eq(args.word.upper())
    rows = frame.loc[mask].copy()
    if rows.empty:
        raise SystemExit(f"Word not found in prediction: {args.word}")

    summary = build_word_summary(rows)
    first = summary.iloc[0].to_dict() if not summary.empty else {}
    phone_cols = [
        "target_phone",
        "start_ms",
        "end_ms",
        "duration_ms",
        "prob_correct",
        "manual_calibrated_error_probability",
        "decision",
        "alignment_quality",
        "review_reason",
        "possible_missing_word",
        "deletion_trigger_source",
        "debug_reason",
    ]
    for col in phone_cols:
        if col not in rows.columns:
            rows[col] = ""

    payload = {
        "word": args.word.upper(),
        "phone_rows": rows[phone_cols].to_dict(orient="records"),
        "word_duration_ms": first.get("word_duration_ms", 0.0),
        "short_phone_ratio": first.get("short_phone_ratio", 0.0),
        "debug_high_error_ratio": first.get("high_error_ratio", 0.0),
        "debug_low_prob_correct_ratio": first.get("low_prob_correct_ratio", 0.0),
        "debug_short_phone_ratio": first.get("short_phone_ratio", 0.0),
        "actual_deletion_trigger_source": first.get("deletion_trigger_source", "none"),
        "debug_reason": first.get("debug_reason", ""),
        "possible_missing_word": bool(first.get("possible_missing_word", False)),
        "missing_word_reason": first.get("missing_word_reason", ""),
        "alignment_quality": first.get("alignment_quality", ""),
        "word_decision": first.get("word_decision", ""),
        "error_type": first.get("error_type", ""),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
