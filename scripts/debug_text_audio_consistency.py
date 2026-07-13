#!/usr/bin/env python
"""Debug word-level alignment between target text and ASR transcript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.text_audio_consistency import check_text_audio_consistency


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug ASR text/audio consistency for one word.")
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--asr-transcript", required=True)
    parser.add_argument("--word", required=True)
    args = parser.parse_args()

    frame, meta = check_text_audio_consistency(target_text=args.target_text, asr_transcript=args.asr_transcript)
    word_rows = frame[frame["target_word"].astype(str).str.upper().eq(args.word.upper())]
    missing = frame[frame["asr_word_status"].eq("deletion")]["target_word"].tolist()
    substitutions = frame[frame["asr_word_status"].eq("substitution")][["target_word", "recognized_word"]].to_dict(orient="records")
    reason = ""
    if not word_rows.empty and bool(word_rows.iloc[0]["asr_missing_word"]):
        reason = "target_word_missing_in_asr_transcript"
    elif not word_rows.empty:
        reason = f"asr_word_status={word_rows.iloc[0]['asr_word_status']}"
    payload = {
        "target_text_normalized": meta["target_text_normalized"],
        "asr_transcript_normalized": meta["asr_transcript_normalized"],
        "word_alignment_table": frame.to_dict(orient="records"),
        "missing_words": missing,
        "substitution_words": substitutions,
        "word": args.word.upper(),
        "deletion_decision_reason": reason,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
