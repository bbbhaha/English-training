#!/usr/bin/env python
"""Print word-level target/ASR mismatch evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.text_audio_consistency import compare_target_with_asr


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug target text versus an ASR transcript.")
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--asr-transcript", required=True)
    args = parser.parse_args()

    frame = compare_target_with_asr(args.target_text, args.asr_transcript)
    columns = [
        "word_index",
        "word",
        "asr_word",
        "asr_edit_op",
        "asr_word_status",
        "text_audio_mismatch_type",
        "text_audio_mismatch_score",
    ]
    print(frame[columns].to_string(index=False))
    missing = frame.loc[frame["asr_missing_word"], "word"].tolist()
    substituted = frame.loc[frame["asr_substituted_word"], ["word", "asr_word"]].to_dict(orient="records")
    print(f"missing_words={missing}")
    print(f"substituted_words={substituted}")


if __name__ == "__main__":
    main()
