#!/usr/bin/env python
"""Check whether ASR transcript is consistent with target text at word level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.text_audio_consistency import check_text_audio_consistency, consistency_to_json_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR-based text/audio consistency check.")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--asr-transcript")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame, meta = check_text_audio_consistency(
        audio_path=args.audio,
        target_text=args.target_text,
        asr_transcript=args.asr_transcript,
    )
    payload = consistency_to_json_payload(frame, meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(frame), "asr_transcript": meta["asr_transcript"]}, ensure_ascii=False, indent=2))
    print(f"Wrote text/audio consistency report to {args.output}")


if __name__ == "__main__":
    main()
