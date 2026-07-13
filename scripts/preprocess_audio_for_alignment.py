#!/usr/bin/env python
"""Convert arbitrary learner audio to 16 kHz mono PCM wav for alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.audio_preprocess import preprocess_audio, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio before forced alignment.")
    parser.add_argument("--input", type=Path, required=True, help="Input wav/mp3 path.")
    parser.add_argument("--output", type=Path, required=True, help="Output 16 kHz mono PCM wav path.")
    parser.add_argument("--trim-silence", action="store_true", help="Trim long leading/trailing silence.")
    parser.add_argument("--silence-threshold", type=float, default=0.01)
    parser.add_argument("--silence-pad-ms", type=float, default=100.0)
    parser.add_argument("--report-output", type=Path, help="Optional JSON audio report path.")
    args = parser.parse_args()

    report = preprocess_audio(
        input_path=args.input,
        output_path=args.output,
        trim_silence=args.trim_silence,
        silence_threshold=args.silence_threshold,
        silence_pad_ms=args.silence_pad_ms,
    )
    write_report(report, args.report_output)


if __name__ == "__main__":
    main()

