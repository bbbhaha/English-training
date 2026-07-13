#!/usr/bin/env python
"""Align a learner wav file to target phones from text or explicit phones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.alignment import align_audio_to_text, save_alignment_csv
from pronunciation.g2p import write_g2p_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create phone-level alignment CSV for a wav + text or wav + phones.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text")
    parser.add_argument("--phones", help="Whitespace or comma separated ARPAbet phones. Overrides --text.")
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/baseline_acoustic_v1/phone_gaussians.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo/alignment.csv")
    parser.add_argument("--g2p-output", type=Path)
    args = parser.parse_args()
    phones = None
    if args.phones:
        phones = [p.strip() for p in args.phones.replace(",", " ").split() if p.strip()]
    if not args.text and not phones:
        raise SystemExit("Provide --text or --phones.")
    alignment, g2p = align_audio_to_text(args.audio, text=args.text, phones=phones, models_path=args.models)
    save_alignment_csv(alignment, args.output)
    if g2p is not None and args.g2p_output:
        write_g2p_json(g2p, args.g2p_output)
    report = {
        "audio": str(args.audio),
        "rows": int(len(alignment)),
        "alignment_quality_counts": alignment["alignment_quality"].value_counts().to_dict() if "alignment_quality" in alignment else {},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote alignment CSV to {args.output}")


if __name__ == "__main__":
    main()
