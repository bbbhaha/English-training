#!/usr/bin/env python
"""Convert SpeechOcean762 to the unified phase-one phone manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phoneme_assessment.speechocean import convert, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data/raw/speechocean762"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/speechocean/phones.csv",
    )
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"SpeechOcean762 not found at {args.input}. "
            "Run scripts/download_speechocean.ps1 first."
        )
    config = json.loads((PROJECT_ROOT / "config/phase1.json").read_text(encoding="utf-8"))
    rows, speaker_splits = convert(args.input, PROJECT_ROOT, config["random_seed"])
    write_csv(rows, args.output)

    split_dir = args.output.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        speakers = sorted(s for s, value in speaker_splits.items() if value == split)
        (split_dir / f"{split}_speakers.txt").write_text(
            "\n".join(speakers) + "\n", encoding="utf-8"
        )
    summary = {
        "phone_rows": len(rows),
        "utterances": len({row["utterance_id"] for row in rows}),
        "speakers": len({row["speaker_id"] for row in rows}),
        "split_speakers": dict(
            Counter(
                (row["split"], row["speaker_id"])
                for row in rows
            )
        ),
        "splits": dict(Counter(row["split"] for row in rows)),
        "labels": dict(Counter(str(row["gold_binary"]) for row in rows)),
        "three_class": dict(Counter(row["gold_three_class"] for row in rows)),
        "phone_groups": dict(Counter(row["phone_group"] for row in rows)),
    }
    # JSON cannot encode tuple keys; report distinct speaker counts separately.
    summary["split_speakers"] = {
        split: len({row["speaker_id"] for row in rows if row["split"] == split})
        for split in ("train", "dev", "test")
    }
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(rows):,} phone rows to {args.output}")
    print(f"Wrote summary and speaker split files to {args.output.parent}")


if __name__ == "__main__":
    main()
