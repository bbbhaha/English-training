#!/usr/bin/env python
"""Validate schema, paths, labels, and speaker isolation in a phone manifest."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "utterance_id", "speaker_id", "target_phone", "phone_index", "start_ms",
    "end_ms", "gold_binary", "error_type", "dataset_source", "split", "audio_path",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/validate_manifest.py MANIFEST.csv")
    path = Path(sys.argv[1])
    split_speakers: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    rows = 0

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            rows += 1
            split_speakers[row["split"]].add(row["speaker_id"])
            if row["gold_binary"] not in {"0", "1"}:
                errors.append(f"line {line_number}: invalid gold_binary")
            if row["start_ms"] and row["end_ms"] and float(row["end_ms"]) < float(row["start_ms"]):
                errors.append(f"line {line_number}: negative interval")
            if not (PROJECT_ROOT / row["audio_path"]).is_file():
                errors.append(f"line {line_number}: missing audio {row['audio_path']}")

    split_names = sorted(split_speakers)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = split_speakers[left] & split_speakers[right]
            if overlap:
                errors.append(f"speaker leakage between {left}/{right}: {sorted(overlap)}")
    if errors:
        print("\n".join(errors[:30]))
        raise SystemExit(f"Validation failed with {len(errors)} error(s)")
    print(f"Validation passed: {rows:,} rows; speaker splits are isolated")


if __name__ == "__main__":
    main()
