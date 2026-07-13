#!/usr/bin/env python
"""Combine dataset manifests while retaining dataset-specific nullable fields."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ROOT / "data/processed/speechocean/phones_aligned.csv",
    ROOT / "data/processed/l2_arctic/phones.csv",
]
OUTPUT = ROOT / "data/processed/combined/phones.csv"


def main() -> None:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    for path in INPUTS:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(reader)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
