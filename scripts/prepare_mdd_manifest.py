#!/usr/bin/env python
"""Create the unified wav2vec2-MDD phone-level manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phoneme_assessment.mdd import assert_speaker_isolation, source_to_mdd_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/processed/speechocean/phones_aligned.csv",
        help="Existing project phone manifest with start_ms/end_ms/gold_binary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/mdd/speechocean_manifest.csv",
    )
    parser.add_argument(
        "--alignment-quality",
        default="pass",
        help="Filter alignment_quality when present. Use 'all' to disable.",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    alignment_quality = None if args.alignment_quality.lower() == "all" else args.alignment_quality
    mdd = source_to_mdd_manifest(source, project_root=PROJECT_ROOT, alignment_quality=alignment_quality)
    assert_speaker_isolation(mdd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mdd.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(mdd):,} MDD phone rows to {args.output}")
    print(mdd.groupby("split")["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
