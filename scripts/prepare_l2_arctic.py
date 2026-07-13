#!/usr/bin/env python
"""Convert L2-ARCTIC Mandarin manual annotations to the phase-one manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phoneme_assessment.l2_arctic import annotation_rows, write_csv
from phoneme_assessment.mdd import assert_speaker_isolation, source_to_mdd_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/L2-ARCTIC-v5.0/Mandarin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/l2_arctic/phones.csv",
    )
    parser.add_argument(
        "--mdd-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/mdd/l2_arctic_manifest.csv",
        help="Optional wav2vec2-MDD manifest with label=1 meaning mispronounced.",
    )
    args = parser.parse_args()

    config = json.loads((PROJECT_ROOT / "config/phase1.json").read_text(encoding="utf-8"))
    speaker_splits = config["l2_arctic"]["speaker_splits"]
    split_by_speaker = {
        speaker: split
        for split, speakers in speaker_splits.items()
        for speaker in speakers
    }

    rows: list[dict[str, object]] = []
    file_counts: Counter[str] = Counter()
    for speaker_dir in sorted(path for path in args.input.iterdir() if path.is_dir()):
        speaker = speaker_dir.name
        if speaker not in split_by_speaker:
            continue
        for annotation in sorted((speaker_dir / "annotation").glob("*.TextGrid")):
            rows.extend(
                annotation_rows(
                    annotation, args.input, speaker, split_by_speaker[speaker], PROJECT_ROOT
                )
            )
            file_counts[speaker] += 1

    write_csv(rows, args.output)
    if args.mdd_output:
        import pandas as pd

        source = pd.DataFrame(rows)
        mdd = source_to_mdd_manifest(source, project_root=PROJECT_ROOT)
        assert_speaker_isolation(mdd)
        args.mdd_output.parent.mkdir(parents=True, exist_ok=True)
        mdd.to_csv(args.mdd_output, index=False, encoding="utf-8-sig")
    summary_path = args.output.with_name("summary.json")
    summary = {
        "annotation_files": dict(file_counts),
        "phone_rows": len(rows),
        "speakers": dict(Counter(str(row["speaker_id"]) for row in rows)),
        "splits": dict(Counter(str(row["split"]) for row in rows)),
        "labels": dict(Counter(str(row["gold_binary"]) for row in rows)),
        "error_types": dict(Counter(str(row["error_type"]) for row in rows)),
        "phone_groups": dict(Counter(str(row["phone_group"]) for row in rows)),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(rows):,} phone rows to {args.output}")
    print(f"Wrote summary to {summary_path}")
    if args.mdd_output:
        print(f"Wrote MDD manifest to {args.mdd_output}")


if __name__ == "__main__":
    main()
