#!/usr/bin/env python
"""Generate SpeechOcean762 phone boundaries with monotonic forced alignment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.acoustic import read_wav_mono
from phoneme_assessment.alignment import align_signal


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def duration_priors(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in read_rows(path):
        if (
            row["split"] == "train"
            and row["gold_binary"] == "1"
            and row["duration_ms"]
            and row["error_type"] != "addition"
        ):
            values[row["target_phone"]].append(float(row["duration_ms"]))
    return {
        phone: float(statistics.median(durations))
        for phone, durations in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/speechocean/phones.csv",
    )
    parser.add_argument(
        "--l2-manifest",
        type=Path,
        default=ROOT / "data/processed/l2_arctic/phones.csv",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "artifacts/baseline_acoustic_v1/phone_gaussians.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/speechocean/phones_aligned.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/processed/speechocean/alignment_report.json",
    )
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    models = joblib.load(args.models)
    priors = duration_priors(args.l2_manifest)
    by_utterance: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if args.split is None or row["split"] == args.split:
            by_utterance[row["utterance_id"]].append(row)

    utterance_ids = sorted(by_utterance)
    if args.limit is not None:
        utterance_ids = utterance_ids[:args.limit]

    output_rows: list[dict[str, str]] = []
    utterance_reports: list[dict] = []
    failures: list[dict[str, str]] = []
    for position, utterance_id in enumerate(utterance_ids, start=1):
        utterance_rows = sorted(
            by_utterance[utterance_id],
            key=lambda row: int(row["phone_index"]),
        )
        try:
            rate, signal = read_wav_mono(ROOT / utterance_rows[0]["audio_path"])
            result = align_signal(
                signal,
                rate,
                [row["target_phone"] for row in utterance_rows],
                models,
                priors,
            )
            durations = []
            for row, (start_ms, end_ms) in zip(
                utterance_rows, result.boundaries_ms
            ):
                duration_ms = end_ms - start_ms
                aligned = dict(row)
                aligned["start_ms"] = f"{start_ms:.1f}"
                aligned["end_ms"] = f"{end_ms:.1f}"
                aligned["duration_ms"] = f"{duration_ms:.1f}"
                aligned["alignment_method"] = "segmental_viterbi_gaussian_v1"
                aligned["alignment_score"] = f"{result.score_per_frame:.6f}"
                aligned["alignment_quality"] = (
                    "review" if duration_ms < 20.0 or duration_ms > 500.0 else "pass"
                )
                output_rows.append(aligned)
                durations.append(duration_ms)
            utterance_reports.append(
                {
                    "utterance_id": utterance_id,
                    "phones": len(utterance_rows),
                    "score_per_frame": result.score_per_frame,
                    "active_start_ms": result.active_start_ms,
                    "active_end_ms": result.active_end_ms,
                    "duration_scale": result.duration_scale,
                    "minimum_phone_ms": min(durations),
                    "maximum_phone_ms": max(durations),
                }
            )
        except Exception as error:
            failures.append(
                {"utterance_id": utterance_id, "error": str(error)}
            )
        if position % 100 == 0:
            print(f"Aligned {position:,}/{len(utterance_ids):,} utterances")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

    report = {
        "method": "segmental Viterbi alignment with L2-ARCTIC Gaussian phone models",
        "requested_utterances": len(utterance_ids),
        "aligned_utterances": len(utterance_reports),
        "failed_utterances": len(failures),
        "aligned_phone_rows": len(output_rows),
        "review_phone_rows": sum(
            row["alignment_quality"] == "review" for row in output_rows
        ),
        "score_per_frame": {
            "median": statistics.median(
                item["score_per_frame"] for item in utterance_reports
            ) if utterance_reports else None,
            "minimum": min(
                (item["score_per_frame"] for item in utterance_reports),
                default=None,
            ),
            "maximum": max(
                (item["score_per_frame"] for item in utterance_reports),
                default=None,
            ),
        },
        "failures": failures[:100],
        "utterances": utterance_reports,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Aligned {len(utterance_reports):,} utterances / "
        f"{len(output_rows):,} phones; failures: {len(failures):,}"
    )
    print(f"Wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
