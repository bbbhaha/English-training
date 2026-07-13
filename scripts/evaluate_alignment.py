#!/usr/bin/env python
"""Evaluate forced alignment against held-out L2-ARCTIC manual boundaries."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/l2_arctic/phones.csv",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "artifacts/baseline_acoustic_v1/phone_gaussians.joblib",
    )
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    models = joblib.load(args.models)
    values: dict[str, list[float]] = defaultdict(list)
    by_utterance: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row["split"] == "train"
            and row["gold_binary"] == "1"
            and row["duration_ms"]
            and row["error_type"] != "addition"
        ):
            values[row["target_phone"]].append(float(row["duration_ms"]))
        if (
            row["split"] in {"dev", "test"}
            and row["error_type"] != "addition"
            and row["target_phone"] in models
            and row["start_ms"]
            and row["end_ms"]
        ):
            by_utterance[row["utterance_id"]].append(row)
    priors = {
        phone: float(statistics.median(durations))
        for phone, durations in values.items()
    }

    errors: list[float] = []
    failures: list[dict[str, str]] = []
    selected = list(sorted(by_utterance.items()))[:args.limit]
    for utterance_id, utterance_rows in selected:
        utterance_rows.sort(key=lambda row: int(row["phone_index"]))
        try:
            rate, signal = read_wav_mono(ROOT / utterance_rows[0]["audio_path"])
            result = align_signal(
                signal,
                rate,
                [row["target_phone"] for row in utterance_rows],
                models,
                priors,
            )
            errors.extend(
                abs(boundary[1] - float(row["end_ms"]))
                for boundary, row in zip(
                    result.boundaries_ms[:-1], utterance_rows[:-1]
                )
            )
        except Exception as error:
            failures.append({"utterance_id": utterance_id, "error": str(error)})

    ordered = sorted(errors)
    report = {
        "utterances": len(selected),
        "failures": len(failures),
        "boundaries": len(errors),
        "median_absolute_error_ms": statistics.median(errors),
        "p75_absolute_error_ms": ordered[int(0.75 * len(ordered))],
        "within_50_ms": sum(error <= 50 for error in errors) / len(errors),
        "within_100_ms": sum(error <= 100 for error in errors) / len(errors),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
