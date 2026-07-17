#!/usr/bin/env python
"""Select high-confidence SpeechOcean762 features for auxiliary phone training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--phones", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--correct-to-error-ratio", type=float, default=2.0)
    args = parser.parse_args()

    features = pd.read_csv(args.features, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    labels = pd.read_csv(
        args.phones,
        encoding="utf-8-sig",
        usecols=["utterance_id", "phone_index", "gold_three_class", "source_score"],
        low_memory=False,
    )
    selected, report = select_high_confidence_features(
        features,
        labels,
        correct_to_error_ratio=args.correct_to_error_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def select_high_confidence_features(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    correct_to_error_ratio: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    metadata = labels.drop_duplicates(["utterance_id", "phone_index"], keep="last")
    merged = features.merge(
        metadata,
        on=["utterance_id", "phone_index"],
        how="left",
        validate="one_to_one",
    )
    if merged["gold_three_class"].isna().any():
        missing = int(merged["gold_three_class"].isna().sum())
        raise ValueError(f"{missing} feature rows have no SpeechOcean label")

    incorrect = merged[merged["gold_three_class"].eq("incorrect")].copy()
    correct = merged[merged["gold_three_class"].eq("correct")].copy()
    correct_count = min(len(correct), int(round(len(incorrect) * correct_to_error_ratio)))
    correct = correct.sample(n=correct_count, random_state=RANDOM_SEED) if correct_count else correct.iloc[0:0]
    selected = pd.concat([incorrect, correct], ignore_index=True)
    selected["gold_phone_state"] = selected["gold_three_class"].map(
        {"correct": "correct", "incorrect": "mispronounced"}
    )
    selected["error_type"] = selected["gold_phone_state"].map(
        {"correct": "correct", "mispronounced": "substitution"}
    )
    selected["split"] = "train"
    selected = selected.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    report = {
        "dataset": "SpeechOcean762",
        "selection": "high-confidence auxiliary phones",
        "excluded_label": "acceptable",
        "correct_to_error_ratio": float(correct_to_error_ratio),
        "input_rows": int(len(features)),
        "selected_rows": int(len(selected)),
        "selected_utterances": int(selected["utterance_id"].nunique()),
        "selected_speakers": int(selected["speaker_id"].nunique()),
        "labels": {
            str(key): int(value)
            for key, value in selected["gold_phone_state"].value_counts().to_dict().items()
        },
    }
    return selected.drop(columns=["gold_three_class", "source_score"]), report


if __name__ == "__main__":
    main()
