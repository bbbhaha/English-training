#!/usr/bin/env python
"""Build conservative word-level labels from SpeechOcean762 expert scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SpeechOcean762 word-level diagnosis labels.")
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "data/raw/speechocean762/resource/scores.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/speechocean/word_labels.csv",
    )
    args = parser.parse_args()
    rows = prepare_word_labels(args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    counts = pd.Series([row["word_label"] for row in rows]).value_counts().to_dict()
    print(json.dumps({"rows": len(rows), "label_counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


def prepare_word_labels(scores_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for utterance_id, utterance in payload.items():
        words = utterance.get("words", []) if isinstance(utterance, dict) else []
        for word_index, word in enumerate(words):
            scores = [_score(value) for value in word.get("phones-accuracy", [])]
            missing_support = _missing_support(word, len(scores))
            label, evidence = classify_word_label(scores, missing_support=missing_support)
            rows.append(
                {
                    "utterance_id": str(utterance_id),
                    "word_index": word_index,
                    "word": str(word.get("text", "")).upper(),
                    "phones": " ".join(str(value) for value in word.get("phones", [])),
                    "phone_scores": " ".join(f"{value:.3f}" for value in scores),
                    "phone_count": len(scores),
                    "score_2_count": sum(value >= 1.5 for value in scores),
                    "score_1_count": sum(0.5 <= value < 1.5 for value in scores),
                    "score_0_count": sum(value < 0.5 for value in scores),
                    "word_accuracy": _score(word.get("accuracy")),
                    "word_label": label,
                    "label_evidence": evidence,
                    "missing_evidence_available": missing_support,
                }
            )
    return rows


def classify_word_label(scores: list[float], *, missing_support: bool = False) -> tuple[str, str]:
    if not scores:
        return "alignment_issue", "no_phone_scores"
    good = sum(value >= 1.5 for value in scores)
    acceptable = sum(0.5 <= value < 1.5 for value in scores)
    incorrect = sum(value < 0.5 for value in scores)
    majority = len(scores) / 2.0
    if good > majority:
        return "word_correct", "phone_score_2_majority"
    if acceptable > majority and incorrect == 0:
        return "acceptable_accent", "phone_score_1_majority_without_zero"
    if incorrect and missing_support:
        return "deletion_or_missing", "phone_score_0_with_asr_ctc_or_duration_missing_evidence"
    if incorrect:
        return "mispronounced", "phone_score_0_without_direct_missing_evidence"
    if good:
        return "word_correct", "mixed_scores_with_good_support"
    return "acceptable_accent", "nonzero_phone_scores_without_good_majority"


def _missing_support(word: dict[str, Any], phone_count: int) -> bool:
    asr_missing = _truthy(word.get("asr_missing_word"))
    blank_ratio = _optional_float(word.get("ctc_blank_ratio"))
    duration = _optional_float(word.get("word_duration_ms"))
    short = duration is not None and duration < max(80.0, phone_count * 20.0)
    return asr_missing or (blank_ratio is not None and blank_ratio >= 0.8) or short


def _score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
