#!/usr/bin/env python
"""Inspect lexicon/G2P coverage for an arbitrary target sentence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.g2p import text_to_phones
from pronunciation.target_words import build_target_word_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Test G2P coverage for arbitrary English words.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = build_target_word_table(args.text)
    result = text_to_phones(args.text, target_word_table=target)
    words = pd.DataFrame(result.words)
    if words.empty:
        words = target.copy()
    words["selected_pronunciation"] = words["selected_pronunciation"].map(_phone_text)
    words["pronunciations"] = words["pronunciations"].map(_variant_text)
    words["target_phone_available"] = words["selected_pronunciation"].ne("") | words["g2p_status"].eq("failed")
    columns = [
        "word_index", "word", "selected_pronunciation", "pronunciations",
        "pronunciation_variant_id", "num_pronunciation_variants", "lexicon_status",
        "g2p_source", "g2p_confidence", "g2p_status", "g2p_error", "error_type",
        "target_phone_available",
    ]
    output = words[columns].sort_values("word_index", kind="stable")
    output.to_csv(args.output_dir / "unseen_words_g2p.csv", index=False, encoding="utf-8-sig")

    expected = set(target["word_index"].astype(int))
    actual = set(output["word_index"].astype(int))
    missing = target.loc[target["word_index"].isin(expected - actual), "word"].tolist()
    report_lines = [
        "# Unseen words G2P report",
        "",
        f"- expected_word_count: {len(target)}",
        f"- output_word_count: {len(output)}",
        f"- missing_words: {missing}",
        "",
        "| index | word | selected pronunciation | variants | source | confidence |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    for row in output.itertuples(index=False):
        report_lines.append(
            f"| {row.word_index} | {row.word} | {row.selected_pronunciation or '<UNK>'} | "
            f"{row.num_pronunciation_variants} | {row.g2p_source} | {row.g2p_confidence} |"
        )
    (args.output_dir / "unseen_words_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(output.to_string(index=False))
    if missing or not output["target_phone_available"].all():
        raise SystemExit(1)


def _phone_text(value: object) -> str:
    return " ".join(str(phone) for phone in (value or []))


def _variant_text(value: object) -> str:
    return " | ".join(" ".join(str(phone) for phone in variant) for variant in (value or []))


if __name__ == "__main__":
    main()
