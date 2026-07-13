#!/usr/bin/env python
"""Analyze false positives on a correct-audio sanity set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def analyze_false_positives(prediction_csv: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(prediction_csv) if prediction_csv.exists() else pd.DataFrame()
    false_pos = frame[frame.get("decision", pd.Series(dtype=str)).astype(str).eq("true_error")].copy()

    by_phone = _count(false_pos, "target_phone", "false_positive_count")
    by_word = _count(false_pos, "word", "false_positive_count")
    by_group = _count(false_pos, "phone_group", "false_positive_count")
    by_phone.to_csv(output_dir / "false_positive_by_phone.csv", index=False, encoding="utf-8-sig")
    by_word.to_csv(output_dir / "false_positive_by_word.csv", index=False, encoding="utf-8-sig")
    by_group.to_csv(output_dir / "false_positive_by_phone_group.csv", index=False, encoding="utf-8-sig")

    weak_vowels = {"AH", "IH", "ER", "R", "AX", "IX"}
    phone_values = set(false_pos.get("target_phone", pd.Series(dtype=str)).astype(str))
    duration = pd.to_numeric(false_pos.get("duration_ms", pd.Series(dtype=float)), errors="coerce")
    prob_correct = pd.to_numeric(false_pos.get("prob_correct", pd.Series(dtype=float)), errors="coerce")
    report = {
        "false_positive_count": int(len(false_pos)),
        "weak_vowel_false_positive_count": int(sum(phone in weak_vowels for phone in phone_values)),
        "short_duration_false_positive_count": int((duration < 50).sum()),
        "high_prob_correct_but_true_error_count": int((prob_correct >= 0.75).sum()),
    }
    markdown = [
        "# Correct Audio False Positive Analysis",
        "",
        f"- False positive phones: {report['false_positive_count']}",
        f"- False positives touching weak vowels/AH/IH/ER/R: {report['weak_vowel_false_positive_count']}",
        f"- False positives with duration_ms < 50: {report['short_duration_false_positive_count']}",
        f"- prob_correct >= 0.75 but true_error: {report['high_prob_correct_but_true_error_count']}",
        "",
        "## Top Phones",
        _markdown_table(by_phone.head(20)),
        "",
        "## Top Words",
        _markdown_table(by_word.head(20)),
        "",
        "## Top Phone Groups",
        _markdown_table(by_group.head(20)),
    ]
    (output_dir / "false_positive_analysis.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


def _count(frame: pd.DataFrame, col: str, count_col: str) -> pd.DataFrame:
    if frame.empty or col not in frame.columns:
        return pd.DataFrame(columns=[col, count_col])
    return frame.groupby(col, dropna=False).size().reset_index(name=count_col).sort_values(count_col, ascending=False)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in frame.astype(str).to_dict(orient="records"):
        lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze true_error false positives in correct-audio predictions.")
    parser.add_argument("prediction_csv", nargs="?", type=Path, default=ROOT / "outputs/sanity_correct/prediction_all.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sanity_correct")
    args = parser.parse_args()
    analyze_false_positives(args.prediction_csv, args.output_dir)
    print(f"Wrote false positive analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
