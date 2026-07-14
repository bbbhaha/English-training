#!/usr/bin/env python
"""Run an end-to-end sentence and audit target-word coverage at every stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.target_words import build_target_word_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug sentence word coverage.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "prediction.csv"
    summary_path = args.output_dir / "word_summary.csv"
    alignment_path = args.output_dir / "alignment.csv"
    g2p_path = args.output_dir / "g2p.json"

    command = [
        sys.executable,
        str(ROOT / "scripts" / "predict_pronunciation.py"),
        "--audio", str(args.audio),
        "--text", args.text,
        "--output", str(prediction_path),
        "--word-summary-output", str(summary_path),
        "--alignment-output", str(alignment_path),
        "--g2p-output", str(g2p_path),
        "--decision-mode", "deletion_only",
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    target = build_target_word_table(args.text)
    g2p_payload = json.loads(g2p_path.read_text(encoding="utf-8"))
    g2p = pd.DataFrame(g2p_payload.get("phones", []))
    alignment = pd.read_csv(alignment_path, encoding="utf-8-sig")
    prediction = pd.read_csv(prediction_path, encoding="utf-8-sig")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    expected = set(target["word_index"].astype(int))
    actual = set(pd.to_numeric(summary["word_index"], errors="coerce").dropna().astype(int))
    missing_indices = sorted(expected - actual)
    missing_words = target.loc[target["word_index"].isin(missing_indices), "word"].tolist()
    report = {
        "expected_word_count": len(target),
        "g2p_word_count": int(g2p["word_index"].nunique()) if not g2p.empty else 0,
        "alignment_word_count": int(alignment["word_index"].nunique()) if not alignment.empty else 0,
        "prediction_word_count": int(prediction["word_index"].nunique()) if not prediction.empty else 0,
        "word_summary_word_count": int(summary["word_index"].nunique()) if not summary.empty else 0,
        "prediction_word_indices": sorted(pd.to_numeric(prediction["word_index"], errors="coerce").dropna().astype(int).unique().tolist()),
        "missing_words": missing_words,
    }
    (args.output_dir / "coverage_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = _markdown_report(target, g2p, alignment, prediction, summary, report)
    (args.output_dir / "coverage_report.md").write_text(markdown, encoding="utf-8")

    print("target_word_table:\n", target.to_string(index=False))
    print("\ng2p_phone_df:\n", g2p.to_string(index=False))
    print("\nalignment_df:\n", alignment.to_string(index=False))
    print("\nprediction word_index:", report["prediction_word_indices"])
    print("\nword_summary:\n", summary.to_string(index=False))
    print("\nmissing_words =", missing_words)
    if missing_words:
        raise SystemExit(1)


def _markdown_report(target, g2p, alignment, prediction, summary, report) -> str:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        available = [column for column in columns if column in frame.columns]
        if not available:
            return "(empty)"
        rows = [[_markdown_cell(value) for value in row] for row in frame[available].itertuples(index=False, name=None)]
        header = "| " + " | ".join(available) + " |"
        separator = "| " + " | ".join("---" for _ in available) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header, separator, *body])

    return "\n".join(
        [
            "# Sentence coverage report",
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(report, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Target words",
            "",
            table(target, ["word_index", "word", "char_start", "char_end"]),
            "",
            "## G2P phones",
            "",
            table(g2p, ["word_index", "word", "target_phone", "g2p_status", "g2p_error"]),
            "",
            "## Alignment",
            "",
            table(alignment, ["word_index", "word", "target_phone", "alignment_quality", "review_reason"]),
            "",
            "## Prediction coverage",
            "",
            table(prediction, ["word_index", "word", "target_phone", "decision", "alignment_quality"]),
            "",
            "## Word summary",
            "",
            table(summary, ["word_index", "word", "word_decision", "alignment_quality", "error_type"]),
            "",
        ]
    )


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
