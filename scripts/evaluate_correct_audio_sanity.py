#!/usr/bin/env python
"""Run a sanity check on audio known to be correct."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_correct_audio_false_positives import analyze_false_positives


def evaluate_manifest(manifest: Path, output_dir: Path, *, run_predictions: bool = True) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_frame = pd.read_csv(manifest)
    predictions: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for _, row in manifest_frame.iterrows():
        utt_id = str(row["utt_id"])
        audio = Path(str(row["audio"]))
        if not audio.is_absolute():
            audio = (manifest.parent / audio).resolve()
        text = str(row["text"])
        pred_path = output_dir / f"{_safe(utt_id)}_prediction.csv"
        word_path = output_dir / f"{_safe(utt_id)}_word_summary.csv"
        if run_predictions:
            _run_prediction(audio, text, utt_id, pred_path, word_path)
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            pred["utt_id"] = utt_id
            pred["expected_label"] = row.get("expected_label", "correct")
            predictions.append(pred)
        if word_path.exists():
            summary = pd.read_csv(word_path)
            summary["utt_id"] = utt_id
            summary["expected_label"] = row.get("expected_label", "correct")
            summaries.append(summary)

    prediction_all = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    word_summary_all = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    prediction_all_path = output_dir / "prediction_all.csv"
    word_summary_all_path = output_dir / "word_summary_all.csv"
    prediction_all.to_csv(prediction_all_path, index=False, encoding="utf-8-sig")
    word_summary_all.to_csv(word_summary_all_path, index=False, encoding="utf-8-sig")
    report = compute_sanity_metrics(prediction_all)
    false_pos = prediction_all[prediction_all.get("decision", pd.Series(dtype=str)).astype(str).eq("true_error")].copy()
    false_pos.to_csv(output_dir / "false_positive_correct_audio.csv", index=False, encoding="utf-8-sig")
    analyze_false_positives(prediction_all_path, output_dir)
    (output_dir / "correct_audio_sanity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "correct_audio_sanity_report.md").write_text(_report_markdown(report), encoding="utf-8")
    return report


def compute_sanity_metrics(frame: pd.DataFrame) -> dict[str, object]:
    total = int(len(frame))
    true_error = frame.get("decision", pd.Series(dtype=str)).astype(str).eq("true_error")
    uncertain = frame.get("decision", pd.Series(dtype=str)).astype(str).eq("uncertain_review")
    false_pos = frame[true_error].copy()
    return {
        "total_phone_count": total,
        "predicted_true_error_count": int(true_error.sum()),
        "correct_audio_true_error_rate": float(true_error.mean()) if total else 0.0,
        "predicted_uncertain_count": int(uncertain.sum()),
        "correct_audio_uncertain_rate": float(uncertain.mean()) if total else 0.0,
        "per_phone_false_positive_count": _dict_count(false_pos, "target_phone"),
        "per_phone_group_false_positive_count": _dict_count(false_pos, "phone_group"),
        "per_word_false_positive_count": _dict_count(false_pos, "word"),
    }


def _run_prediction(audio: Path, text: str, utt_id: str, pred_path: Path, word_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "predict_pronunciation.py"),
        "--audio",
        str(audio),
        "--text",
        text,
        "--utterance-id",
        utt_id,
        "--decision-mode",
        "conservative",
        "--output",
        str(pred_path),
        "--word-summary-output",
        str(word_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)


def _dict_count(frame: pd.DataFrame, col: str) -> dict[str, int]:
    if frame.empty or col not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame.groupby(col, dropna=False).size().sort_values(ascending=False).items()}


def _report_markdown(report: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Correct Audio Sanity Report",
            "",
            f"- total_phone_count: {report['total_phone_count']}",
            f"- predicted_true_error_count: {report['predicted_true_error_count']}",
            f"- correct_audio_true_error_rate: {report['correct_audio_true_error_rate']:.4f}",
            f"- predicted_uncertain_count: {report['predicted_uncertain_count']}",
            f"- correct_audio_uncertain_rate: {report['correct_audio_uncertain_rate']:.4f}",
            "",
            "See `false_positive_correct_audio.csv` and false-positive breakdown CSV files for details.",
        ]
    )


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:80] or "utt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate false positives on correct-audio sanity set.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/sanity_correct/correct_audio_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sanity_correct")
    parser.add_argument("--skip-predictions", action="store_true")
    args = parser.parse_args()
    report = evaluate_manifest(args.manifest, args.output_dir, run_predictions=not args.skip_predictions)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
