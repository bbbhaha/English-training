from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pronunciation.mandarin_deletion_fusion import DEFAULT_MODEL_PATH
from pronunciation.word_deletion_model import word_deletion_detector
from scripts.train_mandarin_deletion_fusion import _score_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real paired word-deletion recordings.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/manual_deletion_pairs")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    frames: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        audio_path = Path(row.audio)
        if not audio_path.is_absolute():
            audio_path = args.project_root / audio_path
        deleted_word_index = int(row.deleted_word_index)
        print(f"Scoring {row.utterance_id}: {audio_path.name}", flush=True)
        evidence = _score_variant(
            audio_path=audio_path,
            transcript=str(row.text),
            speaker_id="manual_pair_speaker",
            split=str(row.split),
            utterance_id=str(row.utterance_id),
            variant_id=str(row.expected_label),
            deleted_word_index=deleted_word_index if deleted_word_index >= 0 else None,
        )
        diagnosed = word_deletion_detector(
            evidence,
            mandarin_fusion_model=args.model,
        )
        diagnosed["baseline_deletion_decision"] = evidence["baseline_deletion_decision"].to_numpy()
        diagnosed["expected_label"] = str(row.expected_label)
        diagnosed["expected_deleted_word"] = str(row.deleted_word)
        diagnosed["manifest_split"] = str(row.split)
        frames.append(diagnosed)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    report = build_report(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "word_predictions.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_report(frame: pd.DataFrame) -> dict[str, Any]:
    target_rows = frame.loc[frame["gold_deletion"].astype(bool)].copy()
    correct_file_rows = frame.loc[frame["expected_label"].eq("correct")].copy()
    return {
        "dataset": "user-recorded real paired deletion set",
        "files": int(frame["utterance_id"].nunique()),
        "deletion_files": int(frame.loc[frame["expected_label"].eq("deletion"), "utterance_id"].nunique()),
        "word_rows": int(len(frame)),
        "gold_deletions": int(frame["gold_deletion"].sum()),
        "fusion_confirmed": _metrics(frame, "deletion_decision", {"deletion"}),
        "fusion_confirmed_or_possible": _metrics(
            frame,
            "deletion_decision",
            {"deletion", "possible_deletion"},
        ),
        "baseline_confirmed": _metrics(frame, "baseline_deletion_decision", {"deletion"}),
        "correct_recording": {
            "words": int(len(correct_file_rows)),
            "confirmed_false_deletions": int(correct_file_rows["deletion_decision"].eq("deletion").sum()),
            "possible_false_deletions": int(correct_file_rows["deletion_decision"].eq("possible_deletion").sum()),
        },
        "target_word_results": [
            {
                "utterance_id": str(row["utterance_id"]),
                "word": str(row["word"]),
                "split": str(row["manifest_split"]),
                "baseline_decision": str(row["baseline_deletion_decision"]),
                "fusion_decision": str(row["deletion_decision"]),
                "fusion_probability": _number(row.get("mandarin_deletion_probability")),
                "asr_missing_word": bool(row.get("asr_missing_word", False)),
                "ctc_deletion_score": _number(row.get("ctc_deletion_score")),
                "ctc_greedy_missing_word": bool(row.get("ctc_greedy_missing_word", False)),
            }
            for _, row in target_rows.sort_values("utterance_id").iterrows()
        ],
        "limitations": (
            "All recordings appear to come from one speaker and one sentence. File-level train/test isolation "
            "does not provide speaker-independent generalization evidence."
        ),
    }


def _metrics(frame: pd.DataFrame, column: str, positive: set[str]) -> dict[str, float | int]:
    predicted = frame[column].fillna("").astype(str).isin(positive)
    gold = frame["gold_deletion"].astype(bool)
    tp = int((predicted & gold).sum())
    fp = int((predicted & ~gold).sum())
    fn = int((~predicted & gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _number(value: object) -> float | None:
    try:
        return None if pd.isna(value) else round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _markdown(report: dict[str, Any]) -> str:
    confirmed = report["fusion_confirmed"]
    broad = report["fusion_confirmed_or_possible"]
    lines = [
        "# Real paired deletion evaluation",
        "",
        f"Files: {report['files']}; deletion files: {report['deletion_files']}; gold deletions: {report['gold_deletions']}",
        "",
        "| Output | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
        f"| Confirmed deletion | {confirmed['precision']:.3f} | {confirmed['recall']:.3f} | {confirmed['f1']:.3f} |",
        f"| Confirmed + possible | {broad['precision']:.3f} | {broad['recall']:.3f} | {broad['f1']:.3f} |",
        "",
        "| File | Target word | Split | Baseline | Fusion | Probability |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for row in report["target_word_results"]:
        lines.append(
            f"| {row['utterance_id']} | {row['word']} | {row['split']} | "
            f"{row['baseline_decision']} | {row['fusion_decision']} | {row['fusion_probability'] or 0.0:.3f} |"
        )
    lines.extend(["", f"Limitation: {report['limitations']}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
