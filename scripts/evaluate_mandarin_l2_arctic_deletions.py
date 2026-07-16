from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pronunciation.ctc_word_deletion import score_audio_word_deletions
from pronunciation.mandarin_deletion_fusion import DEFAULT_MODEL_PATH
from pronunciation.text_audio_consistency import check_text_audio_consistency
from pronunciation.word_deletion_model import word_deletion_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deletion evidence on Mandarin L2-ARCTIC labels.")
    parser.add_argument("--phones", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mandarin_deletion_validation")
    parser.add_argument("--mandarin-deletion-model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    phones = pd.read_csv(args.phones)
    predictions = []
    for utterance_id, group in phones.groupby("utterance_id", sort=False):
        group = group.sort_values("phone_index").copy()
        group["word_index_eval"] = (
            group["word"].fillna("").ne(group["word"].fillna("").shift()).cumsum() - 1
        )
        gold_by_word = group.groupby("word_index_eval")["error_type"].apply(lambda values: values.eq("deletion").all())
        if not bool(gold_by_word.any()):
            continue
        word_summary = (
            group.groupby("word_index_eval", sort=False)
            .agg(
                word=("word", "first"),
                phone_count=("target_phone", "size"),
                start_ms=("start_ms", "min"),
                end_ms=("end_ms", "max"),
                gold_deletion=("error_type", lambda values: bool(values.eq("deletion").all())),
            )
            .reset_index()
            .rename(columns={"word_index_eval": "word_index"})
        )
        word_summary["word_duration_ms"] = word_summary["end_ms"] - word_summary["start_ms"]
        word_summary["alignment_quality"] = "pass"
        word_summary["lexicon_status"] = "cmudict"
        text = str(group["transcript"].iloc[0])
        audio_path = _resolve_audio(args.audio_root, str(group["audio_path"].iloc[0]))
        ctc = score_audio_word_deletions(audio_path, text, local_files_only=True)
        consistency, meta = check_text_audio_consistency(
            audio_path=audio_path,
            target_text=text,
            asr_model="faster_whisper",
        )
        baseline = word_deletion_detector(word_summary, consistency, ctc)
        diagnosed = word_deletion_detector(
            word_summary,
            consistency,
            ctc,
            mandarin_fusion_model=args.mandarin_deletion_model,
        )
        diagnosed["baseline_deletion_decision"] = baseline["deletion_decision"].to_numpy()
        diagnosed["utterance_id"] = utterance_id
        diagnosed["audio_path"] = str(audio_path)
        diagnosed["asr_transcript"] = str(meta.get("asr_transcript", ""))
        predictions.append(diagnosed)

    result = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "word_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "dataset": "L2-ARCTIC-v5.0 Mandarin speakers",
        "scope": "utterances containing at least one fully deleted word",
        "word_count": int(len(result)),
        "gold_deletion_count": int(result.get("gold_deletion", pd.Series(dtype=bool)).sum()),
        "confirmed_deletion": _metrics(result, {"deletion"}),
        "confirmed_or_possible_deletion": _metrics(result, {"deletion", "possible_deletion"}),
        "baseline_confirmed_deletion": _metrics(
            result,
            {"deletion"},
            decision_column="baseline_deletion_decision",
        ),
        "limitations": "This is a small diagnostic subset, not a corpus-wide benchmark.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _metrics(
    frame: pd.DataFrame,
    positive_decisions: set[str],
    *,
    decision_column: str = "deletion_decision",
) -> dict[str, float | int]:
    if frame.empty:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    predicted = frame[decision_column].isin(positive_decisions)
    gold = frame["gold_deletion"].astype(bool)
    tp = int((predicted & gold).sum())
    fp = int((predicted & ~gold).sum())
    fn = int((~predicted & gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _resolve_audio(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
