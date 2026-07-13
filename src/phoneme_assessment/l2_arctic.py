"""L2-ARCTIC manual annotation conversion."""

from __future__ import annotations

from pathlib import Path
import csv

from .phones import SILENCES, normalize_phone, phone_group
from .textgrid import read_interval_tiers, word_at_interval

SPEAKER_GENDER = {"BWC": "M", "LXC": "F", "NCC": "F", "TXHC": "M"}
ERROR_CODES = {"s": "substitution", "d": "deletion", "a": "addition"}


def parse_phone_label(label: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in label.split(",")]
    if len(parts) >= 3 and parts[-1].lower() in ERROR_CODES:
        target, perceived, code = parts[0], parts[1], parts[-1].lower()
        return target, perceived, ERROR_CODES[code]
    return label.strip(), label.strip(), "correct"


def annotation_rows(
    annotation_path: Path,
    corpus_root: Path,
    speaker_id: str,
    split: str,
    project_root: Path,
) -> list[dict[str, object]]:
    tiers = read_interval_tiers(annotation_path)
    phones = tiers.get("phones", [])
    words = tiers.get("words", [])
    utterance_stem = annotation_path.stem
    audio_path = corpus_root / speaker_id / "wav" / f"{utterance_stem}.wav"
    transcript_path = corpus_root / speaker_id / "transcript" / f"{utterance_stem}.txt"
    transcript = (
        transcript_path.read_text(encoding="utf-8").strip()
        if transcript_path.is_file()
        else ""
    )
    rows: list[dict[str, object]] = []

    for phone in phones:
        target_raw, perceived_raw, error_type = parse_phone_label(phone.text)
        target = normalize_phone(target_raw)
        perceived = normalize_phone(perceived_raw)
        if target in SILENCES and error_type != "addition":
            continue
        gold_binary = 1 if error_type == "correct" else 0
        rows.append(
            {
                "utterance_id": f"l2arctic_{speaker_id}_{utterance_stem}",
                "speaker_id": speaker_id,
                "speaker_gender": SPEAKER_GENDER.get(speaker_id, "unknown"),
                "speaker_age": "",
                "native_language": "Mandarin",
                "transcript": transcript,
                "sentence_accuracy": "",
                "sentence_fluency": "",
                "sentence_completeness": "",
                "sentence_prosodic": "",
                "word": word_at_interval(words, phone),
                "word_index": "",
                "word_accuracy": "",
                "word_stress": "",
                "target_phone_raw": target_raw,
                "target_phone": target,
                "perceived_phone_raw": perceived_raw,
                "perceived_phone": perceived,
                "phone_index": len(rows),
                "start_ms": round(phone.start * 1000, 3),
                "end_ms": round(phone.end * 1000, 3),
                "duration_ms": round((phone.end - phone.start) * 1000, 3),
                "source_score": "",
                "gold_binary": gold_binary,
                "attention_binary": "",
                "gold_three_class": "correct" if gold_binary else "incorrect",
                "error_type": error_type,
                "phone_group": phone_group(target),
                "dataset_source": "L2-ARCTIC-v5.0-Mandarin",
                "split": split,
                "official_split": "",
                "audio_path": audio_path.relative_to(project_root).as_posix(),
                "annotation_path": annotation_path.relative_to(project_root).as_posix(),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows were produced")
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
