"""SpeechOcean762 conversion to the unified phase-one phone manifest."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .phones import normalize_phone, phone_group


def read_kaldi_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        result[key] = value.strip()
    return result


def deterministic_dev_speakers(
    speakers: set[str], seed: int, dev_fraction: float = 0.2
) -> set[str]:
    ordered = sorted(
        speakers,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    count = max(1, round(len(ordered) * dev_fraction))
    return set(ordered[:count])


def load_metadata(root: Path, seed: int) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    utterances: dict[str, dict[str, str]] = {}
    official_train_speakers: set[str] = set()

    for official_split in ("train", "test"):
        directory = root / official_split
        utt2spk = read_kaldi_map(directory / "utt2spk")
        wav_scp = read_kaldi_map(directory / "wav.scp")
        text = read_kaldi_map(directory / "text")
        gender = read_kaldi_map(directory / "spk2gender")
        age = read_kaldi_map(directory / "spk2age")
        for utterance_id, speaker_id in utt2spk.items():
            utterances[utterance_id] = {
                "speaker_id": speaker_id,
                "speaker_gender": gender.get(speaker_id, "unknown").upper(),
                "speaker_age": age.get(speaker_id, ""),
                "transcript": text.get(utterance_id, ""),
                "wav": wav_scp[utterance_id].replace("\\", "/"),
                "official_split": official_split,
            }
        if official_split == "train":
            official_train_speakers.update(utt2spk.values())

    dev_speakers = deterministic_dev_speakers(official_train_speakers, seed)
    speaker_splits = {
        metadata["speaker_id"]: (
            "test"
            if metadata["official_split"] == "test"
            else "dev"
            if metadata["speaker_id"] in dev_speakers
            else "train"
        )
        for metadata in utterances.values()
    }
    return utterances, speaker_splits


def convert(root: Path, project_root: Path, seed: int) -> tuple[list[dict], dict[str, str]]:
    scores = json.loads((root / "resource/scores.json").read_text(encoding="utf-8"))
    metadata, speaker_splits = load_metadata(root, seed)
    rows: list[dict] = []

    for utterance_id, item in scores.items():
        meta = metadata.get(utterance_id)
        if meta is None:
            raise ValueError(f"Score entry {utterance_id} has no Kaldi metadata")
        phone_index = 0
        for word_index, word in enumerate(item["words"]):
            phones = word["phones"]
            accuracies = word["phones-accuracy"]
            if len(phones) != len(accuracies):
                raise ValueError(
                    f"{utterance_id} word {word_index}: phone/score length mismatch"
                )
            for raw_phone, raw_score in zip(phones, accuracies):
                score = float(raw_score)
                target = normalize_phone(raw_phone)
                binary = int(score >= 1.0)
                attention = int(score < 1.8)
                three_class = (
                    "correct"
                    if score >= 1.8
                    else "acceptable"
                    if score >= 1.0
                    else "incorrect"
                )
                audio_path = root / meta["wav"]
                rows.append(
                    {
                        "utterance_id": f"speechocean_{utterance_id}",
                        "speaker_id": meta["speaker_id"],
                        "speaker_gender": meta["speaker_gender"],
                        "speaker_age": meta["speaker_age"],
                        "native_language": "Mandarin",
                        "transcript": item.get("text", meta["transcript"]),
                        "sentence_accuracy": item.get("accuracy", ""),
                        "sentence_fluency": item.get("fluency", ""),
                        "sentence_completeness": item.get("completeness", ""),
                        "sentence_prosodic": item.get("prosodic", ""),
                        "word": word["text"],
                        "word_index": word_index,
                        "word_accuracy": word.get("accuracy", ""),
                        "word_stress": word.get("stress", ""),
                        "target_phone_raw": raw_phone,
                        "target_phone": target,
                        "perceived_phone_raw": "",
                        "perceived_phone": "",
                        "phone_index": phone_index,
                        "start_ms": "",
                        "end_ms": "",
                        "duration_ms": "",
                        "source_score": score,
                        "gold_binary": binary,
                        "attention_binary": attention,
                        "gold_three_class": three_class,
                        "error_type": "correct" if binary else "score_error",
                        "phone_group": phone_group(target),
                        "dataset_source": "SpeechOcean762",
                        "split": speaker_splits[meta["speaker_id"]],
                        "official_split": meta["official_split"],
                        "audio_path": audio_path.relative_to(project_root).as_posix(),
                        "annotation_path": "data/raw/speechocean762/resource/scores.json",
                    }
                )
                phone_index += 1
    return rows, speaker_splits


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

