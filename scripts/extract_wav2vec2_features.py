#!/usr/bin/env python
"""Extract phone-segment wav2vec2 embeddings for the MDD main model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phoneme_assessment.acoustic import read_wav_mono
from phoneme_assessment.mdd import assert_speaker_isolation


def load_model(model_name: str, local_files_only: bool):
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=local_files_only)
    model = Wav2Vec2Model.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    return torch, extractor, model


def select_manifest(frame: pd.DataFrame, max_utterances_per_split: int) -> pd.DataFrame:
    if max_utterances_per_split <= 0:
        return frame
    parts = []
    for split in ["train", "dev", "test"]:
        split_frame = frame[frame["split"] == split]
        utts = split_frame["utt_id"].drop_duplicates().head(max_utterances_per_split)
        parts.append(split_frame[split_frame["utt_id"].isin(utts)])
    return pd.concat(parts, ignore_index=True)


def hidden_for_audio(torch, extractor, model, wav_path: Path) -> tuple[np.ndarray, float]:
    rate, signal = read_wav_mono(wav_path)
    inputs = extractor(signal, sampling_rate=rate, return_tensors="pt")
    with torch.no_grad():
        hidden = model(inputs.input_values).last_hidden_state[0].cpu().numpy()
    duration = len(signal) / rate
    return hidden, duration


def pool_span(hidden: np.ndarray, start: float, end: float, audio_duration: float) -> np.ndarray:
    start_index = int(np.floor((start / max(audio_duration, 1e-6)) * len(hidden)))
    end_index = int(np.ceil((end / max(audio_duration, 1e-6)) * len(hidden)))
    start_index = max(0, min(start_index, len(hidden) - 1))
    end_index = max(start_index + 1, min(end_index, len(hidden)))
    return hidden[start_index:end_index].mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/processed/mdd/speechocean_manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/mdd_wav2vec2/features.npz")
    parser.add_argument("--model-name", default="facebook/wav2vec2-base")
    parser.add_argument("--max-utterances-per-split", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    manifest = select_manifest(manifest, args.max_utterances_per_split)
    assert_speaker_isolation(manifest)
    print(f"Rows: {len(manifest):,}; utterances: {manifest['utt_id'].nunique():,}")

    torch, extractor, model = load_model(args.model_name, args.local_files_only)
    embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    for index, (wav_path, group) in enumerate(manifest.groupby("wav_path"), start=1):
        hidden, audio_duration = hidden_for_audio(torch, extractor, model, PROJECT_ROOT / wav_path)
        for _, row in group.sort_values(["utt_id", "phone_index"]).iterrows():
            embeddings.append(pool_span(hidden, float(row["start"]), float(row["end"]), audio_duration).astype(np.float32))
            metadata_rows.append(row.to_dict())
        if index % 25 == 0:
            print(f"Processed {index}/{manifest['wav_path'].nunique()} wav files")

    matrix = np.vstack(embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=matrix,
        model_name=np.array([args.model_name]),
    )
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(args.output.with_suffix(".metadata.csv"), index=False, encoding="utf-8-sig")
    print(f"Wrote embeddings {matrix.shape} to {args.output}")
    print(f"Wrote metadata to {args.output.with_suffix('.metadata.csv')}")


if __name__ == "__main__":
    main()
