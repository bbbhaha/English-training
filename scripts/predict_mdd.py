#!/usr/bin/env python
"""Predict phone-level mispronunciation probabilities with the wav2vec2-MDD model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phoneme_assessment.acoustic import read_wav_mono
from phoneme_assessment.mdd import MDD_COLUMNS, parse_phone_sequence
from phoneme_assessment.phones import phone_group


def load_wav2vec2(model_name: str, local_files_only: bool):
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=local_files_only)
    model = Wav2Vec2Model.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    return torch, extractor, model


def hidden_for_audio(torch, extractor, model, wav_path: Path) -> tuple[np.ndarray, float]:
    rate, signal = read_wav_mono(wav_path)
    inputs = extractor(signal, sampling_rate=rate, return_tensors="pt")
    with torch.no_grad():
        hidden = model(inputs.input_values).last_hidden_state[0].cpu().numpy()
    return hidden, len(signal) / rate


def pool_span(hidden: np.ndarray, start: float, end: float, duration: float) -> np.ndarray:
    start_index = int(np.floor((start / max(duration, 1e-6)) * len(hidden)))
    end_index = int(np.ceil((end / max(duration, 1e-6)) * len(hidden)))
    start_index = max(0, min(start_index, len(hidden) - 1))
    end_index = max(start_index + 1, min(end_index, len(hidden)))
    return hidden[start_index:end_index].mean(axis=0)


def manifest_from_alignment(args: argparse.Namespace) -> pd.DataFrame:
    if args.manifest:
        return pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False)
    if not args.wav_path or not args.alignment_csv:
        raise SystemExit("Provide --manifest, or provide --wav-path plus --alignment-csv.")
    alignment = pd.read_csv(args.alignment_csv, encoding="utf-8-sig", keep_default_na=False)
    required = ["target_phone", "start", "end"]
    missing = [c for c in required if c not in alignment.columns]
    if missing:
        raise SystemExit(f"Alignment CSV missing columns: {missing}")
    rows = []
    for index, row in alignment.iterrows():
        phone = str(row["target_phone"]).strip().upper()
        rows.append(
            {
                "utt_id": args.utt_id or Path(args.wav_path).stem,
                "speaker_id": args.speaker_id,
                "wav_path": str(Path(args.wav_path)),
                "word": row.get("word", ""),
                "target_phone": phone,
                "phone_group": phone_group(phone),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "duration": float(row["end"]) - float(row["start"]),
                "label": -1,
                "split": "predict",
                "dataset_source": "user_input",
                "phone_index": int(row.get("phone_index", index)),
                "gop_score": np.nan,
            }
        )
    return pd.DataFrame(rows, columns=MDD_COLUMNS)


def add_embeddings(frame: pd.DataFrame, wav2vec2_model: str, local_files_only: bool) -> pd.DataFrame:
    torch, extractor, model = load_wav2vec2(wav2vec2_model, local_files_only)
    chunks = []
    for wav_path, group in frame.groupby("wav_path"):
        path = Path(wav_path)
        full_path = path if path.is_absolute() else PROJECT_ROOT / path
        hidden, audio_duration = hidden_for_audio(torch, extractor, model, full_path)
        embeddings = []
        for _, row in group.iterrows():
            embeddings.append(pool_span(hidden, float(row["start"]), float(row["end"]), audio_duration).astype(np.float32))
        emb = np.vstack(embeddings)
        emb_frame = pd.DataFrame(emb, columns=[f"w2v_{i}" for i in range(emb.shape[1])], index=group.index)
        chunks.append(pd.concat([group.reset_index(drop=True), emb_frame.reset_index(drop=True)], axis=1))
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/mdd_wav2vec2/mdd_classifier.joblib"))
    parser.add_argument("--manifest", type=Path, help="MDD manifest rows to predict.")
    parser.add_argument("--wav-path", help="Single wav path; requires --alignment-csv.")
    parser.add_argument("--alignment-csv", type=Path, help="CSV with target_phone,start,end in seconds.")
    parser.add_argument("--utt-id")
    parser.add_argument("--speaker-id", default="unknown")
    parser.add_argument("--wav2vec2-model", default="facebook/wav2vec2-base")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/mdd_wav2vec2/predict_output.csv"))
    args = parser.parse_args()

    rows = manifest_from_alignment(args)
    feature_frame = add_embeddings(rows, args.wav2vec2_model, args.local_files_only)
    bundle = joblib.load(args.model)
    if "normalized_duration_by_phone" in bundle.get("numeric_features", []):
        feature_frame = apply_duration_stats(feature_frame, bundle.get("duration_stats_by_phone", {}))
    pipeline = bundle["pipeline"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    for col in numeric:
        if col not in feature_frame.columns:
            feature_frame[col] = np.nan
    for col in categorical:
        if col not in feature_frame.columns:
            feature_frame[col] = ""
    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    scores = pipeline.predict_proba(feature_frame[numeric + categorical])[:, positive_index]
    out = rows.copy()
    out["mispronounced_probability"] = scores
    out["prediction"] = (scores >= args.threshold).astype(int)
    out["prediction_label"] = np.where(out["prediction"] == 1, "mispronounced", "correct")
    out["threshold"] = args.threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote MDD predictions to {args.output}")


def apply_duration_stats(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = frame.copy()
    durations = pd.to_numeric(out["duration"], errors="coerce")
    global_mean = float(durations.mean())
    global_std = float(durations.std(ddof=0)) or 1.0
    values = []
    for _, row in out.iterrows():
        item = stats.get(str(row["target_phone"]), {"mean": global_mean, "std": global_std})
        values.append((float(row["duration"]) - item["mean"]) / item["std"])
    out["normalized_duration_by_phone"] = values
    return out


if __name__ == "__main__":
    main()
