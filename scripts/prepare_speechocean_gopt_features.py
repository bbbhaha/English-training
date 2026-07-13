#!/usr/bin/env python
"""Prepare SpeechOcean762 sequence features for GOPT-style phone score regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_vocab(values: pd.Series, pad_token: str = "<pad>", unk_token: str = "<unk>") -> dict[str, int]:
    vocab = {pad_token: 0, unk_token: 1}
    for value in sorted(values.dropna().astype(str).unique()):
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def compute_duration_norm(df: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    stats = train.groupby("target_phone")["duration"].agg(["mean", "std"]).replace(0, np.nan)
    global_mean = train["duration"].mean()
    global_std = train["duration"].std() or 1.0
    values = []
    for _, row in df.iterrows():
        if row["target_phone"] in stats.index:
            mean = stats.loc[row["target_phone"], "mean"]
            std = stats.loc[row["target_phone"], "std"]
            if pd.isna(std):
                std = global_std
        else:
            mean = global_mean
            std = global_std
        values.append((row["duration"] - mean) / (std if std else 1.0))
    return pd.Series(values, index=df.index)


def pad_2d(sequences: list[np.ndarray], pad_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(seq) for seq in sequences)
    feat_dim = sequences[0].shape[1]
    out = np.full((len(sequences), max_len, feat_dim), pad_value, dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=bool)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = seq
        mask[i, : len(seq)] = True
    return out, mask


def pad_1d(sequences: list[np.ndarray], dtype, pad_value=0) -> np.ndarray:
    max_len = max(len(seq) for seq in sequences)
    out = np.full((len(sequences), max_len), pad_value, dtype=dtype)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = seq
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gopt_speechocean762.yaml")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    input_csv = args.input_csv or PROJECT_ROOT / config["data"]["input_csv"]
    output = args.output or PROJECT_ROOT / config["data"]["feature_npz"]

    df = pd.read_csv(input_csv, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if "alignment_quality" in df.columns and config["data"].get("alignment_quality", "all").lower() != "all":
        df = df[df["alignment_quality"].astype(str).str.lower() == config["data"]["alignment_quality"].lower()].copy()
    df["source_score"] = pd.to_numeric(df[config["data"]["score_column"]], errors="coerce")
    df = df.dropna(subset=["source_score", "target_phone", "phone_group", "duration_ms", "split", "speaker_id"])
    df["duration"] = pd.to_numeric(df["duration_ms"], errors="coerce") / 1000.0
    min_duration = float(config["data"].get("min_duration", 0.0))
    if min_duration > 0:
        df = df[df["duration"] >= min_duration].copy()
    for col in ["gop_score", "target_log_likelihood", "competitor_log_likelihood"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["phone_index"] = pd.to_numeric(df["phone_index"], errors="coerce").fillna(0).astype(int)
    train = df[df["split"] == "train"].copy()
    df["normalized_duration_by_phone"] = compute_duration_norm(df, train)

    phone_vocab = build_vocab(df["target_phone"])
    group_vocab = build_vocab(df["phone_group"])
    numeric_cols = config["features"]["numeric"]

    utterance_rows = []
    numeric_sequences: list[np.ndarray] = []
    phone_sequences: list[np.ndarray] = []
    group_sequences: list[np.ndarray] = []
    position_sequences: list[np.ndarray] = []
    word_sequences: list[np.ndarray] = []
    score_sequences: list[np.ndarray] = []
    word_score_sequences: list[np.ndarray] = []
    binary_sequences: list[np.ndarray] = []

    for utt_id, group in df.sort_values(["utterance_id", "phone_index"]).groupby("utterance_id"):
        numeric = group[numeric_cols].astype(float).to_numpy(dtype=np.float32)
        phone_ids = group["target_phone"].astype(str).map(lambda x: phone_vocab.get(x, 1)).to_numpy(dtype=np.int64)
        group_ids = group["phone_group"].astype(str).map(lambda x: group_vocab.get(x, 1)).to_numpy(dtype=np.int64)
        pos = np.clip(group["phone_index"].to_numpy(dtype=np.int64), 0, int(config["features"]["max_position"]) - 1)
        word_ids = pd.to_numeric(group.get("word_index", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        word_scores = pd.to_numeric(group.get("word_accuracy", np.nan), errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)
        scores = group["source_score"].to_numpy(dtype=np.float32)
        binary = (scores < float(config["evaluation"]["error_threshold_score"])).astype(np.int64)

        numeric_sequences.append(numeric)
        phone_sequences.append(phone_ids)
        group_sequences.append(group_ids)
        position_sequences.append(pos)
        word_sequences.append(word_ids)
        score_sequences.append(scores)
        word_score_sequences.append(word_scores)
        binary_sequences.append(binary)
        utterance_rows.append(
            {
                "utt_id": utt_id,
                "speaker_id": str(group["speaker_id"].iloc[0]),
                "split": str(group["split"].iloc[0]),
                "length": int(len(group)),
            }
        )

    numeric_array, mask = pad_2d(numeric_sequences)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        numeric=numeric_array,
        phone_ids=pad_1d(phone_sequences, np.int64),
        group_ids=pad_1d(group_sequences, np.int64),
        position_ids=pad_1d(position_sequences, np.int64),
        word_ids=pad_1d(word_sequences, np.int64, pad_value=-1),
        scores=pad_1d(score_sequences, np.float32, pad_value=-1.0),
        word_scores=pad_1d(word_score_sequences, np.float32, pad_value=-1.0),
        binary_labels=pad_1d(binary_sequences, np.int64, pad_value=-1),
        mask=mask,
    )
    metadata = pd.DataFrame(utterance_rows)
    metadata.to_csv(output.with_suffix(".metadata.csv"), index=False, encoding="utf-8-sig")
    manifest_cols = [
        "utterance_id",
        "speaker_id",
        "split",
        "word",
        "word_index",
        "word_accuracy",
        "target_phone",
        "phone_group",
        "phone_index",
        "source_score",
        "duration",
        "gop_score",
    ]
    df[manifest_cols].to_csv(output.with_suffix(".phones.csv"), index=False, encoding="utf-8-sig")
    vocab = {
        "phone_vocab": phone_vocab,
        "group_vocab": group_vocab,
        "numeric_columns": numeric_cols,
        "config": config,
    }
    output.with_suffix(".vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(f"Wrote GOPT features to {output}")
    print(f"Utterances: {len(metadata):,}; phones: {len(df):,}")
    print(df.groupby("split")["source_score"].describe().to_string())


if __name__ == "__main__":
    main()
