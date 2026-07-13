"""Utilities for phone-level mispronunciation detection (MDD)."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .phones import normalize_phone, phone_group


MDD_COLUMNS = [
    "utt_id",
    "speaker_id",
    "wav_path",
    "word",
    "target_phone",
    "phone_group",
    "start",
    "end",
    "duration",
    "label",
    "split",
    "dataset_source",
    "phone_index",
    "gop_score",
]


def source_to_mdd_manifest(
    frame: pd.DataFrame,
    *,
    project_root: Path | None = None,
    alignment_quality: str | None = None,
) -> pd.DataFrame:
    """Convert an existing project phone manifest to MDD format.

    The project historically uses ``gold_binary=1`` for correct/acceptable and
    ``gold_binary=0`` for incorrect.  The MDD pipeline uses the paper-style
    positive class: ``label=1`` means mispronounced and ``label=0`` means
    correct.
    """

    df = frame.copy()
    if alignment_quality and "alignment_quality" in df.columns:
        df = df[df["alignment_quality"].astype(str).str.lower() == alignment_quality.lower()].copy()
    required = ["utterance_id", "speaker_id", "audio_path", "target_phone", "start_ms", "end_ms", "gold_binary", "split"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input manifest is missing required columns: {missing}")

    out = pd.DataFrame()
    out["utt_id"] = df["utterance_id"].astype(str)
    out["speaker_id"] = df["speaker_id"].astype(str)
    out["wav_path"] = df["audio_path"].astype(str)
    out["word"] = df["word"].astype(str) if "word" in df.columns else ""
    out["target_phone"] = df["target_phone"].map(normalize_phone)
    out["phone_group"] = out["target_phone"].map(phone_group)
    out["start"] = pd.to_numeric(df["start_ms"], errors="coerce") / 1000.0
    out["end"] = pd.to_numeric(df["end_ms"], errors="coerce") / 1000.0
    out["duration"] = out["end"] - out["start"]
    out["label"] = 1 - pd.to_numeric(df["gold_binary"], errors="coerce").fillna(1).astype(int)
    out["split"] = df["split"].astype(str)
    out["dataset_source"] = df["dataset_source"].astype(str) if "dataset_source" in df.columns else ""
    if "phone_index" in df.columns:
        out["phone_index"] = pd.to_numeric(df["phone_index"], errors="coerce").fillna(0).astype(int)
    else:
        out["phone_index"] = np.arange(len(out), dtype=int)
    out["gop_score"] = pd.to_numeric(df["gop_score"], errors="coerce") if "gop_score" in df.columns else np.nan
    out = out[MDD_COLUMNS]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["start", "end", "duration", "label"])
    out = out[(out["duration"] > 0) & (out["target_phone"] != "")]
    return out.reset_index(drop=True)


def assert_speaker_isolation(frame: pd.DataFrame) -> None:
    """Raise if any speaker appears in more than one split."""

    split_by_speaker = frame.groupby("speaker_id")["split"].nunique()
    offenders = split_by_speaker[split_by_speaker > 1]
    if not offenders.empty:
        raise ValueError(f"Speakers appear in multiple splits: {offenders.index.tolist()[:10]}")


def parse_phone_sequence(value: str) -> list[str]:
    return [normalize_phone(item) for item in re.split(r"[\s,]+", value.strip()) if normalize_phone(item)]
