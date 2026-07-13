#!/usr/bin/env python
"""Generate high-recall candidate mispronunciation phones for cascade MDD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def add_duration_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    train = out[out["split"] == "train"].copy()
    stats = train.groupby("target_phone")["duration"].agg(["mean", "std"]).replace(0, np.nan)
    global_mean = float(train["duration"].mean())
    global_std = float(train["duration"].std()) or 1.0
    vals = []
    for _, row in out.iterrows():
        if row["target_phone"] in stats.index:
            mean = float(stats.loc[row["target_phone"], "mean"])
            std = float(stats.loc[row["target_phone"], "std"])
            if np.isnan(std):
                std = global_std
        else:
            mean, std = global_mean, global_std
        vals.append((float(row["duration"]) - mean) / (std if std else 1.0))
    out["normalized_duration_by_phone"] = vals
    return out


def load_base_frame(features_npz: Path, scored_csv: Path, min_duration: float) -> pd.DataFrame:
    meta = pd.read_csv(features_npz.with_suffix(".metadata.csv"), encoding="utf-8-sig", keep_default_na=False)
    if "gop_score" in meta.columns:
        meta = meta.drop(columns=["gop_score"])
    scored = pd.read_csv(scored_csv, encoding="utf-8-sig", low_memory=False)
    scored = scored.rename(
        columns={
            "utterance_id": "utt_id",
            "audio_path": "wav_path",
            "start_ms": "start_ms_scored",
            "end_ms": "end_ms_scored",
        }
    )
    cols = [
        "utt_id",
        "speaker_id",
        "target_phone",
        "phone_index",
        "gop_score",
        "target_log_likelihood",
        "competitor_log_likelihood",
        "model_probability",
        "confidence",
        "source_score",
        "word_index",
        "word_accuracy",
    ]
    scored = scored[[c for c in cols if c in scored.columns]].copy()
    key = ["utt_id", "speaker_id", "target_phone", "phone_index"]
    for frame in [meta, scored]:
        frame["utt_id"] = frame["utt_id"].astype(str)
        frame["speaker_id"] = frame["speaker_id"].astype(str)
        frame["target_phone"] = frame["target_phone"].astype(str)
        frame["phone_index"] = pd.to_numeric(frame["phone_index"], errors="coerce").fillna(-1).astype(int)
    df = meta.merge(scored, on=key, how="left", suffixes=("", "_scored"))
    df["gold_label"] = df["label"].astype(int)
    for col in ["gop_score", "target_log_likelihood", "competitor_log_likelihood", "model_probability", "confidence"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df[df["duration"] >= min_duration].copy()
    df = add_duration_norm(df)
    return df.reset_index(drop=True)


def score_candidates(df: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "gop_aggressive":
        values = pd.to_numeric(df["gop_score"], errors="coerce")
        fill = float(values.median()) if not np.isnan(values.median()) else 0.0
        return -values.fillna(fill)
    if strategy == "gop_margin":
        margin = pd.to_numeric(df["competitor_log_likelihood"], errors="coerce") - pd.to_numeric(df["target_log_likelihood"], errors="coerce")
        fill = float(margin.median()) if not np.isnan(margin.median()) else 0.0
        return margin.fillna(fill)
    raise ValueError(f"Unsupported candidate strategy for first implementation: {strategy}")


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }


def choose_high_recall_threshold(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid):
        # Among thresholds reaching target recall, keep the one with best precision.
        idx = valid[np.argmax(precision[:-1][valid])]
        return {"target_recall_met": True, **metrics_at_threshold(labels, scores, float(thresholds[idx]))}
    idx = int(np.argmax(recall[:-1]))
    return {"target_recall_met": False, **metrics_at_threshold(labels, scores, float(thresholds[idx]))}


def write_split(df: pd.DataFrame, split: str, threshold: float, strategy: str, output_dir: Path) -> dict:
    part = df[df["split"] == split].copy()
    part["candidate_label"] = (part["candidate_score"] >= threshold).astype(int)
    part["candidate_source"] = strategy
    cols = [
        "utt_id",
        "speaker_id",
        "wav_path",
        "word",
        "target_phone",
        "phone_group",
        "start",
        "end",
        "duration",
        "normalized_duration_by_phone",
        "gold_label",
        "candidate_source",
        "gop_score",
        "candidate_score",
        "candidate_label",
        "phone_index",
        "split",
    ]
    if "source_score" in part.columns:
        cols.append("source_score")
    path = output_dir / f"mdd_candidates_{split}.csv"
    part[[c for c in cols if c in part.columns]].to_csv(path, index=False, encoding="utf-8-sig")
    labels = part["gold_label"].to_numpy(int)
    return {"split": split, "rows": int(len(part)), "candidate_rows": int(part["candidate_label"].sum()), **metrics_at_threshold(labels, part["candidate_score"].to_numpy(float), threshold)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/candidate_verifier.yaml")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--scored-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--target-recall", type=float, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    features = args.features or PROJECT_ROOT / cfg["data"]["features_npz"]
    scored_csv = args.scored_csv or PROJECT_ROOT / cfg["data"]["scored_csv"]
    strategy = args.strategy or cfg["candidate_generator"]["strategy"]
    target_recall = args.target_recall if args.target_recall is not None else float(cfg["candidate_generator"]["target_recall"])
    min_duration = args.min_duration if args.min_duration is not None else float(cfg["candidate_generator"]["min_duration"])
    df = load_base_frame(features, scored_csv, min_duration)
    df["candidate_score"] = score_candidates(df, strategy)
    dev = df[df["split"] == "dev"]
    selected = choose_high_recall_threshold(dev["gold_label"].to_numpy(int), dev["candidate_score"].to_numpy(float), target_recall)
    threshold = float(selected["threshold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [{"split": "dev_threshold_selection", "strategy": strategy, "target_recall": target_recall, **selected}]
    for split in ["train", "dev", "test"]:
        row = write_split(df, split, threshold, strategy, args.output_dir)
        summaries.append({"strategy": strategy, **row})
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "candidate_generator_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "candidate_generator_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
