#!/usr/bin/env python
"""Systematic diagnostics for MDD data labels, alignment, and model scores."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phoneme_assessment.acoustic import read_wav_mono
from phoneme_assessment.mdd import assert_speaker_isolation


def load_feature_frame(features_path: Path, min_duration: float = 0.0) -> pd.DataFrame:
    matrix = np.load(features_path)["embeddings"]
    metadata = pd.read_csv(features_path.with_suffix(".metadata.csv"), encoding="utf-8-sig", keep_default_na=False)
    if min_duration > 0:
        duration = pd.to_numeric(metadata["duration"], errors="coerce")
        keep = duration >= min_duration
        metadata = metadata.loc[keep].reset_index(drop=True)
        matrix = matrix[keep.to_numpy()]
    feature_columns = pd.DataFrame(matrix, columns=[f"w2v_{i}" for i in range(matrix.shape[1])])
    frame = pd.concat([metadata.reset_index(drop=True), feature_columns], axis=1)
    frame["label"] = frame["label"].astype(int)
    return frame


def add_model_scores(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    if "normalized_duration_by_phone" in bundle.get("numeric_features", []):
        frame = apply_duration_stats(frame, bundle.get("duration_stats_by_phone", {}))
    pipeline = bundle["pipeline"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    for column in numeric:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in categorical:
        if column not in frame.columns:
            frame[column] = ""
    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    out = frame.copy()
    out["score"] = pipeline.predict_proba(out[numeric + categorical])[:, positive_index]
    return out


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


def count_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    table = (
        frame.groupby(group_cols + ["label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "correct_count", 1: "mispronounced_count"})
        .reset_index()
    )
    if "correct_count" not in table.columns:
        table["correct_count"] = 0
    if "mispronounced_count" not in table.columns:
        table["mispronounced_count"] = 0
    table["total"] = table["correct_count"] + table["mispronounced_count"]
    table["mispronounced_rate"] = table["mispronounced_count"] / table["total"].replace(0, np.nan)
    return table


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": None if threshold is None else float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def best_recall_at_precision_threshold(y_true: np.ndarray, scores: np.ndarray, min_precision: float) -> float | None:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return None
    index = valid[np.argmax(recall[:-1][valid])]
    return float(thresholds[index])


def write_score_outputs(frame: pd.DataFrame, output_dir: Path, min_precision: float) -> dict[str, object]:
    y = frame["label"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    pr = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds})
    pr.to_csv(output_dir / "pr_curve.csv", index=False, encoding="utf-8-sig")
    pr_auc = float(auc(recall, precision))

    bins = np.linspace(0.0, 1.0, 51)
    hist_rows = []
    for label, name in [(0, "correct"), (1, "mispronounced")]:
        values = frame.loc[frame["label"] == label, "score"].to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=bins)
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            hist_rows.append({"label": name, "bin_left": left, "bin_right": right, "count": int(count)})
    pd.DataFrame(hist_rows).to_csv(output_dir / "score_histogram_correct_vs_error.csv", index=False, encoding="utf-8-sig")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(frame.loc[frame["label"] == 0, "score"], bins=bins, alpha=0.6, label="correct")
        plt.hist(frame.loc[frame["label"] == 1, "score"], bins=bins, alpha=0.6, label="mispronounced")
        plt.xlabel("mispronounced score")
        plt.ylabel("count")
        plt.title("Score histogram: correct vs mispronounced")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "score_histogram_correct_vs_error.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR curve (AUPRC={pr_auc:.4f})")
        plt.tight_layout()
        plt.savefig(output_dir / "pr_curve.png", dpi=160)
        plt.close()
    except Exception as exc:  # plotting is a diagnostic convenience
        (output_dir / "plot_warning.txt").write_text(str(exc), encoding="utf-8")

    threshold_05 = 0.5
    threshold_f1 = best_f1_threshold(y, scores)
    threshold_precision = best_recall_at_precision_threshold(y, scores, min_precision)
    matrices = {
        "threshold_0_5": metrics_at_threshold(y, scores, threshold_05),
        "best_f1_threshold": metrics_at_threshold(y, scores, threshold_f1),
        f"max_recall_at_precision_{min_precision}": (
            {"found": False, "min_precision": min_precision}
            if threshold_precision is None
            else {"found": True, "min_precision": min_precision, **metrics_at_threshold(y, scores, threshold_precision)}
        ),
        "auprc": pr_auc,
    }
    (output_dir / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2), encoding="utf-8")
    return matrices


def save_segment_wav(signal: np.ndarray, rate: int, start: float, end: float, path: Path) -> None:
    start_sample = max(0, int(round(start * rate)))
    end_sample = min(len(signal), int(round(end * rate)))
    segment = signal[start_sample:end_sample]
    data = np.clip(segment * 32767.0, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


def write_sanity_segments(frame: pd.DataFrame, output_dir: Path, save_wavs: bool, seed: int) -> None:
    sample = frame.sample(n=min(20, len(frame)), random_state=seed).copy()
    sample_columns = ["utt_id", "speaker_id", "word", "target_phone", "phone_group", "start", "end", "duration", "label", "wav_path"]
    sample[sample_columns].to_csv(output_dir / "alignment_sanity_sample_20.csv", index=False, encoding="utf-8-sig")
    if not save_wavs:
        return
    for idx, row in sample.reset_index(drop=True).iterrows():
        wav_path = Path(str(row["wav_path"]))
        full_path = wav_path if wav_path.is_absolute() else PROJECT_ROOT / wav_path
        rate, signal = read_wav_mono(full_path)
        out = output_dir / "segment_wavs" / f"{idx:02d}_{row['utt_id']}_{row['target_phone']}_{row['label']}.wav"
        save_segment_wav(signal, rate, float(row["start"]), float(row["end"]), out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/processed/mdd/speechocean_manifest.csv")
    parser.add_argument("--features", type=Path, help="Optional features.npz for model score diagnostics.")
    parser.add_argument("--model", type=Path, help="Optional trained MDD model.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--min-precision", type=float, default=0.40)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/mdd_diagnostics")
    parser.add_argument("--save-segment-wavs", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    manifest["label"] = manifest["label"].astype(int)
    assert_speaker_isolation(manifest)
    if set(manifest["label"].unique()) - {0, 1}:
        raise SystemExit("MDD label must be 0 correct / 1 mispronounced.")

    data_report = {
        "label_definition": "label=1 mispronounced/error; label=0 correct/acceptable",
        "rows": int(len(manifest)),
        "splits": manifest["split"].value_counts().to_dict(),
        "labels": {str(k): int(v) for k, v in manifest["label"].value_counts().to_dict().items()},
        "start_end_unit": "seconds",
        "invalid_end_le_start": int((pd.to_numeric(manifest["end"]) <= pd.to_numeric(manifest["start"])).sum()),
        "duration_lt_0_03": int((pd.to_numeric(manifest["duration"]) < 0.03).sum()),
        "max_duration_seconds": float(pd.to_numeric(manifest["duration"]).max()),
    }
    (args.output_dir / "data_validation.json").write_text(json.dumps(data_report, indent=2), encoding="utf-8")

    count_table(manifest, ["split"]).to_csv(args.output_dir / "counts_by_split.csv", index=False, encoding="utf-8-sig")
    count_table(manifest, ["target_phone"]).to_csv(args.output_dir / "counts_by_target_phone.csv", index=False, encoding="utf-8-sig")
    count_table(manifest, ["speaker_id"]).to_csv(args.output_dir / "counts_by_speaker.csv", index=False, encoding="utf-8-sig")
    count_table(manifest, ["phone_group"]).to_csv(args.output_dir / "counts_by_phone_group.csv", index=False, encoding="utf-8-sig")
    write_sanity_segments(manifest, args.output_dir, args.save_segment_wavs, args.seed)

    score_report = None
    if args.features and args.model:
        frame = load_feature_frame(args.features, args.min_duration)
        split_frame = frame[frame["split"] == args.split].copy()
        if split_frame.empty:
            raise SystemExit(f"No rows found for split={args.split!r}")
        scored = add_model_scores(split_frame, args.model)
        keep = ["utt_id", "speaker_id", "wav_path", "word", "target_phone", "phone_group", "start", "end", "duration", "label", "split", "score"]
        scored[keep].to_csv(args.output_dir / f"scores_{args.split}.csv", index=False, encoding="utf-8-sig")
        score_report = write_score_outputs(scored, args.output_dir, args.min_precision)

    summary = {"data": data_report, "score_diagnostics": score_report}
    (args.output_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
