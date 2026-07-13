#!/usr/bin/env python
"""Self-supervised speech representation baseline for phase 1.

The script extracts frozen wav2vec2 hidden states for each utterance, pools the
states over aligned phone spans, then trains a lightweight classifier for phone
correctness.  It is designed to be reproducible on CPU; use --max-utterances-per-split
for a quick acceptance/pilot run and omit it for a full run.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.acoustic import read_wav_mono


META_CATEGORICAL = ["target_phone", "phone_group"]
META_NUMERIC = ["duration_ms", "phone_index"]


def load_ssl_model(model_name: str, local_files_only: bool = False):
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=local_files_only)
    model = Wav2Vec2Model.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    return torch, extractor, model


def select_rows(df: pd.DataFrame, max_utterances_per_split: int | None) -> pd.DataFrame:
    if max_utterances_per_split is None or max_utterances_per_split <= 0:
        return df
    parts = []
    for split in ["train", "dev", "test"]:
        split_df = df[df["split"] == split]
        utterances = split_df["utterance_id"].drop_duplicates().head(max_utterances_per_split)
        parts.append(split_df[split_df["utterance_id"].isin(utterances)])
    return pd.concat(parts, ignore_index=True)


def hidden_state_for_audio(torch, extractor, model, audio_path: Path):
    rate, signal = read_wav_mono(audio_path)
    inputs = extractor(signal, sampling_rate=rate, return_tensors="pt")
    with torch.no_grad():
        hidden = model(inputs.input_values).last_hidden_state[0].cpu().numpy()
    return hidden


def pool_phone_embedding(hidden: np.ndarray, start_ms: float, end_ms: float, audio_ms: float) -> np.ndarray:
    if len(hidden) == 0:
        raise ValueError("Empty hidden state")
    start_index = int(np.floor((start_ms / max(audio_ms, 1.0)) * len(hidden)))
    end_index = int(np.ceil((end_ms / max(audio_ms, 1.0)) * len(hidden)))
    start_index = max(0, min(start_index, len(hidden) - 1))
    end_index = max(start_index + 1, min(end_index, len(hidden)))
    segment = hidden[start_index:end_index]
    return segment.mean(axis=0)


def extract_embeddings(df: pd.DataFrame, model_name: str, output_npz: Path, local_files_only: bool) -> tuple[np.ndarray, pd.DataFrame]:
    torch, extractor, model = load_ssl_model(model_name, local_files_only=local_files_only)
    embeddings: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    by_audio: dict[str, pd.DataFrame] = {
        audio_path: group.sort_values("phone_index")
        for audio_path, group in df.groupby("audio_path")
    }
    for index, (audio_path, group) in enumerate(by_audio.items(), start=1):
        full_path = ROOT / audio_path
        rate, signal = read_wav_mono(full_path)
        audio_ms = 1000.0 * len(signal) / rate
        hidden = hidden_state_for_audio(torch, extractor, model, full_path)
        for _, row in group.iterrows():
            embedding = pool_phone_embedding(
                hidden,
                float(row["start_ms"]),
                float(row["end_ms"]),
                audio_ms,
            )
            embeddings.append(embedding.astype(np.float32))
            rows.append(row.to_dict())
        if index % 25 == 0:
            print(f"Extracted SSL embeddings for {index}/{len(by_audio)} utterances")
    matrix = np.vstack(embeddings)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, embeddings=matrix)
    meta = pd.DataFrame(rows)
    meta.to_csv(output_npz.with_suffix(".metadata.csv"), index=False, encoding="utf-8-sig")
    return matrix, meta


def build_classifier(embedding_dim: int) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "meta_num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            META_NUMERIC,
                        ),
                        (
                            "meta_cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            META_CATEGORICAL,
                        ),
                        (
                            "ssl",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            [f"ssl_{i}" for i in range(embedding_dim)],
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def best_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    best_t = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (scores >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_t = float(threshold)
    return best_t


def metric_row(frame: pd.DataFrame, split: str, threshold: float) -> dict[str, object]:
    y_true = frame["gold_binary"].astype(int)
    y_pred = (frame["prob_correct"].astype(float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, frame["prob_correct"].astype(float))
    except ValueError:
        auc = 0.5
    error_precision = tn / (tn + fn) if (tn + fn) else 0.0
    error_recall = tn / (tn + fp) if (tn + fp) else 0.0
    error_f1 = 2 * error_precision * error_recall / (error_precision + error_recall) if (error_precision + error_recall) else 0.0
    return {
        "split": split,
        "model": "wav2vec2_ssl_logreg",
        "calibration": "global_threshold",
        "threshold": round(threshold, 6),
        "n": len(frame),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision_correct_class": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_correct_class": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "auc": round(float(auc), 6),
        "error_precision": round(float(error_precision), 6),
        "error_recall": round(float(error_recall), 6),
        "error_f1": round(float(error_f1), 6),
        "tn_error_predicted_error": int(tn),
        "fp_correct_predicted_error": int(fp),
        "fn_error_predicted_correct": int(fn),
        "tp_correct_predicted_correct": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/speechocean/phones_aligned.csv")
    parser.add_argument("--alignment-quality", default="pass")
    parser.add_argument("--model-name", default="facebook/wav2vec2-base")
    parser.add_argument("--max-utterances-per-split", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/ssl_wav2vec2_v1")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/phase1")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if args.alignment_quality.lower() != "all":
        df = df[df["alignment_quality"].str.lower() == args.alignment_quality.lower()].copy()
    df = select_rows(df, args.max_utterances_per_split)
    df["gold_binary"] = df["gold_binary"].astype(int)
    print(f"SSL input rows: {len(df)}; utterances: {df['utterance_id'].nunique()}")

    embeddings_path = args.output_dir / "phone_embeddings.npz"
    matrix, meta = extract_embeddings(df, args.model_name, embeddings_path, args.local_files_only)
    base_columns = [
        "utterance_id",
        "speaker_id",
        "target_phone",
        "phone_group",
        "duration_ms",
        "phone_index",
        "split",
        "gold_binary",
        "audio_path",
    ]
    feature_frame = meta[base_columns].copy()
    ssl_columns = pd.DataFrame(
        matrix,
        columns=[f"ssl_{i}" for i in range(matrix.shape[1])],
        index=feature_frame.index,
    )
    feature_frame = pd.concat([feature_frame, ssl_columns], axis=1)

    train = feature_frame[feature_frame["split"] == "train"].copy()
    dev = feature_frame[feature_frame["split"] == "dev"].copy()
    test = feature_frame[feature_frame["split"] == "test"].copy()
    if train.empty or dev.empty or test.empty:
        raise SystemExit("Need non-empty train/dev/test splits for SSL baseline.")

    classifier = build_classifier(matrix.shape[1])
    classifier.fit(train.drop(columns=["gold_binary"]), train["gold_binary"])
    classes = list(classifier.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    for frame in [dev, test]:
        frame["prob_correct"] = classifier.predict_proba(frame.drop(columns=["gold_binary"]))[:, positive_index]
    threshold = best_threshold(dev["gold_binary"], dev["prob_correct"].to_numpy())
    metrics = [metric_row(dev, "dev", threshold), metric_row(test, "test", threshold)]

    joblib.dump(classifier, args.output_dir / "ssl_logreg.joblib")
    pd.DataFrame(metrics).to_csv(args.report_dir / "ssl_eval_metrics.csv", index=False, encoding="utf-8-sig")
    predictions = pd.concat([dev, test], ignore_index=True)
    predictions["threshold"] = threshold
    predictions["prediction"] = (predictions["prob_correct"] >= threshold).astype(int)
    keep = [
        "utterance_id",
        "speaker_id",
        "target_phone",
        "phone_group",
        "split",
        "gold_binary",
        "prob_correct",
        "threshold",
        "prediction",
        "audio_path",
    ]
    predictions[keep].to_csv(args.report_dir / "ssl_predictions.csv", index=False, encoding="utf-8-sig")
    summary = [
        "# Self-Supervised Speech Representation Baseline",
        "",
        f"Model: `{args.model_name}`",
        f"Rows: {len(df)}",
        f"Utterances: {df['utterance_id'].nunique()}",
        f"Max utterances per split: {args.max_utterances_per_split}",
        "",
        "## Test Result",
        "",
    ]
    test_row = metrics[-1]
    for key in ["balanced_accuracy", "macro_f1", "auc", "error_precision", "error_recall", "error_f1"]:
        summary.append(f"- {key}: {test_row[key]}")
    summary.append("")
    summary.append("Artifacts: `artifacts/ssl_wav2vec2_v1/`; metrics: `reports/phase1/ssl_eval_metrics.csv`.")
    (args.report_dir / "ssl_baseline_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
