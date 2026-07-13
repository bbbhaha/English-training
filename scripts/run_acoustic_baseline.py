#!/usr/bin/env python
"""Train and evaluate a CPU acoustic-likelihood (GOP-equivalent) baseline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.acoustic import (
    fit_phone_models,
    gop_equivalent_score,
    read_wav_mono,
    segment_features,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def union_fieldnames(rows: list[dict]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def collect_training_observations(
    rows: list[dict[str, str]],
) -> dict[str, list[np.ndarray]]:
    observations: dict[str, list[np.ndarray]] = defaultdict(list)
    by_audio: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row["gold_binary"] == "1":
            by_audio[row["audio_path"]].append(row)
    for audio_path, audio_rows in by_audio.items():
        rate, signal = read_wav_mono(ROOT / audio_path)
        for row in audio_rows:
            observations[row["target_phone"]].append(
                segment_features(
                    signal, rate, float(row["start_ms"]), float(row["end_ms"])
                )
            )
    return observations


def score_rows(rows: list[dict[str, str]], models: dict) -> list[dict]:
    scored: list[dict] = []
    by_audio: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row["split"] in {"train", "dev", "test"}
            and row["target_phone"] in models
            and row["error_type"] != "addition"
        ):
            by_audio[row["audio_path"]].append(row)
    for audio_path, audio_rows in by_audio.items():
        rate, signal = read_wav_mono(ROOT / audio_path)
        for row in audio_rows:
            features = segment_features(
                signal, rate, float(row["start_ms"]), float(row["end_ms"])
            )
            score, competitor, target_ll, competitor_ll = gop_equivalent_score(
                features, row["target_phone"], models
            )
            scored.append(
                {
                    **row,
                    "evidence_score": score,
                    "gop_score": score,
                    "predicted_phone": competitor,
                    "target_log_likelihood": target_ll,
                    "competitor_log_likelihood": competitor_ll,
                }
            )
    return scored


def best_threshold(rows: list[dict]) -> tuple[float, float]:
    scores = np.asarray([float(row["gop_score"]) for row in rows])
    labels = np.asarray([int(row["gold_binary"]) for row in rows])
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best = (-np.inf, float(np.median(scores)))
    for threshold in candidates:
        value = balanced_accuracy_score(labels, scores >= threshold)
        if value > best[0]:
            best = (value, float(threshold))
    return best[1], best[0]


def calibrate_thresholds(dev_rows: list[dict]) -> dict:
    global_threshold, dev_ba = best_threshold(dev_rows)
    groups: dict[str, float] = {}
    for group in sorted({row["phone_group"] for row in dev_rows}):
        subset = [row for row in dev_rows if row["phone_group"] == group]
        counts = Counter(row["gold_binary"] for row in subset)
        if counts["0"] >= 20 and counts["1"] >= 20:
            groups[group] = best_threshold(subset)[0]
    return {
        "global": global_threshold,
        "group": groups,
        "dev_balanced_accuracy_at_global": dev_ba,
    }


def apply_thresholds(rows: list[dict], thresholds: dict) -> None:
    for row in rows:
        threshold = thresholds["group"].get(
            row["phone_group"], thresholds["global"]
        )
        row["threshold"] = threshold
        row["prediction"] = int(float(row["gop_score"]) >= threshold)
        # Distance-to-threshold confidence; monotonic and bounded.
        row["confidence"] = float(
            1.0 / (1.0 + np.exp(-abs(float(row["gop_score"]) - threshold) / 5.0))
        )


def metrics(rows: list[dict]) -> dict:
    labels = np.asarray([int(row["gold_binary"]) for row in rows])
    predictions = np.asarray([int(row["prediction"]) for row in rows])
    scores = np.asarray([float(row["gop_score"]) for row in rows])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    macro = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "rows": len(rows),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "auc": roc_auc_score(labels, scores),
        "error_precision": precision[0],
        "error_recall": recall[0],
        "error_f1": f1[0],
        "correct_precision": precision[1],
        "correct_recall": recall[1],
        "correct_f1": f1[1],
        "support_error": int(support[0]),
        "support_correct": int(support[1]),
        "confusion_matrix_labels_0_1": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


NUMERIC_FEATURES = [
    "gop_score",
    "target_log_likelihood",
    "competitor_log_likelihood",
    "duration_ms",
]
CATEGORICAL_FEATURES = ["target_phone", "phone_group", "predicted_phone"]


def feature_matrix(rows: list[dict]) -> list[dict]:
    return [
        {
            **{name: float(row[name]) for name in NUMERIC_FEATURES},
            **{name: row[name] for name in CATEGORICAL_FEATURES},
        }
        for row in rows
    ]


def train_fusion_classifier(train_rows: list[dict]) -> Pipeline:
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=20260622,
    )
    pipeline = Pipeline(
        [
            ("features", DictVectorizer()),
            ("scale", StandardScaler(with_mean=False)),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(
        feature_matrix(train_rows),
        [int(row["gold_binary"]) for row in train_rows],
    )
    return pipeline


def best_probability_threshold(rows: list[dict]) -> tuple[float, float]:
    probabilities = np.asarray([float(row["model_probability"]) for row in rows])
    labels = np.asarray([int(row["gold_binary"]) for row in rows])
    best = (-np.inf, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        value = balanced_accuracy_score(labels, probabilities >= threshold)
        if value > best[0]:
            best = (value, float(threshold))
    return best[1], best[0]


def apply_fusion_model(
    model: Pipeline,
    rows: list[dict],
    threshold: float,
) -> None:
    probabilities = model.predict_proba(feature_matrix(rows))[:, 1]
    for row, probability in zip(rows, probabilities):
        row["model_probability"] = float(probability)
        row["prediction"] = int(probability >= threshold)
        row["confidence"] = float(
            probability if row["prediction"] else 1.0 - probability
        )


def probability_metrics(rows: list[dict]) -> dict:
    result = metrics(rows)
    labels = np.asarray([int(row["gold_binary"]) for row in rows])
    probabilities = np.asarray([float(row["model_probability"]) for row in rows])
    result["auc"] = roc_auc_score(labels, probabilities)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/l2_arctic/phones.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/baseline_acoustic_v1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.manifest)
    observations = collect_training_observations(rows)
    models = fit_phone_models(observations)
    scored = score_rows(rows, models)
    train_rows = [row for row in scored if row["split"] == "train"]
    dev_rows = [row for row in scored if row["split"] == "dev"]
    test_rows = [row for row in scored if row["split"] == "test"]
    thresholds = calibrate_thresholds(dev_rows)
    apply_thresholds(dev_rows, thresholds)
    apply_thresholds(test_rows, thresholds)
    acoustic_dev_metrics = metrics(dev_rows)
    acoustic_test_metrics = metrics(test_rows)

    fusion_model = train_fusion_classifier(train_rows)
    # Obtain probabilities before selecting the validation threshold.
    apply_fusion_model(fusion_model, train_rows, 0.5)
    apply_fusion_model(fusion_model, dev_rows, 0.5)
    fusion_threshold, fusion_dev_ba = best_probability_threshold(dev_rows)
    apply_fusion_model(fusion_model, train_rows, fusion_threshold)
    apply_fusion_model(fusion_model, dev_rows, fusion_threshold)
    apply_fusion_model(fusion_model, test_rows, fusion_threshold)

    joblib.dump(models, args.output_dir / "phone_gaussians.joblib")
    joblib.dump(fusion_model, args.output_dir / "fusion_classifier.joblib")
    (args.output_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    report = {
        "method": "diagonal Gaussian phone likelihood ratio (GOP-equivalent)",
        "trained_phone_models": len(models),
        "training_frames_by_phone": {
            phone: model.frames for phone, model in sorted(models.items())
        },
        "unsupported_events": "addition events and phones without a train model",
        "acoustic_threshold": {
            "dev": acoustic_dev_metrics,
            "test": acoustic_test_metrics,
        },
        "fusion_classifier": {
            "validation_threshold": fusion_threshold,
            "dev_balanced_accuracy_at_threshold": fusion_dev_ba,
            "dev": probability_metrics(dev_rows),
            "test": probability_metrics(test_rows),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    all_scored_path = args.output_dir / "scored_all_splits.csv"
    with all_scored_path.open("w", encoding="utf-8-sig", newline="") as handle:
        output_rows = train_rows + dev_rows + test_rows
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(output_rows))
        writer.writeheader()
        writer.writerows(output_rows)
    prediction_path = args.output_dir / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        output_rows = dev_rows + test_rows
        writer = csv.DictWriter(handle, fieldnames=union_fieldnames(output_rows))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(report, indent=2))
    print(f"Wrote baseline artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
