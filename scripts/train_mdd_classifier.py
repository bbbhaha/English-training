#!/usr/bin/env python
"""Train a phone-level wav2vec2-MDD classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


META_CATEGORICAL = ["target_phone", "phone_group"]
META_NUMERIC = ["duration", "normalized_duration_by_phone"]
OPTIONAL_NUMERIC = ["gop_score"]
RANDOM_SEED = 42


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
    for col in [c for c in META_NUMERIC if c in frame.columns] + [c for c in OPTIONAL_NUMERIC if c in frame.columns]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in META_CATEGORICAL:
        frame[col] = frame[col].fillna("").astype(str)
    return frame


def feature_columns(frame: pd.DataFrame, use_gop: bool) -> tuple[list[str], list[str]]:
    numeric = [c for c in META_NUMERIC if c in frame.columns] + [c for c in OPTIONAL_NUMERIC if use_gop and c in frame.columns]
    numeric += [c for c in frame.columns if c.startswith("w2v_")]
    return numeric, META_CATEGORICAL


def duration_stats(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for phone, group in train.groupby("target_phone"):
        values = pd.to_numeric(group["duration"], errors="coerce").dropna()
        if values.empty:
            continue
        std = float(values.std(ddof=0))
        stats[str(phone)] = {
            "mean": float(values.mean()),
            "std": std if std > 1e-6 else 1.0,
        }
    return stats


def apply_duration_stats(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = frame.copy()
    global_mean = float(pd.to_numeric(out["duration"], errors="coerce").mean())
    global_std = float(pd.to_numeric(out["duration"], errors="coerce").std(ddof=0)) or 1.0
    values = []
    for _, row in out.iterrows():
        item = stats.get(str(row["target_phone"]), {"mean": global_mean, "std": global_std})
        values.append((float(row["duration"]) - item["mean"]) / item["std"])
    out["normalized_duration_by_phone"] = values
    return out


def build_pipeline(classifier: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical,
            ),
        ]
    )
    if classifier == "logreg":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=RANDOM_SEED,
        )
    elif classifier == "mlp":
        # sklearn MLP has no class_weight in older versions; sample_weight is
        # passed during fit below when supported.
        model = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-4,
            max_iter=200,
            random_state=RANDOM_SEED,
            early_stopping=True,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    return Pipeline([("preprocess", preprocessor), ("classifier", model)])


def sample_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts().to_dict()
    total = len(labels)
    return labels.map(lambda value: total / (2 * counts[int(value)])).to_numpy(dtype=float)


def score_with_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
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
    return pipeline.predict_proba(frame[numeric + categorical])[:, classes.index(1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("artifacts/mdd_wav2vec2/features.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mdd_wav2vec2/mdd_classifier.joblib"))
    parser.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--use-gop", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0.0, help="Drop phone segments shorter than this many seconds.")
    parser.add_argument("--negative-weight", type=float, default=1.0, help="Extra sample weight multiplier for correct/negative class.")
    parser.add_argument("--hard-negative-model", type=Path, help="First-pass model used to identify high-scoring correct phones.")
    parser.add_argument("--hard-negative-top-frac", type=float, default=0.10, help="Top fraction of correct train phones by error score to upweight.")
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    args = parser.parse_args()

    frame = load_feature_frame(args.features, args.min_duration)
    raw_train = frame[frame["split"] == "train"].copy()
    stats = duration_stats(raw_train)
    frame = apply_duration_stats(frame, stats)
    train = frame[frame["split"] == "train"].copy()
    dev = frame[frame["split"] == "dev"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or dev.empty or test.empty:
        raise SystemExit("Features must contain train/dev/test splits.")
    if train["label"].nunique() < 2:
        raise SystemExit(
            "Training split contains only one class. Increase --max-utterances-per-split "
            "during feature extraction or use a manifest with mispronounced examples."
        )
    numeric, categorical = feature_columns(frame, args.use_gop)
    pipeline = build_pipeline(args.classifier, numeric, categorical)
    x_train = train[numeric + categorical]
    y_train = train["label"]
    weights = sample_weights(y_train)
    weights[y_train.to_numpy() == 0] *= args.negative_weight
    hard_negative_count = 0
    if args.hard_negative_model:
        first_pass = joblib.load(args.hard_negative_model)
        train_scores = score_with_bundle(first_pass, train)
        correct_mask = y_train.to_numpy() == 0
        if correct_mask.any():
            cutoff = np.quantile(train_scores[correct_mask], 1.0 - args.hard_negative_top_frac)
            hard_mask = correct_mask & (train_scores >= cutoff)
            weights[hard_mask] *= args.hard_negative_weight
            hard_negative_count = int(hard_mask.sum())
    if args.classifier == "mlp":
        try:
            pipeline.fit(x_train, y_train, classifier__sample_weight=weights)
        except TypeError:
            pipeline.fit(x_train, y_train)
    else:
        pipeline.fit(x_train, y_train, classifier__sample_weight=weights)

    bundle = {
        "pipeline": pipeline,
        "classifier": args.classifier,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "positive_class": "mispronounced",
        "label_definition": "label=1 mispronounced; label=0 correct",
        "random_seed": RANDOM_SEED,
        "duration_stats_by_phone": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    metadata = {
        "features": str(args.features),
        "model": str(args.output),
        "classifier": args.classifier,
        "train_rows": int(len(train)),
        "dev_rows": int(len(dev)),
        "test_rows": int(len(test)),
        "train_labels": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
        "min_duration": args.min_duration,
        "negative_weight": args.negative_weight,
        "hard_negative_model": str(args.hard_negative_model) if args.hard_negative_model else None,
        "hard_negative_count": hard_negative_count,
        "hard_negative_top_frac": args.hard_negative_top_frac,
        "hard_negative_weight": args.hard_negative_weight,
        "numeric_feature_count": len(numeric),
        "categorical_features": categorical,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
