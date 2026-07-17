#!/usr/bin/env python
"""Train a speaker-isolated correct/mispronounced/deleted phone classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.phones import phone_group
from pronunciation.ctc_phone_diagnosis import (
    DEFAULT_PHONE_CTC_MODEL,
    DEFAULT_REFERENCE_PHONE_CTC_MODEL,
    PHONE_STATES,
    REFERENCE_DELETION_MARGIN_THRESHOLD,
    add_phone_model_consensus_features,
    dual_phone_presence_guard,
    phone_equivalence_guard,
    prefix_reference_phone_evidence,
    score_audio_phones_ctc,
)


NUMERIC_FEATURES = [
    "duration_ms",
    "ctc_deletion_margin",
    "ctc_substitution_margin",
    "ctc_target_logit_score",
    "ctc_competing_logit_score",
    "ctc_logit_margin",
    "ctc_target_path_log_probability",
    "reference_ctc_deletion_margin",
    "reference_ctc_substitution_margin",
    "reference_ctc_target_logit_score",
    "reference_ctc_competing_logit_score",
    "reference_ctc_logit_margin",
    "reference_ctc_target_path_log_probability",
    "primary_recognized_available",
    "reference_recognized_available",
    "primary_target_match",
    "reference_target_match",
    "phone_models_recognized_same",
    "dual_deletion_margin_min",
    "dual_deletion_margin_max",
    "dual_substitution_margin_min",
    "dual_substitution_margin_max",
]
CATEGORICAL_FEATURES = ["target_phone", "phone_group"]
RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phones", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=ROOT / "outputs/phone_three_state/l2_arctic_ctc_features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models/phone_three_state_v3.joblib")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/phone_three_state/training_report_v3.json")
    parser.add_argument(
        "--correct-sanity-features",
        type=Path,
        help="Prediction CSV for known-correct speech; constrains deployment false alarms to at most 5%%.",
    )
    parser.add_argument("--ctc-model", default=DEFAULT_PHONE_CTC_MODEL)
    parser.add_argument("--reference-ctc-model", default=DEFAULT_REFERENCE_PHONE_CTC_MODEL)
    parser.add_argument(
        "--classifier",
        choices=["logreg", "hist_gradient_boosting", "random_forest", "extra_trees"],
        default="hist_gradient_boosting",
    )
    parser.add_argument(
        "--aux-features",
        type=Path,
        action="append",
        default=[],
        help="Additional feature CSV used for training only. May be supplied more than once.",
    )
    parser.add_argument("--max-utterances-per-speaker", type=int, default=0)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--resume-feature-extraction", action="store_true")
    parser.add_argument(
        "--refit-on-train-dev",
        action="store_true",
        help="After threshold selection, refit the deployment pipeline on train+dev and evaluate test once.",
    )
    args = parser.parse_args()

    if args.reuse_features and args.feature_cache.is_file():
        features = pd.read_csv(args.feature_cache, encoding="utf-8-sig", keep_default_na=False)
    else:
        source = pd.read_csv(args.phones, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
        source = prepare_source(source, args.max_utterances_per_speaker)
        features = extract_features(
            source,
            args.corpus_root,
            args.ctc_model,
            reference_model_id=args.reference_ctc_model,
            feature_cache=args.feature_cache,
            resume=args.resume_feature_extraction,
        )

    auxiliary_rows: dict[str, int] = {}
    if args.aux_features:
        parts = [features]
        for path in args.aux_features:
            auxiliary = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
            auxiliary["split"] = "train"
            auxiliary_rows[str(path)] = int(len(auxiliary))
            parts.append(auxiliary)
        features = pd.concat(parts, ignore_index=True, sort=False)

    correct_sanity = None
    if args.correct_sanity_features:
        correct_sanity = pd.read_csv(args.correct_sanity_features, encoding="utf-8-sig", keep_default_na=False)
    artifact, report = train_classifier(
        features,
        args.ctc_model,
        correct_sanity=correct_sanity,
        classifier_name=args.classifier,
        reference_model_id=args.reference_ctc_model,
        refit_on_train_dev=args.refit_on_train_dev,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    report.update(
        {
            "model_output": str(args.output),
            "feature_cache": str(args.feature_cache),
            "ctc_model": args.ctc_model,
            "reference_ctc_model": args.reference_ctc_model,
            "classifier": args.classifier,
            "max_utterances_per_speaker": args.max_utterances_per_speaker,
            "correct_sanity_features": str(args.correct_sanity_features or ""),
            "auxiliary_feature_rows": auxiliary_rows,
            "refit_on_train_dev": args.refit_on_train_dev,
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def prepare_source(frame: pd.DataFrame, max_utterances_per_speaker: int) -> pd.DataFrame:
    out = frame.copy()
    out = out[out["error_type"].isin(["correct", "substitution", "deletion"])].copy()
    out = out[~out["target_phone"].astype(str).str.upper().isin(["", "SIL", "SP", "SPN"])].copy()
    out["gold_phone_state"] = out["error_type"].map(
        {"correct": "correct", "substitution": "mispronounced", "deletion": "deleted"}
    )
    if max_utterances_per_speaker <= 0:
        return out
    selected: list[str] = []
    for _, speaker in out.groupby("speaker_id", sort=False):
        summary = speaker.groupby("utterance_id")["gold_phone_state"].agg(
            error_count=lambda values: int(values.ne("correct").sum()),
            deletion_count=lambda values: int(values.eq("deleted").sum()),
        ).reset_index()
        summary["priority"] = summary["deletion_count"] * 5 + summary["error_count"]
        error_utts = summary[summary["error_count"] > 0].sort_values(
            ["priority", "utterance_id"], ascending=[False, True]
        )
        clean_utts = summary[summary["error_count"] == 0].sort_values("utterance_id")
        error_budget = min(len(error_utts), max(1, int(round(max_utterances_per_speaker * 0.8))))
        chosen = error_utts.head(error_budget)["utterance_id"].tolist()
        remaining = max_utterances_per_speaker - len(chosen)
        chosen.extend(clean_utts.head(remaining)["utterance_id"].tolist())
        if len(chosen) < max_utterances_per_speaker:
            chosen.extend(
                error_utts.iloc[error_budget:].head(max_utterances_per_speaker - len(chosen))["utterance_id"].tolist()
            )
        selected.extend(chosen)
    return out[out["utterance_id"].isin(selected)].copy()


def extract_features(
    frame: pd.DataFrame,
    corpus_root: Path,
    model_id: str,
    *,
    reference_model_id: str = DEFAULT_REFERENCE_PHONE_CTC_MODEL,
    feature_cache: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    existing = pd.DataFrame()
    completed: set[str] = set()
    if resume and feature_cache is not None and feature_cache.is_file():
        existing = pd.read_csv(feature_cache, encoding="utf-8-sig", keep_default_na=False)
        completed = set(existing.get("utterance_id", pd.Series(dtype=str)).astype(str))
    rows: list[pd.DataFrame] = []
    first_write = not (resume and feature_cache is not None and feature_cache.is_file())
    groups = list(frame.groupby("utterance_id", sort=False))
    for number, (utterance_id, group) in enumerate(groups, start=1):
        if str(utterance_id) in completed:
            print(f"[{number}/{len(groups)}] resume skip {utterance_id}", flush=True)
            continue
        group = group.sort_values("phone_index", kind="stable").copy()
        audio_path = corpus_root / str(group.iloc[0]["audio_path"])
        evidence = score_audio_phones_ctc(audio_path, group, model_id=model_id, local_files_only=True)
        if not evidence["ctc_phone_model_available"].fillna(False).any():
            print(f"[{number}/{len(groups)}] skipped {utterance_id}: {evidence.iloc[0]['ctc_phone_error']}", flush=True)
            continue
        reference_evidence = score_audio_phones_ctc(
            audio_path,
            group,
            model_id=reference_model_id,
            local_files_only=True,
        )
        reference_evidence = prefix_reference_phone_evidence(reference_evidence)
        metadata = group[
            [
                "utterance_id",
                "speaker_id",
                "split",
                "phone_index",
                "target_phone",
                "duration_ms",
                "gold_phone_state",
                "error_type",
                "perceived_phone",
            ]
        ].copy()
        merged = metadata.merge(
            evidence.drop(columns=["target_phone"], errors="ignore"),
            on="phone_index",
            how="left",
        )
        reference_columns = [
            column
            for column in reference_evidence.columns
            if column not in merged.columns or column in {"word_index", "phone_index"}
        ]
        merged = merged.merge(
            reference_evidence[reference_columns].drop_duplicates(
                ["word_index", "phone_index"],
                keep="last",
            ),
            on=["word_index", "phone_index"],
            how="left",
        )
        merged["phone_group"] = merged["target_phone"].map(phone_group)
        merged = add_phone_model_consensus_features(merged)
        rows.append(merged)
        print(f"[{number}/{len(groups)}] {utterance_id}: {len(merged)} phones", flush=True)
        if feature_cache is not None and len(rows) >= 10:
            _flush_feature_rows(rows, feature_cache, first_write=first_write)
            first_write = False
            rows.clear()
    if feature_cache is not None and rows:
        _flush_feature_rows(rows, feature_cache, first_write=first_write)
        rows.clear()
    if feature_cache is not None and feature_cache.is_file():
        return pd.read_csv(feature_cache, encoding="utf-8-sig", keep_default_na=False)
    if not rows and existing.empty:
        raise RuntimeError("No CTC phone features were produced")
    return pd.concat([existing, *rows], ignore_index=True)


def _flush_feature_rows(rows: list[pd.DataFrame], path: Path, *, first_write: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = pd.concat(rows, ignore_index=True)
    batch.to_csv(
        path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8-sig" if first_write else "utf-8",
    )


def train_classifier(
    features: pd.DataFrame,
    ctc_model: str,
    *,
    correct_sanity: pd.DataFrame | None = None,
    classifier_name: str = "hist_gradient_boosting",
    reference_model_id: str = DEFAULT_REFERENCE_PHONE_CTC_MODEL,
    refit_on_train_dev: bool = False,
) -> tuple[dict, dict]:
    frame = add_phone_model_consensus_features(features)
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(
            frame.get(column, pd.Series(float("nan"), index=frame.index)),
            errors="coerce",
        )
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("").astype(str)
    train = frame[frame["split"] == "train"].copy()
    dev = frame[frame["split"] == "dev"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or dev.empty or test.empty:
        raise ValueError("speaker-isolated train/dev/test rows are required")

    classifier, numeric_steps = _build_classifier(classifier_name)
    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    fit_kwargs: dict[str, object] = {}
    if classifier_name == "hist_gradient_boosting":
        fit_kwargs["classifier__sample_weight"] = compute_sample_weight(
            "balanced",
            train["gold_phone_state"],
        )
    pipeline.fit(train[feature_columns], train["gold_phone_state"], **fit_kwargs)
    sanity = _prepare_correct_sanity(correct_sanity, feature_columns)
    thresholds, dev_metrics, sanity_metrics = tune_thresholds(
        pipeline,
        dev,
        feature_columns,
        correct_sanity=sanity,
    )
    selection_test_metrics = evaluate_split(pipeline, test, feature_columns, thresholds)
    if refit_on_train_dev:
        refit = pd.concat([train, dev], ignore_index=True)
        refit_kwargs: dict[str, object] = {}
        if classifier_name == "hist_gradient_boosting":
            refit_kwargs["classifier__sample_weight"] = compute_sample_weight(
                "balanced",
                refit["gold_phone_state"],
            )
        pipeline.fit(refit[feature_columns], refit["gold_phone_state"], **refit_kwargs)
        sanity_metrics = _correct_sanity_metrics(
            pipeline,
            sanity,
            feature_columns,
            thresholds,
        )
    test_metrics = evaluate_split(pipeline, test, feature_columns, thresholds)
    artifact = {
        "name": "l2_arctic_mandarin_dual_ctc_three_state_v5",
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "thresholds": thresholds,
        "classes": list(PHONE_STATES),
        "ctc_model": ctc_model,
        "reference_ctc_model": reference_model_id,
        "classifier_name": classifier_name,
        "label_definition": {"correct": "读对", "mispronounced": "读错", "deleted": "漏读"},
        "equivalence_guard": "target_match_or_narrow_standard_vowel_variant",
        "reference_deletion_margin_threshold": REFERENCE_DELETION_MARGIN_THRESHOLD,
        "use_reference_hard_gates": False,
        "reference_evidence_policy": "learned dual-CTC features without post-classifier hard gates",
        "random_seed": RANDOM_SEED,
        "refit_on_train_dev": refit_on_train_dev,
    }
    report = {
        "dataset": "L2-ARCTIC-v5.0 Mandarin manual phone annotations with optional training-only auxiliaries",
        "classifier": classifier_name,
        "speaker_split": {
            "train": sorted(train["speaker_id"].unique().tolist()),
            "dev": sorted(dev["speaker_id"].unique().tolist()),
            "test": sorted(test["speaker_id"].unique().tolist()),
        },
        "rows": {"train": len(train), "dev": len(dev), "test": len(test)},
        "labels": {
            split: {str(key): int(value) for key, value in part["gold_phone_state"].value_counts().to_dict().items()}
            for split, part in [("train", train), ("dev", dev), ("test", test)]
        },
        "thresholds": thresholds,
        "correct_sanity_metrics": sanity_metrics,
        "dev_metrics": dev_metrics,
        "selection_test_metrics": selection_test_metrics,
        "test_metrics": test_metrics,
        "refit_rows": int(len(train) + len(dev)) if refit_on_train_dev else int(len(train)),
    }
    return artifact, report


def tune_thresholds(
    pipeline: Pipeline,
    dev: pd.DataFrame,
    columns: list[str],
    *,
    correct_sanity: pd.DataFrame | None = None,
) -> tuple[dict[str, float], dict, dict]:
    probabilities = _probability_frame(pipeline, dev[columns])
    gold = dev["gold_phone_state"].astype(str).to_numpy()
    guard = phone_equivalence_guard(dev).to_numpy(bool)
    presence_guard = dual_phone_presence_guard(dev).to_numpy(bool)
    sanity_probabilities = None
    sanity_guard = None
    sanity_presence_guard = None
    if correct_sanity is not None and not correct_sanity.empty:
        sanity_probabilities = _probability_frame(pipeline, correct_sanity[columns])
        sanity_guard = phone_equivalence_guard(correct_sanity).to_numpy(bool)
        sanity_presence_guard = dual_phone_presence_guard(correct_sanity).to_numpy(bool)
    best: tuple[float, float, float, float, np.ndarray, float] | None = None
    for deletion_threshold in np.arange(0.30, 1.00, 0.01):
        for wrong_threshold in np.arange(0.30, 1.00, 0.01):
            predicted = classify_probabilities(
                probabilities,
                deletion_threshold,
                wrong_threshold,
                equivalence_guard=guard,
                deletion_presence_guard=presence_guard,
            )
            macro = f1_score(gold, predicted, labels=list(PHONE_STATES), average="macro", zero_division=0)
            correct_mask = gold == "correct"
            false_alarm = float(np.mean(predicted[correct_mask] != "correct")) if correct_mask.any() else 0.0
            sanity_false_alarm = 0.0
            if (
                sanity_probabilities is not None
                and sanity_guard is not None
                and sanity_presence_guard is not None
            ):
                sanity_prediction = classify_probabilities(
                    sanity_probabilities,
                    deletion_threshold,
                    wrong_threshold,
                    equivalence_guard=sanity_guard,
                    deletion_presence_guard=sanity_presence_guard,
                )
                sanity_false_alarm = float(np.mean(sanity_prediction != "correct"))
                if sanity_false_alarm > 0.05:
                    continue
            objective = macro - max(0.0, false_alarm - 0.10) * 2.0
            candidate = (
                objective,
                macro,
                -false_alarm,
                -sanity_false_alarm,
                predicted,
                float(deletion_threshold),
                float(wrong_threshold),
            )
            if best is None or candidate[:4] > best[:4]:
                best = candidate
    if best is None:
        raise ValueError("No threshold pair satisfies the correct-sanity false-alarm constraint")
    _, _, _, _, predicted, deletion_threshold, wrong_threshold = best
    thresholds = {"deleted": deletion_threshold, "mispronounced": wrong_threshold}
    sanity_metrics = _correct_sanity_metrics(
        pipeline,
        correct_sanity,
        columns,
        thresholds,
    )
    return thresholds, metric_payload(gold, predicted), sanity_metrics


def evaluate_split(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    columns: list[str],
    thresholds: dict[str, float],
) -> dict:
    probabilities = _probability_frame(pipeline, frame[columns])
    predicted = classify_probabilities(
        probabilities,
        thresholds["deleted"],
        thresholds["mispronounced"],
        equivalence_guard=phone_equivalence_guard(frame).to_numpy(bool),
        deletion_presence_guard=dual_phone_presence_guard(frame).to_numpy(bool),
    )
    return metric_payload(frame["gold_phone_state"].astype(str).to_numpy(), predicted)


def classify_probabilities(
    probabilities: pd.DataFrame,
    deletion_threshold: float,
    wrong_threshold: float,
    *,
    equivalence_guard: np.ndarray | None = None,
    reference_available: np.ndarray | None = None,
    reference_deletion_margin: np.ndarray | None = None,
    reference_equivalence_guard: np.ndarray | None = None,
    deletion_presence_guard: np.ndarray | None = None,
) -> np.ndarray:
    predicted = np.full(len(probabilities), "correct", dtype=object)
    deletion = probabilities["deleted"].to_numpy() >= deletion_threshold
    if deletion_presence_guard is not None:
        deletion &= ~deletion_presence_guard
    guard = np.zeros(len(probabilities), dtype=bool) if equivalence_guard is None else equivalence_guard
    if reference_available is not None and reference_deletion_margin is not None:
        deletion &= (~reference_available) | (
            reference_deletion_margin >= REFERENCE_DELETION_MARGIN_THRESHOLD
        )
    if reference_equivalence_guard is not None:
        guard = guard | reference_equivalence_guard
    wrong = (~deletion) & (~guard) & (probabilities["mispronounced"].to_numpy() >= wrong_threshold)
    predicted[deletion] = "deleted"
    predicted[wrong] = "mispronounced"
    return predicted


def _prepare_correct_sanity(frame: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    out = add_phone_model_consensus_features(frame)
    if "phone_group" not in out.columns:
        out["phone_group"] = out["target_phone"].map(phone_group)
    else:
        missing_group = out["phone_group"].fillna("").astype(str).str.strip().eq("")
        out.loc[missing_group, "phone_group"] = out.loc[missing_group, "target_phone"].map(phone_group)
    for column in NUMERIC_FEATURES:
        out[column] = pd.to_numeric(
            out.get(column, pd.Series(float("nan"), index=out.index)),
            errors="coerce",
        )
    for column in CATEGORICAL_FEATURES:
        out[column] = out.get(column, pd.Series("", index=out.index)).fillna("").astype(str)
    missing = [column for column in columns if column not in out.columns]
    if missing:
        raise KeyError(f"correct sanity features missing columns: {missing}")
    return out


def _build_classifier(classifier_name: str) -> tuple[object, list[tuple[str, object]]]:
    impute = ("impute", SimpleImputer(strategy="median"))
    if classifier_name == "hist_gradient_boosting":
        return (
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_SEED,
            ),
            [impute],
        )
    if classifier_name == "random_forest":
        return (
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=1.0,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            [impute],
        )
    if classifier_name == "extra_trees":
        return (
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            [impute],
        )
    if classifier_name == "logreg":
        return (
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            ),
            [impute, ("scale", StandardScaler())],
        )
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def _correct_sanity_metrics(
    pipeline: Pipeline,
    frame: pd.DataFrame | None,
    columns: list[str],
    thresholds: dict[str, float],
) -> dict:
    if frame is None or frame.empty:
        return {"available": False, "rows": 0, "false_alarm_rate": None}
    probabilities = _probability_frame(pipeline, frame[columns])
    predicted = classify_probabilities(
        probabilities,
        thresholds["deleted"],
        thresholds["mispronounced"],
        equivalence_guard=phone_equivalence_guard(frame).to_numpy(bool),
        deletion_presence_guard=dual_phone_presence_guard(frame).to_numpy(bool),
    )
    values, counts = np.unique(predicted, return_counts=True)
    return {
        "available": True,
        "rows": int(len(frame)),
        "false_alarm_rate": float(np.mean(predicted != "correct")),
        "prediction_counts": {str(value): int(count) for value, count in zip(values, counts)},
    }


def _reference_classification_inputs(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    if "reference_ctc_phone_model_available" not in frame.columns:
        return {}
    available = frame["reference_ctc_phone_model_available"].fillna(False).astype(bool).to_numpy()
    margin = pd.to_numeric(
        frame.get("reference_ctc_deletion_margin"),
        errors="coerce",
    ).fillna(float("-inf")).to_numpy(float)
    guard = phone_equivalence_guard(
        frame,
        recognized_column="reference_recognized_phone",
    ).to_numpy(bool)
    return {
        "reference_available": available,
        "reference_deletion_margin": margin,
        "reference_equivalence_guard": guard,
    }


def metric_payload(gold: np.ndarray, predicted: np.ndarray) -> dict:
    report = classification_report(
        gold,
        predicted,
        labels=list(PHONE_STATES),
        output_dict=True,
        zero_division=0,
    )
    correct_mask = gold == "correct"
    return {
        "macro_f1": float(f1_score(gold, predicted, labels=list(PHONE_STATES), average="macro", zero_division=0)),
        "correct_false_alarm_rate": float(np.mean(predicted[correct_mask] != "correct")) if correct_mask.any() else 0.0,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(gold, predicted, labels=list(PHONE_STATES)).tolist(),
        "label_order": list(PHONE_STATES),
    }


def _probability_frame(pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    raw = pipeline.predict_proba(features)
    classes = [str(value) for value in pipeline.classes_]
    out = pd.DataFrame(0.0, index=features.index, columns=list(PHONE_STATES))
    for index, label in enumerate(classes):
        out[label] = raw[:, index]
    return out


if __name__ == "__main__":
    main()
