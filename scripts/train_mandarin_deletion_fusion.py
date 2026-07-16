from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.textgrid import Interval, read_interval_tiers
from pronunciation.ctc_word_deletion import score_audio_word_deletions
from pronunciation.g2p import text_to_phones
from pronunciation.mandarin_deletion_fusion import FEATURE_COLUMNS, build_mandarin_deletion_features
from pronunciation.target_words import build_target_word_table
from pronunciation.text_audio_consistency import check_text_audio_consistency, normalize_text
from pronunciation.word_deletion_model import word_deletion_detector


SPLIT_BY_SPEAKER = {"BWC": "train", "LXC": "train", "NCC": "dev", "TXHC": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a speaker-independent Mandarin-L1 word-deletion evidence fusion model."
    )
    parser.add_argument("--phones", type=Path, required=True)
    parser.add_argument(
        "--source-project-root",
        type=Path,
        required=True,
        help="Root used by audio_path and annotation_path in the processed L2-ARCTIC CSV.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mandarin_deletion_training")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT / "models/mandarin_deletion_fusion_v1.joblib",
    )
    parser.add_argument("--utterances-per-speaker", type=int, default=8)
    parser.add_argument("--deletions-per-utterance", type=int, default=2)
    parser.add_argument(
        "--additional-training-evidence",
        type=Path,
        action="append",
        default=[],
        help="Optional real-evidence CSV; only rows explicitly marked train are appended.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "mandarin_deletion_features.csv"
    if feature_path.is_file() and not args.rebuild_features:
        evidence = pd.read_csv(feature_path, encoding="utf-8-sig")
    else:
        evidence = collect_training_evidence(
            phones_path=args.phones,
            source_project_root=args.source_project_root,
            output_dir=args.output_dir,
            utterances_per_speaker=args.utterances_per_speaker,
            deletions_per_utterance=args.deletions_per_utterance,
            seed=args.seed,
        )
        evidence.to_csv(feature_path, index=False, encoding="utf-8-sig")

    evidence = append_real_training_evidence(evidence, args.additional_training_evidence)
    artifact, scored, report = train_fusion_model(evidence, seed=args.seed)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_output)
    scored.to_csv(args.output_dir / "scored_examples.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "training_report.md").write_text(
        _markdown_report(report, args.model_output),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote model to {args.model_output}")


def append_real_training_evidence(evidence: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    frames = [evidence.copy()]
    for path in paths:
        extra = pd.read_csv(path, encoding="utf-8-sig")
        split_column = "manifest_split" if "manifest_split" in extra.columns else "split"
        if split_column not in extra.columns:
            raise ValueError(f"Additional evidence has no split column: {path}")
        extra = extra.loc[extra[split_column].fillna("").astype(str).eq("train")].copy()
        if extra.empty:
            continue
        extra["split"] = "train"
        extra["evidence_source"] = "real_manual_pair"
        frames.append(extra)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe = [column for column in ("utterance_id", "variant_id", "word_index") if column in combined.columns]
    return combined.drop_duplicates(dedupe, keep="last") if dedupe else combined


def collect_training_evidence(
    *,
    phones_path: Path,
    source_project_root: Path,
    output_dir: Path,
    utterances_per_speaker: int,
    deletions_per_utterance: int,
    seed: int,
) -> pd.DataFrame:
    phones = pd.read_csv(phones_path)
    invalid_utterances = _utterances_with_full_word_deletions(phones)
    rng = random.Random(seed)
    synthetic_dir = output_dir / "synthetic_audio"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []

    for speaker_id, split in SPLIT_BY_SPEAKER.items():
        speaker_rows = phones.loc[phones["speaker_id"].astype(str).eq(speaker_id)]
        candidates: list[dict[str, Any]] = []
        for utterance_id, group in speaker_rows.groupby("utterance_id", sort=False):
            if str(utterance_id) in invalid_utterances:
                continue
            first = group.iloc[0]
            transcript = str(first.get("transcript", "")).strip()
            audio_path = _resolve(source_project_root, str(first.get("audio_path", "")))
            annotation_path = _resolve(source_project_root, str(first.get("annotation_path", "")))
            if not transcript or not audio_path.is_file() or not annotation_path.is_file():
                continue
            intervals = _exact_word_intervals(annotation_path, transcript)
            if intervals is None or len(intervals) < 4:
                continue
            candidates.append(
                {
                    "utterance_id": str(utterance_id),
                    "transcript": transcript,
                    "audio_path": audio_path,
                    "intervals": intervals,
                }
            )
        rng.shuffle(candidates)
        selected = candidates[: max(1, utterances_per_speaker)]
        for item_number, item in enumerate(selected, start=1):
            print(
                f"[{speaker_id} {item_number}/{len(selected)}] {item['utterance_id']} original",
                flush=True,
            )
            original = _score_variant(
                audio_path=item["audio_path"],
                transcript=item["transcript"],
                speaker_id=speaker_id,
                split=split,
                utterance_id=item["utterance_id"],
                variant_id="original",
                deleted_word_index=None,
            )
            rows.append(original)

            deletion_candidates = list(range(1, len(item["intervals"]) - 1))
            rng.shuffle(deletion_candidates)
            chosen = _choose_diverse_deletions(
                deletion_candidates,
                item["transcript"],
                deletions_per_utterance,
            )
            for word_index in chosen:
                target_interval = item["intervals"][word_index]
                synthetic_path = synthetic_dir / f"{speaker_id}_{item['utterance_id']}_del_{word_index}.wav"
                synthesize_word_deletion(item["audio_path"], synthetic_path, target_interval)
                print(
                    f"[{speaker_id} {item_number}/{len(selected)}] delete word {word_index}",
                    flush=True,
                )
                rows.append(
                    _score_variant(
                        audio_path=synthetic_path,
                        transcript=item["transcript"],
                        speaker_id=speaker_id,
                        split=split,
                        utterance_id=item["utterance_id"],
                        variant_id=f"delete_{word_index}",
                        deleted_word_index=word_index,
                    )
                )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def train_fusion_model(
    evidence: pd.DataFrame,
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    if evidence.empty:
        raise ValueError("No Mandarin deletion evidence was generated")
    features = build_mandarin_deletion_features(evidence)
    labels = evidence["gold_deletion"].astype(int)
    train_mask = evidence["split"].eq("train")
    dev_mask = evidence["split"].eq("dev")
    test_mask = evidence["split"].eq("test")
    if labels.loc[train_mask].nunique() < 2 or labels.loc[dev_mask].nunique() < 2:
        raise ValueError("Train and dev splits must both contain positive and negative examples")

    best_model: Pipeline | None = None
    best_c = 1.0
    best_ap = -1.0
    for regularization in (0.1, 0.3, 1.0, 3.0, 10.0):
        model = _new_model(regularization, seed)
        model.fit(features.loc[train_mask], labels.loc[train_mask])
        dev_probability = model.predict_proba(features.loc[dev_mask])[:, 1]
        score = average_precision_score(labels.loc[dev_mask], dev_probability)
        if score > best_ap:
            best_model, best_c, best_ap = model, regularization, float(score)
    assert best_model is not None

    dev_probability = best_model.predict_proba(features.loc[dev_mask])[:, 1]
    deletion_threshold = _threshold_at_precision(labels.loc[dev_mask].to_numpy(), dev_probability, 0.90)
    possible_threshold = min(deletion_threshold - 0.05, _best_f1_threshold(labels.loc[dev_mask].to_numpy(), dev_probability))
    possible_threshold = max(0.20, possible_threshold)

    scored = evidence.copy()
    scored["mandarin_deletion_probability"] = best_model.predict_proba(features)[:, 1]
    scored["mandarin_deletion_prediction"] = np.where(
        scored["mandarin_deletion_probability"].ge(deletion_threshold),
        "deletion",
        np.where(
            scored["mandarin_deletion_probability"].ge(possible_threshold),
            "possible_deletion",
            "correct",
        ),
    )

    report: dict[str, Any] = {
        "dataset": "L2-ARCTIC-v5.0 Mandarin speakers with synthetic word deletions",
        "speaker_split": {"train": ["BWC", "LXC"], "dev": ["NCC"], "test": ["TXHC"]},
        "feature_columns": FEATURE_COLUMNS,
        "selected_logistic_c": best_c,
        "thresholds": {
            "deletion": deletion_threshold,
            "possible_deletion": possible_threshold,
        },
        "counts": {
            split: {
                "rows": int(mask.sum()),
                "deletions": int(labels.loc[mask].sum()),
                "utterances": int(evidence.loc[mask, "utterance_id"].nunique()),
            }
            for split, mask in (("train", train_mask), ("dev", dev_mask), ("test", test_mask))
        },
        "model_metrics": {
            "train": _metrics(labels.loc[train_mask], scored.loc[train_mask, "mandarin_deletion_probability"], deletion_threshold),
            "dev": _metrics(labels.loc[dev_mask], scored.loc[dev_mask, "mandarin_deletion_probability"], deletion_threshold),
            "test": _metrics(labels.loc[test_mask], scored.loc[test_mask, "mandarin_deletion_probability"], deletion_threshold),
        },
        "baseline_rule_metrics": {
            "dev": _decision_metrics(labels.loc[dev_mask], evidence.loc[dev_mask, "baseline_deletion_decision"]),
            "test": _decision_metrics(labels.loc[test_mask], evidence.loc[test_mask, "baseline_deletion_decision"]),
        },
        "original_audio_false_alarm": {
            split: _original_false_alarm(
                scored.loc[mask],
                deletion_threshold,
            )
            for split, mask in (("dev", dev_mask), ("test", test_mask))
        },
        "standardized_logistic_coefficients": _model_coefficients(best_model),
        "limitations": (
            "Synthetic deletions improve scarce word-deletion coverage, but the held-out test is still small. "
            "Real human deletions must be reported separately."
        ),
    }
    artifact = {
        "name": "mandarin_l1_deletion_fusion_v1",
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "thresholds": report["thresholds"],
        "metadata": report,
    }
    return artifact, scored, report


def synthesize_word_deletion(source: Path, output: Path, interval: Interval, crossfade_ms: float = 8.0) -> None:
    sample_rate, samples = wavfile.read(source)
    values = np.asarray(samples)
    start = max(0, min(len(values), int(round(interval.start * sample_rate))))
    end = max(start, min(len(values), int(round(interval.end * sample_rate))))
    fade = min(int(round(sample_rate * crossfade_ms / 1000.0)), start, len(values) - end)
    if fade > 0:
        ramp_shape = (fade,) + (1,) * (values.ndim - 1)
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float64).reshape(ramp_shape)
        left = values[start - fade:start].astype(np.float64)
        right = values[end:end + fade].astype(np.float64)
        cross = left * (1.0 - ramp) + right * ramp
        if np.issubdtype(values.dtype, np.integer):
            info = np.iinfo(values.dtype)
            cross = np.clip(np.rint(cross), info.min, info.max).astype(values.dtype)
        else:
            cross = cross.astype(values.dtype)
        result = np.concatenate([values[:start - fade], cross, values[end + fade:]], axis=0)
    else:
        result = np.concatenate([values[:start], values[end:]], axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output, sample_rate, result)


def _score_variant(
    *,
    audio_path: Path,
    transcript: str,
    speaker_id: str,
    split: str,
    utterance_id: str,
    variant_id: str,
    deleted_word_index: int | None,
) -> pd.DataFrame:
    target_words = build_target_word_table(transcript, utterance_id=utterance_id)
    g2p = text_to_phones(transcript, target_words)
    g2p_words = pd.DataFrame(g2p.words)
    summary = target_words.merge(
        g2p_words[["word_index", "phones", "lexicon_status", "g2p_source"]],
        on="word_index",
        how="left",
    )
    summary["phone_count"] = summary["phones"].map(lambda value: len(value) if isinstance(value, list) else 0)
    summary["word_duration_ms"] = float("nan")
    summary["alignment_quality"] = "pass"
    consistency, meta = check_text_audio_consistency(
        audio_path=audio_path,
        target_text=transcript,
        asr_model="faster_whisper",
    )
    ctc = score_audio_word_deletions(audio_path, transcript, local_files_only=True)
    diagnosed = word_deletion_detector(summary, consistency, ctc, mandarin_fusion_model=None)
    diagnosed["speaker_id"] = speaker_id
    diagnosed["split"] = split
    diagnosed["utterance_id"] = utterance_id
    diagnosed["variant_id"] = variant_id
    diagnosed["audio_path"] = str(audio_path)
    diagnosed["deleted_word_index"] = -1 if deleted_word_index is None else int(deleted_word_index)
    diagnosed["gold_deletion"] = diagnosed["word_index"].eq(deleted_word_index) if deleted_word_index is not None else False
    diagnosed["baseline_deletion_decision"] = diagnosed["deletion_decision"]
    diagnosed["asr_transcript"] = str(meta.get("asr_transcript", ""))
    return diagnosed


def _exact_word_intervals(annotation_path: Path, transcript: str) -> list[Interval] | None:
    intervals = [interval for interval in read_interval_tiers(annotation_path).get("words", []) if interval.text.strip()]
    target = normalize_text(transcript)
    observed = [tokens[0] for interval in intervals if (tokens := normalize_text(interval.text))]
    return intervals if target == observed else None


def _choose_diverse_deletions(indices: list[int], transcript: str, limit: int) -> list[int]:
    words = normalize_text(transcript)
    function = [index for index in indices if words[index] in {"A", "AN", "AND", "OF", "THE", "TO", "FOR", "IN"}]
    content = [index for index in indices if index not in function]
    chosen: list[int] = []
    for group in (function, content, indices):
        for index in group:
            if index not in chosen:
                chosen.append(index)
            if len(chosen) >= limit:
                return chosen
    return chosen


def _utterances_with_full_word_deletions(phones: pd.DataFrame) -> set[str]:
    invalid: set[str] = set()
    for utterance_id, group in phones.groupby("utterance_id", sort=False):
        ordered = group.sort_values("phone_index")
        word_run = ordered["word"].fillna("").astype(str).ne(ordered["word"].fillna("").astype(str).shift()).cumsum()
        if ordered.groupby(word_run)["error_type"].apply(lambda values: values.eq("deletion").all()).any():
            invalid.add(str(utterance_id))
    return invalid


def _new_model(regularization: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=regularization,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _threshold_at_precision(labels: np.ndarray, probabilities: np.ndarray, minimum_precision: float) -> float:
    candidates = []
    for threshold in np.linspace(0.05, 0.99, 189):
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if tp > 0 and precision >= minimum_precision:
            candidates.append((recall, precision, -threshold, threshold))
    return float(max(candidates)[-1]) if candidates else _best_f1_threshold(labels, probabilities)


def _best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best = (0.0, 0.50)
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = probabilities >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predicted.astype(int),
            average="binary",
            zero_division=0,
        )
        candidate = (float(f1), -float(threshold))
        if candidate > (best[0], -best[1]):
            best = (float(f1), float(threshold))
    return best[1]


def _metrics(labels: pd.Series, probabilities: pd.Series, threshold: float) -> dict[str, float | int]:
    y = labels.astype(int).to_numpy()
    probability = pd.to_numeric(probabilities, errors="coerce").fillna(0.0).to_numpy()
    predicted = probability >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        predicted.astype(int),
        average="binary",
        zero_division=0,
    )
    false_positive = int(np.sum(predicted & (y == 0)))
    negative = int(np.sum(y == 0))
    return {
        "rows": int(len(y)),
        "positives": int(np.sum(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": false_positive / negative if negative else 0.0,
        "average_precision": float(average_precision_score(y, probability)) if len(np.unique(y)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else 0.0,
    }


def _decision_metrics(labels: pd.Series, decisions: pd.Series) -> dict[str, float | int]:
    probability = decisions.fillna("").astype(str).eq("deletion").astype(float)
    return _metrics(labels, probability, 0.5)


def _original_false_alarm(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    original = frame.loc[frame["variant_id"].eq("original")]
    false_alarm = int(original["mandarin_deletion_probability"].ge(threshold).sum())
    return {
        "words": int(len(original)),
        "false_deletions": false_alarm,
        "false_alarm_rate": false_alarm / len(original) if len(original) else 0.0,
    }


def _model_coefficients(model: Pipeline) -> list[dict[str, float | str]]:
    classifier = model.named_steps["classifier"]
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    rows = [
        {"feature": feature, "coefficient": float(coefficient)}
        for feature, coefficient in zip(FEATURE_COLUMNS, coefficients)
    ]
    return sorted(rows, key=lambda row: abs(float(row["coefficient"])), reverse=True)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _markdown_report(report: dict[str, Any], model_path: Path) -> str:
    lines = [
        "# Mandarin L1 deletion fusion training",
        "",
        f"Model: `{model_path}`",
        "",
        "The split is speaker-independent: BWC/LXC train, NCC dev, TXHC test.",
        "",
        "| Split | Precision | Recall | F1 | False positive rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "dev", "test"):
        metric = report["model_metrics"][split]
        lines.append(
            f"| {split} | {metric['precision']:.3f} | {metric['recall']:.3f} | "
            f"{metric['f1']:.3f} | {metric['false_positive_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Confirmed deletion threshold: `{report['thresholds']['deletion']:.3f}`",
            f"Possible deletion threshold: `{report['thresholds']['possible_deletion']:.3f}`",
            "",
            f"Limitation: {report['limitations']}",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
