#!/usr/bin/env python
"""End-to-end pronunciation prediction for wav + target text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.phones import phone_group
from phase15_verification.analysis import add_evidence_columns
from pronunciation.alignment import align_audio_to_text, save_alignment_csv
from pronunciation.asr_consistency import compare_target_with_asr
from pronunciation.audio_preprocess import inspect_audio, preprocess_audio
from pronunciation.calibration import apply_manual_calibrator
from pronunciation.decision import DecisionConfig, apply_decision_rules, apply_deletion_only_override, is_good_alignment
from pronunciation.deletion_detector import build_word_summary, detect_word_deletions
from pronunciation.g2p import write_g2p_json
from pronunciation.final_word_decision import merge_word_diagnosis_into_phones, run_word_level_diagnosis
from pronunciation.text_audio_consistency import (
    check_text_audio_consistency,
    consistency_to_json_payload,
    merge_consistency_into_phone_frame,
    merge_consistency_into_word_summary,
)
from pronunciation.target_words import build_target_word_table, ensure_word_summary_coverage
from pronunciation.verifier import add_verifier_defaults


CATEGORICAL_FEATURES = ["target_phone", "phone_group", "speaker_gender"]
NUMERIC_FEATURES = ["duration_ms", "phone_index", "word_index", "speaker_age"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end phone-level pronunciation diagnosis.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo/prediction.csv")
    parser.add_argument("--utterance-id")
    parser.add_argument("--speaker-id", default="demo")
    parser.add_argument("--speaker-gender", default="")
    parser.add_argument("--speaker-age", type=float, default=0.0)
    parser.add_argument("--alignment-output", type=Path)
    parser.add_argument("--g2p-output", type=Path)
    parser.add_argument("--word-summary-output", type=Path)
    parser.add_argument("--alignment-models", type=Path, default=ROOT / "artifacts/baseline_acoustic_v1/phone_gaussians.joblib")
    parser.add_argument("--phase1-model", type=Path, default=ROOT / "artifacts/phase1_acoustic_fusion_macro_models/feature_logreg.joblib")
    parser.add_argument("--phase15-config", type=Path, default=ROOT / "configs/phase15/aggregator.yaml")
    parser.add_argument("--manual-calibrator", type=Path, default=ROOT / "outputs/phase15_verification/manual_calibration_v2/manual_calibrated_verifier.joblib")
    parser.add_argument("--true-error-threshold", type=float, default=None)
    parser.add_argument("--main-error-threshold", type=float, default=0.05)
    parser.add_argument("--decision-mode", choices=["deletion_only", "conservative", "hardset"], default="deletion_only")
    parser.add_argument("--detect-deletion-as-error", action="store_true")
    parser.add_argument("--enable-asr-consistency-check", action="store_true")
    parser.add_argument("--asr-transcript")
    parser.add_argument("--text-audio-consistency-output", type=Path)
    parser.add_argument("--preprocessed-audio-output", type=Path)
    parser.add_argument("--no-auto-preprocess", action="store_true")
    parser.add_argument("--trim-silence", action="store_true", help="Trim long leading/trailing silence during auto-preprocess.")
    args = parser.parse_args()

    audio_for_alignment = _prepare_audio_for_alignment(args)
    utterance_id = args.utterance_id or args.audio.stem
    target_word_table = build_target_word_table(args.text, utterance_id=utterance_id)
    alignment, g2p = align_audio_to_text(
        audio_for_alignment,
        text=args.text,
        models_path=args.alignment_models,
        target_word_table=target_word_table,
    )
    if args.alignment_output:
        save_alignment_csv(alignment, args.alignment_output)
    if args.g2p_output and g2p is not None:
        write_g2p_json(g2p, args.g2p_output)

    frame = _prediction_frame(alignment, args)
    frame = _score_phase1(frame, args.phase1_model)
    frame = _add_verifier_defaults(frame)
    frame = add_evidence_columns(frame, _load_config(args.phase15_config))
    frame = _apply_manual_calibrator(frame, args.manual_calibrator, args.true_error_threshold)
    frame, _ = detect_word_deletions(frame, mode=args.decision_mode)
    consistency = pd.DataFrame()
    consistency_meta = {"asr_transcript": ""}
    asr_check_enabled = args.enable_asr_consistency_check or bool(args.asr_transcript)
    word_asr_consistency = pd.DataFrame()
    if asr_check_enabled:
        consistency, consistency_meta = check_text_audio_consistency(
            audio_path=audio_for_alignment,
            target_text=args.text,
            asr_transcript=args.asr_transcript,
        )
        if args.text_audio_consistency_output:
            payload = consistency_to_json_payload(consistency, consistency_meta)
            args.text_audio_consistency_output.parent.mkdir(parents=True, exist_ok=True)
            args.text_audio_consistency_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        frame = merge_consistency_into_phone_frame(
            frame,
            consistency,
            asr_transcript=str(consistency_meta.get("asr_transcript", "")),
        )
        frame, _ = detect_word_deletions(frame, mode=args.decision_mode)
        word_asr_consistency = compare_target_with_asr(
            args.text,
            str(consistency_meta.get("asr_transcript", "")),
            asr_confidence=1.0 if args.asr_transcript else 0.7,
        )
    out = _final_output(frame, args)
    out = ensure_prediction_coverage(out, target_word_table, g2p)
    word_summary = build_word_summary(out, mode=args.decision_mode)
    if asr_check_enabled:
        word_summary = merge_consistency_into_word_summary(
            word_summary,
            consistency,
            asr_transcript=str(consistency_meta.get("asr_transcript", "")),
        )
    word_summary = ensure_word_summary_coverage(target_word_table, word_summary)
    if args.decision_mode == "deletion_only":
        out, word_summary = apply_deletion_only_override(
            out,
            word_summary,
            detect_deletion_as_error=args.detect_deletion_as_error,
        )
    word_summary = run_word_level_diagnosis(out, word_summary, word_asr_consistency)
    word_summary = ensure_word_summary_coverage(target_word_table, word_summary)
    out = merge_word_diagnosis_into_phones(out, word_summary)
    out = ensure_prediction_coverage(out, target_word_table, g2p)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    if args.word_summary_output:
        args.word_summary_output.parent.mkdir(parents=True, exist_ok=True)
        word_summary.to_csv(args.word_summary_output, index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": len(out), "decision_counts": out["decision"].value_counts().to_dict()}, indent=2, ensure_ascii=False))
    print(f"Wrote pronunciation prediction to {args.output}")


def _prediction_frame(alignment: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = alignment.copy()
    out["utterance_id"] = args.utterance_id or args.audio.stem
    out["speaker_id"] = args.speaker_id
    out["audio_path"] = str(args.audio)
    out["transcript"] = args.text
    out["speaker_gender"] = args.speaker_gender
    out["speaker_age"] = args.speaker_age
    out["phone_group"] = out["target_phone"].map(phone_group)
    bad_alignment = ~out["alignment_quality"].map(is_good_alignment)
    out.loc[bad_alignment, "decision"] = "uncertain_review"
    out.loc[bad_alignment, "confidence"] = 0.0
    return out


def ensure_prediction_coverage(
    prediction_df: pd.DataFrame,
    target_word_table: pd.DataFrame,
    g2p_result=None,
) -> pd.DataFrame:
    """Ensure final predictions contain every expected target phone and word."""
    prediction = prediction_df.copy()
    if g2p_result is not None and getattr(g2p_result, "phones", None):
        expected = pd.DataFrame(g2p_result.phones)
    else:
        expected = target_word_table[["word_index", "word"]].copy()
        expected["target_phone"] = "<UNK>"
        expected["phone_index"] = range(len(expected))
        expected["word_phone_index"] = 0
        expected["g2p_source"] = "missing"
        expected["g2p_status"] = "failed"
        expected["g2p_error"] = "oov_or_g2p_failed"
    if expected.empty:
        return prediction
    expected = expected.drop(
        columns=[column for column in ("decision", "error_type", "review_reason") if column in expected.columns]
    )
    keys = ["word_index", "phone_index"]
    for frame in (expected, prediction):
        for key in keys:
            if key not in frame.columns:
                frame[key] = pd.NA
            frame[key] = pd.to_numeric(frame[key], errors="coerce").astype("Int64")
    authoritative = {
        "word", "target_phone", "word_phone_index", "g2p_source", "g2p_status", "g2p_error",
        "lexicon_status", "g2p_confidence", "pronunciation_variant_id",
        "num_pronunciation_variants", "selected_pronunciation",
    }
    prediction_columns = keys + [
        column for column in prediction.columns if column not in keys and column not in authoritative
    ]
    out = expected.merge(
        prediction[prediction_columns].drop_duplicates(keys, keep="last"),
        on=keys,
        how="left",
    )
    missing = out["decision"].isna() if "decision" in out.columns else pd.Series(True, index=out.index)
    defaults = {
        "decision": "uncertain_review",
        "error_type": "alignment_issue",
        "alignment_quality": "bad",
        "review_reason": "missing_from_prediction_pipeline",
        "manual_calibrated_error_probability": 0.0,
        "confidence": 0.0,
        "prob_correct": 0.5,
        "model_error_score": 0.5,
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
        else:
            out.loc[missing & out[column].isna(), column] = default
    g2p_failed = out.get("g2p_status", pd.Series("success", index=out.index)).fillna("failed").astype(str).eq("failed")
    out.loc[g2p_failed, "decision"] = "uncertain_review"
    out.loc[g2p_failed, "error_type"] = "g2p_issue"
    out.loc[g2p_failed, "alignment_quality"] = "bad"
    out.loc[g2p_failed, "review_reason"] = "g2p_failed"
    out.loc[g2p_failed, "confidence"] = 0.0
    target_meta = target_word_table.set_index("word_index")
    if "utterance_id" not in out.columns:
        out["utterance_id"] = out["word_index"].map(target_meta["utterance_id"])
    else:
        out["utterance_id"] = out["utterance_id"].fillna(out["word_index"].map(target_meta["utterance_id"]))
    return out.sort_values("phone_index", kind="stable").reset_index(drop=True)


def _prepare_audio_for_alignment(args: argparse.Namespace) -> Path:
    report = inspect_audio(args.audio)
    if not report.exists:
        raise FileNotFoundError(f"Audio file does not exist: {args.audio}")
    if report.warnings:
        warnings.warn(
            "Audio check warnings before alignment: "
            + ", ".join(report.warnings)
            + ". Expected 16 kHz mono 16-bit PCM wav with reasonable duration/RMS.",
            RuntimeWarning,
        )
    if not report.needs_preprocess:
        return args.audio
    if args.no_auto_preprocess:
        warnings.warn("Audio preprocessing is disabled; alignment may fail or fall back to bad rows.", RuntimeWarning)
        return args.audio
    output = args.preprocessed_audio_output or args.output.with_name(f"{args.output.stem}.preprocessed.wav")
    try:
        processed = preprocess_audio(args.audio, output, trim_silence=args.trim_silence)
        if processed.warnings:
            warnings.warn(
                "Preprocessed audio still has warnings: " + ", ".join(processed.warnings),
                RuntimeWarning,
            )
        else:
            warnings.warn(f"Auto-preprocessed audio for alignment: {output}", RuntimeWarning)
        return output
    except Exception as error:
        warnings.warn(
            f"Audio preprocessing failed ({error}); alignment will use original audio and may return bad fallback rows.",
            RuntimeWarning,
        )
        return args.audio


def _score_phase1(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    out = frame.copy()
    if not model_path.exists():
        out["prob_correct"] = 0.5
        out["model_error_score"] = 0.5
        out["prediction"] = 1
        out["confidence"] = 0.5
        return out
    model = joblib.load(model_path)
    expected_features = _expected_model_features(model)
    categorical = [c for c in CATEGORICAL_FEATURES if c in expected_features] or CATEGORICAL_FEATURES
    numeric = [c for c in expected_features if c not in categorical]
    if not expected_features:
        numeric = NUMERIC_FEATURES
        categorical = CATEGORICAL_FEATURES
    for col in categorical:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    for col in numeric:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    classes = list(model.named_steps["model"].classes_)
    feature_order = expected_features or (numeric + categorical)
    prob_correct = model.predict_proba(out[feature_order])[:, classes.index(1)]
    out["prob_correct"] = np.round(prob_correct, 6)
    out["model_error_score"] = np.round(1.0 - prob_correct, 6)
    out["prediction"] = (out["prob_correct"] >= 0.5).astype(int)
    out["confidence"] = np.maximum(out["prob_correct"], 1.0 - out["prob_correct"]).round(6)
    return out


def _expected_model_features(model) -> list[str]:
    try:
        pre = model.named_steps["preprocess"]
        cols: list[str] = []
        for _, _, transformer_cols in pre.transformers_:
            if isinstance(transformer_cols, list):
                cols.extend(transformer_cols)
            else:
                cols.extend(list(transformer_cols))
        return cols
    except Exception:
        return []


def _add_verifier_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    return add_verifier_defaults(frame)


def _apply_manual_calibrator(frame: pd.DataFrame, calibrator_path: Path, threshold_override: float | None) -> pd.DataFrame:
    return apply_manual_calibrator(frame, calibrator_path, threshold_override)


def _final_output(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = apply_decision_rules(
        frame,
        DecisionConfig(
            mode=args.decision_mode,
            hardset_model_error_threshold=args.main_error_threshold,
            hardset_probability_threshold=args.true_error_threshold if args.true_error_threshold is not None else 0.05,
            detect_deletion_as_error=getattr(args, "detect_deletion_as_error", False),
        ),
    )
    g2p_failed = out.get("g2p_status", pd.Series("success", index=out.index)).fillna("failed").astype(str).eq("failed")
    out.loc[g2p_failed, "decision"] = "uncertain_review"
    out.loc[g2p_failed, "error_type"] = "g2p_issue"
    out.loc[g2p_failed, "alignment_quality"] = "bad"
    out.loc[g2p_failed, "review_reason"] = "g2p_failed"
    out.loc[g2p_failed, "confidence"] = 0.0
    keep = [
        "utterance_id",
        "speaker_id",
        "word",
        "word_index",
        "target_phone",
        "phone_index",
        "start_ms",
        "end_ms",
        "duration_ms",
        "model_error_score",
        "prob_correct",
        "manual_calibrated_error_probability",
        "decision",
        "confidence",
        "error_type",
        "alignment_quality",
        "review_reason",
        "g2p_source",
        "g2p_status",
        "g2p_error",
        "word_phone_index",
        "lexicon_status",
        "g2p_confidence",
        "pronunciation_variant_id",
        "num_pronunciation_variants",
        "selected_pronunciation",
        "possible_missing_word",
        "missing_word_reason",
        "deletion_trigger_source",
        "debug_reason",
        "debug_high_error_ratio",
        "debug_low_prob_correct_ratio",
        "debug_short_phone_ratio",
        "word_duration_ms",
        "short_phone_ratio",
        "high_error_ratio",
        "low_prob_correct_ratio",
        "asr_transcript",
        "asr_word_status",
        "asr_missing_word",
        "asr_confidence",
        "alignment_op",
        "recognized_word",
        "deletion_score",
        "deletion_confidence",
        "phonological_relation",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = ""
    return out[keep]


def _merge_review_reasons(existing: str, computed: str) -> str:
    parts: list[str] = []
    for value in (existing, computed):
        for part in str(value).split(";"):
            item = part.strip()
            if item and item not in parts:
                parts.append(item)
    return ";".join(parts)


def _load_config(path: Path) -> dict:
    try:
        from phase15_verification.config import load_config

        return load_config(path)
    except Exception:
        return {}


if __name__ == "__main__":
    main()
