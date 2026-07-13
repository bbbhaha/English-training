from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GOOD_ALIGNMENT_VALUES = {"good", "pass", "ok"}
BAD_ALIGNMENT_VALUES = {"bad", "failed", "alignment_failed"}


@dataclass(frozen=True)
class DecisionConfig:
    mode: str = "conservative"
    correct_threshold: float = 0.75
    true_error_probability_threshold: float = 0.90
    medium_error_probability_threshold: float = 0.60
    confidence_threshold: float = 0.70
    max_prob_correct_for_true_error: float = 0.40
    hardset_model_error_threshold: float = 0.05
    hardset_probability_threshold: float = 0.05
    hardset_confidence_threshold: float = 0.0
    detect_deletion_as_error: bool = False


def is_good_alignment(value: object) -> bool:
    return str(value).strip().lower() in GOOD_ALIGNMENT_VALUES


def apply_decision_rules(frame: pd.DataFrame, config: DecisionConfig | None = None) -> pd.DataFrame:
    config = config or DecisionConfig()
    if config.mode not in {"conservative", "hardset", "deletion_only"}:
        raise ValueError(f"Unsupported decision mode: {config.mode}")
    out = frame.copy()
    _ensure_score_columns(out)
    if config.mode == "deletion_only":
        return _apply_deletion_only(out, config)
    good_alignment = out["alignment_quality"].map(is_good_alignment)
    if config.mode == "hardset":
        return _apply_hardset(out, good_alignment, config)
    return _apply_conservative(out, good_alignment, config)


def _apply_deletion_only(out: pd.DataFrame, config: DecisionConfig) -> pd.DataFrame:
    out, _ = apply_deletion_only_override(out, detect_deletion_as_error=config.detect_deletion_as_error)
    return out


def apply_deletion_only_override(
    prediction_df: pd.DataFrame,
    word_summary_df: pd.DataFrame | None = None,
    *,
    detect_deletion_as_error: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Final deletion-only override with deletion and alignment safety priorities."""
    out = prediction_df.copy()
    _ensure_score_columns(out)
    if "original_alignment_quality" not in out.columns:
        out["original_alignment_quality"] = out["alignment_quality"]
    for col, default in {
        "possible_missing_word": False,
        "deletion_trigger_source": "none",
        "missing_word_reason": "",
        "error_type": "",
        "review_reason": "",
    }.items():
        if col not in out.columns:
            out[col] = default

    if word_summary_df is not None and "word_index" in out.columns and "word_index" in word_summary_df.columns:
        summary_cols = [
            col
            for col in [
                "word_index",
                "deletion_trigger_source",
                "missing_word_reason",
                "possible_missing_word",
                "alignment_quality",
            ]
            if col in word_summary_df.columns
        ]
        summary_map = word_summary_df[summary_cols].copy().rename(
            columns={"alignment_quality": "_word_summary_alignment_quality"}
        )
        out = out.drop(
            columns=[
                col
                for col in [
                    "deletion_trigger_source",
                    "missing_word_reason",
                    "possible_missing_word",
                    "_word_summary_alignment_quality",
                ]
                if col in out.columns
            ]
        ).merge(summary_map, on="word_index", how="left")
        out["deletion_trigger_source"] = out.get("deletion_trigger_source", pd.Series("none", index=out.index)).fillna("none").replace("", "none")
        out["missing_word_reason"] = out.get("missing_word_reason", pd.Series("", index=out.index)).fillna("")
        out["possible_missing_word"] = _bool(out, "possible_missing_word", False)

    trigger = _deletion_trigger_mask(out)
    original_alignment = out["original_alignment_quality"].fillna("").astype(str).str.strip().str.lower()
    current_alignment = out["alignment_quality"].fillna("").astype(str).str.strip().str.lower()
    word_alignment = out.get(
        "_word_summary_alignment_quality",
        pd.Series("", index=out.index),
    ).fillna("").astype(str).str.strip().str.lower()
    bad_alignment = (
        original_alignment.isin(BAD_ALIGNMENT_VALUES)
        | current_alignment.isin(BAD_ALIGNMENT_VALUES)
        | word_alignment.isin(BAD_ALIGNMENT_VALUES)
    )
    out["possible_missing_word"] = trigger
    out.loc[~trigger, "deletion_trigger_source"] = "none"
    out.loc[~trigger, "missing_word_reason"] = ""
    debug_reason = out.get("debug_reason", pd.Series("", index=out.index)).fillna("").astype(str)
    stale_score_suspect = (
        ~trigger
        & ~bad_alignment
        & current_alignment.eq("suspect")
        & debug_reason.str.contains("debug_high_error_ratio", regex=False)
    )
    out.loc[stale_score_suspect, "alignment_quality"] = "pass"

    out["decision"] = "correct"
    out["error_type"] = ""
    out["review_reason"] = ""
    out["manual_calibrated_error_probability"] = 0.0
    out["model_error_score"] = 0.0
    out["confidence"] = 1.0
    out["deletion_score"] = 0.0
    out["deletion_confidence"] = ""

    plain_bad_alignment = ~trigger & bad_alignment
    out.loc[plain_bad_alignment, "alignment_quality"] = "bad"
    out.loc[plain_bad_alignment, "decision"] = "uncertain_review"
    out.loc[plain_bad_alignment, "error_type"] = "alignment_issue"
    out.loc[plain_bad_alignment, "review_reason"] = "bad_alignment"
    out.loc[plain_bad_alignment, "confidence"] = 0.0

    out.loc[trigger, "alignment_quality"] = "suspect"
    out.loc[trigger, "decision"] = "uncertain_review"
    out.loc[trigger, "error_type"] = "possible_deletion"
    out.loc[trigger, "review_reason"] = [
        "possible_missing_word:" + (reason or "word_level_deletion_signal")
        for reason in out.loc[trigger, "missing_word_reason"].fillna("").astype(str)
    ]
    out.loc[trigger, "deletion_score"] = 0.5
    out.loc[trigger, "deletion_confidence"] = "low"
    if detect_deletion_as_error:
        out.loc[trigger, "decision"] = "true_error"
        out.loc[trigger, "error_type"] = "deletion"
        out.loc[trigger, "review_reason"] = "missing_word_detected"
        out.loc[trigger, "deletion_score"] = 1.0
        out.loc[trigger, "deletion_confidence"] = "high"

    summary = None
    if word_summary_df is not None:
        summary = _override_word_summary(word_summary_df.copy(), out, detect_deletion_as_error=detect_deletion_as_error)
    out = out.drop(columns=["_word_summary_alignment_quality"], errors="ignore")
    return out, summary


def _deletion_trigger_mask(frame: pd.DataFrame) -> pd.Series:
    source = (
        frame.get("deletion_trigger_source", pd.Series("none", index=frame.index))
        .fillna("none")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    asr_missing = _bool(frame, "asr_missing_word", False)
    return asr_missing | ~source.isin({"", "none", "nan"})


def _override_word_summary(summary: pd.DataFrame, phones: pd.DataFrame, *, detect_deletion_as_error: bool) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    for col, default in {
        "possible_missing_word": False,
        "deletion_trigger_source": "none",
        "missing_word_reason": "",
        "alignment_quality": "pass",
        "word_decision": "correct",
        "error_type": "",
        "num_phone_true_error": 0,
        "num_phone_uncertain": 0,
        "original_alignment_quality": "",
    }.items():
        if col not in out.columns:
            out[col] = default

    trigger_by_word = phones.groupby("word_index", dropna=False).agg(
        possible_missing_word=("possible_missing_word", "any"),
        deletion_trigger_source=("deletion_trigger_source", _first_non_none),
        missing_word_reason=("missing_word_reason", _first_nonempty),
        num_phone_true_error=("decision", lambda s: int(s.astype(str).eq("true_error").sum())),
        num_phone_uncertain=("decision", lambda s: int(s.astype(str).eq("uncertain_review").sum())),
        has_bad_alignment=("alignment_quality", _has_bad_alignment),
    )
    out = out.drop(
        columns=[
            col
            for col in [
                "possible_missing_word",
                "deletion_trigger_source",
                "missing_word_reason",
                "num_phone_true_error",
                "num_phone_uncertain",
                "has_bad_alignment",
            ]
            if col in out.columns
        ]
    ).merge(trigger_by_word.reset_index(), on="word_index", how="left")
    out["possible_missing_word"] = _bool(out, "possible_missing_word", False)
    out["deletion_trigger_source"] = out["deletion_trigger_source"].fillna("none").replace("", "none")
    out["missing_word_reason"] = out["missing_word_reason"].fillna("")
    out["num_phone_true_error"] = pd.to_numeric(out["num_phone_true_error"], errors="coerce").fillna(0).astype(int)
    out["num_phone_uncertain"] = pd.to_numeric(out["num_phone_uncertain"], errors="coerce").fillna(0).astype(int)
    out["has_bad_alignment"] = _bool(out, "has_bad_alignment", False)

    no_trigger = ~out["possible_missing_word"]
    summary_alignment = out["alignment_quality"].fillna("").astype(str).str.strip().str.lower()
    summary_bad = no_trigger & (out["has_bad_alignment"] | summary_alignment.isin(BAD_ALIGNMENT_VALUES))
    out.loc[no_trigger, "word_decision"] = "correct"
    out.loc[no_trigger, "error_type"] = ""
    summary_debug = out.get("debug_reason", pd.Series("", index=out.index)).fillna("").astype(str)
    stale_score_suspect = (
        no_trigger
        & ~summary_bad
        & summary_alignment.eq("suspect")
        & summary_debug.str.contains("debug_high_error_ratio", regex=False)
    )
    out.loc[stale_score_suspect, "alignment_quality"] = "pass"
    out.loc[summary_bad, "possible_missing_word"] = False
    out.loc[summary_bad, "word_decision"] = "uncertain_review"
    out.loc[summary_bad, "error_type"] = "alignment_issue"
    out.loc[summary_bad, "missing_word_reason"] = "bad_alignment"
    out.loc[summary_bad, "alignment_quality"] = "bad"
    out.loc[out["possible_missing_word"], "alignment_quality"] = "suspect"
    out.loc[out["possible_missing_word"], "word_decision"] = "uncertain_review"
    out.loc[out["possible_missing_word"], "error_type"] = "possible_deletion"
    if detect_deletion_as_error:
        out.loc[out["possible_missing_word"], "word_decision"] = "true_error"
        out.loc[out["possible_missing_word"], "error_type"] = "deletion"
    return out


def _has_bad_alignment(values: pd.Series) -> bool:
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return bool(normalized.isin(BAD_ALIGNMENT_VALUES).any())


def _first_non_none(values: pd.Series) -> str:
    for value in values.fillna("none").astype(str):
        if value and value != "none":
            return value
    return "none"


def _first_nonempty(values: pd.Series) -> str:
    for value in values.fillna("").astype(str):
        if value:
            return value
    return ""


def _apply_conservative(out: pd.DataFrame, good_alignment: pd.Series, config: DecisionConfig) -> pd.DataFrame:
    prob_correct = _num(out, "prob_correct", 0.5).clip(0, 1)
    error_probability = _num(out, "manual_calibrated_error_probability", 0.5).clip(0, 1)
    confidence = _num(out, "confidence", 0.0).clip(0, 1)
    calibration_available = _bool(out, "calibration_available", False)
    relation = out.get("phonological_relation", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    asr_missing = _bool(out, "asr_missing_word", False)
    out["decision"] = "correct"
    out.loc[~good_alignment, "decision"] = "uncertain_review"

    can_decide = good_alignment
    out.loc[can_decide & relation.eq("acceptable_variant"), "decision"] = "acceptable_accent"
    correct = can_decide & (prob_correct >= config.correct_threshold)
    out.loc[correct, "decision"] = "correct"
    out.loc[correct, "manual_calibrated_error_probability"] = np.minimum(
        error_probability.loc[correct],
        _num(out.loc[correct], "model_error_score", 1.0 - prob_correct.loc[correct]).clip(0, 1),
    )
    error_probability = _num(out, "manual_calibrated_error_probability", 0.5).clip(0, 1)

    true_error = (
        can_decide
        & ~correct
        & ~relation.eq("acceptable_variant")
        & calibration_available
        & (error_probability >= config.true_error_probability_threshold)
        & (confidence >= config.confidence_threshold)
        & (prob_correct <= config.max_prob_correct_for_true_error)
        & relation.eq("likely_true_error")
    )
    out.loc[true_error, "decision"] = "true_error"

    uncertain = (
        can_decide
        & ~correct
        & ~true_error
        & (error_probability >= config.medium_error_probability_threshold)
        & ~relation.eq("acceptable_variant")
    )
    out.loc[uncertain, "decision"] = "uncertain_review"
    out.loc[asr_missing, "decision"] = "true_error"
    out.loc[~good_alignment, "confidence"] = 0.0
    return _add_error_fields(out, good_alignment, config)


def _apply_hardset(out: pd.DataFrame, good_alignment: pd.Series, config: DecisionConfig) -> pd.DataFrame:
    model_error = _num(out, "model_error_score", 0.0).clip(0, 1)
    error_probability = _num(out, "manual_calibrated_error_probability", 0.5).clip(0, 1)
    confidence = _num(out, "confidence", 0.0).clip(0, 1)
    true_error = (
        good_alignment
        & (model_error >= config.hardset_model_error_threshold)
        & (error_probability >= config.hardset_probability_threshold)
        & (confidence >= config.hardset_confidence_threshold)
    )
    out["decision"] = "acceptable_accent"
    out.loc[true_error, "decision"] = "true_error"
    out.loc[~good_alignment, "decision"] = "uncertain_review"
    out.loc[~good_alignment, "confidence"] = 0.0
    return _add_error_fields(out, good_alignment, config)


def _add_error_fields(out: pd.DataFrame, good_alignment: pd.Series, config: DecisionConfig) -> pd.DataFrame:
    asr_missing = _bool(out, "asr_missing_word", False)
    possible_missing = _bool(out, "possible_missing_word", False)
    plain_bad_alignment = ~good_alignment & ~possible_missing
    out["error_type"] = np.select(
        [
            asr_missing,
            possible_missing,
            plain_bad_alignment,
            out["decision"].eq("true_error"),
            out["decision"].eq("acceptable_accent"),
            out["decision"].eq("correct"),
        ],
        ["deletion", "possible_deletion", "bad_alignment", "pronunciation_error", "acceptable_accent", "none"],
        default="uncertain",
    )
    out.loc[asr_missing, "decision"] = "true_error"
    out.loc[asr_missing, "error_type"] = "deletion"
    out.loc[asr_missing, "deletion_score"] = 0.85
    out.loc[asr_missing, "deletion_confidence"] = "medium"
    out.loc[asr_missing & possible_missing, "deletion_score"] = 1.0
    out.loc[asr_missing & possible_missing, "deletion_confidence"] = "high"
    out.loc[possible_missing, "alignment_quality"] = "suspect"
    out.loc[possible_missing & ~asr_missing, "decision"] = "uncertain_review"
    out.loc[possible_missing & ~asr_missing, "confidence"] = 0.0
    out.loc[possible_missing & ~asr_missing, "deletion_score"] = 0.5
    out.loc[possible_missing & ~asr_missing, "deletion_confidence"] = "low"
    if config.detect_deletion_as_error:
        out.loc[possible_missing & ~asr_missing, "decision"] = "true_error"
        out.loc[possible_missing & ~asr_missing, "error_type"] = "deletion"
    existing = out.get("review_reason", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    missing_reason = out.get("missing_word_reason", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    computed = np.select(
        [
            asr_missing,
            possible_missing,
            plain_bad_alignment,
            out["decision"].eq("uncertain_review"),
            out["decision"].eq("true_error"),
        ],
        [
            "missing_in_asr_transcript",
            ["possible_missing_word:" + (reason or "word_level_deletion_signal") for reason in missing_reason],
            "alignment_quality_bad;possible_text_audio_mismatch",
            "insufficient_confidence",
            "manual_calibrated_probability_above_threshold",
        ],
        default="",
    )
    out["review_reason"] = [_merge_reasons(a, b) for a, b in zip(existing, computed)]
    return out


def _ensure_score_columns(out: pd.DataFrame) -> None:
    defaults = {
        "prob_correct": 0.5,
        "model_error_score": 0.5,
        "manual_calibrated_error_probability": 0.5,
        "confidence": 0.0,
        "alignment_quality": "",
        "calibration_available": False,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default


def _num(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _bool(frame: pd.DataFrame, col: str, default: bool) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index)
    value = frame[col]
    if value.dtype == bool:
        return value.fillna(default)
    return value.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _merge_reasons(existing: str, computed: str) -> str:
    parts: list[str] = []
    for value in (existing, computed):
        for part in str(value).split(";"):
            item = part.strip()
            if item and item not in parts:
                parts.append(item)
    return ";".join(parts)
