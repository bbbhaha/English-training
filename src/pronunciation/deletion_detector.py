from __future__ import annotations

import numpy as np
import pandas as pd


SHORT_PHONE_MS = 30.0
MIN_MULTI_PHONE_COUNT = 4


def detect_word_deletions(frame: pd.DataFrame, mode: str = "legacy") -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    for col in [
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
    ]:
        if col not in out.columns:
            if col == "possible_missing_word":
                out[col] = False
            elif col in {"missing_word_reason", "deletion_trigger_source", "debug_reason"}:
                out[col] = ""
            else:
                out[col] = 0.0
    if out.empty:
        return out, _empty_summary()

    rows = []
    group_cols = ["word_index", "word"] if "word_index" in out.columns else ["word"]
    for keys, group in out.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        word_index = keys[0] if len(group_cols) > 1 else group.get("word_index", pd.Series([0])).iloc[0]
        word = keys[-1]
        stats = _word_stats(group, mode=mode)
        alignment_values = set(group.get("alignment_quality", pd.Series("", index=group.index)).astype(str).str.lower())
        existing_possible = _truthy_series(group.get("possible_missing_word", pd.Series(False, index=group.index)))
        if mode != "deletion_only" and existing_possible.any():
            reasons = [
                str(v)
                for v in group.get("missing_word_reason", pd.Series("", index=group.index)).fillna("").unique()
                if str(v)
            ]
            stats["possible_missing_word"] = True
            stats["missing_word_reason"] = ";".join(reasons) if reasons else stats["missing_word_reason"]
        elif stats["deletion_trigger_source"] == "none" and not (alignment_values & {"good", "pass", "ok"}):
            stats["possible_missing_word"] = False
            stats["missing_word_reason"] = ""
            stats["deletion_trigger_source"] = "none"
        rows.append({"word": word, "word_index": word_index, **stats})
        mask = group.index
        for col in [
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
        ]:
            out.loc[mask, col] = stats[col]
        if stats["possible_missing_word"]:
            out.loc[mask, "alignment_quality"] = "suspect"
            out.loc[mask, "error_type"] = "possible_deletion"
            out.loc[mask, "review_reason"] = out.loc[mask].apply(
                lambda row: _merge_reason(row.get("review_reason", ""), f"possible_missing_word:{stats['missing_word_reason']}"),
                axis=1,
            )
        elif mode == "deletion_only":
            stale_suspect = out.loc[mask, "alignment_quality"].fillna("").astype(str).str.lower().eq("suspect")
            score_only_suspect = (
                stale_suspect
                & out.loc[mask, "debug_reason"].fillna("").astype(str).str.contains(
                    "debug_high_error_ratio",
                    regex=False,
                )
            )
            if score_only_suspect.any():
                out.loc[mask[score_only_suspect], "alignment_quality"] = "pass"
    return out, pd.DataFrame(rows)


def detect_missing_words(frame: pd.DataFrame, mode: str = "deletion_only") -> tuple[pd.DataFrame, pd.DataFrame]:
    return detect_word_deletions(frame, mode=mode)


def build_word_summary(frame: pd.DataFrame, mode: str = "legacy") -> pd.DataFrame:
    if frame.empty:
        return _empty_summary()
    rows = []
    group_cols = ["word_index", "word"] if "word_index" in frame.columns else ["word"]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        word_index = keys[0] if len(group_cols) > 1 else group.get("word_index", pd.Series([0])).iloc[0]
        word = keys[-1]
        stats = _word_stats(group, mode=mode)
        if "possible_missing_word" in group.columns:
            sources = [
                str(v)
                for v in group.get("deletion_trigger_source", pd.Series("", index=group.index)).fillna("").unique()
                if str(v)
            ]
            if sources:
                stats["deletion_trigger_source"] = next((v for v in sources if v != "none"), "none")
            preserve_existing = mode != "deletion_only" or stats["deletion_trigger_source"] != "none"
            if preserve_existing:
                stats["possible_missing_word"] = bool(_truthy_series(group["possible_missing_word"]).any())
                reasons = [str(v) for v in group.get("missing_word_reason", pd.Series("", index=group.index)).unique() if str(v)]
                stats["missing_word_reason"] = ";".join(reasons) if reasons else stats["missing_word_reason"]
        asr_missing = _truthy_series(group.get("asr_missing_word", pd.Series(False, index=group.index))).any()
        alignment_quality = "suspect" if stats["possible_missing_word"] else _worst_alignment(group)
        decisions = group.get("decision", pd.Series("", index=group.index)).astype(str)
        if asr_missing:
            word_decision = "true_error"
            error_type = "deletion"
            deletion_score = 1.0 if stats["possible_missing_word"] else 0.85
            deletion_confidence = "high" if stats["possible_missing_word"] else "medium"
        elif stats["possible_missing_word"]:
            word_decision = "true_error" if decisions.eq("true_error").any() else "uncertain_review"
            error_type = "deletion" if word_decision == "true_error" else "possible_deletion"
            deletion_score = 0.5
            deletion_confidence = "low"
        else:
            word_decision = _word_decision(decisions)
            error_type = _word_error_type(group, word_decision)
            deletion_score = 0.0
            deletion_confidence = ""
        rows.append(
            {
                "word": word,
                "word_index": word_index,
                "phone_count": stats["phone_count"],
                "word_duration_ms": stats["word_duration_ms"],
                "possible_missing_word": stats["possible_missing_word"],
                "missing_word_reason": stats["missing_word_reason"],
                "deletion_trigger_source": stats["deletion_trigger_source"],
                "debug_reason": stats["debug_reason"],
                "debug_high_error_ratio": stats["debug_high_error_ratio"],
                "debug_low_prob_correct_ratio": stats["debug_low_prob_correct_ratio"],
                "debug_short_phone_ratio": stats["debug_short_phone_ratio"],
                "alignment_quality": alignment_quality,
                "word_decision": word_decision,
                "error_type": error_type,
                "num_phone_true_error": int(decisions.eq("true_error").sum()),
                "num_phone_uncertain": int(decisions.eq("uncertain_review").sum()),
                "short_phone_ratio": stats["short_phone_ratio"],
                "high_error_ratio": stats["high_error_ratio"],
                "low_prob_correct_ratio": stats["low_prob_correct_ratio"],
                "asr_transcript": _first_value(group, "asr_transcript"),
                "asr_word_status": _first_value(group, "asr_word_status", "uncertain"),
                "asr_missing_word": bool(asr_missing),
                "asr_confidence": _first_value(group, "asr_confidence", 0.0),
                "alignment_op": _first_value(group, "alignment_op", "uncertain"),
                "recognized_word": _first_value(group, "recognized_word", ""),
                "deletion_score": deletion_score,
                "deletion_confidence": deletion_confidence,
            }
        )
    return pd.DataFrame(rows)


def _word_stats(group: pd.DataFrame, mode: str = "legacy") -> dict:
    phone_count = int(len(group))
    start = pd.to_numeric(group.get("start_ms", np.nan), errors="coerce")
    end = pd.to_numeric(group.get("end_ms", np.nan), errors="coerce")
    duration = pd.to_numeric(group.get("duration_ms", end - start), errors="coerce")
    error_prob = pd.to_numeric(
        group.get("manual_calibrated_error_probability", pd.Series(0.5, index=group.index)),
        errors="coerce",
    ).fillna(0.5)
    prob_correct = pd.to_numeric(
        group.get("prob_correct", pd.Series(0.5, index=group.index)),
        errors="coerce",
    ).fillna(0.5)
    boundary_missing = start.isna() | end.isna()
    valid_start = start.dropna()
    valid_end = end.dropna()
    word_duration = float(valid_end.max() - valid_start.min()) if len(valid_start) and len(valid_end) else 0.0
    short_ratio = float((duration.fillna(0.0) < SHORT_PHONE_MS).mean()) if phone_count else 0.0
    high_error_ratio = float((error_prob > 0.85).mean()) if phone_count else 0.0
    low_prob_correct_ratio = float((prob_correct < 0.3).mean()) if phone_count else 0.0
    missing_boundary_ratio = float(boundary_missing.mean()) if phone_count else 0.0
    asr_missing = _truthy_series(group.get("asr_missing_word", pd.Series(False, index=group.index))).any()
    debug_reasons = []
    if phone_count >= MIN_MULTI_PHONE_COUNT and short_ratio >= 0.5:
        debug_reasons.append("debug_short_phone_ratio_ge_0.5")
    if phone_count >= MIN_MULTI_PHONE_COUNT and high_error_ratio >= 0.8:
        debug_reasons.append("debug_high_error_ratio_ge_0.8")
    if phone_count >= MIN_MULTI_PHONE_COUNT and low_prob_correct_ratio >= 0.8:
        debug_reasons.append("debug_low_prob_correct_ratio_ge_0.8")

    reasons: list[str] = []
    deletion_trigger_source = "none"
    if mode == "deletion_only":
        extreme_threshold = max(80.0, phone_count * 20.0)
        if asr_missing:
            reasons.append("asr_missing_word")
            deletion_trigger_source = "asr_missing_word"
        elif phone_count >= MIN_MULTI_PHONE_COUNT and missing_boundary_ratio >= 0.8:
            reasons.append("missing_boundary_ratio_ge_0.8")
            deletion_trigger_source = "missing_boundaries"
        elif phone_count >= MIN_MULTI_PHONE_COUNT and word_duration < extreme_threshold:
            reasons.append("extreme_word_duration_compression")
            deletion_trigger_source = "extreme_duration_compression"
    else:
        if phone_count >= MIN_MULTI_PHONE_COUNT and word_duration < 250.0:
            reasons.append("word_duration_lt_250ms")
            deletion_trigger_source = "extreme_duration_compression"
        if phone_count >= MIN_MULTI_PHONE_COUNT and short_ratio >= 0.5:
            reasons.append("short_phone_ratio_ge_0.5")
            deletion_trigger_source = deletion_trigger_source if deletion_trigger_source != "none" else "short_phone_ratio"
        if phone_count >= MIN_MULTI_PHONE_COUNT and high_error_ratio >= 0.8:
            reasons.append("high_error_ratio_ge_0.8")
            deletion_trigger_source = deletion_trigger_source if deletion_trigger_source != "none" else "high_error_ratio"
        if phone_count >= MIN_MULTI_PHONE_COUNT and missing_boundary_ratio >= 0.5:
            reasons.append("missing_boundary_ratio_ge_0.5")
            deletion_trigger_source = "missing_boundaries"
    return {
        "phone_count": phone_count,
        "word_duration_ms": round(word_duration, 3),
        "short_phone_ratio": round(short_ratio, 6),
        "high_error_ratio": round(high_error_ratio, 6),
        "low_prob_correct_ratio": round(low_prob_correct_ratio, 6),
        "missing_boundary_ratio": round(missing_boundary_ratio, 6),
        "possible_missing_word": bool(reasons),
        "missing_word_reason": ";".join(reasons),
        "deletion_trigger_source": deletion_trigger_source,
        "debug_reason": ";".join(debug_reasons),
        "debug_high_error_ratio": round(high_error_ratio, 6),
        "debug_low_prob_correct_ratio": round(low_prob_correct_ratio, 6),
        "debug_short_phone_ratio": round(short_ratio, 6),
    }


def _worst_alignment(group: pd.DataFrame) -> str:
    values = set(group.get("alignment_quality", pd.Series("", index=group.index)).astype(str).str.lower())
    if "bad" in values:
        return "bad"
    if "suspect" in values:
        return "suspect"
    if values & {"good", "pass", "ok"}:
        return "pass"
    return next(iter(values), "")


def _word_decision(decisions: pd.Series) -> str:
    if decisions.eq("true_error").any():
        return "true_error"
    if decisions.eq("uncertain_review").any():
        return "uncertain_review"
    if decisions.eq("acceptable_accent").any():
        return "acceptable_accent"
    if decisions.eq("correct").any():
        return "correct"
    return ""


def _word_error_type(group: pd.DataFrame, decision: str) -> str:
    values = group.get("error_type", pd.Series("", index=group.index)).fillna("").astype(str)
    for preferred in ["alignment_issue", "deletion", "possible_deletion"]:
        if values.eq(preferred).any():
            return preferred
    if decision == "true_error":
        return "pronunciation_error"
    if decision == "uncertain_review":
        return "uncertain"
    if decision == "acceptable_accent":
        return "acceptable_accent"
    if decision == "correct":
        return ""
    nonempty = values[values.ne("")]
    return str(nonempty.iloc[0]) if len(nonempty) else ""


def _merge_reason(existing: object, extra: str) -> str:
    parts: list[str] = []
    for value in (existing, extra):
        for part in str(value or "").split(";"):
            item = part.strip()
            if item and item not in parts:
                parts.append(item)
    return ";".join(parts)


def _truthy_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _first_value(group: pd.DataFrame, col: str, default: object = "") -> object:
    if col not in group.columns or group.empty:
        return default
    value = group[col].iloc[0]
    return default if pd.isna(value) else value


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "word",
            "word_index",
            "phone_count",
            "word_duration_ms",
            "possible_missing_word",
            "missing_word_reason",
            "deletion_trigger_source",
            "debug_reason",
            "debug_high_error_ratio",
            "debug_low_prob_correct_ratio",
            "debug_short_phone_ratio",
            "alignment_quality",
            "word_decision",
            "error_type",
            "num_phone_true_error",
            "num_phone_uncertain",
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
        ]
    )
