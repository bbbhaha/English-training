from __future__ import annotations

import math

import pandas as pd


BAD_ALIGNMENT_VALUES = {"bad", "failed", "alignment_failed"}


def compute_phone_decision(row: pd.Series | dict[str, object]) -> dict[str, object]:
    """Compute one phone decision without word-level or ASR overrides."""
    probability, score_available, source = _phone_error_probability(row)
    alignment = str(row.get("alignment_quality", "")).strip().lower()
    g2p_status = str(row.get("g2p_status", "success")).strip().lower()
    lexicon_status = str(row.get("lexicon_status", "")).strip().lower()
    target_phone = str(row.get("target_phone", "")).strip().upper()

    if g2p_status == "failed" or lexicon_status == "failed" or target_phone == "<UNK>":
        probability = 0.50
        decision = "uncertain_review"
        error_type = "g2p_issue"
        confidence = 0.0
        evidence = "The word has no reliable dictionary or G2P pronunciation; it cannot be judged yet."
    elif alignment in BAD_ALIGNMENT_VALUES:
        probability = max(probability, 0.50)
        decision = "uncertain_review"
        error_type = "alignment_issue"
        confidence = 0.0
        evidence = "Alignment failed; phone pronunciation cannot be reliably judged."
    elif not score_available:
        probability = 0.50
        decision = "uncertain_review"
        error_type = "possible_mispronunciation"
        confidence = 0.0
        evidence = "Phone-level model score is unavailable; manual review is required."
    elif probability >= 0.75:
        decision = "true_error"
        error_type = "mispronunciation"
        confidence = probability
        evidence = "Acoustic phone-level model indicates high error probability."
    elif probability >= 0.45:
        decision = "uncertain_review"
        error_type = "possible_mispronunciation"
        confidence = probability
        evidence = "Phone-level model indicates possible pronunciation issue."
    else:
        decision = "correct"
        error_type = ""
        confidence = 1.0 - probability
        evidence = "Phone-level model indicates acceptable pronunciation."

    probability = round(_clip(probability), 6)
    return {
        "phone_error_probability": probability,
        "phone_error_percent": round(probability * 100.0, 2),
        "phone_decision": decision,
        "phone_error_type": error_type,
        "phone_confidence": round(_clip(confidence), 6),
        "phone_score_source": source,
        "evidence_summary": evidence,
    }


def apply_phone_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    results = [compute_phone_decision(row) for _, row in out.iterrows()]
    for column in [
        "phone_error_probability",
        "phone_error_percent",
        "phone_decision",
        "phone_error_type",
        "phone_confidence",
        "phone_score_source",
        "evidence_summary",
    ]:
        out[column] = [result[column] for result in results]
    # Keep legacy consumers working while phone_* remains authoritative.
    out["decision"] = out["phone_decision"]
    out["error_type"] = out["phone_error_type"]
    out["confidence"] = out["phone_confidence"]
    out["review_reason"] = out["evidence_summary"]
    return out


def summarize_phone_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate phone diagnosis for word_summary.csv without changing phones."""
    if frame.empty or "word_index" not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for word_index, group in frame.groupby("word_index", sort=False, dropna=False):
        decisions = group.get("phone_decision", pd.Series("uncertain_review", index=group.index)).astype(str)
        if decisions.eq("true_error").any():
            word_decision = "has_phone_error"
        elif decisions.eq("uncertain_review").any():
            word_decision = "needs_review"
        else:
            word_decision = "correct"
        probabilities = pd.to_numeric(group.get("phone_error_probability"), errors="coerce")
        starts = pd.to_numeric(group.get("start_ms", pd.Series(dtype=float)), errors="coerce").dropna()
        ends = pd.to_numeric(group.get("end_ms", pd.Series(dtype=float)), errors="coerce").dropna()
        word_duration_ms = float(ends.max() - starts.min()) if len(starts) and len(ends) else float("nan")
        rows.append(
            {
                "word_index": word_index,
                "word": _first(group, "word"),
                "phone_count": int(len(group)),
                "word_duration_ms": word_duration_ms,
                "word_decision": word_decision,
                "error_type": "mispronunciation" if word_decision == "has_phone_error" else (
                    "needs_review" if word_decision == "needs_review" else ""
                ),
                "alignment_quality": _worst_alignment(group),
                "max_phone_error_probability": float(probabilities.max()) if probabilities.notna().any() else 0.5,
                "avg_phone_error_probability": float(probabilities.mean()) if probabilities.notna().any() else 0.5,
                "num_phone_true_error": int(decisions.eq("true_error").sum()),
                "num_phone_uncertain": int(decisions.eq("uncertain_review").sum()),
            }
        )
    return pd.DataFrame(rows)


def _phone_error_probability(row: pd.Series | dict[str, object]) -> tuple[float, bool, str]:
    # Real-audio inference must use the raw model direction first. The manual
    # calibrator was fitted on a small reviewed set and can saturate near 1.0
    # for otherwise high-probability-correct phones on unseen recordings.
    model_error = _number(row.get("model_error_score"))
    if model_error is not None:
        return _clip(model_error), True, "model_error_score"
    prob_correct = _number(row.get("prob_correct"))
    if prob_correct is not None:
        return _clip(1.0 - prob_correct), True, "prob_correct"
    existing = _number(row.get("phone_error_probability"))
    if existing is not None:
        return _clip(existing), True, "phone_error_probability"
    manual = _number(row.get("manual_calibrated_error_probability"))
    if manual is not None:
        return _clip(manual), True, "manual_calibrated_error_probability"
    return 0.50, False, "fallback"


def _number(value: object) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _clip(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _first(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns or group.empty:
        return ""
    values = group[column].dropna().astype(str)
    return values.iloc[0] if len(values) else ""


def _worst_alignment(group: pd.DataFrame) -> str:
    values = set(group.get("alignment_quality", pd.Series("", index=group.index)).fillna("").astype(str).str.lower())
    if values & BAD_ALIGNMENT_VALUES:
        return "bad"
    if "suspect" in values:
        return "suspect"
    if values & {"good", "pass", "ok"}:
        return "pass"
    return next(iter(values), "")
