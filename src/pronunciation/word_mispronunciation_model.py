from __future__ import annotations

from pathlib import Path

import pandas as pd

from pronunciation.mandarin_confusion_prior import classify_mandarin_confusion


BAD_ALIGNMENT = {"bad", "failed", "alignment_failed"}
SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def word_mispronunciation_detector(
    word_features: pd.DataFrame,
    *,
    confusion_config: Path | None = None,
) -> pd.DataFrame:
    out = word_features.copy()
    for column, default in {
        "target_phone_seq": "",
        "predicted_phone_seq": "",
        "ctc_gop_score": float("nan"),
        "min_phone_score": float("nan"),
        "avg_phone_score": float("nan"),
        "asr_edit_op": "uncertain",
        "alignment_quality": "",
        "manual_calibrated_error_probability": float("nan"),
    }.items():
        if column not in out.columns:
            out[column] = default

    decisions, scores, types, severities, summaries = [], [], [], [], []
    for _, row in out.iterrows():
        target = _phones(row.get("target_phone_seq"))
        predicted = _phones(row.get("predicted_phone_seq"))
        avg_score = _number(row.get("avg_phone_score"))
        min_score = _number(row.get("min_phone_score"))
        gop_score = _number(row.get("ctc_gop_score"))
        alignment = str(row.get("alignment_quality", "")).strip().lower()
        asr_op = str(row.get("asr_edit_op", "uncertain")).strip().lower()
        confusion = _strongest_confusion(target, predicted, confusion_config)
        severity = confusion["severity"]
        common = bool(confusion["is_common_mandarin_error"])
        intelligibility = bool(confusion["is_likely_intelligibility_error"])
        phone_difference = bool(predicted and target != predicted)
        acoustic_available = any(pd.notna(value) for value in [avg_score, min_score, gop_score])
        avg = avg_score if pd.notna(avg_score) else gop_score
        low_acoustic = acoustic_available and pd.notna(avg) and avg < 0.45 and (pd.isna(min_score) or min_score < 0.30)
        moderate_acoustic = acoustic_available and pd.notna(avg) and avg < 0.65
        evidence = [f"asr_edit_op={asr_op}", f"confusion={confusion['confusion_type']}"]
        if acoustic_available:
            evidence.append(f"avg_phone_score={avg:.3f}")
            if pd.notna(min_score):
                evidence.append(f"min_phone_score={min_score:.3f}")
        legacy = _number(row.get("manual_calibrated_error_probability"))
        if pd.notna(legacy):
            evidence.append(f"debug_legacy_error_probability={legacy:.3f}")

        if alignment in BAD_ALIGNMENT:
            decision, score = "uncertain_review", 0.0
            evidence.append("bad_alignment")
        elif asr_op == "delete":
            decision, score = "uncertain_review", 0.0
            evidence.append("handled_by_deletion_detector")
        elif common and severity == "low":
            decision = "acceptable_accent" if not low_acoustic else "uncertain_review"
            score = 0.25 if decision == "acceptable_accent" else 0.5
        elif common and intelligibility and low_acoustic:
            decision, score = "mispronounced", 0.9
        elif phone_difference and low_acoustic:
            decision, score = "mispronounced", 0.85
        elif asr_op == "substitute" and low_acoustic:
            decision, score = "mispronounced", 0.8
        elif common and not intelligibility:
            decision, score = ("acceptable_accent", 0.35) if not low_acoustic else ("uncertain_review", 0.55)
        elif phone_difference or asr_op == "substitute" or moderate_acoustic:
            decision, score = "uncertain_review", 0.55
        else:
            decision, score = "correct", 0.0
        decisions.append(decision)
        scores.append(score)
        types.append(confusion["confusion_type"])
        severities.append(severity)
        summaries.append(";".join(evidence))
    out["mispronunciation_score"] = scores
    out["mispronunciation_decision"] = decisions
    out["mandarin_confusion_type"] = types
    out["mandarin_confusion_severity"] = severities
    out["mispronunciation_evidence"] = summaries
    return out


def _strongest_confusion(target: list[str], predicted: list[str], config: Path | None) -> dict:
    best = classify_mandarin_confusion("", "")
    count = max(len(target), len(predicted))
    for index in range(count):
        target_phone = target[index] if index < len(target) else ""
        predicted_phone = predicted[index] if index < len(predicted) else ""
        position = "initial" if index == 0 else ("final" if index == count - 1 else "medial")
        result = classify_mandarin_confusion(
            target_phone,
            predicted_phone,
            position,
            config_path=config,
        )
        if SEVERITY_RANK.get(str(result["severity"]), 0) > SEVERITY_RANK.get(str(best["severity"]), 0):
            best = result
    return best


def _phones(value: object) -> list[str]:
    text = "" if value is None or pd.isna(value) else str(value)
    return [token.strip().upper().rstrip("0123456789") for token in text.split() if token.strip()]


def _number(value: object) -> float:
    try:
        return float("nan") if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return float("nan")
