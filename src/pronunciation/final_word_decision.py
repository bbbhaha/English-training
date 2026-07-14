from __future__ import annotations

import pandas as pd

from pronunciation.ctc_gop import aggregate_word_ctc_gop
from pronunciation.word_deletion_model import word_deletion_detector
from pronunciation.word_mispronunciation_model import word_mispronunciation_detector


def run_word_level_diagnosis(
    phone_frame: pd.DataFrame,
    word_summary: pd.DataFrame,
    asr_consistency: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ctc_features = aggregate_word_ctc_gop(phone_frame)
    base = _merge_features(word_summary, ctc_features)
    base = _merge_features(base, _aggregate_phone_debug(phone_frame))
    deletion = word_deletion_detector(base, asr_consistency, ctc_features)
    mispronunciation = word_mispronunciation_detector(deletion)
    return fuse_word_decisions(mispronunciation)


def fuse_word_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, default in {
        "deletion_decision": "correct",
        "mispronunciation_decision": "correct",
        "deletion_evidence": "",
        "mispronunciation_evidence": "",
    }.items():
        if column not in out.columns:
            out[column] = default
    final_decisions, error_types, evidence = [], [], []
    for _, row in out.iterrows():
        deletion = str(row.get("deletion_decision", "correct"))
        mispronunciation = str(row.get("mispronunciation_decision", "correct"))
        lexicon_status = str(row.get("lexicon_status", "")).strip().lower()
        if lexicon_status == "failed":
            final = "g2p_issue"
        elif deletion == "deletion":
            final = "deletion"
        elif deletion == "possible_deletion":
            final = "possible_deletion"
        elif deletion == "alignment_issue":
            final = "alignment_issue"
        elif mispronunciation == "mispronounced":
            final = "mispronounced"
        elif mispronunciation == "acceptable_accent":
            final = "acceptable_accent"
        elif mispronunciation == "uncertain_review":
            final = "uncertain_review"
        else:
            final = "correct"
        final_decisions.append(final)
        error_types.append("" if final == "correct" else final)
        evidence.append(_join_evidence(row.get("deletion_evidence", ""), row.get("mispronunciation_evidence", "")))
    out["final_word_decision"] = final_decisions
    out["final_error_type"] = error_types
    out["evidence_summary"] = evidence
    return out


def merge_word_diagnosis_into_phones(phone_frame: pd.DataFrame, word_frame: pd.DataFrame) -> pd.DataFrame:
    if phone_frame.empty or word_frame.empty or "word_index" not in phone_frame.columns:
        return phone_frame.copy()
    fields = [
        "word_index",
        "asr_word_status",
        "asr_edit_op",
        "asr_missing_word",
        "deletion_score",
        "deletion_decision",
        "mispronunciation_score",
        "mispronunciation_decision",
        "final_word_decision",
        "final_error_type",
        "mandarin_confusion_type",
        "mandarin_confusion_severity",
        "evidence_summary",
    ]
    available = [column for column in fields if column in word_frame.columns]
    out = phone_frame.drop(columns=[column for column in available if column != "word_index" and column in phone_frame.columns])
    out = out.merge(word_frame[available].drop_duplicates("word_index", keep="last"), on="word_index", how="left")
    final = out.get("final_word_decision", pd.Series("correct", index=out.index)).fillna("correct").astype(str)
    mapping = {
        "deletion": ("true_error", "deletion", "word_deletion_detected"),
        "possible_deletion": ("uncertain_review", "possible_deletion", "possible_word_deletion"),
        "alignment_issue": ("uncertain_review", "alignment_issue", "bad_alignment"),
        "mispronounced": ("true_error", "mispronounced", "word_mispronunciation_detected"),
        "acceptable_accent": ("acceptable_accent", "acceptable_accent", "common_mandarin_accent_pattern"),
        "uncertain_review": ("uncertain_review", "mispronunciation_uncertain", "insufficient_word_mispronunciation_evidence"),
        "correct": ("correct", "", ""),
        "g2p_issue": ("uncertain_review", "g2p_issue", "g2p_failed"),
    }
    for label, (decision, error_type, reason) in mapping.items():
        mask = final.eq(label)
        out.loc[mask, "decision"] = decision
        out.loc[mask, "error_type"] = error_type
        out.loc[mask, "review_reason"] = reason
    return out


def _merge_features(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty or right.empty or "word_index" not in left.columns or "word_index" not in right.columns:
        return left.copy()
    columns = [column for column in right.columns if column not in {"word"}]
    return left.drop(columns=[column for column in columns if column != "word_index" and column in left.columns]).merge(
        right[columns].drop_duplicates("word_index", keep="last"),
        on="word_index",
        how="left",
    )


def _aggregate_phone_debug(phone_frame: pd.DataFrame) -> pd.DataFrame:
    if phone_frame.empty or "word_index" not in phone_frame.columns:
        return pd.DataFrame()
    rows = []
    for word_index, group in phone_frame.groupby("word_index", sort=False, dropna=False):
        legacy = pd.to_numeric(
            group.get("manual_calibrated_error_probability", pd.Series(dtype=float)),
            errors="coerce",
        )
        phone_groups = sorted(
            value for value in group.get("phone_group", pd.Series(dtype=str)).fillna("").astype(str).unique() if value
        )
        rows.append(
            {
                "word_index": word_index,
                "utterance_id": _first_text(group, "utterance_id"),
                "speaker_id": _first_text(group, "speaker_id"),
                "manual_calibrated_error_probability": float(legacy.mean()) if legacy.notna().any() else float("nan"),
                "phone_group": "+".join(phone_groups),
                "lexicon_status": _first_text(group, "lexicon_status"),
                "g2p_source": _first_text(group, "g2p_source"),
                "g2p_confidence": _first_text(group, "g2p_confidence"),
                "g2p_status": _first_text(group, "g2p_status"),
                "g2p_error": _first_text(group, "g2p_error"),
                "pronunciation_variant_id": _first_text(group, "pronunciation_variant_id"),
                "num_pronunciation_variants": _first_text(group, "num_pronunciation_variants"),
                "selected_pronunciation": _first_text(group, "selected_pronunciation"),
            }
        )
    return pd.DataFrame(rows)


def _first_text(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns or group.empty:
        return ""
    values = group[column].dropna().astype(str)
    return values.iloc[0] if len(values) else ""


def _join_evidence(*values: object) -> str:
    parts = []
    for value in values:
        text = "" if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else str(value)
        for item in text.split(";"):
            item = item.strip()
            if item and item not in parts:
                parts.append(item)
    return ";".join(parts)
