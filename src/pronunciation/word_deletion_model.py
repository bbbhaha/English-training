from __future__ import annotations

import pandas as pd


BAD_ALIGNMENT = {"bad", "failed", "alignment_failed"}


def word_deletion_detector(
    word_summary: pd.DataFrame,
    asr_consistency: pd.DataFrame | None = None,
    ctc_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = word_summary.copy()
    out = _merge(out, asr_consistency, _asr_columns())
    out = _merge(out, ctc_features, ["word_index", "ctc_blank_ratio"])
    for column, default in {
        "phone_count": 0,
        "word_duration_ms": 0.0,
        "alignment_quality": "",
        "asr_word_status": "uncertain",
        "asr_edit_op": "uncertain",
        "asr_missing_word": False,
        "asr_substituted_word": False,
        "asr_confidence": 0.0,
        "ctc_blank_ratio": float("nan"),
    }.items():
        if column not in out.columns:
            out[column] = default

    decisions: list[str] = []
    scores: list[float] = []
    evidence: list[str] = []
    for _, row in out.iterrows():
        count = int(_number(row.get("phone_count"), 0.0))
        duration = _number(row.get("word_duration_ms"), 0.0)
        missing = _truthy(row.get("asr_missing_word"))
        alignment = str(row.get("alignment_quality", "")).strip().lower()
        strong_threshold = max(120.0, count * 30.0)
        weak_threshold = max(80.0, count * 20.0)
        blank_ratio = _number(row.get("ctc_blank_ratio"), float("nan"))
        row_evidence = [f"duration_ms={duration:.1f}", f"phone_count={count}"]
        if pd.notna(blank_ratio):
            row_evidence.append(f"ctc_blank_ratio={blank_ratio:.3f}")
        if missing and duration < strong_threshold:
            decision, score = "deletion", 1.0
            row_evidence.append("asr_missing_and_short")
        elif missing:
            decision, score = "possible_deletion", 0.75
            row_evidence.append("asr_missing")
        elif alignment in BAD_ALIGNMENT:
            decision, score = "alignment_issue", 0.0
            row_evidence.append("bad_alignment")
        elif duration < weak_threshold:
            decision, score = "possible_deletion", 0.6
            row_evidence.append("extreme_duration_compression")
        else:
            decision, score = "correct", 0.0
        decisions.append(decision)
        scores.append(score)
        evidence.append(";".join(row_evidence))
    out["deletion_score"] = scores
    out["deletion_decision"] = decisions
    out["deletion_evidence"] = evidence
    return out


def _merge(frame: pd.DataFrame, extra: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if extra is None or extra.empty or "word_index" not in frame.columns or "word_index" not in extra.columns:
        return frame
    available = [column for column in columns if column in extra.columns]
    source = extra
    if "word" in source.columns:
        source = source[source["word"].fillna("").astype(str).str.strip().ne("")]
    right = source[available].drop_duplicates("word_index", keep="last")
    return frame.drop(columns=[column for column in available if column != "word_index" and column in frame.columns]).merge(
        right,
        on="word_index",
        how="left",
    )


def _asr_columns() -> list[str]:
    return [
        "word_index",
        "asr_word_status",
        "asr_edit_op",
        "asr_missing_word",
        "asr_substituted_word",
        "asr_confidence",
        "recognized_word",
    ]


def _number(value: object, default: float) -> float:
    try:
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "y"}
