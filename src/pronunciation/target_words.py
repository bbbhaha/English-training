from __future__ import annotations

import re

import pandas as pd


TARGET_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def build_target_word_table(text: str, utterance_id: str = "") -> pd.DataFrame:
    """Build the canonical word table directly from the original target text."""
    rows = []
    for word_index, match in enumerate(TARGET_WORD_RE.finditer(str(text))):
        original = match.group(0)
        normalized = original.replace("’", "'").upper()
        rows.append(
            {
                "utterance_id": utterance_id,
                "word_index": word_index,
                "word": normalized,
                "normalized_word": normalized,
                "original_word": original,
                "char_start": match.start(),
                "char_end": match.end(),
            }
        )
    return pd.DataFrame(rows, columns=_target_columns())


def ensure_word_summary_coverage(
    target_words: pd.DataFrame,
    word_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Left join diagnostics onto the canonical target words without dropping words."""
    target = target_words.copy()
    summary = word_summary.copy()
    if target.empty:
        return summary
    if "word_index" not in summary.columns:
        summary = pd.DataFrame(columns=["word_index"])
    target["word_index"] = pd.to_numeric(target["word_index"], errors="raise").astype(int)
    summary["word_index"] = pd.to_numeric(summary["word_index"], errors="coerce").astype("Int64")
    target_owned = (
        "word",
        "utterance_id",
        "normalized_word",
        "original_word",
        "char_start",
        "char_end",
    )
    summary = summary.drop(columns=[column for column in target_owned if column in summary.columns])
    out = target.merge(summary.drop_duplicates("word_index", keep="last"), on="word_index", how="left")
    missing = out["alignment_quality"].isna() if "alignment_quality" in out.columns else pd.Series(True, index=out.index)
    defaults = {
        "phone_count": 0,
        "word_duration_ms": float("nan"),
        "word_decision": "uncertain_review",
        "error_type": "alignment_issue",
        "alignment_quality": "bad",
        "missing_word_reason": "missing_from_prediction_pipeline",
        "possible_missing_word": False,
        "deletion_trigger_source": "none",
        "num_phone_true_error": 0,
        "num_phone_uncertain": 1,
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
        elif column in {"phone_count", "num_phone_true_error", "num_phone_uncertain"}:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default).astype(int)
        elif column == "possible_missing_word":
            out[column] = out[column].fillna(False).astype(bool)
        else:
            out.loc[missing & out[column].isna(), column] = default
    return out.sort_values("word_index", kind="stable").reset_index(drop=True)


def _target_columns() -> list[str]:
    return [
        "utterance_id",
        "word_index",
        "word",
        "normalized_word",
        "original_word",
        "char_start",
        "char_end",
    ]
