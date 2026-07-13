from __future__ import annotations

import math

import pandas as pd


SCORE_COLUMNS = ("ctc_gop_score", "gop_score", "phone_score")
PREDICTED_PHONE_COLUMNS = ("predicted_phone", "perceived_phone", "recognized_phone")


def aggregate_word_ctc_gop(phone_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate optional CTC/GOP evidence without fabricating unavailable scores."""
    if phone_frame.empty:
        return pd.DataFrame(columns=_columns())
    rows = []
    groups = ["word_index", "word"] if "word_index" in phone_frame.columns else ["word"]
    for keys, group in phone_frame.groupby(groups, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        word_index = keys[0] if len(groups) > 1 else group.get("word_index", pd.Series([0])).iloc[0]
        word = keys[-1]
        target = [_phone(value) for value in group.get("target_phone", pd.Series("", index=group.index))]
        predicted_col = next((col for col in PREDICTED_PHONE_COLUMNS if col in group.columns), None)
        predicted = [_phone(value) for value in group[predicted_col]] if predicted_col else []
        score_col = next((col for col in SCORE_COLUMNS if col in group.columns), None)
        scores = pd.Series(dtype=float)
        if score_col:
            scores = pd.to_numeric(group[score_col], errors="coerce").dropna().map(_normalize_score)
        blank_ratio = _number(group, "ctc_blank_ratio")
        rows.append(
            {
                "word": word,
                "word_index": word_index,
                "target_phone_seq": " ".join(value for value in target if value),
                "predicted_phone_seq": " ".join(value for value in predicted if value),
                "ctc_gop_score": float(scores.mean()) if len(scores) else float("nan"),
                "min_phone_score": float(scores.min()) if len(scores) else float("nan"),
                "avg_phone_score": float(scores.mean()) if len(scores) else float("nan"),
                "ctc_blank_ratio": blank_ratio,
                "ctc_features_available": bool(len(scores) or any(predicted) or pd.notna(blank_ratio)),
                "ctc_score_source": score_col or "unavailable",
            }
        )
    return pd.DataFrame(rows, columns=_columns())


def _normalize_score(value: float) -> float:
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    if 0.0 <= value <= 2.0:
        return value / 2.0
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _number(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _phone(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    return text.rstrip("0123456789")


def _columns() -> list[str]:
    return [
        "word",
        "word_index",
        "target_phone_seq",
        "predicted_phone_seq",
        "ctc_gop_score",
        "min_phone_score",
        "avg_phone_score",
        "ctc_blank_ratio",
        "ctc_features_available",
        "ctc_score_source",
    ]
