from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .phone_attributes import expected_attributes, normalize_phone


NUMERIC_CANDIDATES = [
    "duration_ms",
    "duration",
    "gop_score",
    "evidence_score",
    "target_log_likelihood",
    "competitor_log_likelihood",
    "alignment_score",
    "prob_correct",
    "mispronounced_probability",
    "confidence",
]

ATTR_FEATURES = [
    "attr_is_vowel",
    "attr_voiced",
    "attr_final_consonant",
    "attr_phone_hash",
]


def add_final_consonant_flag(frame: pd.DataFrame, mapping: dict) -> pd.Series:
    if "phone_index" not in frame.columns:
        return pd.Series(False, index=frame.index)
    group_cols = [c for c in ["utterance_id", "utt_id", "word"] if c in frame.columns]
    if not group_cols:
        group_cols = ["phone_index"]
    phone_index = pd.to_numeric(frame["phone_index"], errors="coerce").fillna(-1)
    max_index = phone_index.groupby([frame[c].astype(str) for c in group_cols]).transform("max")
    is_final = phone_index == max_index
    is_consonant = frame["target_phone"].map(lambda p: expected_attributes(p, mapping).get("vowel_consonant") == "consonant")
    return (is_final & is_consonant).astype(bool)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC_CANDIDATES if c in frame.columns] + [c for c in frame.columns if c.startswith("w2v_")]


def build_feature_matrix(
    frame: pd.DataFrame,
    mapping: dict,
    fit_scaler: StandardScaler | None = None,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    out = frame.copy()
    attr_set = set(ATTR_FEATURES)
    cols = [c for c in (feature_names or feature_columns(out)) if c not in attr_set]
    if not cols:
        out["_bias_feature"] = 0.0
        cols = ["_bias_feature"]
    numeric = out.reindex(columns=cols, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    final_flags = add_final_consonant_flag(out, mapping)
    attr_rows = []
    for phone, final in zip(out["target_phone"], final_flags):
        attrs = expected_attributes(phone, mapping, bool(final))
        attr_rows.append(
            {
                "attr_is_vowel": 1.0 if attrs.get("vowel_consonant") == "vowel" else 0.0,
                "attr_voiced": 1.0 if attrs.get("voicing") else 0.0,
                "attr_final_consonant": 1.0 if attrs.get("final_consonant") else 0.0,
                "attr_phone_hash": (sum(ord(ch) for ch in normalize_phone(phone)) % 29) / 29.0,
            }
        )
    attrs_df = pd.DataFrame(attr_rows, index=out.index)
    combined = pd.concat([numeric, attrs_df], axis=1)
    if feature_names is not None:
        combined = combined.reindex(columns=feature_names, fill_value=0.0)
    matrix = combined.to_numpy(dtype=float)
    scaler = fit_scaler or StandardScaler()
    if fit_scaler is None:
        matrix = scaler.fit_transform(matrix)
    else:
        matrix = scaler.transform(matrix)
    return np.nan_to_num(matrix), scaler, list(combined.columns)
