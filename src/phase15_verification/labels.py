from __future__ import annotations

import numpy as np
import pandas as pd


def speechocean_score_to_label(score: object, mode: str) -> str | int:
    value = int(score)
    if mode == "three_class":
        return {2: "correct", 1: "acceptable_accent", 0: "error"}[value]
    if mode == "binary_recognition":
        return 1 if value == 0 else 0
    if mode == "strict_standardness":
        return 0 if value == 2 else 1
    raise ValueError(f"Unknown label_mode: {mode}")


def infer_error_labels(frame: pd.DataFrame, label_col: str, error_value: str = "auto") -> np.ndarray:
    if label_col not in frame.columns:
        raise ValueError(f"Missing label column: {label_col}")
    values = frame[label_col]
    if error_value != "auto":
        return (values.astype(str) == str(error_value)).to_numpy(dtype=int)
    name = label_col.lower()
    if name in {"gold_binary", "is_correct", "correct"}:
        return (pd.to_numeric(values, errors="coerce") == 0).to_numpy(dtype=int)
    if name in {"gold_label", "label", "error_label", "is_error"}:
        return (pd.to_numeric(values, errors="coerce") == 1).to_numpy(dtype=int)
    lowered = values.astype(str).str.lower()
    if lowered.isin(["error", "mispronounced", "needs_attention"]).any():
        return lowered.isin(["error", "mispronounced", "needs_attention"]).to_numpy(dtype=int)
    numeric = pd.to_numeric(values, errors="coerce")
    return (numeric == 1).to_numpy(dtype=int)


def main_error_score(frame: pd.DataFrame) -> pd.Series:
    if "main_model_error_probability" in frame.columns:
        return pd.to_numeric(frame["main_model_error_probability"], errors="coerce").fillna(0.0).clip(0, 1)
    if "final_error_score" in frame.columns:
        return pd.to_numeric(frame["final_error_score"], errors="coerce").fillna(0.0).clip(0, 1)
    if "mispronounced_probability" in frame.columns:
        return pd.to_numeric(frame["mispronounced_probability"], errors="coerce").fillna(0.0).clip(0, 1)
    if "prob_error" in frame.columns:
        return pd.to_numeric(frame["prob_error"], errors="coerce").fillna(0.0).clip(0, 1)
    if "prob_correct" in frame.columns:
        return (1.0 - pd.to_numeric(frame["prob_correct"], errors="coerce").fillna(1.0)).clip(0, 1)
    if "confidence" in frame.columns and "prediction" in frame.columns:
        confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.5).clip(0, 1)
        pred = pd.to_numeric(frame["prediction"], errors="coerce").fillna(1)
        return np.where(pred == 0, confidence, 1.0 - confidence)
    return pd.Series(np.zeros(len(frame)), index=frame.index)


def main_error_decision(frame: pd.DataFrame) -> pd.Series:
    if "main_model_error_decision" in frame.columns:
        return frame["main_model_error_decision"].astype(int)
    if "prediction" in frame.columns:
        pred = pd.to_numeric(frame["prediction"], errors="coerce").fillna(1)
        if "prob_correct" in frame.columns or "gold_binary" in frame.columns:
            return (pred == 0).astype(int)
        return (pred == 1).astype(int)
    return (main_error_score(frame) >= 0.5).astype(int)
