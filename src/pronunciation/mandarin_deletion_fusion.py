from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "mandarin_deletion_fusion_v2.joblib"

FEATURE_COLUMNS = [
    "asr_missing_word",
    "asr_substituted_word",
    "asr_confidence",
    "asr_context_support",
    "asr_missing_confidence",
    "ctc_deletion_available",
    "ctc_deletion_score",
    "ctc_deletion_margin",
    "ctc_greedy_missing_word",
    "ctc_greedy_substituted_word",
    "ctc_greedy_context_support",
    "recognizer_deletion_agreement",
    "recognizer_substitution_disagreement",
    "phone_count",
    "word_length",
    "relative_word_position",
    "is_function_word",
]

FUNCTION_WORDS = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BUT", "BY", "FOR", "FROM",
    "HE", "HER", "HIM", "HIS", "I", "IN", "IS", "IT", "ITS", "ME", "MY",
    "OF", "ON", "OR", "OUR", "SHE", "THAT", "THE", "THEIR", "THEM", "THEY",
    "THIS", "TO", "US", "WAS", "WE", "WERE", "WITH", "YOU", "YOUR",
}


def build_mandarin_deletion_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stable word-level features available in both training and inference."""
    source = frame.copy()
    index = source.index
    word = source.get("word", pd.Series("", index=index)).fillna("").astype(str).str.upper()
    word_index = pd.to_numeric(source.get("word_index", pd.Series(0, index=index)), errors="coerce").fillna(0.0)
    utterance_max = _utterance_max_word_index(source, word_index)
    features = pd.DataFrame(index=index)
    for column in [
        "asr_missing_word",
        "asr_substituted_word",
        "ctc_deletion_available",
        "ctc_greedy_missing_word",
        "ctc_greedy_substituted_word",
    ]:
        features[column] = _bool_series(source.get(column, pd.Series(False, index=index))).astype(float)
    for column in [
        "asr_confidence",
        "asr_context_support",
        "asr_missing_confidence",
        "ctc_deletion_score",
        "ctc_deletion_margin",
        "ctc_greedy_context_support",
        "phone_count",
    ]:
        features[column] = pd.to_numeric(
            source.get(column, pd.Series(float("nan"), index=index)),
            errors="coerce",
        )
    features["ctc_deletion_margin"] = features["ctc_deletion_margin"].clip(-30.0, 30.0)
    features["recognizer_deletion_agreement"] = (
        features["asr_missing_word"] * features["ctc_greedy_missing_word"]
    )
    features["recognizer_substitution_disagreement"] = (
        features["asr_substituted_word"] * features["ctc_greedy_missing_word"]
    )
    features["word_length"] = word.str.replace("'", "", regex=False).str.len().astype(float)
    features["relative_word_position"] = word_index / utterance_max.clip(lower=1.0)
    features["is_function_word"] = word.isin(FUNCTION_WORDS).astype(float)
    return features[FEATURE_COLUMNS]


def add_mandarin_deletion_fusion_scores(
    frame: pd.DataFrame,
    model_path: str | Path | None,
) -> pd.DataFrame:
    out = frame.copy()
    defaults: dict[str, object] = {
        "mandarin_deletion_model_available": False,
        "mandarin_deletion_probability": float("nan"),
        "mandarin_deletion_model": "",
        "mandarin_deletion_threshold": float("nan"),
        "mandarin_possible_deletion_threshold": float("nan"),
        "mandarin_deletion_model_error": "",
    }
    for column, default in defaults.items():
        out[column] = default
    if model_path is None:
        return out
    path = Path(model_path)
    if not path.is_file():
        out["mandarin_deletion_model_error"] = f"model_not_found:{path}"
        return out
    try:
        artifact = _load_artifact(str(path.resolve()))
        model = artifact["model"]
        features = build_mandarin_deletion_features(out)
        probability = model.predict_proba(features)[:, 1]
        thresholds = artifact.get("thresholds", {})
        out["mandarin_deletion_model_available"] = True
        out["mandarin_deletion_probability"] = np.asarray(probability, dtype=float)
        out["mandarin_deletion_model"] = str(artifact.get("name", path.stem))
        out["mandarin_deletion_threshold"] = float(thresholds.get("deletion", 0.80))
        out["mandarin_possible_deletion_threshold"] = float(thresholds.get("possible_deletion", 0.45))
    except Exception as error:
        out["mandarin_deletion_model_error"] = f"{type(error).__name__}:{error}"
    return out


@lru_cache(maxsize=4)
def _load_artifact(path: str) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("invalid Mandarin deletion fusion artifact")
    expected = artifact.get("feature_columns", FEATURE_COLUMNS)
    if list(expected) != FEATURE_COLUMNS:
        raise ValueError("fusion model feature schema does not match runtime")
    return artifact


def _utterance_max_word_index(frame: pd.DataFrame, word_index: pd.Series) -> pd.Series:
    if "utterance_id" in frame.columns:
        maxima = word_index.groupby(frame["utterance_id"].fillna("").astype(str)).transform("max")
        return maxima.fillna(word_index.max())
    maximum = float(word_index.max()) if len(word_index) else 1.0
    return pd.Series(maximum, index=frame.index, dtype=float)


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(lambda value: value is True or str(value).strip().lower() in {"1", "true", "yes", "y"})
