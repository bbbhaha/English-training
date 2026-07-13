from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import build_feature_matrix
from .labels import main_error_decision
from .phone_attributes import normalize_phone
from .prototype_retrieval import _correct_mask


@dataclass
class OneClassBank:
    models: dict[str, IsolationForest]
    scaler: object
    feature_names: list[str]
    fallback_key: str


def train_oneclass_bank(train: pd.DataFrame, mapping: dict, config: dict | None = None) -> OneClassBank:
    cfg = (config or {}).get("oneclass", config or {})
    min_phone = int(cfg.get("min_phone_samples", 20))
    min_group = int(cfg.get("min_group_samples", 50))
    contamination = float(cfg.get("contamination", 0.08))
    n_estimators = int(cfg.get("n_estimators", 50))
    max_samples = int(cfg.get("max_samples", 512))
    correct = train[_correct_mask(train)].copy().reset_index(drop=True)
    vectors, scaler, feature_names = build_feature_matrix(correct, mapping)
    correct["_phone_norm"] = correct["target_phone"].map(normalize_phone)
    correct["_group_key"] = "group:" + correct.get("phone_group", pd.Series("", index=correct.index)).astype(str)
    models: dict[str, IsolationForest] = {}
    for phone, idx in correct.groupby("_phone_norm").groups.items():
        if len(idx) >= min_phone:
            models[f"phone:{phone}"] = _fit(vectors[list(idx)], contamination, n_estimators, max_samples)
    for group, idx in correct.groupby("_group_key").groups.items():
        if len(idx) >= min_group:
            models[str(group)] = _fit(vectors[list(idx)], contamination, n_estimators, max_samples)
    fallback_key = "global"
    models[fallback_key] = _fit(vectors, contamination, n_estimators, max_samples)
    return OneClassBank(models=models, scaler=scaler, feature_names=feature_names, fallback_key=fallback_key)


def _fit(vectors: np.ndarray, contamination: float, n_estimators: int, max_samples: int) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=min(max_samples, len(vectors)),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(vectors)
    return model


def apply_oneclass_verifier(
    frame: pd.DataFrame,
    bank: OneClassBank,
    mapping: dict,
    config: dict | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    vectors, _, _ = build_feature_matrix(out, mapping, fit_scaler=bank.scaler, feature_names=bank.feature_names)
    main_errors = main_error_decision(out)
    scores = np.zeros(len(out), dtype=float)
    groups = out.get("phone_group", pd.Series("", index=out.index)).astype(str).tolist()
    keys: list[str] = []
    for i, row in enumerate(out.itertuples(index=False)):
        phone_key = f"phone:{normalize_phone(getattr(row, 'target_phone'))}"
        group_key = f"group:{groups[i]}"
        keys.append(phone_key if phone_key in bank.models else group_key if group_key in bank.models else bank.fallback_key)
    for key in sorted(set(keys)):
        idx = np.array([i for i, item in enumerate(keys) if item == key], dtype=int)
        # IsolationForest decision_function is higher for normal samples; invert it so higher means anomaly.
        raw = bank.models[key].decision_function(vectors[idx])
        scores[idx] = 1.0 / (1.0 + np.exp(4.0 * raw))
    decisions: list[str] = []
    for i, anomaly in enumerate(scores):
        if main_errors.iloc[i] and anomaly >= 0.55:
            decisions.append("high_confidence_error")
        elif main_errors.iloc[i]:
            decisions.append("uncertain_review")
        elif anomaly >= 0.70:
            decisions.append("candidate_error_for_review")
        else:
            decisions.append("likely_correct")
    out["oneclass_anomaly_score"] = np.round(scores, 6)
    out["oneclass_verifier_decision"] = decisions
    return out
