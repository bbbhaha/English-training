from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .features import build_feature_matrix
from .labels import main_error_decision
from .phone_attributes import confusable_phones, normalize_phone


@dataclass
class PrototypeBank:
    vectors: np.ndarray
    phones: np.ndarray
    groups: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]


def _correct_mask(frame: pd.DataFrame) -> pd.Series:
    if "gold_binary" in frame.columns:
        return pd.to_numeric(frame["gold_binary"], errors="coerce") == 1
    if "label" in frame.columns:
        return pd.to_numeric(frame["label"], errors="coerce") == 0
    if "gold_label" in frame.columns:
        return pd.to_numeric(frame["gold_label"], errors="coerce") == 0
    raise ValueError("Training manifest needs one of gold_binary, label, or gold_label to identify correct samples.")


def build_prototype_bank(train: pd.DataFrame, mapping: dict) -> PrototypeBank:
    if "target_phone" not in train.columns:
        raise ValueError("Training manifest is missing target_phone.")
    correct = train[_correct_mask(train)].copy()
    if correct.empty:
        raise ValueError("No correct training rows are available for prototype retrieval.")
    vectors, scaler, feature_names = build_feature_matrix(correct, mapping)
    groups = correct.get("phone_group", pd.Series("", index=correct.index)).astype(str).to_numpy()
    phones = correct["target_phone"].map(normalize_phone).to_numpy()
    return PrototypeBank(vectors=vectors, phones=phones, groups=groups, scaler=scaler, feature_names=feature_names)


def save_prototype_bank(bank: PrototypeBank, path) -> None:
    joblib.dump(bank, path)


def load_prototype_bank(path) -> PrototypeBank:
    return joblib.load(path)


def _top_scores(query: np.ndarray, candidates: np.ndarray, top_k: int) -> tuple[float, float]:
    if len(candidates) == 0:
        return 0.0, 0.0
    sims = cosine_similarity(query.reshape(1, -1), candidates).ravel()
    order = np.sort(sims)[::-1]
    return float(order[0]), float(order[: min(top_k, len(order))].mean())


def apply_retrieval_verifier(
    frame: pd.DataFrame,
    bank: PrototypeBank,
    mapping: dict,
    config: dict | None = None,
) -> pd.DataFrame:
    cfg = (config or {}).get("retrieval", config or {})
    top_k = int(cfg.get("top_k", 5))
    low_sim = float(cfg.get("low_similarity_threshold", 0.15))
    margin_threshold = float(cfg.get("positive_margin_threshold", 0.03))
    out = frame.copy()
    vectors, _, _ = build_feature_matrix(out, mapping, fit_scaler=bank.scaler, feature_names=bank.feature_names)
    main_errors = main_error_decision(out)
    same_top1: list[float] = []
    same_topk: list[float] = []
    conf_top1: list[float] = []
    margins: list[float] = []
    decisions: list[str] = []
    frame_groups = out.get("phone_group", pd.Series("", index=out.index)).astype(str).to_numpy()
    for i, row in enumerate(out.itertuples(index=False)):
        phone = normalize_phone(getattr(row, "target_phone"))
        group = frame_groups[i]
        same_mask = bank.phones == phone
        if same_mask.sum() == 0 and group:
            same_mask = bank.groups == group
        confusions = confusable_phones(phone)
        conf_mask = np.isin(bank.phones, confusions) if confusions else np.zeros(len(bank.phones), dtype=bool)
        s1, sk = _top_scores(vectors[i], bank.vectors[same_mask], top_k)
        c1, _ = _top_scores(vectors[i], bank.vectors[conf_mask], top_k)
        margin = s1 - c1
        same_top1.append(round(s1, 6))
        same_topk.append(round(sk, 6))
        conf_top1.append(round(c1, 6))
        margins.append(round(margin, 6))
        if main_errors.iloc[i] and c1 > s1:
            decisions.append("likely_error")
        elif main_errors.iloc[i] and s1 >= low_sim and margin >= margin_threshold:
            decisions.append("likely_correct")
        elif main_errors.iloc[i]:
            decisions.append("uncertain_review")
        elif s1 < low_sim:
            decisions.append("candidate_error_for_review")
        else:
            decisions.append("likely_correct")
    out["proto_same_phone_sim_top1"] = same_top1
    out["proto_same_phone_sim_topk_mean"] = same_topk
    out["proto_confusion_phone_sim_top1"] = conf_top1
    out["proto_margin"] = margins
    out["retrieval_verifier_decision"] = decisions
    return out
