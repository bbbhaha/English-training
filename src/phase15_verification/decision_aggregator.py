from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import add_evidence_columns, audit_reason, normalize_group_key
from .labels import main_error_decision, main_error_score


def aggregate_decisions(frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or {}).get("aggregator", config or {})
    mode = str(cfg.get("aggregator_mode", "weighted_vote"))
    min_error_score = float(cfg.get("min_main_error_score", 0.35))
    min_votes = int(cfg.get("min_support_votes", 2))
    accent_min_error = float(cfg.get("acceptable_accent_min_error_score", 0.20))
    out = frame.copy()
    main_scores = main_error_score(out)
    main_errors = main_error_decision(out)
    final_decisions: list[str] = []
    support_votes: list[int] = []
    final_scores: list[float] = []
    for i, row in out.iterrows():
        votes = 0
        if str(row.get("attribute_verifier_decision", "")) == "high_confidence_error":
            votes += 1
        if str(row.get("retrieval_verifier_decision", "")) == "likely_error":
            votes += 1
        if str(row.get("oneclass_verifier_decision", "")) == "high_confidence_error":
            votes += 1
        attr = float(row.get("attribute_risk_score", 0.0) or 0.0)
        margin = float(row.get("proto_margin", 0.0) or 0.0)
        anomaly = float(row.get("oneclass_anomaly_score", 0.0) or 0.0)
        retrieval_error_score = float(np.clip(0.5 - margin, 0.0, 1.0))
        score = float(np.clip(0.55 * main_scores.loc[i] + 0.20 * attr + 0.15 * anomaly + 0.10 * retrieval_error_score, 0.0, 1.0))
        if main_errors.loc[i] and main_scores.loc[i] >= min_error_score and votes >= min_votes:
            decision = "high_confidence_error"
        elif main_errors.loc[i]:
            decision = "uncertain_review"
        elif score >= accent_min_error:
            decision = "acceptable_accent"
        else:
            decision = "correct"
        support_votes.append(votes)
        final_scores.append(round(score, 6))
        final_decisions.append(decision)
    out["main_model_error_score"] = main_scores.round(6)
    out["main_model_error_decision"] = main_errors.astype(int)
    out["verification_support_votes"] = support_votes
    out["final_error_score"] = final_scores
    out["final_decision"] = final_decisions
    if mode == "strict_consensus":
        out = _apply_strict_consensus(out, config or {})
    out["final_binary_high_precision"] = (out["final_decision"] == "high_confidence_error").astype(int)
    out["final_binary_review_as_error"] = out["final_decision"].isin(["high_confidence_error", "uncertain_review"]).astype(int)
    return out


def _apply_strict_consensus(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = add_evidence_columns(frame, config)
    agg_cfg = config.get("aggregator", {})
    strict = config.get("strict_consensus", {})
    group_thresholds = config.get("phone_group_thresholds", {})
    global_score_threshold = float(agg_cfg.get("final_error_score_threshold", agg_cfg.get("min_main_error_score", 0.35)))
    default_min_evidence = int(strict.get("min_evidence_count", agg_cfg.get("min_support_votes", 2)))
    allow_review = bool(strict.get("allow_review_alignment", False))
    require_main = bool(strict.get("require_main_model_error", True))
    enabled_groups = {normalize_group_key(v) for v in strict.get("enabled_phone_groups", [])}
    enabled_phones = {str(v).upper() for v in strict.get("enabled_target_phones", [])}
    decisions: list[str] = []
    thresholds: list[float] = []
    mins: list[int] = []
    for _, row in out.iterrows():
        group = normalize_group_key(row.get("phone_group", ""))
        phone = str(row.get("target_phone", "")).upper()
        group_key = "final_consonant" if int(row.get("duration_outlier_flag", 0)) == 1 and group != "vowel" else group
        group_cfg = group_thresholds.get(group_key, group_thresholds.get(group, {}))
        score_threshold = float(group_cfg.get("final_error_score_threshold", global_score_threshold))
        min_evidence = int(group_cfg.get("min_evidence_count", default_min_evidence))
        thresholds.append(score_threshold)
        mins.append(min_evidence)
        phone_enabled = not enabled_phones or phone in enabled_phones
        group_enabled = not enabled_groups or group in enabled_groups or group_key in enabled_groups
        main_ok = int(row.get("main_model_error_flag", 0)) == 1 or float(row.get("main_model_error_score", 0.0)) >= score_threshold
        if require_main and not main_ok:
            main_ok = False
        alignment_ok = allow_review or int(row.get("alignment_review_flag", 0)) == 0
        enough_evidence = int(row.get("verifier_evidence_count", 0)) >= min_evidence
        score_ok = float(row.get("final_error_score", 0.0)) >= score_threshold
        if main_ok and enough_evidence and alignment_ok and phone_enabled and group_enabled and score_ok:
            decisions.append("high_confidence_error")
        elif int(row.get("main_model_error_flag", 0)) == 1:
            decisions.append("uncertain_review")
        elif float(row.get("final_error_score", 0.0)) >= float(agg_cfg.get("acceptable_accent_min_error_score", 0.20)):
            decisions.append("acceptable_accent")
        else:
            decisions.append("correct")
    out["final_error_score_threshold"] = thresholds
    out["strict_min_evidence_count"] = mins
    out["final_decision"] = decisions
    out["audit_reason"] = [audit_reason(row) for _, row in out.iterrows()]
    return out
