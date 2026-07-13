from __future__ import annotations

import pandas as pd


def add_verifier_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model_error_score" not in out.columns:
        prob_correct = pd.to_numeric(out.get("prob_correct", 0.5), errors="coerce").fillna(0.5)
        out["model_error_score"] = (1.0 - prob_correct).clip(0, 1)
    out["final_error_score"] = out["model_error_score"]
    out["final_decision"] = "uncertain_review"
    out["main_model_error_score"] = out["model_error_score"]
    out["main_model_error_decision"] = (pd.to_numeric(out["model_error_score"], errors="coerce").fillna(0.0) >= 0.5).astype(int)
    defaults = {
        "attribute_risk_score": 0.0,
        "attribute_mismatch_count": 0,
        "attribute_verifier_decision": "not_run",
        "proto_same_phone_sim_top1": 0.0,
        "proto_same_phone_sim_topk_mean": 0.0,
        "proto_confusion_phone_sim_top1": 0.0,
        "proto_margin": 0.0,
        "retrieval_verifier_decision": "not_run",
        "oneclass_anomaly_score": 0.0,
        "oneclass_verifier_decision": "not_run",
        "verification_support_votes": 0,
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    return out

