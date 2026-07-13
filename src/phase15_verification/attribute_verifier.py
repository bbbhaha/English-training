from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_final_consonant_flag
from .labels import main_error_decision, main_error_score
from .phone_attributes import confusable_phones, expected_attributes, serialize_attributes


def apply_attribute_verifier(
    frame: pd.DataFrame,
    mapping: dict,
    config: dict | None = None,
) -> pd.DataFrame:
    cfg = (config or {}).get("attribute", config or {})
    high = float(cfg.get("high_risk_threshold", 0.55))
    weak = float(cfg.get("weak_risk_threshold", 0.30))
    out = frame.copy()
    final_flags = add_final_consonant_flag(out, mapping)
    main_scores = main_error_score(out)
    main_errors = main_error_decision(out)
    vectors: list[str] = []
    risks: list[float] = []
    mismatch_counts: list[int] = []
    decisions: list[str] = []
    durations = pd.to_numeric(out.get("duration_ms", out.get("duration", pd.Series(0, index=out.index))), errors="coerce").fillna(0)
    duration_median = durations.groupby(out["target_phone"].astype(str)).transform("median").replace(0, np.nan)
    duration_ratio = (durations / duration_median).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    for idx, row in out.iterrows():
        attrs = expected_attributes(row.get("target_phone"), mapping, bool(final_flags.loc[idx]))
        vectors.append(serialize_attributes(attrs))
        risky_phone = 1 if confusable_phones(row.get("target_phone")) else 0
        final_short = 1 if attrs.get("final_consonant") and duration_ratio.loc[idx] < 0.55 else 0
        tense_lax = 1 if attrs.get("vowel_consonant") == "vowel" and row.get("target_phone") in {"IY", "IH", "EY", "EH", "AE", "UW", "UH"} else 0
        mismatch = int(risky_phone) + int(final_short) + int(tense_lax and main_scores.loc[idx] >= 0.25)
        risk = min(1.0, 0.20 * risky_phone + 0.35 * final_short + 0.20 * tense_lax + 0.45 * float(main_scores.loc[idx]))
        risks.append(round(float(risk), 6))
        mismatch_counts.append(mismatch)
        if main_errors.loc[idx] and risk >= high:
            decisions.append("high_confidence_error")
        elif main_errors.loc[idx] and risk < weak:
            decisions.append("uncertain_review")
        elif main_errors.loc[idx]:
            decisions.append("possible_error")
        else:
            decisions.append("likely_correct")
    out["expected_attribute_vector"] = vectors
    out["attribute_risk_score"] = risks
    out["attribute_mismatch_count"] = mismatch_counts
    out["attribute_verifier_decision"] = decisions
    return out
