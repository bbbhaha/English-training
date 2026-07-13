from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def apply_manual_calibrator(
    frame: pd.DataFrame,
    calibrator_path: Path,
    threshold_override: float | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if not calibrator_path.exists():
        out["manual_calibrated_error_probability"] = 0.5
        out["manual_calibrated_decision"] = "uncertain_review"
        out["manual_calibrated_threshold"] = threshold_override if threshold_override is not None else 0.85
        out["calibration_available"] = False
        return out
    bundle = joblib.load(calibrator_path)
    model = bundle["model"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    for col in numeric:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in categorical:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        out["manual_calibrated_error_probability"] = 0.5
        out["manual_calibrated_decision"] = "uncertain_review"
        out["manual_calibrated_threshold"] = threshold_override if threshold_override is not None else 0.85
        out["calibration_available"] = False
        return out
    scores = model.predict_proba(out[numeric + categorical])[:, classes.index(1)]
    threshold = float(threshold_override if threshold_override is not None else bundle.get("threshold", 0.85))
    out["manual_calibrated_error_probability"] = np.round(np.clip(scores, 0.0, 1.0), 6)
    out["manual_calibrated_threshold"] = threshold
    out["manual_calibrated_decision"] = np.where(scores >= threshold, "true_error", "uncertain_review")
    out["calibration_available"] = True
    return out

