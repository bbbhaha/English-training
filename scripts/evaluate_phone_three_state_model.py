#!/usr/bin/env python
"""Evaluate the trained correct/mispronounced/deleted phone classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.ctc_phone_diagnosis import (
    REFERENCE_DELETION_MARGIN_THRESHOLD,
    add_phone_model_consensus_features,
    dual_phone_presence_guard,
    phone_equivalence_guard,
)


PHONE_STATES = ["correct", "mispronounced", "deleted"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.features, encoding="utf-8-sig", keep_default_na=False)
    frame = add_phone_model_consensus_features(frame)
    frame = frame[frame["split"].astype(str).eq(args.split)].copy()
    artifact = joblib.load(args.model)
    columns = list(artifact["feature_columns"])
    for column in artifact.get("numeric_features", []):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    raw = artifact["pipeline"].predict_proba(frame[columns])
    probabilities = pd.DataFrame(raw, columns=artifact["pipeline"].classes_, index=frame.index)
    for state in PHONE_STATES:
        if state not in probabilities.columns:
            probabilities[state] = 0.0
    thresholds = artifact["thresholds"]
    use_reference_hard_gates = bool(artifact.get("use_reference_hard_gates", True))
    predicted = np.full(len(frame), "correct", dtype=object)
    deleted = probabilities["deleted"].to_numpy() >= float(thresholds["deleted"])
    deleted &= ~dual_phone_presence_guard(frame).to_numpy(bool)
    if use_reference_hard_gates and "reference_ctc_phone_model_available" in frame.columns:
        reference_available = (
            frame["reference_ctc_phone_model_available"].fillna(False).astype(bool).to_numpy()
        )
        reference_margin = pd.to_numeric(
            frame["reference_ctc_deletion_margin"],
            errors="coerce",
        ).fillna(float("-inf")).to_numpy(float)
        deleted &= (~reference_available) | (
            reference_margin >= REFERENCE_DELETION_MARGIN_THRESHOLD
        )
    wrong = (~deleted) & (
        probabilities["mispronounced"].to_numpy() >= float(thresholds["mispronounced"])
    )
    guard = phone_equivalence_guard(frame).to_numpy(bool)
    if use_reference_hard_gates and "reference_recognized_phone" in frame.columns:
        guard |= phone_equivalence_guard(
            frame,
            recognized_column="reference_recognized_phone",
        ).to_numpy(bool)
    wrong &= ~guard
    predicted[deleted] = "deleted"
    predicted[wrong] = "mispronounced"
    gold = frame["gold_phone_state"].astype(str).to_numpy()
    report = classification_report(
        gold,
        predicted,
        labels=PHONE_STATES,
        output_dict=True,
        zero_division=0,
    )
    correct = gold == "correct"
    payload = {
        "split": args.split,
        "rows": len(frame),
        "speakers": sorted(frame["speaker_id"].astype(str).unique().tolist()),
        "thresholds": thresholds,
        "use_reference_hard_gates": use_reference_hard_gates,
        "macro_f1": float(f1_score(gold, predicted, labels=PHONE_STATES, average="macro", zero_division=0)),
        "correct_false_alarm_rate": float(np.mean(predicted[correct] != "correct")) if correct.any() else 0.0,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(gold, predicted, labels=PHONE_STATES).tolist(),
        "label_order": PHONE_STATES,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
