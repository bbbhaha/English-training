#!/usr/bin/env python
"""Extract dual-CTC phone features from the SpeechOcean762 training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_phone_three_state_model import extract_features
from pronunciation.ctc_phone_diagnosis import (
    DEFAULT_PHONE_CTC_MODEL,
    DEFAULT_REFERENCE_PHONE_CTC_MODEL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phones", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=ROOT / "outputs/phone_three_state/speechocean_dual_ctc_features_train.csv",
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        default=ROOT / "outputs/phone_three_state/speechocean_aux_selection.json",
    )
    parser.add_argument("--ctc-model", default=DEFAULT_PHONE_CTC_MODEL)
    parser.add_argument("--reference-ctc-model", default=DEFAULT_REFERENCE_PHONE_CTC_MODEL)
    parser.add_argument("--max-utterances", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = pd.read_csv(args.phones, encoding="utf-8-sig", low_memory=False)
    selected = prepare_speechocean_source(source, max_utterances=args.max_utterances)
    report = {
        "dataset": "SpeechOcean762",
        "source_split": "train",
        "strict_label_policy": {
            "correct": "correct",
            "acceptable": "mispronounced",
            "incorrect": "mispronounced",
        },
        "rows": int(len(selected)),
        "utterances": int(selected["utterance_id"].nunique()),
        "speakers": int(selected["speaker_id"].nunique()),
        "labels": {
            str(key): int(value)
            for key, value in selected["gold_phone_state"].value_counts().to_dict().items()
        },
    }
    args.selection_report.parent.mkdir(parents=True, exist_ok=True)
    args.selection_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    features = extract_features(
        selected,
        args.corpus_root,
        args.ctc_model,
        reference_model_id=args.reference_ctc_model,
        feature_cache=args.feature_cache,
        resume=args.resume,
    )
    print(f"Wrote {len(features)} rows to {args.feature_cache}", flush=True)


def prepare_speechocean_source(frame: pd.DataFrame, *, max_utterances: int = 0) -> pd.DataFrame:
    out = frame.copy()
    split_column = "official_split" if "official_split" in out.columns else "split"
    out = out[out[split_column].astype(str).str.lower().eq("train")].copy()
    out = out[~out["target_phone"].fillna("").astype(str).str.upper().isin(["", "SIL", "SP", "SPN"])].copy()
    out["gold_phone_state"] = out["gold_three_class"].map(
        {
            "correct": "correct",
            "acceptable": "mispronounced",
            "incorrect": "mispronounced",
        }
    )
    out = out[out["gold_phone_state"].notna()].copy()
    out["error_type"] = out["gold_phone_state"].map(
        {"correct": "correct", "mispronounced": "substitution"}
    )
    perceived = out["perceived_phone"] if "perceived_phone" in out.columns else pd.Series("", index=out.index)
    out["perceived_phone"] = perceived.fillna("").astype(str)
    out["speaker_id"] = "SO_" + out["speaker_id"].astype(str)
    out["split"] = "train"

    if max_utterances > 0 and out["utterance_id"].nunique() > max_utterances:
        utterances = (
            out.groupby("utterance_id", sort=False)["gold_phone_state"]
            .agg(error_count=lambda values: int(values.ne("correct").sum()))
            .reset_index()
        )
        error = utterances[utterances["error_count"] > 0].sort_values(
            ["error_count", "utterance_id"],
            ascending=[False, True],
        )
        clean = utterances[utterances["error_count"] == 0].sort_values("utterance_id")
        error_budget = min(len(error), max(1, int(round(max_utterances * 0.8))))
        chosen = error.head(error_budget)["utterance_id"].tolist()
        chosen.extend(clean.head(max_utterances - len(chosen))["utterance_id"].tolist())
        if len(chosen) < max_utterances:
            chosen.extend(
                error.iloc[error_budget:].head(max_utterances - len(chosen))["utterance_id"].tolist()
            )
        out = out[out["utterance_id"].isin(chosen)].copy()

    return out.sort_values(["utterance_id", "phone_index"], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    main()
