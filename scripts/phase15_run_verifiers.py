#!/usr/bin/env python
"""Run Phase-1.5 high-precision verification on an existing prediction table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase15_verification.attribute_verifier import apply_attribute_verifier
from phase15_verification.config import load_config
from phase15_verification.decision_aggregator import aggregate_decisions
from phase15_verification.oneclass_verifier import apply_oneclass_verifier, train_oneclass_bank
from phase15_verification.phone_attributes import load_phone_attributes
from phase15_verification.prototype_retrieval import (
    apply_retrieval_verifier,
    build_prototype_bank,
    load_prototype_bank,
)


REQUIRED_INPUT_COLUMNS = {"target_phone"}
OPTIONAL_USEFUL_COLUMNS = {
    "utterance_id",
    "speaker_id",
    "word",
    "phone_group",
    "phone_index",
    "duration_ms",
    "duration",
    "gold_binary",
    "prediction",
    "prob_correct",
    "confidence",
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"File does not exist: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)


def _validate_input(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(frame.columns))
    if missing:
        raise SystemExit(f"Input prediction table is missing required columns: {missing}")
    useful_missing = sorted(OPTIONAL_USEFUL_COLUMNS - set(frame.columns))
    if useful_missing:
        print(f"Optional columns not found; related verifier signals may be weaker: {useful_missing}")
    if not ({"prob_correct", "mispronounced_probability", "prob_error", "confidence"} & set(frame.columns)):
        print("No probability/confidence column found; main error score will default to 0.")


def _load_training_rows(args: argparse.Namespace) -> pd.DataFrame:
    paths = [p for p in [args.train_manifest, args.dev_manifest, args.test_manifest] if p]
    if not paths:
        raise SystemExit("A train manifest is required for retrieval and one-class verifiers.")
    frames = []
    for path in paths:
        frame = _load_csv(path)
        if path == args.train_manifest and "split" in frame.columns:
            frame = frame[frame["split"].astype(str) == args.train_split].copy()
        frames.append(frame)
    train = pd.concat(frames[:1], ignore_index=True)
    if train.empty:
        raise SystemExit("Training manifest is empty after split filtering.")
    return train


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Phase-1.5 verifier columns and final high-precision decisions.")
    parser.add_argument("--input", type=Path, default=ROOT / "reports/phase1_acoustic_fusion_macro/best_model_predictions.csv")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/processed/speechocean/phones_aligned.csv")
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase15/aggregator.yaml")
    parser.add_argument("--attributes", type=Path, default=ROOT / "configs/phase15/phone_attributes.json")
    parser.add_argument("--prototype-bank", type=Path, help="Optional prebuilt prototype bank from phase15_build_prototypes.py.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase15_verification/test_verified_predictions.csv")
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--disable-oneclass", action="store_true")
    args = parser.parse_args()

    predictions = _load_csv(args.input)
    _validate_input(predictions)
    cfg = load_config(args.config)
    mapping = load_phone_attributes(args.attributes)
    out = apply_attribute_verifier(predictions, mapping, cfg)

    train = None
    if not args.disable_retrieval or not args.disable_oneclass:
        train = _load_training_rows(args)

    if not args.disable_retrieval:
        bank = load_prototype_bank(args.prototype_bank) if args.prototype_bank else build_prototype_bank(train, mapping)
        out = apply_retrieval_verifier(out, bank, mapping, cfg)
    else:
        out["retrieval_verifier_decision"] = "not_run"
        out["proto_same_phone_sim_top1"] = 0.0
        out["proto_same_phone_sim_topk_mean"] = 0.0
        out["proto_confusion_phone_sim_top1"] = 0.0
        out["proto_margin"] = 0.0

    if not args.disable_oneclass:
        oneclass = train_oneclass_bank(train, mapping, cfg)
        out = apply_oneclass_verifier(out, oneclass, mapping, cfg)
    else:
        out["oneclass_verifier_decision"] = "not_run"
        out["oneclass_anomaly_score"] = 0.0

    out = aggregate_decisions(out, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved verified predictions: {args.output}")
    print(out["final_decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
