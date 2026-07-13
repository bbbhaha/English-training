#!/usr/bin/env python
"""Build a correct-phone prototype bank for Phase-1.5 retrieval verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase15_verification.phone_attributes import load_phone_attributes
from phase15_verification.prototype_retrieval import build_prototype_bank, save_prototype_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build correct-sample phone prototypes for Phase-1.5.")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/processed/speechocean/phones_aligned.csv")
    parser.add_argument("--attributes", type=Path, default=ROOT / "configs/phase15/phone_attributes.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase15_verification/prototype_bank.joblib")
    parser.add_argument("--split", default="train", help="Training split name to use when the manifest has a split column.")
    args = parser.parse_args()

    train = pd.read_csv(args.train_manifest, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if "split" in train.columns:
        train = train[train["split"].astype(str) == args.split].copy()
    mapping = load_phone_attributes(args.attributes)
    bank = build_prototype_bank(train, mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_prototype_bank(bank, args.output)
    print(f"Saved prototype bank: {args.output}")
    print(f"Prototype rows: {len(bank.vectors)}")


if __name__ == "__main__":
    main()
