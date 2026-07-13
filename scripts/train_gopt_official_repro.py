#!/usr/bin/env python
"""Train GOPT-style phone score regressors for official-feature reproduction variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants-csv", type=Path, default=PROJECT_ROOT / "artifacts/gopt_official_repro/feature_variants.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gopt_speechocean762.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/gopt_official_repro/models")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--only-ready", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    variants = pd.read_csv(args.variants_csv, encoding="utf-8-sig")
    rows = []
    for row in variants.to_dict("records"):
        variant = row["variant"]
        status = row["status"]
        feature_npz = row.get("feature_npz", "")
        if args.only_ready and status != "ready":
            rows.append({**row, "train_status": "skipped_not_ready"})
            continue
        output_dir = args.output_root / variant
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train_gopt_phone_score.py"),
            "--config",
            str(args.config),
            "--features",
            str(feature_npz),
            "--output-dir",
            str(output_dir),
        ]
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        print("Running", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        rows.append({**row, "train_status": "trained", "model_dir": str(output_dir), "checkpoint": str(output_dir / "gopt_phone_score.pt")})

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_root / "train_status.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote train status to {args.output_root / 'train_status.csv'}")


if __name__ == "__main__":
    main()
