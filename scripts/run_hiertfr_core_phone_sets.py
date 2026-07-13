#!/usr/bin/env python
"""Run separate HierTFR-minimal experiments for configured core phone sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/hiertfr_minimal_speechocean.yaml")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs/hiertfr_core_phone_set_results.csv")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_path = args.features or PROJECT_ROOT / config["data"]["feature_npz"]
    rows = []
    for name, phones in config["evaluation"]["core_phone_sets"].items():
        model_dir = PROJECT_ROOT / "artifacts/hiertfr_core_phone_sets" / name
        report_dir = PROJECT_ROOT / "reports/hiertfr_core_phone_sets" / name
        train_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train_hiertfr_minimal.py"),
            "--config",
            str(args.config),
            "--features",
            str(feature_path),
            "--output-dir",
            str(model_dir),
            "--core-phone-set",
            *phones,
        ]
        if args.epochs:
            train_cmd.extend(["--epochs", str(args.epochs)])
        subprocess.run(train_cmd, check=True, cwd=PROJECT_ROOT)
        eval_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate_hiertfr_as_mdd.py"),
            "--config",
            str(args.config),
            "--features",
            str(feature_path),
            "--checkpoint",
            str(model_dir / "hiertfr_minimal.pt"),
            "--output-dir",
            str(report_dir),
            "--core-phone-set",
            *phones,
        ]
        subprocess.run(eval_cmd, check=True, cwd=PROJECT_ROOT)
        metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
        binary = metrics["binary_at_precision_ge_0_40"]
        rows.append(
            {
                "core_phone_set": name,
                "phones": " ".join(phones),
                "phone_mse": metrics["phone_mse"],
                "phone_pcc": metrics["phone_pcc"],
                "word_pcc": metrics["word_pcc"],
                "precision": binary["precision"],
                "recall": binary["recall"],
                "f1": binary["f1"],
                "auc": binary["auc"],
                "auprc": binary["auprc"],
                "fpr": binary["fpr"],
                "fp": binary["fp"],
                "fn": binary["fn"],
                "precision_constraint_found": binary["found"],
                "threshold": binary["threshold"],
                "meets_040_050": bool(binary["precision"] >= 0.40 and binary["recall"] >= 0.50),
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
