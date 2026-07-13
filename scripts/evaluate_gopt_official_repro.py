#!/usr/bin/env python
"""Evaluate GOPT official-feature reproduction variants and write one comparison CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def word_pcc_from_predictions(predictions_csv: Path) -> float:
    df = pd.read_csv(predictions_csv, encoding="utf-8-sig")
    if "word_accuracy" not in df.columns:
        return float("nan")
    grouped = (
        df.dropna(subset=["word_accuracy"])
        .groupby(["utterance_id", "word_index"], dropna=False)
        .agg(gold_word_score=("word_accuracy", "first"), pred_word_score=("predicted_score", "mean"))
        .reset_index()
    )
    if len(grouped) < 2 or grouped["gold_word_score"].nunique() < 2:
        return float("nan")
    return float(pearsonr(grouped["gold_word_score"].to_numpy(float), grouped["pred_word_score"].to_numpy(float)).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-status-csv", type=Path, default=PROJECT_ROOT / "artifacts/gopt_official_repro/models/train_status.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gopt_speechocean762.yaml")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs/gopt_official_repro_results.csv")
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "reports/gopt_official_repro")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    status = pd.read_csv(args.train_status_csv, encoding="utf-8-sig")
    rows = []
    for row in status.to_dict("records"):
        variant = row["variant"]
        if row.get("train_status") != "trained":
            rows.append({"variant": variant, "status": row.get("train_status", row.get("status", "missing"))})
            continue
        out_dir = args.report_root / variant
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate_gopt_as_mdd.py"),
            "--config",
            str(args.config),
            "--features",
            str(row["feature_npz"]),
            "--checkpoint",
            str(row["checkpoint"]),
            "--split",
            args.split,
            "--output-dir",
            str(out_dir),
            "--binary-drop-score-one",
        ]
        print("Running", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        binary = metrics["binary_at_precision_constraint"]
        rows.append(
            {
                "variant": variant,
                "status": "evaluated",
                "split": args.split,
                "phone_mse": metrics["phone_score_mse"],
                "phone_pcc": metrics["phone_score_pcc"],
                "word_pcc": word_pcc_from_predictions(out_dir / "predictions.csv"),
                "binary_precision": binary.get("precision"),
                "binary_recall": binary.get("recall"),
                "binary_f1": binary.get("f1"),
                "binary_auc": binary.get("auc"),
                "binary_auprc": binary.get("auprc"),
                "binary_fpr": binary.get("fpr"),
                "max_recall_at_precision_ge_0_40": binary.get("recall"),
                "precision_constraint_found": binary.get("found"),
                "threshold": binary.get("threshold"),
                "false_positives": binary.get("false_positives"),
                "false_negatives": binary.get("false_negatives"),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
