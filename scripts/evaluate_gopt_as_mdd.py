#!/usr/bin/env python
"""Evaluate GOPT phone-score regression as binary MDD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_gopt_phone_score import GOPTFeatures, GOPTPhoneScoreModel, load_arrays, split_indices


def load_model(checkpoint: Path, device: torch.device) -> tuple[GOPTPhoneScoreModel, dict]:
    state = torch.load(checkpoint, map_location=device)
    config = state["config"]
    model = GOPTPhoneScoreModel(
        numeric_dim=state["numeric_dim"],
        phone_vocab=len(state["vocab"]["phone_vocab"]),
        group_vocab=len(state["vocab"]["group_vocab"]),
        max_position=int(config["features"]["max_position"]),
        config=config,
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


@torch.no_grad()
def collect_predictions(model, arrays, metadata, split, device):
    ds = GOPTFeatures(arrays, split_indices(metadata, split), include_score_one=True)
    loader = DataLoader(ds, batch_size=64)
    split_meta = metadata[metadata["split"] == split].reset_index(drop=True)
    rows = []
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["numeric"], batch["phone_ids"], batch["group_ids"], batch["position_ids"], batch["mask"])
        pred_np = pred.detach().cpu().numpy()
        scores_np = batch["scores"].detach().cpu().numpy()
        mask_np = batch["mask"].detach().cpu().numpy()
        offset = batch_idx * loader.batch_size
        for i in range(pred_np.shape[0]):
            meta = split_meta.iloc[offset + i].to_dict()
            for j in range(pred_np.shape[1]):
                if not mask_np[i, j]:
                    continue
                rows.append(
                    {
                        **meta,
                        "phone_pos": j,
                        "gold_score": float(scores_np[i, j]),
                        "predicted_score": float(pred_np[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["gold_score"].to_numpy()
    p = frame["predicted_score"].to_numpy()
    mse = float(np.mean((y - p) ** 2))
    pcc = float(pearsonr(y, p).statistic) if len(np.unique(y)) > 1 else 0.0
    return {"phone_score_mse": mse, "phone_score_pcc": pcc}


def threshold_metrics(gold_binary: np.ndarray, error_score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (error_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(gold_binary, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(gold_binary, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(gold_binary, pred)),
        "precision": float(precision_score(gold_binary, pred, zero_division=0)),
        "recall": float(recall_score(gold_binary, pred, zero_division=0)),
        "f1": float(f1_score(gold_binary, pred, zero_division=0)),
        "auc": float(roc_auc_score(gold_binary, error_score)) if len(np.unique(gold_binary)) == 2 else 0.5,
        "auprc": float(average_precision_score(gold_binary, error_score)) if len(np.unique(gold_binary)) == 2 else float(np.mean(gold_binary)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def best_recall_at_precision(gold_binary: np.ndarray, error_score: np.ndarray, min_precision: float) -> dict[str, object]:
    precision, recall, thresholds = precision_recall_curve(gold_binary, error_score)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return {"found": False, "precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": None}
    idx = valid[np.argmax(recall[:-1][valid])]
    return {"found": True, **threshold_metrics(gold_binary, error_score, float(thresholds[idx]))}


def evaluate_binary(frame: pd.DataFrame, min_precision: float, drop_score_one: bool) -> tuple[dict, pd.DataFrame]:
    df = frame.copy()
    if drop_score_one:
        df = df[df["gold_score"] != 1.0].copy()
    df["label"] = (df["gold_score"] < 1.0).astype(int)
    # predicted_score < phone-score-threshold means error; use -score as error score.
    error_score = -df["predicted_score"].to_numpy()
    label = df["label"].to_numpy()
    selected = best_recall_at_precision(label, error_score, min_precision)
    if not selected["found"]:
        precision, recall, thresholds = precision_recall_curve(label, error_score)
        idx = int(np.lexsort((recall[:-1], precision[:-1]))[-1])
        selected = {"found": False, **threshold_metrics(label, error_score, float(thresholds[idx]))}
    df["error_score"] = error_score
    df["threshold"] = selected["threshold"]
    df["prediction"] = (error_score >= float(selected["threshold"])).astype(int)
    return selected, df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gopt_speechocean762.yaml")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/gopt_speechocean762/gopt_phone_score.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/gopt_speechocean762")
    parser.add_argument("--min-precision", type=float)
    parser.add_argument("--binary-drop-score-one", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--core-phones", nargs="*")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_path = args.features or PROJECT_ROOT / config["data"]["feature_npz"]
    min_precision = args.min_precision if args.min_precision is not None else float(config["evaluation"]["min_precision"])
    drop_score_one = config["evaluation"]["binary_drop_score_one"] if args.binary_drop_score_one is None else args.binary_drop_score_one
    arrays = load_arrays(feature_path)
    metadata = pd.read_csv(feature_path.with_suffix(".metadata.csv"), encoding="utf-8-sig")
    phones = pd.read_csv(feature_path.with_suffix(".phones.csv"), encoding="utf-8-sig")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _state = load_model(args.checkpoint, device)
    predictions = collect_predictions(model, arrays, metadata, args.split, device)
    # Attach phone-level metadata in sequence order.
    split_phones = phones[phones["split"] == args.split].sort_values(["utterance_id", "phone_index"]).reset_index(drop=True)
    predictions = pd.concat([split_phones.reset_index(drop=True), predictions[["predicted_score"]].reset_index(drop=True)], axis=1)
    predictions["gold_score"] = predictions["source_score"].astype(float)
    if args.core_phones:
        predictions = predictions[predictions["target_phone"].isin(args.core_phones)].copy()

    reg = regression_metrics(predictions)
    binary, pred_rows = evaluate_binary(predictions, min_precision, drop_score_one)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_rows.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "split": args.split,
        "core_phones": args.core_phones or None,
        "binary_drop_score_one": drop_score_one,
        **reg,
        "binary_at_precision_constraint": binary,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = [
        "# GOPT as MDD Evaluation",
        "",
        f"- split: {args.split}",
        f"- binary_drop_score_one: {drop_score_one}",
        f"- core_phones: {args.core_phones or 'all'}",
        f"- phone_score_mse: {reg['phone_score_mse']:.6f}",
        f"- phone_score_pcc: {reg['phone_score_pcc']:.6f}",
        "",
        "## Precision-constrained binary MDD",
    ]
    for k, v in binary.items():
        summary.append(f"- {k}: {v}")
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
