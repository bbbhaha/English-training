#!/usr/bin/env python
"""Evaluate HierTFR-minimal for regression and binary MDD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_hiertfr_minimal import HierTFRDataset, HierTFRMinimal, load_arrays, split_indices, word_targets_from_phone_targets


def load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location=device)
    model = HierTFRMinimal(
        numeric_dim=state["numeric_dim"],
        phone_vocab=len(state["vocab"]["phone_vocab"]),
        group_vocab=len(state["vocab"]["group_vocab"]),
        max_position=int(state.get("max_position", 256)),
        config=state["config"],
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


@torch.no_grad()
def collect_predictions(model, arrays, metadata, split, device):
    ds = HierTFRDataset(arrays, split_indices(metadata, split))
    loader = DataLoader(ds, batch_size=128)
    rows = []
    word_rows = []
    split_meta = metadata[metadata["split"] == split].reset_index(drop=True)
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["numeric"], batch["phone_ids"], batch["group_ids"], batch["position_ids"], batch["word_ids"], batch["mask"])
        phone_scores = out["phone_scores"].detach().cpu().numpy()
        mdd_prob = torch.sigmoid(out["mdd_logits"]).detach().cpu().numpy()
        mask = batch["mask"].detach().cpu().numpy()
        gold_scores = batch["scores"].detach().cpu().numpy()
        offset = batch_idx * loader.batch_size
        for i in range(phone_scores.shape[0]):
            meta = split_meta.iloc[offset + i].to_dict()
            for j in range(phone_scores.shape[1]):
                if mask[i, j]:
                    rows.append(
                        {
                            **meta,
                            "phone_pos": j,
                            "gold_score": float(gold_scores[i, j]),
                            "predicted_phone_score": float(phone_scores[i, j]),
                            "mdd_probability": float(mdd_prob[i, j]),
                        }
                    )
        wt, wm = word_targets_from_phone_targets(batch["word_ids"], batch["word_scores"], batch["mask"])
        word_scores = out["word_scores"].detach().cpu().numpy()
        word_mask = (wm & out["word_mask"]).detach().cpu().numpy()
        wt_np = wt.detach().cpu().numpy()
        for i in range(word_scores.shape[0]):
            meta = split_meta.iloc[offset + i].to_dict()
            for wid in range(word_scores.shape[1]):
                if word_mask[i, wid]:
                    word_rows.append({"utt_id": meta["utt_id"], "word_index": wid, "gold_word_score": float(wt_np[i, wid]), "predicted_word_score": float(word_scores[i, wid])})
    return pd.DataFrame(rows), pd.DataFrame(word_rows)


def pcc(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(pearsonr(y, p).statistic) if len(y) > 1 and len(np.unique(y)) > 1 else 0.0


def binary_metrics(label, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(label, pred, zero_division=0)),
        "recall": float(recall_score(label, pred, zero_division=0)),
        "f1": float(f1_score(label, pred, zero_division=0)),
        "auc": float(roc_auc_score(label, score)) if len(np.unique(label)) == 2 else 0.5,
        "auprc": float(average_precision_score(label, score)) if len(np.unique(label)) == 2 else float(np.mean(label)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }


def best_at_precision(label, score, min_precision):
    precision, recall, thresholds = precision_recall_curve(label, score)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        idx = int(np.lexsort((recall[:-1], precision[:-1]))[-1])
        return {"found": False, **binary_metrics(label, score, thresholds[idx])}
    idx = valid[np.argmax(recall[:-1][valid])]
    return {"found": True, **binary_metrics(label, score, thresholds[idx])}


def best_at_fpr(label, score, max_fpr):
    if len(np.unique(label)) < 2:
        best = binary_metrics(label, score, float(np.max(score) + 1e-6))
        return best
    fpr, tpr, thresholds = roc_curve(label, score)
    valid = np.where(fpr <= max_fpr)[0]
    if len(valid) == 0:
        return binary_metrics(label, score, float(np.max(score) + 1e-6))
    idx = valid[np.argmax(tpr[valid])]
    return binary_metrics(label, score, float(thresholds[idx]))


def summarize_per_phone(df):
    rows = []
    for phone, g in df.groupby("target_phone"):
        if g["label"].nunique() < 2:
            auc = 0.5
            auprc = float(g["label"].mean())
        else:
            auc = float(roc_auc_score(g["label"], g["eval_score"]))
            auprc = float(average_precision_score(g["label"], g["eval_score"]))
        rows.append(
            {
                "target_phone": phone,
                "phone_group": g["phone_group"].iloc[0],
                "support": len(g),
                "positive": int(g["label"].sum()),
                "precision": float(precision_score(g["label"], g["prediction"], zero_division=0)),
                "recall": float(recall_score(g["label"], g["prediction"], zero_division=0)),
                "f1": float(f1_score(g["label"], g["prediction"], zero_division=0)),
                "auc": auc,
                "auprc": auprc,
                "meets_040_050": bool(precision_score(g["label"], g["prediction"], zero_division=0) >= 0.40 and recall_score(g["label"], g["prediction"], zero_division=0) >= 0.50),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/hiertfr_minimal_speechocean.yaml")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/hiertfr_minimal_speechocean/hiertfr_minimal.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/hiertfr_minimal_speechocean")
    parser.add_argument("--score-source", choices=["mdd_head", "score_threshold"], default="mdd_head")
    parser.add_argument("--drop-score1-for-mdd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-score1-all", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--core-phone-set", nargs="*")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_path = args.features or PROJECT_ROOT / config["data"]["feature_npz"]
    arrays = load_arrays(feature_path)
    metadata = pd.read_csv(feature_path.with_suffix(".metadata.csv"), encoding="utf-8-sig")
    phones = pd.read_csv(feature_path.with_suffix(".phones.csv"), encoding="utf-8-sig")
    model, _state = load_model(args.checkpoint, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    phone_pred, word_pred = collect_predictions(model, arrays, metadata, args.split, device)
    split_phones = phones[phones["split"] == args.split].sort_values(["utterance_id", "phone_index"]).reset_index(drop=True)
    pred = pd.concat([split_phones.reset_index(drop=True), phone_pred[["predicted_phone_score", "mdd_probability"]].reset_index(drop=True)], axis=1)
    pred["gold_score"] = pred["source_score"].astype(float)
    if args.drop_score1_all:
        pred = pred[pred["gold_score"] != 1.0].copy()
    if args.core_phone_set:
        pred = pred[pred["target_phone"].isin(args.core_phone_set)].copy()
    regression = {
        "phone_mse": float(np.mean((pred["gold_score"] - pred["predicted_phone_score"]) ** 2)),
        "phone_pcc": pcc(pred["gold_score"], pred["predicted_phone_score"]),
        "word_pcc": pcc(word_pred["gold_word_score"], word_pred["predicted_word_score"]) if len(word_pred) else float("nan"),
    }
    binary = pred.copy()
    if args.drop_score1_for_mdd:
        binary = binary[binary["gold_score"] != 1.0].copy()
    binary["label"] = (binary["gold_score"] < 1.0).astype(int)
    binary["eval_score"] = binary["mdd_probability"] if args.score_source == "mdd_head" else -binary["predicted_phone_score"]
    selected = best_at_precision(binary["label"].to_numpy(), binary["eval_score"].to_numpy(), float(config["evaluation"]["min_precision"]))
    binary["threshold"] = selected["threshold"]
    binary["prediction"] = (binary["eval_score"] >= selected["threshold"]).astype(int)
    fpr_results = {f"max_recall_at_fpr_le_{target:.2f}": best_at_fpr(binary["label"].to_numpy(), binary["eval_score"].to_numpy(), float(target)) for target in config["evaluation"]["fpr_targets"]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    summarize_per_phone(binary).to_csv(args.output_dir / "per_phone_pr_summary.csv", index=False, encoding="utf-8-sig")
    fp = binary[(binary["label"] == 0) & (binary["prediction"] == 1)].copy()
    fp.groupby(["target_phone", "phone_group"]).agg(fp_count=("prediction", "size"), avg_score_correct=("eval_score", "mean")).reset_index().sort_values("fp_count", ascending=False).to_csv(
        args.output_dir / "false_positive_by_phone.csv", index=False, encoding="utf-8-sig"
    )
    cols = ["utterance_id", "speaker_id", "word", "target_phone", "start_ms", "end_ms", "gold_score", "eval_score", "prediction", "threshold", "audio_path"]
    fp[[c for c in cols if c in fp.columns]].head(500).to_csv(args.output_dir / "false_positive_examples.csv", index=False, encoding="utf-8-sig")
    report = {
        "split": args.split,
        "score_source": args.score_source,
        "core_phone_set": args.core_phone_set,
        **regression,
        "binary_at_precision_ge_0_40": selected,
        **fpr_results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
