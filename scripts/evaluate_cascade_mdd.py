#!/usr/bin/env python
"""Evaluate final cascade MDD over all phone samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
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
    roc_curve,
)
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_candidate_verifier import CandidateDataset, CandidateVerifier, load_candidates


def load_model(checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CandidateVerifier(state["audio_dim"], len(state["phone_vocab"]), len(state["group_vocab"]), state["config"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


@torch.no_grad()
def score_candidates(model, state, candidates: pd.DataFrame, features: Path, device) -> pd.DataFrame:
    cand_only = candidates[candidates["candidate_label"].astype(int) == 1].copy()
    full = load_candidates(candidates.attrs["path"], features)
    ds = CandidateDataset(
        full,
        state["phone_vocab"],
        state["group_vocab"],
        audio_mean=state.get("audio_mean"),
        audio_std=state.get("audio_std"),
        numeric_mean=state.get("numeric_mean"),
        numeric_std=state.get("numeric_std"),
    )
    loader = DataLoader(ds, batch_size=512)
    scores = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["audio"], batch["numeric"], batch["phone_ids"], batch["group_ids"])
        scores.append(torch.sigmoid(logits).cpu().numpy())
    full["verifier_score"] = np.concatenate(scores)
    key = ["utt_id", "speaker_id", "target_phone", "phone_index"]
    for frame in [cand_only, full]:
        frame["utt_id"] = frame["utt_id"].astype(str)
        frame["speaker_id"] = frame["speaker_id"].astype(str)
        frame["target_phone"] = frame["target_phone"].astype(str)
        frame["phone_index"] = pd.to_numeric(frame["phone_index"], errors="coerce").fillna(-1).astype(int)
    return cand_only.merge(full[key + ["verifier_score"]], on=key, how="left")


def metrics(label, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(label, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(label, pred)),
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
        return {"found": False, **metrics(label, score, float(thresholds[idx]))}
    idx = valid[np.argmax(recall[:-1][valid])]
    return {"found": True, **metrics(label, score, float(thresholds[idx]))}


def best_at_fpr(label, score, max_fpr):
    if len(np.unique(label)) < 2:
        return metrics(label, score, float(np.max(score) + 1e-6))
    fpr, tpr, thresholds = roc_curve(label, score)
    valid = np.where(fpr <= max_fpr)[0]
    if len(valid) == 0:
        return metrics(label, score, float(np.max(score) + 1e-6))
    idx = valid[np.argmax(tpr[valid])]
    return metrics(label, score, float(thresholds[idx]))


def per_phone_summary(df):
    rows = []
    for phone, g in df.groupby("target_phone"):
        rows.append(
            {
                "target_phone": phone,
                "phone_group": g["phone_group"].iloc[0],
                "support": len(g),
                "positive": int(g["gold_label"].sum()),
                "precision": float(precision_score(g["gold_label"], g["final_prediction"], zero_division=0)),
                "recall": float(recall_score(g["gold_label"], g["final_prediction"], zero_division=0)),
                "f1": float(f1_score(g["gold_label"], g["final_prediction"], zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/candidate_verifier.yaml")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "outputs/mdd_candidates_test.csv")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/candidate_verifier/candidate_verifier.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/cascade_mdd")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    features = args.features or PROJECT_ROOT / cfg["data"]["features_npz"]
    all_phones = pd.read_csv(args.candidates, encoding="utf-8-sig", keep_default_na=False)
    all_phones["utt_id"] = all_phones["utt_id"].astype(str)
    all_phones["speaker_id"] = all_phones["speaker_id"].astype(str)
    all_phones["target_phone"] = all_phones["target_phone"].astype(str)
    all_phones["phone_index"] = pd.to_numeric(all_phones["phone_index"], errors="coerce").fillna(-1).astype(int)
    all_phones.attrs["path"] = args.candidates
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device)
    scored_candidates = score_candidates(model, state, all_phones, features, device)
    key = ["utt_id", "speaker_id", "target_phone", "phone_index"]
    df = all_phones.merge(scored_candidates[key + ["verifier_score"]], on=key, how="left")
    df["verifier_score"] = pd.to_numeric(df["verifier_score"], errors="coerce").fillna(0.0)
    df["cascade_score"] = df["candidate_label"].astype(int) * df["verifier_score"]
    label = df["gold_label"].astype(int).to_numpy()
    score = df["cascade_score"].to_numpy(float)
    selected = best_at_precision(label, score, float(cfg["evaluation"]["min_precision"]))
    df["threshold"] = selected["threshold"]
    df["final_prediction"] = (score >= selected["threshold"]).astype(int)
    candidate_pred = df["candidate_label"].astype(int).to_numpy()
    candidate_recall = float(recall_score(label, candidate_pred, zero_division=0))
    candidate_precision = float(precision_score(label, candidate_pred, zero_division=0))
    true_error_candidates = df[(df["candidate_label"].astype(int) == 1) & (df["gold_label"].astype(int) == 1)]
    verifier_recall = float((true_error_candidates["verifier_score"] >= selected["threshold"]).mean()) if len(true_error_candidates) else 0.0
    verifier_precision_pool = float(precision_score(scored_candidates["gold_label"].astype(int), (scored_candidates["verifier_score"] >= selected["threshold"]).astype(int), zero_division=0))
    fpr_results = {f"max_recall_at_fpr_le_{x:.2f}": best_at_fpr(label, score, float(x)) for x in cfg["evaluation"]["fpr_targets"]}
    report = {
        "candidate_precision": candidate_precision,
        "candidate_recall": candidate_recall,
        "verifier_precision_inside_candidate_pool": verifier_precision_pool,
        "verifier_recall_on_true_candidate_errors": verifier_recall,
        "final_recall_product": candidate_recall * verifier_recall,
        "binary_at_precision_ge_0_40": selected,
        **fpr_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "cascade_predictions.csv", index=False, encoding="utf-8-sig")
    per_phone_summary(df).to_csv(PROJECT_ROOT / "outputs/cascade_per_phone_pr_summary.csv", index=False, encoding="utf-8-sig")
    fp = df[(df["gold_label"].astype(int) == 0) & (df["final_prediction"] == 1)].copy()
    fp.groupby(["target_phone", "phone_group"]).size().reset_index(name="fp_count").sort_values("fp_count", ascending=False).to_csv(
        PROJECT_ROOT / "outputs/cascade_false_positive_by_phone.csv", index=False, encoding="utf-8-sig"
    )
    cols = ["utt_id", "speaker_id", "word", "target_phone", "phone_group", "start", "end", "duration", "gold_label", "verifier_score", "final_prediction", "threshold", "wav_path"]
    fp[[c for c in cols if c in fp.columns]].head(500).to_csv(PROJECT_ROOT / "outputs/cascade_false_positive_examples.csv", index=False, encoding="utf-8-sig")
    removed = df[(df["candidate_label"].astype(int) == 1) & (df["gold_label"].astype(int) == 0) & (df["final_prediction"] == 0)].copy()
    removed[[c for c in cols if c in removed.columns]].head(2000).to_csv(PROJECT_ROOT / "outputs/cascade_removed_false_positives.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
