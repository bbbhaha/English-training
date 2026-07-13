#!/usr/bin/env python
"""Train a false-positive verifier on high-recall candidate phones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_vocab(values: pd.Series) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for value in sorted(values.dropna().astype(str).unique()):
        vocab.setdefault(value, len(vocab))
    return vocab


def load_feature_table(features_npz: Path) -> pd.DataFrame:
    meta = pd.read_csv(features_npz.with_suffix(".metadata.csv"), encoding="utf-8-sig", keep_default_na=False)
    emb = np.load(features_npz)["embeddings"]
    emb_df = pd.DataFrame(emb, columns=[f"w2v_{i}" for i in range(emb.shape[1])])
    return pd.concat([meta.reset_index(drop=True), emb_df], axis=1)


def load_candidates(candidate_csv: Path, features_npz: Path) -> pd.DataFrame:
    cand = pd.read_csv(candidate_csv, encoding="utf-8-sig", keep_default_na=False)
    cand = cand[cand["candidate_label"].astype(int) == 1].copy()
    feats = load_feature_table(features_npz)
    key = ["utt_id", "speaker_id", "target_phone", "phone_index"]
    for frame in [cand, feats]:
        frame["utt_id"] = frame["utt_id"].astype(str)
        frame["speaker_id"] = frame["speaker_id"].astype(str)
        frame["target_phone"] = frame["target_phone"].astype(str)
        frame["phone_index"] = pd.to_numeric(frame["phone_index"], errors="coerce").fillna(-1).astype(int)
    merged = cand.merge(feats[key + [c for c in feats.columns if c.startswith("w2v_")]], on=key, how="left")
    merged["gold_label"] = merged["gold_label"].astype(int)
    for col in ["duration", "normalized_duration_by_phone", "gop_score", "candidate_score"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged.reset_index(drop=True)


class CandidateDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        phone_vocab: dict[str, int],
        group_vocab: dict[str, int],
        audio_mean: np.ndarray | None = None,
        audio_std: np.ndarray | None = None,
        numeric_mean: np.ndarray | None = None,
        numeric_std: np.ndarray | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.w2v_cols = [c for c in frame.columns if c.startswith("w2v_")]
        self.audio = frame[self.w2v_cols].to_numpy(np.float32)
        self.numeric = frame[["duration", "normalized_duration_by_phone", "gop_score", "candidate_score"]].to_numpy(np.float32)
        self.audio_mean = self.audio.mean(axis=0, keepdims=True) if audio_mean is None else audio_mean
        self.audio_std = self.audio.std(axis=0, keepdims=True) + 1e-6 if audio_std is None else audio_std
        self.numeric_mean = self.numeric.mean(axis=0, keepdims=True) if numeric_mean is None else numeric_mean
        self.numeric_std = self.numeric.std(axis=0, keepdims=True) + 1e-6 if numeric_std is None else numeric_std
        self.audio = (self.audio - self.audio_mean) / self.audio_std
        self.numeric = (self.numeric - self.numeric_mean) / self.numeric_std
        self.phone_ids = frame["target_phone"].astype(str).map(lambda x: phone_vocab.get(x, 1)).to_numpy(np.int64)
        self.group_ids = frame["phone_group"].astype(str).map(lambda x: group_vocab.get(x, 1)).to_numpy(np.int64)
        self.labels = frame["gold_label"].to_numpy(np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "audio": torch.tensor(self.audio[idx], dtype=torch.float32),
            "numeric": torch.tensor(self.numeric[idx], dtype=torch.float32),
            "phone_ids": torch.tensor(self.phone_ids[idx], dtype=torch.long),
            "group_ids": torch.tensor(self.group_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class CandidateVerifier(nn.Module):
    def __init__(self, audio_dim: int, phone_vocab_size: int, group_vocab_size: int, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.use_text_gate = bool(m.get("use_text_gate", True))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, m["audio_projection_dim"]), nn.ReLU(), nn.Dropout(m["dropout"]))
        self.phone_emb = nn.Embedding(phone_vocab_size, m["phone_embedding_dim"], padding_idx=0)
        self.group_emb = nn.Embedding(group_vocab_size, m["group_embedding_dim"], padding_idx=0)
        text_dim = m["phone_embedding_dim"] + m["group_embedding_dim"]
        self.gate = nn.Linear(text_dim, m["audio_projection_dim"])
        in_dim = m["audio_projection_dim"] * 2 + text_dim + 4
        self.head = nn.Sequential(
            nn.Linear(in_dim, m["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(m["hidden_dim"], 1),
        )

    def forward(self, audio, numeric, phone_ids, group_ids, return_repr: bool = False):
        audio_repr = self.audio_proj(audio)
        text_repr = torch.cat([self.phone_emb(phone_ids), self.group_emb(group_ids)], dim=-1)
        gate = torch.sigmoid(self.gate(text_repr)) if self.use_text_gate else torch.ones_like(audio_repr)
        gated = audio_repr * gate
        fused = torch.cat([gated, audio_repr, text_repr, numeric], dim=-1)
        logits = self.head(fused).squeeze(-1)
        return (logits, fused) if return_repr else logits


def supervised_contrastive_loss(repr_: torch.Tensor, labels: torch.Tensor, phone_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    if len(labels) < 2:
        return repr_.sum() * 0.0
    z = nn.functional.normalize(repr_, dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    self_mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    same_phone = phone_ids.unsqueeze(0).eq(phone_ids.unsqueeze(1))
    pos_mask = self_mask & same_label & same_phone
    if pos_mask.sum() == 0:
        return repr_.sum() * 0.0
    exp = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True).clamp_min(1e-8))
    return -(log_prob * pos_mask).sum() / pos_mask.sum()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["audio"], batch["numeric"], batch["phone_ids"], batch["group_ids"])
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    y = np.concatenate(labels).astype(int)
    s = np.concatenate(scores)
    pred = (s >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else 0.5,
        "auprc": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float(np.mean(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/candidate_verifier.yaml")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--train-candidates", type=Path, default=PROJECT_ROOT / "outputs/mdd_candidates_train.csv")
    parser.add_argument("--dev-candidates", type=Path, default=PROJECT_ROOT / "outputs/mdd_candidates_dev.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--use-text-gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-contrastive-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--contrastive-weight", type=float)
    parser.add_argument("--contrastive-temperature", type=float)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    features = args.features or PROJECT_ROOT / cfg["data"]["features_npz"]
    output_dir = args.output_dir or PROJECT_ROOT / cfg["data"]["output_dir"]
    train = load_candidates(args.train_candidates, features)
    dev = load_candidates(args.dev_candidates, features)
    phone_vocab = build_vocab(pd.concat([train["target_phone"], dev["target_phone"]]))
    group_vocab = build_vocab(pd.concat([train["phone_group"], dev["phone_group"]]))
    if args.use_text_gate is not None:
        cfg["model"]["use_text_gate"] = args.use_text_gate
    use_contrastive = cfg["training"]["use_contrastive_loss"] if args.use_contrastive_loss is None else args.use_contrastive_loss
    contrastive_weight = cfg["training"]["contrastive_weight"] if args.contrastive_weight is None else args.contrastive_weight
    contrastive_temperature = cfg["training"]["contrastive_temperature"] if args.contrastive_temperature is None else args.contrastive_temperature
    train_ds = CandidateDataset(train, phone_vocab, group_vocab)
    dev_ds = CandidateDataset(
        dev,
        phone_vocab,
        group_vocab,
        audio_mean=train_ds.audio_mean,
        audio_std=train_ds.audio_std,
        numeric_mean=train_ds.numeric_mean,
        numeric_std=train_ds.numeric_std,
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=int(cfg["training"]["batch_size"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateVerifier(len(train_ds.w2v_cols), len(phone_vocab), len(group_vocab), cfg).to(device)
    pos = max(float(train["gold_label"].sum()), 1.0)
    neg = max(float((train["gold_label"] == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best = -1.0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, repr_ = model(batch["audio"], batch["numeric"], batch["phone_ids"], batch["group_ids"], return_repr=True)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"], pos_weight=pos_weight)
            if use_contrastive:
                loss = loss + float(contrastive_weight) * supervised_contrastive_loss(repr_, batch["labels"], batch["phone_ids"], float(contrastive_temperature))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics = evaluate(model, dev_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        history.append(row)
        print(row)
        score = dev_metrics["auprc"]
        if score > best:
            best = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["training"]["early_stopping_patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "phone_vocab": phone_vocab,
            "group_vocab": group_vocab,
            "audio_dim": len(train_ds.w2v_cols),
            "audio_mean": train_ds.audio_mean,
            "audio_std": train_ds.audio_std,
            "numeric_mean": train_ds.numeric_mean,
            "numeric_std": train_ds.numeric_std,
            "use_contrastive_loss": use_contrastive,
        },
        output_dir / "candidate_verifier.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    report = {
        "train_candidates": int(len(train)),
        "dev_candidates": int(len(dev)),
        "train_positive_rate": float(train["gold_label"].mean()),
        "dev_positive_rate": float(dev["gold_label"].mean()),
        "best_dev_auprc": best,
    }
    (output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
