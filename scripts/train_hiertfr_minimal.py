#!/usr/bin/env python
"""Train HierTFR-minimal with phone score regression, word score regression, and MDD auxiliary head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def split_indices(metadata: pd.DataFrame, split: str) -> np.ndarray:
    return metadata.index[metadata["split"] == split].to_numpy(dtype=np.int64)


class HierTFRDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = self.indices[item]
        return {
            "numeric": torch.tensor(self.arrays["numeric"][idx], dtype=torch.float32),
            "phone_ids": torch.tensor(self.arrays["phone_ids"][idx], dtype=torch.long),
            "group_ids": torch.tensor(self.arrays["group_ids"][idx], dtype=torch.long),
            "position_ids": torch.tensor(self.arrays["position_ids"][idx], dtype=torch.long),
            "word_ids": torch.tensor(self.arrays.get("word_ids", np.full_like(self.arrays["phone_ids"], -1))[idx], dtype=torch.long),
            "scores": torch.tensor(self.arrays["scores"][idx], dtype=torch.float32),
            "word_scores": torch.tensor(self.arrays.get("word_scores", np.full_like(self.arrays["scores"], -1))[idx], dtype=torch.float32),
            "binary_labels": torch.tensor(self.arrays["binary_labels"][idx], dtype=torch.float32),
            "mask": torch.tensor(self.arrays["mask"][idx], dtype=torch.bool),
        }


class HierTFRMinimal(nn.Module):
    def __init__(self, *, numeric_dim: int, phone_vocab: int, group_vocab: int, max_position: int, config: dict):
        super().__init__()
        cfg = config["model"]
        hidden = int(cfg["hidden_dim"])
        self.word_pooling = cfg.get("word_pooling", "mean")
        self.phone_embedding = nn.Embedding(phone_vocab, cfg["phone_embedding_dim"], padding_idx=0)
        self.group_embedding = nn.Embedding(group_vocab, cfg["group_embedding_dim"], padding_idx=0)
        self.position_embedding = nn.Embedding(max_position, cfg["position_embedding_dim"])
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_dim, cfg["numeric_projection_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )
        input_dim = cfg["phone_embedding_dim"] + cfg["group_embedding_dim"] + cfg["position_embedding_dim"] + cfg["numeric_projection_dim"]
        self.input_projection = nn.Linear(input_dim, hidden)
        phone_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cfg["attention_heads"],
            dim_feedforward=hidden * 4,
            dropout=cfg["dropout"],
            batch_first=True,
            activation="gelu",
        )
        self.phone_encoder = nn.TransformerEncoder(phone_layer, num_layers=cfg["phone_transformer_layers"])
        self.word_attention = nn.Linear(hidden, 1)
        word_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cfg["attention_heads"],
            dim_feedforward=hidden * 4,
            dropout=cfg["dropout"],
            batch_first=True,
            activation="gelu",
        )
        self.word_encoder = nn.TransformerEncoder(word_layer, num_layers=cfg["word_transformer_layers"])
        self.phone_score_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1), nn.Sigmoid())
        self.mdd_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.word_score_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1), nn.Sigmoid())

    def encode_phone(self, numeric, phone_ids, group_ids, position_ids, mask):
        x = torch.cat(
            [
                self.numeric_projection(numeric),
                self.phone_embedding(phone_ids),
                self.group_embedding(group_ids),
                self.position_embedding(position_ids.clamp_min(0)),
            ],
            dim=-1,
        )
        x = self.input_projection(x)
        return self.phone_encoder(x, src_key_padding_mask=~mask)

    def pool_to_words(self, phone_repr: torch.Tensor, word_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _phones, hidden = phone_repr.shape
        max_words = int(torch.clamp(word_ids[mask].max(), min=0).item()) + 1 if mask.any() else 1
        word_repr = phone_repr.new_zeros((batch, max_words, hidden))
        word_mask = torch.zeros((batch, max_words), dtype=torch.bool, device=phone_repr.device)
        for b in range(batch):
            valid_words = sorted(set(word_ids[b][mask[b] & (word_ids[b] >= 0)].detach().cpu().tolist()))
            for wid in valid_words:
                phone_mask = mask[b] & (word_ids[b] == wid)
                if self.word_pooling == "attention":
                    scores = self.word_attention(phone_repr[b, phone_mask]).squeeze(-1)
                    weights = torch.softmax(scores, dim=0)
                    word_repr[b, wid] = torch.sum(phone_repr[b, phone_mask] * weights.unsqueeze(-1), dim=0)
                else:
                    word_repr[b, wid] = phone_repr[b, phone_mask].mean(dim=0)
                word_mask[b, wid] = True
        word_ids_out = torch.arange(max_words, device=phone_repr.device).unsqueeze(0).expand(batch, -1)
        return word_repr, word_mask, word_ids_out

    def forward(self, numeric, phone_ids, group_ids, position_ids, word_ids, mask):
        phone_repr = self.encode_phone(numeric, phone_ids, group_ids, position_ids, mask)
        phone_scores = self.phone_score_head(phone_repr).squeeze(-1) * 2.0
        mdd_logits = self.mdd_head(phone_repr).squeeze(-1)
        word_repr, word_mask, _ = self.pool_to_words(phone_repr, word_ids, mask)
        word_context = self.word_encoder(word_repr, src_key_padding_mask=~word_mask)
        word_scores = self.word_score_head(word_context).squeeze(-1) * 10.0
        return {"phone_scores": phone_scores, "mdd_logits": mdd_logits, "word_scores": word_scores, "word_mask": word_mask}


def make_phone_filter(vocab: dict, names: list[str] | None, device: torch.device):
    if not names:
        return None
    allowed = [vocab["phone_vocab"][name] for name in names if name in vocab["phone_vocab"]]
    return torch.tensor(allowed, dtype=torch.long, device=device)


def valid_phone_mask(batch, phone_filter, *, drop_score1_all: bool, min_duration: float, duration_idx: int | None):
    valid = batch["mask"] & (batch["scores"] >= 0)
    if drop_score1_all:
        valid &= batch["scores"] != 1.0
    if min_duration and duration_idx is not None:
        valid &= batch["numeric"][:, :, duration_idx] >= min_duration
    if phone_filter is not None and len(phone_filter) > 0:
        valid &= torch.isin(batch["phone_ids"], phone_filter)
    return valid


def word_targets_from_phone_targets(word_ids, word_scores, phone_mask):
    batch = word_ids.shape[0]
    max_words = int(torch.clamp(word_ids[phone_mask].max(), min=0).item()) + 1 if phone_mask.any() else 1
    targets = word_scores.new_full((batch, max_words), -1.0)
    mask = torch.zeros((batch, max_words), dtype=torch.bool, device=word_scores.device)
    for b in range(batch):
        valid_words = sorted(set(word_ids[b][phone_mask[b] & (word_ids[b] >= 0)].detach().cpu().tolist()))
        for wid in valid_words:
            vals = word_scores[b][phone_mask[b] & (word_ids[b] == wid)]
            vals = vals[vals >= 0]
            if len(vals):
                targets[b, wid] = vals.float().mean()
                mask[b, wid] = True
    return targets, mask


def masked_mse(pred, target, mask):
    valid = mask & (target >= 0)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return ((pred[valid] - target[valid]) ** 2).mean()


def masked_bce(logits, target, mask):
    valid = mask & (target >= 0)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[valid], target[valid])


@torch.no_grad()
def evaluate(model, loader, device, phone_filter, args, duration_idx):
    model.eval()
    phone_gold, phone_pred = [], []
    word_gold, word_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["numeric"], batch["phone_ids"], batch["group_ids"], batch["position_ids"], batch["word_ids"], batch["mask"])
        phone_valid = valid_phone_mask(batch, phone_filter, drop_score1_all=args.drop_score1_all, min_duration=args.min_duration, duration_idx=duration_idx)
        phone_gold.append(batch["scores"][phone_valid].detach().cpu().numpy())
        phone_pred.append(out["phone_scores"][phone_valid].detach().cpu().numpy())
        wt, wm = word_targets_from_phone_targets(batch["word_ids"], batch["word_scores"], phone_valid)
        wvalid = wm & out["word_mask"]
        word_gold.append(wt[wvalid].detach().cpu().numpy())
        word_pred.append(out["word_scores"][wvalid].detach().cpu().numpy())
    pg = np.concatenate([x for x in phone_gold if len(x)]) if any(len(x) for x in phone_gold) else np.array([])
    pp = np.concatenate([x for x in phone_pred if len(x)]) if any(len(x) for x in phone_pred) else np.array([])
    wg = np.concatenate([x for x in word_gold if len(x)]) if any(len(x) for x in word_gold) else np.array([])
    wp = np.concatenate([x for x in word_pred if len(x)]) if any(len(x) for x in word_pred) else np.array([])
    return {
        "phone_mse": float(np.mean((pg - pp) ** 2)) if len(pg) else float("nan"),
        "phone_pcc": float(pearsonr(pg, pp).statistic) if len(pg) > 1 and len(np.unique(pg)) > 1 else 0.0,
        "word_pcc": float(pearsonr(wg, wp).statistic) if len(wg) > 1 and len(np.unique(wg)) > 1 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/hiertfr_minimal_speechocean.yaml")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--alpha-mdd", type=float)
    parser.add_argument("--beta-word", type=float)
    parser.add_argument("--drop-score1-for-mdd", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--drop-score1-all", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--core-phone-set", nargs="*")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    feature_path = args.features or PROJECT_ROOT / config["data"]["feature_npz"]
    output_dir = args.output_dir or PROJECT_ROOT / config["data"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(feature_path)
    metadata = pd.read_csv(feature_path.with_suffix(".metadata.csv"), encoding="utf-8-sig")
    vocab = json.loads(feature_path.with_suffix(".vocab.json").read_text(encoding="utf-8"))
    max_position = int(vocab.get("config", {}).get("features", {}).get("max_position", 256))
    numeric_columns = list(vocab.get("numeric_columns", []))
    duration_idx = numeric_columns.index("duration") if "duration" in numeric_columns else None
    args.alpha_mdd = config["training"]["alpha_mdd"] if args.alpha_mdd is None else args.alpha_mdd
    args.beta_word = config["training"]["beta_word"] if args.beta_word is None else args.beta_word
    args.drop_score1_for_mdd = config["training"]["drop_score1_for_mdd"] if args.drop_score1_for_mdd is None else args.drop_score1_for_mdd
    args.drop_score1_all = config["training"]["drop_score1_all"] if args.drop_score1_all is None else args.drop_score1_all
    args.min_duration = config["training"]["min_duration"] if args.min_duration is None else args.min_duration

    train_loader = DataLoader(HierTFRDataset(arrays, split_indices(metadata, "train")), batch_size=config["training"]["batch_size"], shuffle=True)
    dev_loader = DataLoader(HierTFRDataset(arrays, split_indices(metadata, "dev")), batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(HierTFRDataset(arrays, split_indices(metadata, "test")), batch_size=config["training"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phone_filter = make_phone_filter(vocab, args.core_phone_set, device)
    model = HierTFRMinimal(
        numeric_dim=arrays["numeric"].shape[-1],
        phone_vocab=len(vocab["phone_vocab"]),
        group_vocab=len(vocab["group_vocab"]),
        max_position=max_position,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    best_dev = float("inf")
    best_state = None
    stale = 0
    history = []
    epochs = args.epochs or config["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["numeric"], batch["phone_ids"], batch["group_ids"], batch["position_ids"], batch["word_ids"], batch["mask"])
            phone_valid = valid_phone_mask(batch, phone_filter, drop_score1_all=args.drop_score1_all, min_duration=args.min_duration, duration_idx=duration_idx)
            mdd_valid = phone_valid & (batch["scores"] != 1.0 if args.drop_score1_for_mdd else torch.ones_like(phone_valid, dtype=torch.bool))
            wt, wm = word_targets_from_phone_targets(batch["word_ids"], batch["word_scores"], phone_valid)
            loss_phone = masked_mse(out["phone_scores"], batch["scores"], phone_valid)
            loss_mdd = masked_bce(out["mdd_logits"], batch["binary_labels"], mdd_valid)
            loss_word = masked_mse(out["word_scores"], wt, wm & out["word_mask"])
            loss = loss_phone + float(args.alpha_mdd) * loss_mdd + float(args.beta_word) * loss_word
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev = evaluate(model, dev_loader, device, phone_filter, args, duration_idx)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"dev_{k}": v for k, v in dev.items()}}
        history.append(row)
        print(row)
        if dev["phone_mse"] < best_dev:
            best_dev = dev["phone_mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(config["training"]["early_stopping_patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, phone_filter, args, duration_idx)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "vocab": vocab,
            "numeric_dim": arrays["numeric"].shape[-1],
            "max_position": max_position,
            "core_phone_set": args.core_phone_set,
            "alpha_mdd": args.alpha_mdd,
            "beta_word": args.beta_word,
            "drop_score1_for_mdd": args.drop_score1_for_mdd,
            "drop_score1_all": args.drop_score1_all,
            "min_duration": args.min_duration,
        },
        output_dir / "hiertfr_minimal.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    report = {"best_dev_phone_mse": best_dev, "test": test, "core_phone_set": args.core_phone_set}
    (output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
