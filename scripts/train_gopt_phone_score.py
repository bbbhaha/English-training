#!/usr/bin/env python
"""Train a lightweight GOPT-style phone-score regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import joblib
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


class GOPTFeatures(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray, include_score_one: bool = True):
        self.arrays = arrays
        self.indices = indices
        self.include_score_one = include_score_one

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = self.indices[item]
        scores = self.arrays["scores"][idx].copy()
        mask = self.arrays["mask"][idx].copy()
        if not self.include_score_one:
            mask &= scores != 1.0
        return {
            "numeric": torch.tensor(self.arrays["numeric"][idx], dtype=torch.float32),
            "phone_ids": torch.tensor(self.arrays["phone_ids"][idx], dtype=torch.long),
            "group_ids": torch.tensor(self.arrays["group_ids"][idx], dtype=torch.long),
            "position_ids": torch.tensor(self.arrays["position_ids"][idx], dtype=torch.long),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }


class GOPTPhoneScoreModel(nn.Module):
    def __init__(self, *, numeric_dim: int, phone_vocab: int, group_vocab: int, max_position: int, config: dict):
        super().__init__()
        model_cfg = config["model"]
        self.phone_embedding = nn.Embedding(phone_vocab, model_cfg["phone_embedding_dim"], padding_idx=0)
        self.group_embedding = nn.Embedding(group_vocab, model_cfg["group_embedding_dim"], padding_idx=0)
        self.position_embedding = nn.Embedding(max_position, model_cfg["position_embedding_dim"])
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_dim, model_cfg["numeric_projection_dim"]),
            nn.ReLU(),
            nn.Dropout(model_cfg["dropout"]),
        )
        input_dim = (
            model_cfg["phone_embedding_dim"]
            + model_cfg["group_embedding_dim"]
            + model_cfg["position_embedding_dim"]
            + model_cfg["numeric_projection_dim"]
        )
        self.input_projection = nn.Linear(input_dim, model_cfg["hidden_dim"])
        layer = nn.TransformerEncoderLayer(
            d_model=model_cfg["hidden_dim"],
            nhead=model_cfg["attention_heads"],
            dim_feedforward=model_cfg["hidden_dim"] * 4,
            dropout=model_cfg["dropout"],
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=model_cfg["transformer_layers"])
        self.regressor = nn.Sequential(
            nn.LayerNorm(model_cfg["hidden_dim"]),
            nn.Linear(model_cfg["hidden_dim"], 1),
            nn.Sigmoid(),
        )

    def forward(self, numeric, phone_ids, group_ids, position_ids, mask):
        x = torch.cat(
            [
                self.numeric_projection(numeric),
                self.phone_embedding(phone_ids),
                self.group_embedding(group_ids),
                self.position_embedding(position_ids),
            ],
            dim=-1,
        )
        x = self.input_projection(x)
        encoded = self.encoder(x, src_key_padding_mask=~mask)
        return self.regressor(encoded).squeeze(-1) * 2.0


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def split_indices(metadata: pd.DataFrame, split: str) -> np.ndarray:
    return metadata.index[metadata["split"] == split].to_numpy(dtype=np.int64)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask & (target >= 0)
    return ((pred[valid] - target[valid]) ** 2).mean()


@torch.no_grad()
def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    gold = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["numeric"], batch["phone_ids"], batch["group_ids"], batch["position_ids"], batch["mask"])
        valid = batch["mask"] & (batch["scores"] >= 0)
        preds.append(pred[valid].detach().cpu().numpy())
        gold.append(batch["scores"][valid].detach().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gold)
    mse = float(np.mean((y_pred - y_true) ** 2))
    pcc = float(pearsonr(y_true, y_pred).statistic) if len(np.unique(y_true)) > 1 else 0.0
    return {"mse": mse, "pcc": pcc}


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> list[np.ndarray]:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["numeric"], batch["phone_ids"], batch["group_ids"], batch["position_ids"], batch["mask"])
        out.extend(pred.detach().cpu().numpy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gopt_speechocean762.yaml")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--include-score-one", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    feature_path = args.features or PROJECT_ROOT / config["data"]["feature_npz"]
    output_dir = args.output_dir or PROJECT_ROOT / config["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(feature_path.with_suffix(".vocab.json").read_text(encoding="utf-8"))
    arrays = load_arrays(feature_path)
    metadata = pd.read_csv(feature_path.with_suffix(".metadata.csv"), encoding="utf-8-sig")
    include_score_one = config["training"]["include_score_one"] if args.include_score_one is None else args.include_score_one

    train_ds = GOPTFeatures(arrays, split_indices(metadata, "train"), include_score_one=include_score_one)
    dev_ds = GOPTFeatures(arrays, split_indices(metadata, "dev"), include_score_one=True)
    test_ds = GOPTFeatures(arrays, split_indices(metadata, "test"), include_score_one=True)
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GOPTPhoneScoreModel(
        numeric_dim=arrays["numeric"].shape[-1],
        phone_vocab=len(vocab["phone_vocab"]),
        group_vocab=len(vocab["group_vocab"]),
        max_position=int(config["features"]["max_position"]),
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = args.epochs or int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    best_dev = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch["numeric"], batch["phone_ids"], batch["group_ids"], batch["position_ids"], batch["mask"])
            loss = masked_mse(pred, batch["scores"], batch["mask"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics = evaluate_regression(model, dev_loader, device)
        row = {"epoch": epoch, "train_mse": float(np.mean(losses)), **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        history.append(row)
        print(row)
        if dev_metrics["mse"] < best_dev:
            best_dev = dev_metrics["mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_regression(model, test_loader, device)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "vocab": vocab,
            "numeric_dim": arrays["numeric"].shape[-1],
        },
        output_dir / "gopt_phone_score.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    report = {"best_dev_mse": best_dev, "test": test_metrics, "include_score_one": include_score_one}
    (output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
