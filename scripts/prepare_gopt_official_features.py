#!/usr/bin/env python
"""Prepare GOPT official-feature comparison variants.

This script keeps one canonical sequence format:
    numeric, phone_ids, group_ids, position_ids, word_ids, scores, word_scores, mask

It can:
1) pass through the current Gaussian GOP features;
2) create a no-GOP ablation by zeroing GOP-related numeric columns;
3) import an official GOPT intermediate feature CSV/NPZ when supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_speechocean_gopt_features import main as prepare_current_main  # noqa: F401


GOP_COLUMNS = {"gop_score", "target_log_likelihood", "competitor_log_likelihood", "model_probability"}


def copy_sidecars(src: Path, dst: Path) -> None:
    for suffix in [".metadata.csv", ".phones.csv", ".vocab.json"]:
        sidecar = src.with_suffix(suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, dst.with_suffix(suffix))


def copy_variant(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copy_sidecars(src, dst)


def make_no_gop_variant(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(np.load(src))
    vocab = json.loads(src.with_suffix(".vocab.json").read_text(encoding="utf-8"))
    numeric_columns = list(vocab.get("numeric_columns", []))
    numeric = arrays["numeric"].copy()
    for idx, name in enumerate(numeric_columns):
        if name in GOP_COLUMNS or "gop" in name.lower() or "likelihood" in name.lower():
            numeric[:, :, idx] = 0.0
    arrays["numeric"] = numeric
    np.savez_compressed(dst, **arrays)
    copy_sidecars(src, dst)
    vocab["feature_variant"] = "no_gop_features"
    dst.with_suffix(".vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")


def import_official_npz(official: Path, template: Path, dst: Path) -> None:
    """Import official features when they already match utterance/padded layout.

    Expected key aliases:
    - numeric OR features OR gop_features
    Optional standard keys are copied from official when present; otherwise copied
      from the template built from local SpeechOcean metadata.
    """

    official_arrays = dict(np.load(official, allow_pickle=True))
    template_arrays = dict(np.load(template))
    if "numeric" in official_arrays:
        numeric = official_arrays["numeric"]
    elif "features" in official_arrays:
        numeric = official_arrays["features"]
    elif "gop_features" in official_arrays:
        numeric = official_arrays["gop_features"]
    else:
        raise ValueError("Official NPZ must contain one of: numeric, features, gop_features")
    arrays = template_arrays
    arrays["numeric"] = numeric.astype(np.float32)
    for key in ["phone_ids", "group_ids", "position_ids", "word_ids", "scores", "word_scores", "binary_labels", "mask"]:
        if key in official_arrays:
            arrays[key] = official_arrays[key]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **arrays)
    copy_sidecars(template, dst)
    vocab = json.loads(template.with_suffix(".vocab.json").read_text(encoding="utf-8"))
    vocab["feature_variant"] = "official_gop_features"
    vocab["official_source"] = str(official)
    vocab["numeric_columns"] = [f"official_feature_{i}" for i in range(arrays["numeric"].shape[-1])]
    dst.with_suffix(".vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, default=PROJECT_ROOT / "artifacts/gopt_speechocean762/features.npz")
    parser.add_argument("--official-feature-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/gopt_official_repro")
    args = parser.parse_args()

    base = args.base_features
    if not base.exists():
        raise FileNotFoundError(f"Base feature file not found: {base}. Run prepare_speechocean_gopt_features.py first.")

    current_dst = args.output_dir / "current_gaussian_gop_features.npz"
    no_gop_dst = args.output_dir / "no_gop_features.npz"
    copy_variant(base, current_dst)
    make_no_gop_variant(base, no_gop_dst)

    manifest = [
        {"variant": "current_gaussian_gop_features", "feature_npz": str(current_dst), "status": "ready"},
        {"variant": "no_gop_features", "feature_npz": str(no_gop_dst), "status": "ready"},
    ]
    if args.official_feature_file and args.official_feature_file.exists():
        official_dst = args.output_dir / "official_gop_features.npz"
        import_official_npz(args.official_feature_file, base, official_dst)
        manifest.append({"variant": "official_gop_features", "feature_npz": str(official_dst), "status": "ready"})
    else:
        manifest.append(
            {
                "variant": "official_gop_features",
                "feature_npz": "",
                "status": "missing_official_intermediate_features",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(manifest).to_csv(args.output_dir / "feature_variants.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote feature variants to {args.output_dir}")


if __name__ == "__main__":
    main()
