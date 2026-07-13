#!/usr/bin/env python
"""Minimal phase-1 demo predictor.

For public-data demos, pass an utterance id or audio path that exists in the
aligned manifest.  For a new audio file, provide a target-phone sequence; the
script will create approximate equal-length phone segments, which is useful for
interface testing but not a substitute for forced alignment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import wave

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phoneme_assessment.phones import phone_group, normalize_phone


CATEGORICAL_FEATURES = ["target_phone", "phone_group", "speaker_gender"]
NUMERIC_FEATURES = ["duration_ms", "phone_index", "word_index", "speaker_age"]


def audio_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return 1000.0 * handle.getnframes() / handle.getframerate()


def rows_from_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if args.utterance_id:
        rows = manifest[manifest["utterance_id"].astype(str) == str(args.utterance_id)].copy()
    elif args.audio_path:
        audio = str(Path(args.audio_path))
        rows = manifest[
            (manifest["audio_path"].astype(str) == audio)
            | (manifest["audio_path"].astype(str).str.lower() == audio.lower())
            | (manifest["audio_path"].astype(str).apply(lambda p: Path(p).name.lower()) == Path(audio).name.lower())
        ].copy()
    else:
        rows = pd.DataFrame()
    return rows


def rows_from_target_phones(args: argparse.Namespace) -> pd.DataFrame:
    if not args.audio_path or not args.target_phones:
        raise SystemExit("For new audio, provide --audio-path and --target-phones, or use --utterance-id from the manifest.")
    phones = [normalize_phone(p) for p in args.target_phones.replace(",", " ").split() if normalize_phone(p)]
    if not phones:
        raise SystemExit("--target-phones did not contain any usable phones.")
    duration = audio_duration_ms(Path(args.audio_path))
    step = duration / len(phones)
    rows = []
    for idx, phone in enumerate(phones):
        start = idx * step
        end = (idx + 1) * step
        rows.append(
            {
                "utterance_id": args.utterance_id or Path(args.audio_path).stem,
                "speaker_id": args.speaker_id,
                "transcript": args.text or "",
                "word": "",
                "target_phone": phone,
                "phone_index": idx,
                "word_index": 0,
                "speaker_gender": args.speaker_gender,
                "speaker_age": args.speaker_age,
                "start_ms": round(start, 3),
                "end_ms": round(end, 3),
                "duration_ms": round(end - start, 3),
                "phone_group": phone_group(phone),
                "audio_path": str(args.audio_path),
            }
        )
    return pd.DataFrame(rows)


def load_thresholds(path: Path, model_name: str, calibration: str) -> tuple[float, dict[str, float], dict[str, float]]:
    rows = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    rows = rows[rows["model"] == model_name]
    global_rows = rows[rows["level"] == "global"]
    if global_rows.empty:
        raise SystemExit(f"No global threshold for model {model_name!r} in {path}")
    global_threshold = float(global_rows.iloc[0]["threshold"])
    group_thresholds = {
        str(row["key"]): float(row["threshold"])
        for _, row in rows[rows["level"] == "phone_group"].iterrows()
    }
    phone_thresholds = {
        str(row["key"]): float(row["threshold"])
        for _, row in rows[rows["level"] == "target_phone"].iterrows()
    }
    return global_threshold, group_thresholds, phone_thresholds


def choose_threshold(row: pd.Series, calibration: str, global_threshold: float, group_thresholds: dict[str, float], phone_thresholds: dict[str, float]) -> tuple[float, str]:
    if calibration == "global_threshold":
        return global_threshold, "global"
    if calibration == "phone_group_threshold":
        return group_thresholds.get(str(row["phone_group"]), global_threshold), "phone_group"
    if calibration == "target_phone_threshold":
        phone = str(row["target_phone"])
        if phone in phone_thresholds:
            return phone_thresholds[phone], "target_phone"
        return group_thresholds.get(str(row["phone_group"]), global_threshold), "phone_group"
    raise ValueError(calibration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/speechocean/phones_aligned.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/phase1_models/feature_random_forest.joblib")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "reports/phase1/thresholds.csv")
    parser.add_argument("--model-name", default="feature_random_forest")
    parser.add_argument("--calibration", choices=["global_threshold", "phone_group_threshold", "target_phone_threshold"], default="global_threshold")
    parser.add_argument("--utterance-id")
    parser.add_argument("--audio-path")
    parser.add_argument("--text", help="Kept in the output for demo compatibility; current script does not perform G2P.")
    parser.add_argument("--target-phones", help="Required for new audio that is not already in the aligned manifest.")
    parser.add_argument("--speaker-id", default="demo")
    parser.add_argument("--speaker-gender", default="")
    parser.add_argument("--speaker-age", type=float, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase1/demo_prediction.csv")
    args = parser.parse_args()

    rows = rows_from_manifest(args)
    if rows.empty:
        rows = rows_from_target_phones(args)
        rows["demo_alignment_note"] = "approximate_equal_length_segments"
    else:
        rows = rows.sort_values(["utterance_id", "phone_index"]).copy()
        rows["demo_alignment_note"] = "manifest_aligned_boundaries"
    if "alignment_quality" in rows.columns:
        rows = rows[rows["alignment_quality"].astype(str).str.lower().isin(["pass", ""])].copy()
    for col in CATEGORICAL_FEATURES:
        if col not in rows.columns:
            rows[col] = ""
        rows[col] = rows[col].fillna("").astype(str)
    for col in NUMERIC_FEATURES:
        if col not in rows.columns:
            rows[col] = 0
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0)

    model = joblib.load(args.model)
    classes = list(model.named_steps["model"].classes_)
    prob_correct = model.predict_proba(rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, classes.index(1)]
    global_threshold, group_thresholds, phone_thresholds = load_thresholds(args.thresholds, args.model_name, args.calibration)

    out_rows = []
    for (_, row), score in zip(rows.iterrows(), prob_correct):
        threshold, source = choose_threshold(row, args.calibration, global_threshold, group_thresholds, phone_thresholds)
        prediction = int(float(score) >= threshold)
        out_rows.append(
            {
                "utterance_id": row.get("utterance_id", ""),
                "speaker_id": row.get("speaker_id", ""),
                "transcript": row.get("transcript", args.text or ""),
                "word": row.get("word", ""),
                "target_phone": row["target_phone"],
                "phone_index": int(row["phone_index"]),
                "start_ms": row.get("start_ms", ""),
                "end_ms": row.get("end_ms", ""),
                "duration_ms": row.get("duration_ms", ""),
                "prediction": prediction,
                "prediction_label": "correct_or_acceptable" if prediction else "error_or_unacceptable",
                "confidence": round(max(float(score), 1.0 - float(score)), 6),
                "prob_correct": round(float(score), 6),
                "threshold": threshold,
                "threshold_source": source,
                "phone_group": row.get("phone_group", ""),
                "audio_path": row.get("audio_path", args.audio_path or ""),
                "demo_alignment_note": row.get("demo_alignment_note", ""),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote demo prediction to {args.output}")


if __name__ == "__main__":
    main()
