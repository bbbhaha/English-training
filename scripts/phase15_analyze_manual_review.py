#!/usr/bin/env python
"""Analyze manually labeled Phase-1.5 review packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ERROR_LABELS = {"true_error", "acceptable_accent"}
NON_ERROR_LABELS = {"correct", "bad_alignment"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manual review labels and suggest verifier downgrades.")
    parser.add_argument("--input", type=Path, default=Path("outputs/phase15_verification/audit/manual_review_packet.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase15_verification/manual_review_analysis"))
    parser.add_argument("--min-group-size", type=int, default=3)
    parser.add_argument("--target-precision", type=float, default=0.40)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    labeled = df[df["manual_review_label"].astype(str).str.strip() != ""].copy()
    if labeled.empty:
        raise SystemExit("No manual_review_label values found.")
    labeled["manual_review_label"] = labeled["manual_review_label"].astype(str).str.strip()
    usable = labeled[labeled["manual_review_label"].isin(ERROR_LABELS | NON_ERROR_LABELS)].copy()
    usable["manual_is_error"] = usable["manual_review_label"].isin(ERROR_LABELS).astype(int)
    usable["manual_non_error_reason"] = usable["manual_review_label"].where(usable["manual_is_error"] == 0, "")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phone = _group_table(usable, "target_phone")
    pattern = _group_table(usable, "evidence_pattern")
    reason = _group_table(usable, "audit_reason")
    phone.to_csv(args.output_dir / "manual_by_phone.csv", index=False, encoding="utf-8-sig")
    pattern.to_csv(args.output_dir / "manual_by_evidence_pattern.csv", index=False, encoding="utf-8-sig")
    reason.to_csv(args.output_dir / "manual_by_audit_reason.csv", index=False, encoding="utf-8-sig")
    suggestions = pd.concat(
        [
            _suggest(phone, "target_phone", args.min_group_size, args.target_precision),
            _suggest(pattern, "evidence_pattern", args.min_group_size, args.target_precision),
            _suggest(reason, "audit_reason", args.min_group_size, args.target_precision),
        ],
        ignore_index=True,
    )
    suggestions.to_csv(args.output_dir / "suggested_downgrade_rules.csv", index=False, encoding="utf-8-sig")
    summary = {
        "input": str(args.input),
        "labeled_rows": int(len(labeled)),
        "usable_rows": int(len(usable)),
        "label_counts": {str(k): int(v) for k, v in labeled["manual_review_label"].value_counts().to_dict().items()},
        "manual_precision_for_reviewed_high_confidence": round(float(usable["manual_is_error"].mean()), 6),
        "target_precision": args.target_precision,
        "suggested_rule_count": int(len(suggestions)),
        "notes": [
            "Treat true_error and acceptable_accent as pronunciation errors under the correction-focused label policy.",
            "Treat correct and bad_alignment as non-error for high-confidence pronunciation error decisions.",
            "Use suggested downgrade rules as dev evidence only; do not claim test precision improvement until re-evaluated on held-out labels.",
        ],
    }
    (args.output_dir / "manual_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not suggestions.empty:
        print("\nSuggested downgrade rules:")
        print(suggestions.head(20).to_string(index=False))


def _group_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    rows = []
    for key, group in df.groupby(col):
        n = len(group)
        true_error = int(group["manual_is_error"].sum())
        non_error = n - true_error
        label_counts = group["manual_review_label"].value_counts().to_dict()
        rows.append(
            {
                col: str(key),
                "reviewed_count": n,
                "manual_true_error_count": true_error,
                "manual_non_error_count": non_error,
                "manual_precision": round(float(true_error / n), 6) if n else 0.0,
                "acceptable_accent_count": int(label_counts.get("acceptable_accent", 0)),
                "correct_count": int(label_counts.get("correct", 0)),
                "bad_alignment_count": int(label_counts.get("bad_alignment", 0)),
                "mean_final_error_score": round(float(pd.to_numeric(group.get("final_error_score", 0), errors="coerce").fillna(0).mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(["reviewed_count", "manual_non_error_count"], ascending=[False, False])


def _suggest(table: pd.DataFrame, key_col: str, min_group_size: int, target_precision: float) -> pd.DataFrame:
    if table.empty or key_col not in table.columns:
        return pd.DataFrame()
    bad = table[(table["reviewed_count"] >= min_group_size) & (table["manual_precision"] < target_precision)].copy()
    if bad.empty:
        return pd.DataFrame()
    bad.insert(0, "rule_type", key_col)
    bad["recommended_action"] = "downgrade_high_confidence_to_uncertain_review"
    bad["reason"] = (
        "manual precision "
        + bad["manual_precision"].astype(str)
        + " below target "
        + str(target_precision)
        + " on "
        + bad["reviewed_count"].astype(str)
        + " reviewed rows"
    )
    return bad


if __name__ == "__main__":
    main()
