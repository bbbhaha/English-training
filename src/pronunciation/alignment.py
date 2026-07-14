from __future__ import annotations

from pathlib import Path
import wave

import joblib
import pandas as pd

from phoneme_assessment.acoustic import read_wav_mono
from phoneme_assessment.alignment import align_signal
from phoneme_assessment.phones import normalize_phone

from .g2p import G2PResult, text_to_phones


def audio_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return 1000.0 * handle.getnframes() / handle.getframerate()


def align_audio_to_text(
    wav_path: Path,
    text: str | None = None,
    phones: list[str] | None = None,
    models_path: Path | None = None,
    duration_priors: dict[str, float] | None = None,
    target_word_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, G2PResult | None]:
    g2p = text_to_phones(text or "", target_word_table=target_word_table) if phones is None else None
    phone_rows = g2p.phones if g2p is not None else [
        {"word": "", "word_index": 0, "target_phone": normalize_phone(p), "phone_index": i, "g2p_source": "provided"}
        for i, p in enumerate(phones or [])
    ]
    target_phones = [normalize_phone(row["target_phone"]) for row in phone_rows if normalize_phone(row["target_phone"])]
    if not target_phones:
        return ensure_alignment_coverage(pd.DataFrame(phone_rows), _bad_alignment_rows(phone_rows, "missing_phone_sequence")), g2p
    try:
        if models_path is None or not models_path.exists():
            raise FileNotFoundError("No acoustic alignment model was provided.")
        models = joblib.load(models_path)
        priors = duration_priors or {}
        rate, signal = read_wav_mono(wav_path)
        result = align_signal(signal, rate, target_phones, models, priors)
        if not getattr(result, "boundaries_ms", None):
            raise RuntimeError("alignment_returned_no_boundaries")
        if len(result.boundaries_ms) != len(phone_rows):
            raise RuntimeError(
                f"alignment_boundary_count_mismatch:{len(result.boundaries_ms)}!={len(phone_rows)}"
            )
        rows = []
        for row, (start_ms, end_ms) in zip(phone_rows, result.boundaries_ms):
            duration = end_ms - start_ms
            quality = judge_alignment_quality(
                duration_ms=duration,
                phone_count=len(target_phones),
                total_phone_duration_ms=result.active_end_ms - result.active_start_ms,
                audio_duration_ms=len(signal) * 1000.0 / rate,
                method_failed=False,
            )
            rows.append(_alignment_row(row, start_ms, end_ms, quality, "segmental_viterbi_gaussian_v1"))
        return ensure_alignment_coverage(pd.DataFrame(phone_rows), pd.DataFrame(rows)), g2p
    except Exception as error:
        rows = _equal_length_rows(
            wav_path,
            phone_rows,
            reason=f"alignment_failed;possible_text_audio_mismatch;{error}",
        )
        return ensure_alignment_coverage(pd.DataFrame(phone_rows), rows), g2p


def ensure_alignment_coverage(g2p_phone_df: pd.DataFrame, alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Left join alignment results onto all expected G2P phone rows."""
    expected = g2p_phone_df.copy()
    alignment = alignment_df.copy()
    if expected.empty:
        return alignment
    keys = [column for column in ("word_index", "phone_index") if column in expected.columns and column in alignment.columns]
    if "word_index" not in keys:
        raise KeyError("G2P and alignment frames must contain word_index.")
    expected["word_index"] = pd.to_numeric(expected["word_index"], errors="raise").astype(int)
    alignment["word_index"] = pd.to_numeric(alignment["word_index"], errors="coerce").astype("Int64")
    if "phone_index" in keys:
        expected["phone_index"] = pd.to_numeric(expected["phone_index"], errors="raise").astype(int)
        alignment["phone_index"] = pd.to_numeric(alignment["phone_index"], errors="coerce").astype("Int64")
    alignment_fields = [
        column
        for column in ("start_ms", "end_ms", "duration_ms", "alignment_quality", "alignment_method", "review_reason")
        if column in alignment.columns
    ]
    right = alignment[keys + alignment_fields].drop_duplicates(keys, keep="last").rename(
        columns={"review_reason": "_alignment_review_reason"}
    )
    out = expected.merge(
        right,
        on=keys,
        how="left",
    )
    missing = out.get("start_ms", pd.Series(float("nan"), index=out.index)).isna() | out.get(
        "end_ms", pd.Series(float("nan"), index=out.index)
    ).isna()
    for column in ("start_ms", "end_ms", "duration_ms"):
        if column not in out.columns:
            out[column] = float("nan")
    if "alignment_quality" not in out.columns:
        out["alignment_quality"] = "bad"
    out.loc[missing, "alignment_quality"] = "bad"
    if "review_reason" not in out.columns:
        out["review_reason"] = ""
    alignment_reason = out.pop("_alignment_review_reason") if "_alignment_review_reason" in out.columns else pd.Series("", index=out.index)
    successful_g2p = out.get("g2p_status", pd.Series("success", index=out.index)).astype(str).ne("failed")
    out.loc[successful_g2p & alignment_reason.fillna("").ne(""), "review_reason"] = alignment_reason
    out.loc[missing & successful_g2p, "review_reason"] = "alignment_missing"
    if "alignment_method" not in out.columns:
        out["alignment_method"] = "none"
    out.loc[missing & out["alignment_method"].fillna("").eq(""), "alignment_method"] = "none"
    return out.sort_values("phone_index", kind="stable").reset_index(drop=True)


def judge_alignment_quality(
    duration_ms: float,
    phone_count: int,
    total_phone_duration_ms: float,
    audio_duration_ms: float,
    method_failed: bool = False,
) -> str:
    if method_failed or phone_count <= 0:
        return "bad"
    if duration_ms < 20.0 or duration_ms > 500.0:
        return "bad"
    if audio_duration_ms <= 0 or total_phone_duration_ms <= 0:
        return "bad"
    ratio = total_phone_duration_ms / audio_duration_ms
    if ratio < 0.35 or ratio > 1.20:
        return "bad"
    return "pass"


def save_alignment_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _equal_length_rows(wav_path: Path, phone_rows: list[dict], reason: str) -> pd.DataFrame:
    duration = _safe_audio_duration_ms(wav_path)
    if not phone_rows:
        phone_rows = [_placeholder_phone_row()]
    count = max(len(phone_rows), 1)
    step = duration / count
    rows = []
    for index, row in enumerate(phone_rows):
        start = index * step
        end = (index + 1) * step
        rows.append(_alignment_row(row, start, end, "bad", "equal_length_fallback", reason))
    return pd.DataFrame(rows)


def _bad_alignment_rows(phone_rows: list[dict], reason: str) -> pd.DataFrame:
    if not phone_rows:
        phone_rows = [_placeholder_phone_row()]
    return pd.DataFrame([_alignment_row(row, 0.0, 0.0, "bad", "none", reason) for row in phone_rows])


def _alignment_row(row: dict, start_ms: float, end_ms: float, quality: str, method: str, note: str = "") -> dict:
    return {
        "word": row.get("word", ""),
        "word_index": int(row.get("word_index", 0) or 0),
        "target_phone": normalize_phone(row.get("target_phone", "")),
        "phone_index": int(row.get("phone_index", 0) or 0),
        "start_ms": round(float(start_ms), 3),
        "end_ms": round(float(end_ms), 3),
        "duration_ms": round(float(end_ms) - float(start_ms), 3),
        "alignment_quality": quality,
        "alignment_method": method,
        "review_reason": note,
        "g2p_source": row.get("g2p_source", ""),
        "g2p_status": row.get("g2p_status", "success"),
        "g2p_error": row.get("g2p_error", ""),
        "word_phone_index": row.get("word_phone_index", 0),
        "lexicon_status": row.get("lexicon_status", ""),
        "g2p_confidence": row.get("g2p_confidence", ""),
        "pronunciation_variant_id": row.get("pronunciation_variant_id", 0),
        "num_pronunciation_variants": row.get("num_pronunciation_variants", 1),
        "selected_pronunciation": row.get("selected_pronunciation", ""),
        "decision": row.get("decision", ""),
        "error_type": row.get("error_type", ""),
    }


def _safe_audio_duration_ms(wav_path: Path) -> float:
    try:
        return audio_duration_ms(wav_path)
    except Exception:
        return 0.0


def _placeholder_phone_row() -> dict:
    return {
        "word": "",
        "word_index": 0,
        "target_phone": "",
        "phone_index": 0,
        "g2p_source": "missing_phone_sequence",
    }
