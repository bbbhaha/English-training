from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


WORD_RE = re.compile(r"[A-Za-z']+")


def normalize_text(text: str) -> list[str]:
    return [token.strip("'").upper() for token in WORD_RE.findall(str(text)) if token.strip("'")]


def check_text_audio_consistency(
    *,
    target_text: str,
    audio_path: Path | None = None,
    asr_transcript: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transcript = asr_transcript
    asr_source = "manual_transcript"
    asr_available = True
    if transcript is None:
        transcript, asr_source = transcribe_audio(audio_path)
        asr_available = bool(transcript)

    target_words = normalize_text(target_text)
    recognized_words = normalize_text(transcript or "")
    if not asr_available:
        rows = [
            {
                "target_word": word,
                "recognized_word": "",
                "word_index": idx,
                "asr_word_status": "uncertain",
                "asr_missing_word": False,
                "asr_confidence": 0.0,
                "alignment_op": "uncertain",
            }
            for idx, word in enumerate(target_words)
        ]
    else:
        rows = align_words(target_words, recognized_words)

    meta = {
        "target_text_normalized": " ".join(target_words),
        "asr_transcript": transcript or "",
        "asr_transcript_normalized": " ".join(recognized_words),
        "asr_source": asr_source,
        "asr_available": asr_available,
    }
    frame = pd.DataFrame(rows, columns=_columns())
    return frame, meta


def transcribe_audio(audio_path: Path | None) -> tuple[str, str]:
    if audio_path is None:
        return "", "unavailable"
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(audio_path), beam_size=1)
        return " ".join(segment.text.strip() for segment in segments).strip(), "faster_whisper"
    except Exception:
        pass
    try:
        import whisper  # type: ignore

        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path), fp16=False)
        return str(result.get("text", "")).strip(), "openai_whisper"
    except Exception:
        return "", "unavailable"


def align_words(target_words: list[str], recognized_words: list[str]) -> list[dict[str, Any]]:
    n = len(target_words)
    m = len(recognized_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "deletion"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insertion"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target_words[i - 1] == recognized_words[j - 1]:
                candidates = [(dp[i - 1][j - 1], "match")]
            else:
                candidates = [(dp[i - 1][j - 1] + 1, "substitution")]
            candidates.extend(
                [
                    (dp[i - 1][j] + 1, "deletion"),
                    (dp[i][j - 1] + 1, "insertion"),
                ]
            )
            dp[i][j], back[i][j] = min(candidates, key=lambda item: item[0])

    aligned: list[dict[str, Any]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j] if i >= 0 and j >= 0 else ""
        if op in {"match", "substitution"}:
            word_index = i - 1
            status = "match" if op == "match" else "substitution"
            aligned.append(_row(target_words[i - 1], recognized_words[j - 1], word_index, status, op, 1.0))
            i -= 1
            j -= 1
        elif op == "deletion":
            aligned.append(_row(target_words[i - 1], "", i - 1, "deletion", op, 1.0))
            i -= 1
        elif op == "insertion":
            aligned.append(_row("", recognized_words[j - 1], i, "insertion", op, 1.0))
            j -= 1
        else:
            break
    aligned.reverse()
    return aligned


def merge_consistency_into_phone_frame(
    phone_frame: pd.DataFrame,
    consistency: pd.DataFrame,
    *,
    asr_transcript: str = "",
) -> pd.DataFrame:
    out = phone_frame.copy()
    if consistency.empty:
        return _ensure_asr_columns(out, asr_transcript)
    word_rows = consistency[consistency["target_word"].astype(str).ne("")].copy()
    word_rows = word_rows.rename(columns={"target_word": "word"})
    merge_cols = [
        "word_index",
        "asr_word_status",
        "asr_missing_word",
        "asr_confidence",
        "alignment_op",
        "recognized_word",
    ]
    out = out.drop(columns=[col for col in merge_cols if col != "word_index" and col in out.columns])
    out = out.merge(word_rows[merge_cols], on="word_index", how="left")
    out = _ensure_asr_columns(out, asr_transcript)
    missing = _truthy_series(out["asr_missing_word"])
    out.loc[missing, "decision"] = "true_error"
    out.loc[missing, "error_type"] = "deletion"
    out.loc[missing, "deletion_score"] = 1.0
    out.loc[missing, "deletion_confidence"] = "medium"
    out.loc[missing & _truthy_series(out.get("possible_missing_word", pd.Series(False, index=out.index))), "deletion_confidence"] = "high"
    out.loc[missing, "review_reason"] = [
        _merge_reasons(reason, "missing_in_asr_transcript")
        for reason in out.loc[missing, "review_reason"].fillna("").astype(str)
    ]
    return out


def merge_consistency_into_word_summary(
    summary: pd.DataFrame,
    consistency: pd.DataFrame,
    *,
    asr_transcript: str = "",
) -> pd.DataFrame:
    out = summary.copy()
    if consistency.empty:
        return _ensure_asr_columns(out, asr_transcript)
    word_rows = consistency[consistency["target_word"].astype(str).ne("")].copy()
    merge_cols = [
        "word_index",
        "asr_word_status",
        "asr_missing_word",
        "asr_confidence",
        "alignment_op",
        "recognized_word",
    ]
    out = out.drop(columns=[col for col in merge_cols if col != "word_index" and col in out.columns])
    out = out.merge(word_rows[merge_cols], on="word_index", how="left")
    out = _ensure_asr_columns(out, asr_transcript)
    possible = _truthy_series(out.get("possible_missing_word", pd.Series(False, index=out.index)))
    asr_missing = _truthy_series(out["asr_missing_word"])
    out["deletion_score"] = 0.0
    out["deletion_confidence"] = ""
    out.loc[possible, "deletion_score"] = 0.5
    out.loc[possible, "deletion_confidence"] = "low"
    out.loc[asr_missing, "word_decision"] = "true_error"
    out.loc[asr_missing, "error_type"] = "deletion"
    out.loc[asr_missing, "deletion_score"] = 0.85
    out.loc[asr_missing, "deletion_confidence"] = "medium"
    out.loc[asr_missing & possible, "deletion_score"] = 1.0
    out.loc[asr_missing & possible, "deletion_confidence"] = "high"
    return out


def consistency_to_json_payload(frame: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    return {**meta, "words": frame.to_dict(orient="records")}


def _ensure_asr_columns(frame: pd.DataFrame, asr_transcript: str) -> pd.DataFrame:
    out = frame.copy()
    defaults: dict[str, object] = {
        "asr_transcript": asr_transcript,
        "asr_word_status": "uncertain",
        "asr_missing_word": False,
        "asr_confidence": 0.0,
        "alignment_op": "uncertain",
        "recognized_word": "",
        "deletion_score": 0.0,
        "deletion_confidence": "",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    out["asr_transcript"] = asr_transcript
    return out


def _row(target_word: str, recognized_word: str, word_index: int, status: str, op: str, confidence: float) -> dict[str, Any]:
    return {
        "target_word": target_word,
        "recognized_word": recognized_word,
        "word_index": word_index,
        "asr_word_status": status,
        "asr_missing_word": status == "deletion",
        "asr_confidence": confidence,
        "alignment_op": op,
    }


def _columns() -> list[str]:
    return [
        "target_word",
        "recognized_word",
        "word_index",
        "asr_word_status",
        "asr_missing_word",
        "asr_confidence",
        "alignment_op",
    ]


def _truthy_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _merge_reasons(existing: str, extra: str) -> str:
    parts: list[str] = []
    for value in (existing, extra):
        for part in str(value).split(";"):
            item = part.strip()
            if item and item not in parts:
                parts.append(item)
    return ";".join(parts)
