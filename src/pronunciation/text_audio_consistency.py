from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


WORD_RE = re.compile(r"[A-Za-z']+")


def normalize_text(text: str) -> list[str]:
    return [token.strip("'").upper() for token in WORD_RE.findall(str(text)) if token.strip("'")]


def compare_target_with_asr(
    target_text: str,
    asr_transcript: str,
    *,
    asr_confidence: float = 1.0,
) -> pd.DataFrame:
    """Align every target word with an ASR transcript using word edit distance."""
    target = normalize_text(target_text)
    recognized = normalize_text(asr_transcript)
    operations = _align_operations(target, recognized)
    rows: list[dict[str, Any]] = []
    insertions: list[tuple[int, str]] = []
    for target_word, asr_word, word_index, operation in operations:
        if operation == "insert":
            insertions.append((word_index, asr_word))
            continue
        status = {"equal": "matched", "delete": "missing", "replace": "substituted"}[operation]
        mismatch_type = {"equal": "none", "delete": "missing_word", "replace": "substituted_word"}[operation]
        mismatch_score = {"equal": 0.0, "delete": 0.95, "replace": 0.85}[operation]
        rows.append(
            {
                "word_index": word_index,
                "word": target_word,
                "target_word": target_word,
                "asr_word": asr_word,
                "recognized_word": asr_word,
                "asr_edit_op": operation,
                "alignment_op": operation,
                "asr_word_status": status,
                "asr_missing_word": operation == "delete",
                "asr_substituted_word": operation == "replace",
                "asr_inserted_nearby": False,
                "asr_confidence": float(asr_confidence),
                "text_audio_mismatch": operation != "equal",
                "text_audio_mismatch_type": mismatch_type,
                "text_audio_mismatch_score": mismatch_score,
            }
        )

    frame = pd.DataFrame(rows, columns=_comparison_columns())
    for insertion_index, inserted_word in insertions:
        if frame.empty:
            continue
        candidates = frame.index[frame["word_index"].le(max(insertion_index - 1, 0))]
        row_index = candidates[-1] if len(candidates) else frame.index[0]
        frame.loc[row_index, "asr_inserted_nearby"] = True
        if frame.loc[row_index, "text_audio_mismatch_type"] == "none":
            frame.loc[row_index, "asr_word_status"] = "inserted_nearby"
            frame.loc[row_index, "asr_edit_op"] = "insert"
            frame.loc[row_index, "alignment_op"] = "insert"
            frame.loc[row_index, "text_audio_mismatch"] = True
            frame.loc[row_index, "text_audio_mismatch_type"] = "extra_word"
            frame.loc[row_index, "text_audio_mismatch_score"] = 0.70
        current = str(frame.loc[row_index, "asr_word"] or "")
        frame.loc[row_index, "asr_word"] = " ".join(part for part in (current, inserted_word) if part)
        frame.loc[row_index, "recognized_word"] = frame.loc[row_index, "asr_word"]

    mismatch_count = int(frame["text_audio_mismatch"].sum()) if not frame.empty else 0
    if len(target) and mismatch_count / len(target) >= 0.6:
        mismatch = frame["text_audio_mismatch"].astype(bool)
        frame.loc[mismatch, "text_audio_mismatch_type"] = "severe_mismatch"
    return _add_context_evidence(frame)


def check_text_audio_consistency(
    *,
    target_text: str,
    audio_path: Path | None = None,
    asr_transcript: str | None = None,
    asr_model: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transcript = asr_transcript
    source = "manual_transcript"
    automatic_confidence = 1.0
    asr_error = ""
    if transcript is None:
        transcript, source, automatic_confidence, asr_error = transcribe_audio_detailed(
            audio_path,
            asr_model=asr_model,
        )
    available = bool(str(transcript or "").strip())
    if available:
        frame = compare_target_with_asr(
            target_text,
            str(transcript),
            asr_confidence=1.0 if asr_transcript is not None else automatic_confidence,
        )
        # Keep the original check_* API stable; the new compare_* API exposes
        # the normalized matched/missing/substituted vocabulary.
        frame["asr_word_status"] = frame["asr_word_status"].replace(
            {"matched": "match", "missing": "deletion", "substituted": "substitution"}
        )
        frame["alignment_op"] = frame["alignment_op"].replace({"equal": "match", "replace": "substitution"})
    else:
        frame = _not_checked_frame(target_text)
    target_words = normalize_text(target_text)
    recognized_words = normalize_text(transcript or "")
    meta = {
        "target_text_normalized": " ".join(target_words),
        "asr_transcript": transcript or "",
        "asr_transcript_normalized": " ".join(recognized_words),
        "asr_source": source,
        "asr_confidence": automatic_confidence if asr_transcript is None else 1.0,
        "asr_error": asr_error,
        "asr_available": available,
        "text_audio_consistency_status": "checked" if available else "not_checked",
    }
    return frame, meta


def transcribe_audio(audio_path: Path | None, *, asr_model: str = "auto") -> tuple[str, str]:
    transcript, source, _, _ = transcribe_audio_detailed(audio_path, asr_model=asr_model)
    return transcript, source


def transcribe_audio_detailed(
    audio_path: Path | None,
    *,
    asr_model: str = "auto",
) -> tuple[str, str, float, str]:
    if audio_path is None:
        return "", "unavailable", 0.0, "audio_path_missing"
    if asr_model in {"auto", "faster_whisper"}:
        try:
            model = _faster_whisper_model()
            segments, _ = model.transcribe(
                str(audio_path),
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                word_timestamps=True,
            )
            segment_list = list(segments)
            transcript = " ".join(segment.text.strip() for segment in segment_list).strip()
            probabilities = [
                float(word.probability)
                for segment in segment_list
                for word in (getattr(segment, "words", None) or [])
                if getattr(word, "probability", None) is not None
            ]
            confidence = sum(probabilities) / len(probabilities) if probabilities else 0.65
            return transcript, "faster_whisper_base_en", confidence, ""
        except Exception as error:
            if asr_model == "faster_whisper":
                return "", "unavailable", 0.0, f"{type(error).__name__}: {error}"
    if asr_model in {"auto", "whisper"}:
        try:
            import whisper  # type: ignore

            result = whisper.load_model("base").transcribe(str(audio_path), fp16=False)
            return str(result.get("text", "")).strip(), "whisper", 0.65, ""
        except Exception as error:
            if asr_model == "whisper":
                return "", "unavailable", 0.0, f"{type(error).__name__}: {error}"
            pass
    return "", "unavailable", 0.0, "no_supported_asr_backend"


@lru_cache(maxsize=1)
def _faster_whisper_model():
    from faster_whisper import WhisperModel  # type: ignore

    return WhisperModel("base.en", device="cpu", compute_type="int8")


def align_words(target_words: list[str], recognized_words: list[str]) -> list[dict[str, Any]]:
    """Compatibility wrapper for callers using the original row schema."""
    return compare_target_with_asr(" ".join(target_words), " ".join(recognized_words)).to_dict(orient="records")


def merge_consistency_into_phone_frame(
    phone_frame: pd.DataFrame,
    consistency: pd.DataFrame,
    *,
    asr_transcript: str = "",
) -> pd.DataFrame:
    out = _merge_by_word_index(phone_frame, consistency)
    out = _ensure_asr_columns(out, asr_transcript, bool(asr_transcript))
    missing = _truthy_series(out["asr_missing_word"])
    out.loc[missing, "decision"] = "true_error"
    out.loc[missing, "error_type"] = "deletion"
    out.loc[missing, "review_reason"] = "missing_in_asr_transcript"
    return out


def merge_consistency_into_word_summary(
    summary: pd.DataFrame,
    consistency: pd.DataFrame,
    *,
    asr_transcript: str = "",
) -> pd.DataFrame:
    out = _merge_by_word_index(summary, consistency)
    return _ensure_asr_columns(out, asr_transcript, bool(asr_transcript))


def consistency_to_json_payload(frame: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    return {**meta, "words": frame.where(pd.notna(frame), None).to_dict(orient="records")}


def _align_operations(target: list[str], recognized: list[str]) -> list[tuple[str, str, int, str]]:
    n, m = len(target), len(recognized)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "delete"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "insert"
    priority = {"equal": 0, "replace": 1, "delete": 2, "insert": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = "equal" if target[i - 1] == recognized[j - 1] else "replace"
            candidates = [
                (dp[i - 1][j - 1] + (diagonal != "equal"), priority[diagonal], diagonal),
                (dp[i - 1][j] + 1, priority["delete"], "delete"),
                (dp[i][j - 1] + 1, priority["insert"], "insert"),
            ]
            cost, _, operation = min(candidates)
            dp[i][j], back[i][j] = int(cost), operation
    aligned: list[tuple[str, str, int, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        operation = back[i][j]
        if operation in {"equal", "replace"}:
            aligned.append((target[i - 1], recognized[j - 1], i - 1, operation))
            i, j = i - 1, j - 1
        elif operation == "delete":
            aligned.append((target[i - 1], "", i - 1, operation))
            i -= 1
        elif operation == "insert":
            aligned.append(("", recognized[j - 1], i, operation))
            j -= 1
        else:
            break
    aligned.reverse()
    return aligned


def _not_checked_frame(target_text: str) -> pd.DataFrame:
    rows = []
    for index, word in enumerate(normalize_text(target_text)):
        rows.append({
            "word_index": index, "word": word, "target_word": word, "asr_word": "", "recognized_word": "",
            "asr_edit_op": "not_checked", "alignment_op": "not_checked", "asr_word_status": "not_checked",
            "asr_missing_word": False, "asr_substituted_word": False, "asr_inserted_nearby": False,
            "asr_confidence": 0.0, "text_audio_mismatch": False, "text_audio_mismatch_type": "none",
            "text_audio_mismatch_score": 0.0,
        })
    return pd.DataFrame(rows, columns=_comparison_columns())


def _merge_by_word_index(frame: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if consistency.empty or "word_index" not in out.columns:
        return out
    columns = [column for column in _comparison_columns() if column not in {"word", "target_word"} and column in consistency.columns]
    out = out.drop(columns=[column for column in columns if column != "word_index" and column in out.columns])
    return out.merge(consistency[columns].drop_duplicates("word_index", keep="last"), on="word_index", how="left")


def _ensure_asr_columns(frame: pd.DataFrame, transcript: str, available: bool) -> pd.DataFrame:
    out = frame.copy()
    defaults: dict[str, object] = {
        "asr_available": available, "asr_transcript": transcript, "text_audio_consistency_status": "checked" if available else "not_checked",
        "asr_word": "", "recognized_word": "", "asr_edit_op": "not_checked", "alignment_op": "not_checked",
        "asr_word_status": "not_checked", "asr_missing_word": False, "asr_substituted_word": False,
        "asr_inserted_nearby": False, "asr_confidence": 0.0, "text_audio_mismatch": False,
        "text_audio_mismatch_type": "none", "text_audio_mismatch_score": 0.0,
        "asr_context_support": 0.0, "asr_missing_confidence": 0.0,
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
        else:
            out[column] = out[column].fillna(default)
    out["asr_available"] = available
    out["asr_transcript"] = transcript
    out["text_audio_consistency_status"] = "checked" if available else "not_checked"
    return out


def _comparison_columns() -> list[str]:
    return [
        "word_index", "word", "target_word", "asr_word", "recognized_word", "asr_edit_op", "alignment_op",
        "asr_word_status", "asr_missing_word", "asr_substituted_word", "asr_inserted_nearby", "asr_confidence",
        "text_audio_mismatch", "text_audio_mismatch_type", "text_audio_mismatch_score",
        "asr_context_support", "asr_missing_confidence",
    ]


def _add_context_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    matched = out["asr_word_status"].eq("matched")
    supports: list[float] = []
    missing_confidences: list[float] = []
    for position, (_, row) in enumerate(out.iterrows()):
        anchors = []
        if position > 0:
            anchors.append(bool(matched.iloc[position - 1]))
        if position + 1 < len(out):
            anchors.append(bool(matched.iloc[position + 1]))
        context_support = sum(anchors) / len(anchors) if anchors else 0.0
        confidence = float(row.get("asr_confidence", 0.0) or 0.0)
        supports.append(context_support)
        missing_confidences.append(confidence * context_support if bool(row.get("asr_missing_word")) else 0.0)
    out["asr_context_support"] = supports
    out["asr_missing_confidence"] = missing_confidences
    return out


def _truthy_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
