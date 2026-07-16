"""Compatibility wrapper for the unified text/audio consistency module."""

from pronunciation.text_audio_consistency import compare_target_with_asr as _compare


def compare_target_with_asr(target_text: str, asr_transcript: str, *, asr_confidence: float = 1.0):
    """Preserve the original public vocabulary for legacy callers."""
    frame = _compare(target_text, asr_transcript, asr_confidence=asr_confidence)
    frame["asr_word_status"] = frame["asr_word_status"].replace(
        {"matched": "match", "missing": "deletion", "substituted": "substitution"}
    )
    frame["asr_edit_op"] = frame["asr_edit_op"].replace({"equal": "match", "replace": "substitute"})
    return frame

__all__ = ["compare_target_with_asr"]
