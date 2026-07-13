from __future__ import annotations

import re
from typing import Any

import pandas as pd


WORD_RE = re.compile(r"[A-Za-z']+")


def normalize_words(text: object) -> list[str]:
    return [token.strip("'").upper() for token in WORD_RE.findall(str(text)) if token.strip("'")]


def compare_target_with_asr(
    target_text: str,
    asr_transcript: str,
    *,
    asr_confidence: float = 1.0,
) -> pd.DataFrame:
    """Align target and recognized words with Levenshtein edit operations."""
    target = normalize_words(target_text)
    recognized = normalize_words(asr_transcript)
    operations = _align(target, recognized)
    rows: list[dict[str, Any]] = []
    for target_word, recognized_word, word_index, operation in operations:
        status = {
            "match": "match",
            "delete": "deletion",
            "substitute": "substitution",
            "insert": "insertion",
        }[operation]
        rows.append(
            {
                "word": target_word,
                "word_index": word_index,
                "recognized_word": recognized_word,
                "asr_word_status": status,
                "asr_edit_op": operation,
                "asr_missing_word": operation == "delete",
                "asr_substituted_word": operation == "substitute",
                "asr_confidence": float(asr_confidence),
            }
        )
    return pd.DataFrame(rows, columns=_columns())


def _align(target: list[str], recognized: list[str]) -> list[tuple[str, str, int, str]]:
    n, m = len(target), len(recognized)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "delete"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "insert"
    priority = {"match": 0, "substitute": 1, "delete": 2, "insert": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal_op = "match" if target[i - 1] == recognized[j - 1] else "substitute"
            diagonal_cost = dp[i - 1][j - 1] + (diagonal_op != "match")
            candidates = [
                (diagonal_cost, priority[diagonal_op], diagonal_op),
                (dp[i - 1][j] + 1, priority["delete"], "delete"),
                (dp[i][j - 1] + 1, priority["insert"], "insert"),
            ]
            cost, _, operation = min(candidates)
            dp[i][j], back[i][j] = int(cost), operation

    aligned: list[tuple[str, str, int, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        operation = back[i][j]
        if operation in {"match", "substitute"}:
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


def _columns() -> list[str]:
    return [
        "word",
        "word_index",
        "recognized_word",
        "asr_word_status",
        "asr_edit_op",
        "asr_missing_word",
        "asr_substituted_word",
        "asr_confidence",
    ]
