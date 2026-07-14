from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .lexicon import get_best_pronunciation
from .target_words import build_target_word_table


@dataclass
class G2PResult:
    text: str
    normalized_text: str
    words: list[dict[str, Any]]
    phones: list[dict[str, Any]]

    @property
    def phone_sequence(self) -> list[str]:
        return [row["target_phone"] for row in self.phones]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "phone_sequence": self.phone_sequence,
            "words": self.words,
            "phones": self.phones,
        }


def normalize_text(text: str) -> str:
    text = text.upper().replace("’", "'")
    text = re.sub(r"[^A-Z0-9'\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if token]


def text_to_phones(text: str, target_word_table: pd.DataFrame | None = None) -> G2PResult:
    target_words = build_target_word_table(text) if target_word_table is None else target_word_table.copy()
    return target_word_table_to_phones(target_words, text=text)


def target_word_table_to_phones(target_word_table: pd.DataFrame, text: str = "") -> G2PResult:
    """Produce at least one phone row for every canonical target word."""
    word_rows: list[dict[str, Any]] = []
    phone_rows: list[dict[str, Any]] = []
    phone_index = 0
    for _, target_row in target_word_table.sort_values("word_index", kind="stable").iterrows():
        word_index = int(target_row["word_index"])
        word = str(target_row.get("normalized_word", target_row.get("word", ""))).upper()
        lookup = get_best_pronunciation(word)
        phones = list(lookup["selected_pronunciation"])
        source = str(lookup["g2p_source"])
        g2p_status = "success" if lookup["lexicon_status"] != "failed" else "failed"
        g2p_error = str(lookup["g2p_error"])
        if not phones:
            phones = ["<UNK>"]
        start = phone_index
        for word_phone_index, phone in enumerate(phones):
            phone_rows.append(
                {
                    "word": word,
                    "word_index": word_index,
                    "target_phone": phone,
                    "phone_index": phone_index,
                    "word_phone_index": word_phone_index,
                    "g2p_source": source,
                    "g2p_status": g2p_status,
                    "g2p_error": g2p_error,
                    "lexicon_status": lookup["lexicon_status"],
                    "g2p_confidence": lookup["g2p_confidence"],
                    "pronunciation_variant_id": lookup["pronunciation_variant_id"],
                    "num_pronunciation_variants": lookup["num_pronunciation_variants"],
                    "selected_pronunciation": " ".join(phones),
                    "decision": "uncertain_review" if g2p_status == "failed" else "",
                    "error_type": "g2p_issue" if g2p_status == "failed" else "",
                    "review_reason": "g2p_failed" if g2p_status == "failed" else "",
                }
            )
            phone_index += 1
        end = phone_index - 1 if phones else start - 1
        word_rows.append(
            {
                "word": word,
                "word_index": word_index,
                "phones": phones,
                "g2p_source": source,
                "phone_index_start": start,
                "phone_index_end": end,
                "is_oov": lookup["lexicon_status"] == "failed",
                "g2p_status": g2p_status,
                "g2p_error": g2p_error,
                "pronunciations": lookup["pronunciations"],
                "selected_pronunciation": lookup["selected_pronunciation"],
                "lexicon_status": lookup["lexicon_status"],
                "g2p_confidence": lookup["g2p_confidence"],
                "pronunciation_variant_id": lookup["pronunciation_variant_id"],
                "num_pronunciation_variants": lookup["num_pronunciation_variants"],
                "decision": "uncertain_review" if g2p_status == "failed" else "",
                "error_type": "g2p_issue" if g2p_status == "failed" else "",
            }
        )
    normalized_text = " ".join(str(row["word"]) for row in word_rows)
    return G2PResult(text=text, normalized_text=normalized_text, words=word_rows, phones=phone_rows)


def write_g2p_json(result: G2PResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
