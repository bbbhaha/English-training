from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BUILTIN_CMU = {
    "A": ["AH"],
    "AGAIN": ["AH", "G", "EH", "N"],
    "AMERICA": ["AH", "M", "EH", "R", "IH", "K", "AH"],
    "AMERICAN": ["AH", "M", "EH", "R", "AH", "K", "AH", "N"],
    "AN": ["AE", "N"],
    "AND": ["AE", "N", "D"],
    "ARE": ["AA", "R"],
    "BEAR": ["B", "EH", "R"],
    "CALL": ["K", "AO", "L"],
    "CAN": ["K", "AE", "N"],
    "DO": ["D", "UW"],
    "FAMILY": ["F", "AE", "M", "AH", "L", "IY"],
    "GREAT": ["G", "R", "EY", "T"],
    "HAVE": ["HH", "AE", "V"],
    "HE": ["HH", "IY"],
    "HER": ["HH", "ER"],
    "HERE": ["HH", "IH", "R"],
    "HIM": ["HH", "IH", "M"],
    "I": ["AY"],
    "IF": ["IH", "F"],
    "IS": ["IH", "Z"],
    "IT": ["IH", "T"],
    "IT'S": ["IH", "T", "S"],
    "LIKE": ["L", "AY", "K"],
    "MAKE": ["M", "EY", "K"],
    "MIKE": ["M", "AY", "K"],
    "ONE": ["W", "AH", "N"],
    "ORANGE": ["AO", "R", "AH", "N", "JH"],
    "SHE": ["SH", "IY"],
    "THE": ["DH", "AH"],
    "THIS": ["DH", "IH", "S"],
    "TO": ["T", "UW"],
    "WE": ["W", "IY"],
    "WELL": ["W", "EH", "L"],
    "WITH": ["W", "IH", "TH"],
    "YOU": ["Y", "UW"],
}


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


def text_to_phones(text: str) -> G2PResult:
    words = tokenize(text)
    cmu = _load_cmudict()
    fallback = _load_g2p_en()
    word_rows: list[dict[str, Any]] = []
    phone_rows: list[dict[str, Any]] = []
    phone_index = 0
    for word_index, word in enumerate(words):
        phones, source = _lookup_word(word, cmu, fallback)
        start = phone_index
        for phone in phones:
            phone_rows.append(
                {
                    "word": word,
                    "word_index": word_index,
                    "target_phone": phone,
                    "phone_index": phone_index,
                    "g2p_source": source,
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
                "is_oov": source == "oov",
            }
        )
    return G2PResult(text=text, normalized_text=" ".join(words), words=word_rows, phones=phone_rows)


def write_g2p_json(result: G2PResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _lookup_word(word: str, cmu: dict[str, list[str]], fallback) -> tuple[list[str], str]:
    normalized = _strip_possessive(word)
    if normalized in cmu:
        return cmu[normalized], "cmudict"
    if normalized in BUILTIN_CMU:
        return BUILTIN_CMU[normalized], "builtin_cmudict_subset"
    if fallback is not None:
        try:
            phones = [_clean_phone(p) for p in fallback(normalized) if _clean_phone(p)]
            if phones:
                return phones, "g2p_en"
        except Exception:
            pass
    return [], "oov"


def _strip_possessive(word: str) -> str:
    if word.endswith("'S") and len(word) > 2:
        return word[:-2]
    return word


def _clean_phone(phone: str) -> str:
    value = str(phone).strip().upper()
    return re.sub(r"\d+$", "", value)


def _load_cmudict() -> dict[str, list[str]]:
    try:
        import cmudict

        entries = cmudict.entries()
        out: dict[str, list[str]] = {}
        for word, phones in entries:
            key = word.upper()
            out.setdefault(key, [_clean_phone(p) for p in phones])
        return out
    except Exception:
        return {}


def _load_g2p_en():
    try:
        from g2p_en import G2p

        return G2p()
    except Exception:
        return None
