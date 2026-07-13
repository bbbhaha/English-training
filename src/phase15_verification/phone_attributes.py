from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


BOOLEAN_ATTRIBUTES = [
    "rhotic",
    "nasal",
    "stop",
    "fricative",
    "affricate",
    "approximant",
    "lateral",
    "rounded",
    "final_consonant",
]

CONFUSION_SETS = {
    "R": ["L"],
    "L": ["R"],
    "V": ["W", "F"],
    "W": ["V"],
    "F": ["V", "TH"],
    "TH": ["S", "T", "F"],
    "DH": ["Z", "D"],
    "N": ["NG"],
    "NG": ["N"],
    "IY": ["IH"],
    "IH": ["IY"],
    "EY": ["EH"],
    "EH": ["AE", "EY"],
    "AE": ["EH"],
    "UW": ["UH"],
    "UH": ["UW"],
}


def normalize_phone(phone: Any) -> str:
    value = "" if phone is None else str(phone).strip().upper().replace("*", "")
    return re.sub(r"\d+$", "", value)


def load_phone_attributes(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {normalize_phone(k): dict(v) for k, v in raw.items()}


def expected_attributes(phone: Any, mapping: dict[str, dict[str, Any]], final_consonant: bool = False) -> dict[str, Any]:
    value = normalize_phone(phone)
    attrs = dict(mapping.get(value, {}))
    attrs.setdefault("phone", value)
    attrs.setdefault("vowel_consonant", "unknown")
    attrs["final_consonant"] = bool(final_consonant and attrs.get("vowel_consonant") == "consonant")
    for key in BOOLEAN_ATTRIBUTES:
        attrs.setdefault(key, False)
    return attrs


def serialize_attributes(attrs: dict[str, Any]) -> str:
    return json.dumps(attrs, ensure_ascii=True, sort_keys=True)


def confusable_phones(phone: Any) -> list[str]:
    return CONFUSION_SETS.get(normalize_phone(phone), [])
