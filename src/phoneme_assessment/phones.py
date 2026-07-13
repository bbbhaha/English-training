"""Phone normalization and coarse articulatory grouping."""

from __future__ import annotations

import re

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}
STOPS = {"P", "B", "T", "D", "K", "G"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
AFFRICATES = {"CH", "JH"}
NASALS = {"M", "N", "NG"}
LIQUIDS = {"L", "R"}
GLIDES = {"W", "Y"}
SILENCES = {"", "SIL", "SP", "SPN"}


def normalize_phone(phone: str | None) -> str:
    if phone is None:
        return ""
    value = phone.strip().upper().replace("*", "")
    return re.sub(r"\d+$", "", value)


def phone_group(phone: str | None) -> str:
    value = normalize_phone(phone)
    if value in VOWELS:
        return "vowel"
    if value in STOPS:
        return "stop"
    if value in FRICATIVES:
        return "fricative"
    if value in AFFRICATES:
        return "affricate"
    if value in NASALS:
        return "nasal"
    if value in LIQUIDS:
        return "liquid"
    if value in GLIDES:
        return "glide"
    if value in SILENCES:
        return "silence"
    return "other"

