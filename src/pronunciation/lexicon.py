from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUSTOM_LEXICON = ROOT / "configs" / "pronunciation" / "custom_lexicon.yaml"


def load_custom_lexicon(path: str | Path = DEFAULT_CUSTOM_LEXICON) -> dict[str, list[list[str]]]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    lexicon: dict[str, list[list[str]]] = {}
    for word, variants in payload.items():
        normalized = _normalize_word(word)
        cleaned = [_clean_pronunciation(variant) for variant in (variants or [])]
        lexicon[normalized] = [variant for variant in cleaned if variant]
    return lexicon


def lookup_pronunciation(word: str) -> dict[str, Any]:
    return get_best_pronunciation(word)


def get_pronunciations(word: str) -> list[list[str]]:
    return list(get_best_pronunciation(word)["pronunciations"])


def get_best_pronunciation(word: str, context: object = None) -> dict[str, Any]:
    """Return ordered pronunciation candidates and select the first variant for v1."""
    del context
    normalized = _normalize_word(word)
    custom = _custom_lexicon()
    if normalized in custom:
        return _result(word, normalized, custom[normalized], "custom", "high")

    cmu = _cmudict_pronunciations(normalized)
    if cmu:
        return _result(word, normalized, cmu, "cmudict", "high")

    g2p = _g2p_en_pronunciation(normalized)
    if g2p:
        return _result(word, normalized, [g2p], "g2p_en", "medium")

    phonemized = _phonemizer_pronunciation(normalized)
    if phonemized:
        return _result(word, normalized, [phonemized], "phonemizer", "medium")

    return _result(
        word,
        normalized,
        [],
        "failed",
        "low",
        error="oov_or_g2p_failed",
    )


def _result(
    word: str,
    normalized: str,
    pronunciations: list[list[str]],
    status: str,
    confidence: str,
    error: str = "",
) -> dict[str, Any]:
    variants = [_clean_pronunciation(value) for value in pronunciations]
    variants = [value for value in variants if value]
    selected = variants[0] if variants else []
    return {
        "word": str(word),
        "normalized_word": normalized,
        "pronunciations": variants,
        "selected_pronunciation": selected,
        "lexicon_status": status,
        "g2p_confidence": confidence,
        "g2p_source": status,
        "g2p_error": error,
        "pronunciation_variant_id": 0 if selected else -1,
        "num_pronunciation_variants": len(variants),
    }


@lru_cache(maxsize=1)
def _custom_lexicon() -> dict[str, list[list[str]]]:
    return load_custom_lexicon(DEFAULT_CUSTOM_LEXICON)


@lru_cache(maxsize=1)
def _cmudict() -> dict[str, list[list[str]]]:
    try:
        import cmudict

        source = cmudict.dict()
        return {
            str(word).upper(): [_clean_pronunciation(variant) for variant in variants]
            for word, variants in source.items()
        }
    except Exception:
        pass
    try:
        import pronouncing

        words = getattr(pronouncing, "cmudict", None)
        source = words.dict() if words is not None else {}
        return {
            str(word).upper(): [_clean_pronunciation(variant) for variant in variants]
            for word, variants in source.items()
        }
    except Exception:
        return {}


def _cmudict_pronunciations(word: str) -> list[list[str]]:
    variants = _cmudict().get(word, [])
    return _deduplicate(variants)


@lru_cache(maxsize=1)
def _g2p_engine():
    try:
        from g2p_en import G2p

        return G2p()
    except Exception:
        return None


def _g2p_en_pronunciation(word: str) -> list[str]:
    engine = _g2p_engine()
    if engine is None:
        return []
    try:
        return _clean_pronunciation(engine(word))
    except Exception:
        return []


def _phonemizer_pronunciation(word: str) -> list[str]:
    ipa = ""
    try:
        from phonemizer import phonemize

        ipa = str(phonemize(word.lower(), language="en-us", backend="espeak", strip=True))
    except Exception:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable:
            try:
                result = subprocess.run(
                    [executable, "-q", "--ipa=3", "-v", "en-us", word.lower()],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                ipa = result.stdout.strip()
            except Exception:
                ipa = ""
    return _ipa_to_arpabet(ipa)


def _ipa_to_arpabet(ipa: str) -> list[str]:
    if not ipa:
        return []
    mapping = {
        "tʃ": "CH", "dʒ": "JH", "eɪ": "EY", "aɪ": "AY", "ɔɪ": "OY",
        "aʊ": "AW", "oʊ": "OW", "ər": "ER", "ɝ": "ER", "ɚ": "ER",
        "i": "IY", "ɪ": "IH", "e": "EH", "ɛ": "EH", "æ": "AE",
        "ɑ": "AA", "ɒ": "AA", "ɔ": "AO", "ʌ": "AH", "ə": "AH",
        "u": "UW", "ʊ": "UH", "p": "P", "b": "B", "t": "T",
        "d": "D", "k": "K", "ɡ": "G", "g": "G", "f": "F",
        "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
        "ʃ": "SH", "ʒ": "ZH", "h": "HH", "m": "M", "n": "N",
        "ŋ": "NG", "l": "L", "r": "R", "ɹ": "R", "j": "Y", "w": "W",
    }
    cleaned = re.sub(r"[ˈˌː\.\-\s]", "", str(ipa))
    tokens: list[str] = []
    index = 0
    keys = sorted(mapping, key=len, reverse=True)
    while index < len(cleaned):
        key = next((candidate for candidate in keys if cleaned.startswith(candidate, index)), None)
        if key is None:
            return []
        tokens.append(mapping[key])
        index += len(key)
    return tokens


def _normalize_word(word: object) -> str:
    return str(word).replace("’", "'").strip().upper()


def _clean_pronunciation(value: object) -> list[str]:
    tokens = value.split() if isinstance(value, str) else list(value or [])
    return [re.sub(r"\d+$", "", str(token).strip().upper()) for token in tokens if str(token).strip()]


def _deduplicate(variants: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    result = []
    for variant in variants:
        cleaned = _clean_pronunciation(variant)
        key = tuple(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
