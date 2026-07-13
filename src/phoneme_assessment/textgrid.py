"""Small dependency-free parser for Praat long TextGrid interval tiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    text: str


_QUOTED = re.compile(r'=\s*"(.*)"\s*$')
_NUMBER = re.compile(r"=\s*([-+0-9.eE]+)\s*$")


def _quoted_value(line: str) -> str:
    match = _QUOTED.search(line)
    return match.group(1).replace('""', '"') if match else ""


def _number_value(line: str) -> float:
    match = _NUMBER.search(line)
    if not match:
        raise ValueError(f"Expected numeric TextGrid value: {line!r}")
    return float(match.group(1))


def read_interval_tiers(path: str | Path) -> dict[str, list[Interval]]:
    """Read interval tiers from a standard long-form TextGrid file."""
    tiers: dict[str, list[Interval]] = {}
    current_tier: str | None = None
    current: dict[str, float | str] | None = None

    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line.startswith("name ="):
            current_tier = _quoted_value(line)
            tiers.setdefault(current_tier, [])
        elif line.startswith("intervals ["):
            current = {}
        elif current is not None and line.startswith("xmin ="):
            current["start"] = _number_value(line)
        elif current is not None and line.startswith("xmax ="):
            current["end"] = _number_value(line)
        elif current is not None and line.startswith("text ="):
            current["text"] = _quoted_value(line)
            if current_tier and {"start", "end", "text"} <= current.keys():
                tiers[current_tier].append(
                    Interval(
                        start=float(current["start"]),
                        end=float(current["end"]),
                        text=str(current["text"]),
                    )
                )
            current = None
    return tiers


def word_at_interval(words: list[Interval], phone: Interval) -> str:
    """Return the word with the greatest temporal overlap with a phone."""
    best_word = ""
    best_overlap = 0.0
    for word in words:
        overlap = max(0.0, min(word.end, phone.end) - max(word.start, phone.start))
        if word.text.strip() and overlap > best_overlap:
            best_overlap = overlap
            best_word = word.text.strip()
    return best_word

