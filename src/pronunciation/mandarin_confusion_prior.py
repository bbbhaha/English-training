from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "phonology" / "mandarin_english_confusions.yaml"
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
DELETION_PHONES = {"", "DEL", "SIL"}


def classify_mandarin_confusion(
    target_phone: object,
    predicted_phone: object,
    position: str = "medial",
    context: dict[str, Any] | None = None,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    target = _phone(target_phone)
    predicted = _phone(predicted_phone)
    if target == predicted:
        return _result("none", "none", False, False)
    config = _load_config(str(config_path or DEFAULT_CONFIG))
    context = context or {}
    for rule in config.get("rules", []):
        if _matches(rule, target, predicted, position.lower(), context):
            return _result(
                str(rule["name"]),
                str(rule.get("severity", "medium")),
                True,
                bool(rule.get("intelligibility_error", False)),
            )
    return _result("other_phone_difference", "medium", False, False)


@lru_cache(maxsize=8)
def _load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _matches(rule: dict[str, Any], target: str, predicted: str, position: str, context: dict[str, Any]) -> bool:
    positions = {str(value).lower() for value in rule.get("positions", [])}
    if positions and position not in positions:
        return False
    context_flag = rule.get("context_flag")
    if context_flag and not bool(context.get(str(context_flag), False)):
        return False
    targets = {_phone(value) for value in rule.get("targets", [])}
    if targets and target not in targets:
        return False
    if rule.get("target_class") == "consonant" and target in VOWELS:
        return False
    predictions = {_phone(value) for value in rule.get("predictions", [])}
    if predictions and predicted not in predictions:
        return False
    pairs = [({_phone(pair[0]), _phone(pair[1])}) for pair in rule.get("pairs", [])]
    if pairs and {target, predicted} not in pairs:
        return False
    return bool(targets or predictions or pairs or rule.get("target_class") or context_flag)


def _phone(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return text.strip().upper().rstrip("0123456789")


def _result(confusion_type: str, severity: str, common: bool, intelligibility: bool) -> dict[str, Any]:
    return {
        "confusion_type": confusion_type,
        "severity": severity,
        "is_common_mandarin_error": common,
        "is_likely_intelligibility_error": intelligibility,
    }
