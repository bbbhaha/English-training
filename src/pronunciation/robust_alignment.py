from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .g2p import G2PResult


@dataclass
class RobustAlignmentRequest:
    audio_path: Path
    text: str | None
    phones: list[str] | None
    g2p: G2PResult | None = None


class AlignmentBackend(Protocol):
    name: str

    def align(self, request: RobustAlignmentRequest) -> pd.DataFrame:
        """Return phone-level alignment rows or raise a backend-specific error."""


class RobustAligner:
    """Placeholder orchestrator for future MFA, wav2vec2 CTC, or WhisperX backends."""

    def __init__(self, backends: list[AlignmentBackend] | None = None) -> None:
        self.backends = backends or []

    def align(self, request: RobustAlignmentRequest) -> pd.DataFrame:
        errors: list[str] = []
        for backend in self.backends:
            try:
                frame = backend.align(request)
                if frame is not None and not frame.empty:
                    frame = frame.copy()
                    frame["alignment_backend"] = backend.name
                    return frame
            except Exception as error:
                errors.append(f"{backend.name}:{error}")
        joined = ";".join(errors) if errors else "no_alignment_backend_configured"
        raise RuntimeError(joined)

