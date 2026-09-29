from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .wave_data import WaveData

SpeechState = Literal["speech", "silence"]


@dataclass(frozen=True)
class SpeechDecision:
    """Speech detection result for an audio frame."""

    state: SpeechState
    energy: float


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Backend-neutral voice activity detector."""

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        ...

    def detect(self, audio: WaveData) -> SpeechDecision:
        """Classify an audio frame as speech or silence."""
        ...
