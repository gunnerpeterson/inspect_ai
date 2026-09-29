from __future__ import annotations

from typing import Protocol, runtime_checkable

from .wave_data import WaveData


@runtime_checkable
class SpeechTranscriber(Protocol):
    """Backend-neutral speech to text interface."""

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        ...

    async def transcribe(self, audio: WaveData) -> str:
        """Transcribe speech audio."""
        ...
