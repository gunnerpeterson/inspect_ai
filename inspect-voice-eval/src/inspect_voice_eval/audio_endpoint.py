from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .speech_detection import VoiceActivityDetector
from .wave_data import WaveData


class VoiceTurnTimeoutError(RuntimeError):
    """Raised when a voice turn exceeds a bounded wait."""


@dataclass(frozen=True)
class RecordingResult:
    """Recorded response audio and speech boundary timing."""

    audio: WaveData
    speech_started_ms: int
    speech_completed_ms: int


@runtime_checkable
class AudioEndpoint(Protocol):
    """Backend-neutral audio input/output endpoint."""

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        ...

    async def play(self, audio: WaveData) -> None:
        """Play audio to the target application route."""
        ...

    async def record_until(
        self,
        detector: VoiceActivityDetector,
        *,
        frame_duration_ms: int,
        end_silence_ms: int,
        min_speech_ms: int,
        max_recording_ms: int,
    ) -> RecordingResult:
        """Record until speech is followed by sustained silence."""
        ...
