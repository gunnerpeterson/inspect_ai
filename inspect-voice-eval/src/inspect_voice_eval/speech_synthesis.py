from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .wave_data import WaveData


class UnsupportedVoiceControlError(ValueError):
    """Raised when a synthesizer cannot honor requested voice controls."""


@dataclass(frozen=True)
class VoiceRequest:
    """Requested voice controls for speech synthesis."""

    voice_id: str | None = None
    speaker_reference_path: str | None = None
    expressiveness: float | None = None
    cfg_weight: float | None = None
    speaking_rate: float | None = None
    pitch: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def requested_controls(self) -> set[str]:
        """Return controls that have been explicitly requested."""
        controls: set[str] = set()
        if self.voice_id is not None:
            controls.add("voice_id")
        if self.speaker_reference_path is not None:
            controls.add("speaker_reference_path")
        if self.expressiveness is not None:
            controls.add("expressiveness")
        if self.cfg_weight is not None:
            controls.add("cfg_weight")
        if self.speaking_rate is not None:
            controls.add("speaking_rate")
        if self.pitch is not None:
            controls.add("pitch")
        if self.tags:
            controls.add("tags")
        return controls


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Backend-neutral text to speech interface."""

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        ...

    @property
    def supported_controls(self) -> set[str]:
        """Voice controls this backend can honor."""
        ...

    async def synthesize(self, text: str, voice: VoiceRequest) -> WaveData:
        """Create speech audio for text."""
        ...


def reject_unsupported_controls(
    backend_name: str, supported_controls: set[str], voice: VoiceRequest
) -> None:
    """Raise when voice asks for controls the backend does not support."""
    unsupported = voice.requested_controls() - supported_controls
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise UnsupportedVoiceControlError(
            f"{backend_name} does not support requested voice controls: {names}"
        )
