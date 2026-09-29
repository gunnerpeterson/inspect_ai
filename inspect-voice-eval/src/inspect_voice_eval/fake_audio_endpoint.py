from __future__ import annotations

from dataclasses import dataclass, field

from .audio_endpoint import RecordingResult
from .recording import recording_from_audio
from .speech_detection import VoiceActivityDetector
from .wave_data import WaveData


@dataclass
class FakeAudioEndpoint:
    """Deterministic audio endpoint for tests."""

    response_audio: WaveData
    played_audio: list[WaveData] = field(default_factory=list)

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "fake-audio"

    async def play(self, audio: WaveData) -> None:
        """Record played audio for assertions."""
        self.played_audio.append(audio)

    async def record_until(
        self,
        detector: VoiceActivityDetector,
        *,
        frame_duration_ms: int,
        end_silence_ms: int,
        min_speech_ms: int,
        max_recording_ms: int,
    ) -> RecordingResult:
        """Segment deterministic response audio."""
        return recording_from_audio(
            self.response_audio,
            detector,
            frame_duration_ms=frame_duration_ms,
            end_silence_ms=end_silence_ms,
            min_speech_ms=min_speech_ms,
            max_recording_ms=max_recording_ms,
        )
