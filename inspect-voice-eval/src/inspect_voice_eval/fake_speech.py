from __future__ import annotations

from dataclasses import dataclass, field

from .speech_synthesis import VoiceRequest, reject_unsupported_controls
from .wave_data import WaveData


@dataclass(frozen=True)
class FakeSpeechSynthesizer:
    """Deterministic speech synthesizer for tests."""

    audio: WaveData = WaveData.tone(duration_ms=200)

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "fake-tts"

    @property
    def supported_controls(self) -> set[str]:
        """Voice controls this backend can honor."""
        return set()

    async def synthesize(self, text: str, voice: VoiceRequest) -> WaveData:
        """Return deterministic audio."""
        reject_unsupported_controls(self.backend_name, self.supported_controls, voice)
        return self.audio


@dataclass(frozen=True)
class FakeSpeechTranscriber:
    """Deterministic speech transcriber for tests."""

    transcripts_by_audio: dict[str, str] = field(default_factory=dict)
    fail: bool = False

    @classmethod
    def for_audio(cls, audio: WaveData, transcript: str) -> "FakeSpeechTranscriber":
        """Create a fake transcriber for a known audio response."""
        return cls(transcripts_by_audio={audio.to_wav_data_uri(): transcript})

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "fake-transcriber"

    async def transcribe(self, audio: WaveData) -> str:
        """Return transcript for known audio."""
        if self.fail:
            raise RuntimeError("fake transcription failure")
        audio_key = audio.to_wav_data_uri()
        return self.transcripts_by_audio[audio_key]
