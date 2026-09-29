from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceTurnArtifacts(BaseModel):
    """Attachment references for voice turn audio."""

    prompt_audio_ref: str
    response_audio_ref: str


class VoiceTurnBackends(BaseModel):
    """Backend names used for a voice turn."""

    tts: str
    audio: str
    vad: str
    transcriber: str


class VoiceTurnAudioFacts(BaseModel):
    """Scalar audio facts for a voice turn."""

    sample_rate_hz: int
    prompt_duration_ms: int
    response_duration_ms: int


class VoiceTurnTiming(BaseModel):
    """Monotonic millisecond timing marks relative to solver start."""

    solver_started_ms: int = Field(default=0)
    tts_started_ms: int
    tts_completed_ms: int
    playback_started_ms: int
    playback_completed_ms: int
    recording_started_ms: int
    speech_started_ms: int
    speech_completed_ms: int
    recording_completed_ms: int
    transcription_started_ms: int
    transcription_completed_ms: int


class VoiceTurnRecord(BaseModel):
    """Bounded sample-store record for a voice turn."""

    schema_version: int = Field(default=1)
    artifacts: VoiceTurnArtifacts
    backends: VoiceTurnBackends
    audio: VoiceTurnAudioFacts
    timing: VoiceTurnTiming
