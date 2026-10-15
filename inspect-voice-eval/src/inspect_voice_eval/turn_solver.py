from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import anyio
from inspect_ai.log import transcript
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .audio_endpoint import AudioEndpoint, VoiceTurnTimeoutError
from .energy_speech_detection import EnergyVoiceActivityDetector
from .fake_audio_endpoint import FakeAudioEndpoint
from .fake_speech import FakeSpeechSynthesizer, FakeSpeechTranscriber
from .log_attachment_refs import attachment_ref_for_text
from .speech_detection import VoiceActivityDetector
from .speech_synthesis import SpeechSynthesizer, VoiceRequest
from .speech_transcription import SpeechTranscriber
from .turn_record import (
    VoiceTurnArtifacts,
    VoiceTurnAudioFacts,
    VoiceTurnBackends,
    VoiceTurnRecord,
    VoiceTurnTiming,
)
from .wave_data import WaveData

Backend = TypeVar("Backend")
WaitResult = TypeVar("WaitResult")

SpeechSynthesizerSource = SpeechSynthesizer | Callable[[], SpeechSynthesizer]
AudioEndpointSource = AudioEndpoint | Callable[[], AudioEndpoint]
SpeechTranscriberSource = SpeechTranscriber | Callable[[], SpeechTranscriber]
VoiceActivityDetectorSource = (
    VoiceActivityDetector | Callable[[], VoiceActivityDetector]
)


def _default_response_audio() -> WaveData:
    return WaveData.concatenate(
        [
            WaveData.silence(90),
            WaveData.tone(300),
            WaveData.silence(900),
        ]
    )


def _default_recorded_audio() -> WaveData:
    return WaveData.tone(300)


def _make_backend(backend: Backend | Callable[[], Backend]) -> Backend:
    if callable(backend):
        return backend()
    return backend


async def _bounded_wait(
    awaitable: Awaitable[WaitResult], timeout_ms: int
) -> WaitResult:
    try:
        with anyio.fail_after(timeout_ms / 1000):
            return await awaitable
    except TimeoutError as exc:
        raise VoiceTurnTimeoutError("Voice turn stage exceeded timeout") from exc


@dataclass(frozen=True)
class VoiceTurnTimingSettings:
    """Timing settings for one voice turn."""

    frame_duration_ms: int = 30
    end_silence_ms: int = 800
    min_speech_ms: int = 120
    max_recording_ms: int = 120_000
    synthesis_timeout_ms: int = 30_000
    playback_timeout_ms: int = 30_000
    transcription_timeout_ms: int = 30_000


@solver
def voice_turn(
    *,
    synthesizer: SpeechSynthesizerSource | None = None,
    audio_endpoint: AudioEndpointSource | None = None,
    transcriber: SpeechTranscriberSource | None = None,
    detector: VoiceActivityDetectorSource | None = None,
    voice: VoiceRequest | None = None,
    timing: VoiceTurnTimingSettings | None = None,
) -> Solver:
    """Single-turn black-box voice solver."""
    selected_synthesizer = _make_backend(synthesizer or FakeSpeechSynthesizer())
    selected_endpoint = _make_backend(
        audio_endpoint or FakeAudioEndpoint(_default_response_audio())
    )
    selected_transcriber = _make_backend(
        transcriber
        or FakeSpeechTranscriber.for_audio(_default_recorded_audio(), "fake response")
    )
    selected_detector = _make_backend(detector or EnergyVoiceActivityDetector())
    selected_voice = voice or VoiceRequest()
    selected_timing = timing or VoiceTurnTimingSettings()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        started = time.monotonic()

        def mark_ms() -> int:
            return round((time.monotonic() - started) * 1000)

        tts_started_ms = mark_ms()
        prompt_audio = await _bounded_wait(
            selected_synthesizer.synthesize(state.input_text, selected_voice),
            selected_timing.synthesis_timeout_ms,
        )
        tts_completed_ms = mark_ms()

        playback_started_ms = mark_ms()
        await _bounded_wait(
            selected_endpoint.play(prompt_audio), selected_timing.playback_timeout_ms
        )
        playback_completed_ms = mark_ms()

        recording_started_ms = mark_ms()
        response = await selected_endpoint.record_until(
            selected_detector,
            frame_duration_ms=selected_timing.frame_duration_ms,
            end_silence_ms=selected_timing.end_silence_ms,
            min_speech_ms=selected_timing.min_speech_ms,
            max_recording_ms=selected_timing.max_recording_ms,
        )
        recording_completed_ms = mark_ms()

        transcription_started_ms = mark_ms()
        transcript_text = await _bounded_wait(
            selected_transcriber.transcribe(response.audio),
            selected_timing.transcription_timeout_ms,
        )
        transcription_completed_ms = mark_ms()

        state.output = ModelOutput.from_content(
            model=state.model.name,
            content=transcript_text,
        )

        prompt_audio_uri = prompt_audio.to_wav_data_uri()
        response_audio_uri = response.audio.to_wav_data_uri()
        prompt_audio_ref = attachment_ref_for_text(prompt_audio_uri)
        response_audio_ref = attachment_ref_for_text(response_audio_uri)

        transcript().info(
            {
                "voice_turn_audio": {
                    "prompt_audio": prompt_audio_uri,
                    "response_audio": response_audio_uri,
                }
            },
            source="inspect_voice_eval",
        )

        record = VoiceTurnRecord(
            artifacts=VoiceTurnArtifacts(
                prompt_audio_ref=prompt_audio_ref,
                response_audio_ref=response_audio_ref,
            ),
            backends=VoiceTurnBackends(
                tts=selected_synthesizer.backend_name,
                audio=selected_endpoint.backend_name,
                vad=selected_detector.backend_name,
                transcriber=selected_transcriber.backend_name,
            ),
            audio=VoiceTurnAudioFacts(
                sample_rate_hz=response.audio.sample_rate_hz,
                prompt_duration_ms=prompt_audio.duration_ms,
                response_duration_ms=response.audio.duration_ms,
            ),
            timing=VoiceTurnTiming(
                tts_started_ms=tts_started_ms,
                tts_completed_ms=tts_completed_ms,
                playback_started_ms=playback_started_ms,
                playback_completed_ms=playback_completed_ms,
                recording_started_ms=recording_started_ms,
                speech_started_ms=recording_started_ms + response.speech_started_ms,
                speech_completed_ms=recording_started_ms + response.speech_completed_ms,
                recording_completed_ms=recording_completed_ms,
                transcription_started_ms=transcription_started_ms,
                transcription_completed_ms=transcription_completed_ms,
            ),
        )
        state.store.set("voice_turn", record.model_dump())
        return state

    return solve
