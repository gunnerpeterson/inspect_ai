from __future__ import annotations

from .audio_endpoint import RecordingResult, VoiceTurnTimeoutError
from .speech_detection import VoiceActivityDetector
from .wave_data import WaveData


def recording_from_audio(
    audio: WaveData,
    detector: VoiceActivityDetector,
    *,
    frame_duration_ms: int,
    end_silence_ms: int,
    min_speech_ms: int,
    max_recording_ms: int,
) -> RecordingResult:
    """Segment recorded audio with a VAD stopping rule."""
    speech_start: int | None = None
    last_speech_end: int | None = None
    silence_after_speech_ms = 0

    for frame_start_ms, frame in audio.frames(frame_duration_ms):
        frame_end_ms = frame_start_ms + frame.duration_ms
        if frame_end_ms > max_recording_ms:
            break
        decision = detector.detect(frame)
        if decision.state == "speech":
            if speech_start is None:
                speech_start = frame_start_ms
            last_speech_end = frame_end_ms
            silence_after_speech_ms = 0
        elif speech_start is not None and last_speech_end is not None:
            silence_after_speech_ms += frame.duration_ms
            speech_duration_ms = last_speech_end - speech_start
            if (
                speech_duration_ms >= min_speech_ms
                and silence_after_speech_ms >= end_silence_ms
            ):
                return RecordingResult(
                    audio=audio.slice(speech_start, last_speech_end),
                    speech_started_ms=speech_start,
                    speech_completed_ms=last_speech_end,
                )

    if speech_start is not None and last_speech_end is not None:
        raise VoiceTurnTimeoutError("Speech did not end before recording timeout")
    raise VoiceTurnTimeoutError("No speech detected before recording timeout")
