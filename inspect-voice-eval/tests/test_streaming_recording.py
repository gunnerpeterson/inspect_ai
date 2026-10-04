import pytest

from inspect_voice_eval.audio_endpoint import VoiceTurnTimeoutError
from inspect_voice_eval.energy_speech_detection import EnergyVoiceActivityDetector
from inspect_voice_eval.recording import (
    SpeechRecorded,
    SpeechStopState,
    recording_from_audio,
)
from inspect_voice_eval.wave_data import WaveData


def test_streaming_recording_stops_after_speech_and_silence() -> None:
    detector = EnergyVoiceActivityDetector(energy_threshold=0.02)
    response_audio = WaveData.concatenate(
        [
            WaveData.silence(90),
            WaveData.tone(300),
            WaveData.silence(300),
            WaveData.tone(300),
        ]
    )
    stop_state = SpeechStopState(
        detector=detector,
        end_silence_ms=240,
        min_speech_ms=120,
    )

    for frame_start_ms, frame in response_audio.frames(30):
        consumed_duration_ms = frame_start_ms + frame.duration_ms
        progress = stop_state.accept(frame_start_ms, frame)
        if isinstance(progress, SpeechRecorded):
            recording = progress.recording
            break
    else:
        raise AssertionError("speech was not recorded")

    assert recording.audio == response_audio.slice(90, 390)
    assert recording.speech_started_ms == 90
    assert recording.speech_completed_ms == 390
    assert consumed_duration_ms == 630
    assert consumed_duration_ms < response_audio.duration_ms


def test_recording_timeout_respects_frame_boundary() -> None:
    detector = EnergyVoiceActivityDetector(energy_threshold=0.02)
    response_audio = WaveData.concatenate([WaveData.tone(300), WaveData.silence(300)])

    with pytest.raises(VoiceTurnTimeoutError, match="Speech did not end"):
        recording_from_audio(
            response_audio,
            detector,
            frame_duration_ms=30,
            end_silence_ms=240,
            min_speech_ms=120,
            max_recording_ms=510,
        )
