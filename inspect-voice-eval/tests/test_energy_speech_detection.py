import pytest

from inspect_voice_eval.audio_endpoint import VoiceTurnTimeoutError
from inspect_voice_eval.energy_speech_detection import EnergyVoiceActivityDetector
from inspect_voice_eval.recording import recording_from_audio
from inspect_voice_eval.wave_data import WaveData


def test_energy_speech_detection_classifies_tone_and_silence() -> None:
    detector = EnergyVoiceActivityDetector(energy_threshold=0.02)

    assert detector.detect(WaveData.tone(100)).state == "speech"
    assert detector.detect(WaveData.silence(100)).state == "silence"


def test_energy_speech_detection_closes_after_speech_followed_by_silence() -> None:
    detector = EnergyVoiceActivityDetector(energy_threshold=0.02)
    audio = WaveData.concatenate([WaveData.tone(300), WaveData.silence(300)])

    result = recording_from_audio(
        audio,
        detector,
        frame_duration_ms=30,
        end_silence_ms=240,
        min_speech_ms=120,
        max_recording_ms=1_000,
    )

    assert result.speech_started_ms == 0
    assert 270 <= result.speech_completed_ms <= 330


def test_energy_speech_detection_tolerates_short_mid_utterance_pause() -> None:
    detector = EnergyVoiceActivityDetector(energy_threshold=0.02)
    audio = WaveData.concatenate(
        [
            WaveData.tone(210),
            WaveData.silence(90),
            WaveData.tone(210),
            WaveData.silence(300),
        ]
    )

    result = recording_from_audio(
        audio,
        detector,
        frame_duration_ms=30,
        end_silence_ms=240,
        min_speech_ms=120,
        max_recording_ms=1_000,
    )

    assert result.speech_started_ms == 0
    assert result.speech_completed_ms >= 480


def test_energy_speech_detection_times_out_when_silence_never_arrives() -> None:
    detector = EnergyVoiceActivityDetector(energy_threshold=0.02)
    audio = WaveData.tone(600)

    with pytest.raises(VoiceTurnTimeoutError):
        recording_from_audio(
            audio,
            detector,
            frame_duration_ms=30,
            end_silence_ms=240,
            min_speech_ms=120,
            max_recording_ms=600,
        )
