import pytest

from inspect_voice_eval.audio_endpoint import VoiceTurnTimeoutError
from inspect_voice_eval.energy_speech_detection import EnergyVoiceActivityDetector
from inspect_voice_eval.fake_audio_endpoint import FakeAudioEndpoint
from inspect_voice_eval.wave_data import WaveData


@pytest.mark.anyio
async def test_fake_audio_endpoint_records_played_audio() -> None:
    endpoint = FakeAudioEndpoint(
        WaveData.concatenate([WaveData.tone(120), WaveData.silence(240)])
    )
    prompt = WaveData.tone(60)

    await endpoint.play(prompt)

    assert endpoint.played_audio == [prompt]


@pytest.mark.anyio
async def test_fake_audio_endpoint_returns_segmented_response_audio() -> None:
    response = WaveData.concatenate(
        [WaveData.silence(90), WaveData.tone(300), WaveData.silence(300)]
    )
    endpoint = FakeAudioEndpoint(response)

    result = await endpoint.record_until(
        EnergyVoiceActivityDetector(),
        frame_duration_ms=30,
        end_silence_ms=240,
        min_speech_ms=120,
        max_recording_ms=2_000,
    )

    assert result.audio == response.slice(90, 390)
    assert result.speech_started_ms == 90
    assert result.speech_completed_ms == 390


@pytest.mark.anyio
async def test_fake_audio_endpoint_timeout_behavior() -> None:
    endpoint = FakeAudioEndpoint(WaveData.tone(600))

    with pytest.raises(VoiceTurnTimeoutError):
        await endpoint.record_until(
            EnergyVoiceActivityDetector(),
            frame_duration_ms=30,
            end_silence_ms=240,
            min_speech_ms=120,
            max_recording_ms=600,
        )
