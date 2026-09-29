import os

import pytest

from inspect_voice_eval.chatterbox_speech import ChatterboxSpeechSynthesizer
from inspect_voice_eval.faster_whisper_speech import FasterWhisperSpeechTranscriber
from inspect_voice_eval.sounddevice_endpoint import SounddeviceAudioEndpoint
from inspect_voice_eval.speech_synthesis import VoiceRequest
from inspect_voice_eval.wave_data import WaveData


@pytest.mark.anyio
@pytest.mark.chatterbox
@pytest.mark.skipif(
    os.environ.get("INSPECT_VOICE_EVAL_RUN_CHATTERBOX") != "1",
    reason="set INSPECT_VOICE_EVAL_RUN_CHATTERBOX=1 to run Chatterbox smoke test",
)
async def test_chatterbox_synthesizes_audio() -> None:
    synthesizer = ChatterboxSpeechSynthesizer()
    audio = await synthesizer.synthesize("Hello from Inspect.", VoiceRequest())

    assert audio.duration_ms > 0


@pytest.mark.anyio
@pytest.mark.faster_whisper
@pytest.mark.skipif(
    os.environ.get("INSPECT_VOICE_EVAL_RUN_FASTER_WHISPER") != "1",
    reason="set INSPECT_VOICE_EVAL_RUN_FASTER_WHISPER=1 to run faster-whisper smoke test",
)
async def test_faster_whisper_transcribes_audio() -> None:
    transcriber = FasterWhisperSpeechTranscriber(model_size="tiny")
    transcript = await transcriber.transcribe(WaveData.tone(300))

    assert isinstance(transcript, str)


@pytest.mark.anyio
@pytest.mark.sounddevice
@pytest.mark.skipif(
    os.environ.get("INSPECT_VOICE_EVAL_RUN_SOUNDDEVICE") != "1",
    reason="set INSPECT_VOICE_EVAL_RUN_SOUNDDEVICE=1 to run sounddevice smoke test",
)
async def test_sounddevice_endpoint_plays_audio() -> None:
    endpoint = SounddeviceAudioEndpoint()
    await endpoint.play(WaveData.silence(50))
