import pytest

from inspect_voice_eval import (
    ChatterboxSpeechSynthesizer,
    FakeSpeechSynthesizer,
    VoiceRequest,
)
from inspect_voice_eval.speech_synthesis import UnsupportedVoiceControlError


@pytest.mark.anyio
async def test_fake_synthesizer_rejects_unsupported_voice_controls() -> None:
    synthesizer = FakeSpeechSynthesizer()

    with pytest.raises(UnsupportedVoiceControlError, match="voice_id"):
        await synthesizer.synthesize("hello", VoiceRequest(voice_id="speaker-a"))


@pytest.mark.anyio
async def test_chatterbox_declares_reference_speaker_controls() -> None:
    synthesizer = ChatterboxSpeechSynthesizer()

    assert "speaker_reference_path" in synthesizer.supported_controls
    assert "expressiveness" in synthesizer.supported_controls
    assert "cfg_weight" in synthesizer.supported_controls
    assert "voice_id" not in synthesizer.supported_controls
