from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import inspect_voice_eval.kokoro_speech as kokoro_speech
import inspect_voice_eval.piper_speech as piper_speech
from inspect_voice_eval import (
    KokoroSpeechSynthesizer,
    PiperSpeechSynthesizer,
    VoiceRequest,
)
from inspect_voice_eval.speech_synthesis import UnsupportedVoiceControlError
from inspect_voice_eval.wave_data import WaveData


def test_kokoro_backend_name_and_controls() -> None:
    synthesizer = KokoroSpeechSynthesizer(voice_id="af_heart")

    assert synthesizer.backend_name == "kokoro"
    assert synthesizer.supported_controls == {"voice_id", "speaking_rate"}


def test_piper_backend_name_and_controls(tmp_path: Path) -> None:
    synthesizer = PiperSpeechSynthesizer(
        voice_id="en_US-lessac-medium", voice_cache_path=tmp_path
    )

    assert synthesizer.backend_name == "piper"
    assert synthesizer.supported_controls == {"voice_id", "speaking_rate"}


def test_fast_synthesizer_constructors_do_not_import_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def fail_import(package_name: str) -> object:
        imports.append(package_name)
        raise ModuleNotFoundError(package_name)

    monkeypatch.setattr(kokoro_speech, "import_module", fail_import)
    monkeypatch.setattr(piper_speech, "import_module", fail_import)

    kokoro = KokoroSpeechSynthesizer()
    piper = PiperSpeechSynthesizer()

    assert kokoro.backend_name == "kokoro"
    assert piper.backend_name == "piper"
    assert imports == []


@pytest.mark.anyio
async def test_kokoro_rejects_unsupported_voice_controls() -> None:
    synthesizer = KokoroSpeechSynthesizer()

    with pytest.raises(UnsupportedVoiceControlError, match="expressiveness"):
        await synthesizer.synthesize("hello", VoiceRequest(expressiveness=0.5))


@pytest.mark.anyio
async def test_piper_rejects_unsupported_voice_controls(tmp_path: Path) -> None:
    synthesizer = PiperSpeechSynthesizer(voice_cache_path=tmp_path)

    with pytest.raises(UnsupportedVoiceControlError, match="speaker_reference_path"):
        await synthesizer.synthesize(
            "hello", VoiceRequest(speaker_reference_path="speaker.wav")
        )


@pytest.mark.anyio
async def test_kokoro_transcribes_float_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_calls: list[tuple[str, str, float]] = []
    pipeline_inits: list[tuple[str, str, str]] = []

    class FakeKPipeline:
        def __init__(
            self, lang_code: str, repo_id: str, device: str | None = None
        ) -> None:
            pipeline_inits.append((lang_code, repo_id, device or "none"))

        def __call__(self, text: str, *, voice: str, speed: float) -> object:
            pipeline_calls.append((text, voice, speed))
            return [(text, "phonemes", np.array([0.0, 0.5, -0.5], dtype=np.float32))]

    kokoro_package = SimpleNamespace(KPipeline=FakeKPipeline)

    def import_backend(package_name: str) -> object:
        assert package_name == "kokoro"
        return kokoro_package

    monkeypatch.setattr(kokoro_speech, "import_module", import_backend)
    synthesizer = KokoroSpeechSynthesizer(
        voice_id="af_heart", language_code="a", repo_id="local/kokoro", device="cpu"
    )

    audio = await synthesizer.synthesize(
        "hello", VoiceRequest(voice_id="am_adam", speaking_rate=1.25)
    )

    assert audio.sample_rate_hz == 24_000
    assert audio.frame_count == 3
    assert pipeline_inits == [("a", "local/kokoro", "cpu")]
    assert pipeline_calls == [("hello", "am_adam", 1.25)]


@pytest.mark.anyio
async def test_piper_downloads_voice_and_invokes_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloaded_voices: list[tuple[str, Path]] = []
    subprocess_calls: list[list[str]] = []

    class PiperDownloader:
        def download_voice(
            self, voice: str, download_dir: Path, force_redownload: bool = False
        ) -> None:
            downloaded_voices.append((voice, download_dir))
            (download_dir / f"{voice}.onnx").write_bytes(b"model")

    def read_audio(wav_path: Path, *, dtype: str) -> tuple[np.ndarray, int]:
        assert dtype == "float32"
        assert wav_path.exists()
        return np.array([0.0, 0.25, -0.25], dtype=np.float32), 22_050

    soundfile_package = SimpleNamespace(read=read_audio)

    def import_backend(package_name: str) -> object:
        if package_name == "piper.download_voices":
            return PiperDownloader()
        if package_name == "soundfile":
            return soundfile_package
        raise AssertionError(package_name)

    def run_piper(args: list[str], *, input: str, text: bool, check: bool) -> object:
        subprocess_calls.append(args)
        assert input == "hello"
        assert text is True
        assert check is True
        output_path = Path(args[args.index("--output-file") + 1])
        output_path.write_bytes(WaveData.tone(20).to_wav_bytes())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(piper_speech, "import_module", import_backend)
    monkeypatch.setattr("inspect_voice_eval.piper_speech.subprocess.run", run_piper)
    monkeypatch.setattr(
        piper_speech.PiperSpeechSynthesizer, "_piper_cli_path", lambda _: "piper"
    )
    synthesizer = PiperSpeechSynthesizer(
        voice_id="en_US-lessac-medium", voice_cache_path=tmp_path
    )

    audio = await synthesizer.synthesize(
        "hello", VoiceRequest(voice_id="en_US-ryan-medium", speaking_rate=2.0)
    )

    assert audio.sample_rate_hz == 22_050
    assert audio.frame_count == 3
    assert downloaded_voices == [("en_US-ryan-medium", tmp_path)]
    assert subprocess_calls == [
        [
            "piper",
            "--model",
            str(tmp_path / "en_US-ryan-medium.onnx"),
            "--output-file",
            subprocess_calls[0][4],
            "--length-scale",
            "0.5",
        ]
    ]
