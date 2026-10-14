from __future__ import annotations

from types import SimpleNamespace

import pytest

import inspect_voice_eval.mlx_whisper_speech as mlx_whisper_speech
from inspect_voice_eval import MlxWhisperSpeechTranscriber
from inspect_voice_eval.wave_data import WaveData


def test_mlx_whisper_backend_name() -> None:
    transcriber = MlxWhisperSpeechTranscriber(model_repo="mlx-community/whisper-medium")

    assert transcriber.backend_name == "mlx-whisper"
    assert transcriber.model_repo == "mlx-community/whisper-medium"


def test_mlx_whisper_constructor_does_not_import_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def fail_import(package_name: str) -> object:
        imports.append(package_name)
        raise ModuleNotFoundError(package_name)

    monkeypatch.setattr(mlx_whisper_speech, "import_module", fail_import)

    transcriber = MlxWhisperSpeechTranscriber()

    assert transcriber.backend_name == "mlx-whisper"
    assert imports == []


@pytest.mark.anyio
async def test_mlx_whisper_transcribes_with_wav_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_paths: list[str] = []
    observed_repos: list[str] = []
    observed_wav_bytes: list[bytes] = []

    def transcribe(audio_path: str, *, path_or_hf_repo: str) -> dict[str, str]:
        observed_paths.append(audio_path)
        observed_repos.append(path_or_hf_repo)
        with open(audio_path, "rb") as audio_file:
            observed_wav_bytes.append(audio_file.read())
        return {"text": " hello from mlx "}

    mlx_whisper_package = SimpleNamespace(transcribe=transcribe)

    def import_backend(package_name: str) -> object:
        assert package_name == "mlx_whisper"
        return mlx_whisper_package

    monkeypatch.setattr(mlx_whisper_speech, "import_module", import_backend)
    transcriber = MlxWhisperSpeechTranscriber(model_repo="local/medium")

    transcript = await transcriber.transcribe(WaveData.tone(50))

    assert transcript == "hello from mlx"
    assert observed_repos == ["local/medium"]
    assert len(observed_paths) == 1
    assert observed_wav_bytes[0].startswith(b"RIFF")


@pytest.mark.anyio
async def test_mlx_whisper_reports_missing_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(package_name: str) -> object:
        raise ModuleNotFoundError(package_name)

    monkeypatch.setattr(mlx_whisper_speech, "import_module", fail_import)
    transcriber = MlxWhisperSpeechTranscriber()

    with pytest.raises(ModuleNotFoundError, match="mlx_whisper"):
        await transcriber.transcribe(WaveData.silence(10))
