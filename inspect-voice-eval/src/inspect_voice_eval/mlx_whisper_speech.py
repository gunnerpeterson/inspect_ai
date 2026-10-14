from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from tempfile import NamedTemporaryFile
from typing import Protocol, cast

import anyio

from .wave_data import WaveData


class _MlxWhisperPackage(Protocol):
    def transcribe(self, audio: str, *, path_or_hf_repo: str) -> dict[str, str]: ...


@dataclass(frozen=True)
class MlxWhisperSpeechTranscriber:
    """mlx-whisper-backed speech transcriber."""

    model_repo: str = "mlx-community/whisper-medium"

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "mlx-whisper"

    async def transcribe(self, audio: WaveData) -> str:
        """Transcribe audio with mlx-whisper."""
        return await anyio.to_thread.run_sync(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: WaveData) -> str:
        mlx_whisper_package = cast(
            _MlxWhisperPackage,
            import_module("mlx_whisper"),
        )
        audio_file = NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            audio_file.write(audio.to_wav_bytes())
            audio_file.close()
            transcription = mlx_whisper_package.transcribe(
                audio_file.name,
                path_or_hf_repo=self.model_repo,
            )
            return transcription["text"].strip()
        finally:
            audio_file.close()
            os.unlink(audio_file.name)
