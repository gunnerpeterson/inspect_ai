from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module
from tempfile import NamedTemporaryFile
from typing import Protocol, cast

import anyio

from .wave_data import WaveData


class _WhisperSegment(Protocol):
    text: str


class _WhisperModel(Protocol):
    def transcribe(self, audio: str) -> tuple[Iterable[_WhisperSegment], object]: ...


@dataclass
class FasterWhisperSpeechTranscriber:
    """faster-whisper-backed speech transcriber."""

    model_size: str = "base"
    device: str = "cpu"
    precision: str = "int8"
    _model: _WhisperModel | None = field(init=False, default=None)

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "faster-whisper"

    async def transcribe(self, audio: WaveData) -> str:
        """Transcribe audio with faster-whisper."""
        return await anyio.to_thread.run_sync(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: WaveData) -> str:
        model = self._load_model()
        audio_file = NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            audio_file.write(audio.to_wav_bytes())
            audio_file.close()
            segments, _ = model.transcribe(audio_file.name)
            texts = [segment.text.strip() for segment in segments]
            return " ".join(text for text in texts if text)
        finally:
            audio_file.close()
            os.unlink(audio_file.name)

    def _load_model(self) -> _WhisperModel:
        if self._model is None:
            module = import_module("faster_whisper")
            model_cls = module.WhisperModel
            self._model = cast(
                _WhisperModel,
                model_cls(
                    self.model_size,
                    device=self.device,
                    compute_type=self.precision,
                ),
            )
        return self._model
