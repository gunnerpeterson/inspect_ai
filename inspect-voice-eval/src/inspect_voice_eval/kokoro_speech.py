from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Protocol, cast

import anyio
import numpy as np

from .speech_synthesis import VoiceRequest, reject_unsupported_controls
from .wave_data import WaveData


class _KokoroSpeechPart(Protocol):
    audio: object


class _KokoroPipeline(Protocol):
    def __call__(
        self, text: str, *, voice: str, speed: float
    ) -> Iterable[tuple[object, ...] | _KokoroSpeechPart]: ...


class _KokoroPipelineFactory(Protocol):
    def __call__(
        self, *, lang_code: str, repo_id: str, device: str | None
    ) -> _KokoroPipeline: ...


class _KokoroPackage(Protocol):
    KPipeline: _KokoroPipelineFactory


@dataclass
class KokoroSpeechSynthesizer:
    """Kokoro-backed speech synthesizer."""

    voice_id: str = "af_heart"
    language_code: str = "a"
    repo_id: str = "hexgrad/Kokoro-82M"
    device: str | None = None
    _pipeline: _KokoroPipeline | None = field(init=False, default=None)

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "kokoro"

    @property
    def supported_controls(self) -> set[str]:
        """Voice controls this backend can honor."""
        return {"voice_id", "speaking_rate"}

    async def synthesize(self, text: str, voice: VoiceRequest) -> WaveData:
        """Create speech audio with Kokoro."""
        reject_unsupported_controls(self.backend_name, self.supported_controls, voice)
        return await anyio.to_thread.run_sync(self._synthesize_sync, text, voice)

    def _synthesize_sync(self, text: str, voice: VoiceRequest) -> WaveData:
        pipeline = self._load_pipeline()
        selected_voice = voice.voice_id if voice.voice_id is not None else self.voice_id
        speaking_rate = voice.speaking_rate if voice.speaking_rate is not None else 1.0
        if speaking_rate <= 0:
            raise ValueError("speaking_rate must be greater than 0")
        generator = pipeline(text, voice=selected_voice, speed=speaking_rate)
        for speech_part in generator:
            if isinstance(speech_part, tuple):
                samples = speech_part[-1]
            else:
                samples = speech_part.audio
            return _wave_from_float_samples(samples, 24_000)
        raise RuntimeError("kokoro produced no audio")

    def _load_pipeline(self) -> _KokoroPipeline:
        if self._pipeline is None:
            kokoro_package = cast(_KokoroPackage, import_module("kokoro"))
            self._pipeline = kokoro_package.KPipeline(
                lang_code=self.language_code,
                repo_id=self.repo_id,
                device=self.device,
            )
        return self._pipeline


def _wave_from_float_samples(samples: object, sample_rate_hz: int) -> WaveData:
    sample_array = np.asarray(samples, dtype=np.float32).reshape(-1)
    sample_array = np.clip(sample_array, -1.0, 1.0)
    pcm = (sample_array * 32767).astype(np.int16).tobytes()
    return WaveData(pcm=pcm, sample_rate_hz=sample_rate_hz)
