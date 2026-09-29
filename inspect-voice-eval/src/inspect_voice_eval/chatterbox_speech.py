from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Protocol, cast

import anyio

from .speech_synthesis import VoiceRequest, reject_unsupported_controls
from .wave_data import WaveData


class _ChatterboxModel(Protocol):
    sr: int

    def generate(self, text: str, **kwargs: object) -> object: ...


@dataclass
class ChatterboxSpeechSynthesizer:
    """Chatterbox-backed speech synthesizer."""

    device: str = "cpu"
    _model: _ChatterboxModel | None = field(init=False, default=None)

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "chatterbox"

    @property
    def supported_controls(self) -> set[str]:
        """Voice controls this backend can honor."""
        return {"speaker_reference_path", "expressiveness", "cfg_weight"}

    async def synthesize(self, text: str, voice: VoiceRequest) -> WaveData:
        """Create speech audio with Chatterbox."""
        reject_unsupported_controls(self.backend_name, self.supported_controls, voice)
        return await anyio.to_thread.run_sync(self._synthesize_sync, text, voice)

    def _synthesize_sync(self, text: str, voice: VoiceRequest) -> WaveData:
        import numpy as np

        model = self._load_model()
        kwargs: dict[str, object] = {}
        if voice.speaker_reference_path is not None:
            kwargs["audio_prompt_path"] = voice.speaker_reference_path
        if voice.expressiveness is not None:
            kwargs["exaggeration"] = voice.expressiveness
        if voice.cfg_weight is not None:
            kwargs["cfg_weight"] = voice.cfg_weight
        generated = model.generate(text, **kwargs)
        if hasattr(generated, "detach"):
            generated = generated.detach().cpu().numpy()
        samples = np.asarray(generated, dtype=np.float32).reshape(-1)
        samples = np.clip(samples, -1.0, 1.0)
        pcm = (samples * 32767).astype(np.int16).tobytes()
        return WaveData(pcm=pcm, sample_rate_hz=int(model.sr))

    def _load_model(self) -> _ChatterboxModel:
        if self._model is None:
            module = import_module("chatterbox.tts")
            model_cls = module.ChatterboxTTS
            self._model = cast(
                _ChatterboxModel, model_cls.from_pretrained(device=self.device)
            )
        return self._model
