from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, cast

import anyio
import numpy as np

from .speech_synthesis import VoiceRequest, reject_unsupported_controls
from .wave_data import WaveData


class _PiperVoiceDownloader(Protocol):
    def download_voice(
        self, voice: str, download_dir: Path, force_redownload: bool = False
    ) -> None: ...


@dataclass(frozen=True)
class PiperSpeechSynthesizer:
    """Piper-backed speech synthesizer."""

    voice_id: str = "en_US-lessac-medium"
    voice_cache_path: Path | None = None

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "piper"

    @property
    def supported_controls(self) -> set[str]:
        """Voice controls this backend can honor."""
        return {"voice_id", "speaking_rate"}

    async def synthesize(self, text: str, voice: VoiceRequest) -> WaveData:
        """Create speech audio with Piper."""
        reject_unsupported_controls(self.backend_name, self.supported_controls, voice)
        return await anyio.to_thread.run_sync(self._synthesize_sync, text, voice)

    def _synthesize_sync(self, text: str, voice: VoiceRequest) -> WaveData:
        selected_voice = voice.voice_id if voice.voice_id is not None else self.voice_id
        speaking_rate = voice.speaking_rate if voice.speaking_rate is not None else 1.0
        if speaking_rate <= 0:
            raise ValueError("speaking_rate must be greater than 0")
        voice_path = self._voice_model_path(selected_voice)
        wav_file = NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = Path(wav_file.name)
        wav_file.close()
        try:
            subprocess.run(
                [
                    self._piper_cli_path(),
                    "--model",
                    str(voice_path),
                    "--output-file",
                    str(wav_path),
                    "--length-scale",
                    str(1 / speaking_rate),
                ],
                input=text,
                text=True,
                check=True,
            )
            soundfile_package = import_module("soundfile")
            samples, sample_rate_hz = soundfile_package.read(wav_path, dtype="float32")
            return _wave_from_float_samples(samples, int(sample_rate_hz))
        finally:
            wav_path.unlink(missing_ok=True)

    def _voice_model_path(self, voice_id: str) -> Path:
        voice_path = self._voice_cache_path()
        voice_path.mkdir(parents=True, exist_ok=True)
        downloader = cast(
            _PiperVoiceDownloader,
            import_module("piper.download_voices"),
        )
        downloader.download_voice(voice_id, voice_path)
        return voice_path / f"{voice_id}.onnx"

    def _voice_cache_path(self) -> Path:
        if self.voice_cache_path is not None:
            return self.voice_cache_path
        return Path.home() / ".cache" / "inspect-voice-eval" / "piper-voices"

    def _piper_cli_path(self) -> str:
        venv_piper_path = Path(sys.executable).parent / "piper"
        if venv_piper_path.exists():
            return str(venv_piper_path)
        path_piper = shutil.which("piper")
        if path_piper is not None:
            return path_piper
        raise RuntimeError("missing piper CLI")


def _wave_from_float_samples(samples: object, sample_rate_hz: int) -> WaveData:
    sample_array = np.asarray(samples, dtype=np.float32).reshape(-1)
    sample_array = np.clip(sample_array, -1.0, 1.0)
    pcm = (sample_array * 32767).astype(np.int16).tobytes()
    return WaveData(pcm=pcm, sample_rate_hz=sample_rate_hz)
