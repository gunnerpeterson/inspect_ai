from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import anyio

from .audio_endpoint import RecordingResult
from .recording import recording_from_audio
from .speech_detection import VoiceActivityDetector
from .wave_data import WaveData


class SounddeviceAudioEndpoint:
    """Audio endpoint backed by sounddevice and PortAudio."""

    def __init__(
        self,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        sample_rate_hz: int = 16_000,
        channel_count: int = 1,
    ) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate_hz = sample_rate_hz
        self.channel_count = channel_count

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "sounddevice"

    async def play(self, audio: WaveData) -> None:
        """Play audio to the selected output device."""
        await anyio.to_thread.run_sync(self._play_sync, audio)

    async def record_until(
        self,
        detector: VoiceActivityDetector,
        *,
        frame_duration_ms: int,
        end_silence_ms: int,
        min_speech_ms: int,
        max_recording_ms: int,
    ) -> RecordingResult:
        """Record from the selected input device until VAD stops."""
        audio = await anyio.to_thread.run_sync(
            self._record_sync,
            max_recording_ms,
        )
        return recording_from_audio(
            audio,
            detector,
            frame_duration_ms=frame_duration_ms,
            end_silence_ms=end_silence_ms,
            min_speech_ms=min_speech_ms,
            max_recording_ms=max_recording_ms,
        )

    def _play_sync(self, audio: WaveData) -> None:
        import numpy as np

        sd = cast(Any, import_module("sounddevice"))
        samples = np.frombuffer(audio.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.channel_count > 1:
            samples = samples.reshape((-1, audio.channel_count))
        sd.play(
            samples,
            samplerate=audio.sample_rate_hz,
            device=self.output_device,
            blocking=True,
        )

    def _record_sync(self, max_recording_ms: int) -> WaveData:
        import numpy as np

        sd = cast(Any, import_module("sounddevice"))
        frame_count = round(self.sample_rate_hz * max_recording_ms / 1000)
        samples = sd.rec(
            frame_count,
            samplerate=self.sample_rate_hz,
            channels=self.channel_count,
            dtype="int16",
            device=self.input_device,
            blocking=True,
        )
        return WaveData(
            pcm=np.asarray(samples, dtype=np.int16).tobytes(),
            sample_rate_hz=self.sample_rate_hz,
            channel_count=self.channel_count,
        )
