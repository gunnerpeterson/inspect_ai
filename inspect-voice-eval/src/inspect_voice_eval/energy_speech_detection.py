from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .speech_detection import SpeechDecision
from .wave_data import WaveData


@dataclass(frozen=True)
class EnergyVoiceActivityDetector:
    """Voice activity detector based on frame RMS energy."""

    energy_threshold: float = 0.02

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "energy-threshold"

    def detect(self, audio: WaveData) -> SpeechDecision:
        """Classify an audio frame as speech or silence."""
        if audio.sample_width_bytes != 2:
            raise ValueError("EnergyVoiceActivityDetector requires 16-bit PCM audio")
        if not audio.pcm:
            return SpeechDecision(state="silence", energy=0.0)
        total = 0.0
        count = 0
        for (sample,) in struct.iter_unpack("<h", audio.pcm):
            normalized = sample / 32768.0
            total += normalized * normalized
            count += 1
        energy = math.sqrt(total / count)
        return SpeechDecision(
            state="speech" if energy >= self.energy_threshold else "silence",
            energy=energy,
        )
