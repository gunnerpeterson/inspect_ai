from __future__ import annotations

import base64
import io
import math
import struct
import wave
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class WaveData:
    """PCM wave audio held as 16-bit little-endian samples."""

    pcm: bytes
    sample_rate_hz: int
    channel_count: int = 1
    sample_width_bytes: int = 2

    @property
    def frame_count(self) -> int:
        """Number of frames in the audio."""
        bytes_per_frame = self.channel_count * self.sample_width_bytes
        return len(self.pcm) // bytes_per_frame

    @property
    def duration_ms(self) -> int:
        """Audio duration in milliseconds."""
        return round((self.frame_count / self.sample_rate_hz) * 1000)

    def to_wav_bytes(self) -> bytes:
        """Serialize audio to a WAV byte string."""
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wave_file:
            wave_file.setnchannels(self.channel_count)
            wave_file.setsampwidth(self.sample_width_bytes)
            wave_file.setframerate(self.sample_rate_hz)
            wave_file.writeframes(self.pcm)
        return stream.getvalue()

    def to_wav_data_uri(self) -> str:
        """Serialize audio as a WAV data URI."""
        encoded = base64.b64encode(self.to_wav_bytes()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"

    def frames(self, frame_duration_ms: int) -> Iterator[tuple[int, "WaveData"]]:
        """Yield frame start time and audio frames."""
        frames_per_chunk = round(self.sample_rate_hz * frame_duration_ms / 1000)
        bytes_per_frame = self.channel_count * self.sample_width_bytes
        chunk_size = frames_per_chunk * bytes_per_frame
        for start in range(0, len(self.pcm), chunk_size):
            yield (
                round(start / bytes_per_frame / self.sample_rate_hz * 1000),
                WaveData(
                    pcm=self.pcm[start : start + chunk_size],
                    sample_rate_hz=self.sample_rate_hz,
                    channel_count=self.channel_count,
                    sample_width_bytes=self.sample_width_bytes,
                ),
            )

    def slice(self, start_ms: int, end_ms: int) -> "WaveData":
        """Return audio between start and end milliseconds."""
        bytes_per_frame = self.channel_count * self.sample_width_bytes
        start_frame = round(self.sample_rate_hz * start_ms / 1000)
        end_frame = round(self.sample_rate_hz * end_ms / 1000)
        return WaveData(
            pcm=self.pcm[start_frame * bytes_per_frame : end_frame * bytes_per_frame],
            sample_rate_hz=self.sample_rate_hz,
            channel_count=self.channel_count,
            sample_width_bytes=self.sample_width_bytes,
        )

    @classmethod
    def silence(
        cls, duration_ms: int, sample_rate_hz: int = 16_000, channel_count: int = 1
    ) -> "WaveData":
        """Create silent audio."""
        frame_count = round(sample_rate_hz * duration_ms / 1000)
        return cls(
            pcm=b"\x00\x00" * frame_count * channel_count,
            sample_rate_hz=sample_rate_hz,
            channel_count=channel_count,
        )

    @classmethod
    def tone(
        cls,
        duration_ms: int,
        sample_rate_hz: int = 16_000,
        frequency_hz: float = 440.0,
        amplitude: float = 0.25,
    ) -> "WaveData":
        """Create a deterministic sine tone."""
        frame_count = round(sample_rate_hz * duration_ms / 1000)
        samples = bytearray()
        peak = int(32767 * amplitude)
        for idx in range(frame_count):
            value = int(
                peak * math.sin(2 * math.pi * frequency_hz * idx / sample_rate_hz)
            )
            samples.extend(struct.pack("<h", value))
        return cls(pcm=bytes(samples), sample_rate_hz=sample_rate_hz)

    @classmethod
    def concatenate(cls, waves: list["WaveData"]) -> "WaveData":
        """Concatenate audio with matching format."""
        if not waves:
            return cls.silence(0)
        first = waves[0]
        for item in waves:
            if (
                item.sample_rate_hz != first.sample_rate_hz
                or item.channel_count != first.channel_count
                or item.sample_width_bytes != first.sample_width_bytes
            ):
                raise ValueError("All wave data must have matching audio format")
        return cls(
            pcm=b"".join(item.pcm for item in waves),
            sample_rate_hz=first.sample_rate_hz,
            channel_count=first.channel_count,
            sample_width_bytes=first.sample_width_bytes,
        )
