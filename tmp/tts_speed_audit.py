from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from inspect_voice_eval import ChatterboxSpeechSynthesizer, VoiceRequest, WaveData

PROMPT_TEXT = "Hello. Please introduce yourself in one short sentence."
ARTIFACT_DIR = Path("tmp/tts_speed_audit")


@dataclass(frozen=True)
class SuccessfulSpeechRun:
    backend: str
    status: str
    latency_seconds: float
    audio_seconds: float
    latency_per_audio_second: float
    mp3_path: str


@dataclass(frozen=True)
class FailedSpeechRun:
    backend: str
    status: str
    reason: str


SpeechRun = SuccessfulSpeechRun | FailedSpeechRun
SpeechProbe = Callable[[], WaveData]


def write_wav(audio: WaveData, wav_path: Path) -> None:
    wav_path.write_bytes(audio.to_wav_bytes())


def mp3_from_wav(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to write MP3 samples")
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            str(mp3_path),
        ],
        check=True,
    )


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def audio_from_float_samples(samples: object, sample_rate_hz: int) -> WaveData:
    sample_array = np.asarray(samples, dtype=np.float32).reshape(-1)
    sample_array = np.clip(sample_array, -1.0, 1.0)
    pcm = (sample_array * 32767).astype(np.int16).tobytes()
    return WaveData(pcm=pcm, sample_rate_hz=sample_rate_hz)


def require_import(package_name: str) -> object:
    try:
        return import_module(package_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"missing package: {package_name}") from exc


def chatterbox_probe() -> WaveData:
    import anyio

    synthesizer = ChatterboxSpeechSynthesizer(device="cpu")
    return anyio.run(synthesizer.synthesize, PROMPT_TEXT, VoiceRequest())


def chatterbox_turbo_probe() -> WaveData:
    module = require_import("chatterbox.tts_turbo")
    model_cls = module.ChatterboxTurboTTS
    model = model_cls.from_pretrained(device="cpu")
    generated = model.generate(PROMPT_TEXT)
    if hasattr(generated, "detach"):
        generated = generated.detach().cpu().numpy()
    return audio_from_float_samples(generated, int(model.sr))


def kokoro_probe() -> WaveData:
    module = require_import("kokoro")
    pipeline_cls = module.KPipeline
    pipeline = pipeline_cls(lang_code="a")
    generator = pipeline(PROMPT_TEXT, voice="af_heart")
    for speech_part in generator:
        if isinstance(speech_part, tuple):
            audio = speech_part[-1]
        else:
            audio = speech_part.audio
        return audio_from_float_samples(audio, 24_000)
    raise RuntimeError("kokoro produced no audio")


def piper_probe() -> WaveData:
    voice_name = "en_US-lessac-medium"
    voice_dir = ARTIFACT_DIR / "piper_voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    module = require_import("piper.download_voices")
    module.download_voice(voice_name, voice_dir)
    voice_path = voice_dir / "en_US-lessac-medium.onnx"
    piper_path = shutil.which("piper") or str(Path(sys.executable).parent / "piper")
    if not Path(piper_path).exists():
        raise RuntimeError("missing piper CLI")
    wav_path = ARTIFACT_DIR / "piper.wav"
    subprocess.run(
        [
            piper_path,
            "--model",
            str(voice_path),
            "--output-file",
            str(wav_path),
        ],
        input=PROMPT_TEXT,
        text=True,
        check=True,
    )
    samples, sample_rate_hz = sf.read(wav_path, dtype="float32")
    return audio_from_float_samples(samples, int(sample_rate_hz))


def melotts_probe() -> WaveData:
    module = require_import("melo.api")
    model_cls = module.TTS
    model = model_cls(language="EN", device="cpu")
    speaker_ids = model.hps.data.spk2id
    if "EN-US" in speaker_ids:
        speaker_id = speaker_ids["EN-US"]
    else:
        speaker_id = next(iter(speaker_ids.values()))
    wav_path = ARTIFACT_DIR / "melotts.wav"
    model.tts_to_file(PROMPT_TEXT, speaker_id, str(wav_path), speed=1.0)
    samples, sample_rate_hz = sf.read(wav_path, dtype="float32")
    return audio_from_float_samples(samples, int(sample_rate_hz))


def f5_tts_probe() -> WaveData:
    module = require_import("f5_tts.api")
    reference_path = ARTIFACT_DIR / "piper.wav"
    if not reference_path.exists():
        raise RuntimeError("f5-tts requires a reference WAV; run piper first")
    model_cls = module.F5TTS
    model = model_cls(device="cpu")
    samples, sample_rate_hz, _spectrogram = model.infer(
        ref_file=str(reference_path),
        ref_text=PROMPT_TEXT,
        gen_text=PROMPT_TEXT,
        show_info=lambda _message: None,
    )
    return audio_from_float_samples(samples, int(sample_rate_hz))


def fish_speech_probe() -> WaveData:
    require_import("fish_speech")
    raise RuntimeError(
        "fish-speech package is present, but this audit has no local checkpoint path"
    )


def cosyvoice_probe() -> WaveData:
    require_import("cosyvoice")
    raise RuntimeError(
        "cosyvoice package is present, but this audit has no local checkpoint path"
    )


def run_probe(backend: str, probe: SpeechProbe) -> SpeechRun:
    wav_path = ARTIFACT_DIR / f"{backend}.wav"
    mp3_path = ARTIFACT_DIR / f"{backend}.mp3"
    started = time.perf_counter()
    try:
        audio = probe()
        latency_seconds = time.perf_counter() - started
        write_wav(audio, wav_path)
        mp3_from_wav(wav_path, mp3_path)
        audio_seconds = wav_duration_seconds(wav_path)
        return SuccessfulSpeechRun(
            backend=backend,
            status="success",
            latency_seconds=latency_seconds,
            audio_seconds=audio_seconds,
            latency_per_audio_second=latency_seconds / audio_seconds,
            mp3_path=str(mp3_path),
        )
    except Exception as exc:
        return FailedSpeechRun(
            backend=backend,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
        )


def markdown_table(runs: list[SpeechRun]) -> str:
    successful_runs = [run for run in runs if isinstance(run, SuccessfulSpeechRun)]
    failed_runs = [run for run in runs if isinstance(run, FailedSpeechRun)]
    successful_runs.sort(key=lambda run: run.latency_seconds)
    lines = [
        "# TTS Speed Audit Results",
        "",
        f"Prompt: `{PROMPT_TEXT}`",
        "",
        "## Successful local backends",
        "",
        "| Backend | Latency seconds | Audio seconds | Latency per audio second | MP3 |",
        "|---|---:|---:|---:|---|",
    ]
    for run in successful_runs:
        lines.append(
            f"| {run.backend} | {run.latency_seconds:.3f} | {run.audio_seconds:.3f} | {run.latency_per_audio_second:.3f} | `{run.mp3_path}` |"
        )
    lines.extend(
        [
            "",
            "## Unavailable local backends",
            "",
            "| Backend | Reason |",
            "|---|---|",
        ]
    )
    for run in failed_runs:
        lines.append(f"| {run.backend} | {run.reason} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    probes: list[tuple[str, SpeechProbe]] = [
        ("chatterbox", chatterbox_probe),
        ("chatterbox_turbo", chatterbox_turbo_probe),
        ("kokoro", kokoro_probe),
        ("piper", piper_probe),
        ("melotts", melotts_probe),
        ("f5_tts", f5_tts_probe),
        ("fish_speech", fish_speech_probe),
        ("cosyvoice", cosyvoice_probe),
    ]
    runs = []
    for backend, probe in probes:
        print(f"running {backend}", flush=True)
        run = run_probe(backend, probe)
        runs.append(run)
        print(asdict(run), flush=True)

    result_path = ARTIFACT_DIR / "results.json"
    summary_path = ARTIFACT_DIR / "summary.md"
    summary = markdown_table(runs)
    result_path.write_text(json.dumps([asdict(run) for run in runs], indent=2))
    summary_path.write_text(summary)
    print(summary)
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
