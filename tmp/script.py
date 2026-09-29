import anyio
from inspect_voice_eval import (
    EnergyVoiceActivityDetector,
    SounddeviceAudioEndpoint,
    WaveData,
)


async def main() -> None:
    endpoint = SounddeviceAudioEndpoint()
    detector = EnergyVoiceActivityDetector(energy_threshold=0.02)

    print("Playing prompt tone...")
    await endpoint.play(WaveData.tone(duration_ms=700))

    print("Recording. Speak after the tone.")
    response = await endpoint.record_until(
        detector,
        frame_duration_ms=30,
        end_silence_ms=300,
        min_speech_ms=150,
        max_recording_ms=5_000,
    )

    print("Detected speech.")
    print("speech_started_ms:", response.speech_started_ms)
    print("speech_completed_ms:", response.speech_completed_ms)
    print("response_duration_ms:", response.audio.duration_ms)


anyio.run(main)
