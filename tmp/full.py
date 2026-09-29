from inspect_voice_eval import (
    ChatterboxSpeechSynthesizer,
    FasterWhisperSpeechTranscriber,
    SounddeviceAudioEndpoint,
    VoiceTurnTimingSettings,
    voice_turn,
)

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample

task = Task(
    dataset=[
        Sample(
            input="Please say: hello from the voice evaluation test.",
            target="hello from the voice evaluation test",
        )
    ],
    solver=voice_turn(
        synthesizer=ChatterboxSpeechSynthesizer(),
        audio_endpoint=SounddeviceAudioEndpoint(),
        transcriber=FasterWhisperSpeechTranscriber(model_size="tiny"),
        timing=VoiceTurnTimingSettings(max_recording_ms=8_000),
    ),
)

log = eval(
    task,
    model="mockllm/model",
    display="none",
)[0]

sample = log.samples[0]

print("status:", log.status)
print("transcript:", sample.output.completion)

if sample.error is not None:
    print("sample error message:", sample.error.message)
    print("sample error traceback:", sample.error.traceback)

print("store keys:", list(sample.store.keys()))
print("attachments:", list(sample.attachments.keys()))
