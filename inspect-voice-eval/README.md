# Inspect Voice Eval

Inspect extension for single-turn black-box voice evaluation.

The `voice_turn` solver speaks a text prompt to a target application, records the spoken response, transcribes it, and stores the transcript as the Inspect model output so standard scorers such as `match()` work unchanged.

## Install

Base install keeps heavy dependencies out:

```bash
pip install inspect-voice-eval
```

Install only the real backends you need:

```bash
pip install inspect-voice-eval[sounddevice]
pip install inspect-voice-eval[chatterbox]
pip install inspect-voice-eval[faster-whisper]
```

## Minimal fake-backend task

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_voice_eval import voice_turn


@task
def voice_smoke():
    return Task(
        dataset=[Sample(input="Say hello", target="fake response")],
        solver=voice_turn(),
        scorer=match(location="exact"),
    )
```

The default solver dependencies are fake and deterministic so CI does not need audio devices or model downloads.

## Real audio routing

`sounddevice` handles Python audio input/output. It does not route audio between applications by itself. Use OS virtual audio routes:

```text
Python prompt audio -> virtual output -> target app microphone
Target app speaker -> virtual input -> Python recorder
```

Common tools:

- macOS: BlackHole or Loopback
- Windows: VB-CABLE
- Linux: PipeWire/PulseAudio virtual sinks/sources

## Evidence in logs

For every completed voice turn, the solver stores:

- transcript text in `state.output.completion`;
- prompt and response WAV data as Inspect log attachments;
- one bounded `voice_turn` record in the sample store with backend names, attachment refs, audio facts, and monotonic timing marks.

Prefer `.eval` logs for voice evals. JSON logs can become large when audio attachments are retained.

## CI boundary

The default tests prove the Inspect solver protocol and attachment flow. They do not prove real OS audio routing or that a consumer app heard the prompt. Use separate local smoke tests for `sounddevice`, Chatterbox, and faster-whisper.
