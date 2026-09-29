# Voice Audio Solver Extension

## Decision

Build the black-box voice evaluation harness as an Inspect extension, not as an `inspect_ai` core change. The first version uses `sounddevice`/PortAudio for real machine audio I/O, backend-agnostic TTS and transcription interfaces, and fake endpoints for CI.

## User problem

Evaluators need to test voice-native AI applications that expose only realtime audio interfaces. The first target is a single-turn evaluation: speak a prompt to the application, record its spoken response, transcribe it, and score the transcript with existing Inspect scorers.

## Scope

In scope for v0:

- Single-turn voice exchange.
- Text prompt to TTS audio.
- Playback through a selected output device.
- Recording from a selected input device.
- Energy-threshold VAD-based response end detection.
- Transcription to text.
- Store transcript as the solver output so existing scorers work.
- Save prompt and response audio artifacts for audit/debug evidence as Inspect log attachments.
- Record timing metadata, backend names, and attachment references in the sample store.
- Bounded timeouts for every wait.
- Fake audio endpoint for deterministic CI.

Out of scope for v0:

- Multi-turn conversations.
- Barge-in / mid-sentence interruption.
- Prosody or tone scoring.
- Native Inspect View audio player support.
- Core Inspect API changes.
- Automating real consumer applications in CI.

## Inspect integration points

The relevant Inspect APIs are:

- `@solver` / `Solver` from `inspect_ai.solver`: packaged extension solvers are discovered through `project.entry-points.inspect_ai`.
- `TaskState.input_text`: the prompt text for the voice turn.
- `TaskState.output`: set to `ModelOutput.from_content(..., transcript)` so existing scorers read `output.completion`.
- `TaskState.store`: stores one small JSON-serialisable `voice_turn` record. Store changes are logged automatically as `StoreEvent`s.
- `transcript().info(...)`: records custom structured events. v0 uses this to put prompt and response audio data URLs into the transcript so Inspect log condensation stores them as attachments.
- `ContentAudio`: Inspect already models audio content as `wav` or `mp3` data URLs.
- Inspect log condensation: data URLs in messages/events are converted into `attachment://...` references and stored in `EvalSample.attachments`.

One awkward point: Inspect does not expose a public “write attachment and return ref” helper. v0 should isolate attachment-ref creation behind a tiny extension function that mirrors Inspect’s current attachment hash scheme, and should cover it with tests. If Inspect later exposes a public helper, only that function changes.

## Architecture

The extension exposes a custom `@solver`, tentatively called `voice_turn`, that owns the exchange:

```text
TaskState input
→ synthesize prompt audio
→ play prompt audio
→ record target response until VAD silence or timeout
→ transcribe response audio
→ write transcript to TaskState output
→ emit transcript info event with prompt/response audio data URLs
→ record attachment references and timing metadata in the sample store
```

The solver depends on narrow interfaces:

```text
SpeechSynthesizer: text plus voice request -> audio
AudioEndpoint: play(audio), record_until(segment rule)
SpeechTranscriber: audio -> text
VoiceActivityDetector: audio frames -> speech/silence events
```

TTS is backend-agnostic. The extension should support a fake synthesizer for CI and implement Chatterbox as the first real synthesizer, because reference-speaker and expressive controls matter for voice red-team scenarios. Piper remains a later cheap-baseline backend. Each backend declares supported voice controls. If a caller requests a control the selected backend cannot honor, the backend fails clearly rather than silently ignoring the request.

Transcription is also backend-agnostic. The extension should support a fake transcriber for CI and implement faster-whisper as the first real transcriber for cheap local baseline runs. Optional backends such as whisper.cpp, Vosk, or hosted APIs can follow when users need different speed, streaming, or accuracy tradeoffs. The solver treats transcription output as the answer, so transcription failures must be visible rather than coerced into an empty answer.

The first VAD backend uses a simple energy threshold. This keeps v0 dependency-light and makes the stopping rule easy to reason about: speech begins when frame energy crosses a threshold, and the response ends after sustained below-threshold audio. More robust VAD backends, such as WebRTC VAD or Silero VAD, should be added only after tests or real runs show the energy threshold cuts off speech, confuses background noise for speech, or hangs until timeout too often.

The first real audio endpoint uses `sounddevice`, because it supports playback, recording, and full-duplex streams across Linux, macOS, and Windows through PortAudio. OS-level routing remains outside the extension: users still need virtual audio devices such as BlackHole, VB-CABLE, or PipeWire/PulseAudio routes.

Audio artifacts are stored inside Inspect logs as attachments rather than as external sidecar files. The sample store records one small namespaced `voice_turn` record with attachment references, backend names, audio facts, and timing marks. This keeps logs portable across local and remote log storage. Voice evals should prefer `.eval` logs; JSON logs may become large.

The `voice_turn` store record should remain bounded and scalar. Audio bytes belong in attachments, transcript text belongs in `state.output.completion`, and verbose diagnostics should be opt-in. The v0 record shape is:

```json
{
  "voice_turn": {
    "schema_version": 1,
    "artifacts": {
      "prompt_audio_ref": "attachment://...",
      "response_audio_ref": "attachment://..."
    },
    "backends": {
      "tts": "chatterbox",
      "audio": "sounddevice",
      "vad": "energy-threshold",
      "transcriber": "faster-whisper"
    },
    "audio": {
      "sample_rate_hz": 16000,
      "prompt_duration_ms": 2200,
      "response_duration_ms": 4100
    },
    "timing": {
      "solver_started_ms": 0,
      "tts_started_ms": 10,
      "tts_completed_ms": 340,
      "playback_started_ms": 360,
      "playback_completed_ms": 2560,
      "recording_started_ms": 2570,
      "speech_started_ms": 3240,
      "speech_completed_ms": 7340,
      "recording_completed_ms": 7800,
      "transcription_started_ms": 7810,
      "transcription_completed_ms": 8350
    }
  }
}
```

Timing values are monotonic milliseconds relative to solver start, not wall-clock timestamps. This is enough to derive TTS latency, target response latency, recording duration, transcription latency, and total turn time without changing Inspect core.

## Extension package layout

Implement this as a separate package, not under `src/inspect_ai`:

```text
inspect-voice-eval/
  pyproject.toml
  src/inspect_voice_eval/
    __init__.py
    _registry.py
    turn_solver.py
    turn_record.py
    wave_data.py
    speech_synthesis.py
    speech_transcription.py
    speech_detection.py
    audio_endpoint.py
    log_attachment_refs.py
    fake_speech.py
    fake_audio_endpoint.py
    energy_speech_detection.py
    sounddevice_endpoint.py
    chatterbox_speech.py
    faster_whisper_speech.py
  tests/
    test_voice_turn_solver.py
    test_energy_speech_detection.py
    test_log_artifact_refs.py
    test_backend_capabilities.py
    test_fake_audio_endpoint.py
```

`_registry.py` imports `voice_turn` so Inspect can discover the solver through:

```toml
[project.entry-points.inspect_ai]
inspect_voice_eval = "inspect_voice_eval._registry"
```

## Dependency strategy

Do not add audio, TTS, or transcription dependencies to `inspect_ai` core. Package them as extension extras:

```bash
pip install inspect-voice-eval[sounddevice]
pip install inspect-voice-eval[chatterbox]
pip install inspect-voice-eval[faster-whisper]
# Later optional baseline:
pip install inspect-voice-eval[piper]
```

The default CI path uses fake endpoints and should not require real audio devices or model downloads.

## Implementation plan

1. Package skeleton
   - Create the external `inspect_voice_eval` package.
   - Add the Inspect entry point.
   - Export `voice_turn` from the package root.

2. Domain records
   - `wave_data.py`: immutable audio container with PCM bytes/array, sample rate, channel count, duration, and `to_wav_data_uri()`.
   - `turn_record.py`: Pydantic records for artifact refs, backend names, audio facts, and monotonic timing marks.

3. Interfaces
   - `speech_synthesis.py`: `SpeechSynthesizer`, `VoiceRequest`, and supported-control declarations.
   - `speech_transcription.py`: `SpeechTranscriber`.
   - `speech_detection.py`: `VoiceActivityDetector` and speech/silence decisions.
   - `audio_endpoint.py`: `AudioEndpoint` with `play()` and `record_until()`.

4. Fake backends for CI
   - Fake synthesizer returns deterministic WAV audio.
   - Fake audio endpoint records calls and returns deterministic response audio or timeout behavior.
   - Fake transcriber maps known response audio to known text.

5. Energy VAD
   - Implement threshold over frame RMS/energy.
   - Parameters: energy threshold, frame duration, minimum speech duration, end-silence duration, maximum recording duration.
   - Always combine VAD with hard timeout.

6. Real backends
   - `sounddevice_endpoint.py`: plays/records named devices with `sounddevice`.
   - `chatterbox_speech.py`: lazy-imports Chatterbox only when used; supports reference speaker and expressive controls.
   - `faster_whisper_speech.py`: lazy-imports faster-whisper only when used; transcribes local WAV data.

7. Solver
   - `voice_turn(...)` accepts backend instances or backend factories, timeouts, VAD settings, and voice request.
   - Reads `state.input_text`.
   - Tracks monotonic timing marks.
   - Synthesizes prompt audio, plays it, records response audio, transcribes response.
   - Sets `state.output = ModelOutput.from_content(model=state.model.name, content=transcript)`.
   - Emits `transcript().info(...)` with prompt/response audio data URLs.
   - Stores the bounded `voice_turn` record in `state.store`.

8. Attachment refs
   - `log_attachment_refs.py` creates refs matching Inspect log condensation for artifact data URLs.
   - Tests verify refs match the attachments produced after a real Inspect eval log is written and read back.
   - Keep this dependency isolated because it mirrors current Inspect internals.

9. Tests
   - Full Inspect eval with fake backends and `match()` scorer.
   - Transcript becomes `output.completion`.
   - Prompt/response audio data URLs condense into attachments.
   - Store `voice_turn.artifacts.*_audio_ref` points to existing attachments.
   - Energy VAD closes after speech plus silence.
   - Energy VAD tolerates short mid-utterance pause.
   - Recording times out when silence never arrives.
   - Transcriber failure surfaces as sample error, not empty answer.
   - Unsupported voice controls fail clearly.
   - Optional local Chatterbox and faster-whisper tests are skipped unless extras/models are present.

10. Documentation
   - Add README with virtual audio routing diagrams.
   - Document macOS BlackHole, Windows VB-CABLE, and Linux PipeWire/PulseAudio setup.
   - State that CI proves solver protocol, not real consumer-app audio routing.

## Validation commands

For the extension package:

```bash
pytest
ruff format
ruff check --fix
mypy src tests
```

For real local smoke tests, run separate opt-in commands after installing extras and model assets:

```bash
pytest -m chatterbox
pytest -m faster_whisper
pytest -m sounddevice
```

## Open questions

None for v0.

