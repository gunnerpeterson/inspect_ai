# Voice Audio Latency First Pass

## Decision

Implement the first latency pass with three changes only:

1. live VAD recording for the real `sounddevice` endpoint;
2. in-memory prompt audio caching;
3. prompt audio precompute that fills the same cache before `voice_turn` needs prompt audio.

Do not add Chatterbox Turbo, ASR tuning, disk cache, hosted backends, streaming APIs, or a service process in this pass.

## Current code shape

The solver currently does one linear turn in `turn_solver.py`:

```text
synthesize prompt audio
play prompt audio
record response
transcribe response
write output, attachments, and voice_turn record
```

The extension already has useful seams:

- `SpeechSynthesizer.synthesize(text, voice)` returns `WaveData`.
- `AudioEndpoint.record_until(...)` returns `RecordingResult`.
- `FakeAudioEndpoint` and `recording_from_audio(...)` already encode the desired VAD stop semantics.
- `voice_turn(...)` accepts backend instances or factories, so shared backend/cache objects can be passed in.

The main latency issues in current code are:

- `SounddeviceAudioEndpoint.record_until(...)` records a full window with `sd.rec(...)`, then applies VAD afterward.
- `voice_turn(...)` always calls the synthesizer directly; it has no cache/precomputed audio path.
- There is no shared prompt cache object or precompute entry point.

## Implementation plan

### 1. Live VAD recording

Refactor recording logic so VAD state can be updated one frame at a time.

Add a small recording state object or function that accepts `(frame_start_ms, frame)` and either:

- returns no result yet;
- returns `RecordingResult` once speech plus trailing silence completes;
- raises timeout after `max_recording_ms`.

Use that same logic from:

- `recording_from_audio(...)`, preserving fake endpoint behavior;
- `SounddeviceAudioEndpoint.record_until(...)`, while reading from `sounddevice` in chunks.

Acceptance checks:

- Existing VAD tests still pass.
- A new deterministic streaming-recording test proves recording stops after speech plus silence without consuming a full maximum window.
- `FakeAudioEndpoint` and `SounddeviceAudioEndpoint` share the same stopping semantics.

### 2. In-memory prompt audio cache

Add a prompt audio cache with explicit policies:

- `read_through`: return cached audio if present, otherwise synthesize and store;
- `required`: require precomputed audio and fail on miss;
- `disabled`: bypass cache and synthesize directly.

Cache key fields:

- prompt text;
- serialized `VoiceRequest`;
- synthesizer backend name;
- optional model identity string supplied by the caller/backend;
- output audio format facts.

First implementation should be in-memory only. Disk cache is deferred until the key shape is proven stable.

Acceptance checks:

- Repeating the same prompt with `read_through` calls the underlying synthesizer once.
- Changing voice controls changes the cache key.
- `required` fails on missing prompt audio.
- Prompt audio attachments still contain the exact audio that was played.

### 3. Prompt audio precompute

Add a standalone helper that fills the prompt cache for known prompts before solver use:

```text
precompute_prompt_audio(prompts, synthesizer, voice, cache, model_identity)
```

The helper should use bounded TTS concurrency of 1. This avoids concurrent calls into a single local model instance.

The solver should not own the dataset queue. Instead:

1. caller creates a cache;
2. caller optionally starts precompute for known prompts;
3. caller passes the same cache into `voice_turn(...)`;
4. `voice_turn(...)` consumes cached audio according to the selected policy.

Acceptance checks:

- Precompute fills the cache for known prompts.
- A solver run with `required` uses precomputed audio and does not invoke the synthesizer.
- A solver run with `read_through` handles misses by synthesizing and storing.

## Proposed API changes

Extend `voice_turn(...)` with optional prompt cache settings:

```text
prompt_audio_cache: PromptAudioCache | None = None
prompt_audio_cache_policy: PromptAudioCachePolicy = read_through
prompt_audio_model_identity: str | None = None
```

Keep default behavior equivalent to today by using no cache unless one is passed.

Add new files:

```text
prompt_audio_cache.py
prompt_audio_precompute.py
```

Add tests:

```text
test_prompt_audio_cache.py
test_prompt_audio_precompute.py
test_streaming_recording.py
```

## Out of scope for this pass

- Chatterbox Turbo backend.
- faster-whisper decode options.
- disk prompt cache.
- streaming TTS or ASR.
- hosted speech APIs.
- persistent local service process.

## Validation

Run:

```bash
PYTHONPATH=inspect-voice-eval/src:src uv run ruff format inspect-voice-eval
PYTHONPATH=inspect-voice-eval/src:src uv run ruff check inspect-voice-eval
PYTHONPATH=inspect-voice-eval/src:src uv run mypy inspect-voice-eval/src inspect-voice-eval/tests
PYTHONPATH=inspect-voice-eval/src:src uv run pytest -q -c inspect-voice-eval/pyproject.toml inspect-voice-eval/tests
```

For local evidence after implementation, run one sounddevice smoke test and compare `recording_tail_ms` against the saved baseline values of 5.2s and 10.5s.
