# Voice Audio Latency Optimizations

## Decision

Optimize the voice solver around three changes before adding new ASR or hosted backends:

1. Stop real `sounddevice` recording live when VAD detects speech end.
2. Precompute prompt audio ahead of use, beginning as soon as enough prompt text is known and continuing while earlier prompt audio is playing/recording.
3. Cache repeated prompt audio across trials so identical prompt/voice/backend requests do not resynthesize.

These changes preserve the v0 solver evidence model: transcript in `state.output`, prompt and response audio in log attachments, and scalar timing in `state.store["voice_turn"]`.

## Evidence

Current logs and benchmarks show:

- Real `sounddevice` recording tail was 5.2s and 10.5s because recording captured a full window before VAD analysis.
- Warm faster-whisper on the saved 390 ms `Hello.` response is about 0.4-0.7s, so ASR is not the main steady-state bottleneck.
- Warm regular Chatterbox MPS is about 6.7s for the benchmark prompt.
- Warm Chatterbox Turbo MPS is about 2.5-3.4s, but it requires a separate first-time model download and does not support the same expressive controls.
- Full solver replay with warm Turbo TTS and saved response audio is about 3.6-3.7s before real playback and target response time.

## Optimization 1: live VAD recording

### Problem

`SounddeviceAudioEndpoint.record_until(...)` currently records up to `max_recording_ms` and only then runs VAD over the captured audio. This violates the intended `record_until` behavior and adds avoidable tail latency.

### Plan

Read input audio in `frame_duration_ms` chunks, run `VoiceActivityDetector.detect(...)` on each frame, and stop once speech has lasted at least `min_speech_ms` and is followed by `end_silence_ms` of silence. Preserve the hard `max_recording_ms` timeout.

### Acceptance checks

- Returned `RecordingResult.audio`, `speech_started_ms`, and `speech_completed_ms` match the existing fake endpoint semantics.
- A deterministic test proves chunked recording stops after speech plus silence rather than consuming the full maximum window.
- A live smoke run shows `recording_tail_ms` near `end_silence_ms` plus frame rounding rather than 5-10s.

## Optimization 2: prompt audio precompute while evaluation proceeds

### Problem

Prompt TTS is still the main warm steady-state cost when prompt audio is not cached. Even Turbo MPS spends about 2.5-3.4s per prompt; regular Chatterbox MPS spends about 6.7s.

### Plan

Add an explicit prompt-audio preparation path that can synthesize future prompt audio before the solver needs it. The first implementation should use bounded TTS concurrency of 1 so a single local model is not invoked concurrently on itself.

The intended runtime shape is:

```text
prepare first prompt audio
play first prompt audio
while sample 1 is playing/recording/transcribing, prepare sample 2 prompt audio
sample 2 solver reads prepared audio from cache
continue one or more samples ahead
```

### Design constraint

A normal Inspect solver only sees the current `TaskState`; it does not naturally own the whole dataset queue. Therefore prompt precompute should start as a separate helper or runner-level utility that receives known prompts before or alongside eval execution. The solver should consume from the same cache and fail or fall back according to the selected cache policy.

### Acceptance checks

- Benchmark with known prompts shows later samples avoid TTS on the solver critical path.
- Cache misses are visible: strict mode raises, write-through mode synthesizes and stores.
- Precompute progress does not change the final `voice_turn` store record semantics.

## Optimization 3: prompt audio cache for repeated trials

### Problem

Repeated prompts currently resynthesize identical audio. This wastes the dominant prompt-side cost.

### Plan

Add a prompt audio cache keyed by:

- prompt text;
- serialized voice request;
- synthesizer backend name;
- model identity or backend version;
- output audio format.

Support at least an in-memory cache first. Add disk caching only after the key shape is proven stable.

### Cache policies

- `read_through`: use cached audio if present; otherwise synthesize and store.
- `required`: fail if prompt audio is missing; useful for benchmark runs that should prove precompute coverage.
- `disabled`: always synthesize.

### Acceptance checks

- Repeated prompt benchmark shows second use avoids synthesis.
- Cache key changes when voice controls change.
- The prompt audio attachment still represents the exact audio played.

## Recommended implementation order

1. Implement live VAD recording. It is both a correctness fix and a latency fix.
2. Implement prompt audio cache with in-memory storage and strict/read-through policies.
3. Add a precompute helper that fills the cache for known prompts with bounded concurrency.
4. Add benchmark scripts showing no repeated live speech is needed.
5. Consider a proper Chatterbox Turbo backend after manual intelligibility checks of generated Turbo WAVs.

## Non-goals for this pass

- Streaming ASR.
- Streaming tokens directly into TTS.
- Hosted TTS/ASR backends.
- A persistent local TTS/ASR service process.
- Changing Inspect core APIs.

## Remaining implementation steps

The current diff does not implement this optimization plan. It only updates a local smoke script to print derived latencies, while the benchmark scripts and findings documents remain research artifacts. Complete this plan by:

1. Change `SounddeviceAudioEndpoint.record_until(...)` to read input in `frame_duration_ms` chunks, run `VoiceActivityDetector.detect(...)` as frames arrive, and stop after `min_speech_ms` of speech followed by `end_silence_ms` of silence, preserving the hard `max_recording_ms` timeout.
2. Add deterministic coverage for chunked/live recording semantics, including returned `RecordingResult.audio`, `speech_started_ms`, and `speech_completed_ms`.
3. Add an in-memory prompt audio cache keyed by prompt text, serialized voice request, synthesizer backend identity, model identity/version, and audio format.
4. Add cache policies for `read_through`, `required`, and `disabled`; default to `read_through` unless benchmark mode requires strict precomputed audio.
5. Wire `voice_turn(...)` to consume the prompt audio cache without changing output transcript, log attachments, or `state.store["voice_turn"]` semantics.
6. Add a prompt precompute helper that receives known prompts, fills the cache with bounded TTS concurrency of 1, and can run ahead of solver use.
7. Add benchmarks or tests showing repeated prompts avoid synthesis and precomputed prompts are used by later samples.
8. Keep Chatterbox Turbo, streaming ASR, hosted backends, and persistent service mode out of this implementation pass unless a separate design expands scope.
9. Run extension tests and a focused local benchmark/live smoke check to confirm recording tail and prompt synthesis latency improve.

## Current diff review remaining steps

Reviewing `git diff HEAD` shows the extension implementation has not changed yet. The only tracked code diff is a local smoke script update in `tmp/full.py`. The first optimization pass still needs implementation in the extension package:

1. Implement live VAD recording in `SounddeviceAudioEndpoint`.
2. Add deterministic chunked-recording tests.
3. Add the in-memory prompt audio cache and cache policies.
4. Wire `voice_turn(...)` to use the cache without changing log/store semantics.
5. Add the prompt audio precompute helper with TTS concurrency of 1.
6. Add tests or benchmarks proving repeated/precomputed prompts avoid synthesis.
7. Run extension validation and a focused local latency check.

## Open questions

1. Should cache storage be in-memory only for the first implementation, or should disk cache be included immediately?
2. Should cache misses default to read-through synthesis, or should latency benchmark mode require strict precomputed audio?
3. Where should the precompute helper live: a standalone function, a CLI command, or a task construction helper?
