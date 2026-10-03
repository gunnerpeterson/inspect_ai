# Voice Audio Latency Testing Plan

## Purpose

Measure the voice solver latency before choosing implementation work. The current live runs are too slow, but the logs do not yet separate cold model load, warm inference, audio routing, and fixed-window recording costs. This plan defines tests that produce comparable Inspect logs and answer which optimization should happen first.

## Baseline from existing logs

Current successful live runs show:

| Run | Transcript | TTS | Playback | Response latency | Speech | Recording tail | Transcription | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Latest | `Hello.` | 40.3s | 2.8s | 2.6s | 0.4s | 5.2s | 46.4s | 97.7s |
| Earlier | `along.` | 35.5s | 3.1s | 1.7s | 0.3s | 10.5s | 31.6s | 82.6s |

The immediate unknowns are:

1. How much of TTS is cold load vs warm inference?
2. How much of transcription is cold load vs warm inference?
3. Does Apple MPS materially improve Chatterbox latency?
4. Does Chatterbox Turbo materially improve latency enough to justify a separate backend?
5. How much latency is solely caused by fixed-window recording?

## Metrics

Every test must save an Inspect `.eval` log and report these derived values from `sample.store["voice_turn"]["timing"]`:

- `tts_ms = tts_completed_ms - tts_started_ms`
- `playback_ms = playback_completed_ms - playback_started_ms`
- `target_response_latency_ms = speech_started_ms - playback_completed_ms`
- `spoken_response_ms = speech_completed_ms - speech_started_ms`
- `recording_tail_ms = recording_completed_ms - speech_completed_ms`
- `transcription_ms = transcription_completed_ms - transcription_started_ms`
- `total_turn_ms = transcription_completed_ms - solver_started_ms`

Also report:

- transcript text;
- prompt audio duration;
- response audio duration;
- backend names;
- device used for TTS/ASR;
- whether the run was first/cold or later/warm.

## Controls

Keep these stable across comparisons:

- Same prompt text: `Please say: hello from the voice evaluation test.`
- Same expected spoken response: `hello` or `hello from the voice evaluation test`.
- Same microphone/output devices for the small number of live audio-route tests.
- Same room/noise conditions for the small number of live audio-route tests.
- Same saved response audio for backend latency tests.
- Same `max_recording_ms`, `end_silence_ms`, and `frame_duration_ms` unless the test targets recording behavior.
- Same log reader for all reports.

If any control changes, record it next to the result instead of comparing the run as equivalent.

## Human speech policy

Do not require repeated live speech for backend latency tests. Human speech is noisy, inconvenient, and makes results less comparable. Split testing into two paths:

1. Backend latency tests use saved or synthetic response audio. These cover TTS latency, ASR latency, prompt caching, warm/cold backend reuse, and most full-solver timing.
2. Live audio-route tests use the microphone and speakers only when the thing being tested is physical audio routing, recording, or VAD stop behavior.

The preferred reusable response fixture for the first benchmark pass is the already-captured `Hello.` response from `logs/2026-09-29T22-29-10-00-00_task_AiscjAKpxnzi3PRvBso9oe.eval`. That log transcribed the response as `Hello.` and recorded `response_duration_ms: 390`, making it useful for exposing ASR overhead on very short audio. Extract this attachment to `tmp/voice_latency/hello_response.wav` and use it for ASR and replay benchmarks. Later benchmark passes may add a longer phrase such as `hello from the voice evaluation test`, but should not block the first pass on new live speech.

Only final end-to-end smoke tests should require a person to speak, and those should be limited to a small number of runs after backend-only measurements identify the most promising configuration.

## Test 1: cold vs warm regular Chatterbox CPU

### Question

Are the 35-40s TTS timings mostly cold startup or regular per-turn cost?

### Method

Run one Inspect task with three samples using the same backend instances:

- Chatterbox regular backend on `cpu`;
- faster-whisper tiny backend with current settings;
- sounddevice endpoint;
- identical prompt shape for each sample.

### Evidence to collect

Compare `tts_ms` and `transcription_ms` for sample 1 vs samples 2 and 3.

### Interpretation

- If sample 1 is slow and samples 2-3 are much faster, prioritize backend warm reuse and avoid over-investing in replacement backends.
- If all samples remain slow, prioritize Chatterbox Turbo, fast TTS, and ASR alternatives.

## Test 2: regular Chatterbox CPU vs MPS

### Question

Does Apple MPS reduce regular Chatterbox TTS enough to keep regular Chatterbox viable?

### Method

Run the three-sample warm test twice:

1. `ChatterboxSpeechSynthesizer(device="cpu")`
2. `ChatterboxSpeechSynthesizer(device="mps")`

### Evidence to collect

Compare warm sample `tts_ms` values, not just the first sample.

### Decision threshold

If MPS cuts warm TTS by at least 50% without errors or audio regressions, make MPS the recommended local Mac setting. If it does not, regular Chatterbox should remain a quality backend, not the default latency path.

## Test 3: Chatterbox Turbo on MPS

### Question

Does Chatterbox Turbo provide the best local neural TTS latency while preserving acceptable prompt quality?

### Method

Add a temporary experiment script that imports `chatterbox.tts_turbo.ChatterboxTurboTTS` directly and synthesizes the same prompt outside the solver. Measure wall-clock synthesis time and audio duration. If promising, add a proper backend later.

### Evidence to collect

- cold synthesis time;
- warm synthesis time;
- generated audio duration;
- whether the prompt is intelligible;
- unsupported controls such as expressiveness/CFG.

### Decision threshold

If warm Turbo TTS is under 5s for this prompt on MPS, add a dedicated Turbo backend. If it is only modestly faster, prioritize prompt caching or another TTS backend.

## Test 4: faster-whisper warm and decoding options

### Question

Is the 31-46s transcription latency mostly cold load, or are defaults too expensive for short responses?

### Method

Run transcription-only timing over saved response audio attachments or freshly recorded short WAV clips. Compare:

1. current faster-whisper tiny settings;
2. `beam_size=1`;
3. `condition_on_previous_text=False`;
4. both `beam_size=1` and `condition_on_previous_text=False`.

### Evidence to collect

- first transcription time;
- second and third transcription times using the same transcriber instance;
- transcript text for each run;
- response audio duration.

### Decision threshold

If warm transcription for sub-2s audio stays above 5s, test an Apple-specific ASR backend such as MLX Whisper. If warm transcription falls below 2s, keep faster-whisper and expose decoding options.

## Test 5: recording tail measurement

### Question

How much latency comes from fixed-window recording independent of TTS/ASR?

### Method

Use the current live solver logs and later compare against a live-VAD implementation. Before implementation, the expected fixed-window cost is visible as:

`recording_tail_ms = recording_completed_ms - speech_completed_ms`

### Evidence to collect

- current recording tail across at least three live runs;
- configured `max_recording_ms` and `end_silence_ms`;
- response audio duration.

### Decision threshold

If recording tail is consistently more than 1s, live VAD recording is a high-confidence fix. Existing logs already show 5.2s and 10.5s, so this is likely worth doing after backend timing tests.

## Test 6: prompt audio caching bound

### Question

How much can repeated evals gain if prompt text is cached?

### Method

Run the same prompt twice with a manual cache outside the solver or by reusing `WaveData` from the first synthesis in an experiment script.

### Evidence to collect

- first TTS time;
- second prompt audio retrieval time;
- cache key fields needed to avoid stale audio.

### Decision threshold

If prompt text repeats in real workloads, caching is worth adding even if backend latency improves. If prompts are usually unique, caching is a developer-loop optimization only.

## Result table format

Use this format for every run:

| Test | Sample | TTS backend | ASR backend | TTS device | ASR device | TTS | Recording tail | ASR | Total | Transcript |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |

This keeps comparisons auditable and avoids mixing cold and warm runs.

## Benchmark script plans

### Saved response audio extraction

The successful Inspect logs already contain the spoken response audio as an attachment. Use the latest successful log, read `sample.store["voice_turn"]["artifacts"]["response_audio_ref"]`, look up the referenced attachment in `sample.attachments`, decode the WAV data URL, and write one reusable response WAV for benchmark input.

Acceptance checks:

- The extracted WAV duration matches `sample.store["voice_turn"]["audio"]["response_duration_ms"]` within frame rounding.
- Re-transcribing the extracted WAV returns a non-empty transcript.
- The file path and source log path are printed together so later results remain traceable.

### Log latency reader

Create a small local script that accepts one or more `.eval` paths, reads each log with `read_eval_log(...)`, extracts the first sample's `voice_turn` record, and prints the result table format from this document.

Acceptance checks:

- Missing `voice_turn` raises a key error instead of silently skipping the log.
- Failed samples print their sample error before exiting.
- The script reports the source log path, transcript, backend names, audio facts, timing marks, and derived latency fields.

### TTS benchmark

Create a local script that measures prompt synthesis without microphone input. It should run the same prompt multiple times against selected synthesizer variants:

1. regular Chatterbox on CPU;
2. regular Chatterbox on MPS;
3. direct Chatterbox Turbo on MPS if available;
4. any later fast prompt backend.

For each variant, record cold run time, warm run time, output audio duration, backend name, device, and whether the generated prompt sounds intelligible when played separately.

Acceptance checks:

- The script separates first-run timing from later-run timing.
- The same prompt text is used for every variant.
- Generated audio can be written as WAV for manual spot checks.

### ASR benchmark

Create a local script that measures transcription using the extracted response WAV. It should reuse the same saved audio for every trial and compare faster-whisper settings:

1. current default settings;
2. `beam_size=1`;
3. `condition_on_previous_text=False`;
4. both `beam_size=1` and `condition_on_previous_text=False`;
5. any later Apple-specific or hosted ASR backend.

Acceptance checks:

- The script reports first transcription time and later warm transcription times using the same transcriber instance.
- Each trial prints transcript text and response audio duration.
- Any faster setting that changes the transcript materially is flagged rather than treated as an automatic win.

### Full solver replay benchmark

Create a local script that runs normal Inspect evals without live microphone input by using a deterministic replay endpoint that returns the extracted response WAV. This preserves the solver, store, attachment, and log path while removing human speech variability.

Acceptance checks:

- The generated Inspect logs contain `voice_turn` records and attachments.
- The replay endpoint reports a stable backend name so replay runs are distinguishable from real `sounddevice` runs.
- The script runs at least three samples with reused backend instances to expose cold vs warm behavior.

## Recommended order

1. Extract the saved response WAV from the latest successful Inspect log.
2. Add or use a log latency reader so results are cheap to compare.
3. Run the TTS benchmark across CPU, MPS, and Turbo candidates.
4. Run the ASR benchmark against the extracted response WAV.
5. Run the full solver replay benchmark with reused backend instances.
6. Run a small number of live audio-route tests only after backend-only measurements identify the most promising configuration.
7. Decide implementation order from the table.

## Likely outcomes

Current evidence suggests:

- live VAD recording will reliably remove 5-10s;
- model warmup may explain much of the TTS/ASR cost;
- MPS and Turbo are the strongest TTS candidates;
- faster-whisper may be acceptable if warm and configured for short utterances;
- if faster-whisper remains slow when warm, an Apple-specific ASR backend is justified.

