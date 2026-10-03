# Voice Audio Latency Findings

## Purpose

Keep a running record of latency measurements so optimization decisions are based on comparable evidence rather than impressions from one live run.

## Response audio fixture

The first benchmark pass uses the already-captured `Hello.` response from:

```text
logs/2026-09-29T22-29-10-00-00_task_AiscjAKpxnzi3PRvBso9oe.eval
```

Extracted fixture:

```text
tmp/voice_latency/hello_response.wav
```

Fixture facts:

| Field | Value |
| --- | ---: |
| Transcript in source log | `Hello.` |
| Duration from source log | 390 ms |
| Extracted WAV duration | 390 ms |

## Benchmark 1: faster-whisper default on saved `Hello.` WAV

Command:

```bash
.venv/bin/python -u tmp/voice_latency/asr_default_benchmark.py
```

Backend/settings:

| Field | Value |
| --- | --- |
| ASR backend | faster-whisper |
| Model size | tiny |
| Device | default backend setting |
| Precision | default extension setting (`int8`) |
| Audio input | `tmp/voice_latency/hello_response.wav` |
| Audio duration | 390 ms |

Results:

| Trial | Elapsed | Transcript |
| ---: | ---: | --- |
| 1 | 1286 ms | `Hello.` |
| 2 | 517 ms | `Hello.` |
| 3 | 436 ms | `Hello.` |

Finding:

The saved 390 ms response does not require 31-46 seconds to transcribe in isolation. Warm faster-whisper tiny transcription is roughly 0.4-0.5 seconds for this clip. The large transcription timing in earlier full live logs likely included cold model setup, process state, memory pressure after TTS, or other full-pipeline overhead.

Implication:

Do not prioritize replacing faster-whisper until full replay benchmarks show warm ASR is still slow inside the solver path. The next highest-value measurement is TTS cold/warm behavior across CPU, MPS, and Turbo candidates.

## Benchmark 2: Chatterbox regular CPU/MPS and Turbo MPS

Commands:

```bash
.venv/bin/python -u tmp/voice_latency/tts_benchmark.py
.venv/bin/python -u tmp/voice_latency/tts_turbo_benchmark.py
```

Prompt:

```text
Please say: hello from the voice evaluation test.
```

Regular Chatterbox results:

| Variant | Device | Trial | Elapsed | Audio duration |
| --- | --- | ---: | ---: | ---: |
| Regular | CPU | 1 | 34480 ms | 2320 ms |
| Regular | CPU | 2 | 10533 ms | 2680 ms |
| Regular | MPS | 1 | 38278 ms | 2520 ms |
| Regular | MPS | 2 | 6684 ms | 2920 ms |

Turbo Chatterbox results:

| Variant | Device | Trial | Elapsed | Audio duration |
| --- | --- | ---: | ---: | ---: |
| Turbo | MPS | model load | 853251 ms | n/a |
| Turbo | MPS | 1 | 9577 ms | 2800 ms |
| Turbo | MPS | 2 | 2891 ms | 2760 ms |
| Turbo | MPS | 3 | 3352 ms | 2960 ms |

Finding:

Regular Chatterbox has a large cold start and remains slow when warm. MPS improves warm regular Chatterbox from about 10.5 seconds to about 6.7 seconds, but it does not meet the 50% improvement threshold from the testing plan. Chatterbox Turbo requires a separate model download; the first Turbo run spent about 14.2 minutes loading/downloading the model. After that, warm Turbo synthesis was roughly 2.9-3.4 seconds for this prompt.

Implication:

Turbo MPS is the best measured local neural TTS path so far. The warm Turbo result is below the 5 second decision threshold, so a dedicated Turbo backend is justified if the generated audio passes manual intelligibility checks. Regular Chatterbox should remain the expressive/reference-speaker quality backend, not the local latency default. The large one-time Turbo model load should be documented and excluded from steady-state latency expectations.

## Benchmark 3: full solver replay with Turbo TTS and saved `Hello.` response

Command:

```bash
.venv/bin/python -u tmp/voice_latency/full_solver_replay_benchmark.py
```

Setup:

| Field | Value |
| --- | --- |
| Prompt | `Please say: hello from the voice evaluation test.` |
| TTS backend | local benchmark `chatterbox-turbo` protocol implementation |
| TTS device | MPS |
| Audio endpoint | local replay endpoint using `tmp/voice_latency/hello_response.wav` |
| ASR backend | faster-whisper tiny |
| Eval concurrency | `max_samples=1` |
| Log | `tmp/voice_latency/replay_logs/2026-10-02T02-30-13-00-00_task_XWGgZjDqXmTiLPzXjgEXbv.eval` |

Results:

| Sample | Transcript | TTS | Playback | Response latency | Speech | Recording tail | ASR | Total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `Hello.` | 23393 ms | 0 ms | 0 ms | 390 ms | 1 ms | 1386 ms | 25170 ms |
| 2 | `Hello.` | 2574 ms | 0 ms | 0 ms | 390 ms | 1 ms | 737 ms | 3702 ms |
| 3 | `Hello.` | 2509 ms | 0 ms | 0 ms | 390 ms | 2 ms | 717 ms | 3618 ms |

Finding:

With model downloads cached and samples forced to run sequentially, the warm full solver replay path is about 3.6-3.7 seconds before real playback time. The first replay sample still includes Turbo model load and takes about 25.2 seconds total. Warm TTS is about 2.5 seconds and warm ASR is about 0.7 seconds inside the solver path.

The benchmark had to set `max_samples=1`. Inspect otherwise runs samples concurrently, which caused multiple simultaneous Turbo model loads and misleading per-sample TTS timings. For local voice latency benchmarks, sequential sample execution is the clearer baseline unless the test is explicitly measuring concurrent throughput.

Implication:

The target of sub-10-second warm single-turn replay is realistic with Turbo TTS and warm faster-whisper. In real audio runs, add prompt playback duration, target response latency, spoken response duration, and VAD tail to the 3.6-3.7 second warm replay baseline. The remaining implementation priorities are a proper Turbo backend, live VAD recording, and optional prompt audio caching.
